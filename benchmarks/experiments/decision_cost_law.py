"""Decision-cost law (O2): host_tokens ≈ a + b·decisions.

Two sweeps:
1. Bytes at fixed decision count (expect slope ≈ 0)
2. Decisions at fixed bytes (expect slope ≈ tokens/decision)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

# Fixed decision-count sweep
DECISIONS_FIXED = 4
BYTE_SIZES = [1, 64, 1024, 64 * 1024, 256 * 1024]  # KiB

# Fixed bytes sweep
BYTES_FIXED = 64 * 1024  # 64 KiB
DECISION_COUNTS = [1, 2, 4, 8, 16, 32]

# Simple program that reads a buffer and returns a tiny summary
PROGRAM_TEMPLATE = """import json, hashlib
buf = read_buffer("data")
h = hashlib.sha256(bytes(buf)).hexdigest()
print(json.dumps({{"sha256": h, "len": len(buf)})))
"""

@dataclass
class Row:
    sweep: str  # "bytes" or "decisions"
    x_value: int  # KiB or decision count
    decisions: int
    kib: int
    host_tokens: int
    wall_s: float


async def run_jobs(rt: Runtime, sid: str, n_jobs: int, data_kib: int) -> tuple[int, float]:
    """Submit n_jobs with data_kib KiB each, return (total_host_tokens, wall_s)."""
    # Create data once
    data = b"x" * (data_kib * 1024)
    
    # Import buffer
    ref = await rt._buffer_import(
        {"name": "data", "data": data, "dtype": "uint8", "shape": [len(data)]}, sid
    )
    import_tokens = estimate_tokens({
        "buffer_id": ref.buffer_id, "shape": list(ref.shape), "dtype": ref.dtype
    })
    
    tokens = import_tokens
    t0 = time.perf_counter()
    
    job_ids = []
    for i in range(n_jobs):
        job = {
            "processor": "python.vm",
            "op": "run_program",
            "inputs": {"program": PROGRAM_TEMPLATE, "data": {"buffer": ref.buffer_id}},
            "config": {"mode": "sync", "timeout_ms": 60000},
        }
        tokens += estimate_tokens(job)
        result = await rt._submit(job, sid)
        job_ids.append(result.job_id)
        # Summary tokens
        payload = {"job_id": result.job_id, "status": result.status.value, "result": result.result}
        tokens += estimate_tokens(payload)
    
    wall = time.perf_counter() - t0
    return tokens, wall


async def main() -> None:
    rows: list[Row] = []
    
    with tempfile.TemporaryDirectory() as tmp:
        rt = Runtime(
            socket_path=f"{tmp}/run.sock",
            shm_dir=f"{tmp}/shm",
            config={"trace_dir": f"{tmp}/traces"},
        )
        await rt.start()
        sid = (await rt._hello({"protocol": "lmpu/1"})).session_id
        
        try:
            # Sweep 1: Bytes at fixed decision count
            print("=== Sweep 1: Bytes at fixed decisions =", DECISIONS_FIXED, "===")
            for kib in BYTE_SIZES:
                tokens, wall = await run_jobs(rt, sid, DECISIONS_FIXED, kib)
                rows.append(Row("bytes", kib, DECISIONS_FIXED, kib, tokens, wall))
                print(f"  {kib:>6} KiB: {tokens} tokens, {wall:.3f}s  (per job: {tokens/DECISIONS_FIXED:.1f})")
            
            # Sweep 2: Decisions at fixed bytes
            print(f"\n=== Sweep 2: Decisions at fixed bytes = {BYTES_FIXED} KiB ===")
            for n in DECISION_COUNTS:
                tokens, wall = await run_jobs(rt, sid, n, BYTES_FIXED)
                rows.append(Row("decisions", n, n, BYTES_FIXED, tokens, wall))
                print(f"  {n:>2} jobs: {tokens} tokens, {wall:.3f}s  (per job: {tokens/n:.1f})")
                
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()
    
    # Print summary table
    print("\n" + "=" * 70)
    print("DECISION-COST LAW RESULTS")
    print("=" * 70)
    print(f"{'Sweep':<10} {'X':>8} {'Decisions':>9} {'KiB':>8} {'Host Tok':>9} {'Wall s':>8} {'Tok/Dec':>8}")
    print("-" * 70)
    for r in rows:
        print(f"{r.sweep:<10} {r.x_value:>8} {r.decisions:>9} {r.kib:>8} {r.host_tokens:>9} {r.wall_s:>8.3f} {r.host_tokens/r.decisions:>8.1f}")
    
    # Analysis
    bytes_rows = [r for r in rows if r.sweep == "bytes"]
    decisions_rows = [r for r in rows if r.sweep == "decisions"]
    
    print("\n--- Analysis ---")
    
    # Bytes sweep: slope should be ~0
    if len(bytes_rows) >= 2:
        x_vals = [r.x_value for r in bytes_rows]
        y_vals = [r.host_tokens for r in bytes_rows]
        # Simple linear regression
        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        num = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
        den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
        slope_bytes = num / den if den != 0 else 0
        intercept_bytes = y_mean - slope_bytes * x_mean
        print(f"Bytes sweep:  host_tokens ≈ {intercept_bytes:.1f} + {slope_bytes:.4f} × KiB")
        print(f"  Expected: slope ≈ 0 (flat)")
    
    # Decisions sweep: slope should be tokens/decision
    if len(decisions_rows) >= 2:
        x_vals = [r.x_value for r in decisions_rows]
        y_vals = [r.host_tokens for r in decisions_rows]
        n = len(x_vals)
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        num = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
        den = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
        slope_decisions = num / den if den != 0 else 0
        intercept_decisions = y_mean - slope_decisions * x_mean
        print(f"Decisions sweep: host_tokens ≈ {intercept_decisions:.1f} + {slope_decisions:.2f} × decisions")
        print(f"  Expected: slope ≈ tokens/decision (~65-70), intercept ≈ system prompt")
    
    # Save results
    out = Path("benchmarks/results/decision_cost_law")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())