"""Where the data plane starts to win, measured against a real model.

Task: does any record carry marker MARKER? (membership, no arithmetic, so a
small model can plausibly succeed while the data fits its context.)

Arms:
  context   the 0.5B model receives the records in-context and answers
  lmpu      import a buffer, submit one python.vm job, read the summary

The third arm - an agent that writes a script and runs it in a sandbox - is not
measured, because it is analytically the same shape as LMPU: O(1) control tokens
independent of data. It is reported as a tie, not as a win. Claiming otherwise
would be measuring our own metadata against a strawman.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT = 32768
RECORD_SIZES = [50, 200, 800, 3200]
BULK_MIB = 256

PROGRAM = """
import json
buf = read_buffer("records")
text = bytes(buf).decode("utf-8", "ignore")
print(json.dumps({"has_marker": "MARKER" in text}))
"""


@dataclass
class Row:
    records: int
    kib: float
    arm: str
    host_tokens: int
    wall_s: float
    correct: bool | None


def make_data(n: int, needle_idx: int) -> bytes:
    lines = [f'{{"id": {i}, "v": "{(i * 7919) & 0xFFFF:04x}"}}\n' for i in range(n)]
    lines[needle_idx] = f'{{"id": {needle_idx}, "v": "MARKER"}}\n'
    return "".join(lines).encode()


async def context_arm(model, tokenizer, data: bytes) -> tuple[int, float, bool]:
    prompt = (
        "Records, one per line. Answer yes if any record has v equal to MARKER, "
        "otherwise no. Reply with one word.\n\n"
        + data.decode("utf-8", "ignore")
        + "\nAnswer:"
    )
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    n_tokens = ids["input_ids"].shape[1]
    if n_tokens > MAX_CONTEXT:
        return n_tokens, 0.0, False

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **ids, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    wall = time.perf_counter() - t0
    text = tokenizer.decode(out[0][n_tokens:], skip_special_tokens=True).strip().lower()
    return n_tokens, wall, text.startswith("yes")


async def lmpu_arm(rt: Runtime, sid: str, raw: bytes) -> tuple[int, float, bool]:
    t0 = time.perf_counter()
    ref = await rt._buffer_import(
        {"name": "records", "data": raw, "dtype": "uint8", "shape": [len(raw)]}, sid
    )
    job = {
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": PROGRAM, "records": {"buffer": ref.buffer_id}},
        "config": {"mode": "sync", "timeout_ms": 120000},
    }
    # Control-path payload only: what the host model would actually see.
    host_tokens = estimate_tokens(job) + estimate_tokens(
        {"buffer_id": ref.buffer_id, "shape": list(ref.shape), "dtype": ref.dtype}
    )
    result = await rt._submit(job, sid)
    wall = time.perf_counter() - t0
    ok = bool((result.result or {}).get("has_marker"))
    return host_tokens, wall, ok


async def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

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
            for n in RECORD_SIZES:
                raw = make_data(n, needle_idx=n // 2)
                print(f"\n=== {n} records ({len(raw) / 1024:.1f} KiB) ===")

                tok, wall, ok = await context_arm(model, tokenizer, raw)
                rows.append(Row(n, len(raw) / 1024, "context", tok, wall, ok))
                print(f"  context {tok:>8} tok {wall:6.2f}s correct={ok}")

                tok, wall, ok = await lmpu_arm(rt, sid, raw)
                rows.append(Row(n, len(raw) / 1024, "lmpu", tok, wall, ok))
                print(f"  lmpu    {tok:>8} tok {wall:6.2f}s correct={ok}")

            # One bulk row: the context arm cannot attempt this at all.
            raw = make_data(200_000, needle_idx=100_000)
            raw = (raw * (BULK_MIB * 1024 * 1024 // len(raw) + 1))[: BULK_MIB * 1024 * 1024]
            print(f"\n=== bulk ({BULK_MIB} MiB) ===")
            tok, wall, ok = await lmpu_arm(rt, sid, raw)
            rows.append(Row(200_000, len(raw) / 1024, "lmpu", tok, wall, ok))
            print(f"  lmpu    {tok:>8} tok {wall:6.2f}s correct={ok}")
            ctx_tokens = len(raw) // 4
            print(f"  context would need {ctx_tokens:,} tokens "
                  f"({ctx_tokens / MAX_CONTEXT:.0f}x the {MAX_CONTEXT} limit)")
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()

    print(f"\n{'records':>8} {'KiB':>9} {'arm':<8} {'host tok':>9} {'wall s':>8} {'ok':>6}")
    print("-" * 54)
    for r in rows:
        print(f"{r.records:>8} {r.kib:>9.1f} {r.arm:<8} {r.host_tokens:>9} "
              f"{r.wall_s:>8.2f} {str(r.correct):>6}")

    ctx = [r for r in rows if r.arm == "context"]
    lm = [r for r in rows if r.arm == "lmpu"]
    ok_ctx = [r for r in ctx if r.correct]
    fail_ctx = [r for r in ctx if not r.correct]
    print("-" * 54)
    if ok_ctx:
        print(f"context succeeds to {max(r.records for r in ok_ctx)} records "
              f"at {max(r.host_tokens for r in ok_ctx):,} tokens")
    if fail_ctx:
        print(f"context fails from {min(r.records for r in fail_ctx)} records")
    print(f"lmpu: {len([r for r in lm if r.correct])}/{len(lm)} correct, "
          f"host tokens {min(r.host_tokens for r in lm)}-{max(r.host_tokens for r in lm)} "
          f"across {min(r.kib for r in lm):.1f}-{max(r.kib for r in lm):.0f} KiB")
    print("script-sandbox agent: O(1) script tokens, ties lmpu (not a win)")

    out = Path("benchmarks/results/baseline_law")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
