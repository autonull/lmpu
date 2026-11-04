"""Horizon scaling: does cost track decisions or work done?

Two arms, same jobs (a 1.5 s python.vm job each):

  sync      submit, wait, read, repeat        - an agent's blocking tool loop
  horizon   submit all, sleep until any done, read, repeat - control-channel loop

Sweeps N = 1, 4, 16, 40 and reports wall, control tokens per completed job, and
the tokens the model must hold at peak. Results must stay small; handles may
accumulate but are droppable ids, not payloads.
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

JOB_SECONDS = 1.5
SIZES = [1, 4, 16, 40]

PROGRAM = """import json, time
time.sleep({JOB_SECONDS})
print(json.dumps({{"ok": 1, "seed": {seed}}}))
"""


@dataclass
class Row:
    arm: str
    jobs: int
    wall_s: float
    control_tokens: int
    tokens_per_job: float
    peak_tokens: int


async def submit_async(rt: Runtime, sid: str, seed: int) -> tuple[str, int]:
    job = {
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": PROGRAM.format(JOB_SECONDS=JOB_SECONDS, seed=seed)},
        "config": {"timeout_ms": 60000},
    }
    result = await rt._submit(job, sid)
    return result.job_id, estimate_tokens(job)


async def sync_arm(rt: Runtime, sid: str, n: int) -> Row:
    tokens = 0
    held = 0
    peak = 0
    t0 = time.perf_counter()
    for i in range(n):
        job = {
            "processor": "python.vm",
            "op": "run_program",
            "inputs": {"program": PROGRAM.format(JOB_SECONDS=JOB_SECONDS, seed=i)},
            "config": {"mode": "sync", "timeout_ms": 60000},
        }
        tokens += estimate_tokens(job)
        summary = await rt._submit(job, sid)
        payload = {
            "job_id": summary.job_id,
            "status": summary.status.value,
            "result": summary.result,
        }
        got = estimate_tokens(payload)
        tokens += got
        held += got
        peak = max(peak, held)
    wall = time.perf_counter() - t0
    return Row("sync", n, wall, tokens, tokens / n, peak)


async def horizon_arm(rt: Runtime, sid: str, n: int) -> Row:
    tokens = 0
    t0 = time.perf_counter()

    pending: list[str] = []
    for i in range(n):
        job_id, cost = await submit_async(rt, sid, i)
        tokens += cost
        pending.append(job_id)

    # The model holds handles while jobs are in flight; they are ids, not payloads.
    peak = estimate_tokens({"jobs": pending})

    while pending:
        sleep = await rt._sleep({"until": {"any": pending}}, sid)
        if sleep.status == "registered":
            session = rt._sessions[sid]
            await asyncio.wait_for(session.wake_event.wait(), timeout=60)

        done = []
        for job_id in pending:
            record = rt.job_table.get(job_id)
            if record is not None and record.is_terminal():
                done.append(job_id)

        for job_id in done:
            summary = await rt._wait({"job_id": job_id, "timeout_ms": 5000}, sid)
            payload = {
                "job_id": job_id,
                "status": summary.status.value,
                "result": summary.result,
            }
            tokens += estimate_tokens(payload)
            pending.remove(job_id)
        peak = max(peak, estimate_tokens({"jobs": pending}))

    wall = time.perf_counter() - t0
    return Row("horizon", n, wall, tokens, tokens / n, peak)


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rt = Runtime(
            socket_path=f"{tmp}/run.sock",
            shm_dir=f"{tmp}/shm",
            config={"trace_dir": f"{tmp}/traces"},
        )
        await rt.start()
        sid = (await rt._hello({"protocol": "lmpu/1"})).session_id
        rows: list[Row] = []
        try:
            for n in SIZES:
                rows.append(await sync_arm(rt, sid, n))
                rows.append(await horizon_arm(rt, sid, n))
                for r in rows[-2:]:
                    print(
                        f"{r.arm:<8} n={r.jobs:>3}  wall={r.wall_s:>6.2f}s  "
                        f"ctrl_tok={r.control_tokens:>5}  per_job={r.tokens_per_job:>5.1f}  "
                        f"peak={r.peak_tokens:>5}"
                    )
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()

    print(f"\n{'arm':<8} {'jobs':>4} {'wall s':>7} {'ctrl tok':>9} {'tok/job':>8} {'peak':>6}")
    print("-" * 48)
    for r in rows:
        print(
            f"{r.arm:<8} {r.jobs:>4} {r.wall_s:>7.2f} {r.control_tokens:>9} "
            f"{r.tokens_per_job:>8.1f} {r.peak_tokens:>6}"
        )

    hz = [r for r in rows if r.arm == "horizon"]
    sy = [r for r in rows if r.arm == "sync"]
    print("-" * 48)
    print(f"horizon tok/job: {hz[0].tokens_per_job:.1f} (n=1) -> "
          f"{hz[-1].tokens_per_job:.1f} (n=40)")
    print(f"sync    tok/job: {sy[0].tokens_per_job:.1f} (n=1) -> "
          f"{sy[-1].tokens_per_job:.1f} (n=40)")
    print(f"horizon wall at n=40: {hz[-1].wall_s:.1f}s vs sync {sy[-1].wall_s:.1f}s")

    out = Path("benchmarks/results/horizon")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
