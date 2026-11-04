"""Session overcommit: N parked sessions must overlap rather than serialize.

Each session submits one long job, then waits. If the runtime honours its
concurrency limit, wall time scales sublinearly in N; if the dispatcher
serialises, wall time scales linearly.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from lmpu.kernel.runtime import Runtime

SLEEP = 3
PROGRAM = f"import json, time\ntime.sleep({SLEEP})\nprint(json.dumps({{'slept': {SLEEP}}}))"


async def run_case(n: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rt = Runtime(
            socket_path=str(Path(tmp) / "run.sock"),
            shm_dir=str(Path(tmp) / "shm"),
            config={"trace_dir": str(Path(tmp) / "traces")},
        )
        await rt.start()
        sessions = [await rt._hello({"protocol": "lmpu/1"}) for _ in range(n)]
        ids = [s.session_id for s in sessions]
        jobs: list[str] = []
        t0 = time.perf_counter()
        try:
            for index, session_id in enumerate(ids):
                job_id = f"ms{n}-{index}"
                await rt._submit(
                    {
                        "job_id": job_id,
                        "processor": "python.vm",
                        "op": "run_program",
                        "inputs": {"program": PROGRAM},
                        "config": {
                            "mode": "async",
                            "timeout_ms": 60_000,
                            "bench_nonce": job_id,
                        },
                    },
                    session_id,
                )
                jobs.append(job_id)
            await asyncio.gather(
                *(rt.job_table.wait_terminal(job, timeout=120) for job in jobs)
            )
            wall = time.perf_counter() - t0
            print(
                f"n={n}  wall={wall:6.2f}s  ideal={SLEEP}s  "
                f"serial={n * SLEEP}s  overlap={n * SLEEP / wall:.2f}x"
            )
        finally:
            for session_id in ids:
                await rt._close({"detach": False}, session_id)
            await rt.stop()


async def main() -> None:
    for n in (1, 2, 3, 4, 8):
        await run_case(n)


if __name__ == "__main__":
    asyncio.run(main())