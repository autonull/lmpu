"""Trial execution for the benchmark harness.

The runner owns the runtime lifecycle for one run; the harness owns trial
ordering, persistence, and aggregation. Custom runners implement ``TrialRunner``.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lmpu.host.metrics import HostMetrics
from lmpu.kernel.runtime import Runtime

from .config import BenchmarkConfig
from .workloads import WorkloadSpec

logger = logging.getLogger(__name__)


def estimate_tokens(payload: object) -> int:
    """Conservative token estimate for rendered benchmark payloads."""
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, separators=(",", ":"), default=str)
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Inputs for one measured or warmup trial."""

    config: BenchmarkConfig
    workload: WorkloadSpec
    trial: int
    seed: int
    run_dir: Path


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One raw trial result. ``metrics`` uses the canonical A2 schema."""

    workload: str
    trial: int
    seed: int
    job_id: str
    started_at: float
    wall_time_seconds: float
    correct: bool
    metrics: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lmpu.bench.trial.v1",
            "workload": self.workload,
            "trial": self.trial,
            "seed": self.seed,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "wall_time_seconds": self.wall_time_seconds,
            "correct": self.correct,
            "error": self.error,
            "metrics": self.metrics,
        }


class TrialRunner(Protocol):
    """Pluggable trial executor used by the harness."""

    async def setup(self, run_dir: Path) -> None: ...

    async def run(self, context: TrialContext) -> TrialRecord: ...

    async def teardown(self) -> None: ...


class LmpuSyncRunner:
    """Execute workloads through a real runtime session in synchronous mode."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.runtime: Runtime | None = None
        self._runtime_dir: Path | None = None

    async def setup(self, run_dir: Path) -> None:
        # Unix socket paths are length-limited; keep transient runtime files
        # out of the (possibly deep) artifact directory.
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="lmpu-bench-"))
        self.runtime = Runtime(
            socket_path=str(self._runtime_dir / "run.sock"),
            shm_dir=str(self._runtime_dir / "shm"),
            config={
                "trace_dir": str(run_dir / "traces"),
                "batch_window_ms": 1,
                "reaper_interval_ms": 100,
            },
        )
        await self.runtime.start()

    async def run(self, context: TrialContext) -> TrialRecord:
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("runner is not set up")

        hello = await runtime._hello({"protocol": "lmpu/1"})
        session_id = hello.session_id
        job_id = f"bench_{context.seed}_{context.workload.name}_{context.trial}"
        request = {
            "job_id": job_id,
            "processor": context.workload.processor,
            "op": context.workload.op,
            "inputs": context.workload.inputs,
            "config": {"mode": "sync", "timeout_ms": 60_000} | context.workload.config,
        }

        metrics = HostMetrics(session_id=session_id)
        metrics.record_control_tokens(estimate_tokens(request))
        started_at = time.time()
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            submitted = await runtime._submit(request, session_id)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
        except Exception as exc:
            error = str(exc)
            logger.exception("benchmark trial failed", extra={"job_id": job_id})
        wall = time.perf_counter() - started

        if result is not None:
            metrics.record_injected_tokens(estimate_tokens(result))
        payload = metrics.to_dict()
        payload["wall_time_seconds"] = wall

        try:
            await runtime._close({"detach": False}, session_id)
        except Exception:
            logger.exception("benchmark session close failed", extra={"session": session_id})

        return TrialRecord(
            workload=context.workload.name,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=payload,
            error=error,
        )

    async def teardown(self) -> None:
        if self.runtime is not None:
            await self.runtime.stop()
            self.runtime = None
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
            self._runtime_dir = None
