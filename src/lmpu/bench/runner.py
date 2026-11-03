"""Trial data shared by the benchmark harness and baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import BenchmarkConfig
from .workloads import WorkloadSpec


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Inputs for one measured or warmup trial."""

    config: BenchmarkConfig
    workload: WorkloadSpec
    baseline: str
    trial: int
    seed: int
    run_dir: Path


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One raw trial result. ``metrics`` uses the canonical A2 schema."""

    workload: str
    baseline: str
    trial: int
    seed: int
    job_id: str
    started_at: float
    wall_time_seconds: float
    correct: bool
    metrics: dict[str, Any]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "lmpu.bench.trial.v1",
            "workload": self.workload,
            "baseline": self.baseline,
            "trial": self.trial,
            "seed": self.seed,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "wall_time_seconds": self.wall_time_seconds,
            "correct": self.correct,
            "error": self.error,
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


class TrialRunner(Protocol):
    """Pluggable trial executor used by the harness."""

    async def setup(self, run_dir: Path) -> None: ...

    async def run(self, context: TrialContext) -> TrialRecord: ...

    async def teardown(self) -> None: ...
