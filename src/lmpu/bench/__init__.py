"""Benchmark harness for reproducible LMPU evidence."""

from .baselines import BASELINES, BaselineRunner, HFHost, resolve_baselines, score_answer
from .taxonomy import (
    END_TO_END_BASELINES,
    MECHANISM_BASELINES,
    SIMULATION_BASELINES,
)
from .config import BenchmarkConfig
from .harness import (
    BenchmarkArtifactError,
    BenchmarkHarness,
    append_raw,
    detect_hardware,
    read_raw,
    summarize,
)
from .runner import TrialContext, TrialRecord, TrialRunner
from .workloads import WorkloadSpec, get_workload, list_workloads, resolve_workloads

__all__ = [
    "BASELINES",
    "END_TO_END_BASELINES",
    "MECHANISM_BASELINES",
    "SIMULATION_BASELINES",
    "BaselineRunner",
    "BenchmarkArtifactError",
    "BenchmarkConfig",
    "BenchmarkHarness",
    "HFHost",
    "TrialContext",
    "TrialRecord",
    "TrialRunner",
    "WorkloadSpec",
    "append_raw",
    "detect_hardware",
    "get_workload",
    "list_workloads",
    "read_raw",
    "resolve_baselines",
    "resolve_workloads",
    "score_answer",
    "summarize",
]
