"""Benchmark harness for reproducible LMPU evidence."""

from .config import BenchmarkConfig
from .harness import (
    BenchmarkArtifactError,
    BenchmarkHarness,
    append_raw,
    detect_hardware,
    read_raw,
    summarize,
)
from .runner import LmpuSyncRunner, TrialContext, TrialRecord, TrialRunner
from .workloads import WorkloadSpec, get_workload, list_workloads, resolve_workloads

__all__ = [
    "BenchmarkArtifactError",
    "BenchmarkConfig",
    "BenchmarkHarness",
    "LmpuSyncRunner",
    "TrialContext",
    "TrialRecord",
    "TrialRunner",
    "WorkloadSpec",
    "append_raw",
    "detect_hardware",
    "get_workload",
    "list_workloads",
    "read_raw",
    "resolve_workloads",
    "summarize",
]
