"""Canonical host metrics schema and trace consumption."""

from __future__ import annotations

import pytest

from lmpu.host.metrics import EnergySample, HostMetrics, MetricsCollector
from lmpu.kernel.trace import TraceConfig, init_trace_sink, read_trace, shutdown_trace_sink

pytestmark = pytest.mark.unit


def test_schema_covers_every_required_counter() -> None:
    metrics = HostMetrics(session_id="s_metrics")
    payload = metrics.to_dict()

    required = {
        "schema",
        "prefill_tokens",
        "generated_tokens",
        "injected_tokens",
        "control_tokens",
        "forward_passes",
        "wall_time_seconds",
        "vram_peak_mb",
        "gpu_energy_joules",
        "cpu_energy_joules",
    }
    assert required <= payload.keys()
    assert payload["schema"] == "lmpu.host_metrics.v1"


def test_forward_kinds_are_accounted_separately() -> None:
    metrics = HostMetrics(session_id="s_metrics")
    metrics.record_forward("prefill", 5)
    metrics.record_forward("generated", 2)
    metrics.record_forward("injected", 3)
    metrics.record_forward("control", 1)
    metrics.record_energy(gpu_joules=1.5, cpu_joules=0.5)

    payload = metrics.to_dict()

    assert payload["forward_passes"] == 4
    assert payload["prefill_tokens"] == 5
    assert payload["generated_tokens"] == 2
    assert payload["injected_tokens"] == 3
    assert payload["control_tokens"] == 1
    assert payload["total_tokens"] == 11
    assert payload["gpu_energy_joules"] == 1.5


def test_collector_emits_one_forward_record_and_one_snapshot(tmp_path) -> None:
    init_trace_sink(TraceConfig(trace_dir=str(tmp_path)))
    try:
        collector = MetricsCollector()
        collector.record_forward(
            "s_metrics",
            "generated",
            tokens=1,
            energy=EnergySample(gpu_joules=0.25),
            vram_peak_mb=128.0,
        )
        payload = collector.flush("s_metrics")
    finally:
        shutdown_trace_sink()

    records = read_trace(sorted(tmp_path.glob("*.jsonl"))[0])
    forwards = [record for record in records if record["type"] == "host_forward"]
    snapshots = [record for record in records if record["type"] == "host_metrics"]

    assert len(forwards) == 1
    assert forwards[0]["tokens"] == 1
    assert len(snapshots) == 1
    assert snapshots[0]["metrics"] == payload
    assert snapshots[0]["metrics"]["gpu_energy_joules"] == 0.25
    assert snapshots[0]["metrics"]["vram_peak_mb"] == 128.0
