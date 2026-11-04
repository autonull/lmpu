"""Benchmark workload registration and energy helper."""

from __future__ import annotations

import pytest

from lmpu.bench import get_workload, list_workloads
from lmpu.bench.energy import EnergyProbe
from lmpu.bench.workloads import _capability_document, _large_document, _matmul_expected

pytestmark = pytest.mark.unit


def test_workload_set_is_registered() -> None:
    assert set(list_workloads()) == {
        "bigint_mul",
        "matmul",
        "large_context",
        "sleep",
        "large_result",
        "batch",
        "best_case",
        "trivial_arithmetic",
        "small_result",
        "short_job",
        "capability",
        # Context scaling (V3.1)
        "context_1k",
        "context_8k",
        "context_32k",
        "context_128k",
        # Job duration scaling (V3.3)
        "sleep_1s",
        "sleep_3s",
        "sleep_30s",
        "sleep_300s",
        # Batch scaling (V3.4)
        "batch_1",
        "batch_16",
        "batch_64",
        "batch_256",
        "batch_1024",
    }
    assert get_workload("matmul").inputs == {"size": 512, "seed": 7}
    assert get_workload("large_context").baselines[0] == "lmpu.sync"
    assert "app_agent" in get_workload("large_context").baselines
    assert "native" in get_workload("large_context").baselines
    assert get_workload("sleep").inputs["program"].count("sleep(3)") == 1
    assert get_workload("large_result").config["result_bytes"] == 16 * 1024 * 1024
    assert get_workload("batch").native is not None
    assert get_workload("batch").verify(get_workload("batch").native()) is True
    # Negative controls
    assert get_workload("trivial_arithmetic").inputs == {"a": 1, "b": 1}
    assert get_workload("small_result").inputs["program"].count("1024") == 1
    assert get_workload("short_job").inputs["program"].count("sleep(1)") == 1
    # Capability
    assert get_workload("capability").inputs["text"] is not None
    assert len(get_workload("capability").inputs["text"]) > 100_000
    # Context scaling
    assert get_workload("context_1k").inputs["text"] is not None
    assert get_workload("context_128k").inputs["text"] is not None
    assert len(get_workload("context_1k").inputs["text"]) < len(get_workload("context_128k").inputs["text"])
    # Job duration scaling
    assert get_workload("sleep_1s").inputs["program"].count("sleep(1)") == 1
    assert get_workload("sleep_300s").inputs["program"].count("sleep(300)") == 1
    # Batch scaling
    assert get_workload("batch_1").native is not None
    assert get_workload("batch_1024").native is not None


def test_large_context_and_matmul_helpers_are_deterministic() -> None:
    document = _large_document()
    assert len(document) >= 10_000
    assert "SECRET: LMPU-BENCH-ALPHA" in document
    assert _matmul_expected(32, 1) == _matmul_expected(32, 1)
    # Capability document is deterministic and has the secret
    cap_doc = _capability_document()
    assert len(cap_doc) > 100_000
    assert "SECRET: LMPU-CAPABILITY-VERDICT-2026" in cap_doc


async def test_zero_second_idle_baseline_reports_availability() -> None:
    probe = EnergyProbe()
    try:
        baseline = await probe.measure_idle(0)
    finally:
        probe.close()

    assert baseline.seconds == 0.0
    assert baseline.samples == 0
    assert isinstance(baseline.available, bool)
