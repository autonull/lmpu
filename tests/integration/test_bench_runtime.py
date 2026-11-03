"""End-to-end benchmark harness test through the real runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmpu.bench import BenchmarkConfig, BenchmarkHarness, read_raw

pytestmark = pytest.mark.integration


async def test_default_runner_executes_workload_through_runtime(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        workloads=("bigint_mul",),
        trials=1,
        warmups=0,
        seed=3,
        device="cpu",
        results_root=tmp_path / "results",
    )

    summary = await BenchmarkHarness(config).run()

    run_dir = Path(summary["run_id"])
    records = read_raw(run_dir / "raw.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["correct"] is True
    assert record["metrics"]["schema"] == "lmpu.host_metrics.v1"
    assert record["metrics"]["control_tokens"] > 0
    assert record["metrics"]["injected_tokens"] > 0
    assert summary["workloads"]["bigint_mul"]["correctness"]["rate"] == 1.0
