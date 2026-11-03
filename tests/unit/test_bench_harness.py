"""Benchmark harness: config, resume safety, and artifact schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmpu.bench import (
    BenchmarkArtifactError,
    BenchmarkConfig,
    BenchmarkHarness,
    TrialContext,
    TrialRecord,
    get_workload,
    read_raw,
)
from lmpu.host.metrics import HostMetrics

pytestmark = pytest.mark.unit


def _config(tmp_path: Path, **overrides) -> BenchmarkConfig:
    values = {
        "results_root": tmp_path / "results",
        "trials": 3,
        "warmups": 1,
        "seed": 7,
        "device": "cpu",
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.run_dir: Path | None = None

    async def setup(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    async def run(self, context: TrialContext) -> TrialRecord:
        self.calls.append((context.baseline, context.workload.name, context.trial))
        metrics = HostMetrics(session_id=f"s_{context.trial}")
        metrics.record_control_tokens(4)
        metrics.record_injected_tokens(2)
        payload = metrics.to_dict()
        payload["wall_time_seconds"] = 0.10 + context.trial * 0.01
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=f"job_{context.trial}",
            started_at=1.0,
            wall_time_seconds=payload["wall_time_seconds"],
            correct=True,
            metrics=payload,
        )

    async def teardown(self) -> None:
        pass


class FailOnThirdRunner(FakeRunner):
    async def run(self, context: TrialContext) -> TrialRecord:
        if context.trial == 3:
            raise RuntimeError("simulated interruption")
        return await super().run(context)


def test_config_loads_toml_and_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        'name = "x"\nworkloads = ["smoke"]\ntrials = 5\nseed = 11\nresults_root = "artifacts"\n',
        encoding="utf-8",
    )

    config = BenchmarkConfig.load(config_path)

    assert config.workloads == ("smoke",)
    assert config.trials == 5
    assert config.seed == 11
    assert config.results_root == Path("artifacts")
    with pytest.raises(ValueError, match="Unknown benchmark config keys"):
        BenchmarkConfig.from_mapping({"unknown": 1})


def test_workload_alias_resolves_to_canonical_name() -> None:
    assert get_workload("smoke").name == "bigint_mul"
    with pytest.raises(ValueError, match="Unknown workload"):
        get_workload("missing")


async def test_harness_writes_manifest_raw_and_summary(tmp_path: Path) -> None:
    runner = FakeRunner()

    summary = await BenchmarkHarness(_config(tmp_path), runner=runner).run()
    run_dir = Path(summary["run_id"])

    assert runner.calls == [
        ("lmpu.sync", "bigint_mul", -1),
        ("lmpu.sync", "bigint_mul", 1),
        ("lmpu.sync", "bigint_mul", 2),
        ("lmpu.sync", "bigint_mul", 3),
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "lmpu.bench.manifest.v1"
    assert manifest["config"]["seed"] == 7
    assert manifest["commit"]
    assert "gpu" in manifest["hardware"]

    records = read_raw(run_dir / "raw.jsonl")
    assert [record["trial"] for record in records] == [1, 2, 3]
    assert all(record["metrics"]["schema"] == "lmpu.host_metrics.v1" for record in records)
    assert summary["record_count"] == 3
    workload = summary["workloads"]["bigint_mul"]["baselines"]["lmpu.sync"]
    assert workload["wall_time_seconds"]["median"] == pytest.approx(0.12)
    assert len(workload["wall_time_seconds"]["ci95"]) == 2
    assert workload["correctness"]["rate"] == 1.0


async def test_resume_skips_completed_trials_and_appends_only_missing(tmp_path: Path) -> None:
    config = _config(tmp_path, warmups=0)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await BenchmarkHarness(config, runner=FailOnThirdRunner()).run()

    run_dir = next((tmp_path / "results").rglob("raw.jsonl")).parent
    assert [record["trial"] for record in read_raw(run_dir / "raw.jsonl")] == [1, 2]

    resumed = FakeRunner()
    summary = await BenchmarkHarness(config, runner=resumed).run()

    assert resumed.calls == [("lmpu.sync", "bigint_mul", 3)]
    records = read_raw(run_dir / "raw.jsonl")
    assert [record["trial"] for record in records] == [1, 2, 3]
    assert summary["record_count"] == 3


async def test_resume_rejects_a_changed_config(tmp_path: Path) -> None:
    first = _config(tmp_path, trials=1, warmups=0)
    await BenchmarkHarness(first, runner=FakeRunner()).run()

    second = _config(tmp_path, trials=2, warmups=0)
    with pytest.raises(BenchmarkArtifactError, match="different config"):
        await BenchmarkHarness(second, runner=FakeRunner()).run()
