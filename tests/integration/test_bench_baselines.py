"""Baseline adapters: token accounting, program parsing, and end-to-end runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lmpu.bench import (
    BASELINES,
    BaselineRunner,
    BenchmarkConfig,
    BenchmarkHarness,
    get_workload,
    read_raw,
    resolve_baselines,
)
from lmpu.bench.baselines import _program_from_text
from lmpu.host.metrics import estimate_tokens

pytestmark = pytest.mark.integration


class FakeHost:
    """Deterministic host model for model baselines."""

    def __init__(self, answer: str, program: dict[str, Any]):
        self.answer = answer
        self.program = program

    async def start_trial(self) -> None:
        pass

    async def decode(self, prompt: str, max_tokens: int, stop: str = "", session: int = 0) -> str:
        if "STATUS" in prompt:
            return f"ANSWER: {self.answer}"
        if "You have a tool" in prompt:
            if "TOOL RESULT" in prompt:
                return f"ANSWER: {self.answer}"
            return json.dumps(self.program)
        if "TOOL RESULT" in prompt:
            return f"ANSWER: {self.answer}"
        return f"work\nANSWER: {self.answer}"

    async def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def take_energy(self) -> float:
        return 0.0

    async def close(self) -> None:
        pass


def test_program_parser_accepts_fences_and_rejects_invalid_json() -> None:
    payload = _program_from_text('```json\n{"op":"x","inputs":{"a":1}}\n```')
    assert payload == {"op": "x", "inputs": {"a": 1}}
    assert _program_from_text("not json") is None
    assert _program_from_text('{"processor":"p"}') is None


def test_baseline_resolution_and_trace_length() -> None:
    assert resolve_baselines("all") == BASELINES
    assert resolve_baselines(("trace_token",)) == ("trace_token",)
    with pytest.raises(ValueError, match="Unknown baseline"):
        resolve_baselines(("missing",))

    trace = get_workload("bigint_mul").trace()
    assert "RESULT" in trace
    assert estimate_tokens(trace) > 1000


async def test_all_baselines_run_and_emit_canonical_metrics(tmp_path: Path) -> None:
    workload = get_workload("bigint_mul")
    config = BenchmarkConfig(
        workloads=("bigint_mul",),
        baselines=BASELINES,
        trials=1,
        warmups=0,
        seed=5,
        device="cpu",
        results_root=tmp_path / "results",
    )
    host = FakeHost(answer=workload.answer, program=workload.program)

    summary = await BenchmarkHarness(config, runner=BaselineRunner(config, host=host)).run()

    records = read_raw(Path(summary["run_id"]) / "raw.jsonl")
    by_baseline = {record["baseline"]: record for record in records}
    expected = set(workload.baselines or ())
    assert set(by_baseline) == expected
    assert all(record["correct"] for record in records)
    assert all(record["metrics"]["schema"] == "lmpu.host_metrics.v1" for record in records)

    assert by_baseline["direct_cot"]["metrics"]["generated_tokens"] > 0
    assert by_baseline["trace_token"]["metadata"]["simulated"] is True
    assert by_baseline["trace_token"]["metrics"]["generated_tokens"] > 1000
    assert by_baseline["sync_tool_full_result"]["metrics"]["injected_tokens"] > 0
    assert by_baseline["sync_tool_full_result_oracle"]["metrics"]["injected_tokens"] > 0
    assert by_baseline["async_poll_programmatic"]["metadata"]["polls"] >= 1
    assert by_baseline["lmpu.model_async"]["metadata"]["park"]["level"] == "S1"
    assert by_baseline["lmpu.model_async_oracle"]["metadata"]["planner"] == "oracle"
    assert by_baseline["lmpu.model_async_2session"]["metadata"]["sessions"] == 2
    assert by_baseline["app_agent_2session"]["metadata"]["sessions"] == 2
    assert by_baseline["lmpu.sync"]["metrics"]["control_tokens"] > 0
    assert by_baseline["native"]["metadata"]["native"] is True
    assert by_baseline["native"]["metrics"]["forward_passes"] == 0

    stats = summary["workloads"]["bigint_mul"]["baselines"]
    assert set(stats) == expected
    assert all(item["correctness"]["rate"] == 1.0 for item in stats.values())


async def _run_one(tmp_path: Path, workload: str, baseline: str) -> dict:
    config = BenchmarkConfig(
        workloads=(workload,),
        baselines=(baseline,),
        trials=1,
        warmups=0,
        seed=5,
        device="cpu",
        idle_seconds=0.0,
        results_root=tmp_path / f"results-{workload}-{baseline}",
    )
    summary = await BenchmarkHarness(config).run()
    return read_raw(Path(summary["run_id"]) / "raw.jsonl")[0]


async def test_lmpu_sleep_parks_in_s1_with_zero_forward_passes(tmp_path: Path) -> None:
    record = await _run_one(tmp_path, "sleep", "lmpu.sleep")

    assert record["correct"] is True
    assert record["metadata"]["park"]["level"] == "S1"
    assert record["metadata"]["park"]["forward_passes_during_park"] == 0
    assert record["metrics"]["forward_passes"] == 0
    assert record["metrics"]["sleep_duration_seconds"] >= 2.5


async def test_lmpu_slice_reads_16mib_with_bounded_context(tmp_path: Path) -> None:
    record = await _run_one(tmp_path, "large_result", "lmpu.slice")

    assert record["correct"] is True
    assert record["metadata"]["result_bytes"] == 16 * 1024 * 1024
    assert record["metadata"]["derived"] is True
    assert record["metrics"]["peak_context_tokens"] <= 64


async def test_batch_submission_uses_less_control_overhead(tmp_path: Path) -> None:
    record = await _run_one(tmp_path, "batch", "lmpu.individual")
    batched = await _run_one(tmp_path, "batch", "lmpu.batch")

    assert record["correct"] is True
    assert batched["correct"] is True
    assert batched["metrics"]["control_tokens"] < record["metrics"]["control_tokens"]
