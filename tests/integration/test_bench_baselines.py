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
from lmpu.bench.runner import estimate_tokens

pytestmark = pytest.mark.integration


class FakeHost:
    """Deterministic host model for model baselines."""

    def __init__(self, answer: str, program: dict[str, Any]):
        self.answer = answer
        self.program = program

    async def start_trial(self) -> None:
        pass

    async def decode(self, prompt: str, max_tokens: int, stop: str = "") -> str:
        if "You have a tool" in prompt:
            if "TOOL RESULT" in prompt:
                return f"ANSWER: {self.answer}"
            return json.dumps(self.program)
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
    assert set(by_baseline) == set(BASELINES)
    assert all(record["correct"] for record in records)
    assert all(record["metrics"]["schema"] == "lmpu.host_metrics.v1" for record in records)

    assert by_baseline["direct_cot"]["metrics"]["generated_tokens"] > 0
    assert by_baseline["trace_token"]["metadata"]["simulated"] is True
    assert by_baseline["trace_token"]["metrics"]["generated_tokens"] > 1000
    assert by_baseline["sync_tool"]["metrics"]["injected_tokens"] > 0
    assert by_baseline["async_poll"]["metadata"]["polls"] >= 1
    assert by_baseline["lmpu.sync"]["metrics"]["control_tokens"] > 0

    stats = summary["workloads"]["bigint_mul"]["baselines"]
    assert set(stats) == set(BASELINES)
    assert all(item["correctness"]["rate"] == 1.0 for item in stats.values())
