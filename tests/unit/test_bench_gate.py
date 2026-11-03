"""Adversarial fixtures: the harness and report must not support false claims.

Covers the V0.6 exit criterion: a deliberately wrong final answer fails the
claim gate, and a simulated baseline cannot be measured as a claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmpu.bench.baselines import score_answer
from lmpu.bench.report import build_report
from lmpu.bench.workloads import get_workload

pytestmark = pytest.mark.unit


def _record(
    workload: str,
    baseline: str,
    trial: int,
    *,
    correct: bool,
    total_tokens: int,
    metadata: dict | None = None,
) -> dict:
    return {
        "workload": workload,
        "baseline": baseline,
        "trial": trial,
        "started_at": 1000 + trial,
        "wall_time_seconds": 1.0,
        "correct": correct,
        "error": None,
        "metadata": {"simulated": baseline == "trace_token", **(metadata or {})},
        "metrics": {
            "prefill_tokens": 0,
            "generated_tokens": total_tokens,
            "control_tokens": 0,
            "injected_tokens": 0,
            "total_tokens": total_tokens,
            "forward_passes": total_tokens,
            "peak_context_tokens": total_tokens,
            "gpu_energy_joules": 1.0,
            "vram_peak_mb": 1.0,
            "sleep_duration_seconds": 0.0,
        },
    }


def _write_run(tmp_path: Path, records: list[dict], baselines: list[str]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "schema": "lmpu.bench.manifest.v1",
        "commit": "abc123",
        "hardware": {"device": "cpu"},
        "config": {
            "workloads": ["bigint_mul"],
            "baselines": baselines,
            "model": "test-model",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run_dir


def test_score_answer_requires_a_correct_final_answer() -> None:
    workload = get_workload("bigint_mul")

    assert score_answer(workload, workload.answer, None) is True
    assert score_answer(workload, "definitely-wrong", None) is False
    assert score_answer(workload, None, None) is False
    assert score_answer(workload, workload.answer, "boom") is False


def test_wrong_final_answer_fails_the_claim_gate_even_if_tool_succeeded(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            "bigint_mul",
            "sync_tool_full_result",
            trial,
            correct=False,
            total_tokens=200,
            metadata={"tool_correct": True, "answer": "wrong"},
        )
        for trial in range(1, 4)
    ]
    report = build_report(_write_run(tmp_path, records, ["sync_tool_full_result"]))

    cell = report["end_to_end"]["cells"]["bigint_mul"]["sync_tool_full_result"]
    assert cell["correctness"]["rate"] == 0.0
    assert cell["per_correct_task"]["total_tokens"] is None
    assert report["end_to_end"]["claims"]["E_engine_integration"]["status"] != "measured"


def test_treatment_with_wrong_answers_cannot_measure_the_e_claim(tmp_path: Path) -> None:
    records = [
        _record("bigint_mul", "lmpu.model_async", trial, correct=False, total_tokens=10)
        for trial in range(1, 4)
    ] + [
        _record("bigint_mul", "app_agent", trial, correct=True, total_tokens=100)
        for trial in range(1, 4)
    ]
    report = build_report(_write_run(tmp_path, records, ["lmpu.model_async", "app_agent"]))

    claim = report["end_to_end"]["claims"]["E_engine_integration"]
    assert claim["status"] == "not_measured"


def test_simulated_baseline_cannot_win_a_claim(tmp_path: Path) -> None:
    records = [
        _record("bigint_mul", "trace_token", trial, correct=True, total_tokens=1)
        for trial in range(1, 4)
    ]
    report = build_report(_write_run(tmp_path, records, ["trace_token"]))

    assert report["simulation_quarantine"]["excluded_records"] == 3
    assert "trace_token" not in json.dumps(report["mechanism"])
    assert "trace_token" not in json.dumps(report["end_to_end"])
    assert "trace_token" in report["table"]["bigint_mul"]
    assert report["mechanism"]["claims"]["C5_wall_time"]["status"] == "not_measured"


def test_failed_trials_keep_token_accounting(tmp_path: Path) -> None:
    """Failure-path records must retain accumulated metrics (V0.2)."""
    from lmpu.bench.baselines import _failure_record
    from lmpu.bench.runner import TrialContext
    from lmpu.host.metrics import HostMetrics

    workload = get_workload("bigint_mul")
    context = TrialContext(
        config=None,  # type: ignore[arg-type]
        workload=workload,
        baseline="sync_tool_full_result",
        trial=1,
        seed=1,
        run_dir=tmp_path,
    )
    metrics = HostMetrics(session_id="s")
    metrics.record_forward("generated", 40)
    metrics.record_control_tokens(10)
    record = _failure_record(context, 0.0, 0.0, metrics, ValueError("boom"))

    assert record.correct is False
    assert record.metrics["generated_tokens"] == 40
    assert record.metrics["control_tokens"] == 10
    assert record.metrics["total_tokens"] == 50
    assert record.metadata["failed"] is True
