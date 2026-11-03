"""Mechanism and end-to-end report rendering from raw benchmark records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmpu.bench.report import build_report, render_markdown, write_report

pytestmark = pytest.mark.unit


def _record(
    workload: str,
    baseline: str,
    trial: int,
    *,
    wall: float,
    total_tokens: int,
    generated: int = 0,
    injected: int = 10,
    control: int = 5,
    peak_context: int | None = None,
    sleep_seconds: float = 0.0,
    correct: bool = True,
    metadata: dict | None = None,
) -> dict:
    return {
        "workload": workload,
        "baseline": baseline,
        "trial": trial,
        "started_at": 1000 + trial,
        "wall_time_seconds": wall,
        "correct": correct,
        "error": None,
        "metadata": {"simulated": baseline == "trace_token", **(metadata or {})},
        "metrics": {
            "prefill_tokens": 0,
            "generated_tokens": generated,
            "control_tokens": control,
            "injected_tokens": injected,
            "total_tokens": total_tokens,
            "forward_passes": generated,
            "peak_context_tokens": injected if peak_context is None else peak_context,
            "gpu_energy_joules": 1.0,
            "vram_peak_mb": 10.0,
            "sleep_duration_seconds": sleep_seconds,
        },
    }


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "schema": "lmpu.bench.manifest.v1",
        "commit": "abc123",
        "hardware": {"device": "cpu"},
        "config": {
            "workloads": ["bigint_mul", "large_context", "sleep", "large_result", "batch"],
            "baselines": [
                "lmpu.sync",
                "lmpu.sleep",
                "lmpu.slice",
                "lmpu.individual",
                "lmpu.batch",
                "native",
                "sync_tool_full_result",
                "trace_token",
            ],
            "model": "test-model",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = [
        _record("bigint_mul", "lmpu.sync", 1, wall=0.2, total_tokens=20),
        _record("bigint_mul", "lmpu.sync", 2, wall=0.3, total_tokens=20),
        _record(
            "bigint_mul",
            "sync_tool_full_result",
            1,
            wall=1.0,
            total_tokens=200,
            generated=100,
            injected=50,
            metadata={"answer": "wrong", "tool_correct": True},
            correct=False,
        ),
        _record("bigint_mul", "trace_token", 1, wall=0.1, total_tokens=5000, generated=5000),
        _record("bigint_mul", "native", 1, wall=0.05, total_tokens=0),
        _record(
            "sleep",
            "lmpu.sync",
            1,
            wall=3.2,
            total_tokens=8,
            injected=2,
            sleep_seconds=3.0,
        ),
        _record(
            "sleep",
            "lmpu.sleep",
            1,
            wall=3.1,
            total_tokens=8,
            injected=2,
            sleep_seconds=3.0,
        ),
        _record("sleep", "native", 1, wall=3.0, total_tokens=0, sleep_seconds=3.0),
        _record(
            "large_context",
            "lmpu.sync",
            1,
            wall=1.0,
            total_tokens=60,
            injected=50,
        ),
        _record("large_context", "native", 1, wall=0.9, total_tokens=0),
        _record(
            "large_result",
            "lmpu.slice",
            1,
            wall=0.05,
            total_tokens=40,
            control=20,
            injected=20,
            peak_context=40,
            metadata={"derived": True, "result_bytes": 16 * 1024 * 1024},
        ),
        _record(
            "large_result",
            "native",
            1,
            wall=0.04,
            total_tokens=0,
            metadata={"result_bytes": 16 * 1024 * 1024},
        ),
        _record(
            "batch",
            "lmpu.individual",
            1,
            wall=0.4,
            total_tokens=400,
            control=400,
            injected=10,
        ),
        _record(
            "batch",
            "lmpu.batch",
            1,
            wall=0.1,
            total_tokens=30,
            control=20,
            injected=10,
        ),
        _record("batch", "native", 1, wall=0.02, total_tokens=0),
    ]
    (run_dir / "raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return run_dir


def test_report_builds_measured_mechanism_claims(tmp_path: Path) -> None:
    report = build_report(_run_dir(tmp_path))

    assert report["schema"] == "lmpu.bench.report.v2"
    assert report["table"]["bigint_mul"]["lmpu.sync"]["trials"] == 2
    assert report["table"]["bigint_mul"]["lmpu.sync"]["wall_time_seconds"]["median"] == 0.25
    assert report["table"]["bigint_mul"]["lmpu.sync"]["per_correct_task"]["total_tokens"] == 20.0

    c2 = report["mechanism"]["claims"]["C2_sleep_proof"]
    assert c2["status"] == "measured"
    assert c2["treatment"]["forward_passes"] == 0
    assert c2["treatment"]["sleep_duration_seconds"] == pytest.approx(3.0)

    c3 = report["mechanism"]["claims"]["C3_context_protection"]
    assert c3["status"] == "measured"
    assert c3["result_bytes"] == 16 * 1024 * 1024
    assert c3["measured"]["lmpu.slice"]["peak_context_tokens"] <= 64
    assert c3["measured"]["lmpu.slice"]["derived"] is True

    c4 = report["mechanism"]["claims"]["C4_batching"]
    assert c4["status"] == "measured"
    assert c4["speedup"] == pytest.approx(4.0)

    c5 = report["mechanism"]["claims"]["C5_wall_time"]
    assert c5["status"] == "measured"
    assert c5["native_speedup"]["large_result"] == pytest.approx(0.05 / 0.04)
    assert report["timeline"][0]["trial"] == 1


def test_simulation_quarantine_excludes_trace_token_from_claims(tmp_path: Path) -> None:
    report = build_report(_run_dir(tmp_path))

    quarantine = report["simulation_quarantine"]
    assert quarantine["excluded_records"] == 1
    assert quarantine["baselines"] == ["trace_token"]

    mechanism_text = json.dumps(report["mechanism"])
    end_to_end_text = json.dumps(report["end_to_end"])
    assert "trace_token" not in mechanism_text
    assert "trace_token" not in end_to_end_text

    assert "trace_token" in report["table"]["bigint_mul"]
    assert report["table"]["bigint_mul"]["trace_token"]["simulated"] is True


def test_end_to_end_section_scores_final_answers(tmp_path: Path) -> None:
    report = build_report(_run_dir(tmp_path))

    e2e = report["end_to_end"]
    cell = e2e["cells"]["bigint_mul"]["sync_tool_full_result"]
    assert cell["correctness"]["rate"] == 0.0
    assert cell["per_correct_task"]["total_tokens"] is None
    assert e2e["claims"]["E_engine_integration"]["status"] == "not_measured"


def test_report_writes_markdown_and_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)

    path = write_report(run_dir)

    markdown = path.read_text(encoding="utf-8")
    assert "| bigint_mul | lmpu.sync | mechanism | 2 | 2/2 |" in markdown
    assert "## Mechanism claims" in markdown
    assert "## End-to-end results" in markdown
    assert "## Simulation quarantine" in markdown
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["mechanism"]["claims"]["C4_batching"]["status"] == "measured"
    assert "Benchmark report" in render_markdown(payload)
