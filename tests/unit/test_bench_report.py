"""C1-C5 report rendering from raw benchmark records."""

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
    correct: bool = True,
) -> dict:
    return {
        "workload": workload,
        "baseline": baseline,
        "trial": trial,
        "started_at": 1000 + trial,
        "wall_time_seconds": wall,
        "correct": correct,
        "error": None,
        "metadata": {"simulated": baseline == "trace_token"},
        "metrics": {
            "prefill_tokens": 0,
            "generated_tokens": generated,
            "control_tokens": 5,
            "injected_tokens": injected,
            "total_tokens": total_tokens,
            "forward_passes": generated,
            "peak_context_tokens": injected,
            "gpu_energy_joules": 1.0,
            "vram_peak_mb": 10.0,
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
            "workloads": ["bigint_mul", "large_context", "sleep"],
            "baselines": ["lmpu.sync", "sync_tool", "trace_token"],
            "model": "test-model",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = [
        _record("bigint_mul", "lmpu.sync", 1, wall=0.2, total_tokens=20),
        _record("bigint_mul", "lmpu.sync", 2, wall=0.3, total_tokens=20),
        _record(
            "bigint_mul",
            "sync_tool",
            1,
            wall=1.0,
            total_tokens=200,
            generated=100,
            injected=50,
        ),
        _record("bigint_mul", "trace_token", 1, wall=0.1, total_tokens=5000, generated=5000),
        _record(
            "sleep",
            "lmpu.sync",
            1,
            wall=12.2,
            total_tokens=8,
            injected=2,
            correct=True,
        ),
        _record(
            "large_context",
            "lmpu.sync",
            1,
            wall=1.0,
            total_tokens=60,
            injected=50,
        ),
    ]
    (run_dir / "raw.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return run_dir


def test_report_builds_c1_and_partial_claims(tmp_path: Path) -> None:
    report = build_report(_run_dir(tmp_path))

    assert report["schema"] == "lmpu.bench.report.v1"
    assert report["table"]["bigint_mul"]["lmpu.sync"]["trials"] == 2
    assert report["table"]["bigint_mul"]["lmpu.sync"]["wall_time_seconds"]["median"] == 0.25
    assert report["claims"]["C1_token_reduction"]["status"] == "measured"
    assert report["claims"]["C1_token_reduction"]["by_workload"]["bigint_mul"]["sync_tool"][
        "reduction"
    ] == pytest.approx(10.0)
    assert report["claims"]["C2_sleep_proof"]["status"] == "partial"
    assert report["claims"]["C3_context_protection"]["status"] == "partial"
    assert report["claims"]["C4_batching"]["status"] == "not_measured"
    assert report["claims"]["C5_wall_time"]["status"] == "partial"
    assert report["timeline"][0]["trial"] == 1


def test_report_writes_markdown_and_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)

    path = write_report(run_dir)

    markdown = path.read_text(encoding="utf-8")
    assert "| bigint_mul | lmpu.sync | 2 | 2/2 |" in markdown
    assert "C1_token_reduction" in markdown
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["claims"]["C4_batching"]["status"] == "not_measured"
    assert "Benchmark report" in render_markdown(payload)
