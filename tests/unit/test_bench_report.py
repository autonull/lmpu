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
                "sync_tool",
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
            "sync_tool",
            1,
            wall=1.0,
            total_tokens=200,
            generated=100,
            injected=50,
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


def test_report_builds_measured_claims(tmp_path: Path) -> None:
    report = build_report(_run_dir(tmp_path))

    assert report["schema"] == "lmpu.bench.report.v1"
    assert report["table"]["bigint_mul"]["lmpu.sync"]["trials"] == 2
    assert report["table"]["bigint_mul"]["lmpu.sync"]["wall_time_seconds"]["median"] == 0.25
    assert report["claims"]["C1_token_reduction"]["status"] == "measured"
    assert report["claims"]["C1_token_reduction"]["by_workload"]["bigint_mul"]["sync_tool"][
        "reduction"
    ] == pytest.approx(10.0)
    assert "native" not in report["claims"]["C1_token_reduction"]["by_workload"]["bigint_mul"]

    c2 = report["claims"]["C2_sleep_proof"]
    assert c2["status"] == "measured"
    assert c2["treatment"]["forward_passes"] == 0
    assert c2["treatment"]["sleep_duration_seconds"] == pytest.approx(3.0)

    c3 = report["claims"]["C3_context_protection"]
    assert c3["status"] == "measured"
    assert c3["result_bytes"] == 16 * 1024 * 1024
    assert c3["measured"]["lmpu.slice"]["peak_context_tokens"] <= 64
    assert c3["measured"]["lmpu.slice"]["derived"] is True

    c4 = report["claims"]["C4_batching"]
    assert c4["status"] == "measured"
    assert c4["speedup"] == pytest.approx(4.0)

    c5 = report["claims"]["C5_wall_time"]
    assert c5["status"] == "measured"
    assert c5["native_speedup"]["large_result"] == pytest.approx(0.05 / 0.04)
    assert report["timeline"][0]["trial"] == 1


def test_report_writes_markdown_and_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)

    path = write_report(run_dir)

    markdown = path.read_text(encoding="utf-8")
    assert "| bigint_mul | lmpu.sync | 2 | 2/2 |" in markdown
    assert "C1_token_reduction" in markdown
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["claims"]["C4_batching"]["status"] == "measured"
    assert "Benchmark report" in render_markdown(payload)
