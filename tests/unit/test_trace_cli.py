"""Trace CLI replay, gap detection, redaction, and timeline output."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from lmpu.cli import app

runner = CliRunner()


def _write_trace(directory: Path, sequences: list[int]) -> Path:
    records = [
        {
            "type": "event_job_submitted",
            "session": "s_cli",
            "job_id": f"job_{seq}",
            "seq": seq,
            "mono": float(seq),
            "ts": 1000.0 + seq,
            "authorization": "Bearer top-secret",
            "data": {"job_id": f"job_{seq}"},
        }
        for seq in sequences
    ]
    path = directory / "trace-cli-001.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_trace_reports_gap_redacts_and_emits_json(tmp_path: Path) -> None:
    _write_trace(tmp_path, [1, 2, 4])

    result = runner.invoke(
        app,
        ["trace", "latest", "--dir", str(tmp_path), "--gaps", "--redact", "--json"],
    )

    assert result.exit_code == 2
    assert "[REDACTED]" in result.stdout
    assert "Bearer top-secret" not in result.stdout
    assert "Gap in s_cli: missing 3..3" in result.stdout


def test_trace_summarizes_timeline_and_replays(tmp_path: Path) -> None:
    _write_trace(tmp_path, [1, 2, 3])

    timeline = runner.invoke(
        app,
        ["trace", "cli", "--dir", str(tmp_path), "--timeline"],
    )
    assert timeline.exit_code == 0
    assert "Timeline" in timeline.stdout
    assert "event_job_submitted" in timeline.stdout

    replay = runner.invoke(
        app,
        ["trace", "latest", "--dir", str(tmp_path), "--replay", "--speed", "0"],
    )
    assert replay.exit_code == 0
    assert "job_1" in replay.stdout
    assert "job_3" in replay.stdout


def test_trace_without_gaps_is_contiguous(tmp_path: Path) -> None:
    _write_trace(tmp_path, [1, 2, 3])

    result = runner.invoke(app, ["trace", "latest", "--dir", str(tmp_path), "--gaps"])

    assert result.exit_code == 0
    assert "No sequence gaps" in result.stdout
