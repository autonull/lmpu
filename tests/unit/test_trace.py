"""Tests for JSONL trace emission, replay, and payload redaction."""

from __future__ import annotations

import json

import pytest

from lmpu.kernel.trace import (
    TraceConfig,
    TraceSink,
    read_trace,
    redact,
    replay_trace,
)
from lmpu.protocol.messages import (
    Event,
    EventKind,
    JobCompletedData,
    JobSubmittedData,
)

pytestmark = pytest.mark.unit


def _sink(tmp_path, **overrides) -> TraceSink:
    config = TraceConfig(trace_dir=str(tmp_path), **overrides)
    return TraceSink(config)


def _trace_file(tmp_path):
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    return files[0]


def _event(seq: int, session: str = "s_trace") -> Event:
    return Event(
        event_id=f"evt_{seq}",
        seq=seq,
        session=session,
        job_id="job_trace",
        type=EventKind.JOB_SUBMITTED,
        ts_monotonic=float(seq),
        data=JobSubmittedData(
            job_id="job_trace",
            processor="pure.func",
            queue_position=0,
        ),
    )


def test_emit_event_writes_one_json_object_per_line(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.start_trace("t_1", "s_trace")
    sink.emit_event(_event(1))
    sink.emit_job_state("s_trace", "job_trace", "PENDING", "RUNNING")
    sink.close()

    records = read_trace(_trace_file(tmp_path))

    assert all(isinstance(record, dict) for record in records)
    assert records[0]["type"] == "session_start"
    assert records[1]["seq"] == 1
    assert records[1]["type"] == "event_jobsubmitted"
    assert records[2]["type"] == "job_state"


def test_replay_reconstructs_contiguous_session_sequence(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.emit_event(_event(1))
    sink.emit_event(_event(2))
    sink.emit_event(_event(3))
    sink.close()

    replay = replay_trace(_trace_file(tmp_path))

    assert replay.session_sequences == {"s_trace": [1, 2, 3]}
    assert replay.gaps == {}
    assert replay.is_contiguous()


def test_replay_reports_sequence_gap(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.emit_event(_event(1))
    sink.emit_event(_event(3))
    sink.close()

    replay = replay_trace(_trace_file(tmp_path))

    assert replay.session_sequences == {"s_trace": [1, 3]}
    assert replay.gaps == {"s_trace": [[2, 3]]}
    assert not replay.is_contiguous()


def test_replay_keeps_sessions_separate(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.emit_event(_event(1, "s_a"))
    sink.emit_event(_event(1, "s_b"))
    sink.emit_event(_event(2, "s_a"))
    sink.close()

    replay = replay_trace(_trace_file(tmp_path))

    assert replay.session_sequences == {"s_a": [1, 2], "s_b": [1]}
    assert replay.is_contiguous()


def test_sensitive_payloads_are_redacted_by_default(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.emit(
        {
            "type": "security_probe",
            "session": "s_trace",
            "token": "super-secret",
            "nested": {"password": "hunter2", "env": {"API_KEY": "abc"}},
            "safe": "visible",
        }
    )
    sink.close()

    record = read_trace(_trace_file(tmp_path))[0]

    assert record["token"] == "[REDACTED]"
    assert record["nested"]["password"] == "[REDACTED]"
    assert record["nested"]["env"] == "[REDACTED]"
    assert record["safe"] == "visible"


def test_redaction_can_be_disabled(tmp_path) -> None:
    sink = _sink(tmp_path, redact_payloads=False)
    sink.emit({"type": "raw", "token": "visible"})
    sink.close()

    assert read_trace(_trace_file(tmp_path))[0]["token"] == "visible"


def test_redact_recurses_through_lists() -> None:
    value = {"items": [{"secret": "x"}, {"ok": "y"}]}

    assert redact(value) == {
        "items": [{"secret": "[REDACTED]"}, {"ok": "y"}],
    }


def test_invalid_trace_line_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid trace JSON at line 2"):
        read_trace(path)


def test_completed_event_payload_is_preserved(tmp_path) -> None:
    sink = _sink(tmp_path)
    sink.emit_event(
        Event(
            event_id="evt_done",
            seq=1,
            session="s_trace",
            job_id="job_trace",
            type=EventKind.JOB_COMPLETED,
            ts_monotonic=1.0,
            data=JobCompletedData(
                job_id="job_trace",
                result={"result": "42"},
                checksum="blake3:abc",
            ),
        )
    )
    sink.close()

    record = read_trace(_trace_file(tmp_path))[0]

    assert record["data"]["result"] == {"result": "42"}
    assert record["data"]["checksum"] == "blake3:abc"
    assert json.loads(json.dumps(record)) == record
