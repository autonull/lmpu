"""Tests for LMPU protocol messages."""

import pytest
from lmpu.protocol.messages import (
    SubmitMessage,
    PollMessage,
    SleepMessage,
    SleepCondition,
    ModelMessageKind,
    parse_model_message,
    HelloParams,
    HelloResult,
    SubmitParams,
    SubmitResult,
    BufferRef,
    EventKind,
    Event,
    JobSubmittedData,
)


class TestModelMessages:
    """Test model-plane message parsing."""
    
    def test_submit_message(self):
        msg = SubmitMessage(
            job_id="job_123",
            processor="vector.engine",
            op="matmul",
            inputs={"A": {"buffer": "buf_A"}, "B": {"buffer": "buf_B"}},
            outputs={"C": {"buffer": "buf_C"}},
        )
        assert msg.kind == ModelMessageKind.SUBMIT
        assert msg.job_id == "job_123"
        assert msg.processor == "vector.engine"
        assert msg.op == "matmul"
    
    def test_poll_message(self):
        msg = PollMessage(job_id="job_123")
        assert msg.kind == ModelMessageKind.POLL
        assert msg.job_id == "job_123"
    
    def test_sleep_message(self):
        condition = SleepCondition(
            all=["job_1", "job_2"],
            deadline_ms=30000,
        )
        msg = SleepMessage(until=condition, level="S1")
        assert msg.kind == ModelMessageKind.SLEEP
        assert msg.until.all == ["job_1", "job_2"]
        assert msg.level == "S1"
    
    def test_parse_model_message(self):
        data = {
            "version": 1,
            "kind": "submit",
            "job_id": "job_123",
            "processor": "pure.func",
            "op": "bigint_add",
            "inputs": {"a": 1, "b": 2},
        }
        msg = parse_model_message(data)
        assert isinstance(msg, SubmitMessage)
        assert msg.job_id == "job_123"
        assert msg.op == "bigint_add"


class TestControlPlaneMessages:
    """Test control-plane message models."""
    
    def test_hello_params(self):
        params = HelloParams(protocol="lmpu/1", resume_from=10)
        assert params.protocol == "lmpu/1"
        assert params.resume_from == 10
    
    def test_submit_params(self):
        params = SubmitParams(
            job_id="job_123",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
        )
        assert params.job_id == "job_123"
        assert params.processor == "pure.func"
    
    def test_submit_result(self):
        result = SubmitResult(
            job_id="job_123",
            status="pending",
            queue_position=1,
            seq=5,
        )
        assert result.job_id == "job_123"
        assert result.status == "pending"
    
    def test_buffer_ref(self):
        ref = BufferRef(
            buffer_id="buf_123",
            dtype="float32",
            shape=[100, 100],
            bytes=40000,
            checksum="blake3:abc123",
            owner="session_1",
            created_at=1234567890.0,
        )
        assert ref.buffer_id == "buf_123"
        assert ref.shape == [100, 100]


class TestEvents:
    """Test event models."""
    
    def test_job_submitted_event(self):
        data = JobSubmittedData(
            job_id="job_123",
            processor="pure.func",
            queue_position=1,
        )
        event = Event(
            event_id="evt_123",
            seq=1,
            session="session_1",
            job_id="job_123",
            type=EventKind.JOB_SUBMITTED,
            ts_monotonic=123.0,
            data=data,
        )
        assert event.type == EventKind.JOB_SUBMITTED
        assert event.job_id == "job_123"
        assert event.session == "session_1"