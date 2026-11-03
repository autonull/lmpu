"""Tests for LMPU kernel module."""

import pytest
import time

from lmpu.kernel.jobs import (
    JobTable,
    JobRecord,
    JobStatus,
    TransitionError,
    can_transition,
    is_terminal,
    transition,
)
from lmpu.kernel.events import EventLog, SessionEventLog
from lmpu.kernel.clock import RealClock, FakeClock
from lmpu.protocol.messages import ModelJobStatus


class TestJobStateMachine:
    """Test job state machine transitions."""
    
    def test_valid_transitions(self):
        # Test all valid transitions
        assert can_transition(JobStatus.CREATED, JobStatus.PENDING)
        assert can_transition(JobStatus.CREATED, JobStatus.CANCEL_REQUESTED)
        
        assert can_transition(JobStatus.PENDING, JobStatus.RUNNING)
        assert can_transition(JobStatus.PENDING, JobStatus.CANCEL_REQUESTED)
        assert can_transition(JobStatus.PENDING, JobStatus.CANCELLED)
        
        assert can_transition(JobStatus.RUNNING, JobStatus.COMPLETED)
        assert can_transition(JobStatus.RUNNING, JobStatus.FAILED)
        assert can_transition(JobStatus.RUNNING, JobStatus.WAITING_INPUT)
        assert can_transition(JobStatus.RUNNING, JobStatus.WAITING_EVENT)
        assert can_transition(JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED)
        
        assert can_transition(JobStatus.WAITING_INPUT, JobStatus.RUNNING)
        assert can_transition(JobStatus.WAITING_INPUT, JobStatus.CANCEL_REQUESTED)
        assert can_transition(JobStatus.WAITING_INPUT, JobStatus.CANCELLED)
        
        assert can_transition(JobStatus.WAITING_EVENT, JobStatus.RUNNING)
        assert can_transition(JobStatus.WAITING_EVENT, JobStatus.CANCEL_REQUESTED)
        assert can_transition(JobStatus.WAITING_EVENT, JobStatus.CANCELLED)
        
        assert can_transition(JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED)
        assert can_transition(JobStatus.CANCEL_REQUESTED, JobStatus.COMPLETED)
        assert can_transition(JobStatus.CANCEL_REQUESTED, JobStatus.FAILED)
    
    def test_invalid_transitions(self):
        # Terminal states should not transition
        assert not can_transition(JobStatus.COMPLETED, JobStatus.RUNNING)
        assert not can_transition(JobStatus.FAILED, JobStatus.PENDING)
        assert not can_transition(JobStatus.CANCELLED, JobStatus.RUNNING)
        
        # Cannot go backwards
        assert not can_transition(JobStatus.RUNNING, JobStatus.PENDING)
        assert not can_transition(JobStatus.COMPLETED, JobStatus.CREATED)
    
    def test_transition_function(self):
        # Valid transition should not raise
        transition(JobStatus.PENDING, JobStatus.RUNNING)
        
        # Invalid transition should raise
        with pytest.raises(TransitionError):
            transition(JobStatus.COMPLETED, JobStatus.RUNNING)
    
    def test_is_terminal(self):
        assert is_terminal(JobStatus.COMPLETED)
        assert is_terminal(JobStatus.FAILED)
        assert is_terminal(JobStatus.CANCELLED)
        
        assert not is_terminal(JobStatus.CREATED)
        assert not is_terminal(JobStatus.PENDING)
        assert not is_terminal(JobStatus.RUNNING)
        assert not is_terminal(JobStatus.WAITING_INPUT)
        assert not is_terminal(JobStatus.WAITING_EVENT)
        assert not is_terminal(JobStatus.CANCEL_REQUESTED)


class TestJobTable:
    """Test job table operations."""
    
    def test_create_job(self):
        table = JobTable()
        
        job, is_new = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
        )
        
        assert is_new is True
        assert job.job_id.startswith("job_")
        assert job.session_id == "session_1"
        assert job.processor == "pure.func"
        assert job.op == "bigint_add"
        assert job.status == JobStatus.CREATED
    
    def test_idempotent_create(self):
        table = JobTable()
        
        job1, is_new1 = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
            job_id="job_123",
        )
        
        job2, is_new2 = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
            job_id="job_123",
        )
        
        assert is_new1 is True
        assert is_new2 is False
        assert job1.job_id == job2.job_id
    
    def test_idempotency_conflict(self):
        table = JobTable()
        
        table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
            job_id="job_123",
        )
        
        # Different inputs but same job_id should conflict
        with pytest.raises(ValueError, match="Idempotency conflict"):
            table.create_job(
                session_id="session_1",
                processor="pure.func",
                op="bigint_mul",  # Different op
                inputs={"a": 1, "b": 2},
                outputs={},
                job_id="job_123",
            )
    
    def test_transition(self):
        table = JobTable()
        
        job, _ = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
        )
        
        # Valid transitions
        job = table.transition(job.job_id, JobStatus.PENDING)
        assert job.status == JobStatus.PENDING
        
        job = table.transition(job.job_id, JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        
        job = table.transition(job.job_id, JobStatus.COMPLETED)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
    
    def test_invalid_transition(self):
        table = JobTable()
        
        job, _ = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1, "b": 2},
            outputs={},
        )
        
        table.transition(job.job_id, JobStatus.PENDING)
        
        # Invalid: COMPLETED -> RUNNING
        with pytest.raises(TransitionError):
            table.transition(job.job_id, JobStatus.COMPLETED)
            table.transition(job.job_id, JobStatus.RUNNING)
    
    def test_get_by_session(self):
        table = JobTable()
        
        table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1},
            outputs={},
        )
        table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_mul",
            inputs={"a": 2},
            outputs={},
        )
        table.create_job(
            session_id="session_2",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 3},
            outputs={},
        )
        
        jobs_s1 = table.get_by_session("session_1")
        assert len(jobs_s1) == 2
        
        jobs_s2 = table.get_by_session("session_2")
        assert len(jobs_s2) == 1
    
    def test_list_jobs_with_filter(self):
        table = JobTable()
        
        job1, _ = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={"a": 1},
            outputs={},
        )
        table.transition(job1.job_id, JobStatus.PENDING)
        table.transition(job1.job_id, JobStatus.RUNNING)
        table.transition(job1.job_id, JobStatus.COMPLETED)
        
        job2, _ = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_mul",
            inputs={"a": 2},
            outputs={},
        )
        
        # Filter by status
        completed = table.list_jobs(
            session_id="session_1",
            status=[ModelJobStatus.COMPLETED],
        )
        assert len(completed) == 1
        
        # Transition job2 to PENDING
        table.transition(job2.job_id, JobStatus.PENDING)
        
        pending = table.list_jobs(
            session_id="session_1",
            status=[ModelJobStatus.PENDING],
        )
        assert len(pending) == 1
    
    def test_model_status_mapping(self):
        table = JobTable()
        
        job, _ = table.create_job(
            session_id="session_1",
            processor="pure.func",
            op="bigint_add",
            inputs={},
            outputs={},
        )
        
        assert job.to_model_status() == ModelJobStatus.PENDING
        
        table.transition(job.job_id, JobStatus.PENDING)
        table.transition(job.job_id, JobStatus.RUNNING)
        assert job.to_model_status() == ModelJobStatus.RUNNING
        
        table.transition(job.job_id, JobStatus.WAITING_INPUT)
        assert job.to_model_status() == ModelJobStatus.SLEEPING
        
        table.transition(job.job_id, JobStatus.RUNNING)
        table.transition(job.job_id, JobStatus.COMPLETED)
        assert job.to_model_status() == ModelJobStatus.COMPLETED


class TestEventLog:
    """Test event log."""
    
    def test_append_and_get(self):
        log = EventLog()
        
        import ulid
        from lmpu.protocol.messages import Event, EventKind, JobSubmittedData
        
        event = Event(
            event_id=str(ulid.new()),
            seq=0,
            session="session_1",
            job_id="job_1",
            type=EventKind.JOB_SUBMITTED,
            ts_monotonic=time.monotonic(),
            data=JobSubmittedData(
                job_id="job_1",
                processor="pure.func",
                queue_position=1,
            ),
        )
        
        appended = log.append("session_1", event)
        assert appended.seq == 1
        
        events = log.get_since("session_1", 0)
        assert len(events) == 1
        assert events[0].seq == 1
    
    def test_ack(self):
        log = EventLog()
        
        import ulid
        from lmpu.protocol.messages import Event, EventKind, JobSubmittedData
        
        for i in range(5):
            event = Event(
                event_id=str(ulid.new()),
                seq=0,
                session="session_1",
                job_id=f"job_{i}",
                type=EventKind.JOB_SUBMITTED,
                ts_monotonic=time.monotonic(),
                data=JobSubmittedData(
                    job_id=f"job_{i}",
                    processor="pure.func",
                    queue_position=i,
                ),
            )
            log.append("session_1", event)
        
        # Ack up to seq 3
        acked = log.ack("session_1", 3)
        assert acked == 3
        
        # Get unacked
        unacked = log.get_session_ids()  # This is wrong, let me fix
        # Actually test the session log directly
        session_log = log._sessions["session_1"]
        unacked_events = session_log.get_unacked()
        assert len(unacked_events) == 2  # seq 4 and 5
    
    def test_resume_gap_detection(self):
        log = EventLog()
        
        import ulid
        from lmpu.protocol.messages import Event, EventKind, JobSubmittedData
        
        # Add events 1-10
        for i in range(1, 11):
            event = Event(
                event_id=str(ulid.new()),
                seq=0,
                session="session_1",
                job_id=f"job_{i}",
                type=EventKind.JOB_SUBMITTED,
                ts_monotonic=time.monotonic(),
                data=JobSubmittedData(
                    job_id=f"job_{i}",
                    processor="pure.func",
                    queue_position=i,
                ),
            )
            log.append("session_1", event)
        
        # Trim to keep only last 5
        log._sessions["session_1"].trim_to_size(5)
        
        # Resume from 3 should have gap (earliest is 6)
        has_gap = log.has_gap("session_1", 3)
        assert has_gap is True
        
        # Resume from 8 should not have gap
        has_gap = log.has_gap("session_1", 8)
        assert has_gap is False


class TestClocks:
    """Test clock implementations."""
    
    def test_real_clock(self):
        clock = RealClock()
        t1 = clock.now()
        t2 = clock.monotonic()
        assert t1 > 0
        assert t2 > 0
    
    def test_fake_clock(self):
        clock = FakeClock(1000.0)
        
        assert clock.now() == 1000.0
        assert clock.monotonic() == 1000.0
        
        clock.advance(10.0)
        assert clock.now() == 1010.0
        assert clock.monotonic() == 1010.0
    
    def test_fake_clock_timers(self):
        clock = FakeClock(0.0)
        results = []
        
        def callback(value):
            results.append(value)
        
        handle = clock.call_later(5.0, callback, "hello")
        clock.advance(3.0)
        assert len(results) == 0
        
        clock.advance(3.0)
        assert len(results) == 1
        assert results[0] == "hello"
        
        # Cancel timer
        handle2 = clock.call_later(10.0, callback, "world")
        clock.cancel(handle2)
        clock.advance(20.0)
        assert len(results) == 1  # world should not have been called