"""LMPU Kernel - Job state machine and job table."""

from __future__ import annotations

import time
import ulid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..protocol.messages import (
    JobStatus,
    ModelJobStatus,
    EventKind,
    Event,
    JobSubmittedData,
    JobStartedData,
    ProgressData,
    JobCompletedData,
    JobFailedData,
    JobCancelledData,
    NeedsMoreInputData,
    SleepCondition,
    BufferRef,
)
from ..memory.grants import GrantManager


class TransitionError(Exception):
    """Invalid state transition."""
    pass


# Valid state transitions
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.CREATED: {JobStatus.PENDING, JobStatus.CANCEL_REQUESTED},
    JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.WAITING_INPUT,
        JobStatus.WAITING_EVENT,
        JobStatus.CANCEL_REQUESTED,
    },
    JobStatus.WAITING_INPUT: {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED},
    JobStatus.WAITING_EVENT: {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED},
    JobStatus.CANCEL_REQUESTED: {JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),  # Terminal
    JobStatus.FAILED: set(),     # Terminal
    JobStatus.CANCELLED: set(),  # Terminal
}

TERMINAL_STATES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def is_terminal(status: JobStatus) -> bool:
    return status in TERMINAL_STATES


def can_transition(from_status: JobStatus, to_status: JobStatus) -> bool:
    return to_status in VALID_TRANSITIONS.get(from_status, set())


def transition(from_status: JobStatus, to_status: JobStatus) -> None:
    """Validate and perform transition, raising if invalid."""
    if not can_transition(from_status, to_status):
        raise TransitionError(f"Invalid transition: {from_status} -> {to_status}")


@dataclass
class JobRecord:
    """Complete job record."""
    job_id: str
    session_id: str
    processor: str
    op: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    program: Optional[str] = None
    batch: Optional[dict[str, Any]] = None
    config: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.CREATED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[str] = None
    queue_position: int = 0
    priority: int = 5
    timeout_ms: Optional[int] = None
    deadline_ms: Optional[int] = None
    request_hash: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    buffer_refs: list[BufferRef] = field(default_factory=list)
    grants: list[str] = field(default_factory=list)  # grant IDs
    progress: float = 0.0
    progress_message: Optional[str] = None
    retry_count: int = 0
    idempotent: bool = False
    ttl_ms: int = 3600000
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_model_status(self) -> ModelJobStatus:
        """Convert internal status to model-facing status."""
        if self.status in (JobStatus.WAITING_INPUT, JobStatus.WAITING_EVENT):
            return ModelJobStatus.SLEEPING
        if self.status == JobStatus.CREATED:
            return ModelJobStatus.PENDING
        if self.status == JobStatus.CANCEL_REQUESTED:
            return ModelJobStatus.CANCELLED
        return ModelJobStatus(self.status.value)

    def is_terminal(self) -> bool:
        return is_terminal(self.status)

    def elapsed_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return int((end - self.started_at) * 1000)


class JobTable:
    """Thread-safe job table with idempotency and state machine."""

    def __init__(self, grant_manager: Optional[GrantManager] = None):
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_index: dict[str, str] = {}  # request_hash -> job_id
        self._session_jobs: dict[str, set[str]] = {}  # session_id -> job_ids
        self._grant_manager = grant_manager or GrantManager()
        self._lock = None  # asyncio.Lock when used in async context

    def _make_job_id(self) -> str:
        return f"job_{str(ulid.new())}"

    def _compute_request_hash(self, params: dict[str, Any]) -> str:
        import hashlib
        import json
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def create_job(
        self,
        session_id: str,
        processor: str,
        op: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        program: Optional[str] = None,
        batch: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> tuple[JobRecord, bool]:
        """Create a new job or return existing idempotent job.
        
        Returns:
            Tuple of (job_record, is_new)
        """
        config = config or {}
        request_hash = self._compute_request_hash({
            "processor": processor,
            "op": op,
            "inputs": inputs,
            "outputs": outputs,
            "program": program,
            "batch": batch,
            "config": config,
        })

        # Check idempotency
        if job_id and job_id in self._jobs:
            existing = self._jobs[job_id]
            if existing.request_hash == request_hash:
                return existing, False
            raise ValueError(f"Idempotency conflict: job_id {job_id} exists with different request")

        if request_hash in self._idempotency_index:
            existing_id = self._idempotency_index[request_hash]
            if existing_id in self._jobs:
                return self._jobs[existing_id], False

        # Create new job
        job_id = job_id or self._make_job_id()
        record = JobRecord(
            job_id=job_id,
            session_id=session_id,
            processor=processor,
            op=op,
            inputs=inputs,
            outputs=outputs,
            program=program,
            batch=batch,
            config=config,
            status=JobStatus.CREATED,
            request_hash=request_hash,
            priority=config.get("priority", 5),
            timeout_ms=config.get("timeout_ms"),
            deadline_ms=config.get("deadline_ms"),
            idempotent=config.get("idempotent", False),
            ttl_ms=config.get("ttl_ms", 3600000),
        )

        self._jobs[job_id] = record
        self._idempotency_index[request_hash] = job_id
        self._session_jobs.setdefault(session_id, set()).add(job_id)
        return record, True

    def transition(self, job_id: str, new_status: JobStatus) -> JobRecord:
        """Transition job to new status."""
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        
        transition(job.status, new_status)
        old_status = job.status
        job.status = new_status
        
        if new_status == JobStatus.RUNNING and old_status == JobStatus.PENDING:
            job.started_at = time.time()
        elif new_status in TERMINAL_STATES:
            job.completed_at = time.time()
        
        return job

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def get_by_session(self, session_id: str) -> list[JobRecord]:
        job_ids = self._session_jobs.get(session_id, set())
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    def list_jobs(
        self,
        session_id: Optional[str] = None,
        status: Optional[list[ModelJobStatus]] = None,
        limit: int = 100,
    ) -> list[JobRecord]:
        if session_id:
            jobs = self.get_by_session(session_id)
        else:
            jobs = list(self._jobs.values())
        
        if status:
            status_set = set(s.value for s in status)
            jobs = [j for j in jobs if j.status.value in status_set]
        
        # Sort by priority (desc), then created_at (asc)
        jobs.sort(key=lambda j: (-j.priority, j.created_at))
        return jobs[:limit]

    def add_grant(self, job_id: str, grant_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.grants.append(grant_id)

    def release_grants(self, job_id: str) -> int:
        job = self._jobs.get(job_id)
        if not job:
            return 0
        count = 0
        for grant_id in job.grants:
            if self._grant_manager.revoke(grant_id):
                count += 1
        job.grants.clear()
        return count

    def remove(self, job_id: str) -> bool:
        job = self._jobs.pop(job_id, None)
        if job:
            self._idempotency_index.pop(job.request_hash, None)
            self._session_jobs.get(job.session_id, set()).discard(job_id)
            return True
        return False

    def cleanup_expired(self, now: Optional[float] = None) -> list[str]:
        now = now or time.time()
        expired = []
        for job_id, job in list(self._jobs.items()):
            if job.is_terminal():
                age_ms = (now - job.completed_at) * 1000 if job.completed_at else 0
                if age_ms > job.ttl_ms:
                    self.remove(job_id)
                    expired.append(job_id)
        return expired

    def __len__(self) -> int:
        return len(self._jobs)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._jobs