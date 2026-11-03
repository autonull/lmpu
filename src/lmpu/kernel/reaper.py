"""LMPU Kernel - Reaper for cleaning up orphaned resources."""

from __future__ import annotations

import asyncio
import ulid
import time
from typing import Any, Callable, Optional

from .jobs import JobTable, JobStatus
from .events import EventLog
from ..memory.buffers import BufferManager
from ..protocol.messages import EventKind, Event, SleepingData


class Reaper:
    """Cleans up orphaned sessions, dead workers, expired buffers, timed-out jobs."""

    def __init__(
        self,
        job_table: JobTable,
        event_log: EventLog,
        buffer_manager: BufferManager,
        interval_ms: int = 1000,
        job_timeout_ms: int = 300000,  # 5 min default
        buffer_ttl_ms: int = 3600000,  # 1 hour default
        session_grace_ms: int = 5000,  # 5 sec grace after disconnect
    ):
        self.job_table = job_table
        self.event_log = event_log
        self.buffer_manager = buffer_manager
        self.interval_ms = interval_ms
        self.job_timeout_ms = job_timeout_ms
        self.buffer_ttl_ms = buffer_ttl_ms
        self.session_grace_ms = session_grace_ms
        
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._orphaned_sessions: dict[str, float] = {}  # session_id -> disconnect_time
        self._worker_health: dict[str, float] = {}  # worker_id -> last_heartbeat

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def mark_session_disconnected(self, session_id: str) -> None:
        """Mark session as disconnected (grace period starts)."""
        self._orphaned_sessions[session_id] = time.time()

    def mark_session_reconnected(self, session_id: str) -> None:
        """Mark session as reconnected."""
        self._orphaned_sessions.pop(session_id, None)

    def update_worker_heartbeat(self, worker_id: str) -> None:
        self._worker_health[worker_id] = time.time()

    def get_worker_health(self, worker_id: str) -> Optional[float]:
        return self._worker_health.get(worker_id)

    async def _run(self) -> None:
        while self._running:
            try:
                await self._reap_once()
            except Exception as e:
                # Log but don't crash
                print(f"Reaper error: {e}")
            await asyncio.sleep(self.interval_ms / 1000)

    async def _reap_once(self) -> None:
        now = time.time()
        
        # 1. Reap expired jobs
        await self._reap_jobs(now)
        
        # 2. Reap orphaned sessions
        await self._reap_sessions(now)
        
        # 3. Reap expired buffers
        await self._reap_buffers(now)
        
        # 4. Check worker health
        await self._check_workers(now)

    async def _reap_jobs(self, now: float) -> None:
        """Reap timed-out and expired jobs."""
        # Check for timed-out running jobs
        for job in list(self.job_table._jobs.values()):
            if job.status == JobStatus.RUNNING and job.deadline_ms:
                elapsed_ms = (now - job.started_at) * 1000 if job.started_at else 0
                if elapsed_ms > job.deadline_ms:
                    # Trigger timeout
                    await self._timeout_job(job)

            # Check for expired terminal jobs
            if job.is_terminal() and job.completed_at:
                age_ms = (now - job.completed_at) * 1000
                if age_ms > job.ttl_ms:
                    self.job_table.remove(job.job_id)

    async def _timeout_job(self, job) -> None:
        """Handle job timeout."""
        from ..protocol.messages import ErrorCode
        job.status = JobStatus.FAILED
        job.completed_at = time.time()
        job.error = {
            "code": ErrorCode.TIMEOUT.value,
            "message": "Job timed out",
            "retriable": True,
        }
        # Emit event
        event = Event(
            event_id=str(ulid.new()),
            seq=0,  # Will be set by event log
            session=job.session_id,
            job_id=job.job_id,
            type=EventKind.JOB_FAILED,
            ts_monotonic=time.monotonic(),
            data=JobFailedData(
                job_id=job.job_id,
                error_code=ErrorCode.TIMEOUT,
                retriable=True,
                detail="Job exceeded deadline",
            ),
        )
        self.event_log.append(job.session_id, event)

    async def _reap_sessions(self, now: float) -> None:
        """Clean up orphaned sessions."""
        for session_id, disconnect_time in list(self._orphaned_sessions.items()):
            if now - disconnect_time > self.session_grace_ms / 1000:
                # Cancel or detach jobs
                jobs = self.job_table.get_by_session(session_id)
                for job in jobs:
                    if not job.is_terminal():
                        # Default: cancel
                        self.job_table.transition(job.job_id, JobStatus.CANCEL_REQUESTED)
                        # Will be picked up by worker
                
                # Clean up buffers
                self.buffer_manager.cleanup_session(session_id)
                
                # Remove from tracking
                self._orphaned_sessions.pop(session_id, None)
                self.event_log.remove_session(session_id)
                self.job_table._session_jobs.pop(session_id, None)

    async def _reap_buffers(self, now: float) -> None:
        """Clean up expired buffers."""
        # BufferManager handles TTL internally
        # This is a fallback for orphaned buffers
        pass

    async def _check_workers(self, now: float) -> None:
        """Check for dead workers."""
        # Workers report heartbeats; if missing for 3 intervals, mark unhealthy
        # This is handled by WorkerPool, but we track here for visibility
        dead_workers = []
        for worker_id, last_heartbeat in self._worker_health.items():
            if now - last_heartbeat > 3:  # 3 seconds without heartbeat
                dead_workers.append(worker_id)
        
        for worker_id in dead_workers:
            # WorkerPool will handle restart
            pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "orphaned_sessions": len(self._orphaned_sessions),
            "tracked_workers": len(self._worker_health),
        }