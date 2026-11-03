"""LMPU Kernel - Priority scheduler with aging and batching."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from ulid import ULID

from .jobs import JobRecord, JobStatus, JobTable
from ..protocol.messages import ExecutionMode


@dataclass
class ScheduledJob:
    """Job in scheduler queue with effective priority."""
    job: JobRecord
    effective_priority: int
    enqueued_at: float
    batch_key: Optional[str] = None

    def __lt__(self, other: "ScheduledJob") -> bool:
        # Higher priority first, then earlier enqueue time
        if self.effective_priority != other.effective_priority:
            return self.effective_priority > other.effective_priority
        return self.enqueued_at < other.enqueued_at


class Scheduler:
    """Priority scheduler with aging, per-processor limits, and batching."""

    def __init__(
        self,
        job_table: JobTable,
        aging_interval_ms: int = 100,
        aging_priority_boost: int = 1,
        batch_window_ms: int = 2,
        max_batch_size: int = 64,
    ):
        self.job_table = job_table
        self.aging_interval_ms = aging_interval_ms
        self.aging_priority_boost = aging_priority_boost
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        # Per-processor queues
        self._queues: dict[str, asyncio.PriorityQueue[ScheduledJob]] = defaultdict(
            lambda: asyncio.PriorityQueue()
        )
        # Global queue for cross-processor scheduling
        self._global_queue: asyncio.PriorityQueue[ScheduledJob] = asyncio.PriorityQueue()
        
        # Concurrency control
        self._processor_semaphores: dict[str, asyncio.Semaphore] = {}
        self._global_semaphore: Optional[asyncio.Semaphore] = None
        self._processor_limits: dict[str, int] = {}
        self._global_limit: Optional[int] = None

        # Batching
        self._batch_buffers: dict[str, list[ScheduledJob]] = defaultdict(list)
        self._batch_timers: dict[str, asyncio.Task] = {}

        # Worker assignment
        self._job_to_worker: dict[str, str] = {}

    def set_processor_limit(self, processor: str, limit: int) -> None:
        """Set concurrency limit for a processor."""
        self._processor_limits[processor] = limit
        self._processor_semaphores[processor] = asyncio.Semaphore(limit)

    def set_global_limit(self, limit: int) -> None:
        """Set global concurrency limit."""
        self._global_limit = limit
        self._global_semaphore = asyncio.Semaphore(limit)

    def _get_semaphore(self, processor: str) -> asyncio.Semaphore:
        return self._processor_semaphores.get(processor, asyncio.Semaphore(1))

    def _compute_batch_key(self, job: JobRecord) -> Optional[str]:
        """Compute batch key for compatible jobs."""
        config = job.config or {}
        if config.get("mode") != ExecutionMode.ASYNC.value:
            return None
        if job.batch:
            return None  # Already a batch
        # Batch by processor, op, and compatible config
        return f"{job.processor}:{job.op}:{hash(frozenset(config.items()))}"

    def _compute_effective_priority(self, job: JobRecord, now: float) -> int:
        """Compute effective priority with aging."""
        base = job.priority
        if job.status == JobStatus.PENDING:
            wait_time_ms = (now - job.created_at) * 1000
            boost = int(wait_time_ms / self.aging_interval_ms) * self.aging_priority_boost
            return min(base + boost, 9)  # Cap at max priority
        return base

    async def enqueue(self, job: JobRecord) -> None:
        """Add job to scheduler queue."""
        now = time.time()
        effective_priority = self._compute_effective_priority(job, now)
        
        scheduled = ScheduledJob(
            job=job,
            effective_priority=effective_priority,
            enqueued_at=now,
        )

        batch_key = self._compute_batch_key(job)
        if batch_key:
            scheduled.batch_key = batch_key
            self._batch_buffers[batch_key].append(scheduled)
            # Start batch timer if not already running
            if batch_key not in self._batch_timers:
                self._batch_timers[batch_key] = asyncio.create_task(
                    self._flush_batch(batch_key)
                )
        else:
            # Direct to processor queue
            await self._queues[job.processor].put(scheduled)

    async def _flush_batch(self, batch_key: str) -> None:
        """Flush batch buffer after window expires."""
        await asyncio.sleep(self.batch_window_ms / 1000)
        jobs = self._batch_buffers.pop(batch_key, [])
        self._batch_timers.pop(batch_key, None)
        
        if not jobs:
            return

        # Sort by priority and enqueue time
        jobs.sort()
        
        # Take up to max_batch_size
        batch_jobs = jobs[:self.max_batch_size]
        remaining = jobs[self.max_batch_size:]

        # Re-enqueue remaining
        for job in remaining:
            await self._queues[job.job.processor].put(job)

        # Create batch job record
        if len(batch_jobs) > 1:
            # Submit as batch to processor
            first = batch_jobs[0]
            batch_config = {
                "items": [
                    {
                        "inputs": j.job.inputs,
                        "config": j.job.config,
                    }
                    for j in batch_jobs
                ],
                "max_parallel": len(batch_jobs),
            }
            # Update first job to be the batch
            first.job.batch = batch_config
            first.job.config = {**first.job.config, "mode": ExecutionMode.ASYNC.value}
            await self._queues[first.job.processor].put(first)
        else:
            await self._queues[batch_jobs[0].job.processor].put(batch_jobs[0])

    async def dequeue(self, processor: str) -> Optional[ScheduledJob]:
        """Get next job for processor."""
        queue = self._queues.get(processor)
        if not queue or queue.empty():
            return None
        return await queue.get()

    async def acquire_slot(self, processor: str) -> None:
        """Acquire concurrency slot."""
        sem = self._get_semaphore(processor)
        await sem.acquire()
        if self._global_semaphore:
            await self._global_semaphore.acquire()

    def release_slot(self, processor: str) -> None:
        """Release concurrency slot."""
        sem = self._get_semaphore(processor)
        sem.release()
        if self._global_semaphore:
            self._global_semaphore.release()

    def get_queue_depth(self, processor: str) -> int:
        queue = self._queues.get(processor)
        return queue.qsize() if queue else 0

    def get_all_queue_depths(self) -> dict[str, int]:
        return {p: q.qsize() for p, q in self._queues.items()}

    def requeue(self, job: JobRecord) -> None:
        """Re-queue a job (e.g., after worker failure)."""
        job.status = JobStatus.PENDING
        job.queue_position = self.get_queue_depth(job.processor) + 1
        asyncio.create_task(self.enqueue(job))

    def cancel_job(self, job_id: str) -> bool:
        """Remove job from queues if still pending."""
        # Note: PriorityQueue doesn't support removal
        # Job will be filtered when dequeued
        return True

    async def close(self) -> None:
        """Cancel batch timers."""
        for timer in self._batch_timers.values():
            timer.cancel()
        self._batch_timers.clear()