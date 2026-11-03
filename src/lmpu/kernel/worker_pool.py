"""LMPU Kernel - Worker pool and process management."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import ulid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..protocol.messages import (
    EventKind,
    Event,
    JobStartedData,
    ProgressData,
    JobCompletedData,
    JobFailedData,
    JobCancelledData,
    NeedsMoreInputData,
    HeartbeatData,
)
import ulid
from ..kernel.events import EventLog
from ..kernel.jobs import JobTable, JobRecord, JobStatus
from ..kernel.clock import get_clock


@dataclass
class WorkerInfo:
    """Worker process information."""
    worker_id: str
    processor: str
    process: asyncio.subprocess.Process
    pid: int
    manifest: dict[str, Any]
    started_at: float
    last_heartbeat: float
    current_job: Optional[str] = None
    crash_count: int = 0


class WorkerPool:
    """Manages worker processes for processors."""

    def __init__(
        self,
        job_table: JobTable,
        event_log: EventLog,
        python_executable: str = sys.executable,
        worker_module: str = "lmpu.worker",
        max_workers_per_processor: int = 4,
        warm_pool_size: int = 1,
        heartbeat_interval: float = 1.0,
        crash_threshold: int = 3,
    ):
        self.job_table = job_table
        self.event_log = event_log
        self.python_executable = python_executable
        self.worker_module = worker_module
        self.max_workers_per_processor = max_workers_per_processor
        self.warm_pool_size = warm_pool_size
        self.heartbeat_interval = heartbeat_interval
        self.crash_threshold = crash_threshold

        self._workers: dict[str, WorkerInfo] = {}  # worker_id -> info
        self._processor_queues: dict[str, asyncio.Queue[str]] = {}  # processor -> worker_ids
        self._available_workers: dict[str, set[str]] = {}  # processor -> available worker_ids
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start warm worker pool."""
        # Pre-warm workers for built-in processors
        pass

    async def stop(self) -> None:
        """Stop all workers."""
        async with self._lock:
            for worker in list(self._workers.values()):
                await self._terminate_worker(worker.worker_id, graceful=True)
            self._workers.clear()
            self._processor_queues.clear()
            self._available_workers.clear()

    async def _spawn_worker(
        self,
        processor: str,
        manifest: dict[str, Any],
        plugin_path: Optional[str] = None,
    ) -> WorkerInfo:
        """Spawn a new worker process."""
        worker_id = f"worker_{str(ulid.new())}"
        
        # Build command
        cmd = [
            self.python_executable,
            "-m", self.worker_module,
            "--processor", processor,
            "--worker-id", worker_id,
        ]
        if plugin_path:
            cmd.extend(["--plugin", plugin_path])
        
        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        # Wait for ready message
        ready_line = await process.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"Worker {worker_id} failed to start")

        import orjson
        ready_msg = orjson.loads(ready_line)
        if ready_msg.get("type") != "ready":
            raise RuntimeError(f"Worker {worker_id} sent invalid ready: {ready_msg}")

        worker = WorkerInfo(
            worker_id=worker_id,
            processor=processor,
            process=process,
            pid=process.pid or 0,
            manifest=manifest,
            started_at=get_clock().monotonic(),
            last_heartbeat=get_clock().monotonic(),
        )
        
        self._workers[worker_id] = worker
        self._available_workers.setdefault(processor, set()).add(worker_id)
        
        # Start heartbeat monitor
        asyncio.create_task(self._monitor_worker(worker_id))
        
        return worker

    async def _monitor_worker(self, worker_id: str) -> None:
        """Monitor worker stdout/stderr and heartbeats."""
        worker = self._workers.get(worker_id)
        if not worker:
            return

        async def read_stdout():
            while True:
                line = await worker.process.stdout.readline()
                if not line:
                    break
                await self._handle_worker_message(worker_id, line)

        async def read_stderr():
            while True:
                line = await worker.process.stderr.readline()
                if not line:
                    break
                # Log stderr
                print(f"WORKER {worker_id} STDERR: {line.decode().strip()}")

        await asyncio.gather(read_stdout(), read_stderr())
        
        # Process exited
        await worker.process.wait()
        await self._handle_worker_exit(worker_id)

    async def _handle_worker_message(self, worker_id: str, line: bytes) -> None:
        """Handle message from worker."""
        import orjson
        try:
            msg = orjson.loads(line)
        except Exception:
            return

        msg_type = msg.get("type")
        worker = self._workers.get(worker_id)
        if not worker:
            return

        if msg_type == "heartbeat":
            worker.last_heartbeat = get_clock().monotonic()
            self.event_log.append(
                msg.get("session", "unknown"),
                Event(
                    event_id=str(ulid.new()),
                    seq=0,
                    session=msg.get("session", "unknown"),
                    job_id=msg.get("job_id"),
                    type=EventKind.HEARTBEAT,
                    ts_monotonic=get_clock().monotonic(),
                    data=HeartbeatData(worker=worker_id, timestamp=worker.last_heartbeat),
                ),
            )

        elif msg_type == "progress":
            job_id = msg.get("job_id")
            if job_id:
                job = self.job_table.get(job_id)
                if job:
                    job.progress = msg.get("fraction", 0.0)
                    job.progress_message = msg.get("message")
                    self.event_log.append(
                        job.session_id,
                        Event(
                            event_id=str(ulid.new()),
                            seq=0,
                            session=job.session_id,
                            job_id=job_id,
                            type=EventKind.PROGRESS,
                            ts_monotonic=get_clock().monotonic(),
                            data=ProgressData(
                                job_id=job_id,
                                fraction=msg.get("fraction", 0.0),
                                checkpoint=msg.get("checkpoint"),
                                message=msg.get("message"),
                            ),
                        ),
                    )

        elif msg_type == "needs_input":
            job_id = msg.get("job_id")
            if job_id:
                job = self.job_table.get(job_id)
                if job:
                    self.job_table.transition(job_id, JobStatus.WAITING_INPUT)
                    self.event_log.append(
                        job.session_id,
                        Event(
                            event_id=str(ulid.new()),
                            seq=0,
                            session=job.session_id,
                            job_id=job_id,
                            type=EventKind.NEEDS_MORE_INPUT,
                            ts_monotonic=get_clock().monotonic(),
                            data=NeedsMoreInputData(
                                job_id=job_id,
                                prompt=msg.get("prompt", ""),
                                fields=msg.get("fields", []),
                            ),
                        ),
                    )

        elif msg_type == "result":
            await self._handle_worker_result(worker_id, msg)

        elif msg_type == "error":
            await self._handle_worker_error(worker_id, msg)

    async def _handle_worker_result(self, worker_id: str, msg: dict[str, Any]) -> None:
        """Handle successful job result."""
        job_id = msg.get("job_id")
        job = self.job_table.get(job_id)
        if not job:
            return

        self.job_table.transition(job_id, JobStatus.COMPLETED)
        job.result = msg.get("result")
        job.buffer_refs = msg.get("buffer_refs", [])
        
        self.event_log.append(
            job.session_id,
            Event(
                event_id=str(ulid.new()),
                seq=0,
                session=job.session_id,
                job_id=job_id,
                type=EventKind.JOB_COMPLETED,
                ts_monotonic=get_clock().monotonic(),
                data=JobCompletedData(
                    job_id=job_id,
                    result=msg.get("result", {}),
                    buffer_refs=msg.get("buffer_refs", []),
                    checksum=msg.get("checksum", ""),
                    cache_hit=msg.get("cache_hit", False),
                ),
            ),
        )

        # Release worker
        self._release_worker(worker_id)

    async def _handle_worker_error(self, worker_id: str, msg: dict[str, Any]) -> None:
        """Handle job error."""
        job_id = msg.get("job_id")
        job = self.job_table.get(job_id)
        if not job:
            return

        error_code = msg.get("error_code", "internal")
        retriable = msg.get("retriable", False)
        
        self.job_table.transition(job_id, JobStatus.FAILED)
        job.error = {
            "code": error_code,
            "message": msg.get("message", "Worker error"),
            "retriable": retriable,
        }

        self.event_log.append(
            job.session_id,
            Event(
                event_id=str(ulid.new()),
                seq=0,
                session=job.session_id,
                job_id=job_id,
                type=EventKind.JOB_FAILED,
                ts_monotonic=get_clock().monotonic(),
                data=JobFailedData(
                    job_id=job_id,
                    error_code=error_code,
                    retriable=retriable,
                    detail=msg.get("message", "Worker error"),
                ),
            ),
        )

        worker = self._workers.get(worker_id)
        if worker:
            worker.crash_count += 1
            if worker.crash_count >= self.crash_threshold:
                # Quarantine processor
                await self._quarantine_processor(worker.processor)
        
        self._release_worker(worker_id)

    async def _handle_worker_exit(self, worker_id: str) -> None:
        """Handle worker process exit."""
        worker = self._workers.pop(worker_id, None)
        if not worker:
            return

        self._available_workers.get(worker.processor, set()).discard(worker_id)

        # If worker had a job, mark it failed
        if worker.current_job:
            job = self.job_table.get(worker.current_job)
            if job and not job.is_terminal():
                self.job_table.transition(job.job_id, JobStatus.FAILED)
                job.error = {
                    "code": "worker_crashed",
                    "message": "Worker process exited unexpectedly",
                    "retriable": True,
                }
                self.event_log.append(
                    job.session_id,
                    Event(
                        event_id=str(ulid.new()),
                        seq=0,
                        session=job.session_id,
                        job_id=job.job_id,
                        type=EventKind.JOB_FAILED,
                        ts_monotonic=get_clock().monotonic(),
                        data=JobFailedData(
                            job_id=job.job_id,
                            error_code="worker_crashed",
                            retriable=True,
                            detail="Worker process exited",
                        ),
                    ),
                )

    def _release_worker(self, worker_id: str) -> None:
        """Release worker back to pool."""
        worker = self._workers.get(worker_id)
        if worker:
            worker.current_job = None
            self._available_workers.setdefault(worker.processor, set()).add(worker_id)

    async def _terminate_worker(self, worker_id: str, graceful: bool = True) -> None:
        """Terminate a worker."""
        worker = self._workers.get(worker_id)
        if not worker:
            return

        if graceful:
            # Send shutdown command
            if worker.process.stdin:
                import orjson
                worker.process.stdin.write(orjson.dumps({"type": "shutdown"}) + b"\n")
                await worker.process.stdin.drain()
            
            try:
                await asyncio.wait_for(worker.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        if worker.process.returncode is None:
            worker.process.terminate()
            try:
                await asyncio.wait_for(worker.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                worker.process.kill()
                await worker.process.wait()

        self._available_workers.get(worker.processor, set()).discard(worker_id)
        self._workers.pop(worker_id, None)

    async def _quarantine_processor(self, processor: str) -> None:
        """Quarantine a processor due to repeated crashes."""
        # Terminate all workers for this processor
        for worker_id, worker in list(self._workers.items()):
            if worker.processor == processor:
                await self._terminate_worker(worker_id, graceful=False)
        
        # Emit resource warning
        # TODO: emit to all sessions or admin

    async def get_worker(self, processor: str, job: JobRecord) -> WorkerInfo:
        """Get an available worker for a processor, spawning if needed."""
        async with self._lock:
            available = self._available_workers.get(processor, set())
            
            if available:
                worker_id = available.pop()
                worker = self._workers[worker_id]
                worker.current_job = job.job_id
                worker.last_heartbeat = get_clock().monotonic()
                return worker

            # Need to spawn new worker
            if len([w for w in self._workers.values() if w.processor == processor]) >= self.max_workers_per_processor:
                # Wait for available worker
                while True:
                    available = self._available_workers.get(processor, set())
                    if available:
                        worker_id = available.pop()
                        worker = self._workers[worker_id]
                        worker.current_job = job.job_id
                        worker.last_heartbeat = get_clock().monotonic()
                        return worker
                    await asyncio.sleep(0.1)

            # Spawn new worker (manifest would come from registry)
            manifest = {"name": processor, "version": "0.1.0"}
            return await self._spawn_worker(processor, manifest)

    async def send_command(self, worker_id: str, command: dict[str, Any]) -> None:
        """Send command to worker."""
        worker = self._workers.get(worker_id)
        if not worker or not worker.process.stdin:
            raise RuntimeError(f"Worker {worker_id} not available")
        
        import orjson
        worker.process.stdin.write(orjson.dumps(command) + b"\n")
        await worker.process.stdin.drain()

    async def cancel_job(self, job_id: str) -> bool:
        """Send cancel to worker running job."""
        for worker in self._workers.values():
            if worker.current_job == job_id:
                await self.send_command(worker.worker_id, {"type": "cancel", "job_id": job_id})
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_workers": len(self._workers),
            "by_processor": {
                p: len(ids) for p, ids in self._available_workers.items()
            },
            "busy_workers": sum(1 for w in self._workers.values() if w.current_job),
        }