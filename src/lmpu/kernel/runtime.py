"""LMPU Kernel - Runtime facade and main entry point."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ulid
from pydantic import BaseModel

from ..kernel.clock import get_clock
from ..kernel.events import EventLog
from ..kernel.jobs import JobRecord, JobStatus, JobTable
from ..kernel.quotas import QuotaConfig, QuotaManager
from ..kernel.reaper import Reaper
from ..kernel.routing import CapabilityRegistry, ProcessorRegistration, route_job
from ..kernel.scheduler import Scheduler
from ..kernel.trace import TraceConfig, init_trace_sink
from ..kernel.worker_pool import WorkerPool
from ..memory.buffers import BufferManager
from ..memory.grants import GrantManager
from ..memory.shm import SHMBackend
from ..processors.base import Processor
from ..protocol.errors import (
    BufferError,
    JobError,
    LMPUError,
    ProtocolError,
    ValidationError,
)
from ..protocol.messages import (
    PROTOCOL_VERSION,
    AckParams,
    AckResult,
    BufferChecksumParams,
    BufferCreateParams,
    BufferDeleteParams,
    BufferImportParams,
    BufferRef,
    BufferStatParams,
    BufferView,
    CancelParams,
    CancelResult,
    CapabilitiesParams,
    CapabilitiesResult,
    CapabilityDescriptor,
    CloseParams,
    CloseResult,
    ConfigureParams,
    ErrorCode,
    Event,
    EventKind,
    HealthResult,
    HelloParams,
    HelloResult,
    JobCancelledData,
    JobsListParams,
    JobStartedData,
    JobSubmittedData,
    ModelJobStatus,
    PollParams,
    PollResult,
    ReadParams,
    RPCError,
    RPCNotification,
    RPCRequest,
    RPCResponse,
    SleepParams,
    SleepResult,
    SubmitParams,
    SubmitResult,
    WaitParams,
    WaitResult,
    WakeResult,
    WriteParams,
)


def _to_jsonable(value: Any) -> Any:
    """Convert pydantic models and dataclasses to JSON-safe primitives."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


@dataclass
class SessionState:
    """Per-session state."""

    session_id: str
    config: QuotaConfig
    sleep_registration: dict[str, Any] | None = None
    wake_event: asyncio.Event | None = None
    wake_result: dict[str, Any] | None = None
    event_cursor: int = 0
    injection_policy: dict[str, Any] = field(default_factory=dict)
    authorized_processors: set[str] | None = None


class Runtime:
    """LMPU Runtime - main kernel facade."""

    def __init__(
        self,
        socket_path: str = "/tmp/lmpu/run.sock",
        shm_dir: str = "/dev/shm/lmpu",
        config: dict[str, Any] | None = None,
    ):
        self.socket_path = socket_path
        self.shm_dir = shm_dir
        self.config = config or {}

        # Core components
        self.grant_manager = GrantManager()
        self.buffer_manager = BufferManager(
            backend=SHMBackend(self.shm_dir),
            grant_manager=self.grant_manager,
            default_ttl_ms=self.config.get("buffer_ttl_ms", 3600000),
        )
        self.job_table = JobTable(grant_manager=self.grant_manager)
        self.event_log = EventLog(
            max_events_per_session=self.config.get("event_ring", 10000),
        )
        self.scheduler = Scheduler(
            job_table=self.job_table,
            batch_window_ms=self.config.get("batch_window_ms", 2),
            max_batch_size=self.config.get("max_batch", 64),
        )
        self.quota_manager = QuotaManager()
        self.capability_registry = CapabilityRegistry()
        self.reaper = Reaper(
            job_table=self.job_table,
            event_log=self.event_log,
            buffer_manager=self.buffer_manager,
            interval_ms=self.config.get("reaper_interval_ms", 1000),
        )
        self.worker_pool = WorkerPool(
            job_table=self.job_table,
            event_log=self.event_log,
        )
        self.worker_pool.set_terminal_callback(self._on_job_terminal)
        self.worker_pool.set_buffer_provider(self._buffer_specs_for_job)
        self.trace_sink = init_trace_sink(
            TraceConfig(
                trace_dir=self.config.get("trace_dir", "traces"),
                trace_token_events=self.config.get("trace_token_events", "sampled"),
            )
        )

        # Session management
        self._sessions: dict[str, SessionState] = {}
        self._server: asyncio.Server | None = None
        self._running = False
        self._dispatcher_tasks: list[asyncio.Task[None]] = []

        # Register built-in processors
        self._register_builtin_processors()

    def _register_builtin_processors(self) -> None:
        """Register built-in processors."""
        from lmpu.processors.builtin import (
            HardwareStubProcessor,
            NeuralSummaryProcessor,
            PureFunctionProcessor,
            PythonVMProcessor,
            SymbolicPlannerProcessor,
            VectorEngineProcessor,
        )

        processors = [
            PureFunctionProcessor(),
            PythonVMProcessor(),
            VectorEngineProcessor(),
            SymbolicPlannerProcessor(),
            NeuralSummaryProcessor(),
            HardwareStubProcessor(),
        ]

        for proc in processors:
            self.register_processor(proc)

    def register_processor(self, processor: Processor) -> None:
        """Register a processor with the runtime."""
        manifest = processor.manifest
        registration = ProcessorRegistration(
            manifest=manifest,
            executor=manifest.executor if hasattr(manifest, "executor") else "process",
            is_builtin=True,
        )
        self.capability_registry.register(registration)
        self.scheduler.set_processor_limit(
            manifest.name,
            manifest.limits.get("max_concurrent", 4),
        )

    def _on_job_terminal(self, job: JobRecord) -> None:
        """Release job-owned resources exactly once."""
        if job.status == JobStatus.COMPLETED:
            self._refresh_output_checksums(job)
        self.job_table.release_grants(job.job_id)
        self.buffer_manager.release_grants(job.job_id)
        self.quota_manager.on_job_complete(job.session_id)

    def _refresh_output_checksums(self, job: JobRecord) -> None:
        """Recompute cached checksums for buffers written directly by a worker."""
        for _name, spec in job.outputs.items():
            if not isinstance(spec, dict) or not spec.get("buffer"):
                continue
            buffer_id = spec["buffer"]
            meta = self.buffer_manager._metadata.get(buffer_id)
            if meta is None:
                continue
            meta.checksum = self.buffer_manager.backend.compute_checksum(
                meta.session_id,
                buffer_id,
            )

    def _buffer_specs_for_job(self, job: JobRecord) -> list[dict[str, Any]]:
        """Resolve declared input/output buffers into fd-passable specs."""
        specs: list[dict[str, Any]] = []

        def add(name: str, spec: Any, writable: bool) -> None:
            if not isinstance(spec, dict):
                return
            buffer_id = spec.get("buffer")
            if not buffer_id:
                return
            meta = self.buffer_manager._metadata.get(buffer_id)
            if meta is None:
                raise BufferError(
                    f"Buffer not found: {buffer_id}",
                    code=ErrorCode.BUFFER_NOT_FOUND.value,
                )
            specs.append(
                {
                    "name": name,
                    "buffer_id": buffer_id,
                    "dtype": meta.dtype,
                    "shape": meta.shape,
                    "bytes": meta.bytes,
                    "writable": writable,
                    "path": self.buffer_manager.get_buffer_path(buffer_id),
                }
            )

        for name, spec in job.inputs.items():
            add(name, spec, writable=False)
        for name, spec in job.outputs.items():
            add(name, spec, writable=True)
        return specs

    async def start(self) -> None:
        """Start the runtime."""
        self._running = True

        # Ensure socket directory exists
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket
        try:
            Path(self.socket_path).unlink()
        except FileNotFoundError:
            pass

        # Start Unix domain socket server
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self.socket_path,
        )

        # Start background tasks
        await self.reaper.start()
        await self.scheduler._flush_batch("")  # Initialize batch timers
        self._start_dispatchers()

        # Set socket permissions
        Path(self.socket_path).chmod(0o600)

    def _start_dispatchers(self) -> None:
        """Start one dispatch loop per registered processor."""
        for registration in self.capability_registry.list_processors():
            self._dispatcher_tasks.append(
                asyncio.create_task(self._dispatch_loop(registration.name))
            )

    async def _dispatch_loop(self, processor: str) -> None:
        """Dequeue and execute jobs for one processor."""
        while self._running:
            try:
                scheduled = await self.scheduler.next_job(processor)
            except asyncio.CancelledError:
                return
            job = scheduled.job
            if job.is_terminal():
                continue
            if job.status == JobStatus.CANCEL_REQUESTED:
                self._finish_cancelled(job, "cancelled_before_start")
                continue

            await self.scheduler.acquire_slot(processor)
            try:
                await self._run_job(job)
            finally:
                self.scheduler.release_slot(processor)

    async def _run_job(self, job: JobRecord) -> None:
        """Assign a job to a worker and wait for its terminal state."""
        if job.status == JobStatus.CREATED:
            self.job_table.transition(job.job_id, JobStatus.PENDING)
        if job.status == JobStatus.CANCEL_REQUESTED:
            self._finish_cancelled(job, "cancelled_before_start")
            return

        self.job_table.transition(job.job_id, JobStatus.RUNNING)
        self.event_log.append(
            job.session_id,
            Event(
                event_id=str(ulid.new()),
                seq=0,
                session=job.session_id,
                job_id=job.job_id,
                type=EventKind.JOB_STARTED,
                ts_monotonic=get_clock().monotonic(),
                data=JobStartedData(
                    job_id=job.job_id,
                    worker=job.processor,
                    start_time=get_clock().monotonic(),
                ),
            ),
        )

        try:
            worker = await self.worker_pool.get_worker(job.processor, job)
            await self.worker_pool.execute_job(worker, job)
        except Exception as exc:
            self.worker_pool._fail_job(job, "worker_crashed", str(exc), retriable=True)
            return

        timeout = (job.timeout_ms / 1000) if job.timeout_ms else None
        finished = await self.job_table.wait_terminal(job.job_id, timeout=timeout)
        if finished is None:
            self.worker_pool._fail_job(job, "timeout", "Job exceeded wall-clock timeout")
            await self.worker_pool.terminate_for_job(job.job_id)

    def _finish_cancelled(self, job: JobRecord, reason: str) -> None:
        """Complete cancellation when no worker owns the job."""
        if job.is_terminal():
            return
        self.job_table.transition(job.job_id, JobStatus.CANCELLED)
        self.event_log.append(
            job.session_id,
            Event(
                event_id=str(ulid.new()),
                seq=0,
                session=job.session_id,
                job_id=job.job_id,
                type=EventKind.JOB_CANCELLED,
                ts_monotonic=get_clock().monotonic(),
                data=JobCancelledData(job_id=job.job_id, reason=reason),
            ),
        )
        self._on_job_terminal(job)

    async def stop(self) -> None:
        """Stop the runtime."""
        self._running = False

        for task in self._dispatcher_tasks:
            task.cancel()
        if self._dispatcher_tasks:
            await asyncio.gather(*self._dispatcher_tasks, return_exceptions=True)
        self._dispatcher_tasks.clear()
        await self.scheduler.close()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        await self.reaper.stop()
        await self.worker_pool.stop()

        # Clean up socket
        try:
            Path(self.socket_path).unlink()
        except FileNotFoundError:
            pass

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming JSON-RPC connection."""
        from ..protocol.codec import RPCStream
        from ..protocol.messages import RPCNotification, RPCRequest

        stream = RPCStream(reader, writer)
        session_id = None

        try:
            async for msg in stream:
                if isinstance(msg, RPCRequest):
                    response = await self._handle_request(msg, session_id)
                    if response:
                        await stream.send_response(response)
                    # Update session_id from hello
                    if msg.method == "lmpu.hello" and response and response.result:
                        session_id = response.result.get("session_id")
                elif isinstance(msg, RPCNotification):
                    await self._handle_notification(msg, session_id)
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_request(
        self,
        request: RPCRequest,
        session_id: str | None,
    ) -> RPCResponse | None:
        """Handle JSON-RPC request."""
        try:
            if request.method == "lmpu.hello":
                result = await self._hello(request.params)
            elif request.method == "lmpu.capabilities":
                result = await self._capabilities(request.params)
            elif request.method == "lmpu.submit":
                result = await self._submit(request.params, session_id)
            elif request.method == "lmpu.poll":
                result = await self._poll(request.params, session_id)
            elif request.method == "lmpu.wait":
                result = await self._wait(request.params, session_id)
            elif request.method == "lmpu.read":
                result = await self._read(request.params, session_id)
            elif request.method == "lmpu.write":
                result = await self._write(request.params, session_id)
            elif request.method == "lmpu.cancel":
                result = await self._cancel(request.params, session_id)
            elif request.method == "lmpu.sleep":
                result = await self._sleep(request.params, session_id)
            elif request.method == "lmpu.wake":
                result = await self._wake(request.params, session_id)
            elif request.method == "lmpu.ack":
                result = await self._ack(request.params, session_id)
            elif request.method == "lmpu.buffer.create":
                result = await self._buffer_create(request.params, session_id)
            elif request.method == "lmpu.buffer.import":
                result = await self._buffer_import(request.params, session_id)
            elif request.method == "lmpu.buffer.stat":
                result = await self._buffer_stat(request.params, session_id)
            elif request.method == "lmpu.buffer.checksum":
                result = await self._buffer_checksum(request.params, session_id)
            elif request.method == "lmpu.buffer.delete":
                result = await self._buffer_delete(request.params, session_id)
            elif request.method == "lmpu.jobs.list":
                result = await self._jobs_list(request.params, session_id)
            elif request.method == "lmpu.configure":
                result = await self._configure(request.params, session_id)
            elif request.method == "lmpu.health":
                result = await self._health(request.params)
            elif request.method == "lmpu.close":
                result = await self._close(request.params, session_id)
            else:
                raise LMPUError(  # noqa: TRY301 - RPC dispatch boundary
                    message=f"Unknown method: {request.method}",
                    code=ErrorCode.UNKNOWN_METHOD.value,
                )

            return RPCResponse(id=request.id, result=_to_jsonable(result))

        except LMPUError as e:
            return RPCResponse(
                id=request.id,
                error=e.to_rpc_error(),
            )
        except Exception as e:
            return RPCResponse(
                id=request.id,
                error=RPCError(
                    code=ErrorCode.INTERNAL.value,
                    message=str(e),
                    retriable=False,
                ),
            )

    async def _handle_notification(
        self,
        notification: RPCNotification,
        session_id: str | None,
    ) -> None:
        """Handle JSON-RPC notification."""
        # Notifications don't expect responses
        pass

    # --- RPC Method Handlers ---

    async def _hello(self, params: dict) -> HelloResult:
        hello_params = HelloParams(**params)
        session_id = f"s_{ulid.new()}"

        # Create session state
        raw_config = dict(hello_params.session_config or {})
        authorized = raw_config.pop("authorized_processors", None)
        quotas = QuotaConfig(**raw_config)
        self.quota_manager.set_session_quotas(session_id, quotas)

        session_state = SessionState(
            session_id=session_id,
            config=quotas,
            authorized_processors=set(authorized) if authorized is not None else None,
        )
        self._sessions[session_id] = session_state

        # Resume from sequence if requested
        resume_status = "none"
        if hello_params.resume_from is not None:
            if self.event_log.has_gap(session_id, hello_params.resume_from):
                resume_status = "gap"
            else:
                resume_status = "ok"

        return HelloResult(
            session_id=session_id,
            protocol=PROTOCOL_VERSION,
            schema_hash="",  # TODO: compute schema hash
            resume_status=resume_status,
            server_time=get_clock().now(),
        )

    async def _capabilities(self, params: dict) -> CapabilitiesResult:
        caps_params = CapabilitiesParams(**params)
        processors = [
            CapabilityDescriptor(
                name=m.name,
                version=m.version,
                protocol=m.protocol,
                ops=m.ops,
                limits=m.limits,
                determinism=m.determinism,
                latency_class=m.latency_class,
                streaming=m.streaming,
                progress=m.progress,
                interactive=m.interactive,
                persistent_state=m.persistent_state,
                batching=m.batching,
                zero_copy=m.zero_copy,
                sandbox=m.sandbox,
            )
            for m in self.capability_registry.list_processors()
        ]

        matches = None
        recommended = None
        if caps_params.op:
            candidates = self.capability_registry.find_by_op(caps_params.op)
            matches = [c.manifest.name for c in candidates]
            if candidates:
                recommended = candidates[0].manifest.name

        return CapabilitiesResult(
            processors=processors,
            matches=matches,
            recommended=recommended,
        )

    async def _submit(self, params: dict, session_id: str) -> SubmitResult:
        if not session_id:
            raise ProtocolError("No session", code=ErrorCode.PROTOCOL_VERSION.value)

        submit_params = SubmitParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            raise ProtocolError("Session not found", code=ErrorCode.INTERNAL.value)

        # Route against registry capabilities; the session authorization set is
        # a separate boundary and does not alter capability advertisement.
        selected, candidates, reason = route_job(
            self.capability_registry,
            submit_params.op,
            explicit_processor=submit_params.processor,
            allowed_processors=session.authorized_processors,
        )
        if selected is None:
            code = (
                ErrorCode.ACCESS_DENIED.value
                if reason.startswith("access_denied")
                else ErrorCode.CAPABILITY_MISMATCH.value
            )
            raise JobError("Processor not authorized or capable", code=code)
        processor = selected.manifest.name

        config = submit_params.config or {}
        estimated_buffer_bytes = self._estimate_output_bytes(submit_params.outputs)

        # Check quotas
        self.quota_manager.check_job_submit(session_id, estimated_buffer_bytes)

        # Create job
        job, is_new = self.job_table.create_job(
            session_id=session_id,
            processor=processor,
            op=submit_params.op,
            inputs=submit_params.inputs,
            outputs=submit_params.outputs,
            program=submit_params.program,
            batch=submit_params.batch,
            config=config,
            job_id=submit_params.job_id,
        )

        if not is_new:
            # Idempotent return
            return SubmitResult(
                job_id=job.job_id,
                status=job.to_model_status(),
                queue_position=job.queue_position,
                seq=self.event_log.get_last_seq(session_id),
                result=job.result,
                error=job.error,
            )

        # Grant buffer access
        for _name, spec in submit_params.inputs.items():
            if isinstance(spec, dict) and spec.get("buffer"):
                buffer_id = spec["buffer"]
                self.buffer_manager.grant_read(job.job_id, buffer_id)

        for _name, spec in submit_params.outputs.items():
            if isinstance(spec, dict) and spec.get("buffer"):
                buffer_id = spec["buffer"]
                self.buffer_manager.grant_write(job.job_id, buffer_id)

        self.job_table.transition(job.job_id, JobStatus.PENDING)
        job.queue_position = self.scheduler.get_queue_depth(processor) + 1

        # Emit submitted event
        self.event_log.append(
            session_id,
            Event(
                event_id=str(ulid.new()),
                seq=0,
                session=session_id,
                job_id=job.job_id,
                type=EventKind.JOB_SUBMITTED,
                ts_monotonic=get_clock().monotonic(),
                data=JobSubmittedData(
                    job_id=job.job_id,
                    processor=job.processor,
                    queue_position=job.queue_position,
                ),
            ),
        )

        # Enqueue for scheduling
        await self.scheduler.enqueue(job)
        self.quota_manager.on_job_submit(session_id, estimated_buffer_bytes)

        if config.get("mode") == "sync":
            timeout = (config.get("timeout_ms", 0) or 0) / 1000 or None
            finished = await self.job_table.wait_terminal(job.job_id, timeout=timeout)
            if finished is None:
                self.worker_pool._fail_job(job, "timeout", "Job exceeded wait timeout")
                await self.worker_pool.terminate_for_job(job.job_id)
                raise JobError("Job timed out", code=ErrorCode.TIMEOUT.value)
            return SubmitResult(
                job_id=job.job_id,
                status=finished.to_model_status(),
                queue_position=job.queue_position,
                seq=self.event_log.get_last_seq(session_id),
                result=finished.result,
                error=finished.error,
            )

        return SubmitResult(
            job_id=job.job_id,
            status=ModelJobStatus.PENDING,
            queue_position=job.queue_position,
            seq=self.event_log.get_last_seq(session_id),
        )

    def _estimate_output_bytes(self, outputs: dict[str, Any]) -> int:
        """Estimate bytes for declared output buffers."""
        total = 0
        for spec in outputs.values():
            if isinstance(spec, dict) and spec.get("buffer"):
                meta = self.buffer_manager._metadata.get(spec["buffer"])
                if meta is not None:
                    total += meta.bytes
        return total

    async def _poll(self, params: dict, session_id: str) -> PollResult:
        poll_params = PollParams(**params)
        job = self.job_table.get(poll_params.job_id)
        if not job:
            raise JobError("Job not found", code=ErrorCode.UNKNOWN_PROCESSOR.value)

        if job.session_id != session_id:
            raise JobError("Access denied", code=ErrorCode.ACCESS_DENIED.value)

        return PollResult(
            job_id=job.job_id,
            status=job.to_model_status(),
            progress=job.progress if job.status == JobStatus.RUNNING else None,
            elapsed_ms=job.elapsed_ms(),
            queue_position=job.queue_position if job.status == JobStatus.PENDING else None,
        )

    async def _wait(self, params: dict, session_id: str) -> WaitResult:
        wait_params = WaitParams(**params)
        job = self.job_table.get(wait_params.job_id)
        if not job:
            raise JobError("Job not found", code=ErrorCode.UNKNOWN_PROCESSOR.value)

        if job.session_id != session_id:
            raise JobError("Access denied", code=ErrorCode.ACCESS_DENIED.value)

        if job.is_terminal():
            return WaitResult(
                status=job.to_model_status(),
                result=job.result,
                error=job.error,
            )

        # Wait for completion without busy-looping
        timeout = wait_params.timeout_ms / 1000 if wait_params.timeout_ms else None
        finished = await self.job_table.wait_terminal(job.job_id, timeout=timeout)
        if finished is None:
            raise JobError("Wait timeout", code=ErrorCode.TIMEOUT.value)

        return WaitResult(
            status=finished.to_model_status(),
            result=finished.result,
            error=finished.error,
        )

    def _authorize_buffer(self, session_id: str, buffer_id: str, write: bool = False):
        """Return buffer metadata and whether the session owns it."""
        meta = self.buffer_manager._metadata.get(buffer_id)
        if meta is None:
            raise BufferError("Buffer not found", code=ErrorCode.BUFFER_NOT_FOUND.value)
        if meta.session_id == session_id:
            return meta, True
        grant = self.grant_manager.check_access(session_id, buffer_id, 0, meta.bytes, write=write)
        if grant is None:
            raise BufferError("Access denied", code=ErrorCode.ACCESS_DENIED.value)
        return meta, False

    async def _read(self, params: dict, session_id: str) -> BufferView:
        read_params = ReadParams(**params)
        # Check session owns buffer or has grant
        meta, _ = self._authorize_buffer(session_id, read_params.buffer)

        inline_data, derived_ref = self.buffer_manager.read_view(
            read_params.buffer,
            slices=read_params.slices,
            max_inline_bytes=read_params.max_inline_bytes,
        )

        if inline_data is not None:
            return BufferView(
                buffer=read_params.buffer,
                dtype=meta.dtype,
                data=inline_data.tolist(),
            )
        else:
            return BufferView(
                buffer=read_params.buffer,
                dtype=meta.dtype,
                derived_buffer=derived_ref.buffer_id if derived_ref else None,
                note="Result too large, use smaller slice" if derived_ref else None,
            )

    async def _write(self, params: dict, session_id: str) -> BufferRef:
        write_params = WriteParams(**params)
        meta, owned = self._authorize_buffer(session_id, write_params.buffer, write=True)

        # Convert data to numpy array
        import numpy as np

        data = np.array(write_params.data)

        ref = self.buffer_manager.write_view(
            write_params.buffer,
            write_params.slices,
            data,
            session_id,  # Use session as job_id for admin writes
            enforce_grant=not owned,
        )
        return ref

    async def _cancel(self, params: dict, session_id: str) -> CancelResult:
        cancel_params = CancelParams(**params)
        job = self.job_table.get(cancel_params.job_id)
        if not job:
            raise JobError("Job not found", code=ErrorCode.UNKNOWN_PROCESSOR.value)

        if job.session_id != session_id:
            raise JobError("Access denied", code=ErrorCode.ACCESS_DENIED.value)

        if job.is_terminal():
            return CancelResult(
                job_id=job.job_id,
                status="already_terminal",
            )

        # Transition to cancel requested. In a single event-loop thread this
        # is the cancellation linearization point.
        self.job_table.transition(job.job_id, JobStatus.CANCEL_REQUESTED)

        worker_id = await self.worker_pool.cancel_job(job.job_id)
        if worker_id is None:
            self._finish_cancelled(job, cancel_params.reason or "cancel_requested")
        else:
            asyncio.create_task(self._finalize_cancel(job.job_id, worker_id))

        return CancelResult(
            job_id=job.job_id,
            status="cancel_requested",
        )

    async def _finalize_cancel(self, job_id: str, worker_id: str) -> None:
        """Give the worker a grace period, then force terminal cancellation."""
        grace = self.config.get("cancel_grace_ms", 500) / 1000
        await asyncio.sleep(grace)
        job = self.job_table.get(job_id)
        if job is None or job.is_terminal():
            return
        if job.status != JobStatus.CANCEL_REQUESTED:
            return
        self._finish_cancelled(job, "cancel_requested")
        await self.worker_pool.terminate_for_job(job_id)

    async def _sleep(self, params: dict, session_id: str) -> SleepResult:
        sleep_params = SleepParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            raise ProtocolError("Session not found", code=ErrorCode.INTERNAL.value)

        self.quota_manager.check_sleep_register(session_id)

        # Check if already satisfied
        condition = sleep_params.until
        already_satisfied = False
        satisfied_jobs = []
        statuses = {}

        if condition.all:
            all_done = True
            for job_id in condition.all:
                job = self.job_table.get(job_id)
                if job and not job.is_terminal():
                    all_done = False
                    break
                if job:
                    satisfied_jobs.append(job_id)
                    statuses[job_id] = job.to_model_status()
            if all_done:
                already_satisfied = True

        if not already_satisfied and condition.any:
            for job_id in condition.any:
                job = self.job_table.get(job_id)
                if job and job.is_terminal():
                    already_satisfied = True
                    satisfied_jobs.append(job_id)
                    statuses[job_id] = job.to_model_status()
                    break

        if already_satisfied:
            return SleepResult(
                sleep_id="",
                status="already_satisfied",
                woken={
                    "reason": "condition_met",
                    "satisfied_jobs": satisfied_jobs,
                    "statuses": statuses,
                    "already_satisfied": True,
                },
            )

        # Register sleep
        sleep_id = f"sleep_{ulid.new()}"
        session.sleep_registration = {
            "sleep_id": sleep_id,
            "condition": condition,
            "level": sleep_params.level,
            "registered_at": get_clock().monotonic(),
        }
        session.wake_event = asyncio.Event()
        session.wake_result = None
        self.quota_manager.on_sleep_register(session_id)

        return SleepResult(
            sleep_id=sleep_id,
            status="registered",
        )

    async def _wake(self, params: dict, session_id: str) -> WakeResult:
        session = self._sessions.get(session_id or "")
        if not session:
            return WakeResult(status="no_active_sleep")

        if session.wake_event:
            session.wake_event.set()
            session.wake_result = {"reason": "forced_wake"}
            return WakeResult(status="woken")

        return WakeResult(status="no_active_sleep")

    async def _ack(self, params: dict, session_id: str) -> AckResult:
        ack_params = AckParams(**params)
        acked = self.event_log.ack(session_id, ack_params.seq)
        return AckResult(acked=acked)

    async def _buffer_create(self, params: dict, session_id: str) -> BufferRef:
        create_params = BufferCreateParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            raise ProtocolError("Session not found", code=ErrorCode.INTERNAL.value)

        self.quota_manager.check_buffer_create(session_id, 0)  # TODO: calculate size

        ref = self.buffer_manager.create(
            session_id=session_id,
            name=create_params.name,
            dtype=create_params.dtype,
            shape=tuple(create_params.shape),
            persist=create_params.persist,
        )
        self.quota_manager.on_buffer_create(session_id, ref.bytes)
        return ref

    async def _buffer_import(self, params: dict, session_id: str) -> BufferRef:
        import_params = BufferImportParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            raise ProtocolError("Session not found", code=ErrorCode.INTERNAL.value)

        self.quota_manager.check_buffer_create(session_id, 0)

        if import_params.data is not None:
            ref = self.buffer_manager.import_bytes(
                session_id=session_id,
                name=import_params.name,
                data=import_params.data,
                dtype=import_params.dtype,
                shape=tuple(import_params.shape),
            )
        elif import_params.path is not None:
            ref = self.buffer_manager.import_file(
                session_id=session_id,
                name=import_params.name,
                path=import_params.path,
                dtype=import_params.dtype,
                shape=tuple(import_params.shape),
            )
        else:
            raise ValidationError(
                "Either data or path required", code=ErrorCode.INVALID_SCHEMA.value
            )

        self.quota_manager.on_buffer_create(session_id, ref.bytes)
        return ref

    async def _buffer_stat(self, params: dict, session_id: str) -> BufferRef:
        stat_params = BufferStatParams(**params)
        self._authorize_buffer(session_id, stat_params.buffer)
        return self.buffer_manager.stat(stat_params.buffer)

    async def _buffer_checksum(self, params: dict, session_id: str) -> str:
        checksum_params = BufferChecksumParams(**params)
        self._authorize_buffer(session_id, checksum_params.buffer)
        from ..memory.checksum import ChecksumAlgorithm

        return self.buffer_manager.checksum(
            checksum_params.buffer,
            ChecksumAlgorithm(checksum_params.algorithm),
        )

    async def _buffer_delete(self, params: dict, session_id: str) -> None:
        delete_params = BufferDeleteParams(**params)
        meta, _ = self._authorize_buffer(session_id, delete_params.buffer)
        self.quota_manager.on_buffer_delete(session_id, meta.bytes)
        self.buffer_manager.delete(delete_params.buffer)

    async def _jobs_list(self, params: dict, session_id: str) -> list[dict]:
        list_params = JobsListParams(**params)
        jobs = self.job_table.list_jobs(
            session_id=list_params.session_id or session_id,
            status=list_params.status,
            limit=list_params.limit,
        )
        return [
            {
                "job_id": j.job_id,
                "processor": j.processor,
                "op": j.op,
                "status": j.to_model_status().value,
                "progress": j.progress,
                "elapsed_ms": j.elapsed_ms(),
            }
            for j in jobs
        ]

    async def _configure(self, params: dict, session_id: str) -> dict:
        config_params = ConfigureParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            raise ProtocolError("Session not found", code=ErrorCode.INTERNAL.value)

        # Update quotas
        new_quotas = QuotaConfig()
        if config_params.max_outstanding_jobs is not None:
            new_quotas.max_outstanding_jobs = config_params.max_outstanding_jobs
        if config_params.max_buffer_bytes is not None:
            new_quotas.max_buffer_bytes = config_params.max_buffer_bytes
        if config_params.max_sleep_seconds is not None:
            new_quotas.max_sleep_seconds = config_params.max_sleep_seconds
        if config_params.sleep_level is not None:
            new_quotas.sleep_level = config_params.sleep_level

        self.quota_manager.set_session_quotas(session_id, new_quotas)

        # Update injection policy
        if config_params.injection:
            session.injection_policy = config_params.injection

        return {"status": "ok"}

    async def _health(self, params: dict) -> HealthResult:
        return HealthResult(
            status="healthy",
            queue_depth=sum(self.scheduler.get_all_queue_depths().values()),
            worker_health=self.worker_pool.get_stats(),
            uptime_seconds=get_clock().monotonic(),
        )

    async def _close(self, params: dict, session_id: str) -> CloseResult:
        close_params = CloseParams(**params)
        session = self._sessions.get(session_id)
        if not session:
            return CloseResult(cancelled_jobs=[], detached_jobs=[])

        cancelled = []
        detached = []

        jobs = self.job_table.get_by_session(session_id)
        for job in jobs:
            if not job.is_terminal():
                if close_params.detach:
                    detached.append(job.job_id)
                else:
                    self.job_table.transition(job.job_id, JobStatus.CANCEL_REQUESTED)
                    await self.worker_pool.cancel_job(job.job_id)
                    self._finish_cancelled(job, "session_closed")
                    await self.worker_pool.terminate_for_job(job.job_id)
                    cancelled.append(job.job_id)

        # Clean up buffers
        if not close_params.detach:
            self.buffer_manager.cleanup_session(session_id)

        # Remove session
        self._sessions.pop(session_id, None)
        self.quota_manager.remove_session(session_id)
        self.event_log.remove_session(session_id)

        return CloseResult(
            cancelled_jobs=cancelled,
            detached_jobs=detached,
        )
