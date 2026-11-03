"""LMPU Host - Session controller for orchestrating model-kernel interaction."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from lmpu.host.engine import HostEngine, HostState
from lmpu.host.grammar import ConstrainedDecoder
from lmpu.host.parser import ControlBlockParser, IncrementalScanner, SentinelScanner
from lmpu.kernel.quotas import QuotaConfig
from lmpu.kernel.runtime import Runtime
from lmpu.kernel.trace import get_trace_sink
from lmpu.protocol.errors import ValidationError
from lmpu.protocol.messages import (
    EventKind,
    InjectedEvent,
    ModelJobStatus,
    ModelMessage,
    ModelMessageKind,
    SleepLevel,
)
from lmpu.protocol.sentinels import PROC_CLOSE, PROC_EVENT_CLOSE, PROC_EVENT_OPEN, PROC_OPEN


class ControllerState(StrEnum):
    """Session controller states."""

    IDLE = "idle"
    GENERATING = "generating"
    PARSING = "parsing"
    DISPATCHING = "dispatching"
    PARKED = "parked"
    WAKING = "waking"
    CLOSED = "closed"


class SleepAuthority(StrEnum):
    """Who controls sleep decisions."""

    MODEL = "model"
    CONTROLLER = "controller"
    MODEL_WITH_POLICY = "model_with_policy"


VALID_INJECTION_ACTIONS = {"interrupt", "next_turn", "observe", "wake", "inject"}


@dataclass
class InjectionPolicy:
    """Event injection policy configuration."""

    generating: dict[str, str] = field(
        default_factory=lambda: {
            "JobFailed": "interrupt",
            "NeedsMoreInput": "interrupt",
            "JobCompleted": "next_turn",
            "JobCancelled": "next_turn",
            "Progress": "observe",
            "ControlError": "inject",
            "ResultTooLarge": "inject",
        }
    )
    parked: dict[str, str] = field(
        default_factory=lambda: {
            "JobFailed": "wake",
            "NeedsMoreInput": "wake",
            "JobCompleted": "wake",
            "JobCancelled": "wake",
            "Progress": "observe",
            "Woken": "inject",
        }
    )

    def validate(self) -> None:
        valid_events = {kind.value for kind in EventKind}
        for phase, policy in (("generating", self.generating), ("parked", self.parked)):
            for event, action in policy.items():
                if event not in valid_events:
                    raise ValueError(f"Unknown injected event {event!r} in {phase} policy")
                if action not in VALID_INJECTION_ACTIONS:
                    raise ValueError(f"Invalid injection action {action!r} for {event}")
                if phase == "parked" and action in {"interrupt", "next_turn"}:
                    raise ValueError(f"Parked policy for {event} cannot use {action}")


@dataclass
class SessionConfig:
    """Session configuration."""

    max_outstanding_jobs: int = 32
    max_buffer_bytes: int = 2_147_483_648
    max_injected_tokens_per_turn: int = 2048
    max_injected_bytes_per_event: int = 65536
    max_sleep_seconds: int = 300
    sleep_level: str = SleepLevel.S1
    sleep_authority: str = SleepAuthority.MODEL_WITH_POLICY
    wake_on_failure: bool = True
    injection_policy: InjectionPolicy = field(default_factory=InjectionPolicy)
    max_context_tokens: int = 32768

    def __post_init__(self) -> None:
        self.injection_policy.validate()

    def quota_config(self) -> dict[str, Any]:
        """Return only the fields accepted by ``QuotaConfig``."""
        fields = set(QuotaConfig.__dataclass_fields__)
        return {key: value for key, value in self.__dict__.items() if key in fields}


class SessionController:
    """Controls a single session's model-kernel interaction loop."""

    def __init__(
        self,
        session_id: str,
        runtime: Runtime,
        engine: HostEngine,
        decoder: ConstrainedDecoder,
        config: SessionConfig | None = None,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
    ):
        self.session_id = session_id
        self.runtime = runtime
        self.engine = engine
        self.decoder = decoder
        self.config = config or SessionConfig()
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

        self.state = ControllerState.IDLE
        self.host_state: HostState | None = None
        self.prompt = ""
        self.generated_buffer = ""

        self.scanner = SentinelScanner()
        self.incremental_scanner = IncrementalScanner(self.scanner)
        self.parser = ControlBlockParser(max_control_tokens=512)

        self._wake_event: asyncio.Event | None = None
        self._wake_reason: str | None = None
        self._wake_job_statuses: dict[str, str] = {}
        self._sleep_deadline_monotonic: float | None = None

        self._pending_injections: list[InjectedEvent] = []
        self._injected_this_turn = 0

        self._turn_count = 0
        self._start_time = time.time()

    async def initialize(self, initial_prompt: str) -> None:
        """Initialize the session and adopt the runtime-assigned session ID."""
        hello = await self.runtime._hello(
            {
                "protocol": "lmpu/1",
                "session_config": self.config.quota_config(),
            }
        )
        runtime_session_id = (
            hello.get("session_id")
            if isinstance(hello, dict)
            else getattr(hello, "session_id", None)
        )
        if runtime_session_id:
            self.session_id = runtime_session_id

        if hasattr(self.engine, "session_id"):
            self.engine.session_id = self.session_id

        self.prompt = await self._build_prompt(initial_prompt)
        self.host_state = await self.engine.prefill(self.prompt)
        self.state = ControllerState.GENERATING

    async def _build_prompt(self, user_prompt: str) -> str:
        """Build the full prompt with system prompt and live capabilities."""
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)

        caps_result = await self.runtime._capabilities({})
        if caps_result.processors:
            parts.append("Available processors:")
            for processor in caps_result.processors:
                op_names = [
                    op.get("name", "") if isinstance(op, dict) else str(op) for op in processor.ops
                ]
                parts.append(f"  - {processor.name} v{processor.version}: {', '.join(op_names)}")

        parts.extend(
            [
                "Protocol rules:",
                f"  - Emit control blocks using {PROC_OPEN} ... {PROC_CLOSE}",
                "  - Use 'submit' to launch jobs, 'sleep' to wait for completion",
                "  - Read results with 'read_result' using bounded slices",
                "  - Never fabricate computational results",
                f"\nUser request: {user_prompt}",
            ]
        )
        return "\n".join(parts)

    async def run_turn(self) -> bool:
        """Run one generation turn. Returns True if the loop should continue."""
        if self.state == ControllerState.CLOSED:
            return False
        if self.state == ControllerState.PARKED:
            return await self._handle_parked()
        if self.state in {ControllerState.IDLE, ControllerState.GENERATING}:
            return await self._handle_generating()
        return True

    async def run_until_idle(self, max_turns: int = 100) -> int:
        """Run turns until the controller stops or the turn limit is reached."""
        turns = 0
        while turns < max_turns and await self.run_turn():
            turns += 1
        return turns

    async def _handle_generating(self) -> bool:
        """Run one generate/parse/dispatch cycle."""
        if self._context_exhausted():
            self._handle_context_full()
            return False

        # Events with next_turn/inject policy are rendered before decoding.
        if self._pending_injections:
            self._inject_events()
            return True

        prompt = self.prompt + self.generated_buffer
        try:
            free_text = await self.decoder.decode_free(
                prompt=prompt,
                max_tokens=1024,
                stop_sequence=PROC_OPEN,
            )
        except Exception as exc:
            self._record_control_error("engine_error", str(exc))
            self.state = ControllerState.CLOSED
            return False

        if not free_text:
            if not self.incremental_scanner.buffer:
                self.state = ControllerState.IDLE
                return False
            free_text = ""

        self.state = ControllerState.GENERATING
        self.generated_buffer += free_text

        try:
            _, control_msg = self.incremental_scanner.feed(free_text)
        except ValidationError as exc:
            # A malformed block must not kill the session. Reset the scanner and
            # report the failure as an injectable ControlError event.
            self.incremental_scanner = IncrementalScanner(self.scanner)
            self._record_control_error("parser_failure", str(exc))
            return True

        if control_msg is not None:
            self.state = ControllerState.PARSING
            await self._handle_control_message(control_msg)
            return self.state != ControllerState.CLOSED

        return True

    async def _handle_control_message(self, message: ModelMessage) -> None:
        """Dispatch one parsed control message."""
        self.state = ControllerState.DISPATCHING
        handlers = {
            ModelMessageKind.SUBMIT: self._handle_submit,
            ModelMessageKind.SLEEP: self._handle_sleep,
            ModelMessageKind.WAKE: self._handle_wake,
            ModelMessageKind.READ_RESULT: self._handle_read_result,
            ModelMessageKind.WRITE_INPUT: self._handle_write_input,
            ModelMessageKind.CANCEL: self._handle_cancel,
            ModelMessageKind.POLL: self._handle_poll,
            ModelMessageKind.WAIT: self._handle_wait,
            ModelMessageKind.QUERY_CAPABILITIES: self._handle_query_capabilities,
        }
        handler = handlers.get(message.kind)
        if handler is None:
            self._record_control_error(
                "dispatch_failure",
                f"Unsupported model message kind: {message.kind}",
            )
            return
        try:
            await handler(message)
        except Exception as exc:
            self._record_control_error("dispatch_failure", str(exc))
        finally:
            if self.state in {ControllerState.DISPATCHING, ControllerState.PARSING}:
                self.state = ControllerState.GENERATING

    async def _handle_submit(self, message: ModelMessage) -> None:
        """Submit a job to the runtime."""
        from lmpu.protocol.messages import SubmitMessage

        assert isinstance(message, SubmitMessage)
        msg: SubmitMessage = message
        result = await self.runtime._submit(
            {
                "job_id": msg.job_id,
                "processor": msg.processor,
                "op": msg.op,
                "inputs": msg.inputs,
                "outputs": msg.outputs,
                "program": msg.program,
                "batch": msg.batch,
                "config": msg.config,
            },
            self.session_id,
        )
        self._trace_control("submit", result.job_id)

    async def _handle_sleep(self, message: ModelMessage) -> None:
        """Register a sleep condition and park the host."""
        from lmpu.protocol.messages import SleepMessage

        assert isinstance(message, SleepMessage)
        msg: SleepMessage = message
        condition = msg.until
        level = msg.level or self.config.sleep_level

        if self.config.sleep_authority == SleepAuthority.CONTROLLER:
            # Controller-owned sleep still requires an explicit registered
            # condition; a model sleep block is advisory only.
            return

        deadline_ms = condition.deadline_ms
        if self.config.sleep_authority == SleepAuthority.MODEL_WITH_POLICY:
            max_sleep_ms = self.config.max_sleep_seconds * 1000
            deadline_ms = min(deadline_ms, max_sleep_ms) if deadline_ms else max_sleep_ms
            condition.deadline_ms = deadline_ms
            if level != self.config.sleep_level:
                level = self.config.sleep_level

        sleep_result = await self.runtime._sleep(
            {
                "until": condition.model_dump(mode="json"),
                "level": level,
            },
            self.session_id,
        )

        if sleep_result.status == "already_satisfied":
            await self._inject_wake_event(sleep_result.woken or {})
            return

        now = time.monotonic()
        self._sleep_deadline_monotonic = now + (deadline_ms / 1000) if deadline_ms else None
        assert self.host_state is not None
        await self.engine.park(self.host_state, level)
        self._mark_sleep_start()
        self.state = ControllerState.PARKED
        self._wake_event = asyncio.Event()
        trace = get_trace_sink()
        if trace:
            trace.emit_park_enter(self.session_id, level, condition.all or [])

    async def _handle_wake(self, message: ModelMessage) -> None:
        """Handle an explicit wake control block."""
        await self.runtime._wake({}, self.session_id)
        await self.force_wake("model_wake")

    async def _handle_read_result(self, message: ModelMessage) -> None:
        """Read a bounded result view and queue it for injection."""
        from lmpu.protocol.messages import ReadResultMessage

        assert isinstance(message, ReadResultMessage)
        msg: ReadResultMessage = message
        result = await self.runtime._read(
            {
                "buffer": msg.buffer,
                "slices": msg.slices,
                "max_inline_bytes": msg.max_inline_bytes,
            },
            self.session_id,
        )
        self._queue_summary_event(EventKind.BUFFER_UPDATED, msg.job_id, result)

    async def _handle_write_input(self, message: ModelMessage) -> None:
        """Write interactive input through the runtime."""
        from lmpu.protocol.messages import WriteInputMessage

        assert isinstance(message, WriteInputMessage)
        msg: WriteInputMessage = message
        await self.runtime._write(
            {
                "buffer": msg.field,
                "slices": [],
                "data": msg.value,
            },
            self.session_id,
        )

    async def _handle_cancel(self, message: ModelMessage) -> None:
        """Cancel a job."""
        from lmpu.protocol.messages import CancelMessage

        assert isinstance(message, CancelMessage)
        msg: CancelMessage = message
        await self.runtime._cancel(
            {"job_id": msg.job_id, "reason": msg.reason},
            self.session_id,
        )

    async def _handle_poll(self, message: ModelMessage) -> None:
        """Poll a job and queue the status for injection."""
        from lmpu.protocol.messages import PollMessage

        assert isinstance(message, PollMessage)
        msg: PollMessage = message
        result = await self.runtime._poll({"job_id": msg.job_id}, self.session_id)
        self._queue_summary_event(EventKind.PROGRESS, msg.job_id, result)

    async def _handle_wait(self, message: ModelMessage) -> None:
        """Wait for a job and queue the result for injection."""
        from lmpu.protocol.messages import WaitMessage

        assert isinstance(message, WaitMessage)
        msg: WaitMessage = message
        result = await self.runtime._wait(
            {"job_id": msg.job_id, "timeout_ms": msg.timeout_ms},
            self.session_id,
        )
        self._queue_summary_event(EventKind.JOB_COMPLETED, msg.job_id, result)

    async def _handle_query_capabilities(self, message: ModelMessage) -> None:
        """Queue a capability summary for injection."""
        from lmpu.protocol.messages import QueryCapabilitiesMessage

        assert isinstance(message, QueryCapabilitiesMessage)
        msg: QueryCapabilitiesMessage = message
        result = await self.runtime._capabilities({"op": msg.op, "processor": msg.processor})
        self._queue_summary_event(EventKind.PROGRESS, None, result)

    async def _handle_parked(self) -> bool:
        """Wait for a wake event, resume the host, and inject Woken."""
        self.state = ControllerState.PARKED
        if self._wake_event is None:
            self._wake_event = asyncio.Event()

        timeout = float(self.config.max_sleep_seconds)
        if self._sleep_deadline_monotonic is not None:
            timeout = max(0.0, self._sleep_deadline_monotonic - time.monotonic())

        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
        except TimeoutError:
            self._wake_reason = "timeout"
            await self.runtime._wake({}, self.session_id)

        reason = self._wake_reason or "woken"
        statuses = dict(self._wake_job_statuses)

        self.state = ControllerState.WAKING
        assert self.host_state is not None
        await self.engine.resume(self.host_state)
        self._mark_sleep_end()
        await self._inject_wake_event(
            {
                "reason": reason,
                "satisfied_jobs": [],
                "statuses": statuses,
                "already_satisfied": False,
            }
        )

        self.state = ControllerState.GENERATING
        self._wake_event = None
        self._wake_reason = None
        self._wake_job_statuses = {}
        self._sleep_deadline_monotonic = None

        trace = get_trace_sink()
        if trace:
            trace.emit_wake(self.session_id, reason)
        return True

    async def force_wake(self, reason: str = "forced_wake") -> None:
        """Wake a parked session from outside the model loop."""
        self._wake_reason = reason
        if self.state == ControllerState.PARKED and self._wake_event is not None:
            self._wake_event.set()
        if self.session_id:
            await self.runtime._wake({}, self.session_id)

    def _context_exhausted(self) -> bool:
        if self.host_state is None:
            return False
        buffered_tokens = len(self.generated_buffer) // 4
        return len(self.host_state.tokens) + buffered_tokens >= self.config.max_context_tokens

    def _handle_context_full(self) -> None:
        """Report context exhaustion instead of silently truncating."""
        event = InjectedEvent(
            version=1,
            kind="event",
            event=EventKind.RESULT_TOO_LARGE,
            summary={
                "buffer": "",
                "reason": "context_budget_exhausted",
                "max_context_tokens": self.config.max_context_tokens,
                "suggested_slice": [],
            },
        )
        self.generated_buffer += self._render_event(event)
        self._trace_inject(event)
        self.state = ControllerState.CLOSED

    def _select_injection_action(self, event: InjectedEvent) -> str:
        policy = (
            self.config.injection_policy.parked
            if self.state == ControllerState.PARKED
            else self.config.injection_policy.generating
        )
        return policy.get(event.event.value, "observe")

    def enqueue_event(self, event: InjectedEvent) -> str:
        """Apply the configured injection policy to one event."""
        action = self._select_injection_action(event)
        if action == "observe":
            self._trace_inject(event)
            return action

        if action == "wake" and self.state == ControllerState.PARKED:
            self._wake_reason = event.event.value
            if event.job_id and event.status:
                self._wake_job_statuses[event.job_id] = event.status
            if self._wake_event is not None:
                self._wake_event.set()
            self._trace_inject(event)
            return action

        self._pending_injections.append(event)
        self._trace_inject(event)
        return action

    def _inject_events(self) -> None:
        """Render pending events within the per-event and per-turn budgets."""
        self._injected_this_turn = 0

        while self._pending_injections:
            event = self._pending_injections[0]
            rendered = self._render_event(event)
            encoded_size = len(rendered.encode("utf-8"))

            if encoded_size > self.config.max_injected_bytes_per_event:
                event = self._too_large_event(event, encoded_size)
                rendered = self._render_event(event)
                self._trace_inject(event)

            token_estimate = max(1, len(rendered) // 4)
            if self._injected_this_turn + token_estimate > self.config.max_injected_tokens_per_turn:
                break

            self._pending_injections.pop(0)
            self.generated_buffer += rendered
            self._injected_this_turn += token_estimate
            if self.host_state is not None:
                self.host_state.injected_tokens += token_estimate

    def _too_large_event(
        self,
        event: InjectedEvent,
        encoded_size: int,
    ) -> InjectedEvent:
        return InjectedEvent(
            version=1,
            kind="event",
            event=EventKind.RESULT_TOO_LARGE,
            job_id=event.job_id,
            status="truncated",
            summary={
                "buffer": event.job_id or "",
                "reason": "event_exceeds_budget",
                "original_event": event.event.value,
                "bytes": encoded_size,
                "max_bytes": self.config.max_injected_bytes_per_event,
                "suggested_slice": [[0, self.config.max_injected_bytes_per_event, 1]],
            },
        )

    def _render_event(self, event: InjectedEvent) -> str:
        data = event.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(data, separators=(",", ":"))
        return f"\n{PROC_EVENT_OPEN}\n{encoded}\n{PROC_EVENT_CLOSE}\n"

    def _queue_summary_event(
        self,
        kind: EventKind,
        job_id: str | None,
        payload: Any,
    ) -> None:
        summary = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
        self.enqueue_event(
            InjectedEvent(
                version=1,
                kind="event",
                event=kind,
                job_id=job_id,
                summary=summary if isinstance(summary, dict) else {"value": summary},
            )
        )

    async def _inject_wake_event(self, woken_data: dict[str, Any]) -> None:
        """Queue a Woken event after resume."""
        satisfied_jobs = woken_data.get("satisfied_jobs") or []
        statuses = woken_data.get("statuses") or {}
        event = InjectedEvent(
            version=1,
            kind="event",
            event=EventKind.WOKEN,
            job_id=satisfied_jobs[0] if satisfied_jobs else None,
            status=woken_data.get("reason"),
            summary={
                "reason": woken_data.get("reason"),
                "satisfied_jobs": satisfied_jobs,
                "statuses": statuses,
                "already_satisfied": woken_data.get("already_satisfied", False),
            },
        )
        self._pending_injections.append(event)
        self._trace_inject(event)

    def _record_control_error(self, reason: str, message: str) -> None:
        event = InjectedEvent(
            version=1,
            kind="event",
            event=EventKind.CONTROL_ERROR,
            status="rejected",
            summary={"reason": reason, "error": message},
        )
        self.generated_buffer += self._render_event(event)
        self._trace_inject(event)

    def _trace_control(self, kind: str, job_id: str | None) -> None:
        trace = get_trace_sink()
        if trace:
            trace.emit_control_parsed(self.session_id, kind, job_id or "")

    def _trace_inject(self, event: InjectedEvent) -> None:
        trace = get_trace_sink()
        if trace:
            trace.emit_inject(
                self.session_id,
                event.event.value,
                event.job_id or "",
            )

    def _host_metrics(self):
        metrics = getattr(self.engine, "metrics", None)
        if metrics is None or not hasattr(self.engine, "session_id"):
            return None
        return metrics.get_or_create(self.session_id)

    def _mark_sleep_start(self) -> None:
        metrics = self._host_metrics()
        if metrics is not None:
            metrics.start_sleep()

    def _mark_sleep_end(self) -> None:
        metrics = self._host_metrics()
        if metrics is not None:
            metrics.end_sleep()

    async def on_job_completed(
        self,
        job_id: str,
        status: ModelJobStatus,
    ) -> None:
        """Runtime callback for terminal job success."""
        self.enqueue_event(
            InjectedEvent(
                version=1,
                kind="event",
                event=EventKind.JOB_COMPLETED,
                job_id=job_id,
                status=status.value,
            )
        )

    async def on_job_failed(self, job_id: str, error: str) -> None:
        """Runtime callback for terminal job failure."""
        self.enqueue_event(
            InjectedEvent(
                version=1,
                kind="event",
                event=EventKind.JOB_FAILED,
                job_id=job_id,
                status="failed",
                summary={"error": error},
            )
        )

    async def close(self) -> None:
        """Close the session and release runtime resources."""
        self.state = ControllerState.CLOSED
        self._wake_event = None
        metrics = getattr(self.engine, "metrics", None)
        if metrics is not None and hasattr(self.engine, "session_id"):
            metrics.flush(self.engine.session_id)
        await self.runtime._close({"detach": False}, self.session_id)
        await self.engine.close()
