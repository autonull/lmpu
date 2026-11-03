"""LMPU Host - Session controller for orchestrating model-kernel interaction."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

from lmpu.protocol.messages import (
    ModelMessage,
    ModelMessageKind,
    EventKind,
    Event,
    InjectedEvent,
    ModelJobStatus,
    SleepLevel,
    SleepCondition,
)
from lmpu.protocol.errors import ValidationError
from lmpu.kernel.runtime import Runtime
from lmpu.host.engine import HostEngine, HostState, SleepLevel as EngineSleepLevel
from lmpu.host.parser import SentinelScanner, IncrementalScanner, ControlBlockParser
from lmpu.host.grammar import ConstrainedDecoder, create_decoder
from lmpu.kernel.trace import get_trace_sink


class ControllerState(str, Enum):
    """Session controller states."""
    IDLE = "idle"
    GENERATING = "generating"
    PARSING = "parsing"
    DISPATCHING = "dispatching"
    PARKED = "parked"
    WAKING = "waking"
    CLOSED = "closed"


class SleepAuthority(str, Enum):
    """Who controls sleep decisions."""
    MODEL = "model"
    CONTROLLER = "controller"
    MODEL_WITH_POLICY = "model_with_policy"


@dataclass
class InjectionPolicy:
    """Event injection policy configuration."""
    generating: dict[str, str] = field(default_factory=lambda: {
        "JobFailed": "interrupt",
        "NeedsMoreInput": "interrupt",
        "JobCompleted": "next_turn",
        "JobCancelled": "next_turn",
        "Progress": "observe",
    })
    parked: dict[str, str] = field(default_factory=lambda: {
        "JobFailed": "wake",
        "NeedsMoreInput": "wake",
        "JobCompleted": "wake",
        "JobCancelled": "wake",
        "Progress": "observe",
    })


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


class SessionController:
    """Controls a single session's model-kernel interaction loop."""
    
    def __init__(
        self,
        session_id: str,
        runtime: Runtime,
        engine: HostEngine,
        decoder: ConstrainedDecoder,
        config: Optional[SessionConfig] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.session_id = session_id
        self.runtime = runtime
        self.engine = engine
        self.decoder = decoder
        self.config = config or SessionConfig()
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        
        # State
        self.state = ControllerState.IDLE
        self.host_state: Optional[HostState] = None
        self.prompt = ""
        self.generated_buffer = ""
        
        # Components
        self.scanner = SentinelScanner()
        self.incremental_scanner = IncrementalScanner(self.scanner)
        self.parser = ControlBlockParser(
            max_control_tokens=512,
        )
        
        # Wake condition
        self._wake_event: Optional[asyncio.Event] = None
        self._wake_reason: Optional[str] = None
        self._wake_job_statuses: dict[str, ModelJobStatus] = {}
        
        # Injection buffer
        self._pending_injections: list[InjectedEvent] = []
        self._injected_this_turn = 0
        
        # Metrics
        self._turn_count = 0
        self._start_time = time.time()

    async def initialize(self, initial_prompt: str) -> None:
        """Initialize session with prompt."""
        self.prompt = await self._build_prompt(initial_prompt)
        self.host_state = await self.engine.prefill(self.prompt)
        self.state = ControllerState.GENERATING
        
        # Send hello to runtime
        await self.runtime._hello({
            "protocol": "lmpu/1",
            "session_config": self.config.__dict__,
        })

    async def _build_prompt(self, user_prompt: str) -> str:
        """Build full prompt with system prompt and capabilities."""
        parts = []
        
        if self.system_prompt:
            parts.append(self.system_prompt)
        
        # Add capability summary
        caps_result = await self.runtime._capabilities({})
        if caps_result.processors:
            caps_summary = "Available processors:\n"
            for p in caps_result.processors:
                caps_summary += f"  - {p.name} v{p.version}: {', '.join(op['name'] for op in p.ops)}\n"
            parts.append(caps_summary)
        
        parts.append("Protocol rules:")
        parts.append("  - Emit control blocks using <|proc|> ... <|/proc|>")
        parts.append("  - Use 'submit' to launch jobs, 'sleep' to wait for completion")
        parts.append("  - Read results with 'read_result' using bounded slices")
        parts.append("  - Never fabricate computational results")
        
        parts.append(f"\nUser request: {user_prompt}")
        
        return "\n".join(parts)

    async def run_turn(self) -> bool:
        """Run one generation turn. Returns True if should continue."""
        if self.state == ControllerState.CLOSED:
            return False
        
        if self.state == ControllerState.PARKED:
            return await self._handle_parked()
        
        if self.state == ControllerState.GENERATING:
            return await self._handle_generating()
        
        return True

    async def _handle_generating(self) -> bool:
        """Handle generation phase."""
        # Check context budget
        if self.host_state and len(self.host_state.tokens) >= self.config.max_context_tokens:
            # Context full - need to handle
            await self._handle_context_full()
            return True
        
        # Check for pending injections
        if self._pending_injections and self._should_inject_now():
            await self._inject_events()
            return True
        
        # Generate free text until sentinel
        self.state = ControllerState.GENERATING
        
        stop_sequence = "<|proc|>"
        free_text = await self.decoder.decode_free(
            prompt=self.prompt + self.generated_buffer,
            max_tokens=1024,
            stop_sequence=stop_sequence,
        )
        
        self.generated_buffer += free_text
        
        # Scan for control block
        free_text, control_msg = self.incremental_scanner.feed(free_text)
        
        if control_msg:
            # Found control block - parse and dispatch
            self.state = ControllerState.PARSING
            await self._handle_control_message(control_msg)
            return True
        
        # No control block yet, continue generating
        return True

    async def _handle_control_message(self, message: ModelMessage) -> None:
        """Handle parsed control message."""
        self.state = ControllerState.DISPATCHING
        
        try:
            if message.kind == ModelMessageKind.SUBMIT:
                await self._handle_submit(message)
            elif message.kind == ModelMessageKind.SLEEP:
                await self._handle_sleep(message)
            elif message.kind == ModelMessageKind.WAKE:
                await self._handle_wake(message)
            elif message.kind == ModelMessageKind.READ_RESULT:
                await self._handle_read_result(message)
            elif message.kind == ModelMessageKind.WRITE_INPUT:
                await self._handle_write_input(message)
            elif message.kind == ModelMessageKind.CANCEL:
                await self._handle_cancel(message)
            elif message.kind == ModelMessageKind.POLL:
                await self._handle_poll(message)
            elif message.kind == ModelMessageKind.WAIT:
                await self._handle_wait(message)
            elif message.kind == ModelMessageKind.QUERY_CAPABILITIES:
                await self._handle_query_capabilities(message)
        finally:
            self.state = ControllerState.GENERATING

    async def _handle_submit(self, message: ModelMessage) -> None:
        """Handle job submission."""
        from lmpu.protocol.messages import SubmitMessage
        msg: SubmitMessage = message
        
        # Submit to runtime
        result = await self.runtime._submit({
            "job_id": msg.job_id,
            "processor": msg.processor,
            "op": msg.op,
            "inputs": msg.inputs,
            "outputs": msg.outputs,
            "program": msg.program,
            "batch": msg.batch,
            "config": msg.config,
        }, self.session_id)
        
        # Track job
        # Log submission
        trace = get_trace_sink()
        if trace:
            trace.emit_control_parsed(self.session_id, "submit", result.job_id)

    async def _handle_sleep(self, message: ModelMessage) -> None:
        """Handle sleep request."""
        from lmpu.protocol.messages import SleepMessage
        msg: SleepMessage = message
        
        # Check sleep authority
        if self.config.sleep_authority == SleepAuthority.CONTROLLER:
            # Controller decides - ignore model sleep
            return
        elif self.config.sleep_authority == SleepAuthority.MODEL_WITH_POLICY:
            # Validate against policy
            condition = msg.until
            max_sleep = self.config.max_sleep_seconds * 1000
            if condition.deadline_ms and condition.deadline_ms > max_sleep:
                condition.deadline_ms = max_sleep
            # Clamp sleep level
            if msg.level and msg.level != self.config.sleep_level:
                msg.level = self.config.sleep_level
        
        # Register sleep with runtime
        sleep_result = await self.runtime._sleep({
            "until": condition.__dict__,
            "level": msg.level,
        }, self.session_id)
        
        if sleep_result.status == "already_satisfied":
            # Condition already met - inject wake event immediately
            woken = sleep_result.woken
            await self._inject_wake_event(woken)
        else:
            # Park the host
            level = msg.level or self.config.sleep_level
            await self.engine.park(self.host_state, level)
            self.state = ControllerState.PARKED
            self._wake_event = asyncio.Event()
            trace = get_trace_sink()
            if trace:
                trace.emit_park_enter(self.session_id, level, condition.all or [])

    async def _handle_wake(self, message: ModelMessage) -> None:
        """Handle wake request."""
        await self.runtime._wake({}, self.session_id)
        if self._wake_event:
            self._wake_event.set()

    async def _handle_read_result(self, message: ModelMessage) -> None:
        """Handle read result request."""
        from lmpu.protocol.messages import ReadResultMessage
        msg: ReadResultMessage = message
        
        result = await self.runtime._read({
            "buffer": msg.buffer,
            "slices": msg.slices,
            "max_inline_bytes": msg.max_inline_bytes,
        }, self.session_id)
        
        # Inject result into context
        await self._inject_read_result(result)

    async def _handle_write_input(self, message: ModelMessage) -> None:
        """Handle write input request."""
        from lmpu.protocol.messages import WriteInputMessage
        msg: WriteInputMessage = message
        
        await self.runtime._write({
            "buffer": msg.field,  # Using field as buffer for interactive input
            "slices": [],
            "data": msg.value,
        }, self.session_id)

    async def _handle_cancel(self, message: ModelMessage) -> None:
        """Handle cancel request."""
        from lmpu.protocol.messages import CancelMessage
        msg: CancelMessage = message
        
        await self.runtime._cancel({
            "job_id": msg.job_id,
            "reason": msg.reason,
        }, self.session_id)

    async def _handle_poll(self, message: ModelMessage) -> None:
        """Handle poll request."""
        from lmpu.protocol.messages import PollMessage
        msg: PollMessage = message
        
        result = await self.runtime._poll({"job_id": msg.job_id}, self.session_id)
        # Inject poll result
        await self._inject_poll_result(result)

    async def _handle_wait(self, message: ModelMessage) -> None:
        """Handle wait request."""
        from lmpu.protocol.messages import WaitMessage
        msg: WaitMessage = message
        
        result = await self.runtime._wait({
            "job_id": msg.job_id,
            "timeout_ms": msg.timeout_ms,
        }, self.session_id)
        await self._inject_wait_result(result)

    async def _handle_query_capabilities(self, message: ModelMessage) -> None:
        """Handle capability query."""
        from lmpu.protocol.messages import QueryCapabilitiesMessage
        msg: QueryCapabilitiesMessage = message
        
        result = await self.runtime._capabilities({
            "op": msg.op,
            "processor": msg.processor,
        })
        await self._inject_capabilities(result)

    async def _handle_parked(self) -> bool:
        """Handle parked state - wait for wake."""
        self.state = ControllerState.PARKED
        
        # Wait for wake event
        if self._wake_event:
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=300)
            except asyncio.TimeoutError:
                # Sleep deadline exceeded
                await self._handle_wake_timeout()
        
        # Resume host
        self.state = ControllerState.WAKING
        await self.engine.resume(self.host_state)
        
        # Inject wake event
        if self._wake_reason:
            await self._inject_wake_event({
                "reason": self._wake_reason,
                "satisfied_jobs": [],
                "statuses": self._wake_job_statuses,
                "already_satisfied": False,
            })
        
        self.state = ControllerState.GENERATING
        self._wake_event = None
        self._wake_reason = None
        self._wake_job_statuses = {}
        
        trace = get_trace_sink()
        if trace:
            trace.emit_wake(self.session_id, self._wake_reason or "unknown")
        
        return True

    async def _handle_wake_timeout(self) -> None:
        """Handle sleep timeout."""
        self._wake_reason = "timeout"
        # Force wake
        await self.runtime._wake({}, self.session_id)

    def _should_inject_now(self) -> bool:
        """Check if we should inject pending events now."""
        if self._injected_this_turn >= self.config.max_injected_tokens_per_turn:
            return False
        return True

    async def _inject_events(self) -> None:
        """Inject pending events into model context."""
        while self._pending_injections and self._should_inject_now():
            event = self._pending_injections.pop(0)
            
            # Render event as <|proc_event|> block
            event_text = self._render_event(event)
            
            # Add to prompt buffer
            self.generated_buffer += event_text
            self._injected_this_turn += len(event_text) // 4  # rough token estimate
            
            # Update stats
            self.host_state.injected_tokens += self._injected_this_turn
            self.engine.stats_.total_injected_tokens += self._injected_this_turn

    def _render_event(self, event: InjectedEvent) -> str:
        """Render event as proc_event block."""
        import json
        data = event.model_dump(exclude_none=True)
        json_str = json.dumps(data, separators=(",", ":"))
        return f"\n<|proc_event|>\n{json_str}\n<|/proc_event|>\n"

    async def _inject_wake_event(self, woken_data: dict) -> None:
        """Inject wake event."""
        event = InjectedEvent(
            version=1,
            kind="event",
            event=EventKind.WOKEN,
            job_id=woken_data.get("satisfied_jobs", [None])[0],
            status=woken_data.get("reason"),
            summary={
                "satisfied_jobs": woken_data.get("satisfied_jobs", []),
                "statuses": woken_data.get("statuses", {}),
            },
        )
        self._pending_injections.append(event)

    async def _inject_read_result(self, result) -> None:
        """Inject read result."""
        # Render as event
        pass

    async def _inject_poll_result(self, result) -> None:
        """Inject poll result."""
        pass

    async def _inject_wait_result(self, result) -> None:
        """Inject wait result."""
        pass

    async def _inject_capabilities(self, result) -> None:
        """Inject capabilities."""
        pass

    async def _handle_context_full(self) -> None:
        """Handle context budget exhaustion."""
        # Could trigger summarization or truncation
        pass

    async def on_job_completed(self, job_id: str, status: ModelJobStatus) -> None:
        """Callback when job completes (called from runtime)."""
        if self.state == ControllerState.PARKED:
            # Check if this satisfies sleep condition
            # For simplicity, wake on any completion
            self._wake_reason = "job_done"
            self._wake_job_statuses[job_id] = status
            if self._wake_event:
                self._wake_event.set()
        else:
            # Queue for injection
            event = InjectedEvent(
                version=1,
                kind="event",
                event=EventKind.JOB_COMPLETED,
                job_id=job_id,
                status=status.value,
            )
            self._pending_injections.append(event)

    async def on_job_failed(self, job_id: str, error: str) -> None:
        """Callback when job fails."""
        if self.state == ControllerState.PARKED and self.config.wake_on_failure:
            self._wake_reason = "job_failed"
            if self._wake_event:
                self._wake_event.set()
        else:
            event = InjectedEvent(
                version=1,
                kind="event",
                event=EventKind.JOB_FAILED,
                job_id=job_id,
                status="failed",
                summary={"error": error},
            )
            self._pending_injections.append(event)

    async def close(self) -> None:
        """Close session."""
        self.state = ControllerState.CLOSED
        await self.runtime._close({"detach": False}, self.session_id)
        await self.engine.close()