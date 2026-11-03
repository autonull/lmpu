"""LMPU Kernel - Trace sink for JSONL tracing."""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from ..protocol.messages import Event, EventKind

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "env",
    "environment",
    "password",
    "secret",
    "token",
}
_REDACTED = "[REDACTED]"


@dataclass
class TraceConfig:
    enabled: bool = True
    trace_dir: str = "traces"
    trace_token_events: str = "sampled"  # sampled | full | off
    token_sample_rate: int = 100
    max_file_size_mb: int = 100
    max_files: int = 10
    redact_payloads: bool = True


def redact(value: Any) -> Any:
    """Recursively remove sensitive values before a record is persisted."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                result[key] = _REDACTED
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and len(value) > 64_000:
        return value[:64_000] + "...[truncated]"
    return value


@dataclass
class TraceReplay:
    """Parsed trace for timeline replay and integrity checks."""

    records: list[dict[str, Any]]
    session_sequences: dict[str, list[int]]
    gaps: dict[str, list[list[int]]]

    def is_contiguous(self) -> bool:
        return not any(self.gaps.values())


def read_trace(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a canonical JSONL trace file."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Trace line {line_number} is not an object")
            records.append(record)
    return records


def replay_trace(path: str | os.PathLike[str]) -> TraceReplay:
    """Rebuild per-session event order and report sequence gaps.

    Trace records without a numeric ``seq`` are host-side observations and do
    not participate in event replay. A gap means the trace is incomplete and
    must not be presented as a contiguous timeline.
    """
    records = read_trace(path)
    sequences: dict[str, list[int]] = {}
    for record in records:
        session = record.get("session") or record.get("session_id")
        seq = record.get("seq")
        if not isinstance(session, str) or not isinstance(seq, int):
            continue
        sequences.setdefault(session, []).append(seq)

    gaps: dict[str, list[list[int]]] = {}
    for session, seqs in sequences.items():
        ordered = sorted(set(seqs))
        session_gaps = [
            [previous + 1, current]
            for previous, current in zip(ordered, ordered[1:], strict=False)
            if current != previous + 1
        ]
        if session_gaps:
            gaps[session] = session_gaps
    return TraceReplay(records=records, session_sequences=sequences, gaps=gaps)


class TraceSink:
    """JSONL trace sink with rotation."""

    def __init__(self, config: TraceConfig | None = None):
        self.config = config or TraceConfig()
        self._file: TextIO | None = None
        self._current_trace_id: str | None = None
        self._file_size = 0
        self._file_count = 0
        self._token_counter = 0
        self._lock = asyncio.Lock()

        if self.config.enabled:
            self._ensure_dir()
            self._open_new_file()

    def _ensure_dir(self) -> None:
        Path(self.config.trace_dir).mkdir(parents=True, exist_ok=True)

    def _open_new_file(self) -> None:
        if self._file:
            self._file.close()
        self._file_count += 1
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"trace-{timestamp}-{self._file_count:03d}.jsonl"
        path = Path(self.config.trace_dir) / filename
        self._file = path.open("a", buffering=1)
        self._file_size = 0

    def _check_rotation(self) -> None:
        if self._file_size > self.config.max_file_size_mb * 1024 * 1024:
            self._open_new_file()

    def start_trace(self, trace_id: str, session_id: str) -> None:
        self._current_trace_id = trace_id
        self.emit({
            "type": "session_start",
            "trace_id": trace_id,
            "session_id": session_id,
            "ts": time.time(),
            "mono": time.monotonic(),
        })

    def end_trace(self, trace_id: str, session_id: str) -> None:
        self.emit({
            "type": "session_end",
            "trace_id": trace_id,
            "session_id": session_id,
            "ts": time.time(),
            "mono": time.monotonic(),
        })
        if self._current_trace_id == trace_id:
            self._current_trace_id = None

    def emit(self, record: dict[str, Any]) -> None:
        """Emit a trace record (thread-safe for sync calls)."""
        if not self.config.enabled or not self._file:
            return

        # Add trace_id if not present
        if "trace_id" not in record and self._current_trace_id:
            record["trace_id"] = self._current_trace_id

        if self.config.redact_payloads:
            record = redact(record)

        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._file.write(line)
        self._file_size += len(line)
        self._check_rotation()

    async def emit_async(self, record: dict[str, Any]) -> None:
        """Emit a trace record (async)."""
        async with self._lock:
            self.emit(record)

    def emit_event(self, event: Event) -> None:
        """Emit an event as trace record."""
        record = {
            "ts": event.ts_wall,
            "mono": event.ts_monotonic,
            "trace": self._current_trace_id,
            "session": event.session,
            "type": f"event_{event.type.value.lower()}",
            "event_id": event.event_id,
            "seq": event.seq,
        }
        if event.job_id:
            record["job_id"] = event.job_id

        # Add event data (excluding large payloads)
        data_dict = event.data.model_dump() if hasattr(event.data, 'model_dump') else {}
        if event.type in (EventKind.PROGRESS, EventKind.HEARTBEAT):
            # Sample high-frequency events
            self._token_counter += 1
            if (
                self.config.trace_token_events == "sampled"
                and self._token_counter % self.config.token_sample_rate != 0
            ):
                return
            if self.config.trace_token_events == "off":
                return

        record["data"] = data_dict
        self.emit(record)

    def emit_host_forward(self, session_id: str, tokens: int, phase: str) -> None:
        """Emit host forward pass record."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "host_forward",
            "tokens": tokens,
            "phase": phase,
        })

    def emit_control_parsed(self, session_id: str, kind: str, job_id: str) -> None:
        """Emit control block parsed record."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "control_parsed",
            "kind": kind,
            "job_id": job_id,
        })

    def emit_job_state(
        self,
        session_id: str,
        job_id: str,
        from_status: str,
        to_status: str,
    ) -> None:
        """Emit job state transition."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "job_state",
            "job_id": job_id,
            "from": from_status,
            "to": to_status,
        })

    def emit_park_enter(self, session_id: str, level: str, jobs: list[str]) -> None:
        """Emit park enter."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "park_enter",
            "level": level,
            "jobs": jobs,
        })

    def emit_wake(self, session_id: str, reason: str) -> None:
        """Emit wake."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "wake",
            "reason": reason,
        })

    def emit_inject(self, session_id: str, event: str, job_id: str) -> None:
        """Emit event injection."""
        self.emit({
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": "inject",
            "event": event,
            "job_id": job_id,
        })

    def emit_buffer_access(
        self,
        session_id: str,
        job_id: str,
        buffer_id: str,
        action: str,
        byte_range: tuple[int, int] | None = None,
    ) -> None:
        """Emit buffer access audit record."""
        record = {
            "ts": time.time(),
            "mono": time.monotonic(),
            "trace": self._current_trace_id,
            "session": session_id,
            "type": f"buffer_{action}",
            "job_id": job_id,
            "buffer_id": buffer_id,
        }
        if byte_range:
            record["byte_range"] = {"start": byte_range[0], "end": byte_range[1]}
        self.emit(record)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# Global trace sink
_trace_sink: TraceSink | None = None


def get_trace_sink() -> TraceSink | None:
    return _trace_sink


def init_trace_sink(config: TraceConfig | None = None) -> TraceSink:
    global _trace_sink
    _trace_sink = TraceSink(config)
    atexit.register(_trace_sink.close)
    return _trace_sink


def shutdown_trace_sink() -> None:
    global _trace_sink
    if _trace_sink:
        _trace_sink.close()
        _trace_sink = None
