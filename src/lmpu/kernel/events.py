"""LMPU Kernel - Event log with sequence numbers and replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..protocol.messages import (
    Event,
)


@dataclass
class SessionEventLog:
    """Per-session event log with ring buffer."""

    session_id: str
    events: deque = field(default_factory=lambda: deque(maxlen=10000))
    seq_counter: int = 0
    ack_cursor: int = 0  # Highest acknowledged sequence
    event_ids: dict[str, Event] = field(default_factory=dict)

    def append(self, event: Event) -> Event:
        """Append event, assign sequence number, and dedupe by event_id."""
        if event.event_id and event.event_id in self.event_ids:
            return self.event_ids[event.event_id]
        self.seq_counter += 1
        event.seq = self.seq_counter
        self.events.append(event)
        if event.event_id:
            self.event_ids[event.event_id] = event
        return event

    def get_since(self, seq: int) -> list[Event]:
        """Get all events with seq > given sequence."""
        return [e for e in self.events if e.seq > seq]

    def get_range(self, start_seq: int, end_seq: int) -> list[Event]:
        """Get events in sequence range (inclusive)."""
        return [e for e in self.events if start_seq <= e.seq <= end_seq]

    def ack(self, seq: int) -> int:
        """Advance acknowledgement cursor."""
        if seq > self.ack_cursor:
            self.ack_cursor = seq
        return self.ack_cursor

    def get_unacked(self) -> list[Event]:
        """Get events after ack cursor."""
        return [e for e in self.events if e.seq > self.ack_cursor]

    def trim_to_size(self, max_size: int) -> int:
        """Trim events to max size, keeping most recent. Returns count removed."""
        if len(self.events) <= max_size:
            return 0
        # Keep ack_cursor events plus some buffer
        keep_from = max(0, self.ack_cursor - 100)
        if keep_from < len(self.events) - max_size:
            keep_from = len(self.events) - max_size
        removed = 0
        while len(self.events) > max_size and self.events:
            if self.events[0].seq < keep_from:
                self.events.popleft()
                removed += 1
            else:
                break
        return removed


class EventLog:
    """Global event log managing per-session logs."""

    def __init__(self, max_events_per_session: int = 10000):
        self._sessions: dict[str, SessionEventLog] = {}
        self.max_events_per_session = max_events_per_session

    def _get_or_create(self, session_id: str) -> SessionEventLog:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionEventLog(session_id)
        return self._sessions[session_id]

    def append(self, session_id: str, event: Event) -> Event:
        """Append event to session log."""
        log = self._get_or_create(session_id)
        return log.append(event)

    def get_since(self, session_id: str, seq: int) -> list[Event]:
        """Get events since sequence number."""
        log = self._sessions.get(session_id)
        if not log:
            return []
        return log.get_since(seq)

    def get_range(self, session_id: str, start: int, end: int) -> list[Event]:
        log = self._sessions.get(session_id)
        if not log:
            return []
        return log.get_range(start, end)

    def ack(self, session_id: str, seq: int) -> int:
        """Acknowledge events up to sequence."""
        log = self._sessions.get(session_id)
        if not log:
            return 0
        return log.ack(seq)

    def get_ack_cursor(self, session_id: str) -> int:
        log = self._sessions.get(session_id)
        return log.ack_cursor if log else 0

    def get_last_seq(self, session_id: str) -> int:
        log = self._sessions.get(session_id)
        return log.seq_counter if log else 0

    def has_gap(self, session_id: str, resume_from: int) -> bool:
        """Check if there's a gap in retained events."""
        log = self._sessions.get(session_id)
        if not log:
            return False
        if resume_from <= 0:
            return False
        # Check if earliest retained event is after resume_from
        if not log.events:
            return True
        earliest = log.events[0].seq
        return earliest > resume_from

    def remove_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def trim_all(self) -> int:
        total = 0
        for log in self._sessions.values():
            total += log.trim_to_size(self.max_events_per_session)
        return total

    def get_session_ids(self) -> list[str]:
        return list(self._sessions.keys())
