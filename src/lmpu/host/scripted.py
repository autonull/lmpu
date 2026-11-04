"""Deterministic scripted decoder for model-free session scenarios."""

from __future__ import annotations

import json
from typing import Any

from lmpu.protocol.messages import ModelMessage
from lmpu.protocol.sentinels import (
    PROC_CLOSE,
    PROC_EVENT_CLOSE,
    PROC_EVENT_OPEN,
    PROC_OPEN,
)

from .grammar import ConstrainedDecoder


class ScriptedDecoder(ConstrainedDecoder):
    """Return pre-scripted output segments in order.

    Segments may be plain strings or dictionaries:

    - ``{"type": "free", "text": "..."}``
    - ``{"type": "control", "control": {...}}``
    - ``{"type": "malformed_control", "text": "..."}``
    - ``{"type": "event", "event": {...}}``
    - ``{"type": "error", "error": "..."}``
    - ``{"type": "raw", "text": "..."}``
    """

    def __init__(self, segments: list[Any]):
        self.segments = list(segments)
        self.position = 0
        self.prompts: list[str] = []
        self.constrained_prompts: list[str] = []

    @property
    def remaining(self) -> int:
        return max(0, len(self.segments) - self.position)

    def _next_text(self) -> str:
        if self.position >= len(self.segments):
            return ""

        segment = self.segments[self.position]
        self.position += 1
        if isinstance(segment, str):
            return segment
        if not isinstance(segment, dict):
            raise TypeError(f"Unsupported scripted segment: {segment!r}")

        kind = segment.get("type", "raw")
        if kind == "free":
            return str(segment.get("text", ""))
        if kind == "raw":
            return str(segment.get("text", ""))
        if kind == "control":
            control = segment.get("control", {})
            body = (
                json.dumps(control, separators=(",", ":"))
                if isinstance(control, dict)
                else str(control)
            )
            return f"{PROC_OPEN}{body}{PROC_CLOSE}"
        if kind == "malformed_control":
            return f"{PROC_OPEN}{segment.get('text', '')}{PROC_CLOSE}"
        if kind == "event":
            event = segment.get("event", {})
            body = (
                json.dumps(event, separators=(",", ":")) if isinstance(event, dict) else str(event)
            )
            return f"{PROC_EVENT_OPEN}{body}{PROC_EVENT_CLOSE}"
        if kind == "error":
            raise RuntimeError(str(segment.get("error", "scripted engine error")))
        raise ValueError(f"Unknown scripted segment type: {kind}")

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        self.prompts.append(prompt)
        return self._next_text()

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        self.constrained_prompts.append(prompt)
        return self._next_text()


def control_segment(message: ModelMessage) -> dict[str, Any]:
    """Build a scripted control segment from a typed model message."""
    return {
        "type": "control",
        "control": message.model_dump(mode="json", exclude_none=True),
    }


def free_segment(text: str) -> dict[str, Any]:
    return {"type": "free", "text": text}
