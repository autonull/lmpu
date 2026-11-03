"""LMPU Host - Sentinel scanner and parser for control blocks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from lmpu.protocol.sentinels import PROC_OPEN, PROC_CLOSE, PROC_EVENT_OPEN, PROC_EVENT_CLOSE
from lmpu.protocol.messages import parse_model_message, ModelMessage, ControlErrorData, ErrorCode
from lmpu.protocol.errors import ValidationError


@dataclass
class ScanResult:
    """Result of sentinel scanning."""
    text_before: str
    control_block: Optional[str]
    text_after: str
    found_open: bool
    found_close: bool
    open_pos: int
    close_pos: int


class SentinelScanner:
    """Scans model output for sentinel tokens."""
    
    def __init__(
        self,
        open_sentinel: str = PROC_OPEN,
        close_sentinel: str = PROC_CLOSE,
        event_open_sentinel: str = PROC_EVENT_OPEN,
        event_close_sentinel: str = PROC_EVENT_CLOSE,
    ):
        self.open_sentinel = open_sentinel
        self.close_sentinel = close_sentinel
        self.event_open_sentinel = event_open_sentinel
        self.event_close_sentinel = event_close_sentinel
        
        # Compile regex for efficient scanning
        self._open_pattern = re.compile(re.escape(open_sentinel))
        self._close_pattern = re.compile(re.escape(close_sentinel))
        self._event_open_pattern = re.compile(re.escape(event_open_sentinel))
        self._event_close_pattern = re.compile(re.escape(event_close_sentinel))

    def scan(self, text: str) -> ScanResult:
        """Scan text for control block sentinels."""
        # Find opening sentinel
        open_match = self._open_pattern.search(text)
        if not open_match:
            return ScanResult(
                text_before=text,
                control_block=None,
                text_after="",
                found_open=False,
                found_close=False,
                open_pos=-1,
                close_pos=-1,
            )
        
        open_pos = open_match.start()
        after_open = text[open_match.end():]
        
        # Find closing sentinel
        close_match = self._close_pattern.search(after_open)
        if not close_match:
            # Incomplete control block
            return ScanResult(
                text_before=text[:open_pos],
                control_block=None,
                text_after=after_open,
                found_open=True,
                found_close=False,
                open_pos=open_pos,
                close_pos=-1,
            )
        
        close_pos = open_match.end() + close_match.start()
        control_block = after_open[:close_match.start()]
        text_after = after_open[close_match.end():]
        
        return ScanResult(
            text_before=text[:open_pos],
            control_block=control_block.strip(),
            text_after=text_after,
            found_open=True,
            found_close=True,
            open_pos=open_pos,
            close_pos=close_pos,
        )

    def scan_event(self, text: str) -> ScanResult:
        """Scan text for event sentinels."""
        open_match = self._event_open_pattern.search(text)
        if not open_match:
            return ScanResult(
                text_before=text,
                control_block=None,
                text_after="",
                found_open=False,
                found_close=False,
                open_pos=-1,
                close_pos=-1,
            )
        
        open_pos = open_match.start()
        after_open = text[open_match.end():]
        
        close_match = self._event_close_pattern.search(after_open)
        if not close_match:
            return ScanResult(
                text_before=text[:open_pos],
                control_block=None,
                text_after=after_open,
                found_open=True,
                found_close=False,
                open_pos=open_pos,
                close_pos=-1,
            )
        
        close_pos = open_match.end() + close_match.start()
        event_block = after_open[:close_match.start()]
        text_after = after_open[close_match.end():]
        
        return ScanResult(
            text_before=text[:open_pos],
            control_block=event_block.strip(),
            text_after=text_after,
            found_open=True,
            found_close=True,
            open_pos=open_pos,
            close_pos=close_pos,
        )


class ControlBlockParser:
    """Parses and validates control blocks."""
    
    def __init__(
        self,
        max_control_tokens: int = 512,
        repair_whitespace: bool = True,
        repair_trailing_comma: bool = True,
    ):
        self.max_control_tokens = max_control_tokens
        self.repair_whitespace = repair_whitespace
        self.repair_trailing_comma = repair_trailing_comma

    def parse(self, control_text: str) -> Optional[ModelMessage]:
        """Parse control block text into a ModelMessage."""
        if not control_text:
            return None
        
        # Apply minimal repairs
        repaired = self._repair(control_text)
        
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON in control block: {e}",
                code=ErrorCode.INVALID_SCHEMA.value,
                details={"position": e.pos, "text": control_text[:200]},
            )
        
        # Validate version
        version = data.get("version", 1)
        if version != 1:
            raise ValidationError(
                message=f"Unsupported protocol version: {version}",
                code=ErrorCode.PROTOCOL_VERSION.value,
            )
        
        # Parse message
        try:
            message = parse_model_message(data)
        except ValueError as e:
            raise ValidationError(
                message=str(e),
                code=ErrorCode.INVALID_SCHEMA.value,
            )
        
        # Validate control token budget (rough estimate)
        token_estimate = len(control_text) // 4  # rough estimate
        if token_estimate > self.max_control_tokens:
            raise ValidationError(
                message=f"Control block exceeds token budget: {token_estimate} > {self.max_control_tokens}",
                code=ErrorCode.INVALID_SCHEMA.value,
            )
        
        return message

    def _repair(self, text: str) -> str:
        """Apply minimal repairs to JSON."""
        repaired = text
        
        if self.repair_whitespace:
            # Normalize whitespace
            repaired = re.sub(r'\s+', ' ', repaired).strip()
        
        if self.repair_trailing_comma:
            # Remove trailing commas in objects and arrays
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        
        return repaired

    def validate_against_schema(
        self,
        message: ModelMessage,
        schema: dict[str, Any],
    ) -> list[ControlErrorData]:
        """Validate message against JSON schema. Returns list of errors."""
        # Placeholder for JSON schema validation
        # Would use jsonschema library
        return []


class IncrementalScanner:
    """Incremental scanner for streaming model output."""
    
    def __init__(self, scanner: SentinelScanner):
        self.scanner = scanner
        self.buffer = ""
        self.in_control_block = False
        self.control_buffer = ""
    
    def feed(self, text: str) -> tuple[str, Optional[ModelMessage]]:
        """Feed text and return (free_text, control_message)."""
        self.buffer += text
        
        if self.in_control_block:
            # Looking for close sentinel
            result = self.scanner.scan(self.control_buffer + self.buffer)
            if result.found_close:
                self.in_control_block = False
                self.control_buffer = ""
                self.buffer = result.text_after
                # Parse control block
                if result.control_block:
                    parser = ControlBlockParser()
                    control_msg = parser.parse(result.control_block)
                    return result.text_before, control_msg
            else:
                self.control_buffer += self.buffer
                self.buffer = ""
                return "", None
        else:
            # Looking for open sentinel
            result = self.scanner.scan(self.buffer)
            if result.found_open:
                if result.found_close:
                    # Complete control block in buffer
                    self.buffer = result.text_after
                    parser = ControlBlockParser()
                    control_msg = parser.parse(result.control_block) if result.control_block else None
                    return result.text_before, control_msg
                else:
                    # Open sentinel found, but no close yet
                    self.in_control_block = True
                    self.control_buffer = result.text_after
                    self.buffer = result.text_before
                    return result.text_before, None
            else:
                # No sentinels, all free text
                free_text = self.buffer
                self.buffer = ""
                return free_text, None
    
    def flush(self) -> tuple[str, Optional[ModelMessage]]:
        """Flush remaining buffer."""
        if self.in_control_block:
            # Incomplete control block at end
            parser = ControlBlockParser()
            control_msg = parser.parse(self.control_buffer) if self.control_buffer else None
            self.in_control_block = False
            self.control_buffer = ""
            return "", control_msg
        free_text = self.buffer
        self.buffer = ""
        return free_text, None