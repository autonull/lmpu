"""Packaged JSON Schemas for the frozen protocol and trace formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).parent
PROTOCOL_SCHEMA_PATH = SCHEMA_DIR / "protocol-v1.schema.json"
TRACE_SCHEMA_PATH = SCHEMA_DIR / "trace-v1.schema.json"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_protocol_schema() -> dict[str, Any]:
    """Return the frozen protocol-v1 JSON Schema."""
    return _load(PROTOCOL_SCHEMA_PATH)


def load_trace_schema() -> dict[str, Any]:
    """Return the frozen trace-v1 JSON Schema."""
    return _load(TRACE_SCHEMA_PATH)
