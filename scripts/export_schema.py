"""Export the frozen protocol and trace JSON Schemas.

Run ``uv run python scripts/export_schema.py`` after a protocol change, then
commit the regenerated files in ``src/lmpu/schema/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lmpu.protocol.messages import (
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
    CloseParams,
    CloseResult,
    ConfigureParams,
    Event,
    HealthResult,
    HelloParams,
    HelloResult,
    InjectedEvent,
    JobsListParams,
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
    WakeParams,
    WakeResult,
    WriteParams,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "src" / "lmpu" / "schema"

PROTOCOL_MODELS: tuple[type[BaseModel], ...] = (
    RPCRequest,
    RPCResponse,
    RPCNotification,
    RPCError,
    Event,
    InjectedEvent,
    HelloParams,
    HelloResult,
    CapabilitiesParams,
    CapabilitiesResult,
    SubmitParams,
    SubmitResult,
    PollParams,
    PollResult,
    WaitParams,
    WaitResult,
    ReadParams,
    WriteParams,
    CancelParams,
    CancelResult,
    SleepParams,
    SleepResult,
    WakeParams,
    WakeResult,
    AckParams,
    AckResult,
    BufferCreateParams,
    BufferImportParams,
    BufferRef,
    BufferView,
    BufferStatParams,
    BufferChecksumParams,
    BufferDeleteParams,
    JobsListParams,
    ConfigureParams,
    HealthResult,
    CloseParams,
    CloseResult,
)


def build_protocol_schema() -> dict[str, Any]:
    """Build one merged schema whose ``$defs`` hold every public model."""
    definitions: dict[str, Any] = {}
    for model in PROTOCOL_MODELS:
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
        definitions.update(schema.pop("$defs", {}))
        definitions[model.__name__] = schema
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://lmpu.dev/schema/protocol-v1.schema.json",
        "title": "LMPU Protocol v1",
        "$defs": definitions,
    }


def _object(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def build_trace_schema() -> dict[str, Any]:
    """Build the frozen JSONL trace record schema."""
    common = {
        "ts": {"type": "number"},
        "mono": {"type": "number"},
        "session": {"type": "string"},
        "trace": {"type": ["string", "null"]},
    }
    host_metrics = _object(
        [
            "schema",
            "session_id",
            "forward_passes",
            "prefill_tokens",
            "generated_tokens",
            "control_tokens",
            "injected_tokens",
            "total_tokens",
            "peak_context_tokens",
            "vram_peak_mb",
            "gpu_energy_joules",
            "cpu_energy_joules",
            "sleep_duration_seconds",
            "wake_latency_seconds",
            "wall_time_seconds",
        ],
        {
            "schema": {"const": "lmpu.host_metrics.v1"},
            "session_id": {"type": "string"},
            "forward_passes": {"type": "integer"},
            "prefill_tokens": {"type": "integer"},
            "generated_tokens": {"type": "integer"},
            "control_tokens": {"type": "integer"},
            "injected_tokens": {"type": "integer"},
            "total_tokens": {"type": "integer"},
            "peak_context_tokens": {"type": "integer"},
            "vram_peak_mb": {"type": "number"},
            "gpu_energy_joules": {"type": "number"},
            "cpu_energy_joules": {"type": "number"},
            "sleep_duration_seconds": {"type": "number"},
            "wake_latency_seconds": {"type": "number"},
            "wall_time_seconds": {"type": "number"},
        },
    )
    records = [
        _object(
            ["type", "trace_id", "session_id", "ts", "mono"],
            {
                "type": {"enum": ["session_start", "session_end"]},
                "trace_id": {"type": "string"},
                "session_id": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "event_id", "seq", "ts", "mono"],
            {
                "type": {"type": "string", "pattern": "^event_"},
                "event_id": {"type": "string"},
                "seq": {"type": "integer", "minimum": 0},
                "job_id": {"type": "string"},
                "data": {"type": "object"},
                **common,
            },
        ),
        _object(
            ["type", "session", "tokens", "phase", "ts", "mono"],
            {
                "type": {"const": "host_forward"},
                "tokens": {"type": "integer", "minimum": 0},
                "phase": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "metrics", "ts", "mono"],
            {
                "type": {"const": "host_metrics"},
                "metrics": {"$ref": "#/$defs/HostMetricsV1"},
                **common,
            },
        ),
        _object(
            ["type", "session", "kind", "job_id", "ts", "mono"],
            {
                "type": {"const": "control_parsed"},
                "kind": {"type": "string"},
                "job_id": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "job_id", "from", "to", "ts", "mono"],
            {
                "type": {"const": "job_state"},
                "job_id": {"type": "string"},
                "from": {"type": "string"},
                "to": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "level", "jobs", "ts", "mono"],
            {
                "type": {"const": "park_enter"},
                "level": {"type": "string"},
                "jobs": {"type": "array", "items": {"type": "string"}},
                **common,
            },
        ),
        _object(
            ["type", "session", "reason", "ts", "mono"],
            {
                "type": {"const": "wake"},
                "reason": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "event", "job_id", "ts", "mono"],
            {
                "type": {"const": "inject"},
                "event": {"type": "string"},
                "job_id": {"type": "string"},
                **common,
            },
        ),
        _object(
            ["type", "session", "job_id", "buffer_id", "ts", "mono"],
            {
                "type": {"type": "string", "pattern": "^buffer_"},
                "job_id": {"type": "string"},
                "buffer_id": {"type": "string"},
                "byte_range": {
                    "type": "object",
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                },
                **common,
            },
        ),
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://lmpu.dev/schema/trace-v1.schema.json",
        "title": "LMPU Trace v1",
        "$defs": {
            "HostMetricsV1": host_metrics,
            "TraceRecord": {"oneOf": records},
        },
    }


def write_schema(path: Path, schema: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    write_schema(SCHEMA_DIR / "protocol-v1.schema.json", build_protocol_schema())
    write_schema(SCHEMA_DIR / "trace-v1.schema.json", build_trace_schema())


if __name__ == "__main__":
    main()
