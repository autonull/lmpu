"""Frozen protocol/trace schemas and example validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from lmpu.schema import load_protocol_schema, load_trace_schema

ROOT = Path(__file__).resolve().parents[2]


def _builder_module():
    path = ROOT / "scripts" / "export_schema.py"
    spec = importlib.util.spec_from_file_location("export_schema", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator(schema: dict[str, Any], name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"}
    )


def test_committed_schemas_match_generator() -> None:
    builder = _builder_module()

    assert load_protocol_schema() == builder.build_protocol_schema()
    assert load_trace_schema() == builder.build_trace_schema()


@pytest.mark.parametrize(
    ("definition", "example"),
    [
        (
            "RPCRequest",
            {"jsonrpc": "2.0", "id": "1", "method": "hello", "params": {"protocol": "lmpu/1"}},
        ),
        (
            "RPCResponse",
            {
                "jsonrpc": "2.0",
                "id": "1",
                "result": {
                    "session_id": "s_01",
                    "protocol": "lmpu/1",
                    "schema_hash": "abc",
                    "resume_status": "none",
                    "server_time": 1.0,
                },
            },
        ),
        (
            "SubmitParams",
            {
                "job_id": "job_01",
                "processor": "pure.func",
                "op": "bigint_add",
                "inputs": {"a": 1, "b": 2},
                "config": {"mode": "sync"},
            },
        ),
        (
            "SubmitResult",
            {
                "job_id": "job_01",
                "status": "completed",
                "queue_position": 0,
                "seq": 0,
                "result": {"result": "3"},
            },
        ),
        ("SleepParams", {"until": {"all": ["job_01"]}, "level": "S1"}),
        (
            "BufferView",
            {
                "buffer": "buf_01",
                "dtype": "uint8",
                "derived_buffer": "buf_02",
                "note": "use a bounded slice",
            },
        ),
        (
            "Event",
            {
                "event_id": "evt_01",
                "seq": 0,
                "session": "s_01",
                "job_id": "job_01",
                "type": "JobCompleted",
                "ts_monotonic": 1.0,
                "ts_wall": 1000.0,
                "data": {
                    "job_id": "job_01",
                    "status": "completed",
                    "result": {"result": "3"},
                    "checksum": "abc",
                },
            },
        ),
    ],
)
def test_protocol_examples_validate(definition: str, example: dict[str, Any]) -> None:
    _validator(load_protocol_schema(), definition).validate(example)


@pytest.mark.parametrize(
    "record",
    [
        {
            "type": "session_start",
            "trace_id": "t_01",
            "session_id": "s_01",
            "ts": 1000.0,
            "mono": 1.0,
        },
        {
            "type": "event_jobcompleted",
            "event_id": "evt_01",
            "seq": 0,
            "session": "s_01",
            "job_id": "job_01",
            "ts": 1000.0,
            "mono": 1.0,
            "data": {"status": "completed"},
        },
        {
            "type": "host_forward",
            "session": "s_01",
            "tokens": 4,
            "phase": "generated",
            "ts": 1000.0,
            "mono": 1.0,
        },
        {
            "type": "host_metrics",
            "session": "s_01",
            "ts": 1000.0,
            "mono": 1.0,
            "metrics": {
                "schema": "lmpu.host_metrics.v1",
                "session_id": "s_01",
                "forward_passes": 1,
                "prefill_tokens": 2,
                "generated_tokens": 3,
                "control_tokens": 4,
                "injected_tokens": 5,
                "total_tokens": 9,
                "peak_context_tokens": 9,
                "vram_peak_mb": 0.0,
                "gpu_energy_joules": 0.0,
                "cpu_energy_joules": 0.0,
                "sleep_duration_seconds": 0.0,
                "wake_latency_seconds": 0.0,
                "wall_time_seconds": 1.0,
            },
        },
        {
            "type": "park_enter",
            "session": "s_01",
            "level": "S1",
            "jobs": ["job_01"],
            "ts": 1000.0,
            "mono": 1.0,
        },
        {
            "type": "buffer_read",
            "session": "s_01",
            "job_id": "job_01",
            "buffer_id": "buf_01",
            "byte_range": {"start": 0, "end": 1024},
            "ts": 1000.0,
            "mono": 1.0,
        },
    ],
)
def test_trace_examples_validate(record: dict[str, Any]) -> None:
    _validator(load_trace_schema(), "TraceRecord").validate(record)


def test_trace_schema_rejects_unknown_record_type() -> None:
    validator = _validator(load_trace_schema(), "TraceRecord")
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({"type": "not_a_record", "ts": 1.0, "mono": 1.0})
