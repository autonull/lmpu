"""CPU-only integration tests for the reference runtime and worker path."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lmpu.kernel.runtime import Runtime
from lmpu.protocol.codec import RPCStream, open_rpc_connection

pytestmark = pytest.mark.integration


@pytest.fixture
async def runtime(tmp_path):
    rt = Runtime(
        socket_path=str(tmp_path / "runtime.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces"), "batch_window_ms": 5},
    )
    await rt.start()
    yield rt
    await rt.stop()


@pytest.fixture
async def client(runtime):
    stream = await open_rpc_connection(runtime.socket_path)
    yield stream
    stream.close()
    await stream.wait_closed()


async def call(
    stream: RPCStream,
    request_id: int,
    method: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    await stream.send_request(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    )
    response = await asyncio.wait_for(stream.recv(), timeout=15)
    assert response.result is not None
    return response.result


async def hello(stream: RPCStream) -> str:
    result = await call(stream, 1, "lmpu.hello", {"protocol": "lmpu/1"})
    return result["session_id"]


async def test_capabilities_lists_all_processors(client):
    await hello(client)
    result = await call(client, 2, "lmpu.capabilities", {})

    names = {processor["name"] for processor in result["processors"]}
    assert {"pure.func", "python.vm", "vector.engine", "symbolic.planner",
            "neural.summary", "hardware.stub"} <= names


async def test_sync_submit_returns_result(client):
    await hello(client)
    result = await call(
        client,
        2,
        "lmpu.submit",
        {
            "processor": "pure.func",
            "op": "bigint_mul",
            "inputs": {"a": "12345678901234567890", "b": "10"},
            "config": {"mode": "sync", "timeout_ms": 10000},
        },
    )

    assert result["status"] == "completed"
    assert result["result"] == {"result": "123456789012345678900"}


async def test_async_wait_reports_one_terminal_event(runtime, client):
    session_id = await hello(client)
    submitted = await call(
        client,
        2,
        "lmpu.submit",
        {
            "processor": "pure.func",
            "op": "bigint_add",
            "inputs": {"a": 20, "b": 22},
            "config": {"mode": "async"},
        },
    )

    waited = await call(
        client, 3, "lmpu.wait", {"job_id": submitted["job_id"], "timeout_ms": 10000}
    )
    assert waited["status"] == "completed"
    assert waited["result"] == {"result": "42"}

    events = runtime.event_log.get_since(session_id, 0)
    terminal = [
        event for event in events if event.type.value in {"JobCompleted", "JobFailed", "JobCancelled"}
    ]
    assert [event.type.value for event in terminal] == ["JobCompleted"]


async def test_duplicate_submit_does_not_duplicate_execution(client):
    await hello(client)
    params = {
        "job_id": "job_fixed",
        "processor": "pure.func",
        "op": "bigint_add",
        "inputs": {"a": 1, "b": 2},
        "config": {"mode": "async"},
    }
    first = await call(client, 2, "lmpu.submit", params)
    second = await call(client, 3, "lmpu.submit", params)

    assert first["job_id"] == second["job_id"] == "job_fixed"
    waited = await call(
        client, 4, "lmpu.wait", {"job_id": "job_fixed", "timeout_ms": 10000}
    )
    assert waited["result"] == {"result": "3"}


async def test_cancel_running_job_emits_only_cancelled(runtime, client):
    session_id = await hello(client)
    submitted = await call(
        client,
        2,
        "lmpu.submit",
        {
            "processor": "python.vm",
            "op": "run_program",
            "inputs": {"program": "import time\ntime.sleep(5)\nfinish({'ok': 1})"},
            "config": {"mode": "async", "timeout_ms": 10000},
        },
    )
    await asyncio.sleep(0.5)
    cancelled = await call(client, 3, "lmpu.cancel", {"job_id": submitted["job_id"]})
    assert cancelled["status"] == "cancel_requested"

    job = await runtime.job_table.wait_terminal(submitted["job_id"], timeout=5)
    assert job is not None
    assert job.status.value == "cancelled"

    events = runtime.event_log.get_since(session_id, 0)
    terminal = [
        event.type.value
        for event in events
        if event.type.value in {"JobCompleted", "JobFailed", "JobCancelled"}
    ]
    assert terminal == ["JobCancelled"]


async def test_explicit_batch_preserves_item_order(client):
    await hello(client)
    items = [{"inputs": {"a": index, "b": 1}} for index in range(25)]
    submitted = await call(
        client,
        2,
        "lmpu.submit",
        {
            "processor": "pure.func",
            "op": "bigint_add",
            "batch": {"items": items},
            "config": {"mode": "async"},
        },
    )

    waited = await call(
        client, 3, "lmpu.wait", {"job_id": submitted["job_id"], "timeout_ms": 15000}
    )
    results = waited["result"]["results"]
    assert [item["result"] for item in results] == [str(index + 1) for index in range(25)]
    assert waited["result"]["errors"] == []
