"""Deterministic fake-worker message cases."""

from __future__ import annotations

import asyncio

import orjson
import pytest

from lmpu.kernel.events import EventLog
from lmpu.kernel.jobs import JobRecord, JobStatus, JobTable
from lmpu.kernel.worker_pool import WorkerInfo, WorkerPool
from lmpu.protocol.messages import EventKind

pytestmark = pytest.mark.failure


def _setup() -> tuple[WorkerPool, JobTable, EventLog, JobRecord, WorkerInfo]:
    table = JobTable()
    events = EventLog()
    pool = WorkerPool(job_table=table, event_log=events)
    job, created = table.create_job(
        session_id="s_fake",
        processor="pure.func",
        op="echo",
        inputs={},
        outputs={},
    )
    assert created
    table.transition(job.job_id, JobStatus.PENDING)
    table.transition(job.job_id, JobStatus.RUNNING)

    worker = WorkerInfo(
        worker_id="w_fake",
        processor="pure.func",
        process=None,  # type: ignore[arg-type]
        pid=0,
        manifest={"name": "pure.func"},
        started_at=0.0,
        last_heartbeat=0.0,
        current_job=job.job_id,
    )
    pool._workers["w_fake"] = worker
    pool._job_workers[job.job_id] = "w_fake"
    pool._available_workers.setdefault("pure.func", set())
    return pool, table, events, job, worker


async def test_delayed_result_completes_within_deadline() -> None:
    pool, table, _, job, _ = _setup()

    async def deliver_later() -> None:
        await asyncio.sleep(0.05)
        await pool._handle_worker_message(
            "w_fake",
            orjson.dumps({"type": "result", "job_id": job.job_id, "result": {"ok": 1}}),
        )

    task = asyncio.create_task(deliver_later())
    finished = await table.wait_terminal(job.job_id, timeout=1.0)
    await task

    assert finished is not None
    assert finished.status == JobStatus.COMPLETED
    assert finished.result == {"ok": 1}


async def test_duplicate_result_emits_one_terminal_event() -> None:
    pool, table, events, job, _ = _setup()
    message = orjson.dumps({"type": "result", "job_id": job.job_id, "result": {"value": 42}})

    await pool._handle_worker_message("w_fake", message)
    await pool._handle_worker_message("w_fake", message)

    completed = [
        event for event in events.get_since("s_fake", 0) if event.type == EventKind.JOB_COMPLETED
    ]
    assert len(completed) == 1
    assert table.get(job.job_id).status == JobStatus.COMPLETED  # type: ignore[union-attr]


async def test_error_then_late_result_keeps_failed_state() -> None:
    pool, table, events, job, _ = _setup()
    await pool._handle_worker_message(
        "w_fake",
        orjson.dumps(
            {
                "type": "error",
                "job_id": job.job_id,
                "error_code": "execution_error",
                "message": "boom",
            }
        ),
    )
    await pool._handle_worker_message(
        "w_fake",
        orjson.dumps({"type": "result", "job_id": job.job_id, "result": {"late": 1}}),
    )

    assert table.get(job.job_id).status == JobStatus.FAILED  # type: ignore[union-attr]
    assert not [
        event for event in events.get_since("s_fake", 0) if event.type == EventKind.JOB_COMPLETED
    ]


async def test_malformed_json_is_ignored_without_killing_monitor() -> None:
    pool, table, events, job, _ = _setup()

    await pool._handle_worker_message("w_fake", b"{not-json")

    assert table.get(job.job_id).status == JobStatus.RUNNING  # type: ignore[union-attr]
    assert events.get_since("s_fake", 0) == []


async def test_unknown_message_type_is_ignored() -> None:
    pool, table, _, job, _ = _setup()

    await pool._handle_worker_message(
        "w_fake",
        orjson.dumps({"type": "garbage", "job_id": job.job_id}),
    )

    assert table.get(job.job_id).status == JobStatus.RUNNING  # type: ignore[union-attr]


async def test_malformed_progress_is_rejected() -> None:
    pool, table, events, job, _ = _setup()

    await pool._handle_worker_message(
        "w_fake",
        orjson.dumps({"type": "progress", "job_id": job.job_id, "fraction": "not-a-number"}),
    )

    assert table.get(job.job_id).status == JobStatus.RUNNING  # type: ignore[union-attr]
    assert table.get(job.job_id).progress == 0.0  # type: ignore[union-attr]
    assert not [
        event for event in events.get_since("s_fake", 0) if event.type == EventKind.PROGRESS
    ]


async def test_non_object_result_fails_job_instead_of_crashing() -> None:
    pool, table, events, job, _ = _setup()

    await pool._handle_worker_message(
        "w_fake",
        orjson.dumps({"type": "result", "job_id": job.job_id, "result": "oops"}),
    )

    failed = table.get(job.job_id)
    assert failed is not None
    assert failed.status == JobStatus.FAILED
    assert failed.error is not None
    assert failed.error["code"] == "internal"
    assert [event for event in events.get_since("s_fake", 0) if event.type == EventKind.JOB_FAILED]
