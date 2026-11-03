"""Failure-injection cases: timeout, OOM, crash, disconnect, duplicates."""

from __future__ import annotations

import asyncio

import pytest

from lmpu.kernel.events import EventLog
from lmpu.kernel.jobs import JobStatus
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.errors import JobError
from lmpu.protocol.messages import (
    Event,
    EventKind,
    ModelJobStatus,
    ProgressData,
)

pytestmark = pytest.mark.failure


async def _runtime(tmp_path) -> Runtime:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={
            "trace_dir": str(tmp_path / "traces"),
            "reaper_interval_ms": 10,
            "cancel_grace_ms": 50,
        },
    )
    await runtime.start()
    return runtime


def _job(runtime: Runtime, job_id: str):
    job = runtime.job_table.get(job_id)
    assert job is not None
    return job


async def test_timeout_fails_worker_and_emits_no_completion(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        with pytest.raises(JobError, match="timed out"):
            await runtime._submit(
                {
                    "processor": "hardware.stub",
                    "op": "simulate_latency",
                    "inputs": {"duration_ms": 5_000},
                    "config": {"mode": "sync", "timeout_ms": 150},
                },
                session,
            )

        job = runtime.job_table.get_by_session(session)[0]
        assert job.status == JobStatus.FAILED
        assert job.error is not None
        assert job.error["code"] == "timeout"

        await asyncio.sleep(0.1)
        events = runtime.event_log.get_since(session, 0)
        assert not [e for e in events if e.type == EventKind.JOB_COMPLETED]
        assert [e for e in events if e.type == EventKind.JOB_FAILED]
    finally:
        await runtime.stop()


async def test_oom_is_isolated_and_reported(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        submitted = await runtime._submit(
            {
                "processor": "hardware.stub",
                "op": "simulate_oom",
                "inputs": {"bytes": 1_000_000},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )

        assert submitted.status == ModelJobStatus.FAILED
        assert submitted.error is not None
        assert submitted.error["code"] == "oom"

        healthy = await runtime._submit(
            {
                "processor": "hardware.stub",
                "op": "deterministic_compute",
                "inputs": {"seed": 1, "operation": "hash"},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )
        assert healthy.status == ModelJobStatus.COMPLETED
    finally:
        await runtime.stop()


async def test_worker_crash_fails_job_and_pool_recovers(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        crashed = await runtime._submit(
            {
                "processor": "hardware.stub",
                "op": "simulate_crash",
                "inputs": {"signal": 17},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )

        assert crashed.status == ModelJobStatus.FAILED
        assert crashed.error is not None
        assert crashed.error["code"] == "worker_crashed"

        recovered = await runtime._submit(
            {
                "processor": "hardware.stub",
                "op": "deterministic_compute",
                "inputs": {"seed": 2, "operation": "hash"},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )
        assert recovered.status == ModelJobStatus.COMPLETED
    finally:
        await runtime.stop()


async def test_session_disconnect_cancels_jobs_and_cleans_shm(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        submitted = await runtime._submit(
            {
                "processor": "hardware.stub",
                "op": "simulate_latency",
                "inputs": {"duration_ms": 5_000},
                "config": {"mode": "async"},
            },
            session,
        )
        await runtime._buffer_import(
            {
                "name": "kept",
                "data": b"payload",
                "dtype": "uint8",
                "shape": [7],
            },
            session,
        )

        closed = await runtime._close({"detach": False}, session)
        assert submitted.job_id in closed.cancelled_jobs

        job = _job(runtime, submitted.job_id)
        assert job.status == JobStatus.CANCELLED
        assert not (tmp_path / "shm" / session).exists()
    finally:
        await runtime.stop()


async def test_event_before_sleep_returns_already_satisfied(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        submitted = await runtime._submit(
            {
                "processor": "pure.func",
                "op": "bigint_add",
                "inputs": {"a": "1", "b": "2"},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )
        assert submitted.status == ModelJobStatus.COMPLETED

        slept = await runtime._sleep(
            {"until": {"all": [submitted.job_id]}, "level": "S1"},
            session,
        )
        assert slept.status == "already_satisfied"
        assert slept.woken is not None
        assert slept.woken["already_satisfied"] is True
        assert submitted.job_id in slept.woken["satisfied_jobs"]
    finally:
        await runtime.stop()


def test_duplicate_event_delivery_is_deduped_by_event_id() -> None:
    log = EventLog()
    event = Event(
        event_id="evt_duplicate",
        seq=0,
        session="s_dedupe",
        job_id="job_1",
        type=EventKind.PROGRESS,
        ts_monotonic=1.0,
        data=ProgressData(job_id="job_1", fraction=0.5),
    )

    first = log.append("s_dedupe", event)
    second = log.append("s_dedupe", event)

    assert first.seq == second.seq == 1
    assert len(log.get_since("s_dedupe", 0)) == 1


async def test_duplicate_completion_does_not_emit_twice(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        submitted = await runtime._submit(
            {
                "processor": "pure.func",
                "op": "bigint_add",
                "inputs": {"a": "1", "b": "2"},
                "config": {"mode": "sync", "timeout_ms": 10_000},
            },
            session,
        )
        job = _job(runtime, submitted.job_id)

        runtime.worker_pool._complete_job(job, {"result": "3"})

        completed = [
            e for e in runtime.event_log.get_since(session, 0) if e.type == EventKind.JOB_COMPLETED
        ]
        assert len(completed) == 1
    finally:
        await runtime.stop()
