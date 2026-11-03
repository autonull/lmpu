"""Read-only dashboard exposes parked hosts, jobs, and live events."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from lmpu.kernel.runtime import Runtime
from lmpu.observability import dashboard
from lmpu.protocol.messages import Event, EventKind, JobSubmittedData


def _runtime(tmp_path: Path) -> Runtime:
    return Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces")},
    )


def test_dashboard_shows_parked_host_and_live_events(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    hello = asyncio.run(runtime._hello({"protocol": "lmpu/1"}))
    session_id = hello.session_id
    runtime._sessions[session_id].sleep_registration = {
        "sleep_id": "sleep_test",
        "level": "S1",
        "condition": {},
        "registered_at": 0.0,
    }
    runtime.event_log.append(
        session_id,
        Event(
            seq=0,
            session=session_id,
            job_id="job_dash",
            type=EventKind.JOB_SUBMITTED,
            ts_monotonic=1.0,
            data=JobSubmittedData(
                job_id="job_dash",
                processor="pure.func",
                queue_position=0,
            ),
        ),
    )

    try:
        dashboard.init_dashboard(runtime)
        client = TestClient(dashboard.app)
        state = client.get("/api/state").json()

        assert state["host"]["state"] == "parked"
        assert state["host"]["parked_sessions"] == 1
        assert state["host"]["sleep_level"] == "S1"
        assert state["sleep_proof"]["forwards_during_sleep"] == 0
        assert state["events"][0]["type"] == "JobSubmitted"
        assert state["events"][0]["job_id"] == "job_dash"

        html = client.get("/").text
        assert "Parked Host" in html
        assert "Jobs" in html
        assert "Live Events" in html
    finally:
        dashboard.dashboard_state = None


def test_dashboard_reports_active_host_without_sleep(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    asyncio.run(runtime._hello({"protocol": "lmpu/1"}))

    try:
        dashboard.init_dashboard(runtime)
        client = TestClient(dashboard.app)
        state = client.get("/api/state").json()

        assert state["host"]["state"] == "active"
        assert state["host"]["parked_sessions"] == 0
        assert state["sleep_proof"]["level"] is None
    finally:
        dashboard.dashboard_state = None
