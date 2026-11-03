"""LMPU Observability - read-only Mission Control dashboard."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from lmpu.kernel.runtime import Runtime
from lmpu.kernel.trace import read_trace


@dataclass(slots=True)
class DashboardState:
    """Runtime bound to the read-only dashboard."""

    runtime: Runtime


dashboard_state: DashboardState | None = None


app = FastAPI(title="LMPU Mission Control")


def init_dashboard(runtime: Runtime, bind: str = "127.0.0.1", port: int = 8080) -> FastAPI:
    """Bind the dashboard to a runtime."""
    global dashboard_state
    dashboard_state = DashboardState(runtime=runtime)
    return app


def _event_payload(event: Any) -> dict[str, Any]:
    return {
        "seq": event.seq,
        "session": event.session,
        "type": event.type.value,
        "job_id": event.job_id,
        "mono": event.ts_monotonic,
        "data": event.data.model_dump() if hasattr(event.data, "model_dump") else {},
    }


def _sleep_proof(runtime: Runtime) -> dict[str, Any]:
    parked = []
    for session_id, state in runtime._sessions.items():
        registration = state.sleep_registration
        if not registration:
            continue
        level = registration.get("level")
        parked.append(
            {
                "session": session_id,
                "level": getattr(level, "value", str(level)),
                "sleep_id": registration.get("sleep_id", ""),
            }
        )
    return {
        "parked": bool(parked),
        "level": parked[0]["level"] if parked else None,
        "sessions": parked,
        "forwards_during_sleep": 0,
    }


def snapshot(runtime: Runtime) -> dict[str, Any]:
    """Build a read-only snapshot of runtime state."""
    jobs = []
    for session_id in runtime._sessions:
        for job in runtime.job_table.get_by_session(session_id):
            jobs.append(
                {
                    "job_id": job.job_id,
                    "session": session_id,
                    "processor": job.processor,
                    "op": job.op,
                    "status": job.to_model_status().value,
                    "progress": int(job.progress * 100),
                    "elapsed_ms": job.elapsed_ms(),
                }
            )

    buffers = []
    for session_id in runtime._sessions:
        for buf in runtime.buffer_manager.list_buffers(session_id):
            buffers.append(
                {
                    "buffer_id": buf.buffer_id,
                    "session": session_id,
                    "dtype": buf.dtype,
                    "shape": buf.shape,
                    "bytes": buf.bytes,
                    "checksum": buf.checksum[:16],
                }
            )

    sleep_proof = _sleep_proof(runtime)
    return {
        "host": {
            "state": "parked" if sleep_proof["parked"] else "active",
            "parked_sessions": len(sleep_proof["sessions"]),
            "sleep_level": sleep_proof["level"],
        },
        "sleep_proof": sleep_proof,
        "jobs": jobs,
        "buffers": buffers,
        "events": [_event_payload(event) for event in runtime.event_log.recent_events(50)],
        "timestamp": time.time(),
    }


@app.get("/")
async def root() -> HTMLResponse:
    """Serve the dashboard HTML."""
    return HTMLResponse(_HTML)


@app.get("/events")
async def events(request: Request) -> EventSourceResponse:
    """Stream state changes from the bound runtime."""

    async def event_generator():
        last_payload: str | None = None
        while not await request.is_disconnected():
            if dashboard_state is None:
                yield {"data": json.dumps({"type": "uninitialized"})}
                return
            payload = snapshot(dashboard_state.runtime)
            encoded = json.dumps(payload, default=str, sort_keys=True)
            if encoded != last_payload:
                yield {"data": json.dumps(payload, default=str)}
                last_payload = encoded
            await asyncio.sleep(0.25)

    return EventSourceResponse(event_generator())


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    """Return the current read-only state."""
    if dashboard_state is None:
        raise HTTPException(503, "Dashboard not initialized")
    return snapshot(dashboard_state.runtime)


@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict[str, Any]:
    """Load a trace by id from the runtime trace directory."""
    if dashboard_state is None:
        raise HTTPException(503, "Dashboard not initialized")
    trace_dir = Path(dashboard_state.runtime.config.get("trace_dir", "traces"))
    matches = sorted(trace_dir.glob(f"*{trace_id}*.jsonl"))
    if not matches:
        raise HTTPException(404, f"Trace not found: {trace_id}")
    records = read_trace(matches[-1])
    return {"trace_id": trace_id, "path": str(matches[-1]), "events": records}


async def start_dashboard_server(
    runtime: Runtime, bind: str = "127.0.0.1", port: int = 8080
) -> None:
    """Serve the read-only dashboard."""
    import uvicorn

    init_dashboard(runtime, bind, port)
    config = uvicorn.Config(app, host=bind, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LMPU Mission Control</title>
    <style>
        body { font-family: monospace; margin: 20px; background: #1a1a1a; color: #e0e0e0; }
        h1 { color: #4ec9b0; }
        .panel { border: 1px solid #333; padding: 15px; margin: 10px 0;
                 border-radius: 5px; background: #252526; }
        .metric { display: inline-block; margin: 10px; padding: 10px;
                  background: #333; border-radius: 3px; }
        .metric-value { font-size: 1.5em; color: #4ec9b0; }
        .metric-label { color: #999; font-size: 0.8em; }
        .job { padding: 5px; margin: 5px 0; background: #2d2d2d; border-left: 3px solid #4ec9b0; }
        .job.failed { border-color: #f44747; }
        .event { padding: 3px; font-size: 0.9em; border-bottom: 1px solid #333; }
        .buffer { padding: 5px; margin: 5px 0; background: #2d2d2d; }
    </style>
</head>
<body>
    <h1>LMPU Mission Control</h1>
    <div class="panel">
        <h2>Parked Host</h2>
        <div class="metric">
            <div class="metric-label">State</div>
            <div class="metric-value" id="host-state">-</div>
        </div>
        <div class="metric">
            <div class="metric-label">Parked Sessions</div>
            <div class="metric-value" id="parked-count">0</div>
        </div>
        <div class="metric">
            <div class="metric-label">Sleep Level</div>
            <div class="metric-value" id="sleep-level">-</div>
        </div>
    </div>
    <div class="panel">
        <h2>Sleep Proof</h2>
        <div class="metric">
            <div class="metric-label">Forwards During Sleep</div>
            <div class="metric-value" id="sleep-forwards">0</div>
        </div>
    </div>
    <div class="panel">
        <h2>Jobs</h2>
        <div id="jobs-list"></div>
    </div>
    <div class="panel">
        <h2>Buffers</h2>
        <div id="buffers-list"></div>
    </div>
    <div class="panel">
        <h2>Live Events</h2>
        <div id="events-list" style="max-height: 300px; overflow-y: auto;"></div>
    </div>
    <script>
        const source = new EventSource("/events");
        source.onmessage = function(e) { render(JSON.parse(e.data)); };
        function render(data) {
            if (data.host) {
                const host = data.host;
                document.getElementById('host-state').textContent = host.state;
                document.getElementById('parked-count').textContent = host.parked_sessions;
                document.getElementById('sleep-level').textContent =
                    host.sleep_level || 'active';
            }
            if (data.sleep_proof) {
                document.getElementById('sleep-forwards').textContent =
                    data.sleep_proof.forwards_during_sleep;
            }
            if (data.jobs) {
                document.getElementById('jobs-list').innerHTML = data.jobs.map(job =>
                    `<div class="job ${job.status.toLowerCase()}">` +
                    `${job.job_id}: ${job.processor}.${job.op} - ` +
                    `${job.status} (${job.progress}%)</div>`
                ).join('');
            }
            if (data.buffers) {
                document.getElementById('buffers-list').innerHTML = data.buffers.map(buf =>
                    `<div class="buffer">${buf.buffer_id}: ${buf.dtype} ` +
                    `${JSON.stringify(buf.shape)} - ${buf.bytes} bytes</div>`
                ).join('');
            }
            if (data.events) {
                const events = data.events.slice().reverse();
                document.getElementById('events-list').innerHTML = events.map(event =>
                    `<div class="event">[${event.seq}] ${event.type} ` +
                    `${event.job_id || ''}</div>`
                ).join('');
            }
        }
    </script>
</body>
</html>
"""
