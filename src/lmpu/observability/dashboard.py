"""LMPU Observability - Mission Control dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from lmpu.kernel.runtime import Runtime
from lmpu.kernel.trace import get_trace_sink


@dataclass
class DashboardState:
    """Dashboard state for SSE updates."""
    runtime: Runtime
    connected_clients: Set[asyncio.Queue] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)


dashboard_state: Optional[DashboardState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global dashboard_state
    # Startup
    yield
    # Shutdown
    if dashboard_state:
        for queue in dashboard_state.connected_clients:
            await queue.put({"type": "shutdown"})


app = FastAPI(title="LMPU Mission Control", lifespan=lifespan)


def init_dashboard(runtime: Runtime, bind: str = "127.0.0.1", port: int = 8080) -> FastAPI:
    """Initialize dashboard with runtime."""
    global dashboard_state
    dashboard_state = DashboardState(runtime=runtime)
    return app


async def broadcast_update(update: dict[str, Any]) -> None:
    """Broadcast update to all connected clients."""
    if dashboard_state:
        for queue in dashboard_state.connected_clients:
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                pass


@app.get("/")
async def root() -> HTMLResponse:
    """Serve dashboard HTML."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LMPU Mission Control</title>
        <style>
            body { font-family: monospace; margin: 20px; background: #1a1a1a; color: #e0e0e0; }
            h1 { color: #4ec9b0; }
            .panel { border: 1px solid #333; padding: 15px; margin: 10px 0; border-radius: 5px; background: #252526; }
            .metric { display: inline-block; margin: 10px; padding: 10px; background: #333; border-radius: 3px; }
            .metric-value { font-size: 1.5em; color: #4ec9b0; }
            .metric-label { color: #999; font-size: 0.8em; }
            .job { padding: 5px; margin: 5px 0; background: #2d2d2d; border-left: 3px solid #4ec9b0; }
            .job.running { border-color: #dcdcaa; }
            .job.completed { border-color: #4ec9b0; }
            .job.failed { border-color: #f44747; }
            .event { padding: 3px; font-size: 0.9em; border-bottom: 1px solid #333; }
            .event.injected { background: #2d3d2d; }
            .buffer { padding: 5px; margin: 5px 0; background: #2d2d2d; }
            pre { background: #1e1e1e; padding: 10px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>LMPU Mission Control</h1>
        <div id="host-panel" class="panel">
            <h2>Host Engine</h2>
            <div class="metric"><div class="metric-label">State</div><div class="metric-value" id="host-state">-</div></div>
            <div class="metric"><div class="metric-label">Forward Passes</div><div class="metric-value" id="forward-passes">0</div></div>
            <div class="metric"><div class="metric-label">Generated Tokens</div><div class="metric-value" id="gen-tokens">0</div></div>
            <div class="metric"><div class="metric-label">Context Tokens</div><div class="metric-value" id="context-tokens">0</div></div>
            <div class="metric"><div class="metric-label">Sleep Level</div><div class="metric-value" id="sleep-level">-</div></div>
        </div>
        
        <div id="sleep-proof" class="panel">
            <h2>Sleep Proof</h2>
            <div class="metric"><div class="metric-label">Forwards During Sleep</div><div class="metric-value" id="sleep-forwards">0</div></div>
            <div class="metric"><div class="metric-label">Tokens During Sleep</div><div class="metric-value" id="sleep-tokens">0</div></div>
            <div class="metric"><div class="metric-label">GPU Energy (J)</div><div class="metric-value" id="gpu-energy">0</div></div>
            <div class="metric"><div class="metric-label">Wake Latency (ms)</div><div class="metric-value" id="wake-latency">0</div></div>
        </div>
        
        <div id="jobs-panel" class="panel">
            <h2>Jobs</h2>
            <div id="jobs-list"></div>
        </div>
        
        <div id="buffers-panel" class="panel">
            <h2>Buffers</h2>
            <div id="buffers-list"></div>
        </div>
        
        <div id="events-panel" class="panel">
            <h2>Event Stream</h2>
            <div id="events-list" style="max-height: 300px; overflow-y: auto;"></div>
        </div>
        
        <script>
            const evtSource = new EventSource("/events");
            evtSource.onmessage = function(e) {
                const data = JSON.parse(e.data);
                updateDashboard(data);
            };
            
            function updateDashboard(data) {
                if (data.host) {
                    document.getElementById('host-state').textContent = data.host.state;
                    document.getElementById('forward-passes').textContent = data.host.forward_passes;
                    document.getElementById('gen-tokens').textContent = data.host.generated_tokens;
                    document.getElementById('context-tokens').textContent = data.host.context_tokens;
                    document.getElementById('sleep-level').textContent = data.host.sleep_level || 'active';
                }
                if (data.sleep_proof) {
                    document.getElementById('sleep-forwards').textContent = data.sleep_proof.forwards_during_sleep;
                    document.getElementById('sleep-tokens').textContent = data.sleep_proof.tokens_during_sleep;
                    document.getElementById('gpu-energy').textContent = data.sleep_proof.gpu_energy_joules.toFixed(2);
                    document.getElementById('wake-latency').textContent = (data.sleep_proof.wake_latency_ms || 0).toFixed(1);
                }
                if (data.jobs) {
                    const container = document.getElementById('jobs-list');
                    container.innerHTML = data.jobs.map(job => 
                        `<div class="job ${job.status}">${job.job_id}: ${job.processor}.${job.op} - ${job.status} (${job.progress || 0}%)</div>`
                    ).join('');
                }
                if (data.buffers) {
                    const container = document.getElementById('buffers-list');
                    container.innerHTML = data.buffers.map(buf => 
                        `<div class="buffer">${buf.buffer_id}: ${buf.dtype} ${JSON.stringify(buf.shape)} - ${buf.bytes} bytes</div>`
                    ).join('');
                }
                if (data.event) {
                    const container = document.getElementById('events-list');
                    const div = document.createElement('div');
                    div.className = 'event' + (data.event.injected ? ' injected' : '');
                    div.textContent = `[${data.event.seq}] ${data.event.type} ${data.event.job_id || ''}`;
                    container.insertBefore(div, container.firstChild);
                    if (container.children.length > 100) container.removeChild(container.lastChild);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/events")
async def events(request: Request):
    """Server-Sent Events endpoint for real-time updates."""
    queue = asyncio.Queue(maxsize=100)
    if dashboard_state:
        dashboard_state.connected_clients.add(queue)
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {"data": json.dumps(data)}
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "heartbeat"})}
        finally:
            if dashboard_state:
                dashboard_state.connected_clients.discard(queue)
    
    return EventSourceResponse(event_generator())


@app.get("/api/state")
async def get_state():
    """Get current dashboard state."""
    if not dashboard_state:
        raise HTTPException(503, "Dashboard not initialized")
    
    runtime = dashboard_state.runtime
    
    # Collect host state
    host_info = {}
    # Would get from active session
    
    # Collect jobs
    jobs = []
    for session_id in runtime._sessions:
        session_jobs = runtime.job_table.get_by_session(session_id)
        for job in session_jobs:
            jobs.append({
                "job_id": job.job_id,
                "processor": job.processor,
                "op": job.op,
                "status": job.to_model_status().value,
                "progress": int(job.progress * 100),
                "elapsed_ms": job.elapsed_ms(),
            })
    
    # Collect buffers
    buffers = []
    for session_id in runtime._sessions:
        session_buffers = runtime.buffer_manager.list_buffers(session_id)
        for buf in session_buffers:
            buffers.append({
                "buffer_id": buf.buffer_id,
                "dtype": buf.dtype,
                "shape": buf.shape,
                "bytes": buf.bytes,
                "checksum": buf.checksum[:16],
            })
    
    # Sleep proof metrics
    sleep_proof = {
        "forwards_during_sleep": 0,
        "tokens_during_sleep": 0,
        "gpu_energy_joules": 0.0,
        "wake_latency_ms": 0.0,
    }
    
    return {
        "host": host_info,
        "jobs": jobs,
        "buffers": buffers,
        "sleep_proof": sleep_proof,
        "timestamp": time.time(),
    }


@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get trace data for replay."""
    trace_sink = get_trace_sink()
    if not trace_sink:
        raise HTTPException(404, "Trace sink not available")
    
    # Would load trace file
    return {"trace_id": trace_id, "events": []}


@app.post("/api/scenario/{scenario_name}")
async def run_scenario(scenario_name: str):
    """Run a demo scenario."""
    # Would trigger scenario
    return {"status": "started", "scenario": scenario_name}


async def start_dashboard_server(runtime: Runtime, bind: str = "127.0.0.1", port: int = 8080):
    """Start dashboard server."""
    import uvicorn
    init_dashboard(runtime, bind, port)
    config = uvicorn.Config(app, host=bind, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()