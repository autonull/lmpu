"""Pytest configuration and fixtures."""

import asyncio
import pytest
import tempfile
from pathlib import Path

from lmpu.memory import TempSHMBackend, BufferManager
from lmpu.kernel.jobs import JobTable
from lmpu.kernel.events import EventLog
from lmpu.kernel.clock import FakeClock, set_clock, get_clock
from lmpu.kernel.runtime import Runtime


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def fake_clock():
    """Provide a fake clock for deterministic time."""
    clock = FakeClock(0.0)
    set_clock(clock)
    yield clock
    set_clock(RealClock())


@pytest.fixture
def temp_shm_backend():
    """Provide a temporary SHM backend."""
    backend = TempSHMBackend()
    yield backend
    backend.cleanup()


@pytest.fixture
def buffer_manager(temp_shm_backend):
    """Provide a buffer manager with temp backend."""
    return BufferManager(backend=temp_shm_backend)


@pytest.fixture
def job_table():
    """Provide a job table."""
    return JobTable()


@pytest.fixture
def event_log():
    """Provide an event log."""
    return EventLog()


@pytest.fixture
async def runtime():
    """Provide a running runtime instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = Path(tmpdir) / "runtime.sock"
        rt = Runtime(socket_path=str(socket_path))
        await rt.start()
        yield rt
        await rt.stop()


@pytest.fixture
def sample_buffer_data():
    """Sample buffer data for testing."""
    import numpy as np
    return {
        "float32_matrix": np.random.randn(100, 100).astype(np.float32),
        "int64_vector": np.arange(1000, dtype=np.int64),
        "complex64": np.random.randn(50, 50).astype(np.complex64),
    }