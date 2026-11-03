"""End-to-end zero-copy matmul over SCM_RIGHTS buffer passing."""

from __future__ import annotations

import numpy as np
import pytest

from lmpu.kernel.jobs import JobRecord
from lmpu.kernel.runtime import Runtime
from lmpu.kernel.worker_pool import WorkerInfo, WorkerPool
from lmpu.memory.checksum import compute_checksum
from lmpu.protocol.messages import ModelJobStatus

pytestmark = pytest.mark.integration


async def test_vector_matmul_maps_buffers_without_copies(tmp_path) -> None:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={
            "trace_dir": str(tmp_path / "traces"),
            "reaper_interval_ms": 10,
        },
    )
    await runtime.start()
    try:
        hello = await runtime._hello({"protocol": "lmpu/1"})
        session_id = hello.session_id

        rng = np.random.default_rng(7)
        a = rng.standard_normal((32, 64)).astype(np.float32)
        b = rng.standard_normal((64, 16)).astype(np.float32)

        ref_a = await runtime._buffer_import(
            {
                "name": "A",
                "data": a.tobytes(),
                "dtype": "float32",
                "shape": [32, 64],
            },
            session_id,
        )
        ref_b = await runtime._buffer_import(
            {
                "name": "B",
                "data": b.tobytes(),
                "dtype": "float32",
                "shape": [64, 16],
            },
            session_id,
        )
        ref_c = await runtime._buffer_create(
            {"name": "C", "dtype": "float32", "shape": [32, 16]},
            session_id,
        )

        submitted = await runtime._submit(
            {
                "processor": "vector.engine",
                "op": "matmul",
                "inputs": {
                    "A": {"buffer": ref_a.buffer_id},
                    "B": {"buffer": ref_b.buffer_id},
                },
                "outputs": {"C": {"buffer": ref_c.buffer_id}},
                "config": {"mode": "sync", "timeout_ms": 30_000},
            },
            session_id,
        )

        assert submitted.status == ModelJobStatus.COMPLETED, submitted.error
        job = runtime.job_table.get(submitted.job_id)
        assert job is not None
        assert job.copy_count == 0
        assert job.mapped_bytes == a.nbytes + b.nbytes + np.dtype("float32").itemsize * 32 * 16
        assert submitted.result is not None
        assert submitted.result["summary"]["shape"] == [32, 16]

        view = await runtime._read(
            {"buffer": ref_c.buffer_id, "max_inline_bytes": 65536},
            session_id,
        )
        assert view.data is not None
        actual = np.asarray(view.data, dtype=np.float32).reshape(32, 16)
        np.testing.assert_allclose(actual, a @ b, atol=1e-5)
        assert runtime.buffer_manager.checksum(ref_c.buffer_id) == compute_checksum(
            actual.tobytes()
        )

        from lmpu.protocol.messages import EventKind, JobCompletedData

        events = runtime.event_log.get_since(session_id, 0)
        completed = [e for e in events if e.type == EventKind.JOB_COMPLETED]
        assert completed, "missing JobCompleted event"
        assert isinstance(completed[-1].data, JobCompletedData)
        assert completed[-1].data.copy_count == 0
    finally:
        await runtime.stop()


async def test_worker_pool_falls_back_to_inline_copy(tmp_path) -> None:
    pool = WorkerPool(job_table=None, event_log=None)  # type: ignore[arg-type]
    path = tmp_path / "buffer.bin"
    path.write_bytes(b"abc")
    job = JobRecord(
        job_id="job_fallback",
        session_id="s_fallback",
        processor="pure.func",
        op="echo",
        inputs={"x": {"buffer": "buf_x"}},
    )
    pool.set_buffer_provider(
        lambda _job: [
            {
                "name": "x",
                "buffer_id": "buf_x",
                "dtype": "uint8",
                "shape": (3,),
                "bytes": 3,
                "writable": False,
                "path": str(path),
            }
        ]
    )
    worker = WorkerInfo(
        worker_id="w1",
        processor="pure.func",
        process=None,  # type: ignore[arg-type]
        pid=0,
        manifest={},
        started_at=0.0,
        last_heartbeat=0.0,
        fd_socket=None,
    )
    command: dict = {}

    await pool._attach_job_buffers(worker, job, command)

    assert command["buffer_transfer"] == "inline_fallback"
    assert command["buffers_inline"]["x"] == "YWJj"
