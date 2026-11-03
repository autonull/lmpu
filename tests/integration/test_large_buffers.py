"""Large-buffer acceptance: 16 MiB import, bounded slice, derived fallback."""

from __future__ import annotations

import numpy as np
import pytest

from lmpu.kernel.runtime import Runtime

pytestmark = pytest.mark.integration

SIZE = 16 * 1024 * 1024


async def _runtime(tmp_path) -> Runtime:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={
            "trace_dir": str(tmp_path / "traces"),
            "reaper_interval_ms": 10,
        },
    )
    await runtime.start()
    return runtime


async def test_16mib_slice_reads_without_materializing_buffer(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        payload = np.arange(SIZE, dtype=np.uint8) % 251
        ref = await runtime._buffer_import(
            {
                "name": "large",
                "data": payload.tobytes(),
                "dtype": "uint8",
                "shape": [SIZE],
            },
            session,
        )

        start, stop = 1_000_000, 2_000_000
        view = await runtime._read(
            {
                "buffer": ref.buffer_id,
                "slices": [[start, stop, 1]],
                "max_inline_bytes": 2 * 1024 * 1024,
            },
            session,
        )

        assert view.data is not None
        assert np.asarray(view.data, dtype=np.uint8).tolist() == payload[start:stop].tolist()

        inline, derived = runtime.buffer_manager.read_view(
            ref.buffer_id,
            max_inline_bytes=1024 * 1024,
        )
        assert inline is None
        assert derived is not None
        assert derived.bytes == SIZE
        assert derived.shape == [SIZE]
    finally:
        await runtime.stop()


async def test_softmax_writes_mapped_output_without_copy(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = (await runtime._hello({"protocol": "lmpu/1"})).session_id
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4096).astype(np.float32)

        ref_x = await runtime._buffer_import(
            {
                "name": "x",
                "data": x.tobytes(),
                "dtype": "float32",
                "shape": [4096],
            },
            session,
        )
        ref_out = await runtime._buffer_create(
            {"name": "result", "dtype": "float32", "shape": [4096]},
            session,
        )

        submitted = await runtime._submit(
            {
                "processor": "vector.engine",
                "op": "softmax",
                "inputs": {"x": {"buffer": ref_x.buffer_id}},
                "outputs": {"result": {"buffer": ref_out.buffer_id}},
                "config": {"mode": "sync", "timeout_ms": 30_000},
            },
            session,
        )

        assert submitted.status.value == "completed"
        job = runtime.job_table.get(submitted.job_id)
        assert job is not None
        assert job.copy_count == 0

        view = await runtime._read(
            {"buffer": ref_out.buffer_id, "max_inline_bytes": 65536},
            session,
        )
        assert view.data is not None
        result = np.asarray(view.data, dtype=np.float32)
        assert result.shape == (4096,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)
    finally:
        await runtime.stop()
