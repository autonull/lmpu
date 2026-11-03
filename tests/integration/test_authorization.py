"""Authorization boundaries, reaper cleanup, and O(1) large-result context."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmpu.host.engine import ScriptedEngine
from lmpu.host.scripted import ScriptedDecoder
from lmpu.host.session import SessionConfig, SessionController
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.errors import BufferError, JobError
from lmpu.protocol.messages import ErrorCode, ReadResultMessage

pytestmark = pytest.mark.integration

BIG = 16 * 1024 * 1024


async def _runtime(tmp_path, **config) -> Runtime:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces"), "reaper_interval_ms": 10, **config},
    )
    await runtime.start()
    return runtime


async def _hello(runtime: Runtime, **session_config) -> str:
    result = await runtime._hello({"protocol": "lmpu/1", "session_config": session_config or None})
    return result.session_id


async def test_cross_session_buffer_access_is_denied(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        owner = await _hello(runtime)
        intruder = await _hello(runtime)
        ref = await runtime._buffer_import(
            {"name": "owned", "data": b"\x01\x02", "dtype": "uint8", "shape": [2]},
            owner,
        )

        with pytest.raises(BufferError) as read_error:
            await runtime._read({"buffer": ref.buffer_id}, intruder)
        assert read_error.value.code == ErrorCode.ACCESS_DENIED.value

        with pytest.raises(BufferError) as write_error:
            await runtime._write({"buffer": ref.buffer_id, "slices": [], "data": [9, 9]}, intruder)
        assert write_error.value.code == ErrorCode.ACCESS_DENIED.value

        with pytest.raises(BufferError):
            await runtime._buffer_stat({"buffer": ref.buffer_id}, intruder)
        with pytest.raises(BufferError):
            await runtime._buffer_delete({"buffer": ref.buffer_id}, intruder)

        view = await runtime._read({"buffer": ref.buffer_id}, owner)
        assert view.data == [1, 2]
    finally:
        await runtime.stop()


async def test_close_releases_shm_and_grants(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = await _hello(runtime)
        ref = await runtime._buffer_import(
            {"name": "temp", "data": b"abcd", "dtype": "uint8", "shape": [4]},
            session,
        )
        path = runtime.buffer_manager.get_buffer_path(ref.buffer_id)
        assert runtime.grant_manager.has_active_grants(ref.buffer_id) is False

        runtime.buffer_manager.grant_read("job_x", ref.buffer_id)
        assert runtime.grant_manager.has_active_grants(ref.buffer_id) is True

        await runtime._close({}, session)

        assert not (tmp_path / "shm" / session / ref.buffer_id).exists()
        assert path
        assert runtime.grant_manager.has_active_grants(ref.buffer_id) is False
        assert ref.buffer_id not in runtime.buffer_manager._metadata
    finally:
        await runtime.stop()


async def test_ttl_reaper_deletes_expired_buffers(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = await _hello(runtime)
        ref = runtime.buffer_manager.import_bytes(
            session_id=session,
            name="ttl",
            data=b"x",
            dtype="uint8",
            shape=(1,),
            ttl_ms=1,
        )
        runtime.buffer_manager._metadata[ref.buffer_id].created_at -= 10
        reaped = runtime.buffer_manager.reap_expired()

        assert reaped == [ref.buffer_id]
        assert ref.buffer_id not in runtime.buffer_manager._metadata
    finally:
        await runtime.stop()


async def test_session_authorization_is_separate_from_advertisement(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        restricted = await _hello(runtime, authorized_processors=["pure.func"])
        caps = await runtime._capabilities({})
        advertised = {processor.name for processor in caps.processors}
        assert "python.vm" in advertised

        with pytest.raises(JobError) as denied:
            await runtime._submit(
                {"processor": "python.vm", "op": "execute", "config": {"mode": "async"}},
                restricted,
            )
        assert denied.value.code == ErrorCode.ACCESS_DENIED.value

        result = await runtime._submit(
            {
                "op": "bigint_add",
                "inputs": {"a": 20, "b": 22},
                "config": {"mode": "sync", "timeout_ms": 10000},
            },
            restricted,
        )
        assert result.result == {"result": "42"}
    finally:
        await runtime.stop()


async def test_16mib_result_uses_at_most_64_context_tokens(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    controller = SessionController(
        session_id="s_big",
        runtime=runtime,
        engine=ScriptedEngine([]),
        decoder=ScriptedDecoder([]),
        config=SessionConfig(max_context_tokens=100_000),
    )
    try:
        await controller.initialize("read a large result")
        ref = await runtime._buffer_import(
            {
                "name": "huge",
                "data": b"\x00" * BIG,
                "dtype": "uint8",
                "shape": [BIG],
            },
            controller.session_id,
        )

        await controller._handle_read_result(
            ReadResultMessage(
                job_id="job_big",
                buffer=ref.buffer_id,
                max_inline_bytes=1024,
            )
        )
        controller._inject_events()

        rendered = controller._pending_injections
        assert rendered == []
        assert controller.host_state is not None
        assert controller.host_state.injected_tokens <= 64
        assert len(controller.generated_buffer) < 4096
        assert str(BIG) not in controller.generated_buffer
    finally:
        await controller.close()
        await runtime.stop()


async def test_buffer_cycle_does_not_leak_file_descriptors(tmp_path) -> None:
    runtime = await _runtime(tmp_path)
    try:
        session = await _hello(runtime)
        before = len(list(Path("/proc/self/fd").iterdir()))
        for _ in range(10):
            ref = await runtime._buffer_import(
                {"name": "fd", "data": b"abcd", "dtype": "uint8", "shape": [4]},
                session,
            )
            await runtime._read({"buffer": ref.buffer_id}, session)
            await runtime._buffer_delete({"buffer": ref.buffer_id}, session)
        after = len(list(Path("/proc/self/fd").iterdir()))
        assert after <= before + 1
    finally:
        await runtime.stop()
