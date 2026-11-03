"""SDK handshake, submission, buffers, and shutdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmpu.kernel.runtime import Runtime
from lmpu.kernel.trace import replay_trace
from lmpu.sdk import LMPUClient

pytestmark = pytest.mark.integration


async def test_sdk_submit_and_read(tmp_path: Path) -> None:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces")},
    )
    await runtime.start()
    try:
        async with await LMPUClient.connect(str(tmp_path / "run.sock")) as client:
            assert client.session_id.startswith("s_")

            submitted = await client.submit(
                "bigint_add",
                processor="pure.func",
                inputs={"a": 40, "b": 2},
                mode="sync",
            )
            assert submitted.status.value == "completed"
            assert submitted.result == {"result": "42"}

            ref = await client.import_buffer(
                "sdk-buffer",
                "uint8",
                [16 * 1024 * 1024],
                data=b"\x00" * (16 * 1024 * 1024),
            )
            assert ref.bytes == 16 * 1024 * 1024

            view = await client.read(ref.buffer_id, max_inline_bytes=1024)
            assert view.data is None
            assert view.derived_buffer is not None

        traces = sorted((tmp_path / "traces").glob("*.jsonl"))
        assert traces
        replay = replay_trace(traces[-1])
        record_types = {record.get("type") for record in replay.records}
        assert {"event_jobsubmitted", "event_jobstarted", "event_jobcompleted"} <= record_types
        assert replay.is_contiguous()
    finally:
        await runtime.stop()


async def test_sdk_sleep_and_wake(tmp_path: Path) -> None:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces")},
    )
    await runtime.start()
    try:
        async with await LMPUClient.connect(str(tmp_path / "run.sock")) as client:
            submitted = await client.submit(
                "run_program",
                processor="python.vm",
                inputs={"program": "import time\ntime.sleep(1)"},
                mode="async",
            )
            sleep = await client.sleep({"all": [submitted.job_id]}, level="S1")
            assert sleep.status in {"registered", "already_satisfied"}
            await client.wait(submitted.job_id, timeout_ms=10_000)
            woken = await client.wake()
            assert woken.status in {"woken", "no_active_sleep"}
    finally:
        await runtime.stop()
