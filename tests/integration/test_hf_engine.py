"""Opt-in reference-model smoke test. Enable with LMPU_RUN_GPU=1."""

from __future__ import annotations

import os

import pytest

from lmpu.host.hf import HFDecoder, HFEngine
from lmpu.host.session import SessionConfig, SessionController
from lmpu.kernel.runtime import Runtime

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.model,
    pytest.mark.skipif(
        os.environ.get("LMPU_RUN_GPU") != "1",
        reason="set LMPU_RUN_GPU=1 to run the reference model",
    ),
]


async def test_qwen3_5_0_8b_prefill_decode_baseline() -> None:
    engine = HFEngine(model_name="Qwen/Qwen3.5-0.8B", device="cuda", dtype="bfloat16")
    try:
        state = await engine.prefill("Write a list of numbers: 1, 2, 3,")
        assert state.prefill_tokens > 0
        assert type(engine._model).__name__ == "Qwen3_5ForConditionalGeneration"

        text = await engine.decode(state, max_tokens=16, stop_sequences=[])
        assert state.generated_tokens > 0
        assert state.generated_tokens <= 16
        assert text is not None
    finally:
        await engine.close()


async def test_hf_engine_runs_inside_session_loop(tmp_path) -> None:
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={"trace_dir": str(tmp_path / "traces"), "reaper_interval_ms": 10},
    )
    await runtime.start()
    engine = HFEngine(model_name="Qwen/Qwen3.5-0.8B", device="cuda", dtype="bfloat16")
    controller = SessionController(
        session_id="gpu_session",
        runtime=runtime,
        engine=engine,
        decoder=HFDecoder(engine, max_tokens=8),
        config=SessionConfig(max_context_tokens=4096),
    )
    try:
        await controller.initialize("Write a list of numbers: 1, 2, 3,")
        assert engine.stats_.total_prefill_tokens > 0

        await controller.run_turn()

        assert engine.stats_.total_generated_tokens > 0
        assert controller.generated_buffer
    finally:
        await controller.close()
        await runtime.stop()
