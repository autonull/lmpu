"""Opt-in reference-model smoke test. Enable with LMPU_RUN_GPU=1."""

from __future__ import annotations

import os

import pytest

from lmpu.host.hf import HFEngine

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
