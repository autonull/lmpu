"""New pure-function operations used by the initial benchmark workloads."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_matmul_and_text_stats_through_runtime(runtime) -> None:
    hello = await runtime._hello({"protocol": "lmpu/1"})
    session_id = hello.session_id

    matmul = await runtime._submit(
        {
            "processor": "pure.func",
            "op": "matmul",
            "inputs": {"size": 32, "seed": 1},
            "config": {"mode": "sync", "timeout_ms": 30_000},
        },
        session_id,
    )
    assert matmul.result is not None
    assert matmul.result["shape"] == [32, 32]
    assert len(matmul.result["checksum"]) == 16

    text = "alpha beta\nSECRET: LMPU-TEST-CODE\ngamma"
    stats = await runtime._submit(
        {
            "processor": "pure.func",
            "op": "text_stats",
            "inputs": {"text": text},
            "config": {"mode": "sync", "timeout_ms": 30_000},
        },
        session_id,
    )
    assert stats.result is not None
    assert stats.result["chars"] == len(text)
    assert stats.result["words"] == len(text.split())
    assert stats.result["secret"] == "LMPU-TEST-CODE"

    await runtime._close({"detach": False}, session_id)
