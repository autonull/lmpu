"""Headless canonical demos with executable assertions.

Each demo returns a ``CanonicalResult`` instead of raising, so the CLI can run
all three and report a single pass/fail summary. The tests assert ``passed``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lmpu.host.engine import ScriptedEngine
from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.messages import SleepLevel

_RESULT_BYTES = 16 * 1024 * 1024
_MAX_INLINE = 1024
_CONTEXT_TOKEN_BOUND = 64


@dataclass(frozen=True, slots=True)
class CanonicalResult:
    """Outcome of one canonical demo."""

    name: str
    passed: bool
    details: dict[str, Any]


def _runtime(root: Path) -> Runtime:
    return Runtime(
        socket_path=str(root / "run.sock"),
        shm_dir=str(root / "shm"),
        config={"trace_dir": str(root / "traces")},
    )


async def _session(runtime: Runtime) -> str:
    hello = await runtime._hello({"protocol": "lmpu/1"})
    return hello.session_id


async def offload_recovery() -> CanonicalResult:
    """Crash a worker, then prove a fresh worker completes the next job."""
    with tempfile.TemporaryDirectory(prefix="lmpu-offload-") as temp:
        runtime = _runtime(Path(temp))
        await runtime.start()
        try:
            session_id = await _session(runtime)
            crashed = await runtime._submit(
                {
                    "processor": "python.vm",
                    "op": "run_program",
                    "inputs": {"program": "import os\nos._exit(7)"},
                    "config": {"mode": "sync", "timeout_ms": 20_000},
                },
                session_id,
            )
            recovered = await runtime._submit(
                {
                    "processor": "pure.func",
                    "op": "bigint_add",
                    "inputs": {"a": 40, "b": 2},
                    "config": {"mode": "sync"},
                },
                session_id,
            )
            workers_after = len(runtime.worker_pool._workers)
        finally:
            await runtime.stop()

    passed = crashed.error is not None and recovered.result == {"result": "42"}
    return CanonicalResult(
        name="offload-recovery",
        passed=passed,
        details={
            "crash_error": crashed.error,
            "recovered_result": recovered.result,
            "workers_after": workers_after,
        },
    )


async def sleep_proof() -> CanonicalResult:
    """Prove an S1 park freezes forwards and resume matches the baseline."""
    script = [{"type": "free", "text": "after wake"}]
    baseline = ScriptedEngine(list(script))
    baseline_state = await baseline.prefill("canonical prompt")
    expected = await baseline.decode(baseline_state, 4, [])

    engine = ScriptedEngine(list(script))
    state = await engine.prefill("canonical prompt")
    await engine.park(state, SleepLevel.S1)
    frozen = (
        engine.stats_.total_forward_passes,
        engine.stats_.total_generated_tokens,
        engine.position,
    )
    during_sleep = await engine.decode(state, 4, [])
    after_sleep = (
        engine.stats_.total_forward_passes,
        engine.stats_.total_generated_tokens,
        engine.position,
    )
    await engine.resume(state)
    resumed = await engine.decode(state, 4, [])

    passed = (
        during_sleep == ""
        and after_sleep == frozen
        and resumed == expected
        and engine.stats_.total_generated_tokens == baseline.stats_.total_generated_tokens
    )
    return CanonicalResult(
        name="sleep-proof",
        passed=passed,
        details={
            "during_sleep": during_sleep,
            "frozen": frozen,
            "after_sleep": after_sleep,
            "resumed": resumed,
        },
    )


async def context_protection() -> CanonicalResult:
    """Read a 16 MiB buffer without inlining its payload into context."""
    with tempfile.TemporaryDirectory(prefix="lmpu-context-") as temp:
        runtime = _runtime(Path(temp))
        await runtime.start()
        try:
            session_id = await _session(runtime)
            ref = await runtime._buffer_import(
                {
                    "name": "canonical-large",
                    "data": b"\x00" * _RESULT_BYTES,
                    "dtype": "uint8",
                    "shape": [_RESULT_BYTES],
                },
                session_id,
            )
            view = await runtime._read(
                {"buffer": ref.buffer_id, "max_inline_bytes": _MAX_INLINE},
                session_id,
            )
        finally:
            await runtime.stop()

    context_metadata = {
        "buffer": view.buffer,
        "dtype": view.dtype,
        "derived": view.data is None and view.derived_buffer is not None,
    }
    context_tokens = estimate_tokens(context_metadata)
    passed = (
        view.data is None
        and view.derived_buffer is not None
        and ref.bytes == _RESULT_BYTES
        and context_tokens <= _CONTEXT_TOKEN_BOUND
    )
    return CanonicalResult(
        name="context-protection",
        passed=passed,
        details={
            "result_bytes": ref.bytes,
            "derived": context_metadata["derived"],
            "context_tokens": context_tokens,
        },
    )


CANONICAL_DEMOS: dict[str, Callable[[], Awaitable[CanonicalResult]]] = {
    "offload-recovery": offload_recovery,
    "sleep-proof": sleep_proof,
    "context-protection": context_protection,
}


async def run_canonical(names: tuple[str, ...] = ()) -> list[CanonicalResult]:
    """Run the selected canonical demos, or all three when ``names`` is empty."""
    selected = names or tuple(CANONICAL_DEMOS)
    results: list[CanonicalResult] = []
    for name in selected:
        try:
            results.append(await CANONICAL_DEMOS[name]())
        except Exception as exc:
            results.append(CanonicalResult(name=name, passed=False, details={"error": str(exc)}))
    return results
