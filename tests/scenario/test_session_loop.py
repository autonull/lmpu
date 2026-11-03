"""Model-free scenario tests for the scripted session controller."""

from __future__ import annotations

import pytest

from lmpu.host.engine import ScriptedEngine
from lmpu.host.scripted import ScriptedDecoder, control_segment, free_segment
from lmpu.host.session import (
    ControllerState,
    InjectionPolicy,
    SessionConfig,
    SessionController,
)
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.messages import (
    EventKind,
    InjectedEvent,
    ModelJobStatus,
    SleepCondition,
    SleepLevel,
    SleepMessage,
)

pytestmark = pytest.mark.scenario


async def _make_controller(
    tmp_path,
    segments,
    config: SessionConfig | None = None,
):
    runtime = Runtime(
        socket_path=str(tmp_path / "run.sock"),
        shm_dir=str(tmp_path / "shm"),
        config={
            "trace_dir": str(tmp_path / "traces"),
            "reaper_interval_ms": 10,
        },
    )
    engine = ScriptedEngine([])
    decoder = ScriptedDecoder(segments)
    controller = SessionController(
        session_id="local_session",
        runtime=runtime,
        engine=engine,
        decoder=decoder,
        config=config,
    )
    await controller.initialize("scenario prompt")
    return controller, runtime, decoder, engine


async def _cleanup(controller: SessionController, runtime: Runtime) -> None:
    await controller.close()
    await runtime.stop()


async def test_initialize_adopts_runtime_session_and_builds_prompt(tmp_path) -> None:
    controller, runtime, _, _ = await _make_controller(tmp_path, [free_segment("x")])
    try:
        assert controller.session_id.startswith("s_")
        assert controller.session_id in runtime._sessions
        assert "Available processors" in controller.prompt
        assert "pure.func" in controller.prompt
        assert "<|proc|>" in controller.prompt
        assert controller.state == ControllerState.GENERATING
    finally:
        await _cleanup(controller, runtime)


async def test_job_completion_is_injected_on_next_turn(tmp_path) -> None:
    controller, runtime, decoder, _ = await _make_controller(
        tmp_path,
        [free_segment("planning next step")],
    )
    try:
        await controller.on_job_completed("job_alpha", ModelJobStatus.COMPLETED)
        assert len(controller._pending_injections) == 1

        assert await controller.run_turn() is True
        assert decoder.prompts == []
        assert "JobCompleted" in controller.generated_buffer
        assert "job_alpha" in controller.generated_buffer

        assert await controller.run_turn() is True
        assert len(decoder.prompts) == 1
        assert "JobCompleted" in decoder.prompts[0]
        assert "planning next step" in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


async def test_oversized_event_becomes_result_too_large(tmp_path) -> None:
    config = SessionConfig(
        max_injected_bytes_per_event=200,
        max_context_tokens=100_000,
    )
    controller, runtime, _, _ = await _make_controller(
        tmp_path,
        [free_segment("after truncation")],
        config=config,
    )
    try:
        controller.enqueue_event(
            InjectedEvent(
                version=1,
                kind="event",
                event=EventKind.JOB_COMPLETED,
                job_id="job_big",
                summary={"blob": "x" * 2000},
            )
        )
        assert await controller.run_turn() is True

        assert "ResultTooLarge" in controller.generated_buffer
        assert "event_exceeds_budget" in controller.generated_buffer
        assert "suggested_slice" in controller.generated_buffer
        assert "x" * 2000 not in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


async def test_parser_failure_injects_control_error_and_recovers(tmp_path) -> None:
    controller, runtime, decoder, _ = await _make_controller(
        tmp_path,
        [
            {"type": "malformed_control", "text": "{not-json"},
            free_segment("recovered"),
        ],
    )
    try:
        assert await controller.run_turn() is True

        assert "ControlError" in controller.generated_buffer
        assert "parser_failure" in controller.generated_buffer
        assert controller.state != ControllerState.CLOSED

        assert await controller.run_turn() is True
        assert len(decoder.prompts) == 2
        assert "recovered" in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


async def test_engine_error_closes_session(tmp_path) -> None:
    controller, runtime, _, _ = await _make_controller(
        tmp_path,
        [{"type": "error", "error": "backend disconnected"}],
    )
    try:
        assert await controller.run_turn() is False

        assert controller.state == ControllerState.CLOSED
        assert "ControlError" in controller.generated_buffer
        assert "engine_error" in controller.generated_buffer
        assert "backend disconnected" in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


async def test_parked_session_wakes_and_injects_woken_event(tmp_path) -> None:
    sleep = SleepMessage(
        until=SleepCondition(deadline_ms=60_000),
        level=SleepLevel.S1,
    )
    controller, runtime, _, engine = await _make_controller(
        tmp_path,
        [control_segment(sleep), free_segment("after wake")],
    )
    try:
        assert await controller.run_turn() is True
        assert controller.state == ControllerState.PARKED
        assert controller.host_state is not None
        assert controller.host_state.parked is True

        await controller.force_wake("forced_by_test")

        assert await controller.run_turn() is True
        assert controller.state == ControllerState.GENERATING
        assert controller.host_state.parked is False
        assert any(event.event == EventKind.WOKEN for event in controller._pending_injections)

        assert await controller.run_turn() is True
        assert "Woken" in controller.generated_buffer
        assert "forced_by_test" in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


async def test_context_budget_exhaustion_is_reported_not_truncated(tmp_path) -> None:
    config = SessionConfig(max_context_tokens=1)
    controller, runtime, _, _ = await _make_controller(
        tmp_path,
        [free_segment("never generated")],
        config=config,
    )
    try:
        assert await controller.run_turn() is False
        assert controller.state == ControllerState.CLOSED
        assert "ResultTooLarge" in controller.generated_buffer
        assert "context_budget_exhausted" in controller.generated_buffer
    finally:
        await _cleanup(controller, runtime)


def test_invalid_injection_policy_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="Invalid injection action"):
        SessionConfig(
            injection_policy=InjectionPolicy(
                generating={"JobCompleted": "explode"},
            )
        )


def test_generating_and_parked_policies_use_safe_defaults() -> None:
    config = SessionConfig()

    assert config.injection_policy.generating["JobCompleted"] == "next_turn"
    assert config.injection_policy.generating["JobFailed"] == "interrupt"
    assert config.injection_policy.parked["JobCompleted"] == "wake"
    assert config.injection_policy.generating["Progress"] == "observe"


async def test_s1_park_freezes_forwards_and_resume_matches_baseline() -> None:
    script = [{"type": "free", "text": "after wake"}]
    baseline = ScriptedEngine(list(script))
    baseline_state = await baseline.prefill("prompt")
    expected = await baseline.decode(baseline_state, 4, [])

    engine = ScriptedEngine(list(script))
    state = await engine.prefill("prompt")
    await engine.park(state, SleepLevel.S1)

    frozen = (
        engine.stats_.total_forward_passes,
        engine.stats_.total_generated_tokens,
        engine.position,
    )
    assert await engine.decode(state, 4, []) == ""
    assert (
        engine.stats_.total_forward_passes,
        engine.stats_.total_generated_tokens,
        engine.position,
    ) == frozen

    await engine.resume(state)
    assert await engine.decode(state, 4, []) == expected
    assert engine.stats_.total_generated_tokens == baseline.stats_.total_generated_tokens
