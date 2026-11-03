"""Fair baseline adapters for the benchmark harness.

Each adapter emits the canonical host metrics schema. Model baselines use a
``HostModel``; the default implementation is the reference HF engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from lmpu.host.engine import HostStats
from lmpu.host.metrics import EnergySampler, HostMetrics, estimate_tokens
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.messages import ModelJobStatus

from .config import BenchmarkConfig
from .energy import EnergyProbe, IdleBaseline
from .runner import TrialContext, TrialRecord
from .workloads import WorkloadSpec

logger = logging.getLogger(__name__)

BASELINES = (
    "lmpu.sync",
    "lmpu.sleep",
    "lmpu.slice",
    "lmpu.individual",
    "lmpu.batch",
    "native",
    "direct_cot",
    "trace_token",
    "sync_tool",
    "async_poll",
)
MODEL_BASELINES = frozenset({"direct_cot", "sync_tool", "async_poll"})
TERMINAL_STATUSES = frozenset(
    {ModelJobStatus.COMPLETED, ModelJobStatus.FAILED, ModelJobStatus.CANCELLED}
)


def resolve_baselines(selection: str | tuple[str, ...]) -> tuple[str, ...]:
    """Validate and normalize baseline names."""
    if isinstance(selection, str):
        names = BASELINES if selection == "all" else (selection,)
    else:
        names = BASELINES if "all" in selection else tuple(selection)
    unknown = set(names) - set(BASELINES)
    if unknown:
        raise ValueError(f"Unknown baseline: {sorted(unknown)[0]}")
    return names


class HostModel(Protocol):
    """Minimal host-model surface used by the model baselines."""

    async def start_trial(self) -> None: ...

    async def decode(self, prompt: str, max_tokens: int, stop: str = "") -> str: ...

    async def count_tokens(self, text: str) -> int: ...

    def take_energy(self) -> float: ...

    async def close(self) -> None: ...


class HFHost:
    """Reference HF engine adapter with a fresh decoder state per trial."""

    def __init__(self, config: BenchmarkConfig):
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        for noisy in (
            "huggingface_hub",
            "transformers",
            "httpx",
            "httpx2",
            "httpcore",
            "urllib3",
            "filelock",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        from lmpu.host.hf import HFDecoder, HFEngine

        self.config = config
        self.engine = HFEngine(model_name=config.model, device=config.device)
        self.decoder = HFDecoder(self.engine, max_tokens=config.max_tokens)
        self._energy = EnergySampler()
        self._trial_energy = 0.0

    async def start_trial(self) -> None:
        self.engine.stats_ = HostStats()
        self.decoder.reset()
        self._energy.sample()
        self._trial_energy = 0.0

    async def decode(self, prompt: str, max_tokens: int, stop: str = "") -> str:
        text = await self.decoder.decode_free(prompt, max_tokens, stop)
        self._trial_energy += self._energy.sample().gpu_joules
        return text

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        await self.engine._initialize()
        tokenizer = self.engine._tokenizer
        if tokenizer is None:
            return estimate_tokens(text)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def take_energy(self) -> float:
        energy = self._trial_energy
        self._trial_energy = 0.0
        return energy

    async def close(self) -> None:
        self._energy.close()
        await self.engine.close()


def _program_from_text(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or "op" not in payload or "inputs" not in payload:
        return None
    return payload


def _request(
    program: dict[str, Any],
    workload: WorkloadSpec,
    job_id: str,
    mode: str | None,
) -> dict[str, Any]:
    raw_inputs = program.get("inputs")
    if raw_inputs == "$INPUTS":
        inputs = dict(workload.inputs)
    elif isinstance(raw_inputs, dict):
        inputs = {
            key: (workload.inputs[key] if value == "$CONTEXT" and key in workload.inputs else value)
            for key, value in raw_inputs.items()
        }
    else:
        inputs = raw_inputs

    config: dict[str, Any] = {"timeout_ms": 60_000, "bench_nonce": job_id}
    config.update(workload.config)
    if mode is not None:
        config["mode"] = mode
    request = {
        "job_id": job_id,
        "processor": program.get("processor", workload.processor),
        "op": program["op"],
        "inputs": inputs,
        "config": config,
    }
    if isinstance(program.get("batch"), dict):
        request["batch"] = program["batch"]
    return request


def _result_text(result: Any) -> str:
    return json.dumps(result, separators=(",", ":"), default=str)


def _summary_text(result: Any, max_chars: int = 200) -> str:
    text = _result_text(result)
    return text if len(text) <= max_chars else text[:max_chars] + "…"


async def _record_prefill(metrics: HostMetrics, host: HostModel, text: str) -> int:
    tokens = await host.count_tokens(text)
    if tokens:
        metrics.record_forward("prefill", tokens)
    return tokens


async def _record_generated(metrics: HostMetrics, host: HostModel, text: str) -> int:
    tokens = await host.count_tokens(text)
    for _ in range(tokens):
        metrics.record_forward("generated", 1)
    return tokens


async def _record_injected(metrics: HostMetrics, host: HostModel, text: str) -> int:
    tokens = await host.count_tokens(text)
    if tokens:
        metrics.record_forward("injected", tokens)
    return tokens


def _payload(metrics: HostMetrics, wall: float) -> dict[str, Any]:
    payload = metrics.to_dict()
    payload["wall_time_seconds"] = wall
    return payload


class BaselineRunner:
    """Run one of the registered baselines for each trial."""

    def __init__(self, config: BenchmarkConfig, host: HostModel | None = None):
        self.config = config
        self._host = host
        self._energy = EnergyProbe()
        self.runtime: Runtime | None = None
        self._runtime_dir: Path | None = None

    async def setup(self, run_dir: Path) -> None:
        # Unix socket paths are length-limited; keep transient runtime files
        # out of the (possibly deep) artifact directory.
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="lmpu-bench-"))
        self.runtime = Runtime(
            socket_path=str(self._runtime_dir / "run.sock"),
            shm_dir=str(self._runtime_dir / "shm"),
            config={
                "trace_dir": str(run_dir / "traces"),
                "batch_window_ms": 1,
                "reaper_interval_ms": 100,
            },
        )
        await self.runtime.start()
        if self._host is None and any(
            baseline in MODEL_BASELINES for baseline in self.config.baselines
        ):
            self._host = HFHost(self.config)

    async def run(self, context: TrialContext) -> TrialRecord:
        handlers = {
            "lmpu.sync": self._run_lmpu_sync,
            "lmpu.sleep": self._run_lmpu_sleep,
            "lmpu.slice": self._run_lmpu_slice,
            "lmpu.individual": self._run_lmpu_individual,
            "lmpu.batch": self._run_lmpu_batch,
            "native": self._run_native,
            "direct_cot": self._run_direct_cot,
            "trace_token": self._run_trace_token,
            "sync_tool": self._run_sync_tool,
            "async_poll": self._run_async_poll,
        }
        handler = handlers.get(context.baseline)
        if handler is None:
            raise ValueError(f"Unknown baseline: {context.baseline}")

        idle = await self._energy.measure_idle(self.config.idle_seconds)
        started_at = time.time()
        started = time.perf_counter()
        try:
            record = await handler(context, started_at)
        except Exception as exc:
            wall = time.perf_counter() - started
            logger.warning(
                "baseline failed: %s",
                exc,
                extra={"baseline": context.baseline, "trial": context.trial},
            )
            metrics = HostMetrics(session_id=f"bench_{context.baseline}_{context.trial}")
            record = TrialRecord(
                workload=context.workload.name,
                baseline=context.baseline,
                trial=context.trial,
                seed=context.seed,
                job_id="",
                started_at=started_at,
                wall_time_seconds=wall,
                correct=False,
                metrics=_payload(metrics, wall),
                error=str(exc),
            )
        return self._with_idle(record, idle)

    def _with_idle(self, record: TrialRecord, idle: IdleBaseline) -> TrialRecord:
        metadata = dict(record.metadata)
        metadata["idle"] = idle.to_dict()
        measured = float(record.metrics.get("gpu_energy_joules", 0.0))
        if measured > 0 and idle.average_watts is not None:
            idle_joules = idle.average_watts * record.wall_time_seconds
            metadata["energy"] = {
                "measured_joules": measured,
                "idle_joules": idle_joules,
                "corrected_joules": max(0.0, measured - idle_joules),
            }
        return replace(record, metadata=metadata)

    def _host_model(self) -> HostModel:
        if self._host is None:
            raise RuntimeError("host model is not configured")
        return self._host

    async def _open_session(self) -> str:
        runtime = self._require_runtime()
        hello = await runtime._hello({"protocol": "lmpu/1"})
        return hello.session_id

    def _require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise RuntimeError("runtime is not set up")
        return self.runtime

    async def _close_session(self, session_id: str) -> None:
        try:
            await self._require_runtime()._close({"detach": False}, session_id)
        except Exception:
            logger.exception("benchmark session close failed", extra={"session": session_id})

    def _job_id(self, context: TrialContext, suffix: str = "") -> str:
        return (
            f"bench_{context.seed}_{context.baseline}_"
            f"{context.workload.name}_{context.trial}{suffix}"
        )

    async def _run_lmpu_sync(self, context: TrialContext, started_at: float) -> TrialRecord:
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        job_id = self._job_id(context)
        request = {
            "job_id": job_id,
            "processor": context.workload.processor,
            "op": context.workload.op,
            "inputs": context.workload.inputs,
            "config": {
                "mode": "sync",
                "timeout_ms": 60_000,
                "bench_nonce": job_id,
            }
            | context.workload.config,
        }
        metrics.record_control_tokens(estimate_tokens(request))
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            submitted = await runtime._submit(request, session_id)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        if result is not None:
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
        )

    async def _run_lmpu_sleep(self, context: TrialContext, started_at: float) -> TrialRecord:
        """Submit async, park in S1 until the job completes, then wake."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        job_id = self._job_id(context)
        request = {
            "job_id": job_id,
            "processor": context.workload.processor,
            "op": context.workload.op,
            "inputs": context.workload.inputs,
            "config": {
                "mode": "async",
                "timeout_ms": 60_000,
                "bench_nonce": job_id,
            }
            | context.workload.config,
        }
        metrics.record_control_tokens(estimate_tokens(request))
        timeout = context.workload.config.get("timeout_ms", 60_000) / 1000
        started = time.perf_counter()
        error: str | None = None
        finished = None
        sleep_seconds = 0.0
        sleep_id = ""
        try:
            await runtime._submit(request, session_id)
            sleep_result = await runtime._sleep(
                {"until": {"all": [job_id]}, "level": "S1"},
                session_id,
            )
            sleep_id = sleep_result.sleep_id
            park_started = time.perf_counter()
            finished = await runtime.job_table.wait_terminal(job_id, timeout=timeout)
            sleep_seconds = time.perf_counter() - park_started
            metrics.sleep_duration_seconds = sleep_seconds
            await runtime._wake({}, session_id)
            if finished is None:
                raise TimeoutError("sleep treatment exceeded the job timeout")
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        result = finished.result if finished is not None else None
        if finished is not None and finished.error:
            error = json.dumps(finished.error, default=str)
        if result is not None:
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "park": {
                    "level": "S1",
                    "sleep_id": sleep_id,
                    "sleep_seconds": sleep_seconds,
                    "forward_passes_during_park": 0,
                }
            },
        )

    async def _run_lmpu_slice(self, context: TrialContext, started_at: float) -> TrialRecord:
        """Import a large buffer and read it through the bounded-slice path."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        size = int(context.workload.config.get("result_bytes", 16 * 1024 * 1024))
        max_inline = int(context.workload.config.get("max_inline_bytes", 1024))
        started = time.perf_counter()
        error: str | None = None
        derived = False
        try:
            ref = await runtime._buffer_import(
                {
                    "name": "large_result",
                    "data": b"\x00" * size,
                    "dtype": "uint8",
                    "shape": [size],
                },
                session_id,
            )
            read_request = {"buffer": ref.buffer_id, "max_inline_bytes": max_inline}
            metrics.record_control_tokens(estimate_tokens(read_request))
            view = await runtime._read(read_request, session_id)
            derived = view.data is None and view.derived_buffer is not None
            summary = {
                "buffer": view.buffer,
                "dtype": view.dtype,
                "derived": derived,
                "note": view.note,
            }
            metrics.record_injected_tokens(estimate_tokens(summary))
        except Exception as exc:
            error = str(exc)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id="",
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and derived,
            metrics=_payload(metrics, wall),
            error=error,
            metadata={"result_bytes": size, "derived": derived, "max_inline_bytes": max_inline},
        )

    async def _run_lmpu_individual(self, context: TrialContext, started_at: float) -> TrialRecord:
        """Submit each batch item as its own synchronous job."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        items = context.workload.inputs.get("items", [])
        started = time.perf_counter()
        results: list[Any] = []
        errors: list[str] = []
        try:
            for index, item in enumerate(items):
                job_id = self._job_id(context, f"_{index}")
                request = {
                    "job_id": job_id,
                    "processor": context.workload.processor,
                    "op": item.get("op", context.workload.op),
                    "inputs": item.get("inputs", {}),
                    "config": {
                        "mode": "sync",
                        "timeout_ms": 60_000,
                        "bench_nonce": job_id,
                    },
                }
                metrics.record_control_tokens(estimate_tokens(request))
                try:
                    submitted = await runtime._submit(request, session_id)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if submitted.error:
                    errors.append(json.dumps(submitted.error, default=str))
                results.append(submitted.result)
            metrics.record_injected_tokens(
                estimate_tokens(_summary_text({"results": results, "errors": errors}))
            )
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        combined = {"results": results, "errors": errors}
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id="",
            started_at=started_at,
            wall_time_seconds=wall,
            correct=not errors and context.workload.verify(combined),
            metrics=_payload(metrics, wall),
            error=errors[0] if errors else None,
            metadata={"submissions": len(results), "errors": errors},
        )

    async def _run_lmpu_batch(self, context: TrialContext, started_at: float) -> TrialRecord:
        """Submit all batch items in one job."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        job_id = self._job_id(context)
        request = _request(context.workload.program, context.workload, job_id, "sync")
        metrics.record_control_tokens(estimate_tokens(request))
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            submitted = await runtime._submit(request, session_id)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        if result is not None:
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={"batched": True},
        )

    async def _run_native(self, context: TrialContext, started_at: float) -> TrialRecord:
        """Run the workload computation in-process without the runtime."""
        native = context.workload.native
        if native is None:
            raise ValueError(f"Workload {context.workload.name} has no native baseline")
        metrics = HostMetrics(session_id=f"bench_native_{context.trial}")
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            result = await asyncio.to_thread(native)
        except Exception as exc:
            error = str(exc)
        wall = time.perf_counter() - started
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id="",
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={"native": True},
        )

    async def _run_direct_cot(self, context: TrialContext, started_at: float) -> TrialRecord:
        host = self._host_model()
        session_id = f"bench_direct_cot_{context.trial}"
        metrics = HostMetrics(session_id=session_id)
        started = time.perf_counter()
        error: str | None = None
        answer: str | None = None
        response = ""
        try:
            await host.start_trial()
            await _record_prefill(metrics, host, context.workload.prompt)
            response = await host.decode(context.workload.prompt, self.config.max_tokens, "")
            await _record_generated(metrics, host, response)
            metrics.record_energy(gpu_joules=host.take_energy())
            answer = context.workload.parse_answer(response)
        except Exception as exc:
            error = str(exc)
        wall = time.perf_counter() - started
        correct = error is None and answer is not None and context.workload.verify_answer(answer)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id="",
            started_at=started_at,
            wall_time_seconds=wall,
            correct=correct,
            metrics=_payload(metrics, wall),
            error=error,
            metadata={"answer": answer, "response": response},
        )

    async def _run_trace_token(self, context: TrialContext, started_at: float) -> TrialRecord:
        session_id = f"bench_trace_token_{context.trial}"
        metrics = HostMetrics(session_id=session_id)
        started = time.perf_counter()
        trace = context.workload.trace()
        tokens = estimate_tokens(trace)
        for _ in range(tokens):
            metrics.record_forward("generated", 1)
        wall = time.perf_counter() - started
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id="",
            started_at=started_at,
            wall_time_seconds=wall,
            correct=True,
            metrics=_payload(metrics, wall),
            metadata={"simulated": True, "trace_tokens": tokens},
        )

    async def _model_program(
        self, context: TrialContext, metrics: HostMetrics
    ) -> tuple[str, dict[str, Any], str]:
        host = self._host_model()
        await host.start_trial()
        prompt = context.workload.program_prompt
        await _record_prefill(metrics, host, prompt)
        text = await host.decode(prompt, self.config.max_tokens, "")
        await _record_generated(metrics, host, text)
        program = _program_from_text(text)
        if program is None:
            raise ValueError("model did not emit a valid tool program")
        return prompt + text, program, text

    async def _run_sync_tool(self, context: TrialContext, started_at: float) -> TrialRecord:
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        try:
            prefix, program, _ = await self._model_program(context, metrics)
            request = _request(program, context.workload, self._job_id(context), "sync")
            metrics.record_control_tokens(estimate_tokens(request))
            submitted = await runtime._submit(request, session_id)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
            if result is not None:
                result_text = _result_text(result)
                followup = (
                    prefix
                    + "\nTOOL RESULT:\n"
                    + result_text
                    + '\nReply with a final line "ANSWER: <integer>".'
                )
                await _record_injected(metrics, host, followup[len(prefix) :])
                response = await host.decode(followup, self.config.max_tokens, "")
                await _record_generated(metrics, host, response)
                metrics.record_energy(gpu_joules=host.take_energy())
                final_answer = context.workload.parse_answer(response)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=self._job_id(context),
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={"program": program, "answer": final_answer},
        )

    async def _run_async_poll(self, context: TrialContext, started_at: float) -> TrialRecord:
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        polls = 0
        try:
            _, program, _ = await self._model_program(context, metrics)
            metrics.record_energy(gpu_joules=host.take_energy())
            request = _request(program, context.workload, job_id, None)
            metrics.record_control_tokens(estimate_tokens(request))
            submitted = await runtime._submit(request, session_id)
            deadline = time.monotonic() + 60.0
            while True:
                poll = await runtime._poll({"job_id": job_id}, session_id)
                polls += 1
                status_text = json.dumps(
                    {
                        "status": poll.status.value,
                        "progress": poll.progress,
                        "elapsed_ms": poll.elapsed_ms,
                    },
                    separators=(",", ":"),
                )
                metrics.record_control_tokens(await host.count_tokens(status_text))
                if poll.status in TERMINAL_STATUSES:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("async polling exceeded 60 seconds")
                await asyncio.sleep(self.config.poll_interval_ms / 1000)

            waited = await runtime._wait({"job_id": job_id, "timeout_ms": 1000}, session_id)
            result = waited.result
            if waited.error:
                error = json.dumps(waited.error, default=str)
            if result is not None:
                summary = _summary_text(result)
                await _record_injected(metrics, host, summary)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=error is None and context.workload.verify(result),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "polls": polls,
                "submitted_status": submitted.status.value,
            },
        )

    async def teardown(self) -> None:
        if self._host is not None:
            await self._host.close()
            self._host = None
        self._energy.close()
        if self.runtime is not None:
            await self.runtime.stop()
            self.runtime = None
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
            self._runtime_dir = None
