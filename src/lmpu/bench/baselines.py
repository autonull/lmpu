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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from lmpu.host.engine import HostStats
from lmpu.host.metrics import EnergySampler, HostMetrics, estimate_tokens
from lmpu.kernel.runtime import Runtime
from lmpu.protocol.messages import ModelJobStatus

from .config import BenchmarkConfig
from .energy import EnergyProbe, IdleBaseline
from .runner import TrialContext, TrialRecord
from .taxonomy import (
    ALL_BASELINES,
    END_TO_END_BASELINES,
    split_oracle,
)
from .workloads import WorkloadSpec

logger = logging.getLogger(__name__)

BASELINES = ALL_BASELINES
MODEL_BASELINES = frozenset(END_TO_END_BASELINES)
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

    async def decode(
        self, prompt: str, max_tokens: int, stop: str = "", session: int = 0
    ) -> str: ...

    async def count_tokens(self, text: str) -> int: ...

    def take_energy(self) -> float: ...

    async def close(self) -> None: ...


class HFHost:
    """Reference HF engine adapter with a fresh decoder state per trial."""

    def __init__(self, config: BenchmarkConfig, sessions: int = 1, use_constrained: bool = False):
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
        self._decoders = [
            HFDecoder(self.engine, max_tokens=config.max_tokens, use_constrained=use_constrained)
            for _ in range(sessions)
        ]
        self.decoder = self._decoders[0]
        self._energy = EnergySampler()
        self._trial_energy = 0.0

    async def start_trial(self) -> None:
        self.engine.stats_ = HostStats()
        for decoder in self._decoders:
            decoder.reset()
        self._energy.sample()
        self._trial_energy = 0.0

    async def decode(self, prompt: str, max_tokens: int, stop: str = "", session: int = 0) -> str:
        text = await self._decoders[session].decode_free(prompt, max_tokens, stop)
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
        # The model outputs JSON with unescaped content in the "program" field.
        # Try to extract the program code directly from the response.
        program_code = _extract_program_code(candidate)
        if program_code:
            return {
                "processor": "python.vm",
                "op": "run_program",
                "inputs": {"program": program_code},
            }
        # Fallback: try regex matches
        for match in re.finditer(r"\{.*\}", candidate, re.DOTALL):
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict) and "op" in payload and "inputs" in payload:
                    return payload
            except json.JSONDecodeError:
                continue
        return None
    if not isinstance(payload, dict) or "op" not in payload or "inputs" not in payload:
        return None
    return payload


def _schema_for_workload(processor: str, op: str) -> dict[str, Any]:
    """Generate JSON schema for constrained decoding based on workload processor/op."""
    if processor == "pure.func":
        # Define input schemas for each pure.func op
        # Use loose schemas for constrained decoding speed; exact values validated later
        input_schemas = {
            "bigint_mul": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
            "bigint_add": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
            "matmul": {
                "type": "object",
                "properties": {"size": {"type": "integer"}, "seed": {"type": "integer"}},
                "required": ["size", "seed"],
            },
            "text_stats": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "batch": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string"},
                                "inputs": {"type": "object"},
                            },
                            "required": ["op", "inputs"],
                        },
                    }
                },
                "required": ["items"],
            },
        }
        inputs_schema = input_schemas.get(op, {"type": "object"})
        return {
            "type": "object",
            "properties": {
                "processor": {"type": "string", "enum": [processor]},
                "op": {"type": "string", "enum": [op]},
                "inputs": inputs_schema,
            },
            "required": ["processor", "op", "inputs"],
        }
    elif processor == "python.vm":
        return {
            "type": "object",
            "properties": {
                "processor": {"type": "string", "enum": ["python.vm"]},
                "op": {"type": "string", "enum": ["run_program"]},
                "inputs": {
                    "type": "object",
                    "properties": {"program": {"type": "string"}},
                    "required": ["program"],
                },
            },
            "required": ["processor", "op", "inputs"],
        }
    else:
        # Fallback: allow any processor/op
        return {
            "type": "object",
            "properties": {
                "processor": {"type": "string"},
                "op": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": ["processor", "op", "inputs"],
        }


def _extract_program_code(text: str) -> str | None:
    """Extract Python program code from model response that has malformed JSON."""
    # The model outputs something like:
    # {"program": "import hashlib\ntime.sleep(60)\nunit = b'...\\n'"}
    # with a mix of literal newlines (between statements) and escaped \\n (inside strings).
    # Strategy: find the program field, extract content, then normalize escapes.

    # First try: find the complete JSON object first
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            candidate = match.group(0)
            # Fix common JSON issues: unescaped newlines in string values
            fixed = _fix_json_string_escapes(candidate)
            payload = json.loads(fixed)
            if isinstance(payload, dict) and payload.get("op") == "run_program":
                program = payload.get("inputs", {}).get("program")
                if isinstance(program, str) and program.strip():
                    return program
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fallback: regex extraction with smart escape handling
    match = re.search(r'"program"\s*:\s*"(.+?)"\s*\}\s*\}', text, re.DOTALL)
    if match:
        raw = match.group(1)
        return _normalize_program_escapes(raw)

    # Fallback: more permissive regex
    match = re.search(r'"program"\s*:\s*"(.+)"\s*\}\s*\}', text, re.DOTALL)
    if match:
        raw = match.group(1)
        return _normalize_program_escapes(raw)

    # Fallback: code blocks
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _fix_json_string_escapes(text: str) -> str:
    """Fix invalid JSON by escaping unescaped newlines/tabs in string values."""
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        ch = text[i]
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"' and not escape_next:
            result.append(ch)
            in_string = not in_string
        elif ch in "\n\r\t" and in_string:
            # Escape unescaped control chars inside strings
            if ch == "\n":
                result.append("\\n")
            elif ch == "\r":
                result.append("\\r")
            elif ch == "\t":
                result.append("\\t")
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def _normalize_program_escapes(raw: str) -> str:
    """Normalize escape sequences in extracted program code.

    The raw capture may contain:
    - Literal newlines between statements (keep as literal newlines)
    - Already-escaped \\n inside string literals (convert to literal newlines)
    - Double-escaped \\\\n (convert to literal newlines)

    Returns a string with ACTUAL newlines for Python execution.
    """
    # First, handle double-escaping: \\\\n -> \n (literal newline)
    raw = raw.replace("\\\\n", "\n").replace("\\\\t", "\t").replace("\\\\r", "\r")
    # Then, handle single-escaping: \\n -> \n (literal newline)
    raw = raw.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    # Keep literal newlines as-is
    return raw


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


def _tool_result_suffix(workload: WorkloadSpec, body: str) -> str:
    return f'\nTOOL RESULT:\n{body}\nReply with a final line "{workload.answer_line}".'


def score_answer(workload: WorkloadSpec, answer: str | None, error: str | None) -> bool:
    """Score a model baseline on its final answer (V0.1).

    A correct tool result with a wrong or missing final answer is incorrect.
    """
    return error is None and answer is not None and workload.verify_answer(answer)


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


@dataclass
class PhaseTimer:
    """Accumulate per-phase wall time (V1.4)."""

    phases: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    def as_dict(self) -> dict[str, float]:
        return dict(self.phases)


def _failure_record(
    context: TrialContext,
    started_at: float,
    started: float,
    metrics: HostMetrics,
    exc: Exception,
    metadata: dict[str, Any] | None = None,
) -> TrialRecord:
    """Build a failed trial record that retains accumulated token accounting."""
    wall = time.perf_counter() - started
    merged = {"failed": True, **(metadata or {})}
    return TrialRecord(
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
        metadata=merged,
    )


def _job_context(job_id: str) -> Any:
    from lmpu.processors.base import JobContext

    return JobContext(
        job_id=job_id,
        session_id="app_agent",
        buffers={},
        send_progress=lambda *_: None,
        send_needs_input=lambda *_: None,
        submit_child=lambda *_: None,
        log=lambda *_, **__: None,
        trace=lambda *_, **__: None,
        cancelled=lambda: False,
    )


async def _execute_in_process(program: dict[str, Any], workload: WorkloadSpec, job_id: str) -> Any:
    """Execute a model-chosen program without the LMPU runtime (app-level path)."""
    from lmpu.processors.base import ProcessorContext, Result
    from lmpu.processors.builtin.pure_function import PureFunctionProcessor
    from lmpu.processors.builtin.python_vm import PythonVMProcessor

    processor_name = program.get("processor", workload.processor)
    job = _request(program, workload, job_id, None)
    if processor_name == "pure.func":
        processor: Any = PureFunctionProcessor()
    elif processor_name == "python.vm":
        processor = PythonVMProcessor()
    else:
        raise ValueError(f"app_agent cannot execute processor: {processor_name}")
    await processor.setup(ProcessorContext({}))
    result: Result = await processor.execute(job, _job_context(job_id))
    if not result.success:
        raise ValueError(result.error_message or "in-process execution failed")
    return result.data


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
            sessions = (
                2
                if any(self._is_two_session(baseline) for baseline in self.config.baselines)
                else 1
            )
            # Enable constrained decoding for model planner arms
            self._host = HFHost(self.config, sessions=sessions, use_constrained=True)

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
            "sync_tool_full_result": self._run_sync_tool_full_result,
            "async_poll_programmatic": self._run_async_poll_programmatic,
            "async_poll_model": self._run_async_poll_model,
            "session_reopen": self._run_session_reopen,
            "app_agent": self._run_app_agent,
            "engine_native": self._run_engine_native,
            "lmpu.model_async": self._run_model_async,
            "lmpu.model_sync": self._run_model_sync,
            "lmpu.model_async_2session": self._run_model_async_2session,
            "app_agent_2session": self._run_app_agent_2session,
        }
        for name in END_TO_END_BASELINES:
            stem, oracle = split_oracle(name)
            if oracle and stem in handlers:
                handlers[name] = handlers[stem]
        handler = handlers.get(context.baseline)
        if handler is None:
            raise ValueError(f"Unknown baseline: {context.baseline}")

        idle = await self._energy.measure_idle(self.config.idle_seconds)
        started_at = time.time()
        started = time.perf_counter()
        holder: dict[str, Any] = {}
        try:
            record = await handler(context, started_at, holder)
        except Exception as exc:
            logger.warning(
                "baseline failed: %s",
                exc,
                extra={"baseline": context.baseline, "trial": context.trial},
            )
            metrics = holder.get("metrics")
            if metrics is None:
                metrics = HostMetrics(session_id=f"bench_{context.baseline}_{context.trial}")
            if self._host is not None:
                try:
                    metrics.record_energy(gpu_joules=self._host.take_energy())
                except Exception:
                    logger.exception("host energy collection failed on the failure path")
            record = _failure_record(
                context,
                started_at,
                started,
                metrics,
                exc,
                {"phases": (holder.get("phases") or {})},
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

    @staticmethod
    def _is_two_session(baseline: str) -> bool:
        stem, _ = split_oracle(baseline)
        return stem in ("lmpu.model_async_2session", "app_agent_2session")

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

    async def _run_lmpu_sync(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
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
        # Energy sampling for mechanism baselines
        self._energy.sample()
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            submit_started = time.perf_counter()
            submitted = await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        # Record energy after trial
        gpu_joules = self._energy.sample().gpu_joules
        metrics.record_energy(gpu_joules=gpu_joules)
        if result is not None:
            inject_started = time.perf_counter()
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
            phases.record("ingest", time.perf_counter() - inject_started)
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
            metadata={"phases": phases.as_dict()},
        )

    async def _run_lmpu_sleep(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Submit async, park in S1 until the job completes, then wake."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
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
        # Energy sampling
        self._energy.sample()
        started = time.perf_counter()
        error: str | None = None
        finished = None
        sleep_seconds = 0.0
        sleep_id = ""
        try:
            submit_started = time.perf_counter()
            await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            sleep_result = await runtime._sleep(
                {"until": {"all": [job_id]}, "level": "S1"},
                session_id,
            )
            sleep_id = sleep_result.sleep_id
            park_started = time.perf_counter()
            finished = await runtime.job_table.wait_terminal(job_id, timeout=timeout)
            sleep_seconds = time.perf_counter() - park_started
            metrics.sleep_duration_seconds = sleep_seconds
            phases.record("park", sleep_seconds)
            wake_started = time.perf_counter()
            await runtime._wake({}, session_id)
            metrics.record_wake_latency(time.perf_counter() - wake_started)
            phases.record("wake", metrics.wake_latency_seconds)
            if finished is None:
                raise TimeoutError("sleep treatment exceeded the job timeout")
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        # Record energy after trial
        gpu_joules = self._energy.sample().gpu_joules
        metrics.record_energy(gpu_joules=gpu_joules)
        result = finished.result if finished is not None else None
        if finished is not None and finished.error:
            error = json.dumps(finished.error, default=str)
        if result is not None:
            inject_started = time.perf_counter()
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
            phases.record("ingest", time.perf_counter() - inject_started)
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
                },
                "phases": phases.as_dict(),
            },
        )

    async def _run_lmpu_slice(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Import a large buffer and read it through the bounded-slice path."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        size = int(context.workload.config.get("result_bytes", 16 * 1024 * 1024))
        max_inline = int(context.workload.config.get("max_inline_bytes", 1024))
        # Energy sampling
        self._energy.sample()
        started = time.perf_counter()
        error: str | None = None
        derived = False
        try:
            import_started = time.perf_counter()
            ref = await runtime._buffer_import(
                {
                    "name": "large_result",
                    "data": b"\x00" * size,
                    "dtype": "uint8",
                    "shape": [size],
                },
                session_id,
            )
            phases.record("import", time.perf_counter() - import_started)
            read_request = {"buffer": ref.buffer_id, "max_inline_bytes": max_inline}
            metrics.record_control_tokens(estimate_tokens(read_request))
            read_started = time.perf_counter()
            view = await runtime._read(read_request, session_id)
            derived = view.data is None and view.derived_buffer is not None
            summary = {
                "buffer": view.buffer,
                "dtype": view.dtype,
                "derived": derived,
                "note": view.note,
            }
            metrics.record_injected_tokens(estimate_tokens(summary))
            phases.record("ingest", time.perf_counter() - read_started)
        except Exception as exc:
            error = str(exc)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        # Record energy after trial
        gpu_joules = self._energy.sample().gpu_joules
        metrics.record_energy(gpu_joules=gpu_joules)
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
            metadata={
                "result_bytes": size,
                "derived": derived,
                "max_inline_bytes": max_inline,
                "phases": phases.as_dict(),
            },
        )

    async def _model_tool_run(
        self,
        context: TrialContext,
        started_at: float,
        holder: dict[str, Any],
        *,
        blocking: bool,
    ) -> TrialRecord:
        """Treatment arm: host model in the loop; LMPU runtime submits, parks, wakes.

        ``blocking`` selects ``lmpu.model_sync`` (mode=sync submit); otherwise
        ``lmpu.model_async`` parks in S1 with zero host decode while parked.
        Bounded ingestion for both; the only delta is park vs block.
        """
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        sleep_id = ""
        sleep_seconds = 0.0
        timeout = context.workload.config.get("timeout_ms", 60_000) / 1000
        try:
            prefix, program, _ = await self._model_program(context, metrics, phases)
            if blocking:
                request = _request(program, context.workload, job_id, "sync")
                metrics.record_control_tokens(estimate_tokens(request))
                submit_started = time.perf_counter()
                submitted = await runtime._submit(request, session_id)
                phases.record("submit", time.perf_counter() - submit_started)
                result = submitted.result
                if submitted.error:
                    error = json.dumps(submitted.error, default=str)
            else:
                request = _request(program, context.workload, job_id, None)
                metrics.record_control_tokens(estimate_tokens(request))
                submit_started = time.perf_counter()
                await runtime._submit(request, session_id)
                phases.record("submit", time.perf_counter() - submit_started)
                sleep_result = await runtime._sleep(
                    {"until": {"all": [job_id]}, "level": "S1"},
                    session_id,
                )
                sleep_id = sleep_result.sleep_id
                park_started = time.perf_counter()
                finished = await runtime.job_table.wait_terminal(job_id, timeout=timeout)
                sleep_seconds = time.perf_counter() - park_started
                metrics.sleep_duration_seconds = sleep_seconds
                phases.record("park", sleep_seconds)
                wake_started = time.perf_counter()
                await runtime._wake({}, session_id)
                metrics.record_wake_latency(time.perf_counter() - wake_started)
                phases.record("wake", metrics.wake_latency_seconds)
                if finished is None:
                    raise TimeoutError("sleep treatment exceeded the job timeout")
                result = finished.result
                if finished.error:
                    error = json.dumps(finished.error, default=str)
            if result is not None:
                suffix = _tool_result_suffix(context.workload, _summary_text(result))
                _, final_answer = await self._decode_answer(
                    context, host, metrics, phases, prefix, suffix
                )
        finally:
            await self._close_session(session_id)
            try:
                metrics.record_energy(gpu_joules=host.take_energy())
            except Exception:
                logger.exception("host energy collection failed")
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        _, oracle = split_oracle(context.baseline)
        metadata: dict[str, Any] = {
            "program": program,
            "answer": final_answer,
            "tool_correct": tool_correct,
            "planner": "oracle" if oracle else "model",
            "ingest_mode": "bounded_summary",
            "phases": phases.as_dict(),
        }
        if not blocking:
            metadata["park"] = {
                "level": "S1",
                "sleep_id": sleep_id,
                "sleep_seconds": sleep_seconds,
                "forward_passes_during_park": 0,
            }
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata=metadata,
        )

    async def _run_model_async(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        return await self._model_tool_run(context, started_at, holder, blocking=False)

    async def _run_model_sync(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        return await self._model_tool_run(context, started_at, holder, blocking=True)

    async def _two_session_run(
        self,
        context: TrialContext,
        started_at: float,
        holder: dict[str, Any],
        *,
        app_level: bool,
    ) -> TrialRecord:
        """Two concurrent sessions on the best-case workload; oracle-planned.

        ``app_level=False``: two LMPU sessions parked in S1 on one runtime.
        ``app_level=True``: two app-level in-process tasks on one shared engine
        with two independent KV-warm decoders; no LMPU runtime.
        """
        host = self._host_model()
        await host.start_trial()
        runtime = None if app_level else self._require_runtime()
        session_ids = (
            [f"bench_{context.baseline}_{context.trial}_{i}" for i in (0, 1)]
            if app_level
            else [await self._open_session() for _ in range(2)]
        )
        metrics = HostMetrics(session_id="+".join(session_ids))
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_ids = [self._job_id(context, f"_s{i}") for i in (0, 1)]
        program = context.workload.program
        if program is None:
            raise ValueError(
                f"Two-session arms require a gold program for workload {context.workload.name}"
            )
        error: str | None = None
        final_answers: list[str | None] = [None, None]
        tool_oks = [False, False]
        polls = [0, 0]
        sleep_id = ""
        timeout = context.workload.config.get("timeout_ms", 180_000) / 1000
        try:
            for _ in session_ids:
                await _record_prefill(metrics, host, context.workload.prompt)
            submit_started = time.perf_counter()
            if app_level:

                async def in_process_leg(index: int) -> tuple[Any, int]:
                    task = asyncio.create_task(
                        _execute_in_process(program, context.workload, job_ids[index])
                    )
                    deadline = time.monotonic() + timeout + 30
                    count = 0
                    while not task.done():
                        count += 1
                        if time.monotonic() > deadline:
                            task.cancel()
                            raise TimeoutError(
                                "in-process tool execution exceeded the two-session deadline"
                            )
                        await asyncio.sleep(self.config.poll_interval_ms / 1000)
                    return task.result(), count

                async def app_leg(index: int) -> tuple[Any, int]:
                    try:
                        return await in_process_leg(index)
                    except Exception as exc:
                        return ({"error": str(exc)}, 0)

                async with asyncio.TaskGroup() as group:
                    leg_a = group.create_task(app_leg(0))
                    leg_b = group.create_task(app_leg(1))
                legs = [leg_a.result(), leg_b.result()]
                results = [legs[0][0], legs[1][0]]
                polls = [legs[0][1], legs[1][1]]
                phases.record("execute", time.perf_counter() - submit_started)
            else:
                assert runtime is not None
                for i, session_id in enumerate(session_ids):
                    request = _request(program, context.workload, job_ids[i], None)
                    metrics.record_control_tokens(estimate_tokens(request))
                    await runtime._submit(request, session_id)
                phases.record("submit", time.perf_counter() - submit_started)
                for session_id in session_ids:
                    result = await runtime._sleep(
                        {"until": {"all": job_ids}, "level": "S1"},
                        session_id,
                    )
                    sleep_id = result.sleep_id
                park_started = time.perf_counter()
                finished = [
                    await runtime.job_table.wait_terminal(jid, timeout=timeout) for jid in job_ids
                ]
                sleep_seconds = time.perf_counter() - park_started
                metrics.sleep_duration_seconds = sleep_seconds
                phases.record("park", sleep_seconds)
                wake_started = time.perf_counter()
                for session_id in session_ids:
                    await runtime._wake({}, session_id)
                metrics.record_wake_latency(time.perf_counter() - wake_started)
                phases.record("wake", metrics.wake_latency_seconds)
                results = [
                    item.result if item is not None and item.error is None else None
                    for item in finished
                ]
                leg_errors = [
                    json.dumps(item.error, default=str)
                    for item in finished
                    if item is not None and item.error
                ]
                if any(item is None for item in finished):
                    leg_errors.append("two-session wait exceeded the job timeout")
                if leg_errors:
                    error = "; ".join(leg_errors)
            for i in (0, 1):
                result = results[i]
                tool_oks[i] = result is not None and context.workload.verify(result)
                if result is None:
                    continue
                suffix = _tool_result_suffix(context.workload, _summary_text(result))
                ingest_started = time.perf_counter()
                await _record_injected(metrics, host, suffix)
                phases.record("ingest", time.perf_counter() - ingest_started)
                conversation = context.workload.prompt + suffix
                answer_started = time.perf_counter()
                response = await host.decode(conversation, self.config.max_tokens, "", session=i)
                phases.record("answer", time.perf_counter() - answer_started)
                await _record_generated(metrics, host, response)
                final_answers[i] = context.workload.parse_answer(response)
            metrics.record_energy(gpu_joules=host.take_energy())
        finally:
            if not app_level and runtime is not None:
                for session_id in session_ids:
                    await self._close_session(session_id)
        wall = time.perf_counter() - started
        answers_ok = [score_answer(context.workload, final_answers[i], error) for i in (0, 1)]
        correct = error is None and all(answers_ok)
        metadata = {
            "program": program,
            "answers": final_answers,
            "answer": final_answers[0],
            "tool_correct": tool_oks,
            "planner": "oracle",
            "ingest_mode": "bounded_summary",
            "sessions": 2,
            "polls": polls,
            "phases": phases.as_dict(),
        }
        if not app_level:
            metadata["park"] = {
                "level": "S1",
                "sleep_id": sleep_id,
                "forward_passes_during_park": 0,
            }
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=",".join(job_ids),
            started_at=started_at,
            wall_time_seconds=wall,
            correct=correct,
            metrics=_payload(metrics, wall),
            error=error,
            metadata=metadata,
        )

    async def _run_model_async_2session(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        return await self._two_session_run(context, started_at, holder, app_level=False)

    async def _run_app_agent_2session(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        return await self._two_session_run(context, started_at, holder, app_level=True)

    async def _run_lmpu_individual(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Submit each batch item as its own synchronous job."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        items = context.workload.inputs.get("items", [])
        # Energy sampling
        self._energy.sample()
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
                submit_started = time.perf_counter()
                try:
                    submitted = await runtime._submit(request, session_id)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                finally:
                    phases.record("submit", time.perf_counter() - submit_started)
                if submitted.error:
                    errors.append(json.dumps(submitted.error, default=str))
                results.append(submitted.result)
            metrics.record_injected_tokens(
                estimate_tokens(_summary_text({"results": results, "errors": errors}))
            )
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        # Record energy after trial
        gpu_joules = self._energy.sample().gpu_joules
        metrics.record_energy(gpu_joules=gpu_joules)
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
            metadata={"submissions": len(results), "errors": errors, "phases": phases.as_dict()},
        )

    async def _run_lmpu_batch(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Submit all batch items in one job."""
        runtime = self._require_runtime()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        job_id = self._job_id(context)
        request = _request(context.workload.program, context.workload, job_id, "sync")
        metrics.record_control_tokens(estimate_tokens(request))
        # Energy sampling
        self._energy.sample()
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            submit_started = time.perf_counter()
            submitted = await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        # Record energy after trial
        gpu_joules = self._energy.sample().gpu_joules
        metrics.record_energy(gpu_joules=gpu_joules)
        if result is not None:
            inject_started = time.perf_counter()
            metrics.record_injected_tokens(estimate_tokens(_summary_text(result)))
            phases.record("ingest", time.perf_counter() - inject_started)
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
            metadata={"batched": True, "phases": phases.as_dict()},
        )

    async def _run_native(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Run the workload computation in-process without the runtime."""
        native = context.workload.native
        if native is None:
            raise ValueError(f"Workload {context.workload.name} has no native baseline")
        metrics = HostMetrics(session_id=f"bench_native_{context.trial}")
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        try:
            execute_started = time.perf_counter()
            result = await asyncio.to_thread(native)
            phases.record("execute", time.perf_counter() - execute_started)
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
            metadata={"native": True, "phases": phases.as_dict()},
        )

    async def _run_direct_cot(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        host = self._host_model()
        session_id = f"bench_direct_cot_{context.trial}"
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        error: str | None = None
        answer: str | None = None
        response = ""
        try:
            await host.start_trial()
            prefill_started = time.perf_counter()
            await _record_prefill(metrics, host, context.workload.prompt)
            phases.record("prefill", time.perf_counter() - prefill_started)
            answer_started = time.perf_counter()
            response = await host.decode(context.workload.prompt, self.config.max_tokens, "")
            phases.record("answer", time.perf_counter() - answer_started)
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
            metadata={"answer": answer, "response": response, "phases": phases.as_dict()},
        )

    async def _run_trace_token(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        session_id = f"bench_trace_token_{context.trial}"
        metrics = HostMetrics(session_id=session_id)
        holder["metrics"] = metrics
        holder["phases"] = {}
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
        self, context: TrialContext, metrics: HostMetrics, phases: PhaseTimer | None = None
    ) -> tuple[str, dict[str, Any], str]:
        host = self._host_model()
        await host.start_trial()
        _, oracle = split_oracle(context.baseline)
        if oracle:
            gold = context.workload.program
            if gold is None:
                raise ValueError(
                    f"Oracle planner requires a gold program for workload {context.workload.name}"
                )
            await _record_prefill(metrics, host, context.workload.prompt)
            if phases is not None:
                phases.record("plan", 0.0)
            return context.workload.prompt, dict(gold), ""
        prompt = context.workload.program_prompt
        plan_started = time.perf_counter()
        await _record_prefill(metrics, host, prompt)

        # Use constrained decoding for model planner arms
        if isinstance(self._host, HFHost):
            engine = self._host.engine
            if engine._model is not None and engine._tokenizer is not None:
                import torch

                # Dynamic schema based on workload's processor and op
                proc = context.workload.processor
                op = context.workload.op
                schema = _schema_for_workload(proc, op)
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_transformers_prefix_allowed_tokens_fn,
                )

                parser = JsonSchemaParser(schema)
                prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
                    engine._tokenizer, parser
                )
                inputs = engine._tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(engine._device_obj)
                with torch.inference_mode():
                    output = engine._model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_tokens,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                        do_sample=False,
                        pad_token_id=engine._tokenizer.eos_token_id,
                    )
                generated_ids = output[0][inputs["input_ids"].shape[1] :]
                text = engine._tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                text = await host.decode(prompt, self.config.max_tokens, "")
        else:
            text = await host.decode(prompt, self.config.max_tokens, "")

        if phases is not None:
            phases.record("plan", time.perf_counter() - plan_started)
        await _record_generated(metrics, host, text)
        program = _program_from_text(text)
        if program is None:
            raise ValueError("model did not emit a valid tool program")
        return prompt + text, program, text

    async def _decode_answer(
        self,
        context: TrialContext,
        host: HostModel,
        metrics: HostMetrics,
        phases: PhaseTimer,
        prefix: str,
        suffix: str,
    ) -> tuple[str, str | None]:
        """Inject ``suffix`` into the KV-warm conversation and decode an answer turn."""
        ingest_started = time.perf_counter()
        await _record_injected(metrics, host, suffix)
        phases.record("ingest", time.perf_counter() - ingest_started)
        conversation = prefix + suffix
        answer_started = time.perf_counter()
        response = await host.decode(conversation, self.config.max_tokens, "")
        phases.record("answer", time.perf_counter() - answer_started)
        await _record_generated(metrics, host, response)
        metrics.record_energy(gpu_joules=host.take_energy())
        return response, context.workload.parse_answer(response)

    async def _run_sync_tool_full_result(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        job_id = self._job_id(context)
        try:
            prefix, program, _ = await self._model_program(context, metrics, phases)
            request = _request(program, context.workload, job_id, "sync")
            metrics.record_control_tokens(estimate_tokens(request))
            submit_started = time.perf_counter()
            submitted = await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            result = submitted.result
            if submitted.error:
                error = json.dumps(submitted.error, default=str)
            if result is not None:
                result_text = _result_text(result)
                suffix = _tool_result_suffix(context.workload, result_text)
                _, final_answer = await self._decode_answer(
                    context, host, metrics, phases, prefix, suffix
                )
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "answer": final_answer,
                "tool_correct": tool_correct,
                "ingest_mode": "full_result",
                "phases": phases.as_dict(),
            },
        )

    async def _poll_until_terminal(
        self,
        runtime: Runtime,
        session_id: str,
        job_id: str,
        metrics: HostMetrics,
        phases: PhaseTimer,
        timeout_seconds: float = 60.0,
    ) -> Any:
        deadline = time.monotonic() + timeout_seconds
        polls = 0
        submitted = None
        poll_started = time.perf_counter()
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
            metrics.record_control_tokens(estimate_tokens(status_text))
            if poll.status in TERMINAL_STATUSES:
                submitted = poll
                break
            if time.monotonic() > deadline:
                raise TimeoutError("async polling exceeded the job timeout")
            await asyncio.sleep(self.config.poll_interval_ms / 1000)
        phases.record("poll", time.perf_counter() - poll_started)
        return polls, submitted

    async def _run_async_poll_programmatic(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        polls = 0
        try:
            prefix, program, _ = await self._model_program(context, metrics, phases)
            request = _request(program, context.workload, job_id, None)
            metrics.record_control_tokens(estimate_tokens(request))
            submit_started = time.perf_counter()
            await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            polls, _ = await self._poll_until_terminal(runtime, session_id, job_id, metrics, phases)
            waited = await runtime._wait({"job_id": job_id, "timeout_ms": 1000}, session_id)
            result = waited.result
            if waited.error:
                error = json.dumps(waited.error, default=str)
            if result is not None:
                summary = _summary_text(result)
                suffix = _tool_result_suffix(context.workload, summary)
                _, final_answer = await self._decode_answer(
                    context, host, metrics, phases, prefix, suffix
                )
        finally:
            await self._close_session(session_id)
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "answer": final_answer,
                "tool_correct": tool_correct,
                "polls": polls,
                "ingest_mode": "bounded_summary",
                "phases": phases.as_dict(),
            },
        )

    async def _run_async_poll_model(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Model-in-the-loop polling: each status check is a model turn."""
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        polls = 0
        submitted: Any = None
        try:
            prefix, program, _ = await self._model_program(context, metrics, phases)
            conversation = prefix
            request = _request(program, context.workload, job_id, None)
            metrics.record_control_tokens(estimate_tokens(request))
            submit_started = time.perf_counter()
            submitted = await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            deadline = time.monotonic() + context.workload.config.get("timeout_ms", 60_000) / 1000
            while True:
                polls += 1
                poll = await runtime._poll({"job_id": job_id}, session_id)
                status = {
                    "status": poll.status.value,
                    "progress": poll.progress,
                    "elapsed_ms": poll.elapsed_ms,
                }
                if poll.status in TERMINAL_STATUSES:
                    waited = await runtime._wait({"job_id": job_id, "timeout_ms": 1000}, session_id)
                    result = waited.result
                    if waited.error:
                        error = json.dumps(waited.error, default=str)
                    status["result"] = _summary_text(result) if result is not None else None
                status_text = json.dumps(status, separators=(",", ":"))
                suffix = (
                    "\nSTATUS: "
                    + status_text
                    + '\nReply "POLL" to keep waiting, or "ANSWER: <value>" with the answer.'
                )
                await _record_injected(metrics, host, suffix)
                conversation += suffix
                poll_started = time.perf_counter()
                response = await host.decode(conversation, 32, "")
                phases.record("poll", time.perf_counter() - poll_started)
                await _record_generated(metrics, host, response)
                conversation += response
                if "ANSWER" in response.upper() or poll.status in TERMINAL_STATUSES:
                    final_answer = context.workload.parse_answer(response)
                    break
                if polls >= 32 or time.monotonic() > deadline:
                    raise TimeoutError("model polling exceeded the deadline")  # noqa: TRY301 - caught below to emit a failed record
                await asyncio.sleep(self.config.poll_interval_ms / 1000)
        except Exception as exc:
            error = error or str(exc)
        finally:
            await self._close_session(session_id)
            try:
                metrics.record_energy(gpu_joules=host.take_energy())
            except Exception:
                logger.exception("host energy collection failed")
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "answer": final_answer,
                "tool_correct": tool_correct,
                "polls": polls,
                "submitted_status": submitted.status.value if submitted else None,
                "ingest_mode": "bounded_summary",
                "phases": phases.as_dict(),
            },
        )

    async def _run_session_reopen(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Async job, then a fresh session: pays full re-prefill instead of KV-warm wake."""
        runtime = self._require_runtime()
        host = self._host_model()
        session_id = await self._open_session()
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        job_timeout = context.workload.config.get("timeout_ms", 60_000) / 1000
        try:
            _, program, _ = await self._model_program(context, metrics, phases)
            request = _request(program, context.workload, job_id, None)
            metrics.record_control_tokens(estimate_tokens(request))
            submit_started = time.perf_counter()
            await runtime._submit(request, session_id)
            phases.record("submit", time.perf_counter() - submit_started)
            polls, _ = await self._poll_until_terminal(
                runtime, session_id, job_id, metrics, phases, job_timeout
            )
            waited = await runtime._wait({"job_id": job_id, "timeout_ms": 1000}, session_id)
            result = waited.result
            if waited.error:
                error = json.dumps(waited.error, default=str)
        finally:
            await self._close_session(session_id)
        # Reopen: discard the KV cache and re-prefill the full context.
        try:
            await host.start_trial()
            result_text = _result_text(result) if result is not None else "{}"
            full_prompt = (
                context.workload.prompt
                + f"\n\nTOOL RESULT:\n{result_text}\n"
                + f'Reply with a final line "{context.workload.answer_line}".'
            )
            prefill_started = time.perf_counter()
            await _record_prefill(metrics, host, full_prompt)
            phases.record("reprefill", time.perf_counter() - prefill_started)
            answer_started = time.perf_counter()
            response = await host.decode(full_prompt, self.config.max_tokens, "")
            phases.record("answer", time.perf_counter() - answer_started)
            await _record_generated(metrics, host, response)
            metrics.record_energy(gpu_joules=host.take_energy())
            final_answer = context.workload.parse_answer(response)
        except Exception as exc:
            error = error or str(exc)
        finally:
            try:
                metrics.record_energy(gpu_joules=host.take_energy())
            except Exception:
                logger.exception("host energy collection failed")
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "answer": final_answer,
                "tool_correct": tool_correct,
                "ingest_mode": "full_reprefill",
                "phases": phases.as_dict(),
            },
        )

    async def _app_style_run(
        self,
        context: TrialContext,
        started_at: float,
        holder: dict[str, Any],
        *,
        polling: bool,
    ) -> TrialRecord:
        """App-level agent loop: same model, prompt and budget as LMPU arms.

        Tool call is executed in-process (no LMPU runtime), the session stays
        KV-warm via the engine's prefix cache, and only a bounded result
        summary enters context. ``polling`` selects the programmatic poll loop
        (``app_agent``) or a blocking in-process call (``engine_native``).
        """
        host = self._host_model()
        session_id = f"bench_{context.baseline}_{context.trial}"
        metrics = HostMetrics(session_id=session_id)
        phases = PhaseTimer()
        holder["metrics"] = metrics
        holder["phases"] = phases.phases
        started = time.perf_counter()
        job_id = self._job_id(context)
        error: str | None = None
        result: Any = None
        program: dict[str, Any] | None = None
        final_answer: str | None = None
        polls = 0
        try:
            prefix, program, _ = await self._model_program(context, metrics, phases)
            execute_started = time.perf_counter()
            if polling:
                task = asyncio.create_task(_execute_in_process(program, context.workload, job_id))
                deadline = (
                    time.monotonic() + context.workload.config.get("timeout_ms", 60_000) / 1000
                )
                while not task.done():
                    polls += 1
                    if time.monotonic() > deadline:
                        task.cancel()
                        raise TimeoutError("in-process tool execution exceeded 60 seconds")  # noqa: TRY301 - caught below to emit a failed record
                    await asyncio.sleep(self.config.poll_interval_ms / 1000)
                result = task.result()
            else:
                result = await _execute_in_process(program, context.workload, job_id)
            phases.record("execute", time.perf_counter() - execute_started)
            summary = _summary_text(result)
            suffix = _tool_result_suffix(context.workload, summary)
            _, final_answer = await self._decode_answer(
                context, host, metrics, phases, prefix, suffix
            )
        except Exception as exc:
            error = str(exc)
            try:
                metrics.record_energy(gpu_joules=host.take_energy())
            except Exception:
                logger.exception("host energy collection failed")
        wall = time.perf_counter() - started
        tool_correct = result is not None and context.workload.verify(result)
        return TrialRecord(
            workload=context.workload.name,
            baseline=context.baseline,
            trial=context.trial,
            seed=context.seed,
            job_id=job_id,
            started_at=started_at,
            wall_time_seconds=wall,
            correct=score_answer(context.workload, final_answer, error),
            metrics=_payload(metrics, wall),
            error=error,
            metadata={
                "program": program,
                "answer": final_answer,
                "tool_correct": tool_correct,
                "polls": polls,
                "ingest_mode": "bounded_summary",
                "phases": phases.as_dict(),
            },
        )

    async def _run_app_agent(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """App-level agent: prefix-cached session, bounded result, programmatic polling."""
        return await self._app_style_run(context, started_at, holder, polling=True)

    async def _run_engine_native(
        self, context: TrialContext, started_at: float, holder: dict[str, Any]
    ) -> TrialRecord:
        """Engine-native session reuse without the LMPU protocol: blocking in-process call."""
        return await self._app_style_run(context, started_at, holder, polling=False)

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
