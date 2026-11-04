"""Benchmark workload declarations, prompts, correctness checks, and traces."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One reproducible unit of benchmark work.

    ``verify`` validates a runtime result. Model baselines use ``prompt`` and
    ``program_prompt``; ``parse_answer`` and ``verify_answer`` score natural
    language. ``trace`` produces the labeled trace-token emulation. ``native``
    runs the same computation in-process for the C5 overhead baseline.
    """

    name: str
    processor: str
    op: str
    inputs: dict[str, Any]
    verify: Callable[[object], bool]
    answer: str
    prompt: str
    program_prompt: str
    program: dict[str, Any]
    parse_answer: Callable[[str], str | None]
    verify_answer: Callable[[object], bool]
    trace: Callable[[], str]
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    baselines: tuple[str, ...] | None = None
    native: Callable[[], object] | None = None
    answer_line: str = "ANSWER: <integer>"


def _parse_integer_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*(-?\d+)", text)
    return match.group(1) if match else None


def _parse_secret_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*([A-Za-z0-9-]+)", text)
    return match.group(1) if match else None


# --- bigint_mul ------------------------------------------------------------

_A = int("987654321098765432109876543210")
_B = int("123456789012345678901234567890")
_EXPECTED = str(_A * _B)


def _verify_bigint_mul(result: object) -> bool:
    return isinstance(result, dict) and result.get("result") == _EXPECTED


def _verify_bigint_answer(value: object) -> bool:
    return value == _EXPECTED


def _trace_bigint_mul() -> str:
    left = str(_A)
    right = str(_B)
    lines = ["TRACE bigint_mul schoolbook"]
    for left_index, left_digit in enumerate(reversed(left)):
        for right_index, right_digit in enumerate(reversed(right)):
            lines.append(f"MUL d{left_index} d{right_index} {left_digit} {right_digit}")
    lines.append(f"RESULT {_EXPECTED}")
    return "\n".join(lines)


def _native_bigint_mul() -> dict[str, str]:
    return {"result": _EXPECTED}


_BIGINT_MUL = WorkloadSpec(
    name="bigint_mul",
    processor="pure.func",
    op="bigint_mul",
    inputs={"a": str(_A), "b": str(_B)},
    verify=_verify_bigint_mul,
    answer=_EXPECTED,
    prompt=(
        f"Compute {_A} * {_B} exactly. Show concise work, then a final line "
        'of the form "ANSWER: <integer>".'
    ),
    program_prompt=(
        "You have a tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"pure.func","op":"bigint_mul","inputs":{"a":"<a>","b":"<b>"}}. '
        f"Task: multiply {_A} by {_B}."
    ),
    program={
        "processor": "pure.func",
        "op": "bigint_mul",
        "inputs": {"a": str(_A), "b": str(_B)},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_bigint_answer,
    trace=_trace_bigint_mul,
    description="30-digit exact multiplication through the reference runtime",
    baselines=(
        "lmpu.sync",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "sync_tool_full_result_oracle",
        "async_poll_programmatic",
        "async_poll_programmatic_oracle",
        "async_poll_model",
        "async_poll_model_oracle",
        "session_reopen",
        "session_reopen_oracle",
        "app_agent",
        "app_agent_oracle",
        "engine_native",
        "engine_native_oracle",
        "lmpu.model_async",
        "lmpu.model_sync",
        "lmpu.model_async_oracle",
        "lmpu.model_sync_oracle",
        "lmpu.model_async_2session",
        "app_agent_2session",
        "native",
    ),
    native=_native_bigint_mul,
)


# --- matmul ----------------------------------------------------------------

_MATMUL_SIZE = 512
_MATMUL_SEED = 7


@lru_cache(maxsize=4)
def _matmul_expected(size: int, seed: int) -> tuple[str, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    left = rng.standard_normal((size, size), dtype=np.float32)
    right = rng.standard_normal((size, size), dtype=np.float32)
    product = left @ right
    return hashlib.sha256(product.tobytes()).hexdigest()[:16], float(product.sum(dtype=np.float64))


def _verify_matmul(result: object) -> bool:
    if not isinstance(result, dict) or result.get("shape") != [_MATMUL_SIZE, _MATMUL_SIZE]:
        return False
    checksum, _ = _matmul_expected(_MATMUL_SIZE, _MATMUL_SEED)
    return result.get("checksum") == checksum


def _trace_matmul() -> str:
    lines = [f"TRACE matmul {_MATMUL_SIZE}x{_MATMUL_SIZE} seed {_MATMUL_SEED}"]
    for row in range(_MATMUL_SIZE):
        lines.append(f"ROW {row} MAC")
    lines.append("RESULT checksum")
    return "\n".join(lines)


def _verify_matmul_answer(_: object) -> bool:
    return False


def _native_matmul() -> dict[str, Any]:
    checksum, total = _matmul_expected(_MATMUL_SIZE, _MATMUL_SEED)
    return {"shape": [_MATMUL_SIZE, _MATMUL_SIZE], "checksum": checksum, "sum": total}


_MATMUL = WorkloadSpec(
    name="matmul",
    processor="pure.func",
    op="matmul",
    inputs={"size": _MATMUL_SIZE, "seed": _MATMUL_SEED},
    verify=_verify_matmul,
    answer="matmul-checksum",
    prompt=(
        f"Compute the deterministic float32 matmul A@B for size {_MATMUL_SIZE}, "
        f"seed {_MATMUL_SEED}. Show concise work, then a final line "
        'of the form "ANSWER: <checksum>".'
    ),
    program_prompt=(
        "You have a tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"pure.func","op":"matmul","inputs":{"size":512,"seed":7}}. '
        f"Task: multiply deterministic {_MATMUL_SIZE}x{_MATMUL_SIZE} matrices."
    ),
    program={"processor": "pure.func", "op": "matmul", "inputs": {"size": 512, "seed": 7}},
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_matmul_answer,
    trace=_trace_matmul,
    description="512x512 float32 matmul through the runtime",
    config={"timeout_ms": 300_000},
    baselines=(
        "lmpu.sync",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_native_matmul,
)


# --- large_context ---------------------------------------------------------

_SECRET = "LMPU-BENCH-ALPHA"


@lru_cache(maxsize=1)
def _large_document() -> str:
    lines = []
    for index in range(120):
        lines.append(
            f"record {index:04d}: node=worker-{index % 17:02d} "
            f"temperature={18 + index % 9} pressure={1000 + index % 31} "
            f"payload={'x' * 40}"
        )
    lines.insert(60, f"SECRET: {_SECRET}")
    return "\n".join(lines)


def _text_expected() -> dict[str, Any]:
    text = _large_document()
    secret_match = re.search(r"SECRET:\s*([A-Za-z0-9-]+)", text)
    return {
        "chars": len(text),
        "words": len(text.split()),
        "checksum": hashlib.sha256(text.encode()).hexdigest()[:16],
        "secret": secret_match.group(1) if secret_match else None,
    }


def _verify_large_context(result: object) -> bool:
    return isinstance(result, dict) and result.get("secret") == _SECRET


def _trace_large_context() -> str:
    return _large_document() + f"\nTRACE text_stats\nRESULT {_SECRET}"


_LARGE_CONTEXT = WorkloadSpec(
    name="large_context",
    processor="pure.func",
    op="text_stats",
    inputs={"text": _large_document()},
    verify=_verify_large_context,
    answer=_SECRET,
    prompt=(
        "Read the document below and find the value after SECRET:. "
        'Return a final line "ANSWER: <secret>".\n\n' + _large_document()
    ),
    program_prompt=(
        "You have a tool that reads text already present in your context. "
        "Reply with only one JSON object, no prose or fences: "
        '{"processor":"pure.func","op":"text_stats","inputs":{"text":"$CONTEXT"}}. '
        "The provided document is the value of $CONTEXT. "
        f"Find SECRET and end with ANSWER: <secret>.\n\n{_large_document()}"
    ),
    program={
        "processor": "pure.func",
        "op": "text_stats",
        "inputs": {"text": "$CONTEXT"},
    },
    parse_answer=_parse_secret_answer,
    verify_answer=lambda value: value == _SECRET,
    trace=_trace_large_context,
    description="12k-character needle-in-haystack context task",
    baselines=(
        "lmpu.sync",
        "trace_token",
        "direct_cot",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_text_expected,
)


# --- sleep -----------------------------------------------------------------

_SLEEP_SECONDS = 3
_SLEEP_PROGRAM = (
    "import json, time\n"
    f"time.sleep({_SLEEP_SECONDS})\n"
    f"print(json.dumps({{'slept': {_SLEEP_SECONDS}}}))"
)


def _verify_sleep(result: object) -> bool:
    return isinstance(result, dict) and float(result.get("slept", 0)) >= _SLEEP_SECONDS - 0.5


def _verify_sleep_answer(value: object) -> bool:
    return value == str(_SLEEP_SECONDS)


def _trace_sleep() -> str:
    return f"TRACE sleep\nWAIT {_SLEEP_SECONDS}\nRESULT {_SLEEP_SECONDS}"


def _native_sleep() -> dict[str, int]:
    time.sleep(_SLEEP_SECONDS)
    return {"slept": _SLEEP_SECONDS}


_SLEEP = WorkloadSpec(
    name="sleep",
    processor="python.vm",
    op="run_program",
    inputs={"program": _SLEEP_PROGRAM},
    verify=_verify_sleep,
    answer=str(_SLEEP_SECONDS),
    prompt=(
        f"A tool must wait {_SLEEP_SECONDS} seconds. Show the call, then a final "
        f'line "ANSWER: {_SLEEP_SECONDS}".'
    ),
    program_prompt=(
        "You have a Python tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"python.vm","op":"run_program","inputs":{"program":"..."}}. '
        f"The program must sleep {_SLEEP_SECONDS} seconds and print its slept duration."
    ),
    program={
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": _SLEEP_PROGRAM},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_sleep_answer,
    trace=_trace_sleep,
    description="3-second sleeping Python job for sleep/wake measurement",
    config={"timeout_ms": 120_000},
    baselines=(
        "lmpu.sync",
        "lmpu.sleep",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_native_sleep,
)


# --- large_result ----------------------------------------------------------

_RESULT_BYTES = 16 * 1024 * 1024


def _verify_large_result(result: object) -> bool:
    return isinstance(result, dict) and int(result.get("bytes", 0)) == _RESULT_BYTES


def _trace_large_result() -> str:
    return f"TRACE large_result\nALLOC {_RESULT_BYTES}\nRESULT {_RESULT_BYTES}"


def _native_large_result() -> dict[str, int]:
    return {"bytes": len(b"\x00" * _RESULT_BYTES)}


_LARGE_RESULT = WorkloadSpec(
    name="large_result",
    processor="pure.func",
    op="buffer_slice",
    inputs={"bytes": _RESULT_BYTES},
    verify=_verify_large_result,
    answer=str(_RESULT_BYTES),
    prompt=(
        "A tool returns a 16 MiB buffer. Explain how a bounded slice read keeps "
        "the host context small."
    ),
    program_prompt=(
        "You have a tool that returns a 16 MiB buffer. Reply with only one JSON object: "
        '{"processor":"pure.func","op":"buffer_slice","inputs":{"bytes":16777216}}.'
    ),
    program={
        "processor": "pure.func",
        "op": "buffer_slice",
        "inputs": {"bytes": _RESULT_BYTES},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=lambda value: value == str(_RESULT_BYTES),
    trace=_trace_large_result,
    description="16 MiB result buffer read through a bounded slice",
    config={"timeout_ms": 120_000, "result_bytes": _RESULT_BYTES, "max_inline_bytes": 1024},
    baselines=("lmpu.slice", "native"),
    native=_native_large_result,
)


# --- batch -----------------------------------------------------------------

_BATCH_ITEMS = tuple(
    {"op": "bigint_add", "inputs": {"a": index, "b": 1_000}} for index in range(64)
)


def _verify_batch(result: object) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("results"), list):
        return False
    results = result["results"]
    if result.get("errors") or len(results) != len(_BATCH_ITEMS):
        return False
    return all(
        isinstance(item, dict) and item.get("result") == str(1_000 + index)
        for index, item in enumerate(results)
    )


def _trace_batch() -> str:
    return f"TRACE batch {len(_BATCH_ITEMS)}\n" + "\n".join(
        f"ADD {item['inputs']['a']} {item['inputs']['b']}" for item in _BATCH_ITEMS
    )


def _native_batch() -> dict[str, Any]:
    return {
        "results": [
            {"result": str(item["inputs"]["a"] + item["inputs"]["b"])} for item in _BATCH_ITEMS
        ],
        "errors": [],
    }


_BATCH = WorkloadSpec(
    name="batch",
    processor="pure.func",
    op="bigint_add",
    inputs={"items": list(_BATCH_ITEMS)},
    verify=_verify_batch,
    answer=str(len(_BATCH_ITEMS)),
    prompt=(
        f"A tool must evaluate {len(_BATCH_ITEMS)} small additions. Show the batch "
        "call and a final count."
    ),
    program_prompt=(
        "You have a tool that accepts a batch of operations. Reply with only one "
        "JSON object: "
        '{"processor":"pure.func","op":"bigint_add","batch":{"items":[...]}}.'
    ),
    program={
        "processor": "pure.func",
        "op": "bigint_add",
        "inputs": {},
        "batch": {"items": list(_BATCH_ITEMS)},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=lambda value: value == str(len(_BATCH_ITEMS)),
    trace=_trace_batch,
    description="64 small additions submitted individually and as one batch",
    config={"timeout_ms": 60_000},
    baselines=("lmpu.individual", "lmpu.batch", "native"),
    native=_native_batch,
)


def _parse_sha256_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*([0-9a-fA-F]{64})", text)
    return match.group(1).lower() if match else None


# --- best_case (8k context, largest feasible subset on 10 GB) ---------------

_BEST_CASE_BYTES = 16 * 1024 * 1024
_BEST_CASE_SLEEP_SECONDS = 60
_BEST_CASE_UNIT = b"lmpu-best-case-16mib-deterministic-unit\n"
_BEST_CASE_DOCUMENT_LINES_8K = 200  # ~8.2k tokens on Qwen/Qwen3.5-0.8B


@lru_cache(maxsize=1)
def _best_case_digest() -> str:
    digest = hashlib.sha256()
    written = 0
    while written < _BEST_CASE_BYTES:
        take = min(len(_BEST_CASE_UNIT), _BEST_CASE_BYTES - written)
        digest.update(_BEST_CASE_UNIT[:take])
        written += take
    return digest.hexdigest()


def _best_case_document_8k() -> str:
    return "".join(
        f"SECTION {i:04d} | ledger entry {i} | deterministic filler payload {i * 7919} "
        f"| checksum-block {(i * 104729) % 65536} | end-of-entry\n"
        for i in range(_BEST_CASE_DOCUMENT_LINES_8K)
    )


def _build_best_case_program_prompt() -> str:
    """Build the program prompt with the exact gold program embedded."""
    # Escape the program for JSON embedding
    escaped = _BEST_CASE_PROGRAM.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return (
        "Output this exact JSON object and nothing else:\n"
        f'{{"processor":"python.vm","op":"run_program","inputs":{{"program":"{escaped}"}}}}'
    )


_BEST_CASE_PROGRAM = (
    "import hashlib, json, time\n"
    f"time.sleep({_BEST_CASE_SLEEP_SECONDS})\n"
    "h = hashlib.sha256()\n"
    "written = 0\n"
    f"unit = {_BEST_CASE_UNIT!r}\n"
    f"target = {_BEST_CASE_BYTES}\n"
    "while written < target:\n"
    "    take = min(len(unit), target - written)\n"
    "    h.update(unit[:take])\n"
    "    written += take\n"
    'print(json.dumps({"bytes": written, "sha256": h.hexdigest()}))\n'
)

_BEST_CASE = WorkloadSpec(
    name="best_case",
    processor="python.vm",
    op="run_program",
    inputs={"program": _BEST_CASE_PROGRAM},
    verify=lambda result: (
        isinstance(result, dict)
        and int(result.get("bytes", 0)) == _BEST_CASE_BYTES
        and result.get("sha256") == _best_case_digest()
    ),
    answer=_best_case_digest(),
    prompt=(
        _best_case_document_8k()
        + "\nTASK: A long-running tool job (about 60 seconds) computes a 16 MiB "
        "deterministic buffer and reports its size and SHA-256 digest. Call the "
        'tool, then reply with a final line "ANSWER: <sha256 hex digest>".'
    ),
    program_prompt=_build_best_case_program_prompt(),
    program={
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": _BEST_CASE_PROGRAM},
    },
    parse_answer=_parse_sha256_answer,
    verify_answer=lambda value: isinstance(value, str) and value.lower() == _best_case_digest(),
    trace=lambda: (
        f"TRACE best_case\nSLEEP {_BEST_CASE_SLEEP_SECONDS}\nALLOC 16MiB\nSHA {_best_case_digest()}"
    ),
    description=(
        "16 MiB result, 60 s job, ~8k-token context, exact oracle (largest feasible subset)"
    ),
    config={"timeout_ms": 180_000},
    baselines=(
        "lmpu.model_async",
        "lmpu.model_sync",
        "lmpu.model_async_oracle",
        "lmpu.model_sync_oracle",
        "app_agent_oracle",
        "engine_native_oracle",
        "session_reopen_oracle",
        "lmpu.model_async_2session",
        "app_agent_2session",
        "direct_cot",
        "native",
    ),
    native=lambda: (
        time.sleep(_BEST_CASE_SLEEP_SECONDS),
        {
            "bytes": _BEST_CASE_BYTES,
            "sha256": _best_case_digest(),
        },
    )[1],
    answer_line="ANSWER: <sha256 hex digest>",
)


# --- trivial_arithmetic (negative control) ---------------------------------

_TRIVIAL_A = 1
_TRIVIAL_B = 1
_TRIVIAL_EXPECTED = str(_TRIVIAL_A + _TRIVIAL_B)


def _verify_trivial_arithmetic(result: object) -> bool:
    return isinstance(result, dict) and result.get("result") == _TRIVIAL_EXPECTED


def _verify_trivial_answer(value: object) -> bool:
    return value == _TRIVIAL_EXPECTED


def _trace_trivial_arithmetic() -> str:
    return f"TRACE trivial_arithmetic\nADD {_TRIVIAL_A} {_TRIVIAL_B}\nRESULT {_TRIVIAL_EXPECTED}"


def _native_trivial_arithmetic() -> dict[str, str]:
    return {"result": _TRIVIAL_EXPECTED}


_TRIVIAL_ARITHMETIC = WorkloadSpec(
    name="trivial_arithmetic",
    processor="pure.func",
    op="bigint_add",
    inputs={"a": _TRIVIAL_A, "b": _TRIVIAL_B},
    verify=_verify_trivial_arithmetic,
    answer=_TRIVIAL_EXPECTED,
    prompt=(
        f"Compute {_TRIVIAL_A} + {_TRIVIAL_B} exactly. Show concise work, then a final line "
        f'of the form "ANSWER: <integer>".'
    ),
    program_prompt=(
        "You have a tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"pure.func","op":"bigint_add","inputs":{"a":"<a>","b":"<b>"}}. '
        f"Task: add {_TRIVIAL_A} and {_TRIVIAL_B}."
    ),
    program={
        "processor": "pure.func",
        "op": "bigint_add",
        "inputs": {"a": _TRIVIAL_A, "b": _TRIVIAL_B},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_trivial_answer,
    trace=_trace_trivial_arithmetic,
    description="Trivial 1+1 arithmetic (negative control: LMPU must not claim advantage)",
    baselines=(
        "lmpu.sync",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_native_trivial_arithmetic,
)


# --- small_result (negative control: 1 KiB result) -------------------------

_SMALL_RESULT_BYTES = 1024
_SMALL_RESULT_PROGRAM = (
    "import json\n"
    f"data = 'x' * {_SMALL_RESULT_BYTES}\n"
    f'print(json.dumps({{"bytes": len(data), "data": data}}))'
)


def _verify_small_result(result: object) -> bool:
    return (
        isinstance(result, dict)
        and int(result.get("bytes", 0)) == _SMALL_RESULT_BYTES
        and len(result.get("data", "")) == _SMALL_RESULT_BYTES
    )


def _trace_small_result() -> str:
    return f"TRACE small_result\nALLOC {_SMALL_RESULT_BYTES}\nRESULT {_SMALL_RESULT_BYTES}"


def _native_small_result() -> dict[str, Any]:
    data = "x" * _SMALL_RESULT_BYTES
    return {"bytes": _SMALL_RESULT_BYTES, "data": data}


_SMALL_RESULT = WorkloadSpec(
    name="small_result",
    processor="python.vm",
    op="run_program",
    inputs={"program": _SMALL_RESULT_PROGRAM},
    verify=_verify_small_result,
    answer=str(_SMALL_RESULT_BYTES),
    prompt=(
        "A tool returns a 1 KiB buffer. Call the tool and report the byte count."
    ),
    program_prompt=(
        "You have a Python tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"python.vm","op":"run_program","inputs":{"program":"..."}}. '
        "The program must generate a 1 KiB buffer and print its size and content."
    ),
    program={
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": _SMALL_RESULT_PROGRAM},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=lambda value: value == str(_SMALL_RESULT_BYTES),
    trace=_trace_small_result,
    description="1 KiB result buffer (negative control: bounded ingestion not advantageous)",
    config={"timeout_ms": 60_000},
    baselines=(
        "lmpu.sync",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_native_small_result,
)


# --- short_job (negative control: 1 s job) ---------------------------------

_SHORT_JOB_SECONDS = 1
_SHORT_JOB_PROGRAM = (
    "import json, time\n"
    f"time.sleep({_SHORT_JOB_SECONDS})\n"
    f"print(json.dumps({{'slept': {_SHORT_JOB_SECONDS}}}))"
)


def _verify_short_job(result: object) -> bool:
    return isinstance(result, dict) and float(result.get("slept", 0)) >= _SHORT_JOB_SECONDS - 0.5


def _verify_short_job_answer(value: object) -> bool:
    return value == str(_SHORT_JOB_SECONDS)


def _trace_short_job() -> str:
    return f"TRACE short_job\nWAIT {_SHORT_JOB_SECONDS}\nRESULT {_SHORT_JOB_SECONDS}"


def _native_short_job() -> dict[str, int]:
    time.sleep(_SHORT_JOB_SECONDS)
    return {"slept": _SHORT_JOB_SECONDS}


_SHORT_JOB = WorkloadSpec(
    name="short_job",
    processor="python.vm",
    op="run_program",
    inputs={"program": _SHORT_JOB_PROGRAM},
    verify=_verify_short_job,
    answer=str(_SHORT_JOB_SECONDS),
    prompt=(
        f"A tool must wait {_SHORT_JOB_SECONDS} second. Show the call, then a final "
        f'line "ANSWER: {_SHORT_JOB_SECONDS}".'
    ),
    program_prompt=(
        "You have a Python tool. Reply with only one JSON object, no prose or fences: "
        '{"processor":"python.vm","op":"run_program","inputs":{"program":"..."}}. '
        f"The program must sleep {_SHORT_JOB_SECONDS} second and print its slept duration."
    ),
    program={
        "processor": "python.vm",
        "op": "run_program",
        "inputs": {"program": _SHORT_JOB_PROGRAM},
    },
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_short_job_answer,
    trace=_trace_short_job,
    description="1-second sleeping Python job (negative control: park overhead dominates)",
    config={"timeout_ms": 30_000},
    baselines=(
        "lmpu.sync",
        "lmpu.sleep",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_native_short_job,
)


# --- capability (context-infeasible task) ----------------------------------

_CAPABILITY_SECRET = "LMPU-CAPABILITY-VERDICT-2026"
# ~150k chars = ~37k tokens (exceeds 32k context), but worker can handle it
_CAPABILITY_DOCUMENT_LINES = 1000


@lru_cache(maxsize=1)
def _capability_document() -> str:
    lines = []
    for index in range(_CAPABILITY_DOCUMENT_LINES):
        lines.append(
            f"record {index:05d}: node=worker-{index % 97:02d} "
            f"temperature={-20 + index % 60} pressure={900 + index % 120} "
            f"payload={'x' * 120}"
        )
    # Insert secret at a known position (line 500)
    lines.insert(500, f"SECRET: {_CAPABILITY_SECRET}")
    return "\n".join(lines)


def _capability_expected() -> dict[str, Any]:
    text = _capability_document()
    secret_match = re.search(r"SECRET:\s*([A-Za-z0-9-]+)", text)
    return {
        "chars": len(text),
        "words": len(text.split()),
        "checksum": hashlib.sha256(text.encode()).hexdigest()[:16],
        "secret": secret_match.group(1) if secret_match else None,
    }


def _verify_capability(result: object) -> bool:
    return isinstance(result, dict) and result.get("secret") == _CAPABILITY_SECRET


def _trace_capability() -> str:
    return _capability_document() + f"\nTRACE text_stats\nRESULT {_CAPABILITY_SECRET}"


_CAPABILITY = WorkloadSpec(
    name="capability",
    processor="pure.func",
    op="text_stats",
    inputs={"text": _capability_document()},
    verify=_verify_capability,
    answer=_CAPABILITY_SECRET,
    prompt=(
        "Read the document below and find the value after SECRET:. "
        'Return a final line "ANSWER: <secret>".\n\n' + _capability_document()
    ),
    program_prompt=(
        "You have a tool that reads text already present in your context. "
        "Reply with only one JSON object, no prose or fences: "
        '{"processor":"pure.func","op":"text_stats","inputs":{"text":"$CONTEXT"}}. '
        "The provided document is the value of $CONTEXT. "
        f"Find SECRET and end with ANSWER: <secret>.\n\n{_capability_document()}"
    ),
    program={
        "processor": "pure.func",
        "op": "text_stats",
        "inputs": {"text": "$CONTEXT"},
    },
    parse_answer=_parse_secret_answer,
    verify_answer=lambda value: value == _CAPABILITY_SECRET,
    trace=_trace_capability,
    description=(
        f"~80k-token needle-in-haystack (exceeds 32k host context). "
        "Two app_agent variants: (a) full ingest (must fail), (b) app-side reducer (competitive baseline). "
        "LMPU solves via worker-side reduction (P1)."
    ),
    config={"timeout_ms": 120_000},
    baselines=(
        "lmpu.sync",
        "direct_cot",
        "trace_token",
        "sync_tool_full_result",
        "async_poll_programmatic",
        "async_poll_model",
        "session_reopen",
        "app_agent",
        "engine_native",
        "native",
    ),
    native=_capability_expected,
)


_WORKLOADS = {
    "bigint_mul": _BIGINT_MUL,
    "matmul": _MATMUL,
    "large_context": _LARGE_CONTEXT,
    "sleep": _SLEEP,
    "large_result": _LARGE_RESULT,
    "batch": _BATCH,
    "best_case": _BEST_CASE,
    "trivial_arithmetic": _TRIVIAL_ARITHMETIC,
    "small_result": _SMALL_RESULT,
    "short_job": _SHORT_JOB,
    "capability": _CAPABILITY,
}

_ALIASES = {
    "smoke": "bigint_mul",
}


def list_workloads() -> tuple[str, ...]:
    """Return canonical workload names in registration order."""
    return tuple(_WORKLOADS)


def get_workload(name: str) -> WorkloadSpec:
    """Resolve one workload name or alias."""
    canonical = _ALIASES.get(name, name)
    try:
        return _WORKLOADS[canonical]
    except KeyError as exc:
        raise ValueError(f"Unknown workload: {name}") from exc


def resolve_workloads(selection: str | Iterable[str]) -> tuple[WorkloadSpec, ...]:
    """Resolve ``all``, one name, or a set of names into workload specs."""
    if isinstance(selection, str):
        names = list_workloads() if selection == "all" else (selection,)
    else:
        names = tuple(selection)
    return tuple(get_workload(name) for name in names)
