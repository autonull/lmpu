"""Benchmark workload declarations, prompts, correctness checks, and traces."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One reproducible unit of benchmark work.

    ``verify`` validates a runtime result. Model baselines use ``prompt`` and
    ``program_prompt``; ``parse_answer`` and ``verify_answer`` score natural
    language. ``trace`` produces the labeled trace-token emulation.
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
)


# --- matmul ----------------------------------------------------------------

_MATMUL_SIZE = 2048
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
        '{"processor":"pure.func","op":"matmul","inputs":{"size":2048,"seed":7}}. '
        f"Task: multiply deterministic {_MATMUL_SIZE}x{_MATMUL_SIZE} matrices."
    ),
    program={"processor": "pure.func", "op": "matmul", "inputs": {"size": 2048, "seed": 7}},
    parse_answer=_parse_integer_answer,
    verify_answer=_verify_matmul_answer,
    trace=_trace_matmul,
    description="2048x2048 float32 matmul through the runtime",
    config={"timeout_ms": 300_000},
)


# --- large_context ---------------------------------------------------------

_SECRET = "LMPU-2048-ALPHA"


@lru_cache(maxsize=1)
def _large_document() -> str:
    lines = []
    for index in range(500):
        lines.append(
            f"record {index:04d}: node=worker-{index % 17:02d} "
            f"temperature={18 + index % 9} pressure={1000 + index % 31} "
            f"payload={'x' * 40}"
        )
    lines.insert(250, f"SECRET: {_SECRET}")
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
    description="50k-character needle-in-haystack context task",
)


# --- sleep -----------------------------------------------------------------

_SLEEP_SECONDS = 12
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
    description="12-second sleeping Python job for sleep/wake measurement",
    config={"timeout_ms": 120_000},
)


_WORKLOADS = {
    "bigint_mul": _BIGINT_MUL,
    "matmul": _MATMUL,
    "large_context": _LARGE_CONTEXT,
    "sleep": _SLEEP,
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
