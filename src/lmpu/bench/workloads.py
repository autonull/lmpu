"""Benchmark workload declarations, prompts, correctness checks, and traces."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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


_A = int("987654321098765432109876543210")
_B = int("123456789012345678901234567890")
_EXPECTED = str(_A * _B)


def _verify_bigint_mul(result: object) -> bool:
    return isinstance(result, dict) and result.get("result") == _EXPECTED


def _parse_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*(-?\d+)", text)
    return match.group(1) if match else None


def _verify_answer(value: object) -> bool:
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


_PROMPT = (
    f"Compute {_A} * {_B} exactly. Show concise work, then a final line "
    'of the form "ANSWER: <integer>".'
)
_PROGRAM_PROMPT = (
    "You have a tool. Reply with only one JSON object, no prose or fences: "
    '{"processor":"pure.func","op":"bigint_mul","inputs":{"a":"<a>","b":"<b>"}}. '
    f"Task: multiply {_A} by {_B}."
)
_PROGRAM = {"processor": "pure.func", "op": "bigint_mul", "inputs": {"a": str(_A), "b": str(_B)}}

_BIGINT_MUL = WorkloadSpec(
    name="bigint_mul",
    processor="pure.func",
    op="bigint_mul",
    inputs={"a": str(_A), "b": str(_B)},
    verify=_verify_bigint_mul,
    answer=_EXPECTED,
    prompt=_PROMPT,
    program_prompt=_PROGRAM_PROMPT,
    program=_PROGRAM,
    parse_answer=_parse_answer,
    verify_answer=_verify_answer,
    trace=_trace_bigint_mul,
    description="30-digit exact multiplication through the reference runtime",
)

_WORKLOADS = {
    "bigint_mul": _BIGINT_MUL,
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
