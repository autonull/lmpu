"""Benchmark workload declarations and correctness checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """One reproducible unit of benchmark work."""

    name: str
    processor: str
    op: str
    inputs: dict[str, Any]
    verify: Callable[[object], bool]
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)


_A = int("987654321098765432109876543210")
_B = int("123456789012345678901234567890")


def _verify_bigint_mul(result: object) -> bool:
    return isinstance(result, dict) and result.get("result") == str(_A * _B)


_BIGINT_MUL = WorkloadSpec(
    name="bigint_mul",
    processor="pure.func",
    op="bigint_mul",
    inputs={"a": str(_A), "b": str(_B)},
    verify=_verify_bigint_mul,
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
