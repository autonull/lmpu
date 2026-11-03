"""Benchmark configuration with file loading and explicit overrides."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Pre-registered benchmark inputs.

    ``seed`` seeds per-trial work. ``trials`` counts measured runs only;
    ``warmups`` runs are discarded and never written to the raw log.
    """

    name: str = "lmpu-bench"
    workloads: tuple[str, ...] = ("bigint_mul",)
    trials: int = 20
    warmups: int = 3
    seed: int = 20261103
    device: str = "cpu"
    model: str = "Qwen/Qwen3.5-0.8B"
    results_root: Path = Path("benchmarks/results")
    resume: bool = True

    def __post_init__(self) -> None:
        if not self.workloads:
            raise ValueError("At least one workload is required")
        if self.trials < 0:
            raise ValueError("trials must be non-negative")
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")
        if not self.results_root:
            raise ValueError("results_root must not be empty")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> BenchmarkConfig:
        """Build a config from a TOML/JSON mapping, rejecting unknown keys."""
        known = {item.name for item in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"Unknown benchmark config keys: {sorted(unknown)}")

        values = dict(mapping)
        if "workloads" in values:
            values["workloads"] = tuple(values["workloads"])
        if "results_root" in values:
            values["results_root"] = Path(values["results_root"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        """Load a flat TOML or JSON config file."""
        source = Path(path)
        raw = source.read_bytes()
        if source.suffix == ".toml":
            mapping = tomllib.loads(raw.decode("utf-8"))
        elif source.suffix == ".json":
            mapping = json.loads(raw)
        else:
            raise ValueError(f"Unsupported config format: {source.suffix}")
        if not isinstance(mapping, dict):
            raise TypeError("Benchmark config must be a mapping")
        return cls.from_mapping(mapping)

    def with_overrides(self, **overrides: Any) -> BenchmarkConfig:
        """Return a copy with the non-``None`` overrides applied."""
        applied = {key: value for key, value in overrides.items() if value is not None}
        if "workloads" in applied and isinstance(applied["workloads"], str):
            applied["workloads"] = (applied["workloads"],)
        if "results_root" in applied:
            applied["results_root"] = Path(applied["results_root"])
        return replace(self, **applied)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config for the artifact manifest."""
        return {
            "name": self.name,
            "workloads": list(self.workloads),
            "trials": self.trials,
            "warmups": self.warmups,
            "seed": self.seed,
            "device": self.device,
            "model": self.model,
            "results_root": str(self.results_root),
            "resume": self.resume,
        }
