"""Resume-safe benchmark harness and artifact writer."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import re
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .baselines import BaselineRunner, resolve_baselines
from .config import BenchmarkConfig
from .runner import TrialContext, TrialRunner
from .workloads import WorkloadSpec, resolve_workloads

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "lmpu.bench.manifest.v1"
SUMMARY_SCHEMA = "lmpu.bench.summary.v1"
SUMMARY_METRIC_KEYS = (
    "forward_passes",
    "prefill_tokens",
    "generated_tokens",
    "control_tokens",
    "injected_tokens",
    "total_tokens",
    "peak_context_tokens",
    "vram_peak_mb",
    "gpu_energy_joules",
    "cpu_energy_joules",
    "sleep_duration_seconds",
    "wake_latency_seconds",
)


class BenchmarkArtifactError(RuntimeError):
    """Raised when resuming into an incompatible artifact directory."""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit[:12] if commit else "nogit",
        "full_commit": commit or None,
        "dirty": bool(status),
    }


def detect_hardware(device: str) -> dict[str, Any]:
    """Capture the hardware and runtime manifest used for a run."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "device": device,
        "gpu": None,
    }
    if not device.startswith("cuda"):
        return info

    try:
        import torch
    except ImportError:
        return info

    try:
        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        }
    except Exception:
        logger.exception("GPU detection failed")
    return info


def _run_directory(
    config: BenchmarkConfig,
    hardware: dict[str, Any],
    git: dict[str, Any],
    now: float,
) -> Path:
    stamp = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%d")
    gpu = hardware.get("gpu") or {}
    device_name = gpu.get("name") if isinstance(gpu, dict) else None
    device_slug = _slug(device_name or config.device)
    return Path(config.results_root) / stamp / device_slug / git["commit"]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_raw(path: Path) -> list[dict[str, Any]]:
    """Read a raw JSONL log.

    A truncated final line is ignored so an interrupted run can resume. Any
    other malformed line means the artifact is corrupt.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) and not text.endswith("\n"):
                logger.warning("ignoring truncated final raw record", extra={"line": index})
                break
            raise BenchmarkArtifactError(f"Invalid raw record at line {index}: {exc}") from exc
        records.append(record)
    return records


def append_raw(path: Path, record: dict[str, Any]) -> None:
    """Append one complete record and flush it to disk."""
    line = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bootstrap_ci(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values))) for _ in range(2000)
    )
    return [medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians)) - 1]]


def summarize(
    records: list[dict[str, Any]],
    workloads: tuple[WorkloadSpec, ...],
    baselines: tuple[str, ...],
) -> dict[str, Any]:
    """Aggregate raw trials into medians and bootstrap 95% intervals."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["workload"], record.get("baseline", "lmpu.sync"))
        grouped.setdefault(key, []).append(record)

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "record_count": len(records),
        "workloads": {},
    }
    for workload in workloads:
        by_baseline: dict[str, Any] = {}
        for baseline in baselines:
            trials = grouped.get((workload.name, baseline), [])
            walls = [float(record["wall_time_seconds"]) for record in trials]
            correctness = [bool(record["correct"]) for record in trials]
            metric_medians = {}
            for key in SUMMARY_METRIC_KEYS:
                values = [
                    float(record["metrics"][key])
                    for record in trials
                    if key in record.get("metrics", {})
                ]
                if values:
                    metric_medians[key] = statistics.median(values)

            by_baseline[baseline] = {
                "trials": len(trials),
                "wall_time_seconds": {
                    "median": statistics.median(walls) if walls else None,
                    "ci95": _bootstrap_ci(walls, workload_hash(f"{workload.name}:{baseline}")),
                },
                "metrics": metric_medians,
                "correctness": {
                    "passes": sum(correctness),
                    "total": len(correctness),
                    "rate": (sum(correctness) / len(correctness)) if correctness else None,
                },
                "failures": [
                    {"trial": record["trial"], "error": record["error"]}
                    for record in trials
                    if not record["correct"]
                ],
                "metadata": {
                    "simulated": any(
                        record.get("metadata", {}).get("simulated") for record in trials
                    ),
                    "polls": [
                        record["metadata"]["polls"]
                        for record in trials
                        if "polls" in record.get("metadata", {})
                    ],
                },
            }
        summary["workloads"][workload.name] = {
            "description": workload.description,
            "baselines": by_baseline,
        }
    return summary


def workload_hash(name: str) -> int:
    """Stable per-workload bootstrap seed."""
    return sum(ord(char) for char in name) * 7919


class BenchmarkHarness:
    """Run trials into a resume-safe artifact directory."""

    def __init__(
        self,
        config: BenchmarkConfig,
        runner: TrialRunner | None = None,
        hardware: dict[str, Any] | None = None,
        git: dict[str, Any] | None = None,
        clock: Any = None,
    ):
        self.config = config
        self.runner = runner or BaselineRunner(config)
        self.hardware = hardware
        self.git = git
        self.clock = clock or time.time

    def _prepare(self) -> tuple[Path, dict[str, Any]]:
        root = Path(__file__).resolve().parents[3]
        hardware = self.hardware or detect_hardware(self.config.device)
        git = self.git or _git_info(root)
        run_dir = _run_directory(self.config, hardware, git, self.clock())
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "config": self.config.to_dict(),
            "commit": git["commit"],
            "full_commit": git.get("full_commit"),
            "dirty": git["dirty"],
            "hardware": hardware,
            "started_at": datetime.fromtimestamp(self.clock(), tz=UTC).isoformat(),
            "run_dir": str(run_dir),
        }
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config") != manifest["config"]:
                raise BenchmarkArtifactError(
                    f"Run directory {run_dir} was created with a different config; "
                    "choose another results root or remove the artifact directory"
                )
        else:
            _atomic_write_json(manifest_path, manifest)
        return run_dir, manifest

    async def run(self) -> dict[str, Any]:
        workloads = resolve_workloads(self.config.workloads)
        baselines = resolve_baselines(self.config.baselines)
        run_dir, manifest = self._prepare()
        raw_path = run_dir / "raw.jsonl"
        if not self.config.resume and raw_path.exists():
            raise BenchmarkArtifactError(
                f"Resume is disabled but {raw_path} already exists; remove the artifact directory"
            )
        records = read_raw(raw_path)
        completed = {
            (record.get("baseline", "lmpu.sync"), record["workload"], int(record["trial"]))
            for record in records
            if "workload" in record and "trial" in record
        }
        pairs = [(workload, baseline) for workload in workloads for baseline in baselines]

        await self.runner.setup(run_dir)
        try:
            for warmup in range(self.config.warmups):
                ordered = list(pairs)
                random.Random(self.config.seed + warmup).shuffle(ordered)
                for workload, baseline in ordered:
                    context = TrialContext(
                        config=self.config,
                        workload=workload,
                        baseline=baseline,
                        trial=-(warmup + 1),
                        seed=self.config.seed + warmup,
                        run_dir=run_dir,
                    )
                    try:
                        await self.runner.run(context)
                    except Exception:
                        logger.exception(
                            "warmup failed",
                            extra={"workload": workload.name, "baseline": baseline},
                        )

            for trial in range(1, self.config.trials + 1):
                ordered = list(pairs)
                random.Random(self.config.seed + trial).shuffle(ordered)
                for workload, baseline in ordered:
                    if (baseline, workload.name, trial) in completed:
                        continue
                    context = TrialContext(
                        config=self.config,
                        workload=workload,
                        baseline=baseline,
                        trial=trial,
                        seed=self.config.seed + trial,
                        run_dir=run_dir,
                    )
                    record = await self.runner.run(context)
                    payload = {"run_id": manifest["run_dir"], **record.to_dict()}
                    append_raw(raw_path, payload)
                    records.append(payload)
                    completed.add((baseline, workload.name, trial))
        finally:
            await self.runner.teardown()

        summary = summarize(records, workloads, baselines)
        summary["run_id"] = manifest["run_dir"]
        summary["config"] = self.config.to_dict()
        summary["commit"] = manifest["commit"]
        _atomic_write_json(run_dir / "summary.json", summary)
        return summary
