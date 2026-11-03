"""Render C1-C5 benchmark reports from raw harness artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_KEYS = (
    "prefill_tokens",
    "generated_tokens",
    "control_tokens",
    "injected_tokens",
    "total_tokens",
)
ENERGY_KEYS = ("gpu_energy_joules", "vram_peak_mb")
TIMING_KEYS = ("sleep_duration_seconds", "wake_latency_seconds")
PRIMARY_BASELINES = ("lmpu.sync", "lmpu.slice", "lmpu.sleep", "lmpu.batch")


def _primary_baseline(table: dict[str, Any], workload: str) -> str | None:
    return next(
        (baseline for baseline in PRIMARY_BASELINES if baseline in table.get(workload, {})),
        None,
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ci95(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values))) for _ in range(2000)
    )
    return [medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians)) - 1]]


def _cdf(values: list[float]) -> list[float]:
    return sorted(values)


def _group(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["workload"], record["baseline"])].append(record)
    return grouped


def _stats(trials: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    walls = [float(record["wall_time_seconds"]) for record in trials]
    correctness = [bool(record["correct"]) for record in trials]
    tokens = {
        key: _median(
            [float(record["metrics"][key]) for record in trials if key in record["metrics"]]
        )
        for key in TOKEN_KEYS
    }
    energy = {
        key: _median(
            [float(record["metrics"][key]) for record in trials if key in record["metrics"]]
        )
        for key in ENERGY_KEYS
    }
    timing = {
        key: _median(
            [float(record["metrics"][key]) for record in trials if key in record["metrics"]]
        )
        for key in TIMING_KEYS
    }
    result_bytes = _median(
        [
            float(record["metadata"]["result_bytes"])
            for record in trials
            if "result_bytes" in record.get("metadata", {})
        ]
    )
    return {
        "trials": len(trials),
        "wall_time_seconds": {"median": _median(walls), "ci95": _ci95(walls, seed)},
        "wall_time_cdf": _cdf(walls),
        "tokens": tokens,
        "forward_passes": _median(
            [
                float(record["metrics"]["forward_passes"])
                for record in trials
                if "forward_passes" in record["metrics"]
            ]
        ),
        "peak_context_tokens": _median(
            [
                float(record["metrics"]["peak_context_tokens"])
                for record in trials
                if "peak_context_tokens" in record["metrics"]
            ]
        ),
        "token_cdf": {key: _cdf([float(r["metrics"][key]) for r in trials]) for key in TOKEN_KEYS},
        "energy": energy,
        "timing": timing,
        "result_bytes": result_bytes,
        "derived": any(record.get("metadata", {}).get("derived") for record in trials),
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
        "simulated": any(record.get("metadata", {}).get("simulated") for record in trials),
    }


def _reduction(numerator: float | None, denominator: float | None) -> float | None:
    if not numerator or not denominator:
        return None
    return denominator / numerator


def build_report(run_dir: Path) -> dict[str, Any]:
    """Build a report dictionary from a harness artifact directory."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = _group(records)
    baselines = manifest["config"].get("baselines", [])
    workloads = manifest["config"].get("workloads", [])

    table: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        table[workload] = {}
        for baseline in baselines:
            trials = grouped.get((workload, baseline), [])
            if trials:
                table[workload][baseline] = _stats(trials, len(trials) * 7919)

    c1 = {}
    c1_complete = True
    for workload in workloads:
        treatment = table.get(workload, {}).get("lmpu.sync")
        if not treatment:
            continue
        treatment_tokens = treatment["tokens"]["total_tokens"]
        c1[workload] = {
            baseline: {
                "baseline_total_tokens": stats["tokens"]["total_tokens"],
                "reduction": _reduction(treatment_tokens, stats["tokens"]["total_tokens"]),
                "correctness": stats["correctness"],
            }
            for baseline, stats in table[workload].items()
            if baseline in {"direct_cot", "trace_token", "sync_tool", "async_poll"}
        }
        for stats in c1[workload].values():
            if stats["correctness"]["rate"] != 1.0 or stats["reduction"] is None:
                c1_complete = False

    sleep = table.get("sleep", {})
    sleep_treatment = sleep.get("lmpu.sleep")
    large_result = table.get("large_result", {})
    slice_stats = large_result.get("lmpu.slice")
    batch = table.get("batch", {})
    individual = batch.get("lmpu.individual")
    batched = batch.get("lmpu.batch")
    native_by_workload = {
        workload: table[workload]["native"]
        for workload in workloads
        if "native" in table.get(workload, {})
    }

    c2_measured = (
        sleep_treatment is not None
        and (sleep_treatment["forward_passes"] or 0) == 0
        and sleep_treatment["correctness"]["rate"] == 1.0
    )
    c3_measured = (
        slice_stats is not None
        and slice_stats["peak_context_tokens"] is not None
        and slice_stats["peak_context_tokens"] <= 64
        and slice_stats["correctness"]["rate"] == 1.0
    )
    c4_measured = (
        individual is not None
        and batched is not None
        and individual["correctness"]["rate"] == 1.0
        and batched["correctness"]["rate"] == 1.0
        and batched["wall_time_seconds"]["median"] <= individual["wall_time_seconds"]["median"]
    )
    c5_measured = (
        bool(workloads)
        and len(native_by_workload) == len(workloads)
        and all(stats["correctness"]["rate"] == 1.0 for stats in native_by_workload.values())
    )

    def _duration(stats: dict[str, Any] | None) -> dict[str, Any] | None:
        if stats is None:
            return None
        return {
            "forward_passes": stats["forward_passes"],
            "sleep_duration_seconds": stats["timing"]["sleep_duration_seconds"],
            "wake_latency_seconds": stats["timing"]["wake_latency_seconds"],
            "wall_time_seconds": stats["wall_time_seconds"]["median"],
            "correctness": stats["correctness"],
        }

    report: dict[str, Any] = {
        "schema": "lmpu.bench.report.v1",
        "run_dir": str(run_dir),
        "manifest": manifest,
        "table": table,
        "claims": {
            "C1_token_reduction": {
                "status": ("measured" if c1_complete else "partial") if c1 else "not_measured",
                "by_workload": c1,
            },
            "C2_sleep_proof": {
                "status": ("measured" if c2_measured else ("partial" if sleep else "not_measured")),
                "treatment": _duration(sleep_treatment),
                "measured": {
                    baseline: {
                        "generated_tokens": stats["tokens"]["generated_tokens"],
                        "forward_passes": stats["forward_passes"],
                        "gpu_energy_joules": stats["energy"]["gpu_energy_joules"],
                        "wall_time_seconds": stats["wall_time_seconds"]["median"],
                    }
                    for baseline, stats in sleep.items()
                },
                "note": (
                    "The lmpu.sleep treatment parks the host in S1 for the measured "
                    "job window and performs zero forward passes while parked."
                ),
                "limitation": None
                if c2_measured
                else "No lmpu.sleep/S1 treatment is present in this run.",
            },
            "C3_context_protection": {
                "status": (
                    "measured" if c3_measured else ("partial" if slice_stats else "not_measured")
                ),
                "result_bytes": slice_stats["result_bytes"] if slice_stats else None,
                "measured": None
                if slice_stats is None
                else {
                    "lmpu.slice": {
                        "control_tokens": slice_stats["tokens"]["control_tokens"],
                        "injected_tokens": slice_stats["tokens"]["injected_tokens"],
                        "peak_context_tokens": slice_stats["peak_context_tokens"],
                        "derived": slice_stats["derived"],
                        "correctness": slice_stats["correctness"],
                    }
                },
                "note": (
                    "A 16 MiB shared-memory buffer is represented in context by "
                    "bounded-slice metadata; no slice payload is inlined."
                ),
                "limitation": None
                if c3_measured
                else "No 16 MiB bounded-slice probe is present in this run.",
            },
            "C4_batching": {
                "status": (
                    "measured"
                    if c4_measured
                    else ("partial" if individual or batched else "not_measured")
                ),
                "individual": None
                if individual is None
                else {
                    "wall_time_seconds": individual["wall_time_seconds"]["median"],
                    "control_tokens": individual["tokens"]["control_tokens"],
                },
                "batched": None
                if batched is None
                else {
                    "wall_time_seconds": batched["wall_time_seconds"]["median"],
                    "control_tokens": batched["tokens"]["control_tokens"],
                },
                "speedup": _reduction(
                    batched["wall_time_seconds"]["median"] if batched else None,
                    individual["wall_time_seconds"]["median"] if individual else None,
                ),
                "note": "One batch submission replaces one submission per item.",
                "limitation": None
                if c4_measured
                else "Batched and individual submissions are not both present.",
            },
            "C5_wall_time": {
                "status": (
                    "measured"
                    if c5_measured
                    else ("partial" if native_by_workload else "not_measured")
                ),
                "by_workload": {
                    workload: {
                        baseline: stats["wall_time_seconds"]["median"]
                        for baseline, stats in table.get(workload, {}).items()
                    }
                    for workload in workloads
                },
                "native_speedup": {
                    workload: _reduction(
                        stats["wall_time_seconds"]["median"],
                        table[workload][primary]["wall_time_seconds"]["median"],
                    )
                    for workload, stats in native_by_workload.items()
                    if (primary := _primary_baseline(table, workload)) is not None
                },
                "note": (
                    "native runs the same computation in-process; LMPU includes "
                    "worker, IPC, and control-plane overhead."
                ),
                "limitation": None
                if c5_measured
                else "A native baseline is missing for at least one workload.",
            },
        },
        "timeline": [
            {
                "workload": record["workload"],
                "baseline": record["baseline"],
                "trial": record["trial"],
                "started_at": record["started_at"],
                "wall_time_seconds": record["wall_time_seconds"],
            }
            for record in sorted(records, key=lambda item: item["started_at"])
        ],
        "limitations": [
            "Trace-token is a labelled simulation and is not a wall-time or energy result.",
            "C2 measures the S1 park window; the treatment performs no model decode by "
            "construction.",
            "C3 inlines derived-buffer metadata only, not any slice payload.",
            "C4 uses 64 pure-function additions; it is control-plane evidence, not "
            "model-quality evidence.",
            "C5 native runs in-process and excludes worker, IPC, and control-plane overhead.",
            "Incorrect or failed baselines are retained; token reductions require "
            "equal task correctness.",
            "Failures and incorrect answers are retained in the table.",
        ],
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable report."""
    manifest = report["manifest"]
    lines = [
        "# Benchmark report",
        "",
        f"- Run: `{report['run_dir']}`",
        f"- Commit: `{manifest['commit']}`",
        f"- Device: `{manifest['hardware'].get('device')}`",
        f"- Model: `{manifest['config'].get('model')}`",
        "",
        "## Results",
        "",
        "| Workload | Baseline | Trials | Correct | Wall median (s) | 95% CI | "
        "Total tokens | GPUs J |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for workload, baselines in report["table"].items():
        for baseline, stats in baselines.items():
            wall = stats["wall_time_seconds"]
            ci = wall["ci95"]
            ci_text = "-" if ci is None else f"{ci[0]:.3f}–{ci[1]:.3f}"
            lines.append(
                f"| {workload} | {baseline} | {stats['trials']} | "
                f"{stats['correctness']['passes']}/{stats['correctness']['total']} | "
                f"{wall['median']:.4f} | {ci_text} | "
                f"{stats['tokens']['total_tokens'] or 0:.0f} | "
                f"{stats['energy']['gpu_energy_joules'] or 0:.2f} |"
            )

    lines += ["", "## Claims", ""]
    for name, claim in report["claims"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Status: **{claim['status']}**")
        if name == "C1_token_reduction" and claim.get("by_workload"):
            lines.append("")
            lines.append("| Workload | Baseline | Baseline tokens | Reduction | Correct |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for workload, baselines in claim["by_workload"].items():
                for baseline, stats in baselines.items():
                    reduction = stats["reduction"]
                    reduction_text = "-" if reduction is None else f"{reduction:.2f}x"
                    lines.append(
                        f"| {workload} | {baseline} | {stats['baseline_total_tokens'] or 0:.0f} | "
                        f"{reduction_text} | {stats['correctness']['passes']}/"
                        f"{stats['correctness']['total']} |"
                    )
        if claim.get("note"):
            lines.append("")
            lines.append(claim["note"])
        if claim.get("limitation"):
            lines.append("")
            lines.append(claim["limitation"])
        lines.append("")

    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    """Write `report.json` and `report.md` into the run directory."""
    report = build_report(run_dir)
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path = run_dir / "report.md"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return markdown_path


def latest_run(root: Path) -> Path:
    """Return the newest harness run directory under ``root``."""
    candidates = list(root.rglob("raw.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No raw.jsonl under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a benchmark report")
    parser.add_argument("run_dir", nargs="?", help="Harness artifact directory")
    parser.add_argument("--latest", action="store_true", help="Use the newest run")
    parser.add_argument(
        "--results-root", default="benchmarks/results", help="Search root for --latest"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run(Path(args.results_root))
    path = write_report(run_dir)
    logger.info("report: %s", path)


if __name__ == "__main__":
    main()
