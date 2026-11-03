"""Render benchmark reports from raw harness artifacts.

Two sections, never mixed (V0.5):

- ``mechanism``: lmpu.* instrumentation and the native control (claims C2-C5).
  Mechanism evidence only; not an end-to-end claim.
- ``end_to_end``: host-model baselines (claim E). Every arm is scored on its
  decoded final answer.

Simulated records (``metadata.simulated``) are quarantined from every claim
table (V0.3); they remain visible in the raw table, flagged.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .taxonomy import (
    END_TO_END_BASELINES,
    TREATMENT_PREFIX,
    baseline_kind,
    record_is_simulated,
)

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


def _per_correct_task(trials: list[dict[str, Any]]) -> dict[str, float | None]:
    """Total cost per correct task; failures are paid for, not skipped (V0.4)."""
    correct = sum(1 for record in trials if record["correct"])
    if not correct:
        return {
            "total_tokens": None,
            "forward_passes": None,
            "wall_time_seconds": None,
            "gpu_joules": None,
        }

    def total(key: str) -> float:
        return sum(float(record["metrics"].get(key, 0.0)) for record in trials)

    return {
        "total_tokens": total("total_tokens") / correct,
        "forward_passes": total("forward_passes") / correct,
        "wall_time_seconds": sum(float(record["wall_time_seconds"]) for record in trials) / correct,
        "gpu_joules": total("gpu_energy_joules") / correct,
    }


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
        "token_cdf": {
            key: _cdf([float(r["metrics"][key]) for r in trials if key in r["metrics"]])
            for key in TOKEN_KEYS
        },
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
        "simulated": any(record_is_simulated(record) for record in trials),
        "per_correct_task": _per_correct_task(trials),
        "phases": _phases_summary(trials),
    }


def _phases_summary(trials: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for record in trials:
        for name, seconds in (record.get("metadata", {}).get("phases") or {}).items():
            totals[name] += float(seconds)
    return dict(totals) if totals else {}


def _paired_delta(
    treatment_trials: list[dict[str, Any]],
    reference_trials: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any] | None:
    """Paired per-trial differences (treatment minus reference) on shared trials."""
    reference = {record["trial"]: record for record in reference_trials}
    token_deltas: list[float] = []
    wall_deltas: list[float] = []
    for record in treatment_trials:
        other = reference.get(record["trial"])
        if other is None:
            continue
        token_deltas.append(
            float(record["metrics"]["total_tokens"]) - float(other["metrics"]["total_tokens"])
        )
        wall_deltas.append(float(record["wall_time_seconds"]) - float(other["wall_time_seconds"]))
    if len(token_deltas) < 2:
        return None
    return {
        "n": len(token_deltas),
        "total_tokens_mean_delta": statistics.median(token_deltas),
        "total_tokens_delta_ci95": _ci95(token_deltas, seed),
        "wall_time_mean_delta": statistics.median(wall_deltas),
        "wall_time_delta_ci95": _ci95(wall_deltas, seed + 1),
    }


def _reduction(numerator: float | None, denominator: float | None) -> float | None:
    if not numerator or not denominator:
        return None
    return denominator / numerator


def _e_claim(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    treatments: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for workload, workload_cells in cells.items():
        for baseline, stats in workload_cells.items():
            if baseline.startswith(TREATMENT_PREFIX):
                treatments[workload].append((baseline, stats))
    if not treatments:
        return {
            "status": "not_measured",
            "note": (
                "No lmpu.model_* treatment arm is present; Batch 1 adds "
                "lmpu.model_async/lmpu.model_sync before the end-to-end claim "
                "can be evaluated."
            ),
        }

    verdicts = []
    for workload, arms in treatments.items():
        for arm, stats in arms:
            competitors = {
                baseline: cell
                for baseline, cell in cells[workload].items()
                if baseline != arm and baseline in END_TO_END_BASELINES
            }
            correct_competitors = [
                cell for cell in competitors.values() if cell["correctness"]["rate"] == 1.0
            ]
            treatment_ok = stats["correctness"]["rate"] == 1.0
            # Hardest to beat: the correct competitor with the fewest tokens.
            strongest = (
                min((cell["tokens"]["total_tokens"] or 0.0) for cell in correct_competitors)
                if correct_competitors
                else None
            )
            # Reduction is reference-over-treatment; >=1 means the treatment wins.
            reduction = (
                _reduction(stats["tokens"]["total_tokens"] or None, strongest)
                if strongest
                else None
            )
            verdicts.append(
                {
                    "workload": workload,
                    "treatment": arm,
                    "treatment_correct": treatment_ok,
                    "strongest_correct_competitor_tokens": strongest,
                    "token_reduction_vs_strongest_correct": reduction,
                }
            )
    if verdicts and all(
        item["treatment_correct"]
        and item["token_reduction_vs_strongest_correct"] is not None
        and item["token_reduction_vs_strongest_correct"] >= 1.1
        for item in verdicts
    ):
        status = "measured"
    elif any(item["treatment_correct"] for item in verdicts):
        status = "partial"
    else:
        status = "not_measured"
    return {"status": status, "arms": verdicts}


def build_report(run_dir: Path) -> dict[str, Any]:
    """Build a report dictionary from a harness artifact directory."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped = _group(records)
    quarantined = [record for record in records if record_is_simulated(record)]
    claim_records = [record for record in records if not record_is_simulated(record)]
    claim_grouped = _group(claim_records)
    baselines = manifest["config"].get("baselines", [])
    workloads = manifest["config"].get("workloads", [])

    table: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        table[workload] = {}
        for baseline in baselines:
            trials = grouped.get((workload, baseline), [])
            if trials:
                table[workload][baseline] = _stats(trials, len(trials) * 7919)

    claim_cells: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        claim_cells[workload] = {}
        for baseline in baselines:
            trials = claim_grouped.get((workload, baseline), [])
            if trials:
                claim_cells[workload][baseline] = _stats(trials, len(trials) * 7919)

    end_to_end_table = {
        workload: {
            baseline: stats
            for baseline, stats in cells.items()
            if baseline_kind(baseline) == "end_to_end"
        }
        for workload, cells in claim_cells.items()
    }

    paired: dict[str, dict[str, Any]] = {}
    reference = "direct_cot"
    for workload, cells in claim_cells.items():
        paired[workload] = {}
        reference_trials = claim_grouped.get((workload, reference), [])
        for baseline in cells:
            if baseline == reference or baseline_kind(baseline) != "end_to_end":
                continue
            trials = claim_grouped.get((workload, baseline), [])
            delta = _paired_delta(trials, reference_trials, len(trials) * 7919)
            if delta is not None:
                paired[workload][baseline] = delta

    sleep = claim_cells.get("sleep", {})
    sleep_treatment = sleep.get("lmpu.sleep")
    large_result = claim_cells.get("large_result", {})
    slice_stats = large_result.get("lmpu.slice")
    batch = claim_cells.get("batch", {})
    individual = batch.get("lmpu.individual")
    batched = batch.get("lmpu.batch")
    native_by_workload = {
        workload: cells["native"] for workload, cells in claim_cells.items() if "native" in cells
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
        "schema": "lmpu.bench.report.v2",
        "run_dir": str(run_dir),
        "manifest": manifest,
        "table": table,
        "simulation_quarantine": {
            "excluded_records": len(quarantined),
            "baselines": sorted({record["baseline"] for record in quarantined}),
            "note": (
                "trace_token is a labelled simulation; it cannot appear in any "
                "claim table. Its records remain in the raw table, flagged."
            ),
        },
        "mechanism": {
            "note": (
                "Mechanism-only evidence (S1 park, bounded slices, batching, "
                "overhead). Not an end-to-end efficiency claim."
            ),
            "claims": {
                "C2_sleep_proof": {
                    "status": (
                        "measured" if c2_measured else ("partial" if sleep else "not_measured")
                    ),
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
                        "measured"
                        if c3_measured
                        else ("partial" if slice_stats else "not_measured")
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
                            for baseline, stats in cells.items()
                        }
                        for workload, cells in claim_cells.items()
                    },
                    "native_speedup": {
                        workload: _reduction(
                            stats["wall_time_seconds"]["median"],
                            claim_cells[workload][primary]["wall_time_seconds"]["median"],
                        )
                        for workload, stats in native_by_workload.items()
                        if (primary := _primary_baseline(claim_cells, workload)) is not None
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
        },
        "end_to_end": {
            "note": (
                "Host model in the loop; every arm is scored on its decoded final "
                "answer. Token reductions require equal task correctness."
            ),
            "reference": reference,
            "cells": end_to_end_table,
            "paired_vs_reference": paired,
            "claims": {"E_engine_integration": _e_claim(end_to_end_table)},
        },
        "timeline": [
            {
                "workload": record["workload"],
                "baseline": record["baseline"],
                "trial": record["trial"],
                "started_at": record["started_at"],
                "wall_time_seconds": record["wall_time_seconds"],
                "simulated": record_is_simulated(record),
            }
            for record in sorted(records, key=lambda item: item["started_at"])
        ],
        "limitations": [
            "Trace-token is a labelled simulation; it cannot support any claim.",
            "C2 measures the S1 park window; the treatment performs no model decode "
            "by construction.",
            "C3 inlines derived-buffer metadata only, not any slice payload.",
            "C4 uses 64 pure-function additions; it is control-plane evidence, not "
            "model-quality evidence.",
            "C5 native runs in-process and excludes worker, IPC, and control-plane overhead.",
            "Per-correct-task cost divides total spend by correct tasks; failures "
            "are paid for, not skipped.",
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
        "| Workload | Baseline | Kind | Trials | Correct | Wall median (s) | 95% CI | "
        "Total tokens | Tokens/correct | GPUs J |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for workload, baselines in report["table"].items():
        for baseline, stats in baselines.items():
            wall = stats["wall_time_seconds"]
            ci = wall["ci95"]
            ci_text = "-" if ci is None else f"{ci[0]:.3f}–{ci[1]:.3f}"
            per_task = stats.get("per_correct_task", {}).get("total_tokens")
            per_task_text = "-" if per_task is None else f"{per_task:.0f}"
            flag = " (simulated)" if stats.get("simulated") else ""
            lines.append(
                f"| {workload} | {baseline}{flag} | {baseline_kind(baseline)} | "
                f"{stats['trials']} | "
                f"{stats['correctness']['passes']}/{stats['correctness']['total']} | "
                f"{wall['median']:.4f} | {ci_text} | "
                f"{stats['tokens']['total_tokens'] or 0:.0f} | {per_task_text} | "
                f"{stats['energy']['gpu_energy_joules'] or 0:.2f} |"
            )

    lines += ["", "## Mechanism claims", "", report["mechanism"]["note"], ""]
    for name, claim in report["mechanism"]["claims"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Status: **{claim['status']}**")
        if name == "C5_wall_time" and claim.get("native_speedup"):
            lines.append("")
            lines.append("| Workload | native/treatment wall ratio |")
            lines.append("| --- | ---: |")
            for workload, ratio in claim["native_speedup"].items():
                ratio_text = "-" if ratio is None else f"{ratio:.2f}x"
                lines.append(f"| {workload} | {ratio_text} |")
        if claim.get("note"):
            lines.append("")
            lines.append(claim["note"])
        if claim.get("limitation"):
            lines.append("")
            lines.append(claim["limitation"])
        lines.append("")

    e2e = report["end_to_end"]
    lines += ["", "## End-to-end results", "", e2e["note"], ""]
    lines.append(f"Reference: `{e2e['reference']}` (paired trials, same seed and order).")
    lines.append("")
    for workload, cells in e2e["cells"].items():
        if not cells:
            continue
        lines.append(f"### {workload}")
        lines.append("")
        lines.append(
            "| Baseline | Trials | Correct | Total tokens | Tokens/correct | "
            "Paired Δ tokens (median) |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for baseline, stats in cells.items():
            per_task = stats.get("per_correct_task", {}).get("total_tokens")
            per_task_text = "-" if per_task is None else f"{per_task:.0f}"
            delta = e2e["paired_vs_reference"].get(workload, {}).get(baseline)
            delta_text = "-" if delta is None else f"{delta['total_tokens_mean_delta']:.0f}"
            lines.append(
                f"| {baseline} | {stats['trials']} | "
                f"{stats['correctness']['passes']}/{stats['correctness']['total']} | "
                f"{stats['tokens']['total_tokens'] or 0:.0f} | {per_task_text} | {delta_text} |"
            )
        lines.append("")
    claim = e2e["claims"]["E_engine_integration"]
    lines.append(f"### E_engine_integration — status: **{claim['status']}**")
    lines.append("")
    if claim.get("note"):
        lines.append(claim["note"])
    lines.append("")

    quarantine = report["simulation_quarantine"]
    lines += [
        "## Simulation quarantine",
        "",
        f"Excluded records: {quarantine['excluded_records']} "
        f"(baselines: {', '.join(quarantine['baselines']) or 'none'}).",
        "",
        quarantine["note"],
        "",
        "## Limitations",
        "",
    ]
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
