"""Addressable-cost ceiling analysis (V0.8; plan section 0.1).

LMPU can only reduce result ingestion, polling/wait, re-prefill, and idle host
compute. This module decomposes harness artifacts per baseline and computes

    ceiling_fraction = (ingestion + polling + re-prefill + idle) / total

on the token axis for cells that carry the needed accounting, and reports
every unmeasurable component explicitly instead of estimating it. Run against
existing artifacts before any new GPU matrix; a short instrumented pilot is
allowed when they lack per-phase totals.

CLI: ``uv run python -m lmpu.bench.ceiling <run_dir>`` (writes ``ceiling.json``
next to ``raw.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PHASE_KEYS = (
    "prefill_tokens",
    "generated_tokens",
    "control_tokens",
    "injected_tokens",
    "total_tokens",
    "forward_passes",
    "gpu_energy_joules",
    "sleep_duration_seconds",
    "wake_latency_seconds",
    "wall_time_seconds",
)


def decompose(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-cell (workload/baseline) host cost, splitting measurable phases."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["workload"], record["baseline"])].append(record)

    cells: dict[str, Any] = {}
    for (workload, baseline), trials in sorted(grouped.items()):
        totals = {}
        for key in PHASE_KEYS:
            values = [
                float(record["metrics"][key])
                for record in trials
                if key in record.get("metrics", {})
            ]
            if values:
                totals[key] = {"sum": sum(values), "median": statistics.median(values)}
        phases: dict[str, float] = defaultdict(float)
        has_per_phase = False
        for record in trials:
            for name, seconds in (record.get("metadata", {}).get("phases") or {}).items():
                phases[name] += float(seconds)
                has_per_phase = True
        correct = sum(1 for record in trials if record["correct"])
        cells[f"{workload}/{baseline}"] = {
            "workload": workload,
            "baseline": baseline,
            "trials": len(trials),
            "correct": correct,
            "totals": totals,
            "phases": dict(phases),
            "has_per_phase_instrumentation": has_per_phase,
        }
    return cells


def ceiling(cells: dict[str, Any]) -> dict[str, Any]:
    """Compute the addressable fraction per baseline.

    Token-axis decomposition only. ``re_prefill`` is measurable only for cells
    with per-phase instrumentation (this batch adds it); ``idle`` is reported
    from recorded park windows. Anything else is reported as not_measured.
    """
    results: dict[str, Any] = {}
    for cell_key, cell in cells.items():
        total = cell["totals"].get("total_tokens", {}).get("sum")
        ingestion = cell["totals"].get("injected_tokens", {}).get("sum")
        control = cell["totals"].get("control_tokens", {}).get("sum")
        if not total:
            results[cell_key] = {
                "status": "not_measured",
                "reason": "no token accounting",
                "workload": cell["workload"],
                "baseline": cell["baseline"],
            }
            continue
        addressable = (ingestion or 0.0) + (control or 0.0)
        re_prefill: float | None = None
        if cell["has_per_phase_instrumentation"]:
            re_prefill = cell["phases"].get("reprefill")
        addressable_measured = addressable + (re_prefill or 0.0)
        results[cell_key] = {
            "status": "measured",
            "workload": cell["workload"],
            "baseline": cell["baseline"],
            "correct": cell["correct"],
            "trials": cell["trials"],
            "total_tokens": total,
            "ingestion_tokens": ingestion or 0.0,
            "control_tokens": control or 0.0,
            "reprefill_seconds": re_prefill,
            "addressable_tokens": addressable_measured,
            "addressable_fraction": addressable_measured / total,
            "idle_seconds": cell["totals"].get("sleep_duration_seconds", {}).get("sum"),
            "polling_note": (
                "control_tokens is used as the polling/wait proxy; they include "
                "submission overhead and cannot be split further until the "
                "Batch 1 instrumented run"
            ),
            "reprefill_note": (
                "re-prefill tokens are not separable from initial prefill in "
                "artifacts without per-phase instrumentation"
            ),
        }
    return results


def build_ceiling_report(run_dir: Path) -> dict[str, Any]:
    """Build the ceiling report from a harness artifact directory."""
    raw_path = run_dir / "raw.jsonl"
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells = decompose(records)
    return {
        "schema": "lmpu.bench.ceiling.v1",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "baselines": cells,
        "ceiling": ceiling(cells),
        "verdict_rule": (
            "Record FAIL for the performance thesis only after the ceiling is "
            "computed on the best-case workload (plan 0.1). Artifacts without "
            "the best-case cell are reported as inconclusive, and the decisive "
            "number comes from the Batch 1 instrumented pilot."
        ),
        "best_case_measured": False,
        "verdict": "inconclusive",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute the addressable-cost ceiling")
    parser.add_argument("run_dir", help="Harness artifact directory containing raw.jsonl")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_dir = Path(args.run_dir)
    report = build_ceiling_report(run_dir)
    out = run_dir / "ceiling.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for cell_key, cell in sorted(report["ceiling"].items()):
        if cell.get("status") == "measured":
            logger.info(
                "%s: correct %d/%d, addressable %.1f%% of %.0f tokens",
                cell_key,
                cell["correct"],
                cell["trials"],
                100 * cell["addressable_fraction"],
                cell["total_tokens"],
            )
        else:
            logger.info("%s: %s", cell_key, cell.get("reason"))
    logger.info("ceiling report: %s", out)


if __name__ == "__main__":
    main()
