"""Benchmark runner entry point and async facade."""

from __future__ import annotations

import argparse
import asyncio
import logging

from .baselines import BASELINES, resolve_baselines
from .config import BenchmarkConfig
from .harness import BenchmarkHarness
from .workloads import list_workloads

logger = logging.getLogger(__name__)


async def run_benchmark(
    workload: str | None = "all",
    trials: int | None = None,
    output: str | None = None,
    *,
    baselines: tuple[str, ...] | None = None,
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run a benchmark selection and return the summary payload.

    ``workload=None`` keeps the workloads declared by ``config``; ``"all"``
    selects every registered workload; any other value selects one.
    """
    overrides: dict[str, object] = {
        "trials": trials,
        "results_root": output,
    }
    if baselines is not None:
        overrides["baselines"] = resolve_baselines(baselines)
    if workload == "all":
        overrides["workloads"] = list_workloads()
    elif workload:
        overrides["workloads"] = (workload,)

    resolved = (config or BenchmarkConfig()).with_overrides(**overrides)
    return await BenchmarkHarness(resolved).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run LMPU benchmarks")
    parser.add_argument("--config", help="TOML or JSON benchmark config")
    parser.add_argument("--workload", help="Workload name or 'all'; defaults to config")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=[*BASELINES, "all"],
        help="Baseline name or 'all'; repeatable; defaults to config",
    )
    parser.add_argument("--trials", type=int, help="Measured trials per workload")
    parser.add_argument("--warmups", type=int, help="Discarded warmup trials")
    parser.add_argument("--seed", type=int, help="Base seed")
    parser.add_argument("--device", help="Host device, e.g. cpu or cuda")
    parser.add_argument("--model", help="Reference model name")
    parser.add_argument("--output", help="Results root directory")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume checks")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = BenchmarkConfig.load(args.config) if args.config else BenchmarkConfig()
    config = base.with_overrides(
        trials=args.trials,
        warmups=args.warmups,
        seed=args.seed,
        device=args.device,
        model=args.model,
        results_root=args.output,
        resume=False if args.no_resume else None,
    )
    summary = asyncio.run(
        run_benchmark(
            workload=args.workload,
            baselines=None if args.baseline is None else tuple(args.baseline),
            config=config,
        )
    )

    logger.info("run: %s", summary.get("run_id"))
    for name, workload in summary.get("workloads", {}).items():
        for baseline, stats in workload["baselines"].items():
            wall = stats["wall_time_seconds"]
            logger.info(
                "%s / %s: %s trials, median %.4fs, correctness %s/%s",
                name,
                baseline,
                stats["trials"],
                wall["median"] or 0.0,
                stats["correctness"]["passes"],
                stats["correctness"]["total"],
            )


if __name__ == "__main__":
    main()
