"""Energy measurement (R2/P6): Whole-trial NVML power via nvidia-smi dmon.

Measures:
  - Idle baseline (runtime up, no sessions)
  - Single session active (dataplane job)
  - Multiple parked sessions (4, 8)
  - Multiple active sessions (4, 8)

Reports median ± IQR over N trials. Only claims deltas outside IQR.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from lmpu.kernel.runtime import Runtime


@dataclass
class TrialResult:
    label: str
    duration_s: float
    power_samples: list[float]  # watts per sample
    median_w: float
    q1_w: float
    q3_w: float
    iqr_w: float
    energy_j: float  # median_w * duration_s


@dataclass
class ExperimentConfig:
    trial_duration_s: float = 10.0
    num_trials: int = 20
    dmon_interval_ms: int = 100
    gpu_index: int = 0


class EnergyExperiment:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self.results: list[TrialResult] = []

    def _start_dmon(self) -> subprocess.Popen[bytes]:
        """Start nvidia-smi dmon in background, return Popen object."""
        cmd = [
            "nvidia-smi",
            "dmon",
            "-s", "p",  # power only
            "-i", str(self.config.gpu_index),
            "-d", str(self.config.dmon_interval_ms),
            "-o", "T",  # timestamp
        ]
        # Redirect output to PIPE so we can read it
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read_dmon_power(self, proc: subprocess.Popen[bytes]) -> list[float]:
        """Read power samples from dmon process."""
        power_samples = []
        if proc.stdout:
            for line in proc.stdout:
                line = line.decode().strip()
                if line.startswith("#") or not line:
                    continue
                # Format: timestamp gpu_id pwr
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        power_samples.append(float(parts[2]))
                    except ValueError:
                        pass
        return power_samples

    async def _run_idle_trial(self) -> TrialResult:
        """Runtime up, no sessions."""
        dmon = self._start_dmon()
        await asyncio.sleep(0.5)  # let dmon warm up
        t0 = time.perf_counter()
        await asyncio.sleep(self.config.trial_duration_s)
        t1 = time.perf_counter()
        dmon.terminate()
        dmon.wait(timeout=2)

        power = self._read_dmon_power(dmon)
        return self._make_result("idle", t1 - t0, power)

    async def _run_single_active_trial(self, tmpdir: Path) -> TrialResult:
        """One session doing a dataplane job."""
        rt = Runtime(
            socket_path=str(tmpdir / "run.sock"),
            shm_dir=str(tmpdir / "shm"),
            config={"trace_dir": str(tmpdir / "traces")},
        )
        await rt.start()
        hello = await rt._hello({"protocol": "lmpu/1"})
        sid = hello.session_id

        dmon = self._start_dmon()
        await asyncio.sleep(0.5)
        t0 = time.perf_counter()

        # Run a moderate dataplane job
        data = np.random.default_rng(42).bytes(64 * 1024 * 1024)  # 64 MiB
        ref = await rt._buffer_import(
            {"name": "source", "data": data, "dtype": "uint8", "shape": [64 * 1024 * 1024]}, sid
        )
        hash_ref = await rt._buffer_create(
            {"name": "hash", "dtype": "uint8", "shape": [64]}, sid
        )
        await rt._submit(
            {
                "processor": "pure.func",
                "op": "sha256",
                "inputs": {"data": {"buffer": ref.buffer_id}},
                "outputs": {"hash": {"buffer": hash_ref.buffer_id}},
                "config": {"mode": "sync", "timeout_ms": 60_000},
            },
            sid,
        )

        t1 = time.perf_counter()
        dmon.terminate()
        dmon.wait(timeout=2)

        await rt._close({"detach": False}, sid)
        await rt.stop()

        power = self._read_dmon_power(dmon)
        return self._make_result("single_active", t1 - t0, power)

    async def _run_parked_trial(self, tmpdir: Path, num_sessions: int) -> TrialResult:
        """Multiple sessions parked (created but idle)."""
        rt = Runtime(
            socket_path=str(tmpdir / "run.sock"),
            shm_dir=str(tmpdir / "shm"),
            config={"trace_dir": str(tmpdir / "traces")},
        )
        await rt.start()

        sessions = []
        for _ in range(num_sessions):
            hello = await rt._hello({"protocol": "lmpu/1"})
            sessions.append(hello.session_id)

        dmon = self._start_dmon()
        await asyncio.sleep(0.5)
        t0 = time.perf_counter()
        await asyncio.sleep(self.config.trial_duration_s)
        t1 = time.perf_counter()
        dmon.terminate()
        dmon.wait(timeout=2)

        for sid in sessions:
            await rt._close({"detach": False}, sid)
        await rt.stop()

        power = self._read_dmon_power(dmon)
        return self._make_result(f"parked_{num_sessions}", t1 - t0, power)

    async def _run_active_trial(self, tmpdir: Path, num_sessions: int) -> TrialResult:
        """Multiple sessions active (concurrent jobs)."""
        rt = Runtime(
            socket_path=str(tmpdir / "run.sock"),
            shm_dir=str(tmpdir / "shm"),
            config={"trace_dir": str(tmpdir / "traces")},
        )
        await rt.start()

        sessions = []
        for _ in range(num_sessions):
            hello = await rt._hello({"protocol": "lmpu/1"})
            sessions.append(hello.session_id)

        dmon = self._start_dmon()
        await asyncio.sleep(0.5)
        t0 = time.perf_counter()

        # Launch concurrent jobs
        tasks = []
        for sid in sessions:
            data = np.random.default_rng(42).bytes(16 * 1024 * 1024)  # 16 MiB each
            ref = await rt._buffer_import(
                {"name": "source", "data": data, "dtype": "uint8", "shape": [16 * 1024 * 1024]}, sid
            )
            hash_ref = await rt._buffer_create(
                {"name": "hash", "dtype": "uint8", "shape": [64]}, sid
            )
            task = rt._submit(
                {
                    "processor": "pure.func",
                    "op": "sha256",
                    "inputs": {"data": {"buffer": ref.buffer_id}},
                    "outputs": {"hash": {"buffer": hash_ref.buffer_id}},
                    "config": {"mode": "sync", "timeout_ms": 60_000},
                },
                sid,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        t1 = time.perf_counter()
        dmon.terminate()
        dmon.wait(timeout=2)

        for sid in sessions:
            await rt._close({"detach": False}, sid)
        await rt.stop()

        power = self._read_dmon_power(dmon)
        return self._make_result(f"active_{num_sessions}", t1 - t0, power)

    def _make_result(self, label: str, duration: float, power_samples: list[float]) -> TrialResult:
        if not power_samples:
            return TrialResult(label, duration, [], 0, 0, 0, 0, 0)

        median_w = statistics.median(power_samples)
        q1_w = statistics.quantiles(power_samples, n=4)[0]
        q3_w = statistics.quantiles(power_samples, n=4)[2]
        iqr_w = q3_w - q1_w
        energy_j = median_w * duration

        return TrialResult(
            label=label,
            duration_s=duration,
            power_samples=power_samples,
            median_w=median_w,
            q1_w=q1_w,
            q3_w=q3_w,
            iqr_w=iqr_w,
            energy_j=energy_j,
        )

    async def run_all(self, output_dir: Path) -> list[TrialResult]:
        all_results = []

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            # Run each scenario for N trials
            scenarios = [
                ("idle", self._run_idle_trial, ()),
                ("single_active", self._run_single_active_trial, (tmpdir,)),
                ("parked_4", self._run_parked_trial, (tmpdir, 4)),
                ("parked_8", self._run_parked_trial, (tmpdir, 8)),
                ("active_4", self._run_active_trial, (tmpdir, 4)),
                ("active_8", self._run_active_trial, (tmpdir, 8)),
            ]

            for label, fn, args in scenarios:
                print(f"\n=== {label} ===")
                trial_results = []

                for trial in range(self.config.num_trials):
                    print(f"  Trial {trial + 1}/{self.config.num_trials}...", end=" ", flush=True)
                    result = await fn(*args)
                    trial_results.append(result)
                    print(f"median={result.median_w:.1f}W IQR={result.iqr_w:.1f}W")

                # Aggregate across trials
                medians = [r.median_w for r in trial_results]
                overall_median = statistics.median(medians)
                overall_q1 = statistics.quantiles(medians, n=4)[0]
                overall_q3 = statistics.quantiles(medians, n=4)[2]
                overall_iqr = overall_q3 - overall_q1
                overall_energy = overall_median * statistics.median([r.duration_s for r in trial_results])

                agg = TrialResult(
                    label=label,
                    duration_s=statistics.median([r.duration_s for r in trial_results]),
                    power_samples=[],
                    median_w=overall_median,
                    q1_w=overall_q1,
                    q3_w=overall_q3,
                    iqr_w=overall_iqr,
                    energy_j=overall_energy,
                )
                all_results.append(agg)

                # Save raw trials
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / f"{label}_trials.jsonl", "w") as f:
                    for r in trial_results:
                        f.write(json.dumps(asdict(r)) + "\n")

        return all_results


def print_summary(results: list[TrialResult]) -> None:
    print("\n" + "=" * 80)
    print("ENERGY SUMMARY (median ± IQR over trials)")
    print("=" * 80)
    print(f"{'Scenario':<15} {'Median (W)':>14} {'Q1 (W)':>10} {'Q3 (W)':>10} {'IQR (W)':>10} {'Energy (J)':>12}")
    print("-" * 80)

    idle = next((r for r in results if r.label == "idle"), None)

    for r in results:
        delta = ""
        if idle and r.label != "idle":
            diff = r.median_w - idle.median_w
            if abs(diff) > idle.iqr_w:
                delta = f"  Δ={diff:+.1f}W*"
            else:
                delta = f"  Δ={diff:+.1f}W (inside IQR)"
        print(f"{r.label:<15} {r.median_w:>10.1f} ± {r.iqr_w:>5.1f}  {r.q1_w:>10.1f}  {r.q3_w:>10.1f}  {r.iqr_w:>10.1f}  {r.energy_j:>10.1f}{delta}")

    print("-" * 80)
    print("* = delta outside idle IQR (statistically meaningful)")
    print()

    # Key claim: idle watts per parked session ≈ 0
    if idle:
        parked_4 = next((r for r in results if r.label == "parked_4"), None)
        parked_8 = next((r for r in results if r.label == "parked_8"), None)
        if parked_4:
            per_session_4 = (parked_4.median_w - idle.median_w) / 4
            per_session_8 = (parked_8.median_w - idle.median_w) / 8 if parked_8 else 0
            print(f"Idle watts per parked session:")
            print(f"  4 sessions: {per_session_4:.2f} W/session (Δ vs idle: {parked_4.median_w - idle.median_w:.1f}W)")
            if parked_8:
                print(f"  8 sessions: {per_session_8:.2f} W/session (Δ vs idle: {parked_8.median_w - idle.median_w:.1f}W)")


async def main() -> None:
    # Check nvidia-smi dmon availability
    try:
        subprocess.run(["nvidia-smi", "dmon", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: nvidia-smi dmon not available")
        return

    config = ExperimentConfig(
        trial_duration_s=5.0,
        num_trials=10,
        dmon_interval_ms=200,
    )

    output_dir = Path("benchmarks/results/energy")
    exp = EnergyExperiment(config)
    results = await exp.run_all(output_dir)

    print_summary(results)

    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())