# Benchmark methodology

The harness is in `src/lmpu/bench/`. It records one raw JSONL row per
`(baseline, workload, trial)`. Resume is append-only: a completed key is skipped,
and a truncated final line is ignored.

## Commands

```bash
make bench-smoke   # CPU, one workload, lmpu.sync
make bench         # configs/benchmarks.toml, all workloads and baselines
```

Artifacts are written under `benchmarks/results/<date>/<device>/<commit>/`:

- `manifest.json`: config, commit, dirty flag, platform, GPU.
- `raw.jsonl`: one canonical `lmpu.bench.trial.v1` record per trial.
- `summary.json`: per-baseline medians, bootstrap 95% intervals, and failures.

## Baselines

- **`lmpu.sync`**: submit and wait through one LMPU session; summary-only result.
- **`direct_cot`**: model reasons in natural language with no tool.
- **`trace_token`**: labelled interpreter-driven trace emulation. It is a token
  baseline, not a wall-time or energy baseline.
- **`sync_tool`**: PAL-style program generation, synchronous execution, full
  result ingestion, then a final model turn.
- **`async_poll`**: program submission plus status polling and small result
  ingestion. Poll counts are recorded in trial metadata.

`configs/benchmarks.toml` fixes the seed, model, token limit, device, and trial
counts. Trial order is shuffled per trial with a seed derived from the base seed.

## Workloads

- `bigint_mul`: two 50-digit operands, exact integer result.
- `matmul`: 2048 x 2048 deterministic float32 matrix product.
- `large_context`: find a marker in a 40k+ character document.
- `sleep`: a 12-second blocking `python.vm` job.

## Energy

`energy.EnergyProbe` samples GPU power through NVML. `idle_seconds` sets the
pre-trial idle window. The measured idle rate is stored on every record as
`metadata.idle`, and measured GPU energy is compared against the idle-rate
estimate in `metadata.energy`.

The canonical `gpu_energy_joules` field is the measured value for the trial.
The energy baseline is not subtracted from the stored measurement because the
raw value remains auditable; consumers can use `metadata.energy.corrected_joules`.

RAPL CPU energy is not measured by this harness. When NVML is unavailable,
`metadata.idle.available` is false and zero GPU joules means "not measured",
not "no energy". S1 sleep retains KV memory and is near-idle, not zero energy.
