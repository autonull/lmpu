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
- `report.json` / `report.md`: C1–C5 tables, CDFs, timeline, and limitations.

Run `make report` to render the newest artifact directory. Published reports live
in `benchmarks/reports/`:

- `gate-b-initial`: first C1–C5 pass with C2–C5 limitations.
- `gate-b-evidence`: measured C2 (S1 zero forwards), C3 (16 MiB bounded slice),
  C4 (batch vs individual), and C5 (native baseline). C1 remains partial
  because the small-budget model baselines fail correctness; the report keeps
  those failures instead of hiding them.

Run settings are intentionally small for reproducibility: 512² matmul, 12k
context, 3 s sleep, 64-token default budget. Increase them explicitly for a
paper run.

## Baselines

- **`lmpu.sync`**: submit and wait through one LMPU session; summary-only result.
  Mechanism-only; not an end-to-end arm.
- **`lmpu.sleep`**: submit asynchronously, register an S1 sleep, park for the job
  window, and wake. The treatment performs no decode, so
  `forward_passes` is zero while parked; park duration is recorded in
  `metadata.park`. Mechanism-only.
- **`lmpu.slice`**: import a 16 MiB shared-memory buffer and read it through the
  bounded-slice path. Only derived-buffer metadata enters context.
  Mechanism-only.
- **`lmpu.individual`**: submit each batch item as its own synchronous job.
  Mechanism-only.
- **`lmpu.batch`**: submit all batch items in one job. Mechanism-only.
- **`native`**: run the same workload computation in-process, without the
  runtime, for C5 overhead comparison. Mechanism-only.
- **`direct_cot`**: model reasons in natural language with no tool. End-to-end.
- **`trace_token`**: labelled interpreter-driven trace emulation. It is a token
  simulation, quarantined from every claim table by `report.py`.
- **`sync_tool_full_result`**: PAL-style program generation, synchronous execution, full
  result ingestion, then a final model turn scored on the decoded answer.
  End-to-end.
- **`async_poll_programmatic`**: program submission plus programmatic status
  polling, bounded-result ingestion, and a final model turn scored on the
  decoded answer. End-to-end.
- **`async_poll_model`**: model-in-the-loop polling; each status check is a
  model turn, bounded-result ingestion, answer scored on the decoded answer.
  End-to-end.
- **`session_reopen`**: async job, then a fresh session that re-prefills the
  full context for the answer turn (no KV reuse). End-to-end.
- **`app_agent`**: app-level agent loop without the LMPU runtime; the tool runs
  in-process, the session stays KV-warm via the engine prefix cache, only a
  bounded result summary enters context, and polling is programmatic.
  End-to-end.
- **`engine_native`**: engine-native session reuse without the LMPU protocol;
  blocking in-process call, KV-warm session, bounded result. End-to-end.

Every model baseline decodes a final answer and is scored on it. A correct
tool result with a wrong final answer is `correct=False`; both `tool_correct`
and the decoded `answer` are retained in trial metadata. Failed trials keep
accumulated token accounting.

`configs/benchmarks.toml` fixes the seed, model, token limit, device, and trial
counts. Trial order is shuffled per trial with a seed derived from the base seed.

## Workloads

- `bigint_mul`: two 50-digit operands, exact integer result.
- `matmul`: 512 x 512 deterministic float32 matrix product.
- `large_context`: find a marker in a 12k+ character document.
- `sleep`: a 3-second blocking `python.vm` job.
- `large_result`: a 16 MiB buffer read through a bounded slice (C3).
- `batch`: 64 small additions submitted individually and as one batch (C4).

Each workload declares its applicable baselines, so a run never attempts a
model baseline on a non-model workload. `native` runs the same computation
in-process for C5.

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
