# Benchmark report

- Run: `benchmarks/results/l0-pilot/2026-11-03/NVIDIA-GeForce-RTX-3080/cb000e530401`
- Commit: `cb000e530401`
- Device: `cuda`
- Model: `Qwen/Qwen3.5-0.8B`

## Results

| Workload | Baseline | Kind | Trials | Correct | Wall median (s) | 95% CI | Total tokens | Tokens/correct | GPUs J |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| sleep | direct_cot | end_to_end | 3 | 3/3 | 10.0095 | 9.979–10.067 | 279 | 279 | 969.71 |
| sleep | sync_tool_full_result | end_to_end | 3 | 0/3 | 9.8510 | 9.753–9.923 | 305 | - | 959.53 |
| sleep | async_poll_programmatic | end_to_end | 3 | 0/3 | 10.0727 | 9.978–10.164 | 305 | - | 986.91 |
| sleep | async_poll_model | end_to_end | 3 | 0/3 | 9.8965 | 9.873–9.942 | 305 | - | 973.63 |
| sleep | session_reopen | end_to_end | 3 | 0/3 | 9.9631 | 9.953–10.060 | 305 | - | 978.15 |
| sleep | app_agent | end_to_end | 3 | 0/3 | 9.8966 | 9.807–9.952 | 305 | - | 967.84 |
| sleep | engine_native | end_to_end | 3 | 0/3 | 9.9104 | 9.872–10.023 | 305 | - | 970.02 |
| bigint_mul | direct_cot | end_to_end | 3 | 0/3 | 9.9421 | 9.778–9.943 | 340 | - | 966.24 |
| bigint_mul | sync_tool_full_result | end_to_end | 3 | 0/3 | 9.9617 | 9.959–10.300 | 427 | - | 0.00 |
| bigint_mul | async_poll_programmatic | end_to_end | 3 | 0/3 | 10.2715 | 10.271–21.957 | 540 | - | 0.00 |
| bigint_mul | async_poll_model | end_to_end | 3 | 0/3 | 12.6953 | 12.448–12.774 | 567 | - | 1235.76 |
| bigint_mul | session_reopen | end_to_end | 3 | 0/3 | 19.7496 | 19.713–20.264 | 805 | - | 964.30 |
| bigint_mul | app_agent | end_to_end | 3 | 0/3 | 10.0330 | 9.854–10.055 | 366 | - | 962.04 |
| bigint_mul | engine_native | end_to_end | 3 | 0/3 | 9.8858 | 9.834–10.147 | 366 | - | 976.54 |

## Mechanism claims

Mechanism-only evidence (S1 park, bounded slices, batching, overhead). Not an end-to-end efficiency claim.

### C2_sleep_proof

Status: **partial**

The lmpu.sleep treatment parks the host in S1 for the measured job window and performs zero forward passes while parked.

No lmpu.sleep/S1 treatment is present in this run.

### C3_context_protection

Status: **not_measured**

A 16 MiB shared-memory buffer is represented in context by bounded-slice metadata; no slice payload is inlined.

No 16 MiB bounded-slice probe is present in this run.

### C4_batching

Status: **not_measured**

One batch submission replaces one submission per item.

Batched and individual submissions are not both present.

### C5_wall_time

Status: **not_measured**

native runs the same computation in-process; LMPU includes worker, IPC, and control-plane overhead.

A native baseline is missing for at least one workload.


## End-to-end results

Host model in the loop; every arm is scored on its decoded final answer. Token reductions require equal task correctness.

Reference: `direct_cot` (paired trials, same seed and order).

### sleep

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_cot | 3 | 3/3 | 279 | 279 | - |
| sync_tool_full_result | 3 | 0/3 | 305 | - | 26 |
| async_poll_programmatic | 3 | 0/3 | 305 | - | 26 |
| async_poll_model | 3 | 0/3 | 305 | - | 26 |
| session_reopen | 3 | 0/3 | 305 | - | 26 |
| app_agent | 3 | 0/3 | 305 | - | 26 |
| engine_native | 3 | 0/3 | 305 | - | 26 |

### bigint_mul

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_cot | 3 | 0/3 | 340 | - | - |
| sync_tool_full_result | 3 | 0/3 | 427 | - | 87 |
| async_poll_programmatic | 3 | 0/3 | 540 | - | 200 |
| async_poll_model | 3 | 0/3 | 567 | - | 227 |
| session_reopen | 3 | 0/3 | 805 | - | 465 |
| app_agent | 3 | 0/3 | 366 | - | 26 |
| engine_native | 3 | 0/3 | 366 | - | 26 |

### E_engine_integration — status: **not_measured**

No lmpu.model_* treatment arm is present; Batch 1 adds lmpu.model_async/lmpu.model_sync before the end-to-end claim can be evaluated.

## Simulation quarantine

Excluded records: 0 (baselines: none).

trace_token is a labelled simulation; it cannot appear in any claim table. Its records remain in the raw table, flagged.

## Limitations

- Trace-token is a labelled simulation; it cannot support any claim.
- C2 measures the S1 park window; the treatment performs no model decode by construction.
- C3 inlines derived-buffer metadata only, not any slice payload.
- C4 uses 64 pure-function additions; it is control-plane evidence, not model-quality evidence.
- C5 native runs in-process and excludes worker, IPC, and control-plane overhead.
- Per-correct-task cost divides total spend by correct tasks; failures are paid for, not skipped.
- Incorrect or failed baselines are retained; token reductions require equal task correctness.
- Failures and incorrect answers are retained in the table.
