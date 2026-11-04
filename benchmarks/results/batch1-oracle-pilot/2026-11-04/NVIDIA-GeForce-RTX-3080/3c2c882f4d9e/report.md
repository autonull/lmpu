# Benchmark report

- Run: `benchmarks/results/batch1-oracle-pilot/2026-11-04/NVIDIA-GeForce-RTX-3080/3c2c882f4d9e`
- Commit: `3c2c882f4d9e`
- Device: `cuda`
- Model: `Qwen/Qwen3.5-0.8B`

## Results

| Workload | Baseline | Kind | Trials | Correct | Wall median (s) | 95% CI | Total tokens | Tokens/correct | GPUs J |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| best_case | lmpu.model_async_oracle | end_to_end | 3 | 3/3 | 72.3006 | 71.269–75.093 | 8061 | 8061 | 6885.66 |
| best_case | lmpu.model_sync_oracle | end_to_end | 3 | 3/3 | 73.2290 | 71.204–73.512 | 8064 | 8064 | 6867.95 |
| best_case | app_agent_oracle | end_to_end | 3 | 3/3 | 75.0161 | 73.062–86.816 | 7920 | 7920 | 8310.92 |
| best_case | engine_native_oracle | end_to_end | 3 | 3/3 | 71.8090 | 71.727–72.903 | 7920 | 7920 | 6820.62 |
| best_case | session_reopen_oracle | end_to_end | 3 | 3/3 | 73.3284 | 71.640–78.184 | 31136 | 31119 | 1184.05 |
| best_case | lmpu.model_async_2session | end_to_end | 3 | 3/3 | 147.0750 | 142.767–147.598 | 16128 | 16128 | 14011.77 |
| best_case | app_agent_2session | end_to_end | 3 | 3/3 | 83.9372 | 82.814–91.018 | 15840 | 15840 | 7993.85 |
| best_case | direct_cot | end_to_end | 3 | 0/3 | 11.5881 | 10.782–12.302 | 7822 | - | 1102.54 |

## Mechanism claims

Mechanism-only evidence (S1 park, bounded slices, batching, overhead). Not an end-to-end efficiency claim.

### C2_sleep_proof

Status: **not_measured**

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

### best_case

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lmpu.model_async_oracle | 3 | 3/3 | 8061 | 8061 | 239 |
| lmpu.model_sync_oracle | 3 | 3/3 | 8064 | 8064 | 242 |
| app_agent_oracle | 3 | 3/3 | 7920 | 7920 | 98 |
| engine_native_oracle | 3 | 3/3 | 7920 | 7920 | 98 |
| session_reopen_oracle | 3 | 3/3 | 31136 | 31119 | 23314 |
| lmpu.model_async_2session | 3 | 3/3 | 16128 | 16128 | 8306 |
| app_agent_2session | 3 | 3/3 | 15840 | 15840 | 8018 |
| direct_cot | 3 | 0/3 | 7822 | - | - |

### E_engine_integration — status: **partial**


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
