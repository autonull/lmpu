# Benchmark report

- Run: `benchmarks/reports/gate-b-evidence`
- Commit: `3a42d6a5a011`
- Device: `cuda`
- Model: `Qwen/Qwen3.5-0.8B`

## Results

| Workload | Baseline | Kind | Trials | Correct | Wall median (s) | 95% CI | Total tokens | Tokens/correct | GPUs J |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| bigint_mul | lmpu.sync | mechanism | 3 | 3/3 | 0.0006 | 0.001–0.001 | 86 | 86 | 0.00 |
| bigint_mul | native | mechanism | 3 | 3/3 | 0.0002 | 0.000–0.000 | 0 | 0 | 0.00 |
| bigint_mul | direct_cot | end_to_end | 3 | 0/3 | 2.5835 | 2.560–2.601 | 148 | - | 246.88 |
| bigint_mul | trace_token (simulated) | simulation | 3 | 3/3 | 0.0016 | 0.002–0.002 | 3473 | 3473 | 0.00 |
| bigint_mul | sync_tool | end_to_end | 3 | 0/3 | 7.8546 | 2.572–16.262 | 0 | - | 0.00 |
| bigint_mul | async_poll | end_to_end | 3 | 0/3 | 2.5958 | 2.570–2.596 | 0 | - | 0.00 |
| matmul | lmpu.sync | mechanism | 3 | 3/3 | 0.0166 | 0.011–0.061 | 70 | 70 | 0.00 |
| matmul | native | mechanism | 3 | 3/3 | 0.0002 | 0.000–0.013 | 0 | 0 | 0.00 |
| matmul | direct_cot | end_to_end | 3 | 0/3 | 2.5555 | 2.546–2.624 | 104 | - | 246.23 |
| matmul | trace_token (simulated) | simulation | 3 | 3/3 | 0.0006 | 0.001–0.001 | 1519 | 1519 | 0.00 |
| matmul | sync_tool | end_to_end | 3 | 0/3 | 2.3942 | 2.388–2.413 | 0 | - | 0.00 |
| matmul | async_poll | end_to_end | 3 | 0/3 | 2.4067 | 2.403–2.499 | 0 | - | 0.00 |
| large_context | lmpu.sync | mechanism | 3 | 3/3 | 0.0008 | 0.001–0.001 | 3291 | 3291 | 0.00 |
| large_context | native | mechanism | 3 | 3/3 | 0.0005 | 0.001–0.001 | 0 | 0 | 0.00 |
| large_context | trace_token (simulated) | simulation | 3 | 3/3 | 0.0012 | 0.001–0.001 | 3196 | 3196 | 0.00 |
| sleep | lmpu.sync | mechanism | 3 | 3/3 | 3.1155 | 3.102–3.200 | 69 | 69 | 0.00 |
| sleep | lmpu.sleep | mechanism | 3 | 3/3 | 3.1226 | 3.105–4.601 | 70 | 70 | 0.00 |
| sleep | native | mechanism | 3 | 3/3 | 3.0003 | 3.000–3.001 | 0 | 0 | 0.00 |
| sleep | direct_cot | end_to_end | 3 | 3/3 | 2.5590 | 2.553–6.808 | 87 | 87 | 246.36 |
| sleep | trace_token (simulated) | simulation | 3 | 3/3 | 0.0000 | 0.000–0.000 | 6 | 6 | 0.00 |
| sleep | sync_tool | end_to_end | 3 | 0/3 | 2.5711 | 2.547–2.574 | 0 | - | 0.00 |
| sleep | async_poll | end_to_end | 3 | 0/3 | 2.6066 | 2.593–6.223 | 0 | - | 0.00 |
| large_result | lmpu.slice | mechanism | 3 | 3/3 | 0.0182 | 0.018–0.021 | 43 | 43 | 0.00 |
| large_result | native | mechanism | 3 | 3/3 | 0.0012 | 0.001–0.005 | 0 | 0 | 0.00 |
| batch | lmpu.individual | mechanism | 3 | 3/3 | 0.0196 | 0.019–1.534 | 3624 | 3624 | 0.00 |
| batch | lmpu.batch | mechanism | 3 | 3/3 | 0.0011 | 0.001–0.003 | 853 | 853 | 0.00 |
| batch | native | mechanism | 3 | 3/3 | 0.0003 | 0.000–0.001 | 0 | 0 | 0.00 |

## Mechanism claims

Mechanism-only evidence (S1 park, bounded slices, batching, overhead). Not an end-to-end efficiency claim.

### C2_sleep_proof

Status: **measured**

The lmpu.sleep treatment parks the host in S1 for the measured job window and performs zero forward passes while parked.

### C3_context_protection

Status: **measured**

A 16 MiB shared-memory buffer is represented in context by bounded-slice metadata; no slice payload is inlined.

### C4_batching

Status: **measured**

One batch submission replaces one submission per item.

### C5_wall_time

Status: **measured**

| Workload | native/treatment wall ratio |
| --- | ---: |
| bigint_mul | 3.72x |
| matmul | 72.12x |
| large_context | 1.45x |
| sleep | 1.04x |
| large_result | 14.65x |
| batch | 3.99x |

native runs the same computation in-process; LMPU includes worker, IPC, and control-plane overhead.


## End-to-end results

Host model in the loop; every arm is scored on its decoded final answer. Token reductions require equal task correctness.

Reference: `direct_cot` (paired trials, same seed and order).

### bigint_mul

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_cot | 3 | 0/3 | 148 | - | - |
| sync_tool | 3 | 0/3 | 0 | - | -148 |
| async_poll | 3 | 0/3 | 0 | - | -148 |

### matmul

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_cot | 3 | 0/3 | 104 | - | - |
| sync_tool | 3 | 0/3 | 0 | - | -104 |
| async_poll | 3 | 0/3 | 0 | - | -104 |

### sleep

| Baseline | Trials | Correct | Total tokens | Tokens/correct | Paired Δ tokens (median) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_cot | 3 | 3/3 | 87 | 87 | - |
| sync_tool | 3 | 0/3 | 0 | - | -87 |
| async_poll | 3 | 0/3 | 0 | - | -87 |

### E_engine_integration — status: **not_measured**

No lmpu.model_* treatment arm is present; Batch 1 adds lmpu.model_async/lmpu.model_sync before the end-to-end claim can be evaluated.

## Simulation quarantine

Excluded records: 12 (baselines: trace_token).

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
