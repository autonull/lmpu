# Benchmark report

- Run: `benchmarks/results/2026-11-03/NVIDIA-GeForce-RTX-3080/3a42d6a5a011`
- Commit: `3a42d6a5a011`
- Device: `cuda`
- Model: `Qwen/Qwen3.5-0.8B`

## Results

| Workload | Baseline | Trials | Correct | Wall median (s) | 95% CI | Total tokens | GPUs J |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| bigint_mul | lmpu.sync | 3 | 3/3 | 0.0006 | 0.001–0.001 | 86 | 0.00 |
| bigint_mul | native | 3 | 3/3 | 0.0002 | 0.000–0.000 | 0 | 0.00 |
| bigint_mul | direct_cot | 3 | 0/3 | 2.5835 | 2.560–2.601 | 148 | 246.88 |
| bigint_mul | trace_token | 3 | 3/3 | 0.0016 | 0.002–0.002 | 3473 | 0.00 |
| bigint_mul | sync_tool | 3 | 0/3 | 7.8546 | 2.572–16.262 | 0 | 0.00 |
| bigint_mul | async_poll | 3 | 0/3 | 2.5958 | 2.570–2.596 | 0 | 0.00 |
| matmul | lmpu.sync | 3 | 3/3 | 0.0166 | 0.011–0.061 | 70 | 0.00 |
| matmul | native | 3 | 3/3 | 0.0002 | 0.000–0.013 | 0 | 0.00 |
| matmul | direct_cot | 3 | 0/3 | 2.5555 | 2.546–2.624 | 104 | 246.23 |
| matmul | trace_token | 3 | 3/3 | 0.0006 | 0.001–0.001 | 1519 | 0.00 |
| matmul | sync_tool | 3 | 0/3 | 2.3942 | 2.388–2.413 | 0 | 0.00 |
| matmul | async_poll | 3 | 0/3 | 2.4067 | 2.403–2.499 | 0 | 0.00 |
| large_context | lmpu.sync | 3 | 3/3 | 0.0008 | 0.001–0.001 | 3291 | 0.00 |
| large_context | native | 3 | 3/3 | 0.0005 | 0.001–0.001 | 0 | 0.00 |
| large_context | trace_token | 3 | 3/3 | 0.0012 | 0.001–0.001 | 3196 | 0.00 |
| sleep | lmpu.sync | 3 | 3/3 | 3.1155 | 3.102–3.200 | 69 | 0.00 |
| sleep | lmpu.sleep | 3 | 3/3 | 3.1226 | 3.105–4.601 | 70 | 0.00 |
| sleep | native | 3 | 3/3 | 3.0003 | 3.000–3.001 | 0 | 0.00 |
| sleep | direct_cot | 3 | 3/3 | 2.5590 | 2.553–6.808 | 87 | 246.36 |
| sleep | trace_token | 3 | 3/3 | 0.0000 | 0.000–0.000 | 6 | 0.00 |
| sleep | sync_tool | 3 | 0/3 | 2.5711 | 2.547–2.574 | 0 | 0.00 |
| sleep | async_poll | 3 | 0/3 | 2.6066 | 2.593–6.223 | 0 | 0.00 |
| large_result | lmpu.slice | 3 | 3/3 | 0.0182 | 0.018–0.021 | 43 | 0.00 |
| large_result | native | 3 | 3/3 | 0.0012 | 0.001–0.005 | 0 | 0.00 |
| batch | lmpu.individual | 3 | 3/3 | 0.0196 | 0.019–1.534 | 3624 | 0.00 |
| batch | lmpu.batch | 3 | 3/3 | 0.0011 | 0.001–0.003 | 853 | 0.00 |
| batch | native | 3 | 3/3 | 0.0003 | 0.000–0.001 | 0 | 0.00 |

## Claims

### C1_token_reduction

Status: **partial**

| Workload | Baseline | Baseline tokens | Reduction | Correct |
| --- | --- | ---: | ---: | ---: |
| bigint_mul | direct_cot | 148 | 1.72x | 0/3 |
| bigint_mul | trace_token | 3473 | 40.38x | 3/3 |
| bigint_mul | sync_tool | 0 | - | 0/3 |
| bigint_mul | async_poll | 0 | - | 0/3 |
| matmul | direct_cot | 104 | 1.49x | 0/3 |
| matmul | trace_token | 1519 | 21.70x | 3/3 |
| matmul | sync_tool | 0 | - | 0/3 |
| matmul | async_poll | 0 | - | 0/3 |
| large_context | trace_token | 3196 | 0.97x | 3/3 |
| sleep | direct_cot | 87 | 1.26x | 3/3 |
| sleep | trace_token | 6 | 0.09x | 3/3 |
| sleep | sync_tool | 0 | - | 0/3 |
| sleep | async_poll | 0 | - | 0/3 |

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

native runs the same computation in-process; LMPU includes worker, IPC, and control-plane overhead.

## Limitations

- Trace-token is a labelled simulation and is not a wall-time or energy result.
- C2 measures the S1 park window; the treatment performs no model decode by construction.
- C3 inlines derived-buffer metadata only, not any slice payload.
- C4 uses 64 pure-function additions; it is control-plane evidence, not model-quality evidence.
- C5 native runs in-process and excludes worker, IPC, and control-plane overhead.
- Incorrect or failed baselines are retained; token reductions require equal task correctness.
- Failures and incorrect answers are retained in the table.
