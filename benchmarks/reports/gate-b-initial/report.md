# Benchmark report

- Run: `benchmarks/reports/gate-b-initial`
- Commit: `a48d0bcccb16`
- Device: `cuda`
- Model: `Qwen/Qwen3.5-0.8B`

## Results

| Workload | Baseline | Trials | Correct | Wall median (s) | 95% CI | Total tokens | GPUs J |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| bigint_mul | lmpu.sync | 2 | 2/2 | 0.0009 | 0.001–0.001 | 86 | 0.00 |
| bigint_mul | direct_cot | 2 | 0/2 | 1.5789 | 1.450–1.708 | 116 | 179.65 |
| bigint_mul | trace_token | 2 | 2/2 | 0.0016 | 0.002–0.002 | 3473 | 0.00 |
| bigint_mul | sync_tool | 2 | 0/2 | 1.9102 | 1.787–2.033 | 0 | 0.00 |
| bigint_mul | async_poll | 2 | 0/2 | 1.4537 | 1.303–1.604 | 0 | 0.00 |
| matmul | lmpu.sync | 2 | 2/2 | 0.0333 | 0.025–0.041 | 70 | 0.00 |
| matmul | direct_cot | 2 | 0/2 | 1.7061 | 1.603–1.809 | 72 | 234.07 |
| matmul | trace_token | 2 | 2/2 | 0.0007 | 0.001–0.001 | 1519 | 0.00 |
| matmul | sync_tool | 2 | 2/2 | 3.1636 | 2.735–3.592 | 238 | 373.39 |
| matmul | async_poll | 2 | 2/2 | 1.6629 | 1.595–1.731 | 213 | 220.06 |
| large_context | lmpu.sync | 2 | 2/2 | 0.0011 | 0.001–0.001 | 3291 | 0.00 |
| large_context | trace_token | 2 | 2/2 | 0.0026 | 0.001–0.004 | 3196 | 0.00 |
| sleep | lmpu.sync | 2 | 2/2 | 3.1684 | 3.166–3.170 | 69 | 0.00 |
| sleep | direct_cot | 2 | 0/2 | 1.5254 | 1.334–1.717 | 55 | 180.90 |
| sleep | trace_token | 2 | 2/2 | 0.0000 | 0.000–0.000 | 6 | 0.00 |
| sleep | sync_tool | 2 | 0/2 | 1.5732 | 1.445–1.701 | 0 | 0.00 |
| sleep | async_poll | 2 | 0/2 | 1.6803 | 1.571–1.790 | 0 | 0.00 |

## Claims

### C1_token_reduction

Status: **partial**

| Workload | Baseline | Baseline tokens | Reduction | Correct |
| --- | --- | ---: | ---: | ---: |
| bigint_mul | direct_cot | 116 | 1.35x | 0/2 |
| bigint_mul | trace_token | 3473 | 40.38x | 2/2 |
| bigint_mul | sync_tool | 0 | - | 0/2 |
| bigint_mul | async_poll | 0 | - | 0/2 |
| matmul | direct_cot | 72 | 1.03x | 0/2 |
| matmul | trace_token | 1519 | 21.70x | 2/2 |
| matmul | sync_tool | 238 | 3.40x | 2/2 |
| matmul | async_poll | 213 | 3.04x | 2/2 |
| large_context | trace_token | 3196 | 0.97x | 2/2 |
| sleep | direct_cot | 55 | 0.80x | 0/2 |
| sleep | trace_token | 6 | 0.09x | 2/2 |
| sleep | sync_tool | 0 | - | 0/2 |
| sleep | async_poll | 0 | - | 0/2 |

### C2_sleep_proof

Status: **partial**

No lmpu.sleep/S1 treatment is wired; synchronous and polling baselines are shown for context only.

### C3_context_protection

Status: **partial**

The workload demonstrates bounded injection on a 12k-character context, not a 16 MiB result buffer.

### C4_batching

Status: **not_measured**

No batched-vs-individual submission probe is wired.

### C5_wall_time

Status: **partial**

Native-only timings are not recorded; LMPU includes worker and control-plane overhead.

## Limitations

- Trace-token is a labelled simulation and is not a wall-time or energy result.
- C2 uses synchronous/polling baselines because lmpu.sleep is not yet wired.
- C3 uses a 12k-character context task, not a 16 MiB result.
- C4 batching has no probe in this report.
- C5 lacks a separately measured native baseline.
- Incorrect or failed baselines are retained; token reductions require equal task correctness.
- Failures and incorrect answers are retained in the table.
