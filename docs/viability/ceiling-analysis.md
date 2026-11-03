# L0 — Cost decomposition and addressable-cost ceiling (V0.8)

**Date:** 2026-11-03. **Status:** ceiling verdict INCONCLUSIVE — best-case cell
not yet measurable; everything else closed. Batch 1 (oracle arms) is required
to resolve it. Do not advance past Batch 1 on the strength of this document.

## Method

`lmpu.bench.ceiling` decomposes harness artifacts per (workload, baseline):
prefill, generated, control, injected, total tokens, forward passes, GPU
joules, wall, park/wake windows, per-phase wall (this batch's instrumentation).
The addressable fraction is

    (ingestion + polling-proxy + re-prefill + idle) / total

measured on the token axis. Components that artifacts cannot measure are
reported as `not_measured`; none are estimated silently. `control_tokens`
serves as the polling/wait proxy until the Batch 1 instrumented run splits
poll traffic from submissions; `reprefill` is measurable only for cells with
per-phase instrumentation.

## Existing artifacts (`benchmarks/reports/gate-b-evidence`)

- No correct end-to-end tool-arm cells exist. Every `sync_tool`/`async_poll`
  trial failed at the program step (0/3) and recorded **zero tokens** because
  the failure path discarded accounting — the defect V0.2 closes. Re-rendered
  with the fixed harness, the same raw log shows the failures at 0 tokens;
  the accounting gap is proven, not hypothetical.
- Only correct end-to-end cell: `sleep/direct_cot` (3/3) — 261 tokens, of
  which addressable (ingestion + control) = **0%**. `direct_cot` ingests
  nothing; there is nothing for LMPU to remove. (3 idle seconds of the wall
  are, however, exactly what the S1 park attacks in the sleep phase.)
- Mechanism cells (`lmpu.*`) are ~100% "addressable" by construction: their
  entire token budget is control + injected traffic. They are the addressable
  workload, not a baseline to beat.
- **Reading:** existing artifacts cannot support either a PASS or a FAIL for
  the performance thesis. They prove the measurement defect and nothing else.

## Instrumented pilot (`benchmarks/reports/l0-pilot`)

Short pilot, authorized by plan §0.1 ("if they lack per-phase totals, run a
short instrumented pilot instead of re-running the full matrix"). Config:
`sleep` + `bigint_mul`, 7 end-to-end arms, 3 trials, cuda,
`max_tokens=256`, seed 20261103. **42 records.** Not the full matrix; one
dataset, no re-runs.

Findings:

1. **The 0.8B model-planner path fails outright at this budget.** 0/3 correct
   on every tool arm on both workloads; the model fills all 256 tokens on
   every turn and echoes the `<a>`/`$CONTEXT` placeholder values literally
   (e.g. `invalid literal for int() with base 10: '<a>'`). Only
   `sleep/direct_cot` passes (3/3). Total failed-attempt accounting:
   15,736 tokens retained (previously 0 — V0.2 verified end-to-end on GPU).
2. **Per-phase instrumentation works.** `plan` ≈ 369 s of 42-trial wall;
   `poll` 9.3 s; `submit` 0.4 s; `answer` 89.4 s across trials that reached
   the answer turn (almost none — failures stop at plan).
3. **Addressable fraction is 0% on every correct-or-failed cell** — because
   no correct tool arm ran at all. There is no ingestion/polling to address
   when the plan step never lands. This is the calibration defect from gate-b,
   reproduced with proper accounting, not a refutation of the thesis.
4. Direct-cot reference on sleep: 837 tokens median, 3/3 correct. That is the
   floor any tool arm must beat on cost at equal correctness.

## Verdict

**INCONCLUSIVE — not FAIL.**

- FAIL requires showing ceiling <10% on the best-case workload (16 MiB
  result, 60 s job, 32k context). That cell does not exist in the harness
  yet (it is V2.3 work) and was not measurable in this batch.
- What this batch proves: the harness can no longer hide failures, mixed
  claim tables are gone, the end-to-end arms run mechanically, and the
  model-planner arm as prompted is not viable — calibration (V2.1) and
  oracle planning (V1.8) are mandatory gates before any end-to-end number.
- Stop rule for L1: run the oracle-planner best-case arms (V1.1/V1.2/V1.8)
  first. If LMPU loses to `app_agent` in both single- and two-session
  oracle arms, stop and record FAIL. If either arm wins or the probes
  (P1-P3) show signal, proceed to Batch 2 with the calibration fixes.

## Defects recorded

- 0.8B never emits EOS within 256 tokens; `max_tokens` calibration is part
  of V2.1, not a harness afterthought.
- Program prompts with `<a>`-style placeholders are echoed literally; V2.1
  needs few-shot prompts with concrete values filled in.
- Mid-pilot harness defect: `async_poll_model` crashed when the plan step
  failed (`submitted` unbound). Fixed; the pilot dataset was discarded and
  re-run cleanly from scratch after the fix.
