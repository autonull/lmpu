# LMPU Viability — Pre-registration (Batch 0 / L0)

**Status:** committed before the first measured run of Batch 1.
**Authority:** `TODO.3.md` (sections 0.1, 0.2, V0, V1, V2). This artifact records the
hypotheses, thresholds, task suite, exclusions, and analysis code for the
viability ladder. No threshold in this document may be changed after results
are observed; changes require a new dated revision committed before new runs.

## 1. Hypotheses under test

| ID | Hypothesis | Threshold |
|----|------------|-----------|
| H-E | Engine integration beats the strongest app-level baseline at equal correctness | ≥10% cost (tokens or GPU J) per correct task, tight 95% CI; 20% stretch |
| H-B | LMPU solves context-infeasible tasks | Succeeds where a no-reducer app baseline fails; competitive with an app-side reducer |
| H-C | Batched work lowers control overhead | ≥10× lower control overhead vs individual submissions (64 jobs) |
| H-D | Program composition amortizes planning | Plan DAG beats N host round trips by ≥2× tokens or latency |
| H-P | Portable across model families | Same direction in a second family/class |
| H-I | Adoption cost is reasonable | Integration effort lower than or bounded vs app-side reducer |

Mechanism hypotheses (V1.5, unchanged from the plan):

| Mechanism | Hypothesis | Threshold |
|---|---|---|
| S1 park | Host performs no decode while parked | 0 forward passes; park energy within 5% of idle |
| KV reuse | Extension turns avoid full re-prefill | re-prefill tokens < 2% of prompt tokens |
| Bounded result | Large results stay out of context | 16 MiB → ≤64 context tokens; no payload inline |
| Batching | Submission overhead is not O(N) | 64 jobs → ≥10× lower control overhead than individual |
| Wake | Park/resume preserves answer quality | Output equals non-parked baseline |

## 2. Decisive path (plan 0.1)

1. **L0 (this batch).** Harness integrity, cost decomposition, addressable
   ceiling on existing artifacts, instrumented pilot if the artifacts lack
   per-phase totals. Stop if ceiling <10% on the best-case workload.
2. **L1.** Oracle-planner best case (16 MiB result, 60 s job, 32k context,
   exact oracle), single- and two-session, plus probes P1-P3. Stop only if
   both oracle arms lose to `app_agent` and the probes show no signal.
3. **L2+.** Model-planner best case; capability and negative controls;
   scaling; full matrix; verdict. Only after the stopping rules allow.

## 3. Task suite and arms

Workloads (`src/lmpu/bench/workloads.py`): `bigint_mul`, `matmul`,
`large_context`, `sleep`, `large_result`, `batch`. Oracle verification is
deterministic; `parse_answer`/`verify_answer` score natural language.

End-to-end arms (host model in the loop, V1.3):

- `direct_cot` — no tool.
- `sync_tool_full_result` — PAL-style program, sync execution, full result ingestion.
- `async_poll_programmatic` — programmatic polling, bounded result.
- `async_poll_model` — model-in-the-loop polling, bounded result.
- `session_reopen` — fresh session, full re-prefill for the answer turn.
- `app_agent` — app-level agent: in-process tool, KV-warm engine prefix cache,
  bounded (200-char) result ingestion, programmatic polling. No LMPU runtime.
- `engine_native` — engine-native session reuse, blocking in-process call,
  bounded ingestion. No LMPU runtime.
- `lmpu.model_async` / `lmpu.model_sync` (Batch 1) — treatment.

Mechanism-only arms: `lmpu.sync`, `lmpu.sleep`, `lmpu.slice`, `lmpu.individual`,
`lmpu.batch`, `native`. Simulated: `trace_token` (quarantined from claims).

Every model arm decodes a final answer and is scored on it; a correct tool
result with a wrong final answer is `correct=False`. Trials pair on shared
`(seed, trial)` with randomized order per trial.

## 4. Exclusions (written reasons, V0.7)

- **Prefix caching** — included via the engine's KV-reuse decoder in every
  model arm (`HFDecoder._advance` appends deltas); `session_reopen` exists to
  price re-prefill against it.
- **Programmatic polling** — included (`async_poll_programmatic`, `app_agent`),
  `poll_interval_ms = 50`, pre-registered.
- **Streaming results** — excluded from the arm set: injecting result chunks
  into context is strictly dominated by bounded ingestion for any result
  larger than the budget, and a streaming baseline that inlines the same
  bytes as `sync_tool_full_result` is already covered by it. The bounded-summary
  arm (200 chars) is the fair app-level implementation. Streaming is revisited
  in W5 only if the verdict passes.
- **vLLM / continuous batching and prefix-cache interception** — X1, optional;
  not in the viability path until L1/L2 signal exists.
- **Production-engine baseline (vLLM or equivalent)** — W3, after V6.
- **Constrained decoding (V2.2)** — promoted into the model-planner batch only
  after L1 passes; not an L0/L1 concern.

## 5. Analysis code

- Raw artifacts → `lmpu.bench.harness` (resume-safe JSONL, bootstrap 95% CIs,
  per-trial paired order).
- Claims → `lmpu.bench.report` (`report.v2`): mechanism (C2-C5) and
  end-to-end (`E_engine_integration`) sections, simulation quarantine,
  per-correct-task cost (failures are paid for, not skipped), paired deltas
  vs `direct_cot` on shared trials.
- Ceiling → `lmpu.bench.ceiling` (`ceiling.json`).

## 6. Seeds, budgets, and fairness

Seed: `20261103` (config default). Model: `Qwen/Qwen3.5-0.8B` on cuda
(bf16), temperature 0, greedy decode. `max_tokens` is a calibration variable
set per batch (64 was invalid; V2.1 sets 128/256/512 with 0.8B and 2B).
Hardware manifest, commit hash, and config are recorded in each artifact's
`manifest.json`. Randomized trial order uses `Random(seed + trial)` and is
identical across arms.

## 7. Verdict rules (unchanged from TODO.3)

Claim-level verdicts E, B, C, D, P, I; then portfolio PASS / NARROW / PIVOT /
FAIL per V6. Inconclusive is not PASS: rerun once with doubled n, then decide.
No threshold softening after results.
