# LMPU Viability — Pre-registration (Batch 1 / L1)

**Status:** committed before the first measured run of Batch 1.
**Authority:** `TODO.3.md` (§0.1, §0.2, V1.1–V1.2, V1.8, V1.10, §9 Batch 1).
This artifact records the decisive experiment, its arms, budgets, and stop
rules for Batch 1 (the L1 oracle best case). No threshold here may change
after results are observed.

## 1. The decisive experiment (plan §0.1)

Workload: `best_case` (`src/lmpu/bench/workloads.py`) —

- **16 MiB result**: python.vm job computes SHA-256 over a deterministic
  16 MiB buffer (repeated `lmpu-best-case-16mib-deterministic-unit\n` chunks)
  and prints `{"bytes": 16777216, "sha256": "<hex>"}`.
- **60 s job**: the program sleeps 60 s before hashing; wall per trial ≥ 60 s.
- **~8k-token context** (largest feasible subset on 10 GB): the prompt carries
  a 200-line deterministic ledger document (~7.6k tokens with
  Qwen/Qwen3.5-0.8B tokenizer) as a read-only distractor; the answer is
  tool-derived, never document-derived. (The 32k variant was an arithmetic
  error: 32,818 tokens exceeded `HFEngine.max_context=32768` and every model
  arm generated 1–2 tokens before the context-full guard fired.)
- **Exact oracle**: sha256 of the deterministic buffer is known in advance;
  `verify` checks `bytes == 16 MiB` and the digest; `verify_answer` checks the
  decoded `ANSWER: <sha256 hex digest>` line.

Arms (all oracle-planned: gold program injected, zero planning tokens; the
model planner is Batch 2 and remains known-void until V2.1 calibration):

| Arm | Path | Why it is here |
|---|---|---|
| `lmpu.model_async_oracle` | LMPU runtime; submit → S1 park (zero host decode) → wake → bounded ingest → answer | Treatment (V1.1) |
| `lmpu.model_sync_oracle` | LMPU runtime; blocking `mode=sync` submit → bounded ingest → answer | Treatment, blocking delta (V1.2) |
| `app_agent_oracle` | App-level agent; in-process tool, shared-engine KV-warm prefix cache, bounded 200-char ingestion, programmatic polling | Strongest app-level competitor |
| `engine_native_oracle` | Engine-native session reuse, blocking in-process call, bounded ingestion | Engine boundary without the LMPU protocol |
| `session_reopen_oracle` | LMPU job, then a fresh session re-prefills the full conversation | P3 probe: re-prefill vs KV-warm wake |
| `lmpu.model_async_2session` | Two concurrent LMPU sessions, one runtime, both jobs parked in S1, answers decoded per session | Two-session best case |
| `app_agent_2session` | Two concurrent app-level tasks on one shared engine with two independent KV-warm decoders, no LMPU runtime | Two-session competitor |
| `direct_cot` | No tool; cannot compute the digest | Capability floor reference |

Excluded with reasons:

- `sync_tool_full_result` / `sync_tool_full_result_oracle` — full ingestion of
  the result is infeasible: the tool result is 16 MiB. This is the capability
  boundary V1.9 will record.
- Model-planner arms (`lmpu.model_async`, `app_agent`, …) — Batch 2 (L2),
  gated on V2.1 calibration.
- `async_poll_model` — model-in-the-loop polling; model planning is not
  calibrated; excluded until Batch 2.
- Streaming ingestion — strictly dominated by bounded ingestion at any result
  size (pre-registration Batch 0, §4 exclusions).

**Revision (2026-11-03):** Reduced context from 32k to 8k (200 lines) to fit
`HFEngine.max_context=32768` with answer headroom on a 10 GB card. This is
the "largest feasible subset" per the plan §1. The 32k variant was never
measured; no prior data is invalidated.

## 2. Metrics and analysis

- Primary endpoint (per correct task, paid for, not skipped): host tokens
  (`total_tokens`), forward passes, GPU joules, wall time.
- Secondary: accuracy per arm, `peak_context_tokens`, `injected_tokens`,
  per-phase wall (`prefill/plan/submit/park/wake/ingest/answer/reprefill`),
  `sleep_duration_seconds`, `wake_latency_seconds`.
- Paired trials on shared `(seed, trial)`; randomized per-trial order
  (`Random(seed + trial)`); seed `20261103`.
- Bootstrap 95% CIs (`lmpu.bench.report`, `report.v2`); claim tables are
  split end-to-end / mechanism; simulation quarantined.
- KV-warm claim (KV reuse hypothesis): compare `session_reopen_oracle`
  re-prefill tokens vs `lmpu.model_async_oracle` wake+append delta. The
  engine reuses the KV cache when the next prompt extends the known text;
  the wake path appends only the injected suffix.

## 3. Probes (V1.10)

- **P1 — Worker-side reducer**: the gold program reduces inside the worker
  (sha256 computed in the sandboxed subprocess); the host sees only a
  bounded ~200-char summary. Signal: `injected_tokens` on the answer turn
  (~200 chars ≈ ≤60 tokens) vs the 16 MiB payload a naive agent would pay.
  Recorded from the best_case arms directly.
- **P2 — Plan graph**: the composed single-job plan
  (`fetch → extract → verify` in one python.vm submission) is the oracle
  stand-in for a plan DAG. Signal: host tokens and wall of the composed
  arm vs a decomposed two-step variant. No new executor is built in
  Batch 1; a null result here means P2 needs the plan-graph work before
  claim D is testable, and that is recorded as such.
- **P3 — KV wake vs session reopen**: `lmpu.model_async_oracle` wake latency
  vs `session_reopen_oracle` re-prefill tokens/wall on the identical prompt.

## 4. Budgets and fairness

Model `Qwen/Qwen3.5-0.8B` on cuda (bf16), temperature 0 (greedy), seed
`20261103`. `max_tokens = 256` for the answer budget (the answer line is
short; planning is absent in oracle mode). `poll_interval_ms = 50`.
`idle_seconds = 1`. Workload `timeout_ms = 180_000`. Warmups 0. Trials per
arm: **3** (pilot; the batch-scale n is set only from measured variance, not
chosen now). One raw dataset feeds every analysis; no cell is re-run to
"confirm". Randomized arm order per trial.

## 5. Stop rule (Batch 1)

L1 stops the plan only if `lmpu.model_async_oracle` loses to `app_agent_oracle`
in **both** the single-session and the two-session arms **and** the probes
show no signal (P1 injected-token bound not demonstrated; P3 wake not
cheaper than re-prefill; P2 showing no composition advantage). Otherwise the
result is recorded as-is and Batch 2 (calibration + model planner) proceeds
with the defect list from this run. A single-session loss does not alone
falsify the design.

## 6. Non-negotiables

- No claim is recorded from a failed baseline; failed arms are retained in
  the tables with their errors.
- The 0.8B model's free-text planning is not part of Batch 1; any
  model-planner numbers are void and must not support a verdict.
- If the two-session arms exhaust VRAM (2 × 32k KV heads), the arms record
  the failure and the memory ceiling stands in for the claim; the single-
  session arms still decide Batch 1.
- Commit hash, hardware manifest, and config are in the artifact's
  `manifest.json`; reports are copied to `benchmarks/reports/batch1-oracle-pilot/`.
