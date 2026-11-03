# Strongest-baseline audit (V0.7)

**Date:** 2026-11-03. **Scope:** what the end-to-end arm set includes, what it
excludes, and why. Companion to `docs/viability/preregistration.md` §4.

## Evaluated and included

| Technique | Where it lives | Notes |
|---|---|---|
| Prefix caching / KV reuse | Every model arm, via `HFDecoder._advance` (appends delta tokens to the live KV cache, re-prefills only on divergence) | `session_reopen` exists to price KV-warm reuse against a fresh full prefill on the same model, prompt and budget. |
| Programmatic polling | `async_poll_programmatic`, `app_agent` | 50 ms poll interval (`poll_interval_ms`), pre-registered in the config; poll counts recorded in trial metadata; zero host-model tokens spent on waits. |
| Bounded result ingestion | `app_agent`, `engine_native`, `async_poll_programmatic`, `async_poll_model` (200-char summary) | Mirrors what a correct app-level integration does for large tool results. |
| Full result ingestion | `sync_tool_full_result` | The naive PAL-style app: easy to beat, retained to bound the naive case. |
| In-process tool execution | `app_agent`, `engine_native` (via builtin processors; no LMPU runtime) | The strongest app-level integration: no IPC, no control plane, library-call cost. LMPU must win against *this*, not against a strawman. |

## Excluded, with reasons

| Technique | Reason for exclusion |
|---|---|
| Streaming result chunks into context | Injecting the same bytes as chunks costs the same tokens as ingesting them at once and is strictly worse than bounded ingestion for large results; any workload where streaming helps is one where bounded ingestion also helps, so the bounded arm is the binding constraint. Revisit in W5 (capability matrix) only if the verdict passes. |
| vLLM / continuous batching / prefix-cache interception | X1 spike, explicitly outside the viability path until L1/L2 signal exists. Production-engine baseline is W3, after V6. |
| Best-of-N / self-consistency | Retry policy is part of the calibrated model budget (V2.1), not an app-side trick; adding it here would confound fairness invariants. Revisit in V2.7 (adversarial red-team). |
| Grammar-constrained decoding | V2.2; promoted only after L1 passes. The control channel in Batch 1 arms stays free-text so the arms compare on equal footing. |
| Model-driven polling as the polling default | Included as `async_poll_model` (a deliberately expensive arm), but programmatic polling is the strongest fair default and is what `app_agent` uses. |

## Fairness invariants (V0.4)

Same model (`Qwen/Qwen3.5-0.8B`, bf16, cuda), same prompts
(`program_prompt`), same greedy decode and `max_tokens` per arm within a
run, same seed and hardware, randomized per-trial order (`Random(seed +
trial)`), paired trials on shared `(seed, trial)`. Per-correct-task cost is
reported for every cell (failures are paid for, not skipped), and paired
per-trial deltas vs `direct_cot` are reported in
`report.json` → `end_to_end.paired_vs_reference`.
