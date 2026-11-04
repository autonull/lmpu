# LMPU Development Plan — TODO.4.md

**Revision 2** · 2026-11-04 · supersedes revision 1
**Purpose**: salvage the research by separating what is *proven* from what was *artifact*, and by converting the planner blocker into a bounded, measurable result.

---

## 1. Verdict first

The evidence supports a **capability-and-cost** claim, not a **latency** claim.
Three results are binary and defensible. One result (planner latency) is a real
limit that we currently misreport as a bug. The rest is noise that must not ship.

| Claim | Status | Strength |
|---|---|---|
| C1 · Context-infeasible tasks are solvable with an unchanged 32k model | **PROVEN** (capability, 3/3) | ⭐⭐⭐ Headline |
| C2 · Batch submission removes ~27× control round-trips | **PROVEN** (0.0009 s vs 0.0243 s @ 64 items) | ⭐⭐⭐ |
| C3 · Large results ingest at bounded cost (43 tok for 16 MiB) | **PROVEN** | ⭐⭐⭐ |
| C4 · No false advantage on trivial tasks | **PROVEN** (9/9 negative controls) | ⭐⭐ Credibility |
| C5 · Parking is nearly free (~0.2 J overhead) | **PROVEN** | ⭐⭐ |
| C6 · Token parity with app-level prefix cache | **NEGATIVE** (E=NARROW) | ⚠️ Must state |
| C7 · Model planner is usable per-call | **NOT PROVEN** — 16 tok/s constrained | ❌ Blocker |
| C8 · Energy advantage over prefix-cached chat | **NOT PROVEN** — below noise | ❌ Must not claim |

**Strategic consequence.** C6 and C8 say LMPU does not beat a well-built
prefix-cached agent on the common path. C7 says per-call planning is too slow on
consumer hardware. Therefore the only honest thesis is:

> LMPU wins **capability** (C1) and **amortized cost** (C2, C3), and only when
> the plan is amortized over work that is expensive relative to planning.

That framing is publishable. It is also narrower than the original thesis. The
plan below makes it precise and measurable rather than hiding it.

---

## 2. Diagnostic findings (measured this session)

These change the plan. They are not assumptions.

| Measurement | Value | Implication |
|---|---|---|
| Qwen2.5-1.5B, `generate`, eager | 71.7 tok/s | Model is fine; 1.5B is viable |
| Qwen2.5-1.5B, `generate`, **sdpa** | **80.1 tok/s** | Free 12% — set `attn_implementation="sdpa"` |
| Constrained (lm-format-enforcer) | **16.3 tok/s** | **5× penalty — this is the "15 s/trial"** |
| Peak VRAM, 1.5B bf16 | 3.1 GB / 10 GB | **2–3 concurrent sessions fit → V3.5 runnable** |
| Plain decode inside `HFEngine` | 37–47 tok/s vs 80 fresh | Engine loop leaks ~2× (per-token NVML in `_record`) |

**Root cause of the planner blocker**: it is not the model and not the KV cache.
It is two implementation defects:

1. `HFEngine._forward` → `_record` → `energy_sampler.sample()` calls **NVML per
   token**. A syscall per token costs more than the forward pass at these sizes.
2. `lm-format-enforcer`'s `prefix_allowed_tokens_fn` **re-parses the whole JSON
   prefix every token** (O(n²)). This is the 16 tok/s.

Neither is fundamental. Both are fixable in ≤2 days, and the second has a
drop-in replacement (`outlines` FSM, or `xgrammar`) that is O(1) per token.

**Realistic post-fix planning cost**: 150-token program at 50–60 tok/s ≈ **2.5–3 s**.
That is *usable for amortized tasks* and is 5–6× better than today.

---

## 3. Opportunity ledger

Ranked by (expected value × feasibility) ÷ cost. This is the whole plan in one table.

| # | Opportunity | Value | Cost | Verdict |
|---|---|---|---|---|
| O1 | Fix `HFEngine` token loop (sdpa, batch NVML, drop per-token `_record`) | Unblocks all planner measurement; 2× everywhere | 2 h | **DO FIRST** |
| O2 | Swap grammar backend → `outlines` FSM | 16 → 50+ tok/s; planner becomes measurable | 4 h | **DO SECOND** |
| O3 | **Crossover experiment**: planner-cost ÷ task-duration sweep | Turns the blocker into the headline result | 1 d | **DO THIRD** |
| O4 | Capability ladder 32k→64k→128k→256k (binary cliff) | Strongest single figure in the paper | 4 h GPU | DO |
| O5 | Concurrency 1/2/3 sessions (3.1 GB each ⇒ fits) | Completes V3.5; kills the "doesn't scale" objection | 3 h GPU | DO |
| O6 | Replace noise-energy with **bounded-interval** reporting | Removes an indefensible claim | 3 h | DO |
| O7 | Chunked-RAG baseline for capability arm | Pre-empts the obvious reviewer objection | 1 d | DO |
| O8 | Cost model + sensitivity (V4.3/V4.5) | Converts J and s into a decision | 1 d | DO |
| O9 | Robustness soak (V5) | Needed for release gate only | 2 d | DEFER |
| O10 | Model scaling (V3.2) | Only 1.5B fits cleanly; low marginal value | — | **CUT** |
| O11 | Qwen3.5 multimodal planner | Fails structurally (tensor mismatch) | — | **CUT** |
| O12 | RAPL CPU energy in container | No `/sys/class/powercap`; unfixable here | — | **DOCUMENT GAP** |
| O13 | Chasing sub-2 s planner for *per-call* use | Wrong target — see §4 | — | **CUT** |

### The reframe that saves the research (O3, expanded)

Stop asking *"is the planner fast?"* and ask *"at what task size does LMPU's
planning cost amortize?"* Define:

```
break_even(task) = planner_cost / (baseline_cost(task) − worker_cost(task))
```

- `planner_cost` ≈ 2.5–3 s of model time (measured, post-O2).
- `baseline_cost(task)` grows with task duration (blocking call, re-prefill per turn).
- `worker_cost(task)` ≈ 0 s of model time (the worker runs off-GPU).

Prediction: below the crossover, direct-CoT wins; above it, LMPU wins by an
increasing margin. **That curve is the paper.** It also makes C7 an *input* to a
result instead of a failure — which is the difference between "our planner is
slow" and "we characterised when planning pays off."

---

## 4. Revised plan

### Phase 0 — De-artifact the harness (½ day) · gate G0

- [ ] `attn_implementation="sdpa"` default in `HFEngine._initialize`.
- [ ] Remove per-token energy sampling; sample every 16 tokens **and** at
      boundaries. `_record` must be allocation-light.
- [ ] Replace the manual decode loop with `model.generate` + `DynamicCache`
      (keep `append`/KV-reuse paths intact — they are load-bearing for P3).
- [ ] **Gate G0**: `bigint_mul` free decode ≥ 70 tok/s inside the harness.
      If not, stop and profile before touching the planner.

### Phase 1 — Planner, made honest and fast (1.5 days) · gate G1

- [ ] Install and wire `outlines` (FSM, O(1)/token) behind the existing
      `create_decoder` seam. Keep lm-format-enforcer as fallback.
- [ ] Add a **planner-invocation counter** already present in the harness
      (`plan` phase) — ensure it is emitted per trial.
- [ ] Calibration table: 0.5B / 1.5B × budgets {128, 256} × {free, constrained}
      × {outlines, lmfe} → correctness and tok/s.
- [ ] **Gate G1**: constrained ≥ 45 tok/s **and** valid-JSON ≥ 95% on 3 workloads.
      If G1 fails, accept oracle planner and go to Phase 2 via the oracle path —
      do **not** spend more than 1.5 days here.

### Phase 2 — The crossover curve (1 day) · gate G2 · **the headline**

- [ ] Workload ladder by *task duration*: `sleep_1s/3s/30s`, `batch_64/1024`,
      `context_8k/32k`.
- [ ] Arms: `direct_cot` (raw model, no tool), `lmpu.model_async` (planner +
      async worker), `lmpu.model_sync`, `lmpu.sync` (oracle), `native`.
- [ ] Record per arm: wall, model-tokens, GPU-J, correctness.
- [ ] Emit `benchmarks/reports/crossover.{json,md}`: break-even duration per
      workload class with a confidence interval.
- [ ] **Gate G2**: a monotone break-even curve exists and is explained. If LMPU
      never wins, that is the finding — state it and stop.

### Phase 3 — Capability, made airtight (1 day) · gate G3

- [ ] Ladder `context_32k/64k/128k/256k`; per rung assert the 32k model
      **cannot** ingest and LMPU **can** solve.
- [ ] Add **chunked-RAG baseline** (O7): same model, retrieve top-k chunks.
      Report where chunking fails and why (needle outside top-k, cross-chunk
      reasoning).
- [ ] **Gate G3**: LMPU solves ≥ 90% on rungs above the context window while
      full-context and chunked-RAG are strictly worse.

### Phase 4 — Cost, energy, concurrency (1.5 days) · gate G4

- [ ] Energy: measure whole-trial power via NVML externally; report
      **median ± IQR** and **refuse to claim any delta below the noise floor**.
      Delete every "0.21 J" style point estimate from the record.
- [ ] `V4.3/V4.5`: cost per *correct task* with price ∈ {0.05,…,0.50} $/kWh;
      sensitivity table.
- [ ] `V3.5`: 1/2/3 concurrent sessions (VRAM allows), report throughput and p99.
- [ ] **Gate G4**: LMPU cost-per-correct-task is lower than direct-CoT on ≥2 of
      3 amortized workload classes, with non-overlapping IQRs.

### Phase 5 — Release (1.5 days) · gate G5

- [ ] Mechanical verdict from gates (all evidence pre-registered).
- [ ] Rewrite claims to exactly C1–C5; explicitly disclose C6–C8.
- [ ] Viability report + raw JSONL archived under `benchmarks/results/`.
- [ ] **Gate G5**: every claim in the report maps to a gate and a raw file.

---

## 5. What was cut, and why

- **V3.2 model scaling** — only 1.5B fits with margin; a second scale adds cost
  without changing the mechanism argument.
- **Qwen3.5 multimodal planner** — structural tensor mismatch in the logits
  processor; not a tuning problem.
- **Sub-2 s per-call planner** — wrong objective. The thesis is amortization;
  chasing per-call latency optimises against the claim.
- **RAPL CPU joules** — no powercap device in this container. Document the gap;
  never fabricate joules.
- **V5 soak** — release-gate work. Runs only after G2/G3 pass, because a soak on
  a broken claim is expensive.

---

## 6. Schedule and resources

| Phase | Wall | GPU | Gate |
|---|---|---|---|
| 0 · de-artifact | ½ d | 20 min | G0 free ≥ 70 tok/s |
| 1 · planner | 1.5 d | 1 h | G1 ≥ 45 tok/s, ≥ 95% valid |
| 2 · crossover | 1 d | 2 h | G2 monotone break-even |
| 3 · capability | 1 d | 2 h | G3 ≥ 90% above window |
| 4 · cost/concurrency | 1.5 d | 2 h | G4 non-overlapping IQRs |
| 5 · release | 1.5 d | 30 min | G5 claim↔evidence map |

Total ≈ 7 working days, ≈ 8 GPU-hours. Hard stop at day 4.5 if G1 fails:
fall back to the oracle path and run Phases 2–5 unchanged (the crossover curve
is still the result; only the planner-cost input changes).

**Execution rule (per AGENTS.md)**: every GPU cell ≤ 5 min foreground, else
`nohup … > logs/<name>.log 2>&1 &` with a pre-registered kill time.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| G1 fails even with outlines | MED | Oracle path already proven; cost input becomes the measured oracle→model delta |
| Crossover shows LMPU never wins | LOW–MED | **This is a result.** Record it; do not tune to hide it |
| Capability result attacked as "just RAG" | HIGH | O7 chunked-RAG baseline is mandatory, not optional |
| Energy numbers re-litigated | MED | Interval reporting + explicit noise floor (§4 Phase 4) |
| Reviewer rejects "narrow" thesis | MED | Narrow and true beats broad and false; C6/C8 disclosure is the defence |

---

## 8. Definition of done

**Minimum viable salvage.** G2 and G3 pass with pre-registered gates; C1–C5
published with raw artifacts; C6–C8 disclosed. No claim rests on a point estimate
inside the noise floor.

**Full salvage.** Additionally G1 and G4 pass: planner ≥ 45 tok/s at ≥ 95%
validity, and LMPU cost-per-correct-task beats direct-CoT with non-overlapping
intervals on ≥ 2 of 3 amortized classes.

---

## 9. Commands

```bash
# G0 — harness decode speed (must be >= 70 tok/s)
uv run python -m lmpu.bench.run_benchmark --workload bigint_mul \
  --baseline lmpu.model_sync --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-tokens 256 --trials 3 --device cuda --output benchmarks/results/g0

# G1 — planner calibration (free vs constrained vs outlines)
uv run python -m lmpu.bench.calibrate --models 0.5B,1.5B \
  --budgets 128,256 --decoders free,lmfe,outlines --output benchmarks/results/g1

# G2 — the crossover curve (the headline)
uv run python -m lmpu.bench.run_benchmark \
  --workload sleep_1s --workload sleep_3s --workload sleep_30s \
  --workload batch_64 --workload batch_1024 \
  --baseline direct_cot --baseline lmpu.model_async --baseline lmpu.sync \
  --baseline native --trials 5 --device cuda --idle-seconds 5 \
  --output benchmarks/results/g2-crossover

# G3 — capability ladder
uv run python -m lmpu.bench.run_benchmark \
  --workload context_32k --workload context_64k \
  --workload context_128k --workload context_256k \
  --baseline direct_cot --baseline lmpu.sync --baseline chunked_rag \
  --trials 3 --device cuda --output benchmarks/results/g3-capability

# G4 — cost + concurrency
uv run python -m lmpu.bench.report --results benchmarks/results \
  --prices 0.05,0.15,0.30,0.50 --output benchmarks/reports/cost.md
```

---

## 10. One-line summary

> Fix the two implementation defects that disguise a real limit as a bug (Phase 0–1),
> then publish the **crossover curve** and the **capability cliff** (Phase 2–3) —
> because the honest, defensible LMPU claim is *capability plus amortized cost*,
> not per-call latency.
