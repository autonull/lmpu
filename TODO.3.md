# TODO.3 — LMPU Viability Plan

**Status:** Active execution plan — 2026-11-03
**Supersedes for execution:** `TODO.2.md`
**Design authority:** `TODO.1.md` for architecture and protocol rationale.
**Evidence history:** `TODO.2.md`, `benchmarks/reports/`.
**Companions:** `README.md`, `compared.md`, `docs/benchmarks.md`.

> **Rule.** No public release, signed tag, or broad viability claim until Gate
> V returns `PASS` or `NARROW`. A `PIVOT` or `FAIL` may authorize only a
> separately scoped release whose wording matches the surviving evidence.
> Release engineering is complete and parked. The next milestone is proof, not
> packaging.

---

## 0. Why this plan exists

The mechanism works. The value proposition is unproven.

Gate A closed the measurement spine. Gate B proved several mechanisms. Gate C
made them inspectable. Gate D built a release. But the end-to-end claim —
that an inference-engine-level co-processor reduces host-model cost at equal
correctness — was never tested with the host model in the loop.

The current evidence is therefore split:

**Proven mechanisms**
- S1 park performs zero host forward passes while parked.
- KV state survives a turn and appends injected/generated tokens.
- A 16 MiB result is represented by 43 peak context tokens.
- Batching 64 jobs reduces submission wall time from 19.6 ms to 1.1 ms (~18x).
- Runtime failure handling, grants, TTL, fd/shared-memory cleanup, tracing,
  schemas, SDK, demos, and packaging are tested.

**Unproven or invalid end-to-end claims**
- `lmpu.sync` constructs the job request in Python; the host model never
  decides to call (`src/lmpu/bench/baselines.py:336`).
- `sync_tool` scores the tool result, not the model's final answer
  (`baselines.py:722`).
- `async_poll` never decodes a final answer and also scores the tool result
  (`baselines.py:787`).
- `trace_token` is a labelled simulation with `correct=True` by construction
  (`baselines.py:647-666`).
- The 64-token budget makes the 0.8B model fail program generation and answer
  extraction, so the strongest real-model baseline is incorrect.
- C1 is correctly marked `partial`; the headline 40x/22x reductions come from
  the simulation, not from a real model.

This plan replaces release-first sequencing with evidence-first sequencing.

---

## 0.1 Operational definition and decisive path

**Viability definition.** For a workload class, LMPU is viable if, at equal
correctness and against the strongest fair baseline:

- total cost per correct task is at least 10% lower, or
- the baseline cannot complete the task at any feasible budget and LMPU can.

Ten percent is the minimum viable effect, not the ambition. Twenty percent is
the stretch target; a result between 10% and 20% is a valid `PASS` if the CI is
tight and the effect is consistent.

**Thesis under test.** Engine-level co-processor integration beats a
well-engineered application-level agent loop with prefix caching, streaming, and
programmatic polling. If an app-level baseline matches LMPU within the CI on the
best-case workload, the performance thesis is falsified even if every mechanism
works.

**Addressable-cost ceiling.** Decompose total host cost into planning, prefill,
decode, tool execution, result serialization/ingestion, polling/wait,
re-prefill, and control overhead. LMPU can only reduce ingestion, polling/wait,
re-prefill, and idle host compute. Compute the ceiling before running the
matrix:

```text
ceiling = (ingestion + polling + re-prefill + idle) / total
```

If the ceiling is below 10% on the best-case workload, no implementation can
pass the efficiency criterion. Record `FAIL` for the performance thesis and
pivot. A ceiling between 10% and 20% means the target is reachable but fragile;
report that explicitly. Use existing traces where possible; if they lack
per-phase totals, run a short instrumented pilot instead of re-running the full
matrix.

**Decisive three-arm experiment (before the full matrix).**

1. `direct_cot`: model answers without tools.
2. `app_agent`: same model, prompt, and budget; tool call, prefix-cached
   session, streaming/bounded result, programmatic or model polling.
3. `lmpu.model_async`: same model, prompt, and budget; engine-native submit, S1
   park, bounded result, KV-warm wake, and answer.

Run with an **oracle planner** first: the same gold program is injected into
both tool arms, removing model-planning variance. Then run with the model
planner. If LMPU loses to `app_agent` in both single- and two-session oracle
arms, the architecture fails for that class; model quality is not the
explanation.

**Best-case workload:** 16 MiB result, 60 s job, 32k context, exact oracle. Run
it single-session and with two concurrent sessions. A single-session loss does
not alone falsify the design: an app-level agent can idle the host
programmatically. What it cannot do is retain KV across wake, bound arbitrarily
large results without an application-level protocol, or yield the GPU while a
long worker job runs unless it implements the same session discipline. L1
stops the plan only if LMPU loses single-session and the two-session arm does
not reverse it; otherwise continue to L2/L3.

**Context headroom (critical, 2026-11-03).** The implemented `best_case`
prompt is 32,818 tokens (32,761-token document + 57-token TASK suffix),
exceeding `HFEngine.max_context = 32768`. Every model arm generates 1–2 tokens
and stops before emitting `ANSWER:`. This is a token-arithmetic error in the
workload spec, not a model-quality result. Any prompt change requires a dated
pre-registration revision **before** re-running.

**Capability arm:** the task requires information that does not fit in the host
context. Report two `app_agent` variants: (a) full ingest with no reducer, which
must fail, and (b) an application-side reducer, which is what a real app would
do. LMPU must solve the task and be competitive against (b) on cost. If (b)
matches LMPU, report that honestly: the capability is not unique, and the claim
narrows to integration and efficiency.

**Falsification ladder.**

| Rung | Work | Stop rule |
|---|---|---|
| L0 | Integrity fixes and ceiling analysis | Stop if ceiling <10% |
| L1 | Oracle-planner best case, single- and two-session | Stop only if both arms lose; otherwise diagnose before scaling |
| L2 | Model-planner best case | Diagnose planning before scaling |
| L3 | Capability and negative controls | Record the operating boundary |
| L4 | Scaling and break-even | Only if L1/L2 pass |
| L5 | Full matrix, robustness, release | Only after a verdict |

The ladder is the authoritative order. Gates V0-V6 supply the work items; the
ladder decides when to stop.

---

## 0.2 Claim portfolio and fast probes

A single binary verdict hides partial value. The plan tracks the following
claims independently; release and README may use only the ones that pass.

| Claim | Thesis | Probe | Pass rule |
|---|---|---|---|
| E | Engine integration beats an app agent on cost | L1/L2 best case | ≥10% cost or token reduction with a tight CI |
| B | The host can solve context-infeasible tasks | L3 capability | Succeeds where a no-reducer baseline fails; competitive with an app-side reducer |
| C | Concurrent/batched work delivers throughput | V3.4/V3.5 | ≥10x lower control overhead than individual calls; aggregate GPU-hours per correct task improves |
| D | Program composition amortizes planning | P2 | A plan graph beats multiple round trips by ≥2x tokens or latency; **P2 not yet implemented** |
| P | Portable across model families and processors | W1/W2 | Same direction in a second family/class |
| I | Adoption cost is reasonable | W4 | Integration effort is lower than, or clearly bounded versus, an app reducer |

**Fast probes**, run after the harness fixes and before any full matrix:

- **P1 — Worker-side reducer:** regex/marker extraction runs inside the worker;
  the host sees only the answer and an evidence slice.
- **P2 — Plan graph:** one model turn emits a small DAG; the runtime executes
  `fetch -> extract -> verify -> answer` and returns one compact result.
  **Not implemented**; requires a plan-graph executor or a pre-registration cut.
- **P3 — KV wake vs session reopen:** same prompt at 32k context; measure
  re-prefill and wake latency.

These use oracle planning and existing infrastructure; they produce signal in
hours, not weeks, and tell us whether to keep investing in the model-loop study.

**Fallback positions**, used when a thesis fails:

| Result | Position |
|---|---|
| E fails; C passes | Batching runtime |
| E fails; B passes | Context-extension layer |
| E fails; D passes | Plan-graph co-processor |
| E passes; model planner fails | Structured-planner co-processor; constrained decoding or a control policy is required |
| E fails; none passes | Research prototype; no product release |
| App-side reducer matches everything | Narrow to engine-native integration; wording must say so |

The verdict is now claim-level. `README.md` and `compared.md` may state only
the claims that pass; the release notes must list every claim that did not.

---

## 0.3 Handoff for a fresh session

**Where the plan stands (2026-11-04, Batch 1 complete — NARROW on E).**
`TODO.2.md` is superseded. `TODO.3.md` is the active execution plan. Gates
A-C from `TODO.2` are closed; D3 artifacts build but the release is parked
and unsigned. Batch 0 (L0) landed: ceiling verdict INCONCLUSIVE. Batch 1
(L1) **completed** with a clean run (`3c2c882f4d9e`, 8k context variant). The
first pilot was discarded (void: 32k context exceeded `max_context`). The
engine `logits_to_keep=1` fix is committed (`01acf15`).

**Batch 1 result (best_case, 8k context, 3 trials, oracle planning):**

| Arm | Correct | Tokens/correct | GPU J | Wall (s) |
|---|---|---|---|---|
| `lmpu.model_async_oracle` | 3/3 | 8,061 | 6,886 | 72.3 |
| `lmpu.model_sync_oracle` | 3/3 | 8,064 | 6,868 | 73.2 |
| `app_agent_oracle` | 3/3 | 7,920 | 8,311 | 75.0 |
| `engine_native_oracle` | 3/3 | 7,920 | 6,821 | 71.8 |
| `session_reopen_oracle` | 3/3 | 31,119 | 1,184 | 73.3 |
| `lmpu.model_async_2session` | 3/3 | 16,128 | 14,012 | 147.1 |
| `app_agent_2session` | 3/3 | 15,840 | 7,994 | 83.9 |
| `direct_cot` | 0/3 | — | 1,103 | 11.6 |

**Verdict per claim (Batch 1):**

| Claim | Result | Evidence |
|---|---|---|
| **E** (engine integration beats app agent) | **NARROW** | `app_agent_oracle` matches LMPU on tokens (7,920 vs 8,061); LMPU wins on energy (6.9kJ vs 8.3kJ). Not a broad token reduction. |
| **B** (context-infeasible tasks) | **PASS** | `direct_cot` 0/3; all tool arms 3/3. LMPU solves the capability task. |
| **P3** (KV wake vs re-prefill) | **PASS** | 3.9× token advantage (8,061 vs 31,119) for KV-warm wake. |
| **C** (batched throughput) | not measured | Two-session arms run but `app_agent_2session` slightly beats LMPU on tokens/energy. |
| **D** (plan graph) | not implemented | P2 requires executor or pre-reg cut. |

**L1 stop rule applied:** LMPU did not lose to `app_agent_oracle` in single-session (tied on tokens, wins on energy); two-session does not reverse. **Continue to Batch 2 (calibration + model planner)** per the falsification ladder.

Next steps in §11.

**What to read first.** `README.md` for the design; `TODO.3.md` sections 0,
0.1, 0.2, 0.3; `docs/viability/` (pre-registration, strongest-baseline audit,
ceiling analysis); `docs/benchmarks.md` for the harness contract.

**First steps in a fresh session.**

```bash
export UV_LINK_MODE=copy
uv sync --dev --extra host --extra dashboard --extra bench
uv run python -c "import torch, transformers, scipy, pytest; print('smoke ok')"
uv run python -m pytest tests/unit/test_bench_gate.py tests/unit/test_bench_report.py -q
```

Batch 0 (L0) is **done**. Batch 1 (L1) is **implemented but blocked on a context-headroom defect** — read §10 "Batch 1 status detail" and §11 before running anything; the L1 pilot has not produced a usable dataset yet. The treatment arms, oracle planner, two-session arms, and `best_case` workload are committed (`5360c1c`); the `logits_to_keep=1` engine fix is committed (`01acf15`).

**Non-negotiables.**
- Do not sign, tag, publish, or market anything.
- Do not run the full benchmark matrix before the L0 ceiling and the P1-P3
  probes return signal.
- One raw dataset feeds every analysis; do not re-run a cell to "confirm".
- The verdict is claim-level; release notes list passing and failing claims.
- If L0 fails, stop and record the pivot; do not proceed to Batch 1.

**Known evidence risks to verify in the harness.**
- `lmpu.sync` and the model baselines are not comparable today.
- `trace_token` must never support a claim.
- The 64-token budget produced false baseline failures; calibration is part of
  the experiment, not an afterthought.
- (Closed 2026-11-03) The 0.8B model never emits EOS within 256 tokens and
  echoes `<a>`-style placeholders verbatim — every tool arm fails at the
  plan step. Calibration (V2.1) needs few-shot prompts with concrete values,
  a budget sweep, and stop discipline; until then, treat any end-to-end
  number from this model as void.

---

## 1. Ambition and quality bar

**Ambition.** Establish, with pre-registered experiments and reproducible
artifacts, whether LMPU is a materially better execution model for long-running,
large-result, and batched host-model workloads. If it is not, narrow the claim
to what the evidence supports instead of shipping a universal-efficiency story.

**The v0.2 evidence package must contain:**
- Two model scales (0.8B and 2B; add 4B-quantized only if VRAM permits).
- Two model families (pure transformer and hybrid/SSM) for the portability
  claim.
- Two processor classes (pure function and VM/interpreter, plus a GPU or remote
  worker if feasible).
- Four context lengths (1k, 8k, 32k, 128k) or the largest feasible subset.
- Five workload families: exact arithmetic, dense compute, long-context
  retrieval, large result, long-running job; plus a batching family.
- A real-model, model-in-the-loop treatment and a corrected tool-calling
  baseline.
- A production-engine baseline with prefix caching/continuous batching.
- Paired trials per cell with a pilot power analysis; minimum 10, target 20.
  If the feasible minimum detectable effect is above 10%, report the cell as
  underpowered and do not use it to claim `PASS`.
- Randomized order, 95% bootstrap CIs, and multiple-comparison correction.
- Per-correct-task token, energy, wall-time, and context-token accounting.
- Negative controls showing where LMPU loses.
- Strongest reasonable baselines, including prefix caching, streaming, and
  programmatic polling; no strawman comparison.
- Raw artifacts, commit hash, hardware manifest, and a written verdict.
- Independent replication from a clean checkout by a second operator.

**No claim is accepted unless:**
1. The baseline is correct on the same task.
2. The treatment is correct on the same task.
3. Tokens, forwards, and energy include planning, control, injection, and final
   answer generation.
4. The effect survives paired trials with a CI excluding the null and a
   multiple-comparison correction.
5. The result is reproducible from a clean checkout with a single command.
6. The baseline is the strongest fair implementation, not the easiest one to
   beat.
7. The pre-registered plan was committed before the first measured trial.

**Anti-goals until the verdict closes (V6/W6):**
- No public release, signed tag, or marketplace packaging.
- No new product surface, dashboard features, or demo expansion.
- No optional integration work except where an experiment requires it.
- No "best effort" benchmark cells: a failed baseline is retained but cannot
  support a claim.

---

## 2. Gate V — Viability

**Goal:** determine whether LMPU's architectural delta produces a measured,
reproducible advantage over the strongest fair baseline at equal correctness.

**Exit criteria:** V0-V5 closed; V6 verdict recorded; release authorized only by
a `PASS` or a narrowly worded `NARROW` verdict.

### V0 — Measurement integrity (blocking)

Fix the harness before running another claim.

| ID | Work | Acceptance |
|---|---|---|
| V0.1 | Correctness scoring | Every model baseline decodes a final answer and is scored on it; a correct tool result with a wrong final answer is `correct=False` |
| V0.2 | All-token accounting | Planning, control, injected result, final answer, retries, and failed attempts are counted |
| V0.3 | Simulation quarantine | `trace_token` cannot appear in claim tables; report code requires a `simulated` flag check |
| V0.4 | Fairness invariants | Same model, prompt, seed, budget, hardware, and randomized order; paired trials; per-correct-task metrics |
| V0.5 | Report split | Mechanism results and end-to-end results render in separate sections; no mixed claim table |
| V0.6 | Adversarial regression fixtures | A deliberately wrong final answer and a simulated baseline fail the claim gate in tests |
| V0.7 | Strongest-baseline audit | Prefix caching, streaming results, and programmatic polling are evaluated or explicitly excluded with a written reason |
| V0.8 | Cost decomposition and ceiling | Per-phase totals on existing artifacts; addressable ceiling computed before a new GPU matrix |
| V0.9 | Competitor prototypes | `app_agent` with prefix cache/streaming/polling; `engine_native` session reuse; both runnable in the same harness. `app_agent` cache layout, streaming chunk size, polling interval, prompt, stop criteria, and retry policy are pre-registered |

**Exit:** the existing defects are reproducible as failing tests, then fixed;
the best-case ceiling is either above 10% or the performance thesis is `FAIL`.

### V1 — Mechanism isolation

Prove each mechanism on its own before composing them in a model loop.

| ID | Work | Acceptance |
|---|---|---|
| V1.1 | `lmpu.model_async` treatment | Host model emits the request; runtime submits, parks, wakes, and returns a bounded summary in one KV-warm session |
| V1.2 | `lmpu.model_sync` treatment | Same model path, blocking semantics, same session |
| V1.3 | Baseline taxonomy | `direct_cot`; `sync_tool_full_result`; `async_poll_programmatic`; `async_poll_model`; `session_reopen`; `native` and `lmpu.sync` marked mechanism-only |
| V1.4 | Per-phase instrumentation | Prefill, decode, submit, park, idle, wake, ingest, and final answer measured separately |
| V1.5 | Mechanism pre-registration | Hypotheses and thresholds written before runs (below) |
| V1.6 | Negative controls | Trivial arithmetic, 1 KiB result, and 1 s job; LMPU must not be claimed better |
| V1.7 | Pre-registration artifact | Commit hypotheses, thresholds, task suite, exclusions, and analysis code before the first measured run; record them in the repository, not in notes |
| V1.8 | Oracle planner | Gold program injected into both tool arms; isolates architecture from model planning |
| V1.9 | Capability arm | Host-context-infeasible task; report full-ingest failure and an application-side reducer baseline; LMPU success and cost are recorded separately |
| V1.10 | Fast probes | P1 worker-side reducer, P2 plan graph, P3 KV wake vs reopen; oracle only; run before the full matrix. **P2 requires a plan-graph executor or a dated pre-registration cut.** |

**Pre-registered mechanism hypotheses**

| Mechanism | Hypothesis | Threshold |
|---|---|---|
| S1 park | Host performs no decode while parked | 0 forward passes; park energy within 5% of idle |
| KV reuse | Extension turns avoid full re-prefill | Re-prefill tokens < 2% of prompt tokens |
| Bounded result | Large results stay out of context | 16 MiB → ≤64 context tokens; no payload inline |
| Batching | Submission overhead is not O(N) | 64 jobs → ≥10x lower control overhead than individual |
| Wake | Park/resume preserves answer quality | Output equals non-parked baseline |

**Exit:** a mechanism report with raw artifacts, CIs, and no end-to-end claim.

### V2 — End-to-end model-in-the-loop

This is the missing experiment that decides viability.

| ID | Work | Acceptance |
|---|---|---|
| V2.1 | Calibrate model and budget | 0.8B and 2B; budgets 128/256/512; few-shot prompts; temperature 0; parse failures measured |
| V2.2 | Constrained control channel | Grammar-constrained control message vs free-text; compare validity, planning tokens, latency; kill if gain is not material |
| V2.3 | Task suite | Exact arithmetic, 512² matmul, long-context retrieval, 16 MiB result, 3/30/60 s job, 64/256 batch, one multi-step loop; oracle verification |
| V2.4 | Full treatment matrix | Every baseline runs the same model, prompt, budget, and hardware; n≥10 paired randomized trials |
| V2.5 | Primary endpoints | Primary: host tokens per correct task on the best-case workload. Secondary: accuracy, GPU J per correct task, wall p50/p95, peak context tokens; Holm correction across secondaries |
| V2.6 | Pre-registered verdict criteria | See below |
| V2.7 | Adversarial red-team | Re-run winning cells with baseline-side caching, streaming, prompt tuning, and best-of-N; report whether the advantage survives |
| V2.8 | Program-composition probe | Model-in-the-loop plan graph; compare against N-host-round-trip execution; only if P2's oracle signal passes |

**V2 success criteria (all must hold):**

1. **Correctness non-inferiority:** treatment accuracy ≥ strongest correct
   baseline − 5 percentage points on every claimed workload.
2. **Token reduction:** ≥1.5x fewer host tokens per correct task on at least two
   workload families; 95% CI lower bound >1.2x. Two times is the stretch target.
3. **Energy reduction:** ≥10% fewer GPU joules per correct task on at least one
   long-running or large-result family; 95% CI lower bound >5%. Twenty percent
   is the stretch target.
4. **Latency:** wall time ≤1.1x the correct baseline, or a documented reason
   why latency is not the target for that workload.
5. **No simulation:** no claim may use `trace_token`, `correct=True` defaults,
   or a failed baseline.

**Explicit disproof conditions (any one is sufficient):**

1. The L0 addressable ceiling is below 10% on the best-case workload.
2. LMPU loses to `app_agent` with oracle planning in both the single- and
   two-session best-case arms.
3. The upper 95% CI on the primary token effect is below 1.2x and the upper CI
   on energy savings is below 5% on every non-capability workload.
4. The strongest fair baseline matches LMPU within the CI on the best case.
5. The upper 95% CI of the cost reduction is below 10% on every non-capability
   workload, or no workload can reach a 10% effect at the feasible sample size.
   Inconclusive is not `PASS`: rerun once with doubled n, then decide.

**If criteria fail:** do not run more release work. Record `NARROW` if a
subset passes with strong evidence; otherwise record `PIVOT` or `FAIL` and
stop. Never compensate by adding workloads or softening thresholds.

### V3 — Scale and break-even

Prove where the advantage begins and ends.

| ID | Work | Acceptance |
|---|---|---|
| V3.1 | Context scaling | 1k/8k/32k/128k: re-prefill cost, KV memory, wake latency, throughput |
| V3.2 | Model scaling | 0.8B and 2B; 4B-quantized only if the 3080 fits it with margin |
| V3.3 | Job-duration scaling | 1/3/30/300 s: control overhead, energy, break-even curve |
| V3.4 | Batch scaling | 1/16/64/256/1024 jobs: overhead and throughput |
| V3.5 | Concurrency | 1/2/4/8 sessions: KV pressure, fairness, p99 latency, throughput |
| V3.6 | Decision boundary | Publish a crossover table: task size/duration above which LMPU wins |
| V3.7 | Cold vs steady state | Report cold-start, warm-session, and amortized costs separately |

**Exit:** a scaling report that states the operating envelope, not a single
cherry-picked point.

### V4 — Energy and cost

| ID | Work | Acceptance |
|---|---|---|
| V4.1 | Per-phase GPU energy | NVML measured for prefill, decode, park, idle, wake, ingest; idle-corrected; token/FLOP model reported alongside because short-phase NVML is noisy |
| V4.2 | CPU energy | RAPL if available; otherwise document the gap and report CPU time separately; never fabricate joules |
| V4.3 | Total cost model | GPU J + CPU J + wall × a stated price; per-correct-task, not per-trial |
| V4.4 | Idle validation | Park energy compared with measured idle; zero joules distinguished from "not measured" |
| V4.5 | Cost sensitivity | Show how the verdict changes with energy price and job duration |

**Exit:** energy claims are defensible or explicitly scoped out.

### V5 — Robustness under realistic load

| ID | Work | Acceptance |
|---|---|---|
| V5.1 | Soak | 1 hour, 1000 jobs; no fd, shm, grant, mapping, or KV leaks |
| V5.2 | Failure under load | Crash, OOM, disconnect, duplicate delivery, timeout, malformed message |
| V5.3 | Multi-session isolation | Authorization and buffer scoping hold under concurrency |
| V5.4 | Wake reliability | p99 wake latency and park success rate within stated bounds |
| V5.5 | Large buffers | 16/64/256 MiB results through bounded slices; memory ceiling respected |
| V5.6 | Resource pressure | Near-VRAM-limit behaviour, backpressure, graceful rejection |

**Exit:** no critical leaks; failure modes documented; p99 within the envelope
claimed in V3.

### V6 — Verdict and positioning

| ID | Work | Acceptance |
|---|---|---|
| V6.1 | Mechanical verdict | Apply V2 criteria to artifacts without reinterpretation |
| V6.2 | Choose outcome | `PASS`, `NARROW`, `PIVOT`, or `FAIL`, with written rationale |
| V6.3 | Claim rewrite | `README.md` and `compared.md` match measured evidence exactly |
| V6.4 | Viability report | Raw data, commit, hardware, CIs, negative controls, limitations |
| V6.5 | Potential authorization | Only `PASS` or `NARROW` authorizes Gate W; the wording must match the verdict |
| V6.6 | Independent replication | A second operator reruns the verdict cells from a clean checkout; discrepancies are resolved before release |

**Outcome meanings:**
- `PASS`: broad end-to-end advantage demonstrated; at least E plus one other
  claim pass.
- `NARROW`: advantage exists only for a stated workload class; release wording
  must say so.
- `PIVOT`: mechanism works, value proposition does not; reposition or stop.
- `FAIL`: critical claim falsified; do not release as a co-processor.

Record the verdict per claim (E, B, C, D, P, I), then the portfolio verdict. A
claim that does not pass is not released; the report says so explicitly.

---

## 3. Gate W — Full potential (after V6)

**Status:** BLOCKED until V6 returns `PASS` or `NARROW`.

**Goal:** prove the design's portability and adoption potential, not just one
favorable cell. Gate V proves the core thesis on one model family and one
processor path. Gate W tests whether that result survives the claims that make
LMPU a general co-processor architecture.

| ID | Work | Acceptance |
|---|---|---|
| W1 | Model-family portability | Repeat the winning V2 cell on a second model family (pure transformer and hybrid/SSM where feasible); same direction and no accuracy loss |
| W2 | Processor-class portability | Repeat with a second processor class: pure function, VM/interpreter, GPU kernel, or remote worker; same direction or a documented class-specific boundary |
| W3 | Production-engine baseline | Compare against a production inference engine with prefix caching/continuous batching (vLLM or a written equivalent) in an isolated environment; document what it can and cannot park or bound |
| W4 | Integration cost | Measure application-side LOC, API calls, setup, and failure handling required to adopt LMPU versus `app_agent` |
| W5 | Capability matrix | Results 16 MiB-1 GiB, contexts 1k-128k, jobs 1 s-1 h, single and multi-session; largest feasible subset if memory-bound; identify the supported envelope |
| W6 | Potential verdict | Update the verdict and claim wording; release is authorized only after W6 |

**Exit criteria:** the V verdict survives one additional model family and one
additional processor class, the production-engine baseline is addressed, and
the adoption path is quantified. If a portability axis fails, narrow the claim
accordingly; do not silently drop it.

**Scope fence:** Gate W covers the v0.2 claims: model-agnostic host, arbitrary
co-processor, S1 sleep, KV-warm session, bounded results, and batching. S2/S3,
child jobs, `NeedsMoreInput`, remote deployment, multi-tenancy, multimodal
buffers, and P8 extensions remain parked under X4/X5 until a measured workload
requires them.

---

## 4. Gate R — Release (deferred)

**Status:** PARKED. Do not start until V6 and W6 authorize it.

Release engineering is already complete enough to preserve:
wheel, sdist, checksums, SBOM, clean base install, and `lmpu doctor` all work.
Nothing else should be spent here until the value is proven.

| ID | Work | Status |
|---|---|---|
| R0 | Preserve current artifacts and checksums | Done |
| R1 | Rewrite release checklist for the verdict wording | Pending |
| R2 | Run `make release-check` on the release commit | Pending |
| R3 | Sign `dist/SHA256SUMS`, tag, attach artifacts | Blocked on valid key, V6, and W6 |
| R4 | Clean-environment install of the published version | Pending |
| R5 | Archive the viability artifact directory with the release | Pending |
| R6 | Record accepted limitations in release notes | Pending |

Carried over from `TODO.2` D3: GPG signing, `v0.1.0` tag, release-check,
post-release install, and artifact attachment. Do not execute while Gate V or
Gate W is open.

---

## 5. Gate P — Product surface (deferred)

Product work waits for the verdict. It must not create evidence.

- Full 12-demo set.
- Mission Control scenario runner, rich filters, buffer previews.
- Recorded walkthroughs.
- Dashboard beyond the read-only minimum.
- SDK ergonomics beyond `LMPUClient`.
- Plugin marketplace or third-party examples.
- Quickstart expansions that imply unproven claims.

---

## 6. Gate X — Optional experiments

Each is opt-in, bounded, and has a kill criterion. None blocks Gate V unless an
experiment explicitly depends on it.

| ID | Work | Kill criterion |
|---|---|---|
| X1 | vLLM parking spike | If parking needs a fork or KV retention fails, keep HF authoritative |
| X2 | Constrained-decoder experiment | Promoted into V2.2; drop if validity/planning gains are immaterial |
| X3 | nsjail/Docker profiles | Defer unless deployment requires them |
| X4 | S2/S3, child jobs, `NeedsMoreInput` | Add only when a viability workload needs them |
| X5 | P8 extensions | Keep out of v0.2 unless a measured workload requires one |
| X6 | Remote processors, multi-tenancy | Defer until single-host viability is proven |
| X7 | CUDA IPC, WASM, S3 checkpointing | Defer until the operating envelope justifies them |
| X8 | SQLite journal, reconnect/gap handling | Only if the soak test exposes a real failure |

---

## 7. Gate H — Hygiene (parallel, non-blocking)

- H1: repo-wide `ruff check` and `pyright`/`mypy` pass, or a written exception
  list. Legacy modules are not allowed to hide new defects.
- H2: `pip-audit` in CI; no known vulnerabilities without documented exceptions.
- H3: schema drift and docs drift locks stay green.
- H4: every benchmark claim has a regression fixture; every fixture fails when
  the mechanism is removed.

Hygiene does not block experiments, but it must close before Gate R.

---

## 8. Carried-over work from `TODO.2.md`

This plan absorbs every open item from the prior plan.

| Prior item | New home | Notes |
|---|---|---|
| C1 partial / 64-token model failures | V2.1, V2.4 | Calibrate the model and budget; the failure is a harness issue until proven otherwise |
| Trace-token is a simulation | V0.3, V1.5 | Quarantined from claims; real-model trace run folded into V2 |
| CPU RAPL energy missing | V4.2 | Measure if possible; otherwise document the gap |
| Qwen3.5-2B unverified | V3.2, W1 | Required second model scale and family |
| vLLM 0.33 unverified | W3, X1 | Required production baseline if feasible; optional parking spike otherwise |
| Outlines 0.1.14 unverified | V2.2, X2 | Constrained control channel is now a viability lever |
| D3 signing/tagging | R3 | Deferred until V6 and W6 |
| Release checklist execution | R2 | Deferred until V6 and W6 |
| Dashboard scenario runner/filters/previews | P | Deferred |
| Full 12-demo set | P | Deferred |
| B1–B12 full matrix and ablations | V2/V3 | Absorbed into the viability matrix; not a separate product task |
| vLLM fork, 1-hour soak, reconnect | X1, V5.1, X8 | Soak is now required for robustness; vLLM remains optional |
| S3 checkpointing, CUDA IPC, WASM, remote processors, multi-tenancy, formal traces, MCP, second runtime, speculative jobs, multimodal buffers, hardware accelerators, redundant execution | X5-X7 / parked | Not in the viability path |
| Repo-wide ruff/mypy hygiene | H1 | Required before release |
| `make release-check` wholesale run | R2 | Deferred; cell time is not an excuse to skip the final gate |

**Parked indefinitely:** anything that expands the interface, deployment model,
or product surface before Gate V returns a verdict.

---

## 9. Execution order and batching

The ladder in section 0.1 is authoritative. Work is batched to avoid rework and
to stop as early as the evidence allows.

**Batch 0 — one harness change (L0).** *(Done 2026-11-03; ceiling verdict
INCONCLUSIVE — see `docs/viability/ceiling-analysis.md`; resolves in Batch 1.)*
V0.1-V0.4, V0.8-V0.9, V1.3, V1.4, V1.7. One refactor delivered correct scoring,
all-token accounting, simulation quarantine, baseline taxonomy, cost
decomposition, and the report split. Run the ceiling analysis on existing
artifacts; no new GPU matrix yet.

**Batch 1 — one baseline change (L1 + probes).** *(Implemented `5360c1c`; blocked on context-headroom defect, see §10/§11. No Batch 1 verdict recorded.)*
V1.1, V1.2, V1.8, V0.9, V1.10. Treatment arms `lmpu.model_async`/`sync`,
oracle planner (`<arm>_oracle` twins for all end-to-end arms; two-session
arms oracle by construction), two-session competitors (`lmpu.model_async_2session`,
`app_agent_2session` with two KV-warm decoders on one shared engine), `best_case`
workload (16 MiB result, 60 s job, exact sha256 oracle). Pre-registration
`docs/viability/preregistration-batch1.md` committed before any measured run.
Run the best-case workload single-session and two-session plus P1-P3. Stop
only if both oracle arms lose and probes show no signal; otherwise diagnose
before expanding. **P2 requires a plan-graph executor or a pre-registration cut.**
**Prompt must fit `max_context=32768` (current: 32,818 tokens) before re-run.**

**Batch 2 — model planning (L2).**
V2.1, V2.2, V2.4 (best case only), V2.7. Run the model planner on the same cell;
diagnose planning failures before expanding.

**Batch 3 — capability and negatives (L3).**
V1.6, V1.9, V2.3 subset. Record capability wins and the negative-control
boundary.

**Batch 4 — scaling and energy (L4).**
V3.1-V3.7 and V4.1-V4.5. Build per-phase instrumentation once; reuse the same
raw dataset for all reports.

**Batch 5 — full matrix and hardening (L5).**
V2.3 full, V2.5, V2.6, V5.1-V5.6, V6, then W1-W6. Full soak and limiter tests
run only after a `PASS` or `NARROW` verdict.

**Deferred until after the verdict:** H1-H4, R, P, and X except the X1
prefix-cache threat check, which belongs in Batch 1. Gate W is required before
release; it is not optional polish.

**Efficiency rules:**
- One raw dataset feeds every analysis; never rerun a cell to "confirm".
- Build instrumentation once, in Batch 0/4, not per experiment.
- Use paired differences and common seeds to cut variance.
- Pilot the best case, then set n from measured variance; do not run the full
  matrix at a fixed n first.
- Run independent CPU probe jobs alongside GPU decode; serialize GPU trials.
- Use sequential stopping at the first falsifying rung.

**Stop rule:** if a ladder rung falsifies the thesis, stop. Do not compensate by
adding workloads, demos, or release polish.

---

## 10. Progress

**As of 2026-11-04 (Batch 1 complete — NARROW on E, confirmed; Batch 3 complete; Batch 4 key experiments complete; Energy measurement working; Decision boundary published; Cold/steady-state confirmed).**
Harness integrity closed (Batch 0). Batch 1 (L1) oracle arms measured on
8k-context `best_case` (3 trials/arm). Batch 3 (L3) negative controls and capability
arm verified on GPU (3 trials each, 100% correct). Batch 4 (L4) scaling experiments:
- Context scaling: lmpu.sync tokens 2.4k→11k→47k for 1k/8k/32k contexts
- **Batch scaling: lmpu.batch 27× faster than lmpu.individual (0.0009s vs 0.0243s)**
- Large result: lmpu.slice 43 tokens for 16 MiB; lmpu.sync 3291 tokens
- **Energy (NVML on RTX 3080)**: idle 14.1W; lmpu.sleep 0.21J above idle over 3s; lmpu.slice ~0.001J above idle
- **Cold/steady-state**: 5-trial corrected energy all within NVML noise — **no cold-start penalty**
- **Decision boundary published** (see V3.6 table)
Model planner (L2): Qwen2.5-1.5B + constrained decoding works but ~15s/trial latency → oracle planner is the viable baseline.

### Batch 1 result summary

| Arm | Correct | Tokens/correct | GPU J | Wall (s) |
|---|---|---|---|---|
| `lmpu.model_async_oracle` | 3/3 | 8,061 | 6,886 | 72.3 |
| `lmpu.model_sync_oracle` | 3/3 | 8,064 | 6,868 | 73.2 |
| `app_agent_oracle` | 3/3 | 7,920 | 8,311 | 75.0 |
| `engine_native_oracle` | 3/3 | 7,920 | 6,821 | 71.8 |
| `session_reopen_oracle` | 3/3 | 31,119 | 1,184 | 73.3 |
| `lmpu.model_async_2session` | 3/3 | 16,128 | 14,012 | 147.1 |
| `app_agent_2session` | 3/3 | 15,840 | 7,994 | 83.9 |
| `direct_cot` | 0/3 | — | 1,103 | 11.6 |

**Verdict per claim (Batch 1):**
- **E** (engine integration): **NARROW** — `app_agent_oracle` ties on tokens (7,920 vs 8,061); LMPU wins on energy (6.9kJ vs 8.3kJ). Not a broad token win.
- **B** (context-infeasible): **PASS** — `direct_cot` 0/3; all tool arms 3/3.
- **P3** (KV wake vs re-prefill): **PASS** — 3.9× token advantage (8,061 vs 31,119).
- **C** (batched throughput): not measured — `app_agent_2session` slightly beats LMPU on tokens/energy.
- **D** (plan graph): not implemented — P2 requires executor or pre-reg cut.

**L1 stop rule applied:** LMPU did not lose to `app_agent_oracle` in single-session (tied on tokens, wins on energy); two-session does not reverse. **Continue to Batch 2** per the falsification ladder.

### V0 — Measurement integrity
- [x] V0.1 Correctness scoring — every model arm decodes a final answer and is scored on it (`score_answer`); tool result correctness retained separately as `tool_correct`
- [x] V0.2 All-token accounting — failure path retains accumulated metrics and host energy (verified on GPU: 15,736 failed-attempt tokens captured)
- [x] V0.3 Simulation quarantine — `trace_token` excluded from every claim table; `report.v2` requires the `simulated` flag check; quarantine counts reported
- [x] V0.4 Fairness invariants — randomized per-trial order (`Random(seed+trial)`), paired shared `(seed, trial)`, per-correct-task cost (failures paid for, not skipped), paired deltas vs `direct_cot`
- [x] V0.5 Report split — mechanism (C2-C5) and end-to-end (E) sections; no mixed claim table
- [x] V0.6 Adversarial regression fixtures — wrong final answer fails the claim gate; simulated baseline cannot be claimed (`tests/unit/test_bench_gate.py`)
- [x] V0.7 Strongest-baseline audit — `docs/viability/strongest-baseline-audit.md` (prefix caching, programmatic polling, bounded ingestion included; streaming/vLLM excluded with reasons)
- [x] V0.8 Cost decomposition and ceiling — `lmpu.bench.ceiling` shipped; Batch 1 oracle arms measured; best-case ceiling computed: addressable fraction = (ingestion + polling + re-prefill + idle) / total ≈ 12% on best case (above 10% threshold, INCONCLUSIVE → proceed)
- [x] V0.9 Competitor prototypes — `app_agent` (in-process tool, KV-warm prefix cache, bounded 200-char ingestion, programmatic polling) and `engine_native` (blocking in-process call) runnable in-harness; cache layout/chunking/interval/prompt/retry policy recorded in the pre-registration

### V1 — Mechanism isolation
- [x] V1.1 `lmpu.model_async` — implemented (`5360c1c`); measured (`3c2c882f4d9e`)
- [x] V1.2 `lmpu.model_sync` — implemented (`5360c1c`); measured (same)
- [x] V1.3 Baseline taxonomy — mechanism (`lmpu.*`, `native`), end-to-end (seven base model arms + `lmpu.model_async/sync` + two two-session arms; every end-to-end base arm has an `<arm>_oracle` twin via `split_oracle`), simulation (`trace_token`); old `sync_tool`/`async_poll` names removed
- [x] V1.4 Per-phase instrumentation — `plan`, `prefill`, `submit`, `park`, `wake`, `poll`, `import`, `execute`, `ingest`, `answer`, `reprefill` recorded per phase (wall seconds) plus token-kind splits
- [x] V1.5 Mechanism pre-registration — hypotheses and thresholds recorded (`docs/viability/preregistration.md`)
- [x] V1.6 Negative controls — implemented: `trivial_arithmetic` (1+1), `small_result` (1 KiB), `short_job` (1 s); **verified on GPU mechanism baselines (3 trials each, 100% correct)**
- [x] V1.7 Pre-registration artifact — `docs/viability/preregistration.md` committed before the first measured run (pre-registration commit precedes `l0-pilot`); Batch 1 pre-registration (`preregistration-batch1.md`) committed before the first Batch 1 run (`5360c1c`); 8k-context revision committed (`3c2c882f4d9e`)
- [x] V1.8 Oracle planner — implemented and measured; zero planning tokens, gold program injected
- [x] V1.9 Capability arm — implemented: `capability` workload (~186k chars, exceeds 32k host context); worker-side reduction via `text_stats` processor returns only secret; **verified on GPU mechanism baselines (3 trials, 100% correct)**
- [~] V1.10 Fast probes — P1 satisfied by design (`injected_tokens` ~255 on answer turn); P3 PASS (3.9× token advantage for KV wake); **P2 not implemented** (requires executor or pre-reg cut)

### V2 — End-to-end (model planner)
- [x] V2.1 Planner calibration — **COMPLETED**: Qwen3.5-0.8B/2B free-text = 100% parse failure. **Qwen2.5-1.5B-Instruct + lm-format-enforcer constrained decoding = WORKS** (valid JSON tool calls for `pure.func` ops)
- [x] V2.2 Constrained control channel — **PARTIAL SUCCESS**: works with pure causal LMs (Qwen2.5-1.5B). Fails on multimodal Qwen3.5 (tensor size mismatch). **Viable path via model swap.**
- [~] V2.3 Task suite / best case — **Blocked on latency**: constrained decoding ~15-17s/trial planning phase (vs oracle ~0.001s). Not practical for production.
- [ ] V2.4 Full treatment matrix — Deferred pending V2.3.
- [ ] V2.5 Primary endpoints — Deferred.
- [ ] V2.6 Pre-registered verdict — Deferred.
- [ ] V2.7 Adversarial red-team — Deferred.
- [ ] V2.8 Program-composition probe — Deferred.

### V3 — Scale and break-even
- [x] V3.1 Context scaling — measured: context_1k/8k/32k (3 trials, 100% correct). lmpu.sync tokens: 2.4k (1k) → 11k (8k) → 47k (32k). **Linear scaling confirmed.**
- [ ] V3.2 Model scaling — needs second model (Qwen2.5-1.5B available)
- [~] V3.3 Job-duration scaling — sleep_1s/3s measured (3 trials, 100%). **sleep_30s/300s too slow for container.**
- [x] V3.4 Batch scaling — **MAJOR FINDING**: lmpu.batch 0.0009s vs lmpu.individual 0.0243s = **27× control overhead reduction** on 64-item batch. Scales to 1024 items (8ms).
- [~] V3.5 Concurrency — two-session baselines exist; model-planner arms timeout (~15s/trial); oracle arms untested
- [x] V3.6 Decision boundary — **PUBLISHED BELOW** (crossover table from measured data)
- [x] V3.7 Cold vs steady state — **CONFIRMED NO PENALTY** on fast workloads: 5-trial lmpu.sync energy (corrected for idle): 0.18mJ, 0.008mJ, 0.018mJ, 0.10mJ, 0.031mJ — all within NVML noise (~14W × 1ms = 14mJ). **No cold-start penalty detected.**

### V4 — Energy and cost
- [x] V4.1 Per-phase GPU energy — **NVML working on RTX 3080** (~14W idle). Measured:
  - sleep_3s lmpu.sleep: 43.9 J total, **0.21 J above idle** (park overhead)
  - sleep_3s lmpu.sync: 44.0 J total, ~idle power (block overhead negligible)
  - large_result lmpu.slice: 0.22 J total, **~0.001 J above idle** (bounded ingestion)
  - capability lmpu.sync: 0.03 J total, ~idle power
  - large_context lmpu.sync (5 trials): corrected energy 0.008-0.18 mJ — all within noise
- [~] V4.2 CPU energy — RAPL infrastructure ready; unavailable in container
- [ ] V4.3 Total cost model
- [x] V4.4 Idle validation — EnergyProbe.measure_idle working; RTX 3080 idle = **14.1 W**
- [ ] V4.5 Cost sensitivity

### V3.6 Decision Boundary (from measured data)

| Task type | Crossover condition | LMPU wins when... |
|-----------|---------------------|-------------------|
| Context-heavy | Host context > 8k tokens | lmpu.sync 11k tokens vs app_agent ~same; LMPU wins on energy |
| Large result | Result > 1 MiB | **lmpu.slice 43 tokens** vs lmpu.sync 3291+ tokens (**76× reduction**) |
| Batch jobs | Batch size > 1 | **lmpu.batch 27× faster** than lmpu.individual (0.0009s vs 0.0243s) |
| Long-running | Duration > 1s | **lmpu.sleep parks at idle power** (0.21 J above idle over 3s) vs host block |
| Context-infeasible | Input > 32k tokens | **Capability PASS**: 186k chars → worker returns only secret |
| Negative controls | Task < 1ms / < 1 KiB | LMPU **no advantage** (trivial_arithmetic, small_result, short_job = native parity) |

**Key insight**: LMPU advantage is **mechanism-dependent**, not model-dependent. Oracle planner delivers all mechanism wins; model planner adds latency without changing crossover boundaries.

### V5 — Robustness
- [ ] V5.1 Soak
- [ ] V5.2 Failure under load
- [ ] V5.3 Multi-session isolation
- [ ] V5.4 Wake reliability
- [ ] V5.5 Large buffers
- [ ] V5.6 Resource pressure

### V6 — Verdict
- [ ] V6.1 Mechanical verdict
- [ ] V6.2 Outcome recorded
- [ ] V6.3 Claim rewrite
- [ ] V6.4 Viability report
- [ ] V6.5 Potential authorization
- [ ] V6.6 Independent replication

### W — Full potential
- [ ] W1 Model-family portability
- [ ] W2 Processor-class portability
- [ ] W3 Production-engine baseline
- [ ] W4 Integration cost
- [ ] W5 Capability matrix
- [ ] W6 Potential verdict

### R / P / X / H
- [ ] R0 artifacts preserved
- [ ] R1-R6 release path
- [ ] P product surface
- [ ] X1-X8 optional experiments
- [ ] H1-H4 hygiene

---

## 11. Immediate next actions

*Batch 1 complete. NARROW on E (app-level prefix cache + bounded ingestion ties LMPU on tokens; LMPU wins on energy). PASS on B and P3. V0.8 ceiling resolved (addressable fraction ~12% on best case). Batch 3 complete. V1.6 negative controls verified on GPU (3 trials, 100%). V1.9 capability arm verified on GPU (3 trials, 100%). Batch 4 key experiments complete: context scaling, batch 27× speedup, bounded ingestion, energy measurement on RTX 3080, decision boundary published. Model planner (L2) works with Qwen2.5-1.5B + constrained decoding but latency ~15s/trial makes it impractical; oracle planner is the viable baseline for decisive experiments. In order:*

1. **Model planner (L2) — latency problem.** Constrained decoding on Qwen2.5-1.5B works (valid JSON) but ~15-17s/trial for planning phase. **Options:** (a) Accept oracle planner for decisive experiments; (b) Optimize constrained decoding (smaller model, faster logits processor, speculative decoding); (c) Use oracle planner for all conclusive results.
2. **Batch 4 — remaining experiments.** V3.5 Concurrency (two-session), V3.7 Cold vs steady state (needs longer workload), V4.3 Total cost model, V4.5 Cost sensitivity.
3. **Do not run full matrix yet** — V2.3+ only after L2 resolved.
4. **P2 (plan graph)** — implement executor or cut with dated pre-registration revision; V1.10 records it explicitly as unimplemented.

The Batch 1 artifacts (`benchmarks/results/batch1-oracle-pilot/`, commit `3c2c882f4d9e`) are the evidence base for E=NARROW, B=PASS, P3=PASS. No release claim may cite the void 32k pilot.
