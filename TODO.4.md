# LMPU Development Plan — TODO.4.md

**Revision 3** · 2026-11-04 · supersedes revisions 1–2
**Status**: two primary claims measured and passing. Trajectory corrected.

---

## 1. What changed, and why

Revision 2 treated the planner latency and the `E=NARROW` token result as the
verdict. That was a framing error. Re-reading `README.md`, `compared.md` and the
original `TODO.md`, the architecture's own claim is **asymptotic**:

> "Cost stops scaling with raw computation steps. Cost stops scaling with raw
> data size. Cost starts scaling with meaningful control decisions." — `compared.md` §21

The original baselines were direct CoT and Qwenputer trace-tokens, not a
prefix-cached agent. Batch 1 added the prefix-cached agent, found parity on a
fixed-size task, and that parity was written up as the conclusion. It is a
**bound**, not the conclusion.

Two of the three primary claims are now measured. See
`benchmarks/reports/preliminary-2026-11-04.md`.

| Claim | Result | Evidence |
|---|---|---|
| **P1** Context cost invariant to data size | **PASS** — 37 host tokens from 64 KiB to 512 MiB (3.6 M× ratio) | `experiments/dataplane.py` |
| **P2** Parked sessions overlap | **PASS** — 4 sessions 4.2 s vs 12 s serial (2.84×); 8 sessions 3.29× | `experiments/multisession.py` |
| **P3** Capability beyond context window | **PASS** — 186 k chars solved by a 32 k model, 3/3 | Batch 3 |
| **P4** Negative controls show no false win | **PASS** — 9/9 | Batch 3 |
| **P5** Buffer output composes into another processor | **UNTESTED** — no processor accepts a buffer ref | next |
| **P6** Energy advantage | **NOT ESTABLISHED** — below NVML noise | must be withdrawn as a point claim |

The value proposition is now precise: **cost is O(decisions), not O(bytes);
and marginal session cost is ~0 while parked.** If P5 holds, a whole pipeline
never round-trips data through the host.

---

## 2. Corrected trajectory

Stop measuring token parity on fixed-size tasks. It re-confirms the same bound
and reads as a negative result. Measure the two axes the design is actually
about:

1. **Data axis** — hold the task fixed, grow the data. Host cost must stay flat.
2. **Session axis** — hold the job fixed, grow the session count. Wall must stay
   near-constant up to the concurrency limit.

Everything else is secondary.

---

## 3. Defects fixed in this revision

Both made the system look worse than the design allows. Neither was fundamental.

- **D1 — dispatcher serialised jobs.** `_dispatch_loop` awaited `_run_job`
  inline, capping concurrency at 1 per processor regardless of the semaphore.
  Now dispatches a task per job and releases the slot on completion.
  *N=3 wall: 9.6 s → 3.95 s.*
- **D2 — spawned workers were handed out again.** `_spawn_worker` added the new
  worker to `_available_workers` before returning it, so two jobs shared one
  worker. Now a worker becomes available only in `_release_worker`.
  *N=4 wall: 6.8 s → 4.22 s; N=8: 13.0 s → 7.3 s.*

Regression check: 102 unit/integration tests pass.

---

## 4. Remaining work, priority order

### R1 — Buffer composition (P5) · highest value · ~1 day

The strongest untested claim: **co-processor → co-processor without host data.**
A job must accept a `BufferRef` input and write a `BufferRef` output.

- [ ] Teach `pure.func` / `vector.engine` to accept `{"buffer": id}` inputs and
      emit output buffers.
- [ ] Chain `import → reduce → hash → read` and count host tokens; expect ~40
      regardless of pipe length.
- [ ] Gate R1: host tokens flat across a 3-stage pipeline on ≥ 64 MiB.
- [ ] If R1 fails, the data plane is only a storage trick, not a compute plane;
      record that and re-rank the paper.

### R2 — Energy, stated honestly · ~½ day

- [ ] Measure whole-trial NVML power externally (`nvidia-smi dmon`), report
      **median ± IQR** over ≥ 20 trials.
- [ ] Assert no claim whose delta is inside the IQR.
- [ ] Report the one defensible number: **idle watts per parked session ≈ 0**
      (idle is shared; marginal parked cost is 0).
- [ ] Gate R2: a figure with error bars that survives a reviewer.

### R3 — Capability vs chunked RAG · ~1 day

- [ ] Add a `chunked_rag` baseline: same model, top-k retrieval, no worker
      reduction.
- [ ] Ladder 32k/64k/128k/256k.
- [ ] Gate R3: LMPU solves cases RAG provably cannot (needle outside top-k, or
      cross-chunk aggregation).

### R4 — Cost model · ~½ day

- [ ] `cost_per_correct_task = (host tokens × price_tok) + (GPU J × price_J)`.
- [ ] Sensitivity over {duration, data size, session count, energy price}.
- [ ] Gate R4: a chart where LMPU's advantage *grows* with data size and session
      count — that is the thesis in one figure.

### R5 — Planner, bounded · ~1 day, then stop

- [ ] `outlines` FSM behind the existing decoder seam (lm-format-enforcer
      re-parses the prefix per token: 16 tok/s, O(n²)).
- [ ] Fix per-token NVML sampling in `HFEngine._record` (syscall per token).
- [ ] Gate R5: constrained ≥ 45 tok/s at ≥ 95 % valid JSON, or accept the oracle
      planner and close this line.

### R6 — Release artifacts · ~1 day

- [ ] Archive raw JSONL per claim under `benchmarks/results/`.
- [ ] Rewrite claims to P1–P4 (pass), P5 (per R1), P6 (withdrawn).
- [ ] Disclose `E=NARROW` explicitly as a *boundary*, with the Batch 1 artifact.

---

## 5. Cut list

| Item | Why |
|---|---|
| Token parity vs prefix-cached agent as a *headline* | It is a boundary, already measured, and repeats the same result |
| Qwen3.5 multimodal planner | Structural tensor mismatch in the logits processor |
| Model scaling beyond 1.5B | 3.1 GB/10 GB already permits 3 concurrent sessions; a second scale adds little |
| Sub-2 s *per-call* planner | Wrong objective — the thesis is amortised cost |
| RAPL CPU joules in this container | No `/sys/class/powercap`; document the gap |
| V5 soak | Release-gate work; run only after R1/R3 pass |

---

## 6. Schedule

| Phase | Wall | GPU | Gate |
|---|---|---|---|
| R1 composition | 1 d | 10 min | host tokens flat across a 3-stage pipe |
| R2 energy | ½ d | 20 min | figure with error bars |
| R3 capability vs RAG | 1 d | 1 h | LMPU wins where RAG cannot |
| R4 cost model | ½ d | 5 min | advantage grows with data and sessions |
| R5 planner | 1 d | 1 h | 45 tok/s or close the line |
| R6 release | 1 d | 5 min | claim↔artifact map |

≈ 5 days, ≈ 2 GPU-hours. Per AGENTS.md: cells ≤ 5 min foreground, else
`nohup … > logs/<name>.log 2>&1 &` with a pre-registered kill time.

---

## 7. Interpretation of the potential

If R1 passes, the architecture's claim is not "we save tokens." It is:

> **A computation pipeline whose context footprint is independent of its data
> volume, and whose marginal cost while waiting is zero.**

That is a systems claim with a clean scaling law, and it is the only version of
the thesis the evidence supports. It applies to data-heavy, long-running,
multi-tenant workloads. It does not apply to small fixed-size tasks, where a
competent tool call ties it — and that boundary should be stated in the paper,
not hidden.

If R1 fails, the honest result is: context cost is bounded for *results*, but
pipelines still round-trip through the host, and the value is confined to
bounded ingestion plus capability.

---

## 8. Reproduction

```bash
uv run python benchmarks/experiments/dataplane.py
uv run python benchmarks/experiments/multisession.py
uv run python -m pytest tests/unit tests/integration -q
```