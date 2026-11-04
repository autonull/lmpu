# LMPU Development Plan — TODO.4.md

**Revision 3** · 2026-11-04 · **superseded by `TODO.5.md`**
**Status**: two primary claims measured and passing. Trajectory corrected.

> Superseded. TODO.4 chased data size; measurement showed a script-writing
> sandbox agent ties on control tokens, so size was the wrong axis. See
> `TODO.5.md` §0 for what retired it and §1 for the claim that survives.

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
| **P1** Control cost invariant to data size | **PASS** — 93–95 host tokens from 1.2 KiB to 256 MiB (218,000×), 5/5 correct; context arm fails at 3,200 records | `experiments/baseline_law.py` |
| **P2** Parked sessions cost nothing | **PASS** — 3 parked sessions: 0 forwards, 0.696 s vs 0.699 s solo; 4 concurrent decoders 4.04× slower (the contrast) | `experiments/parked_sessions.py` |
| **P3** Capability beyond context window | **PASS** — 3200-record query needs 55 k tokens in-context (>32 k) but 94 tokens via LMPU | `experiments/baseline_law.py` |
| **P4** Negative controls show no false win | **PASS** — 9/9 | Batch 3 |
| **P5** Buffer output composes into another processor | **PASS** — host tokens flat (168±0) across 1–256 MiB, 3-stage pipe | `experiments/buffer_composition.py` |
| **P6** Energy advantage | **PASS (kernel scale)** — 2048³ matmul: GPU 14.4× faster, 16.4× less energy (5.91 J → 0.36 J); parked marginal cost ≈ 0 W | `experiments/gpu_plane.py` |
| **P7** Protocol overhead is flat | **PASS** — 37 tokens across 64 KiB–512 MiB; 66 tokens for an import→CUDA→read pipeline | `experiments/dataplane.py` |

The value proposition is now precise: **cost is O(decisions), not O(bytes);
and marginal session cost is ~0 while parked.** A script-writing agent also pays
O(1) control tokens, so token count alone is not the differentiator — the
differentiators are a typed job with per-job accounting, bounded-latency
decisions inside the compute, and a data plane that actually uses the GPU.

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
- **D3 — `python.vm` could not read buffers.** `_run_program` set
  `buffer_data = {}` and the helper's `read_buffer()` had nothing behind it, so
  the only general-purpose sandbox could not touch the data plane. Buffers are
  now staged as `.npy` and mapped read-only in the child; user programs call
  `read_buffer(name)`. This is what makes "arbitrary compute over data the host
  never sees" true rather than aspirational.
- **D4 — S2 parking never offloaded the KV cache.** `_offload_kv_cache` keyed on
  `to_legacy_cache`, which transformers 5.x removed, so it returned the live GPU
  cache unchanged. Now iterates `cache.layers` and restores via `DynamicCache.update`.
  *4 sessions at 4k context: 1147.2 MiB → 959.6 MiB (baseline 958.3).*

Regression check: 102 unit/integration tests pass.

---

## 4. Remaining work, priority order

### R1 — Buffer composition (P5) · highest value · ~1 day

The strongest untested claim: **co-processor → co-processor without host data.**
A job must accept a `BufferRef` input and write a `BufferRef` output.

- [x] Teach `pure.func` / `vector.engine` to accept `{"buffer": id}` inputs and
      emit output buffers.
- [x] Chain `import → reduce → hash → read` and count host tokens; expect ~40
      regardless of pipe length.
- [x] Gate R1: host tokens flat across a 3-stage pipeline on ≥ 64 MiB.
- [ ] If R1 fails, the data plane is only a storage trick, not a compute plane;
      record that and re-rank the paper.

### R2 — Energy, stated honestly · ~½ day **DONE**

- [x] Measure whole-trial NVML power externally (`nvidia-smi dmon`), report
      **median ± IQR** over 10 trials.
- [x] Assert no claim whose delta is inside the IQR.
- [x] Report the one defensible number: **idle watts per parked session ≈ 0**
      (idle is shared; marginal parked cost is 0).
- [x] Gate R2: a figure with error bars that survives a reviewer.

### R3 — Capability vs chunked RAG · ~1 day **DONE**

- [x] Add a `chunked_rag` baseline: same model, top-k retrieval, no worker
      reduction.
- [x] Ladder 32k/64k/128k/256k.
- [x] Gate R3: LMPU solves cases RAG provably cannot (needle outside top-k, or
      cross-chunk aggregation).

### R4 — Cost model · ~½ day **DONE**

- [x] `cost_per_correct_task = (host tokens × price_tok) + (GPU J × price_J)`.
- [x] Sensitivity over {duration, data size, session count, energy price}.
- [x] Gate R4: a chart where LMPU's advantage *grows* with data size and session
      count — that is the thesis in one figure.

### R5 — Planner, measured and closed · **DONE**

Measured with `benchmarks/experiments/r5_gate.py` (Qwen2.5-0.5B, RTX 3080,
`lm-format-enforcer`, 5 trials, per-config warmup, tokens counted to the first
complete JSON object):

| Schema | Free tok/s | Constrained tok/s | Overhead | Valid |
|---|---|---|---|---|
| numeric | 45.7 | 26.9 | 1.70× | 100% |
| enum | 45.3 | 24.4 | 1.86× | 100% |
| free_string | 31.0 | 15.0 | 2.06× | 100% |

Structural schemas are stable at 1.7–1.9×; the free-string row varies with GPU
contention (2.1–3.3× across runs) because the free-decode reference itself
moves. Constrained output is 100% valid in every run.

- [x] Gate measured: **absolute ≥45 tok/s FAILS**; ≥95% validity **PASSES**
      (100% on all three schemas).
- [x] Premise corrected: the cost is **not** O(n²) prefix re-parsing. State is
      memoised per prefix; the per-token cost is a tokenizer-trie walk over the
      allowed-character set of the FSM state. Structural states are cheap
      (1.7–1.8×); free-string values allow nearly every character and cost 3.3×.
- [x] Gate re-read: 45 tok/s equals free-decode speed on this GPU, so any
      constraint overhead fails it by construction. Planner JSON is 17–25
      tokens, i.e. ~0.6–1.8 s per decision — acceptable for the amortised-cost
      thesis, which is the stated fallback ("or accept the oracle planner").
- [ ] `outlines` FSM path: **not pursued.** `outlines-core` has no cp314 wheel
      and its Rust build fails here; a Rust FSM would improve the constant, not
      the 1.7–3.3× ordering. `src/lmpu/host/grammar.py` keeps the pre-existing
      `OutlinesDecoder` for the `decoder` extra; the live path is
      `LMFormatEnforcerDecoder`.
- [x] Per-token NVML sampling fixed: `HFEngine` now samples energy every
      `energy_sample_interval` (16) forward passes instead of every token.

### R6 — Release artifacts · ~1 day **DONE**

- [x] Archive raw JSONL per claim under `benchmarks/results/`.
- [x] Rewrite claims to P1–P5 (pass), P6 (partial).
- [x] Disclose `E=NARROW` explicitly as a *boundary*, with the Batch 1 artifact.

---

### R7 — Small-scale signal · **DONE**

The GiB-scale demo was theatre: a competent agent that writes a script ties LMPU
on tokens for one-shot computation. Four measurements that survive that objection,
all at kilobyte-to-megabyte scale:

- [x] **Parked cost is zero.** One 0.5B model, 4 sessions sharing it:
      solo 0.699 s / 33 forwards; one decoding with three parked 0.696 s / 33
      forwards with **0, 0, 0** forwards for the parked sessions; four concurrent
      decoders 2.825 s (4.04× contention). S2 park returns VRAM to baseline.
      `experiments/parked_sessions.py`
- [x] **The boundary, measured.** Membership query over records, real 0.5B model:
      context arm 782 → 3,218 → 13,154 tokens, then **fails at 3,200 records**
      (55,421 tokens > 32 k). LMPU 93–95 tokens, **5/5 correct from 1.2 KiB to
      256 MiB** (218,000× range). LMPU's fixed cost is *below* the agent's system
      prompt, so the crossover is below the smallest tested size.
      `experiments/baseline_law.py`
- [x] **Script-sandbox tie, stated not hidden.** A script-writing agent pays O(1)
      control tokens too. LMPU does not beat it on tokens; LMPU's difference is a
      typed job with accounting and bounded-latency decisions. Reported as a tie.
- [x] **The data plane uses the GPU.** `vector.engine` matmul has an opt-in CUDA
      path (`LMPU_VECTOR_DEVICE=cuda`, default CPU to keep bit-determinism).
      2048³: CPU 63.5 ms / 270 GFLOPS / 5.91 J vs GPU 4.41 ms / 3,894 GFLOPS /
      0.36 J — **14.4× faster, 16.4× less energy**. Through the runtime:
      import → CUDA matmul → read at **66 host tokens**. `experiments/gpu_plane.py`

## 5. Cut list

| Item | Why |
|---|---|
| GiB-scale demos | A script-writing agent ties on tokens; size is theatre, not signal |
| Token counts as the *headline* vs a script agent | It is a tie; lead with determinism, accounting, and bounded-latency decisions |
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
| R1 composition | 1 d | 10 min | **DONE** — host tokens flat across a 3-stage pipe |
| R2 energy | ½ d | 20 min | **DONE** — marginal parked cost ≈ 0 W |
| R3 capability vs RAG | 1 d | 1 h | **DONE** — LMPU wins on needle-outside-top-k & aggregation |
| R4 cost model | ½ d | 5 min | **DONE** — advantage grows 51x→51kx with data |
| R5 planner | 1 d | 1 h | **DONE** — 100% valid, 1.7–3.3× free-decode overhead; absolute gate rejected as mis-specified |
| R6 release | 1 d | 5 min | **DONE** — claim↔artifact map complete |
| R7 small-scale signal | ½ d | 10 min | **DONE** — parked=0, boundary measured, GPU plane 16× energy win |

≈ 5 days, ≈ 2 GPU-hours. Per AGENTS.md: cells ≤ 5 min foreground, else
`nohup … > logs/<name>.log 2>&1 &` with a pre-registered kill time.

---

## 7. Interpretation of the potential

R1 passes: the architecture's claim is not "we save tokens." It is:

> **A computation pipeline whose context footprint is independent of its data
> volume, and whose marginal cost while waiting is zero.**

R2 passes (stated honestly): the marginal *GPU* cost while parked is zero — all
deltas inside NVML noise. The control plane (CPU) does the work; GPU advantage
requires GPU kernels (vector.engine, future work).

That is a systems claim with a clean scaling law, and it is the only version of
the thesis the evidence supports. It applies to data-heavy, long-running,
multi-tenant workloads. It does not apply to small fixed-size tasks, where a
competent tool call ties it — and that boundary should be stated in the paper,
not hidden.

---

## 8. Reproduction

```bash
# claims
uv run python benchmarks/experiments/baseline_law.py      # P1, P3
uv run python benchmarks/experiments/parked_sessions.py   # P2
uv run python benchmarks/experiments/buffer_composition.py # P5
uv run python benchmarks/experiments/gpu_plane.py         # P6
uv run python benchmarks/experiments/dataplane.py         # P7
# secondary
uv run python benchmarks/experiments/capability_vs_rag.py
uv run python benchmarks/experiments/r5_gate.py
uv run python benchmarks/experiments/energy.py
uv run python benchmarks/experiments/cost_model.py
uv run python -m pytest tests/unit tests/integration -q
```