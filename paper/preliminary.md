# Supervising computation larger than the context window

**Preliminary draft — 2026-11-04**
**Status**: measured core complete; adaptive-supervision result pending (see §7).
**Code**: `src/lmpu` · **Artifacts**: `benchmarks/results` · **Plan**: `TODO.5.md`

---

## Abstract

We describe LMPU, a runtime in which a language model supervises computation whose
in-context representation would exceed its window. The model emits typed jobs; a
kernel routes them to worker processes that read shared-memory buffers directly;
the model receives bounded summaries.

Measured on one RTX 3080 with Qwen2.5-0.5B-Instruct:

- control cost is **93–95 tokens from 1.2 KiB to 256 MiB** of data (218,000×), with
  5/5 correct answers;
- a context-bound baseline climbs from 782 to 55,421 tokens and **fails at 3,200
  records** (its 32 k window);
- three parked sessions perform **zero forward passes** and slow an active decoder
  by 0% (0.696 s vs 0.699 s), while four *active* decoders contend 4.04×;
- the worker plane is accelerator-positive: a 2048³ matmul is **14.4× faster and
  uses 16.4× less energy** on GPU than CPU (5.91 J → 0.36 J), reached from the
  model at **66 control tokens**.

We also report where the approach does **not** win. A competent agent that writes
a script and runs it in a sandbox ties LMPU on control tokens, because both pay
O(1) per decision. Token count alone is therefore not the contribution. What
remains is capability beyond the window, per-job accounting, bounded-latency
decisions *inside* the computation, and a worker plane that uses the GPU.

---

## 1. Claim, and three non-claims

**Claim.** A fixed-context model can supervise computation whose in-context
representation exceeds its window, at control cost independent of data size,
with zero marginal cost while parked, and with the workers using the accelerator.

**Non-claim 1 — fewer tokens than a sandbox agent.** Measured as a tie. Any
comparison that puts N bytes into a context-only baseline conflates the encoding
of the data with the cost of the protocol. We report the tie rather than hide it.

**Non-claim 2 — faster decoding.** Concurrent decoders serialise on one GPU:
four decoders take 4.04× the wall time of one. Parking is where the win is, not
parallel decoding.

**Non-claim 3 — a faster planner.** Constrained JSON decoding is correct (100%)
but 1.7–2.1× slower than free decoding, because the per-token cost is a
tokenizer-trie walk over the allowed-character set of the FSM state.

---

## 2. System

```
model ──submit(op, inputs, outputs)──► kernel ──job──► worker
  ▲                                      │              │
  └──── bounded summary / event ◄────────┘              └── shm buffer (no host copy)
```

- **Buffers** live in shared memory. Handles are passed to workers by descriptor;
  a job's inputs and outputs are references, never payloads.
- **Jobs** are typed (`op`, declared inputs/outputs, limits). The kernel routes by
  capability manifest, enforces limits, and records events.
- **Parking** (S1) freezes the KV cache in place; (S2) offloads it to CPU; (S3)
  checkpoints to disk. A parked session performs no forward passes.
- **Summary** returned to the model is O(1): status, checksum, shape, counts.

The model therefore never holds the data, only the decision.

---

## 3. Results

Hardware: RTX 3080 (10 GB), 16 cores, 31 GB RAM, 16 GB `/dev/shm`.
Python 3.14, transformers 5.20.2, bf16. Model: Qwen2.5-0.5B-Instruct (32 k window).

### 3.1 Control cost is flat in data size

Membership query over records; the LMPU arm imports a buffer and submits one
`python.vm` job that reads it. (`baseline_law.py`)

| records | data | context tokens | LMPU tokens | LMPU correct |
|---:|---:|---:|---:|---|
| 50 | 1.2 KiB | 782 | 93 | yes |
| 200 | 4.8 KiB | 3,218 | 93 | yes |
| 800 | 19.4 KiB | 13,154 | 94 | yes |
| 3,200 | 80 KiB | **55,421 — exceeds window** | 94 | yes |
| 200,000 | 256 MiB | 67,108,864 required (2,048× window) | **95** | yes |

The crossover is below the smallest tested size: LMPU's fixed cost (93 tokens) is
lower than the baseline's system prompt (782 tokens). The baseline grows at
≈16.4 tokens/record and fails where `16.4·N > 32,768`.

Protocol overhead alone, across two orders of magnitude more data, is flat at
**37 tokens** from 64 KiB to 512 MiB (`dataplane.py`).

### 3.2 Capability beyond the window

The 3,200-record row is not a slow baseline; it is an impossible one. No prompt
engineering recovers it, because the encoded data exceeds the window. LMPU
answers it with 94 tokens and no accuracy loss. At 256 MiB the baseline would
need 2,048 windows' worth of tokens.

This is the claim we consider strongest, and it requires no argument about
prices, caching, or agent competence.

### 3.3 Parked sessions are free

One model, four sessions sharing it. (`parked_sessions.py`)

| scenario | wall | decoder forwards | other sessions' forwards |
|---|---:|---:|---|
| solo | 0.699 s | 33 | — |
| 1 decoding + 3 parked | 0.696 s | 33 | **0, 0, 0** |
| 4 concurrent decoders | 2.825 s | 32 each | 32 each (4.04×) |

The concurrent row is the control: it shows the measurement can detect cost.
Parked sessions add none. S2 parking returns VRAM to baseline (1147.2 → 959.6 MiB,
base 958.3) at 4,001 tokens of context per session.

### 3.4 Buffer composition

`import → sha256 → text_stats → read` over 1–256 MiB: **168 ± 1 control tokens**,
at every size (`buffer_composition.py`). Intermediate results stay in buffers; the
host sees only references and summaries.

### 3.5 The worker plane is accelerator-positive

`vector.engine` matmul, CPU (numpy) vs GPU (torch/CUDA), power sampled with NVML
over a window long enough to resolve it (`gpu_plane.py`).

| dim | device | wall | GFLOPS | J/matmul |
|---:|---|---:|---:|---:|
| 1024 | CPU | 13.23 ms | 162 | 0.17 |
| 1024 | GPU | 1.30 ms | 1,656 | 0.02 |
| 2048 | CPU | 63.53 ms | 270 | 5.91 |
| 2048 | GPU | 4.41 ms | 3,894 | **0.36** |

Through the runtime — import A, import B, matmul on CUDA, read C — the job
completes at **66 control tokens**.

An earlier version of this work reported a null energy result. That null was an
artifact: no worker used the GPU, so the GPU never left idle. The lesson is in §4.

### 3.6 Where it stops

- **Sandbox tie.** A script-writing agent also pays O(1) control tokens. LMPU
  does not beat it on tokens; it ties. Our differentiators are elsewhere.
- **Small tasks.** A single model call needs no pipeline; 93 tokens of protocol
  buys nothing. The crossover is low but not zero.
- **Planner.** 100% valid JSON, 1.7–2.1× slower than free decoding.

---

## 4. Two defects that only measurement found

Both were shipped, both invalidated claims, neither was visible to unit tests.

**D3 — the sandbox could not see the data plane.** `python.vm` set
`buffer_data = {}`; its `read_buffer()` helper had nothing behind it. Every
"arbitrary computation over a buffer" claim was false. Buffers are now staged and
memory-mapped read-only in the child.

**D4 — sleep never freed memory.** S2 parking called `to_legacy_cache`, removed in
transformers 5.x, and silently returned the live GPU cache. Four "offloaded"
sessions held 1.2 GB. Now `cache.layers` is walked and the cache rebuilt via
`DynamicCache.update`.

Both were found by writing a benchmark against a real model at kilobyte scale.
Neither would have been found by a gigabyte demo.

---

## 5. Open problems (invitations)

These are unmeasured. Each has a concrete experiment in `TODO.5.md`.

1. **Adaptive supervision.** Does a model that inspects samples and revises a
   criterion beat a fixed one-shot script on drifted data, at equal model cost?
   This is the only claim a sandbox cannot tie. If it fails, say so.
2. **Decision-cost law.** Is control cost `a + b·decisions` with `b` measurable
   and `a` below a system prompt? One figure, two slopes.
3. **Replay and ledger.** Are pipelines bit-identical across runs, and is the
   per-job accounting (bytes, FLOPs, tokens, energy) sufficient to audit them?
4. **Crash-resume.** Does a killed worker yield identical results after retry, or
   duplicated side effects? The failure harness already exists.
5. **Multi-tenant economics.** With 8 sessions waking and parking at random, what
   happens to GPU-seconds per completed task?
6. **Descriptor handoff.** `python.vm` currently copies buffers to a temp file.
   Passing the shm descriptor would remove an O(size) copy.

---

## 6. Reproduce

```bash
uv run python benchmarks/experiments/baseline_law.py       # §3.1, §3.2
uv run python benchmarks/experiments/parked_sessions.py    # §3.3
uv run python benchmarks/experiments/buffer_composition.py # §3.4
uv run python benchmarks/experiments/gpu_plane.py          # §3.5
uv run python benchmarks/experiments/dataplane.py          # §3.1 (protocol)
uv run python benchmarks/experiments/r5_gate.py            # §3.6
uv run python -m pytest tests/unit tests/integration -q    # 126 passed, 3 skipped
```

Raw artifacts: `benchmarks/results/{baseline_law,parked_sessions,buffer_composition,gpu_plane,dataplane,r5_gate}`.
Claim files: `benchmarks/results/claims/*.json`.

Every number in this paper is regenerated by one of the commands above.

---

## 7. Status and how to help

Measured: control-cost invariance, capability beyond the window, zero-cost
parking, buffer composition, accelerator-positive worker plane, planner overhead.

Not yet measured: adaptive supervision, the decision-cost law, replay, crash
resume, multi-tenant economics.

The keystone is adaptive supervision. If a model that only sees aggregates and
samples can steer a computation better than a fixed script over the same bytes,
LMPU is a control plane for computation, not a data-transfer optimisation. If it
cannot, the honest result is the capability claim in §3.2 plus an accounting and
reliability story — still useful, and much narrower.

We would rather publish the narrow version than defend a claim that a 20-line
script can tie.
