# LMPU Development Plan — TODO.5.md

**Revision 1** · 2026-11-04 · supersedes TODO.4.md
**Status**: measured core complete. One claim left that a sandbox cannot tie.

---

## 0. What changed since TODO.4

TODO.4 chased size. That was the wrong axis, and three measurements retired it:

1. **A script-writing sandbox agent ties LMPU on control tokens.** Both pay
   O(1) per decision, independent of bytes. Any claim of the form "we use fewer
   tokens than an agent over N bytes" is false as soon as the agent can run code.
2. **`vector.engine` never used the GPU.** Every worker was numpy/CPU, so the
   P6 energy null was an artifact of a CPU-only data plane, not of the design.
   With an opt-in CUDA matmul: 2048³ runs 14.4× faster and uses **16.4× less
   energy** (5.91 J → 0.36 J).
3. **Two real defects surfaced only when we measured.** `python.vm` could not
   read buffers at all (`buffer_data = {}`), and S2 parking never offloaded the
   KV cache (`to_legacy_cache` was removed in transformers 5.x), so "sleep"
   freed nothing. Both are fixed; both were invisible to the old benchmarks.

The lesson for planning: measure the claim against a *strong* baseline, at
whatever scale makes the ratio legible. Kilobyte-scale experiments found two
shipped bugs. Gigabyte demos would have found nothing.

---

## 1. The claim we can actually win

> **A fixed-context model can supervise computation whose in-context
> representation exceeds its window — at flat control cost, with zero marginal
> cost while parked, and with the worker plane using the accelerator.**

What this is *not*:
- not "fewer tokens than an agent over the same bytes" (tie, measured),
- not "faster decoding" (the GPU serialises decoders),
- not "a better planner" (constrained decoding costs 1.7–2.1× free decode).

What remains, and is defensible:
- **Capability**: queries whose in-context cost exceeds the window.
- **Adaptive supervision**: model decisions *inside* the compute, at bounded
  latency and flat context. A one-shot script cannot express this.
- **Accounting**: per-job bytes, FLOPs, tokens, energy; a replayable ledger.
- **Serving economics**: parked sessions cost zero forward passes and zero W.

---

## 2. Measured core (do not re-litigate)

| # | Claim | Evidence | Number |
|---|---|---|---|
| P1 | Control cost flat in data size | `baseline_law.py` | 93–95 tok, 1.2 KiB → 256 MiB (218,000×), 5/5 |
| P2 | Parked sessions cost nothing | `parked_sessions.py` | 0 forwards, 0.696 s vs 0.699 s; 4 concurrent 4.04× |
| P3 | Capability beyond the window | `baseline_law.py` | 3,200 records → 55 k tok in-context; LMPU 94 |
| P5 | Buffer output composes | `buffer_composition.py` | 168±0 tok, 3-stage, 1–256 MiB |
| P6 | GPU plane is energy-positive | `gpu_plane.py` | 2048³: 14.4× faster, 16.4× less energy |
| P7 | Protocol overhead flat | `dataplane.py` | 37 tok, 64 KiB → 512 MiB |
| — | Boundary | `E_NARROW_boundary.json` | sandbox agent ties on tokens |
| — | Planner | `r5_gate.py` | 100% valid, 1.7–2.1× slower than free decode |

Test state: `126 passed, 3 skipped`.

---

## 3. Opportunities, ranked by what they would prove

### O1 — Adaptive supervision (the keystone) · ~2 days · highest value

The only claim a script cannot tie. Design a task where a fixed criterion is
insufficient *by construction*:

- 200k synthetic records across 20 shards; the label distribution drifts at
  shard 10 (e.g. a second feature becomes predictive).
- Arm A (**one-shot script**): a program with a threshold chosen before seeing
  the data.
- Arm B (**adaptive**): the model sees only aggregates and 20 samples per shard,
  then sets/revises the threshold and re-runs the scan. Model tokens stay flat;
  decisions grow linearly.
- Arm C (**oracle**): threshold fitted with full labels — upper bound.

**Best case**: B recovers ≥90% of C's F1 at ≤5 model decisions per shard, while
A loses ≥20 F1 points on the drifted half. That is a task-level win that no
one-shot script can produce, at constant context.
**Kill criterion**: if A ≈ B on every task we can construct, adaptive
supervision has no measurable value and the project should say so.

### O2 — Decision-cost law · ~½ day

Generalise P1 into a two-variable law rather than a single flat line:

- Sweep bytes at fixed decision count (expect slope ≈ 0).
- Sweep decision count at fixed bytes (expect slope ≈ tokens/decision).

**Best case**: `host_tokens ≈ a + b·(decisions)`, with `b` measured and `a`
below a system prompt. Report both slopes with confidence intervals. This is the
one figure that makes the thesis quantitative instead of anecdotal.

### O3 — Replay and ledger · ~½ day

Run one pipeline 5× and diff: result checksum, per-job byte/FLOP/token counts,
event log. Then run the same task through a hosted model five times and diff the
answers.

**Best case**: pipeline replays are bit-identical end to end; model answers vary.
Differentiator for auditable settings; cheap to measure. **Do not** claim
determinism against a temperature-0 agent's final answer — measure only what is
actually stable.

### O4 — Crash-resume · ~½ day

Kill a worker mid-pipeline (SIGKILL), assert the job is retried elsewhere and the
pipeline completes with an identical result. The failure-injection harness
already exists (`tests/failure/`).

**Best case**: N/10 kills recovered, identical output, bounded retry cost.
Reliability is a stronger differentiator than tokens for anyone running long jobs.

### O5 — Multi-tenant economics · ~1 day

Real model, 8 sessions: each wakes, issues 3 decisions, parks. Compare GPU
seconds and wall against serial. **Best case**: GPU-seconds per completed task
falls because parked time costs nothing; wall does not degrade for the active
session.

### O6 — fd-based buffer handoff · engineering enabler · ~½ day

`python.vm` currently stages buffers as `.npy` in a temp dir — correct, but it
copies the whole buffer to disk. Pass descriptors (memfd or the shm fd) and
`np.memmap` in the child. Required before any multi-GB adaptive run; not required
for O1–O5 at the scales they need.

---

## 4. Not doing

| Item | Why |
|---|---|
| GiB-scale demos | A script ties on tokens; size is theatre |
| Token counts as the headline vs a script agent | It is a tie, measured |
| `outlines` / Rust FSM | No cp314 wheel, Rust build fails; improves a constant, not the ordering |
| Model scaling past 1.5B | 10 GB VRAM already supports several sessions; adds little |
| RAPL CPU joules | No `/sys/class/powercap` in this container |
| Sub-2 s per-call planner | Wrong objective; the thesis is amortised cost |

---

## 5. Paper plan

Target: a preliminary paper shareable with developers, not a submission.
Living draft: `paper/preliminary.md` (write once O1 is measured; scaffold now).

| Section | Must contain | Depends on |
|---|---|---|
| Abstract | fixed-context supervision, flat control cost, GPU-positive plane | P1–P7 |
| Claim & non-claims | the sandbox tie, stated first | boundary artifact |
| System | agent → kernel → worker; shm + FD handoff; park levels | code |
| Results | P1, P2, P3, P5, P6, P7 tables with commands | artifacts |
| Adaptive supervision | O1 result or an explicit negative | O1 |
| Boundaries | planner overhead, small tasks, sandbox tie | R5, boundary |
| Engineering findings | D3 (sandbox blind), D4 (sleep never freed VRAM) | this doc §0 |
| Open problems | O2–O5 as invitations to collaborate | — |
| Reproduce | exact commands, hardware, versions | — |

Rule: every number in the paper maps to a file under `benchmarks/results/`.
No number that cannot be regenerated by one command.

---

## 6. Order

1. **O2** (½ day) — cheapest, turns P1 into a law, and its harness feeds O1.
2. **O1** (2 days) — keystone; decides whether adaptive supervision is real.
3. **O3 + O4** (1 day) — both cheap, both strengthen the paper's credibility.
4. **Paper** (1 day) — draft after O1, finalise after O3/O4.
5. **O5, O6** — only if O1 lands and someone wants multi-tenant numbers.

Cells ≤5 min foreground; longer runs go to `nohup … > logs/<name>.log`.

---

## 7. Honest failure modes

- **O1 null**: adaptive supervision is not measurably better than a fixed
  criterion. Then the paper's contribution is the capability + accounting +
  GPU plane, and adaptive control is reported as an open negative. Say it plainly.
- **O2 degenerates**: if `b` is dominated by prompt overhead at every decision
  count we test, the law is trivial. Report it as such.
- **O3 null**: replay is deterministic but so is temperature-0 inference. Then the
  differentiator is the ledger, not determinism; claim only the ledger.
- **O4 flaky**: a killed worker occasionally duplicates side effects. That is a
  real finding; fix exactly-once semantics or document at-least-once.
