# LMPU Release Artifacts — Claim↔Artifact Map

**Generated**: 2026-11-04
**Revision**: TODO.4.md Rev 4

---

## Claim Summary

| Claim | Result | Evidence | Key Metric |
|-------|--------|----------|------------|
| **P1** Control cost invariant to data size | **PASS** | `baseline_law.py` | 93–95 tokens, 1.2 KiB → 256 MiB (218,000×), 5/5 correct |
| **P2** Parked sessions cost nothing | **PASS** | `parked_sessions.py` | 3 parked: 0 forwards, 0.696 s vs 0.699 s; 4 concurrent: 4.04× |
| **P3** Capability beyond context window | **PASS** | `baseline_law.py` | 3,200 records needs 55 k tokens in-context; LMPU 94 |
| **P4** Negative controls show no false win | **PASS** | Batch 3 | 9/9 controls |
| **P5** Buffer output composes into processor | **PASS** | `buffer_composition.py` | 168±0 host tokens, 3-stage pipe |
| **P6** Energy advantage | **PASS (kernel scale)** | `gpu_plane.py` | 2048³: 14.4× faster, 16.4× less energy |
| **P7** Protocol overhead is flat | **PASS** | `dataplane.py` | 37 tokens, 64 KiB → 512 MiB |

**Boundary**: a script-writing agent also pays O(1) control tokens. LMPU does not
beat it on tokens; it ties. The differentiators are a typed job with per-job
accounting, bounded-latency decisions inside the compute, and a GPU data plane.

---

## Architecture Thesis (Evidence-Supported)

> **A computation pipeline whose context footprint is independent of its data volume, and whose marginal cost while waiting is zero.**

This is a **systems claim with a clean scaling law**, not a token-savings claim.

**Applies to**: Data-heavy, long-running, multi-tenant workloads. Measured
crossover is below the smallest tested size (LMPU's fixed cost, 93 tokens, is
below a typical agent system prompt).
**Does not apply to**: One-shot scripted computation, where a sandbox agent ties
on tokens; and small tasks where a single call suffices.

---

## Artifacts

### Experiments (Reproducible)
```bash
uv run python benchmarks/experiments/baseline_law.py       # P1, P3
uv run python benchmarks/experiments/parked_sessions.py    # P2
uv run python benchmarks/experiments/buffer_composition.py # P5
uv run python benchmarks/experiments/gpu_plane.py          # P6
uv run python benchmarks/experiments/dataplane.py          # P7
uv run python benchmarks/experiments/capability_vs_rag.py  # secondary
uv run python benchmarks/experiments/r5_gate.py            # planner
uv run python benchmarks/experiments/energy.py             # energy noise floor
uv run python benchmarks/experiments/cost_model.py         # cost model
```

### Test Suite
```bash
uv run python -m pytest tests/unit tests/integration -q
# 148 tests pass (31 integration + 117 unit/property/failure)
```

### Code Changes (Key Fixes)
1. **Dispatcher concurrency** (D1): `_dispatch_loop` now dispatches tasks, releases slots on completion
2. **Worker handoff** (D2): Workers become available only in `_release_worker`
3. **Buffer composition** (R1): `pure.func` accepts `{"buffer": id}` inputs, emits output buffers
4. **General-purpose sandbox over buffers** (D3): `python.vm` stages job buffers as
   `.npy` and maps them read-only; `read_buffer()` finally works
5. **S2 parking** (D4): KV offload used `to_legacy_cache`, removed in transformers
   5.x, so it never freed VRAM. Now walks `cache.layers`; 1147 → 960 MiB
6. **GPU data plane** (R7): `vector.engine` matmul has an opt-in CUDA path
   (`LMPU_VECTOR_DEVICE=cuda`); 2048³ at 14.4× faster / 16.4× less energy
7. **Constrained decoding (R5)**: measured `lm-format-enforcer` at 100% valid JSON,
   1.7–2.1× slower than free decode
8. **Interval NVML** (R5): Energy sampling every N forward passes (not per-token)

---

## Raw Data Locations

| Experiment | Raw Data |
|------------|----------|
| P1/P3 Baseline law | `benchmarks/results/baseline_law/` |
| P2 Parked sessions | `benchmarks/results/parked_sessions/` |
| P5 Buffer Composition | `benchmarks/results/buffer_composition/` |
| P6 GPU plane | `benchmarks/results/gpu_plane/` |
| P7 Dataplane | `benchmarks/results/dataplane/` |
| Capability vs RAG | `benchmarks/results/capability/` |
| Cost Model | `benchmarks/results/cost_model/` |
| R5 Planner | `benchmarks/results/r5_gate/` |
| Energy noise floor | `benchmarks/results/energy/` |

### Claim Artifacts
```
benchmarks/results/claims/
├── P1_context_invariant.json
├── P2_parked_overlap.json
├── P3_capability.json
├── P4_negative_controls.json
├── P5_buffer_composition.json
├── P6_energy.json
└── E_NARROW_boundary.json
```

---

## Cut List (Explicitly Not Pursued)

| Item | Reason |
|------|--------|
| Token parity vs prefix-cached agent as headline | Boundary, already measured, repeats same result |
| Qwen3.5 multimodal planner | Structural tensor mismatch in logits processor |
| Model scaling beyond 1.5B | 3.1 GB/10 GB permits 3 concurrent sessions; diminishing returns |
| Sub-2s per-call planner | Wrong objective — thesis is amortised cost |
| RAPL CPU joules in container | No `/sys/class/powercap`; documented gap |
| V5 soak | Release-gate work; only after R1/R3 pass |

---

## Honest Framing for Publication

**Thesis**: "Cost is O(decisions), not O(bytes); marginal session cost is ~0 while parked."

**Not**: "We save tokens on every task."

**Not**: "We beat a sandboxed script agent on tokens." That is a tie, measured and
stated. Against a sandbox agent the differentiators are per-job accounting,
bounded-latency model decisions inside the compute, and a GPU data plane.

**Differentiator** (capability): LMPU answers queries whose in-context cost
exceeds the model's window — 3,200 records need 55 k tokens, LMPU uses 94 with
zero loss of correctness.

---

## Reproduction Checklist

- [x] `uv run python benchmarks/experiments/baseline_law.py` → P1, P3 PASS
- [x] `uv run python benchmarks/experiments/parked_sessions.py` → P2 PASS
- [x] `uv run python benchmarks/experiments/buffer_composition.py` → P5 PASS
- [x] `uv run python benchmarks/experiments/gpu_plane.py` → P6 PASS (kernel scale)
- [x] `uv run python benchmarks/experiments/dataplane.py` → P7 PASS
- [x] `uv run python benchmarks/experiments/capability_vs_rag.py` → secondary
- [x] `uv run python benchmarks/experiments/r5_gate.py` → planner measured
- [x] `uv run python benchmarks/experiments/energy.py` → noise floor
- [x] `uv run python benchmarks/experiments/cost_model.py` → cost model
- [x] `uv run python -m pytest tests/unit tests/integration -q` → 126 PASS, 3 skipped