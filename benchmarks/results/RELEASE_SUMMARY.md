# LMPU Release Artifacts — Claim↔Artifact Map

**Generated**: 2026-11-04
**Revision**: TODO.4.md Rev 3

---

## Claim Summary

| Claim | Result | Evidence | Key Metric |
|-------|--------|----------|------------|
| **P1** Context cost invariant to data size | **PASS** | `dataplane.py` | 37 host tokens, 64 KiB → 512 MiB (3.6M×) |
| **P2** Parked sessions overlap | **PASS** | `multisession.py` | 2.84× (4 sess), 3.26× (8 sess) |
| **P3** Capability beyond context window | **PASS** | Batch 3 | 186k chars by 32k model (3/3) |
| **P4** Negative controls show no false win | **PASS** | Batch 3 | 9/9 controls |
| **P5** Buffer output composes into processor | **PASS** | `buffer_composition.py` | 168±0 host tokens, 3-stage pipe |
| **P6** Energy advantage | **PARTIAL** | `energy.py` | Marginal parked cost ≈ 0 W |

**Boundary**: **E=NARROW** — Token parity with prefix-cached agent on fixed-size tasks is a *bound*, not the thesis.

---

## Architecture Thesis (Evidence-Supported)

> **A computation pipeline whose context footprint is independent of its data volume, and whose marginal cost while waiting is zero.**

This is a **systems claim with a clean scaling law**, not a token-savings claim.

**Applies to**: Data-heavy, long-running, multi-tenant workloads.
**Does not apply to**: Small fixed-size tasks (competent tool call ties it).

---

## Artifacts

### Experiments (Reproducible)
```bash
uv run python benchmarks/experiments/dataplane.py          # P1
uv run python benchmarks/experiments/multisession.py       # P2
uv run python benchmarks/experiments/buffer_composition.py # P5
uv run python benchmarks/experiments/energy.py             # P6
uv run python benchmarks/experiments/capability_vs_rag.py  # P3 (vs RAG)
uv run python benchmarks/experiments/cost_model.py         # R4
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
4. **Constrained decoding (R5)**: measured `lm-format-enforcer` at 100% valid JSON, 1.7–3.3× slower than free decode (26.7 / 25.2 / 13.7 tok/s vs ~45)
5. **Interval NVML** (R5): Energy sampling every N forward passes (not per-token)

---

## Raw Data Locations

| Experiment | Raw Data |
|------------|----------|
| P1 Dataplane | `benchmarks/results/dataplane/` |
| P2 Multisession | `benchmarks/results/multisession/` |
| P5 Buffer Composition | `benchmarks/results/buffer_composition/` |
| P6 Energy | `benchmarks/results/energy/` |
| R3 Capability vs RAG | `benchmarks/results/capability/` |
| R4 Cost Model | `benchmarks/results/cost_model/` |
| R5 Planner | `benchmarks/results/r5_gate/` |

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

**Differentiator** (R3): LMPU solves needle-outside-top-k and cross-chunk aggregation where RAG provably cannot — this is the *capability* win; the token/session claims are the *enabling mechanism*.

---

## Reproduction Checklist

- [x] `uv run python benchmarks/experiments/dataplane.py` → P1 PASS
- [x] `uv run python benchmarks/experiments/multisession.py` → P2 PASS
- [x] `uv run python benchmarks/experiments/buffer_composition.py` → P5 PASS
- [x] `uv run python benchmarks/experiments/energy.py` → P6 PARTIAL
- [x] `uv run python benchmarks/experiments/capability_vs_rag.py` → R3 PASS
- [x] `uv run python benchmarks/experiments/cost_model.py` → R4 PASS
- [x] `uv run python benchmarks/experiments/r5_gate.py` → R5 measured (100% valid, 1.7–3.3× overhead)
- [x] `uv run python -m pytest tests/unit tests/integration -q` → 126 PASS, 3 skipped