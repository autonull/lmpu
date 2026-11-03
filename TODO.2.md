# TODO.2 — LMPU Execution Plan

**Status:** Active execution plan — 2026-11-03  
**Supersedes for execution:** `TODO.1.md`  
**Design authority:** `TODO.1.md` remains the detailed reference for architecture, schemas, phase rationale, and original acceptance text. This file owns order, scope, and progress.  
**Companions:** `README.md`, `compared.md`

> **Rule.** Do not execute phases linearly. Close the evidence gates first. Product surface and optional integrations wait until the measurements are credible.

---

## 0. Current state

### Working
- Protocol models, JSON-RPC codec, runtime RPC surface.
- Typed shared-memory buffers, grants, bounded reads/writes, checksum refresh.
- `SCM_RIGHTS` worker fd passing, worker mmap, inline fallback, `copy_count` evidence.
- Six built-in processors register and pass a representative contract suite.
- Scripted session loop: injection policy, budgets, `ResultTooLarge`, parser recovery, forced wake.
- Reference GPU path: Qwen3.5-0.8B loads and runs inside the session loop via `HFDecoder`.
- Failure matrix: timeout, OOM, crash/recovery, disconnect, duplicate delivery, malformed worker messages.
- Trace JSONL emission, replay, gap detection, redaction.
- CPU suite: **119 passed, 4 skipped**; opt-in GPU tests: **2 passed**.

### Known gaps
- `HFDecoder` re-prefills every turn; no cross-turn KV reuse.
- Trace/metrics lack host token, forward-pass, and energy coverage.
- P1 exit gate is not closed: grants/reaper audit, routing/auth split, demo assertions, 16 MiB context-token proof.
- `uv.lock` has a host/vLLM dependency conflict; vLLM and Outlines are not installed.
- Legacy `mypy`/`ruff` findings remain outside the changed-file policy.
- Qwen3.5-2B, vLLM 0.33, and Outlines 0.1.14 are unverified.

---

## 1. Execution rules

1. Work top-down by gate. Do not start Gate C while Gate A is open.
2. One active item at a time; commit each item with its acceptance evidence.
3. Every claim needs a measured artifact or automated assertion. No demo-only evidence.
4. Prefer deletion and consolidation over new modules.
5. Optional integrations have explicit kill criteria and never block a gate.
6. Update this file when an item closes. Keep `TODO.1.md` as design history only.

---

## 2. Gate A — Measurement spine

**Goal:** make the core claims measurable and trustworthy.

**Exit criteria:** S1 park shows zero forward passes; HF turns reuse KV; metrics are emitted once and consumed by trace; grants and routing are audited; 16 MiB fits the context-token bound; the environment is reproducible.

| ID | Work | Acceptance | Depends |
|---|---|---|---|
| A1 | `HFDecoder` KV reuse | Injected text and generated tokens append to the live cache; two turns do not re-prefill the full prompt; forward-pass count matches tokens | — |
| A2 | Metrics/trace schema | One schema emits prefill tokens, generated tokens, injected tokens, control tokens, forward passes, wall time, VRAM, energy hooks | A1 |
| A3 | S1 park/resume proof | Park freezes forwards and generated tokens; resume matches baseline output; assertion is automated | A1, A2 |
| A4 | Grants/refcounts/TTL/reaper audit | Negative authorization tests; no leaked grants, fds, mappings, or shm after terminal/cancel/close | — |
| A5 | Routing/auth split | Capability advertisement is separate from session authorization; routing contract test | — |
| A6 | 16 MiB context bound | A 16 MiB result is represented by ≤64 tokens of context metadata; assertion covers the real session path | A2 |
| A7 | Environment reproducibility | Resolve `uv.lock` host/vLLM conflict; document optional vLLM env; `make check` scoped green on changed files | — |

**Gate A order:** A7 → A1 → A2 → A3 → A4 → A5 → A6.

---

## 3. Gate B — Benchmark spine

**Goal:** produce the first credible C1–C5 report. Start as soon as A1–A3 are usable.

**Exit criteria:** harness reproduces a run from a clean checkout; four baselines are fair; energy method is documented; three workloads produce raw artifacts and CIs.

| ID | Work | Acceptance | Depends |
|---|---|---|---|
| B1 | Harness | Config, seeds, trials, raw artifacts, resume-safe logs | A2 |
| B2 | Baselines | Direct CoT, trace-token, synchronous tool, async polling | B1 |
| B3 | Energy sampling | NVML path, idle baseline, optional RAPL, documented methodology | A2 |
| B4 | Initial workloads | One compute, one large-context, one sleep-heavy; correctness checks | B1 |
| B5 | C1–C5 report | Medians, CIs, CDFs, timelines, raw data, limitations | B2, B3, B4 |

**Do not** implement B1–B12 or the full ablation matrix before the first report.

---

## 4. Gate C — Product surface

**Goal:** make the evidence inspectable without expanding scope.

**Exit criteria:** three canonical demos pass in assertion mode; trace CLI replays a run; one minimal dashboard shows a parked host and live events.

| ID | Work | Acceptance | Depends |
|---|---|---|---|
| C1 | Three canonical demos | Kernel offload + recovery; S1 sleep proof; 16 MiB context protection; all headless | A3, A6, B5 |
| C2 | `lmpu trace` CLI | Replay, gap detection, redaction, timeline summary | A2 |
| C3 | Minimal dashboard | Read-only localhost: live events, job board, parked-host panel | C2, B5 |

**Deferred:** full 12-demo set, Mission Control scenario runner, rich filters, buffer previews, recorded walkthroughs.

---

## 5. Gate D — Release surface

**Goal:** publish a coherent, reproducible artifact.

| ID | Work | Acceptance | Depends |
|---|---|---|---|
| D1 | RFC 0001 + JSON Schema | Protocol examples validate; trace schema frozen | A2, B5 |
| D2 | Docs + SDK + packaging | Quickstart matches code; plugin example runs; `lmpu==0.1.0` builds | C1, C2 |
| D3 | Release checklist | License, SBOM, pins, signed artifact, security notes, reproducible tag | D2 |

---

## 6. Gate E — Optional validation

These items do not block A–D. Each has a kill criterion.

| ID | Work | Kill criterion |
|---|---|---|
| E1 | vLLM parking spike | If parking needs a fork or KV retention fails, keep HF authoritative |
| E2 | Constrained-decoder experiment | If control-token reduction is not material, keep free-text + sentinel parsing |
| E3 | nsjail/Docker profiles | If deployment does not require them, defer to P6 |
| E4 | S2/S3, child jobs, `NeedsMoreInput` | Add only when a canonical demo or benchmark needs them |
| E5 | P8 extensions | Keep out of v0.1.0 |

---

## 7. Consolidation decisions

- **RFC + schema:** write once in D1, then update incrementally. Do not maintain parallel documents.
- **ADRs:** batch into one short decision record before D1.
- **Contract tests:** generate per-op cases from manifests after op sets freeze; keep the representative suite as the gate.
- **Demos:** three canonical demos, each backed by automated assertions. No demo-only behavior.
- **Metrics:** one schema in A2. Trace CLI, dashboard, energy, and benchmarks consume it.
- **Dashboard:** one minimal viewer. Defer Mission Control and multi-task YAML.
- **Dependencies:** keep vLLM/Outlines in an optional isolated environment. Do not force Python 3.12/3.13 into the core environment.
- **Lint/types:** changed-file policy during development; one hygiene pass before D3.

---

## 8. Progress

### Gate A
- [ ] A1 HFDecoder KV reuse
- [ ] A2 Metrics/trace schema
- [ ] A3 S1 park/resume proof
- [ ] A4 Grants/refcounts/TTL/reaper audit
- [ ] A5 Routing/auth split
- [ ] A6 16 MiB context bound
- [ ] A7 Environment reproducibility

### Gate B
- [ ] B1 Harness
- [ ] B2 Baselines
- [ ] B3 Energy sampling
- [ ] B4 Initial workloads
- [ ] B5 C1–C5 report

### Gate C
- [ ] C1 Three canonical demos
- [ ] C2 `lmpu trace` CLI
- [ ] C3 Minimal dashboard

### Gate D
- [ ] D1 RFC 0001 + JSON Schema
- [ ] D2 Docs + SDK + packaging
- [ ] D3 Release checklist

### Gate E
- [ ] E1 vLLM spike
- [ ] E2 Constrained-decoder experiment
- [ ] E3 Sandbox profiles
- [ ] E4 S2/S3, child jobs, `NeedsMoreInput`
- [ ] E5 P8 extensions

---

## 9. Immediate next actions

1. **A7:** fix `uv.lock`/optional dependency isolation; confirm reproducible test command.
2. **A1:** implement cross-turn KV reuse in `HFDecoder`; add forward-pass assertion.
3. **A2:** define and emit the metrics schema from the HF session path.
4. **A4/A5:** batch the grants/reaper audit and routing/auth split.

---

## 10. Traceability to `TODO.1.md`

| New | Old |
|---|---|
| A1 | P3-02 follow-up |
| A2 | P0-07, P3-11 |
| A3 | P3-08 |
| A4 | P1-06 |
| A5 | P1-07 |
| A6 | P1-14, P5.5-06 |
| A7 | P0-02, P0-03, P0-08 (environment) |
| B1–B5 | P5.5-01 … P5.5-05 |
| C1 | P1-10, P2-10, P3-10, P4-07 |
| C2 | P5-01 |
| C3 | P5-02, P5-03, P5-08 (minimal) |
| D1 | P0-04, P7-01 |
| D2 | P7-02 … P7-05 |
| D3 | P7-09 |
| E1 | P6-01, P6-02 |
| E2 | P3-04 |
| E3 | P2-12 |
| E4 | P3-09, P4-02, P4-05, P8-01 |
| E5 | P8-02 … P8-14 |

---

## 11. Deferred backlog

Not cancelled, intentionally out of the v0.1.0 critical path:

- Full 12-demo set.
- Mission Control scenario runner and rich dashboard panels.
- B1–B12 full workload matrix and all ablations.
- vLLM fork, SQLite journal, reconnect/gap handling, 1-hour soak (promote to a gate only if vLLM becomes authoritative).
- S3 checkpointing, CUDA IPC, WASM, remote processors, multi-tenancy, formal traces, MCP, second runtime, speculative jobs, multimodal buffers, hardware accelerators, redundant execution.
- Full repo-wide `mypy`/`ruff` hygiene.
