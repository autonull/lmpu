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
- Cross-turn KV reuse: injected text and generated tokens append to the live cache; full prefill happens only on a non-extension prompt.
- Canonical host metrics schema: prefill/generated/injected/control tokens, forward passes, wall time, VRAM peak, and NVML energy hooks; one `host_metrics` trace record per session.
- Failure matrix: timeout, OOM, crash/recovery, disconnect, duplicate delivery, malformed worker messages.
- Trace JSONL emission, replay, gap detection, redaction.
- Session authorization boundary: capability advertisement stays registry-wide; buffer and processor access are session-scoped.
- Grant/refcount per-buffer release, TTL reaping, close-time shm cleanup, and fd-leak assertions.
- Benchmark harness: TOML/JSON config, explicit seeds, warmups, measured trials, resume-safe `raw.jsonl`, hardware/commit manifest, summary medians and bootstrap CIs, and a real-runtime sync trial runner.
- Fair baselines: direct CoT, labelled trace-token emulation, PAL-style sync tool, and summary-only async polling, plus `lmpu.sync`; randomized per-trial order, baseline-aware resume keys, and per-baseline summaries. Model baselines use the shared HF host and decoding budget.
- Energy: NVML idle probe before every trial, per-model-trial GPU joules, idle-rate correction metadata, and a documented methodology in `docs/benchmarks.md`.
- Initial workloads: `bigint_mul`, `matmul` (512²), `large_context` (12k-char secret lookup), and `sleep` (3 s); all verified through `lmpu.sync` one-trial artifacts.
- C1–C5 report generator: `src/lmpu/bench/report.py`; first published artifacts in `benchmarks/reports/gate-b-initial/` with raw data, medians, CIs, CDFs, timeline, and explicit claim limitations.
- CPU suite: **132 passed, 5 skipped**; opt-in GPU tests: **3 passed**. B1–B5 add 15 targeted tests (10 unit, 5 integration).

### Known gaps
- Trace/metrics lack CPU energy coverage (RAPL is root-only on the reference host).
- P1 exit gate is not closed: canonical demo assertions and the 16 MiB dashboard proof (Gate C).
- Report claims are partial: C1/C3/C5 are bounded by current workloads and failed strong baselines; C2 lacks `lmpu.sleep`; C4 lacks a batching probe.
- Trace-token baseline is a labelled simulation; a small real-model trace run remains.
- CPU RAPL energy is not measured; NVML GPU energy is the only energy path.
- Qwen3.5-2B, vLLM 0.33, and Outlines 0.1.14 are unverified (Gate E).
- Legacy `mypy`/`ruff` findings remain outside the changed-file policy.

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

**Status:** CLOSED 2026-11-03.

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
- [x] A1 HFDecoder KV reuse — `tests/integration/test_kv_reuse.py`, opt-in `test_hf_decoder_reuses_kv_across_injected_turns`
- [x] A2 Metrics/trace schema — `src/lmpu/host/metrics.py`, `tests/unit/test_metrics.py`
- [x] A3 S1 park/resume proof — `test_s1_park_freezes_forwards_and_resume_matches_baseline`
- [x] A4 Grants/refcounts/TTL/reaper audit — `test_release_grants_decrements_only_the_matching_buffer`, `tests/integration/test_authorization.py`
- [x] A5 Routing/auth split — `route_job(..., allowed_processors=...)`, `test_session_authorization_is_separate_from_advertisement`
- [x] A6 16 MiB context bound — `test_16mib_result_uses_at_most_64_context_tokens`
- [x] A7 Environment reproducibility — `uv.lock` resolves, `docs/optional-vllm.md`

### Gate B
- [x] B1 Harness — `src/lmpu/bench/`, `configs/benchmarks.toml`, `tests/unit/test_bench_harness.py`, `tests/integration/test_bench_runtime.py`; `make bench-smoke` runs through the real runtime
- [x] B2 Baselines — `src/lmpu/bench/baselines.py`, baseline-aware harness records/summary/resume, `--baseline` CLI, `tests/integration/test_bench_baselines.py`; direct CoT, trace-token, sync tool, async polling, and LMPU sync
- [x] B3 Energy sampling — `src/lmpu/bench/energy.py`, per-trial idle metadata and corrected GPU joules, `docs/benchmarks.md`
- [x] B4 Initial workloads — `matmul` and `text_stats` ops, `matmul` / `large_context` / `sleep` workload specs, `tests/unit/test_bench_workloads.py`, `tests/integration/test_bench_processor_ops.py`
- [x] B5 C1–C5 report — `src/lmpu/bench/report.py`, `make report`, first artifacts in `benchmarks/reports/gate-b-initial/`; C2–C5 limitations recorded

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

Gate A is closed and the first B5 report is published. Continue Gate B evidence work:

1. Close C2 by wiring an `lmpu.sleep` S1 treatment and measuring zero forwards.
2. Close C4 with a batched-vs-individual submission probe.
3. Replace C3's 12k context probe with a 16 MiB result probe.
4. Add a separately measured native baseline for C5.

Keep Gate C closed until the partial claims are resolved or explicitly accepted.

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
