# TODO.1 — LMPU Development Plan

**Status:** Accepted baseline for implementation — 2026-11-03  
**Supersedes:** `TODO.md` (proof-of-concept blueprint)  
**Companions:** `README.md` (architecture specification), `compared.md` (positioning)  
**Repository state:** CPU reference runtime is running end to end (49 tests green, 6 integration tests); GPU/model paths are not yet verified. See §0.3 and §20.

> **Thesis.** The host language model is a semantic control plane. Native processors do the computation. The host submits a small job, parks its generation loop, and wakes to a compact buffer reference. The expensive model is used for decisions, not for simulating execution.

---

## 0. How to use this plan

This is an execution plan, not a second specification. It defines:

- the architecture we will actually build;
- the contracts that must remain stable;
- the order of work and the gate that closes each phase;
- the test and benchmark strategy that keeps the project honest;
- the decisions that are accepted and how changes are controlled.

Checkbox conventions:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[!]` blocked; record the blocker in the phase log

Estimates use `S` (≤1 day), `M` (2–4 days), `L` (1–2 weeks). They assume one senior engineer familiar with Python async, PyTorch, and Linux IPC. With two engineers, parallel tracks are called out in §13.

**Release target for this plan:** `lmpu 0.1.0`, a single-node, single-tenant, Linux-first proof of concept that is credible enough to support an RFC, a systems paper, and external plugin authors.

### 0.1 Reference development environment

Measured on 2026-11-03:

| Item | Value | Consequence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB, GA102 | 0.8B comfortable; 2B comfortable; 4B tight; 9B needs quantization |
| Driver / CUDA | 615.78.08 / CUDA 13.4 | Recent enough for current PyTorch |
| GPU power | 13.5 W idle, 320 W cap; NVML `power.draw.average` available (±5 W) | GPU energy can be sampled without root |
| CPU energy | Intel RAPL present, but `energy_uj` is root-only | CPU joules need a udev rule or privileged helper |
| Host RAM / shm | 16 GiB `/dev/shm` | 5 GB synthetic buffers are feasible in shm |
| Disk | 26 GiB free (95% used) | Tight for 4B/9B weights and artifacts; 0.8B/2B preferred |
| Python | 3.14.8 | Use 3.12/3.13 if vLLM or a decoder lacks 3.14 wheels |
| Libraries | PyTorch 2.15.0, Transformers 4.57.6, tokenizers 0.22.2, NumPy 2.5.4 | Transformers 4.57.6 rejects `qwen3_5`; latest is 5.22.1. vLLM 0.33.0 and Outlines 0.1.14 are available but not installed |
| HF cache | Qwen3.5-0.8B tokenizer/config present; weights must be downloaded | First run needs network and ~1.6 GB |
| Dependency policy | See Appendix D | Optional extras keep the core CPU-only |

This table is part of the plan: benchmark results must record the actual host, not the reference host.

### 0.2 Non-goals for 0.1

- No multi-tenant isolation, fair-share scheduling, or remote authentication.
- No remote processors, GPU buffers, CUDA IPC, or S3 checkpoint/migration.
- No production-grade sandbox; public mode uses a safe profile, but the threat model is still limited.
- No end-to-end fine-tuned control policy; constrained decoding and prompting are the 0.1 path.
- No formal verification, hardware attestation, or exactly-once execution.
- No full OS, hypervisor, or distributed scheduler claim.

These are scoped out so the PoC can prove the execution model without turning into an infrastructure project.

### 0.3 Implementation progress (2026-11-03)

This section is the running progress record. It is updated before each commit; phase
checkboxes below carry the same status. Status means *verified by a test or a run*,
not merely "a file exists".

| Area | Status | Evidence |
|---|---|---|
| Package, CLI, test tree, dependency metadata | Done | `pyproject.toml`, `src/lmpu/`, `tests/`, `Makefile` |
| Protocol models, discriminated unions, JSON-RPC codec | Done | `src/lmpu/protocol/`, `tests/unit/test_protocol.py` |
| Job state machine, idempotency, terminal waiters | Done | `src/lmpu/kernel/jobs.py`, `tests/unit/test_kernel.py` |
| Event log: sequence, ring buffer, ack, gap detection | Done | `src/lmpu/kernel/events.py`, unit tests |
| Buffer manager, shm backend, NumPy views, BLAKE3 | Done | `src/lmpu/memory/`, `tests/unit/test_memory.py` |
| Grants, TTL, reaper, provenance | Partial | grants/TTL/reaper exist; provenance not fully audited |
| Capability registry, routing, capability query | Partial | registry and routing exist; no routing contract test |
| RPC surface (`hello` … `close`) | Done | `src/lmpu/kernel/runtime.py` |
| Worker execution path (dispatch, sync submit, wait, timeout, cancel) | Done | `tests/integration/test_runtime_execution.py` (6 tests) |
| Explicit batch API and async batch coalescing with per-job fanout | Done | integration test; manual 100-item batch run |
| `pure.func`, `python.vm`, `vector.engine`, `symbolic.planner`, `neural.summary`, `hardware.stub` | Partial | all register and return results; all six pass the representative contract suite (`tests/contract/test_processor_contract.py`); per-op coverage is incomplete |
| `vector.engine` | Partial | fd-mapped 16 KiB matmul completes with `copy_count == 0` and refreshed output checksum; other ops still use placeholder paths |
| Worker fd passing | Partial | `SCM_RIGHTS` works end-to-end; inline base64 fallback is implemented and tested |
| Failure injection | Done | Real-worker timeout/OOM/crash/disconnect plus deterministic fake-worker delayed, duplicate, and malformed-message cases |
| `python.vm` timeout and cancellation enforcement | Partial | parent-side hard cancel verified; child-side sandbox limits incomplete |
| Host engine ABC, scripted engine, parser, grammar, session controller | Partial | scripted session loop, injection policy, context/event budgets, `ResultTooLarge`, parser-failure recovery, and forced wake are covered by `tests/scenario/test_session_loop.py`; HF engine still unverified |
| HF engine and GPU path | Partial | Qwen3.5-0.8B loads via `Qwen3_5ForConditionalGeneration`; 23.8 tok/s decode, 1691.8 MiB peak VRAM. `HFDecoder` runs inside `SessionController`; per-turn prefill (no cross-turn KV reuse). 2B, vLLM, and Outlines unverified |
| Trace sink and metrics | Partial | JSONL sink, redaction, and replay tests exist (`tests/unit/test_trace.py`); host token/energy metrics missing |
| Benchmarks, Mission Control, RFC/SDK/docs | Not started | directories exist only |

**Immediate next work (in order).**

1. Optimize `HFDecoder`: append injected text to the live KV cache instead of re-prefilling each turn.
2. Finish P0-08 where the environment allows: Qwen3.5-2B load, vLLM 0.33, Outlines 0.1.14.
3. Wire trace token/energy metrics into the HF session path.

**Completed in this pass (2026-11-03).**

- Processor contract suite for all six built-ins: manifest truth, representative op success plus checksum, progress-claim backing, unknown-op errors, strict determinism.
- Property tests for JSON-RPC framing round-trip and typed buffer import/slice/write without NumPy divergence.
- Trace replay helper (`read_trace`, `replay_trace`, contiguous-gap detection) and sensitive-payload redaction by default.
- Processor checksums now use the shared BLAKE3 helper instead of `hashlib.blake3`, which does not exist on Python 3.14.
- Scripted host loop: `ScriptedDecoder`, runtime session adoption, policy-driven event injection, event/context budgets, `ResultTooLarge`, parser-failure recovery, engine-error handling, and forced wake.
- Scenario suite `tests/scenario/test_session_loop.py` covers nine model-free controller cases.
- `SCM_RIGHTS` fd passing over a per-worker socketpair, worker-side mmap, output-buffer checksum refresh, inline base64 fallback, and `copy_count`/`mapped_bytes` evidence in `JobCompleted`. End-to-end matmul test: `tests/integration/test_zero_copy_matmul.py`.
- Failure-injection suite: real-worker timeout, OOM, crash/recovery, disconnect cleanup; event-id dedupe and duplicate-completion guards; event-before-sleep returns `already_satisfied`.
- All remaining `vector.engine` ops write mapped outputs through one `_finalize` path; 16 MiB bounded-slice and mapped-softmax acceptance tests added.
- P0-08 baseline for Qwen3.5-0.8B on RTX 3080: Transformers 5.22.1, Python 3.14, `Qwen3_5ForConditionalGeneration`, 23.8 tok/s decode, 1691.8 MiB peak VRAM. Command: `python scripts/bench_hf.py`. Fixed `AutoModelForCausalLM` → `AutoModelForImageTextToText` and the decode attention mask; this raised decode from 6.1 to 23.8 tok/s.
- `HFDecoder` adapter wires the reference engine into `SessionController`; CLI `run --engine hf` now uses it. Opt-in test proves prefill/decode inside the session loop.
- Deterministic fake-worker suite: delayed result, duplicate terminal delivery, error-then-late-result, invalid JSON, unknown type, malformed progress, and non-object results. Hardened worker-message handling against malformed payloads.

**P1 exit gate — still open.**

- Grants/refcount release audit and negative authorization tests (P1-06).
- Routing contract test and advertisement/authorization separation (P1-07).
- Demos 1 and 2 in headless assertion mode (P1-10).
- End-to-end 16 MiB buffer slice read and buffer-import provenance checks (P1-14).

**Known red items.**

- `make check` is not green: `mypy` still reports pre-existing annotation errors; `ruff` is runnable but not yet clean.
- P0-08 remaining: Qwen3.5-2B, vLLM 0.33, and Outlines 0.1.14 are not installed/verified; 0.8B is verified.
- `HFDecoder` re-prefills every turn; cross-turn KV reuse is not implemented.

---

## 1. Naming, scope, and decisions

### 1.1 Naming

| Concept | Name | Notes |
|---|---|---|
| Architecture | **LMPU** — Language Model Processing Unit | Parallels CPU/GPU/TPU; the model is the control plane, the LMPU is the execution plane |
| Runtime | **LMPU Runtime** or **LMPU Kernel** | Avoid calling the PoC a full OS; it is an OS-like kernel |
| Protocol | **LMPU Protocol v1** | Wire version string `lmpu/1` |
| Python package / CLI | `lmpu` | One package, one CLI |
| Model-facing sentinels | `<\|proc\|> ... <\|/proc\|>` and `<\|proc_event\|> ... <\|/proc_event\|>` | Recommended; see §1.2 |
| Unit of work | **job** | Never “coproc call” in code |
| Execution backend | **processor** | Replaces `coprocessor` in wire messages; shorter and clearer |
| Host suspension | **park / resume** internally; **sleep / wake** in the model protocol | Avoids collision with vLLM’s global `sleep()` |

`LMPU` describes the execution unit; the contribution is the *runtime and protocol* around it. Documentation should say “LMPU architecture” and “LMPU Runtime,” not imply that a new chip is required.

### 1.2 Sentinel selection

`<proc>` is short, but it collides with:

- the Unix `/proc` filesystem and general “process” terminology;
- HTML/XML-like tag parsing and documentation renderers;
- ordinary prose and code comments;
- tokenizer splitting and stop-sequence matching.

`<|proc|>` follows the reserved-special-token convention used by modern tokenizers (`<|endoftext|>`, `<|im_start|>`). It is visually unambiguous, unlikely to occur naturally, and easier to scan incrementally across token boundaries. The plan assumes `<|proc|>`; changing it is one constant in `lmpu/protocol/sentinels.py`, but the project should ship exactly one canonical form.

**Tokenizer measurement.** With `Qwen/Qwen3.5-0.8B` and Transformers 4.57.6:

| Sentinel | Tokenization | Token count |
|---|---|---|
| `<\|p\|>` | `<`, `\|`, `p`, `\|`, `>` | 5 |
| `<\|proc\|>` | `<`, `\|`, `proc`, `\|`, `>` | 5 |
| `<\|process\|>` | `<`, `\|`, `process`, `\|`, `>` | 5 |
| `<\|/p\|>` | `<`, `\|`, `/`, `p`, `\|`, `>` | 6 |
| `<\|/proc\|>` | `<`, `\|`, `/`, `proc`, `\|`, `>` | 6 |
| `<\|/process\|>` | `<`, `\|`, `/`, `process`, `\|`, `>` | 6 |

The shortest string does **not** produce fewer tokens. `<|p|>` becomes a one-token win only if it is added as a special token, which resizes the embedding matrix and requires the model to learn a new token. Constrained decoding can force the token, but free-generation validity and model likelihood suffer. Recommendation: keep `<|proc|>` as a string sentinel for 0.1; revisit special-token training only after the free-generation baseline exists.

### 1.3 Decision register

| ID | Decision | Choice | Status |
|---|---|---|---|
| D-01 | Architecture and protocol name | LMPU; `lmpu/1`; `<\|proc\|>` string; P3-13 tests `<\|p\|>` as a special token | Accepted |
| D-02 | Core language and tooling | Python 3.12+, asyncio, `uv`, `pydantic` v2, `orjson`, NumPy | Accepted |
| D-03 | Control-plane transport | JSON-RPC 2.0 over Unix domain socket; WebSocket later | Accepted |
| D-04 | Buffer plane | POSIX shared memory + `mmap`; `SCM_RIGHTS` fd passing; BLAKE3 | Accepted |
| D-05 | Host engine strategy | `HostEngine` abstraction; HF reference first; vLLM parking via a version-pinned monkey-patch adapter in Phase 6; scripted engine for tests | Accepted |
| D-06 | Sleep levels | S1 (KV resident) default; S2 (KV offload) under memory pressure; S3 (checkpoint) deferred | Accepted |
| D-07 | Event delivery | Per-session monotonic sequence; at-least-once; ack + replay | Accepted |
| D-08 | Idempotency | Client-supplied `job_id` plus optional idempotency key | Accepted |
| D-09 | Sandbox posture | Pluggable; subprocess + rlimits for trusted development; safe profile required for public mode; never a security boundary by itself | Accepted |
| D-10 | Test gates | CPU `core` lane always; GPU `e2e` lane required for release | Accepted |
| D-11 | Evidence strategy | Pre-registered benchmark, five baselines, CDR and Token Tax metrics | Accepted |
| D-12 | License | Apache-2.0 | Accepted |
| D-13 | Model-facing surface | Small, safe subset; buffer CRUD lives in the trusted admin API | Accepted |
| D-14 | Processor terminology | `processor` in wire and code; `co-processor` only in prose | Accepted |
| D-15 | Sleep authority | Configurable: `model`, `controller`, or `model_with_policy`; default `model_with_policy` | Accepted |
| D-16 | vLLM integration | Version-pinned monkey-patch adapter preferred; no fork by default; SGLang fallback | Accepted |
| D-17 | Sandbox selection | Configurable profile; public mode requires nsjail/Docker; subprocess only for trusted local development | Accepted |
| D-18 | Event injection | Configurable per event and per session; safe defaults | Accepted |
| D-19 | Flagship demo | Mission Control dashboard with multiple concurrent demo tasks | Accepted |
| D-20 | Primary evidence claim | Zero host forwards plus O(1) context are primary; GPU energy is secondary | Accepted |
| D-21 | Tenancy | Single-tenant 0.1; admission hooks only | Accepted |
| D-22 | Buffer provenance | `buffer.import` from files and bytes in 0.1; URLs and databases later | Accepted |
| D-23 | Portability claim | Interface portable; reference implementation transformer-only | Accepted |
| D-24 | Decoding evidence | Report constrained and unconstrained validity; defer SFT | Accepted |
| D-25 | Context-bloat scale | 1 GB for CI; 5 GB for the paper | Accepted |
| D-26 | Energy measurement | GPU joules in 0.1; CPU RAPL optional | Accepted |
| D-27 | Model choice | Qwen3.5-0.8B for CI; Qwen3.5-2B for demos; 4B quantized optional | Accepted |
| D-28 | MCP positioning | Separate layer with a bridge | Accepted |

### 1.4 Changes from `TODO.md`

The blueprint is sound; these changes make it buildable and defensible:

1. **Rename the system and wire fields.** `coproc` becomes `lmpu`; `coprocessor` becomes `processor`; sentinels become reserved-token form. This avoids a crowded namespace and clarifies the contribution.
2. **Split model plane from control plane.** The model emits small JSON blocks. The trusted session controller translates them to JSON-RPC. The model never gets buffer-delete or quota authority.
3. **Add delivery semantics.** Sequence numbers, event IDs, ack cursors, replay on resume, and idempotent submit are mandatory. Without them, async wake is a lost-event bug generator.
4. **Make the job state machine explicit.** Terminal states are immutable and exactly one terminal event is emitted. Cancellation has a stated linearization point.
5. **Replace “callback mode” with execution mode.** `sync`, `async`, `interrupt` describe behavior; callbacks are an implementation detail.
6. **Define sleep levels.** “Sleep” is not one thing. S1 keeps KV resident, S2 offloads it, S3 checkpoints. The energy claim depends on which one is measured.
7. **Separate sleeping jobs from sleeping sessions.** A processor can wait for input or an event; the host can park its generation loop. They are related but not the same state.
8. **Use shared memory properly.** `/dev/shm` plus `mmap` plus fd passing; BLAKE3 for integrity; Arrow only for tabular/string data. Do not promise “zero-copy” where a copy remains.
9. **Treat vLLM sleep as a different feature.** vLLM’s global sleep offloads weights to free VRAM. The LMPU claim needs *per-request parking with KV retention*. That is a scheduler feature, not the existing global sleep primitive.
10. **Make tests model-free by default.** The GPU lane proves the integration; it does not carry the correctness of the kernel. This keeps CI fast and avoids brittle prompt tests.
11. **Pre-register the benchmark.** Strong baselines, raw results, confidence intervals, and ablations. A demo is not evidence.
12. **Delay the dashboard.** Trace-first observability; the dashboard is a view over the trace, not the source of truth.

---

## 2. Target architecture

### 2.1 Layers

```text
┌───────────────────────────── Host process ──────────────────────────────┐
│ HostEngine (HF / vLLM / scripted / OpenAI-degraded)                     │
│   ├─ SentinelScanner + two-phase constrained decoder                    │
│   ├─ SessionController: generate · parse · dispatch · park · resume     │
│   └─ HostMetrics: forward passes · tokens · context · energy            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ JSON control blocks (model plane)
                                │ JSON-RPC over UDS (control plane)
┌───────────────────────────────▼─────────────────────────────────────────┐
│ LMPU Kernel                                                             │
│   Router · CapabilityRegistry · JobTable · Scheduler · EventLog          │
│   Clock · Timers · Quotas · Reaper · TraceSink · Admin API              │
└───────────────┬───────────────────────────────┬─────────────────────────┘
                │ UDS + SCM_RIGHTS              │ read-only HTTP/SSE
                ▼                               ▼
      Worker pool (subprocess / nsjail)     Dashboard + trace viewer
      pure.func · python.vm · vector.engine
      symbolic.planner · neural.summary · hardware.stub
                │
                ▼
      /dev/shm buffers (mmap, typed views, grants)
```

### 2.2 Module map

```text
lmpu/
  protocol/        messages, JSON-RPC codec, errors, sentinels, schema
  kernel/          runtime facade, jobs, scheduler, events, timers, quotas, reaper
  memory/          buffer manager, shm backend, checksums, grants, views
  processors/
    base.py        Processor ABC, manifest, context
    worker.py      worker main loop and parent-side client
    sandbox/       subprocess, nsjail, docker, wasm (stub)
    builtin/       pure_function, python_vm, vector_engine, symbolic_planner,
                   neural_summary, hardware_stub
  host/
    engine.py      HostEngine ABC and HostState
    hf.py          Hugging Face reference engine
    vllm.py        parked-request adapter (Phase 6)
    scripted.py    deterministic engine for tests
    parser.py      sentinel scanner, validator, repair policy
    grammar.py     two-phase constrained decoding
    session.py     SessionController
    prompts/       versioned prompt templates
    metrics.py     forward/token/context accounting
  observability/   trace, metrics, dashboard
  bench/           workloads, baselines, metrics, plots
  synth/           synthetic trace generator (Phase 8)
  cli.py           run, demo, bench, trace, doctor, plugin
configs/
demos/
tests/
docs/
  rfcs/0001-lmpu-protocol.md
  architecture.md
  threat-model.md
  benchmarks.md
  adr/             architecture decision records
```

### 2.3 Flagship data flow

1. The controller prefills the prompt with a compact capability summary.
2. The host decodes free text; the scanner stops at `<|proc|>`.
3. The constrained decoder emits a valid `submit` message; the controller validates it.
4. The kernel validates buffers, grants, quotas, and processor capability. It returns `accepted`.
5. The host emits `sleep` for the job ID. The kernel registers the waiter *atomically with a check for an already-terminal job*.
6. The host parks: no decode calls, no forward passes, no token generation. S1 keeps the KV cache in VRAM.
7. The worker executes at native speed. Progress events go to the event log and dashboard; they are not injected by default.
8. On completion, the kernel appends `JobCompleted` with a sequence number and resolves the waiter.
9. The controller resumes, injects a compact event block, and continues generation.
10. If the host wants data, it requests a bounded slice. The full buffer never enters context.

### 2.4 Invariants

These are test targets, not aspirations.

- **I1.** Every job emits exactly one terminal event (`JobCompleted`, `JobFailed`, or `JobCancelled`).
- **I2.** Per-session event sequence numbers are strictly increasing. Retained events form a contiguous suffix; any eviction gap is reported on resume, never hidden.
- **I3.** Buffer bytes enter model context only through an explicit read, bounded by `max_inline_bytes`.
- **I4.** A job cannot read or write a buffer without a matching grant.
- **I5.** During S1 park, host forward passes and generated tokens are exactly zero.
- **I6.** Terminal state is immutable. Cancellation linearizes at the atomic terminal transition; no completion event follows a cancellation.
- **I7.** Every state transition is journaled with a cause and timestamp.
- **I8.** Every completed result carries a checksum and provenance (`processor`, version, config hash, determinism class).
- **I9.** Control-plane messages are bounded: ≤4 KiB per message, ≤64 KiB per injected event, ≤512 control tokens per model block.
- **I10.** Unknown protocol majors are rejected; unknown additive fields are tolerated within a major.

---

## 3. LMPU Protocol v1

### 3.1 Two surfaces

| Surface | Speaker | Transport | Authority |
|---|---|---|---|
| Model plane | Host model ↔ session controller | JSON blocks inside sentinels | Small safe subset only |
| Control plane | Session controller / tools ↔ kernel | JSON-RPC 2.0 over UDS | Jobs, buffers, quotas, admin |
| Admin plane | CLI / dashboard ↔ kernel | Same JSON-RPC methods, separate namespace | Full buffer CRUD, traces, health |

The model MUST NOT be able to delete buffers, change quotas, or read another session’s memory. The controller mediates and can refuse a model request before it reaches the kernel.

### 3.2 Control-plane envelope

Use JSON-RPC 2.0 for the trusted wire. This avoids inventing request IDs, error objects, batches, and notifications.

```json
{"jsonrpc":"2.0","id":"c-17","method":"lmpu.submit","params":{...}}
{"jsonrpc":"2.0","id":"c-17","result":{"job_id":"job_01H...","status":"pending","seq":41}}
{"jsonrpc":"2.0","method":"lmpu.event","params":{"session":"s_1","seq":42,"event":{...}}}
```

Methods:

| Method | Purpose |
|---|---|
| `lmpu.hello` | Open session, negotiate version, resume from sequence |
| `lmpu.capabilities` | Query processors and ops |
| `lmpu.submit` | Submit a job or batch |
| `lmpu.poll` | Read status without blocking |
| `lmpu.wait` | Wait for terminal state with timeout |
| `lmpu.read` | Read a bounded buffer view |
| `lmpu.write` | Write a buffer or interactive field |
| `lmpu.cancel` | Request cancellation |
| `lmpu.sleep` | Register a host wake condition |
| `lmpu.wake` | Force wake or cancel a sleep registration |
| `lmpu.ack` | Advance the client’s event cursor |
| `lmpu.buffer.*` | Create/import/export/stat/checksum/delete (admin) |
| `lmpu.jobs.list` | Dashboard and diagnostics |
| `lmpu.configure` | Update session limits or runtime policy (admin) |
| `lmpu.health` | Liveness, queue depth, worker health |
| `lmpu.close` | Close session; cancel or detach jobs |

Response shapes:

```json
{"job_id":"job_01H...","status":"pending","queue_position":3}
{"job_id":"job_01H...","status":"running","progress":0.42,"elapsed_ms":812}
{"status":"completed","result":{"summary":{"shape":[2048,2048],"dtype":"float32","bytes":16777216},"buffer":"buf_C","checksum":"blake3:..."}}
{"buffer":"buf_C","view":{"slices":[[0,4],[0,4]]},"dtype":"float32","data":[[...]]}
{"matches":["vector.engine","hardware.stub"],"recommended":"vector.engine","details":[...]}
```

`lmpu.configure` accepts session quotas and policy overrides:

```json
{"max_outstanding_jobs":32,"max_buffer_bytes":2147483648,
 "max_sleep_seconds":300,"sleep_level":"S1",
 "injection":{"generating":{"JobCompleted":"next_turn"}}}
```

It is admin-only and cannot exceed runtime maxima. A rejected override returns `quota_exceeded` with the limiting field.

### 3.3 Model-plane message kinds

Keep the README vocabulary for traceability:

```text
submit, poll, wait, read_result, write_input, cancel, sleep, wake,
query_capabilities
```

`configure` and buffer lifecycle are admin-only. A model may request a capability query; it may not grant itself resources.

Example submit:

```json
{
  "version": 1,
  "kind": "submit",
  "job_id": "job_vec_001",
  "processor": "vector.engine",
  "op": "matmul",
  "inputs": {
    "A": {"buffer": "buf_A"},
    "B": {"buffer": "buf_B"}
  },
  "outputs": {
    "C": {"buffer": "buf_C"}
  },
  "config": {
    "mode": "async",
    "timeout_ms": 10000,
    "priority": 5,
    "determinism": "strict"
  }
}
```

`job_id` is optional; if omitted, the kernel allocates one and returns it. If supplied, it doubles as the idempotency key.

Job-spec fields:

| Field | Required | Meaning |
|---|---|---|
| `processor` | No | Explicit processor; omit to route by capability |
| `op` | Yes | Operation name |
| `inputs` | No | Mapping of name to a JSON value or `{"buffer": "..."}` |
| `outputs` | No | Mapping of name to a buffer ref; runtime allocates if absent |
| `program` | python.vm only | Program text or path; hashed into the request |
| `batch` | No | `{op, items: [{inputs, config?}], max_parallel?}` for batch jobs |
| `config.mode` | No | `sync`, `async` (default), `interrupt` |
| `config.timeout_ms` | No | Wall-clock timeout; starts at submission |
| `config.deadline_ms` | No | Absolute deadline from submission |
| `config.priority` | No | Integer 0–9; higher runs first |
| `config.memory_limit_bytes` | No | Hard memory limit for the worker |
| `config.determinism` | No | `strict` or `best_effort` |
| `config.progress_interval_ms` | No | Minimum spacing for progress events |
| `config.idempotent` | No | Whether automatic retry is allowed |
| `config.ttl_ms` | No | Result retention after terminal state |

Batch example:

```json
{"version":1,"kind":"submit","processor":"pure.func","op":"bigint_add",
 "batch":{"items":[{"inputs":{"a":1,"b":2}},{"inputs":{"a":3,"b":4}}]},
 "config":{"mode":"async"}}
```

Interactive input:

```json
{"version":1,"kind":"write_input","job_id":"job_plan_002","field":"goal","value":"D"}
```

`write_input` is valid only while the job is `WAITING_INPUT`. It is idempotent for the same field and value. It can also write a bounded buffer slice for an interactive processor, but the target must be declared in the job’s inputs or fields.

Execution modes:

| Mode | Behavior |
|---|---|
| `sync` | Kernel waits for terminal state; RPC result contains the result summary. A trace event is still recorded. |
| `async` | Kernel returns `accepted`; completion is delivered as an event. |
| `interrupt` | Async, but completion and failure events may interrupt a generating segment at the next token boundary. |

### 3.4 Job state machine

```text
CREATED
  └─> PENDING ──> RUNNING ──> COMPLETED
         │           ├──────> FAILED
         │           ├──────> WAITING_INPUT ──> RUNNING
         │           ├──────> WAITING_EVENT ──> RUNNING
         │           └──────> CANCEL_REQUESTED ──> CANCELLED
         └──────────> CANCEL_REQUESTED ──> CANCELLED
```

- `PENDING` maps to the model-facing status `pending`.
- `RUNNING` maps to `running`.
- `WAITING_INPUT` and `WAITING_EVENT` map to `sleeping` in the model-facing status.
- `CANCEL_REQUESTED` is internal; it becomes `CANCELLED` when the worker stops or the hard-kill completes.
- Terminal states are `COMPLETED`, `FAILED`, `CANCELLED`.
- `WAITING_INPUT` and `WAITING_EVENT` may transition to `CANCEL_REQUESTED`.
- `WAITING_EVENT` is used when a processor explicitly waits for an external event. The job record stores the predicate. It is distinct from a parked host.
- A transition table in code is the single source of truth. Unit tests exhaustively cover allowed and forbidden transitions.

Cancellation rule: `cancel` performs an atomic compare-and-set from a non-terminal state to `CANCEL_REQUESTED`, then signals the worker. If the worker wins the race with `COMPLETED`, the job stays completed and the cancel response says `already_terminal`. If cancellation wins, the terminal event is `JobCancelled` and no `JobCompleted` is emitted. A cancellation test asserts that no terminal event is emitted twice and that no completed event follows a cancelled event.

### 3.5 Events

| Event | Injection default | Payload |
|---|---|---|
| `JobSubmitted` | observe | job id, processor, queue position |
| `JobStarted` | observe | worker, start time |
| `Progress` | observe (throttled) | fraction, checkpoint, message |
| `JobCompleted` | inject at next turn | status, result summary, buffer refs, checksum |
| `JobFailed` | inject + interrupt | error code, retriable, detail |
| `JobCancelled` | inject at next turn | reason |
| `NeedsMoreInput` | inject + interrupt | prompt, fields, job id |
| `Sleeping` | observe | session, sleep id, conditions |
| `Woken` | inject | reason, satisfied jobs, statuses |
| `BufferCreated` | observe | buffer ref |
| `BufferUpdated` | observe | buffer ref, checksum |
| `ResourceWarning` | observe | quota, current, limit |
| `ControlError` | inject | position, expected schema, message |
| `ResultTooLarge` | inject | buffer ref, why it was truncated |
| `Heartbeat` | never inject | worker liveness |

Injection policy is configured per session, not hard-coded in the event bus. Defaults:

- `JobCompleted`, `JobFailed`, `JobCancelled`, `NeedsMoreInput`, `Woken`, `ControlError`, `ResultTooLarge`: inject (next turn or wake, as configured).
- `Progress`, `JobSubmitted`, `JobStarted`, `Buffer*`, `Heartbeat`: observe only.
- `JobFailed` and `NeedsMoreInput` may interrupt a generating segment; `Progress` never does.
- The controller renders injected events as `<|proc_event|> ... <|/proc_event|>` blocks with a compact summary. The model never receives raw worker stdout.

```text
<|proc_event|>
{"version":1,"kind":"event","event":"Woken","job_id":"job_vec_001",
 "status":"completed","summary":{"output_buffer":"buf_C","shape":[2048,2048],
 "dtype":"float32","bytes":16777216,"checksum":"blake3:..."}}
<|/proc_event|>
```

### 3.6 Ordering, delivery, and resume

- Each session has a monotonically increasing `seq`, starting at 1.
- Every event has a ULID `event_id`, `seq`, `session`, `job_id`, `type`, `ts_monotonic`, and `data`.
- Delivery is at-least-once. The host deduplicates by `event_id`; the model context deduplicates by `(job_id, terminal)`.
- The kernel retains a ring buffer of events per session (default 10,000). Phase 6 adds optional SQLite durability.
- `hello` carries `resume_from`; the kernel replays retained events with `seq > resume_from` before new ones.
- `ack` advances the cursor. A session that never acks still receives events within retention.
- If a required event has been evicted, `hello_ok` returns `resume_status: "gap"` and the controller must reconcile job states explicitly. It must not silently continue.

### 3.7 Errors

```json
{
  "code": "quota_exceeded",
  "message": "Session buffer quota exceeded",
  "job_id": "job_01H...",
  "retriable": false,
  "details": {"limit_bytes": 2147483648, "current_bytes": 2200000000}
}
```

Canonical codes:

```text
invalid_schema, unknown_method, unknown_processor, unknown_op,
capability_mismatch, buffer_not_found, buffer_too_large, access_denied,
quota_exceeded, backpressure, timeout, cancelled,
worker_crashed, oom, checksum_mismatch, runtime_restarted,
protocol_version, idempotency_conflict, already_terminal, internal

`cancel_requested` is a job status, not an error. `cancelled` is the terminal error/status after the worker stops.
```

### 3.8 Versioning

- Wire major is `lmpu/1`; a client with a different major is rejected with `protocol_version`.
- Additive fields are allowed within a major. Unknown fields are ignored at the boundary but preserved in traces.
- Capability descriptors carry `protocol` and `processor_version`.
- `hello_ok` returns `schema_hash`; mismatch is a warning, not an automatic failure, unless `strict_schema: true`.
- No dual-format compatibility shims in 0.1. One canonical schema.

### 3.9 Model-facing grammar

The model emits free text and control blocks. The controller enforces:

```text
free text (any tokens)
<|proc|>
{ JSON per schema }
<|/proc|>
free text ...
```

Two-phase decoding:

| Phase | Decoder |
|---|---|
| `FREE` | Normal sampling with stop sequence `<\|proc\|>` |
| `CONTROL` | JSON-schema-constrained decoding; hard stop at `<\|/proc\|>`; max 512 tokens |
| `FREE` | Resume normal sampling |

Implementation choices, to be settled in a Phase 3 spike:

- `outlines` + Transformers, if the custom state machine is stable;
- `lm-format-enforcer`, if it integrates more cleanly with a manual loop;
- fallback: free JSON plus strict validation and a single retry.

Repair policy: **no silent repair by default.** Whitespace and a trailing comma may be normalized, and the fact is logged. Anything else returns `ControlError` with a byte position and a minimal example. Silent repair hides model errors and corrupts benchmark token counts.

Evidence policy: report both constrained and unconstrained control-block validity. Constrained decoding is allowed for the headline benchmark, with its overhead reported separately. Defer SFT to Phase 8 unless unconstrained validity is unusable.

---

## 4. Buffers and memory

### 4.1 Model

- A buffer is a typed, named region with an owner session, a checksum, a shape, and a dtype.
- Data lives in `/dev/shm/lmpu/<session>/<buffer_id>` on Linux; fallback to `/tmp` on other platforms.
- Metadata lives in a kernel table and a sidecar manifest for diagnosis.
- Workers receive file descriptors over `SCM_RIGHTS`, then `mmap` the region read-only or read-write. No base64, no JSON arrays for large data.
- The host sees a `BufferRef`: id, dtype, shape, bytes, checksum, and a short summary. Never the bytes, unless it explicitly reads a bounded slice.
- Each buffer records provenance: `source` (`inline`, `file`, `job`), producing job id, original path or hash, and the creating session. Provenance is visible in the trace and the dashboard.

`BufferRef`:

```json
{
  "buffer_id": "buf_C",
  "dtype": "float32",
  "shape": [2048, 2048],
  "bytes": 16777216,
  "checksum": "blake3:9f2c...",
  "owner": "s_1",
  "created_at": 1735920000.123,
  "ttl_ms": 3600000
}
```

### 4.2 Dtypes and layouts

- NumPy dtypes: `bool`, `int8/16/32/64`, `uint8/16/32/64`, `float16/32/64`, `complex64/128`.
- `bfloat16` via `ml_dtypes`; required for parity with model tensors.
- Opaque bytes and UTF-8 strings as `bytes`; Arrow IPC for tables and columnar data.
- C-contiguous by default; stride-aware views for slices and transposes.
- Shape and stride validation at every boundary. Negative strides are rejected in v1.

### 4.3 Operations

```text
create(name, dtype, shape, persist=False) -> BufferRef
import_bytes(name, data, dtype, shape) -> BufferRef
import_file(name, path, dtype, shape) -> BufferRef  # path under import_roots, symlink-checked
export_bytes(ref, offset, length) -> bytes        # bounded, admin
export_file(ref, path) -> None                    # bounded, admin
read_view(ref, slices, max_inline_bytes) -> View | BufferRef
write_view(ref, slices, data) -> BufferRef
stat(ref) -> BufferRef + summary
checksum(ref, algorithm="blake3") -> str
delete(ref) -> None
```

Views use per-axis `start:stop:step`. A view either returns inline JSON (≤ `max_inline_bytes`, default 64 KiB) or a derived `BufferRef` with a note telling the host to read a smaller slice. The controller enforces the context budget after the kernel returns.

### 4.4 Integrity and lifecycle

- BLAKE3 by default; SHA-256 available for external audit compatibility.
- Checksums are computed on write for buffers ≤64 MiB; lazy for larger; `verify_on_read` is configurable.
- Grants are per job: `read`, `write`, `read_write`; default deny. A job may only touch declared buffers.
- Buffers are reference-counted by running jobs. Deletion waits for the last grant to release.
- Session close deletes non-persistent buffers after a grace period. Persistent buffers survive jobs, not runtime restarts in 0.1.
- A reaper removes expired buffers and orphaned shm files. It is tested by killing a session mid-job.

### 4.5 Large data and streaming

- `write_chunk` / `read_chunk` with `offset` and `length` for data larger than memory.
- `import_stream` accepts an iterator of chunks and computes the checksum incrementally.
- The context-bloat demo (1 GB CI, 5 GB paper) uses a generated synthetic dataset and chunked reads; it does not allocate the full dataset in Python at once.
- The vector engine may memory-map an input and process tiles; the first implementation may materialize one tile at a time.
- “Zero-copy” is claimed only when the worker maps the same shm file. The trace records `copy_count` and `mapped_bytes` so the claim is auditable.

---

## 5. LMPU Kernel

### 5.1 Components

| Component | Responsibility |
|---|---|
| `Runtime` | Facade, lifecycle, session registry, admin API |
| `JobTable` | Job records, state machine, idempotency, terminal events |
| `Router` | Capability matching, explicit routing, load-aware selection |
| `Scheduler` | Priority queues, per-processor concurrency, batching, fairness |
| `EventLog` | Per-session sequence, ring buffer, ack cursor, replay |
| `Clock` | `now`, `sleep`, `call_later`; fake clock for tests |
| `QuotaManager` | Per-session and global limits; admission control |
| `Reaper` | Orphan sessions, dead workers, expired buffers, timed-out jobs |
| `TraceSink` | JSONL trace, optional OpenTelemetry |
| `WorkerPool` | Warm workers, health, restart, crash quarantine |

### 5.2 Scheduling

- Priority is an integer 0–9; higher wins. Aging prevents starvation: a job gains effective priority every 100 ms in queue.
- FIFO within equal effective priority; round-robin across sessions at the same priority.
- Per-processor concurrency limits; global semaphore.
- Batching coalescer: same `(processor, op, config-compatible)` jobs within a 2 ms window, up to the processor’s declared max batch. Individual jobs still get individual results.
- Admission control before queueing: outstanding jobs per session (default 32), buffer bytes per session (default 2 GiB), sleep registrations per session (1), GPU KV budget (when applicable).
- Reject early with `quota_exceeded`; do not accept a job that cannot be scheduled within its deadline.
- 0.1 quota dimensions: outstanding jobs, buffer bytes, CPU seconds, GPU seconds, wall time, and sleeping registrations. Token and FLOP accounting are metrics only; token/FLOP quotas are deferred.

### 5.3 Executors

| Executor | Use |
|---|---|
| `Inline` | Tiny pure functions; runs in a thread when >1 ms to protect the event loop |
| `ThreadPool` | NumPy/BLAS work that releases the GIL; per-processor bounded pool |
| `Process` | Python VM, isolation, crash containment, rlimits |
| `Remote` | Deferred; gRPC/HTTP processor endpoint |
| `GPU` | Deferred; CuPy/CUDA IPC; the vector engine may opt in |

An executor must never block the kernel event loop for more than 5 ms. This is a reviewed invariant and a performance test.

### 5.4 Timeouts, cancellation, and crashes

- Timeout starts at submission, not at worker start; deadline and queue time count.
- On timeout: cooperative cancel signal, then `SIGTERM`, then `SIGKILL` after a 500 ms grace. Terminal state is `FAILED/timeout`.
- Cancellation API returns immediately with `cancel_requested`; the terminal event follows. Tests cover both orders.
- A worker crash becomes `FAILED/worker_crashed`. The pool restarts the worker. Three crashes from the same job signature quarantine the processor and emit `ResourceWarning`.
- A job that exceeds memory or CPU limits receives `FAILED/oom` or `FAILED/quota_exceeded`; the sandbox enforces limits, not the job itself.
- Orphan job policy: session close cancels jobs by default; `detach: true` keeps them with a TTL and no event delivery.

### 5.5 Idempotency and caching

- `submit` with a `job_id` that already exists returns the existing job if the request hash matches; otherwise `idempotency_conflict`.
- The request hash covers processor, op, inputs, outputs, config, and program hash.
- Optional result cache keyed by `(processor, version, op, input checksums, config hash)` for deterministic ops.
- Cache hits emit `JobCompleted` with `cache_hit: true`; they do not re-run the worker.
- Side-effectful processors declare `idempotent: false`; the runtime never auto-retries them.

### 5.6 Recovery

Phase 1–5: in-memory state; a kernel crash loses jobs and buffers. This is acceptable for the PoC and documented.

Phase 6: append-only SQLite journal for job records and events; on restart, non-terminal jobs become `FAILED/runtime_restarted`; buffers are reconciled from the shm manifest. Exactly-once execution is not promised; exactly-once *result visibility* is promised for retained events.

---

## 6. Processors and the SDK

### 6.1 Processor contract

```python
class Processor(ABC):
    manifest: Manifest

    async def setup(self, ctx: ProcessorContext) -> None: ...
    async def execute(self, job: Job, ctx: JobContext) -> Result: ...
    async def cancel(self, job_id: str) -> None: ...
    async def shutdown(self) -> None: ...
```

`JobContext` provides:

```text
read_buffer(name) / write_buffer(name)
progress(fraction, message=None)
request_input(fields, prompt)
submit_child(processor, op, inputs, config)
log / trace
cancelled
```

### 6.2 Manifest

```json
{
  "name": "vector.engine",
  "version": "0.1.0",
  "protocol": "lmpu/1",
  "ops": [
    {"name": "matmul", "inputs": ["A", "B"], "outputs": ["C"]}
  ],
  "limits": {"max_buffer_bytes": 1073741824, "max_batch": 32},
  "determinism": ["strict", "best_effort"],
  "latency_class": "ms",
  "streaming": true,
  "progress": true,
  "interactive": false,
  "persistent_state": false,
  "batching": true,
  "zero_copy": true,
  "sandbox": ["subprocess"]
}
```

Capabilities are claims, and the contract suite tests them. A processor that declares `progress: true` but never emits progress fails conformance.

Registration and routing:

- Built-in processors register at kernel startup; external processors register through an entry point or an explicit manifest file.
- An explicit `processor` bypasses routing but is validated against the capability manifest.
- Auto-routing filters by `op`, input/output constraints, memory limit, determinism requirement, and sandbox profile.
- Surviving processors are scored by explicit preference, current load, latency class, and locality. Ties break by name for reproducibility.
- A route decision is recorded in the trace with the candidate set and the chosen reason.

### 6.3 Worker protocol

- Parent launches a worker: `python -m lmpu.worker --plugin <module> --socket <path>`.
- Handshake: worker sends `ready` with manifest and pid; parent validates.
- Commands: `execute`, `cancel`, `shutdown`; events: `accepted`, `progress`, `needs_input`, `result`, `error`, `heartbeat`.
- Length-prefixed JSONL over UDS; buffer fds passed with `SCM_RIGHTS`.
- Heartbeat every 1 s; three missed heartbeats mark the worker unhealthy.
- Worker stdout/stderr are captured, bounded, and attached to the trace on failure. Never injected into model context by default.
- `lmpu.worker` is small and dependency-light; plugins import their own heavy libraries in the worker.

### 6.4 SDK ergonomics

The three-line promise:

```python
from lmpu import processor

@processor(op="double", inputs={"x": "int"}, outputs={"y": "int"})
def double(x): return {"y": 2 * x}
```

`lmpu plugin new` generates a manifest, a worker entry point, and a contract-test skeleton. `lmpu plugin test` runs the conformance suite. This is an adoption feature; it must be excellent.

### 6.5 Built-in processors

#### `pure.func`

- Ops: `bigint_add`, `bigint_mul`, `mod_pow`, `sha256`, `blake3`, `json_query`, `lookup`, `batch`.
- Stateless, deterministic, exact. Runs in the inline/thread executor; heavy bit operations may go to a process.
- Batch mode accepts `items` and returns ordered results plus per-item errors.
- Used for demos 1, 2, 9, 10 and for kernel microbenchmarks.

#### `python.vm`

- Ops: `run_program`, `run_file`.
- Program string or path; helper module `lmpu` with `read_buffer`, `write_buffer`, `progress`, `request_input`, `finish`, `log`.
- Runs with `python -I` in a fresh temp cwd; minimal environment; no inherited secrets. Do not use `-S` unless the worker bootstraps an explicit, audited `sys.path` for the helper and NumPy.
- Default limits: wall timeout, CPU seconds, address space, file size, open files, no network namespace (`unshare -n` when available).
- `NeedsMoreInput` parks the job in `WAITING_INPUT`; `write_input` resumes it.
- Persistent state via an explicit `state` buffer; not automatic in 0.1.
- Sandbox is best-effort by default; the README and `lmpu doctor` say so.

#### `vector.engine`

- Ops: `matmul`, `dot`, `fft`, `conv1d`, `softmax`, `layer_norm`.
- NumPy/SciPy; BLAS thread count configurable. `determinism: strict` sets BLAS threads to 1 and fixes seeds where relevant. Strict means reproducible on the same processor version, dtype, shape, host, BLAS build, and thread count; cross-machine bit-exactness is not promised. `best_effort` may use multithreaded BLAS.
- Validates dtype, shape, contiguity; returns a checksum and a compact summary (mean/std/min/max for floats).
- Reads and writes shm buffers directly; records `copy_count`.
- GPU path via CuPy is optional and off by default; the capability manifest says whether it is active.
- Used for the flagship sleep/wake and context-bloat demos.

#### `symbolic.planner`

- Ops: `shortest_path`, `bfs_plan`, `constraint_solve`, `schedule`.
- Deterministic tie-breaking; graph input as JSON or a buffer; progress for large graphs.
- `NeedsMoreInput` when `goal` or a required field is missing.
- Used for interactive demo 7.

#### `neural.summary`

- Ops: `summarize_text_buffer`, `embed_text_buffer`.
- Backend A: extractive TextRank on CPU (default; deterministic, test-friendly).
- Backend B: small local model (Qwen3.5-0.8B or a sentence transformer) when configured.
- Input and output are buffers; the model context receives `chars`, `preview`, and a buffer ref, not the full summary.
- GPU memory budget is checked against the host; in 0.1 the default is CPU to avoid contention.

#### `hardware.stub`

- Simulated device latency and deterministic output; demonstrates latency classes, async completion, and capability descriptors.
- Later replaced by a CUDA kernel, FPGA endpoint, or DSP worker without changing the interface.
- Used for capability and latency demos; optional fourth job in concurrency demo 9.

### 6.6 Composition

A processor may submit child jobs through `ctx.submit_child`. Limits: depth ≤4, total child jobs ≤64 per root job, inherited deadline and budget, cycle detection by `(processor, op, input hash)`. Child events are attributed to the parent in the trace. Composition is tested with a small “orchestrator” processor in Phase 4.

---

## 7. Host integration

### 7.1 `HostEngine` abstraction

```python
class HostEngine(ABC):
    async def prefill(self, prompt: str) -> HostState: ...
    async def decode(self, state: HostState, max_tokens: int, stop: list[str]) -> str: ...
    async def park(self, state: HostState, level: SleepLevel) -> None: ...
    async def resume(self, state: HostState) -> None: ...
    def stats(self) -> HostStats: ...
    async def close(self) -> None: ...
```

`HostState` contains token ids, attention mask/cache, generated-token accounting, forward-pass count, and parked status. The abstraction is intentionally narrow: generate, park, resume, measure. Anything wider leaks engine internals into the controller.

Portability scope: the **interface** is model-agnostic; the 0.1 **implementations** are transformer-only (HF and vLLM). SSM/RNN/multimodal claims are limited to “the interface can express them” until a second adapter exists.

### 7.2 Hugging Face reference engine

- Use the class selected by Transformers 5.x for `qwen3_5` (`AutoModelForImageTextToText` or `Qwen3_5ForConditionalGeneration`) and `AutoTokenizer`; run the text-only decode path. `AutoModelForCausalLM` is not sufficient for Qwen3.5. Use `torch.inference_mode()`.
- Manual decode loop, not `generate()`: one forward per token; `DynamicCache` (or the model’s cache object) retained.
- Stop conditions: sentinel string, max tokens, EOS, context budget.
- Forward-pass counter increments once per model call; tests assert the exact count.
- S1 park: stop calling the model; cache stays in VRAM. No `empty_cache()`.
- S2 park: move cache tensors to pinned CPU memory; free GPU blocks; `resume` moves them back. Verify that cache length and logits match the pre-park state.
- Token accounting: prompt/prefill tokens, generated text tokens, control tokens, injected event tokens, total context tokens.
- Default model: `Qwen/Qwen3.5-0.8B` for CI and fast tests; `Qwen/Qwen3.5-2B` recommended for recorded demos on the 10 GB reference GPU. `Qwen/Qwen3.5-4B` is optional and may need 4-bit quantization; `Qwen/Qwen3.5-9B` does not fit comfortably. Qwen3.5 is multimodal and supports a 262k context, but LMPU uses the text path only. It requires Transformers 5.x (`qwen3_5` is rejected by 4.57.6) and a recent vLLM. Thinking mode disabled. No code depends on Qwen specifics.

### 7.3 vLLM parked-request adapter

This is a separate milestone, not a drop-in `sleep()` call.

- vLLM’s existing `sleep()` is a global memory-management primitive: it offloads weights to free VRAM. It does **not** pause one request while preserving its KV blocks.
- LMPU needs a request-level `PAUSED` state: remove the request from the active batch, keep its KV blocks pinned (S1) or swap them to CPU (S2), and re-admit it on wake without re-prefill.
- **Preferred integration: a version-pinned Python monkey-patch adapter, not a fork.** Patch only pure-Python scheduler/engine surfaces:
  1. `EngineCore.step` or the scheduler loop: skip `PAUSED` requests.
  2. `Scheduler.schedule`: exclude parked requests from active batches.
  3. Request state/status: add `PAUSED` and its transitions.
  4. KV block manager: pin blocks for S1 or swap for S2; never free on park.
  5. Async output path: hold the request output queue across park and resume.
- Monkey-patching is possible because these are Python classes, but they are **private APIs**. Patches are version-sensitive and can break across vLLM releases. Therefore:
  - ship a separate `lmpu-vllm-adapter` package with a compatibility matrix;
  - `lmpu doctor` checks the vLLM version and each patch point; fail closed with a clear error;
  - run the contract suite against every supported vLLM version;
  - never patch CUDA/C++ kernels; if a required hook is in C++, upstream it or fork.
- Fallback order: (1) monkey-patch, (2) S2 swap using public engine APIs, (3) SGLang pause/resume, (4) documented limitation.
- **No fork by default.** Fork only if the monkey-patch cannot retain KV blocks and the Phase 6 spike proves a fork is necessary.
- Upstream a minimal design note or PR after the adapter works.
- The adapter must report `parked_requests`, `pinned_kv_bytes`, `swapped_kv_bytes`, `monkey_patch_version`, and `wake_latency`.

### 7.4 Scripted engine

- Reads a deterministic script: a list of free-text segments and control blocks.
- Used for kernel integration tests, parser tests, demos without a GPU, and CI.
- Not used for the official efficiency benchmark; it cannot prove model behavior.
- The scripted engine is also the fuzzing target for the session controller.

### 7.5 OpenAI-compatible degraded adapter

- Useful as a baseline and for users without local GPUs.
- No KV freeze: each turn re-prefills. It must be labeled `degraded: no_kv_freeze`.
- Never use it for the headline sleep claim.

### 7.6 Session controller

States: `IDLE`, `GENERATING`, `PARSING`, `DISPATCHING`, `PARKED`, `WAKING`, `CLOSED`.

Loop, simplified:

```python
while not closed:
    if parked:
        wake_event = await runtime.wait_for_wake(session)
        await engine.resume(state)
        inject(wake_event)
        parked = False
        continue

    segment = await engine.decode(state, stop=SENTINEL_OPEN)
    control = parser.parse(segment)

    if control is None:
        continue

    result = await runtime.dispatch(session, control)

    if control.kind == "sleep":
        wake_event = await runtime.register_sleep(session, control)
        if wake_event is not None:          # condition already satisfied
            inject(wake_event)
            continue
        await engine.park(state, level=resolve_level(control, config))
        parked = True
        continue

    for event in runtime.take_injectable_events(session):
        inject(event)
```

`resolve_level` clamps the requested sleep level to `session.sleep_level` and the configured maximum; the model cannot pin unlimited KV. The real loop must handle: event arrives while generating, `NeedsMoreInput` while parked, forced wake, cancel, parser failure, context budget exhaustion, engine error, and session close. Each is a test case.

Sleep authority is configurable. The model does not have to own the decision:

| Mode | Behavior | Use |
|---|---|---|
| `model` | The model’s `sleep` message is honored; the controller validates syntax and deadline only | Research and demos |
| `controller` | The controller parks based on job latency and policy; the model cannot sleep arbitrarily | Production-leaning |
| `model_with_policy` (default) | The model proposes; the controller enforces `max_sleep_seconds`, `wake_on_failure`, and the deadline, and may force wake | Default |

Injection policy is configurable per event and per session. Safe defaults:

| Event class | While generating | While parked |
|---|---|---|
| `JobFailed`, `NeedsMoreInput` | `interrupt` at next token boundary | `wake` |
| `JobCompleted`, `JobCancelled` | `next_turn` | `wake` |
| `Progress` | `observe` | `observe` |
| `Woken` | `inject` immediately | `inject` after resume |

Config keys: `session.injection.generating.<Event>`, `session.injection.parked.<Event>`, with values `interrupt | next_turn | observe | wake | inject`. Invalid combinations fail at startup, not at runtime.

Context budget:

- `max_injected_bytes_per_event` (default 64 KiB)
- `max_injected_tokens_per_turn` (default 2,048)
- `max_context_tokens` (model-dependent; default 32k)
- If an event exceeds the budget, inject `ResultTooLarge` with a buffer ref and a suggested slice. Never truncate silently mid-JSON.

### 7.7 Prompts

- System prompt: role, available processors (compact capability summary), protocol rules, sleep guidance, read-slice guidance.
- Rules: never fabricate results; never compute large arithmetic or matrices yourself; submit, then sleep for long jobs; read only the slice you need; prefer `async` for jobs over 100 ms.
- Few-shot examples: exact arithmetic, matmul + sleep, read slice, interactive input.
- Prompt version is recorded in every trace. Prompts live in `lmpu/host/prompts/` with a version string.
- The capability summary is generated from the registry, not hand-maintained. If a processor is absent, the prompt does not mention it.

### 7.8 Training path (deferred)

Phase 8, not required for 0.1:

- Generate synthetic traces from the scripted engine and successful model runs.
- SFT on host tokens, including control blocks; no loss during simulated sleep intervals.
- Optional preference optimization to reduce unnecessary chain-of-thought and polling.
- Evaluate on held-out tasks with the same benchmark harness; report control-block validity, job success, token count, and correctness.

---

## 8. Sleep and wake

### 8.1 Sleep levels

| Level | Host generation | KV cache | GPU memory | Wake latency | Use |
|---|---|---|---|---|---|
| S1 | stopped | VRAM, pinned | retained | lowest | default single-session demo |
| S2 | stopped | offloaded to pinned CPU RAM | freed | medium | memory pressure, multi-session |
| S3 | stopped | checkpointed to disk or another worker | freed | highest | migration, long jobs, deferred |

The headline claim must name the level. “Sleep” with S1 still holds KV memory; GPU idle power is not zero. Report energy against an idle baseline, not against zero.

### 8.2 Wake conditions

```json
{
  "version": 1,
  "kind": "sleep",
  "until": {
    "all": ["job_vec_001"],
    "any": [],
    "events": ["JobFailed", "NeedsMoreInput"],
    "deadline_ms": 30000
  },
  "level": "S1"
}
```

Semantics:

- `all`: all listed jobs have reached a terminal state.
- `any`: any listed job has reached a terminal state.
- `events`: additional event types that wake the host.
- `deadline_ms`: on expiry, emit `Woken` with `reason: "timeout"`; this is not an error.
- A failed job satisfies `all`; include per-job statuses in `Woken`. Optionally set `success_only: true` to wake only on successful completion of all.
- Only one active sleep registration per session. A second `sleep` returns the current registration; `wake` cancels it.
- `wake` is also called on session close and on engine error.
- `level` is optional. The controller resolves it against `session.sleep_level` and the configured maximum; the model cannot demand S3 or pin unlimited KV.
- Empty `all` and `any` with no `events` means “wait for external wake or deadline.” A `sleep` with no conditions and no deadline is rejected.
- `wake` is idempotent: waking an already-awake session is a no-op, not an error.
- A session-global sleep has no job ids and waits on `events` or `deadline_ms`. A job-scoped sleep lists jobs in `all`/`any`; `wake` targets the active registration. To wake only for one job, use `any: [job_id]`.

### 8.3 Race handling

The sleep path is the most likely place for lost events. The implementation must be atomic:

1. Lock the session.
2. Check whether the condition is already satisfied by terminal jobs or retained events.
3. If satisfied, return `Woken` immediately with `already_satisfied: true`.
4. Otherwise register the waiter and park.
5. On the event path, append the event and resolve waiters in the same critical section.

Tests: completion before sleep, duplicate completion, cancel during sleep, timeout during sleep, disconnect during sleep, two jobs completing in one tick, event eviction while parked.

### 8.4 Metrics

- `sleep_duration_seconds`
- `host_forward_passes_during_sleep` (must be 0)
- `host_generated_tokens_during_sleep` (must be 0)
- `host_gpu_energy_during_sleep_joules`
- `wake_latency_seconds` (event append → first generated token)
- `kv_bytes_resident`
- `kv_bytes_offloaded`
- `sleeping_sessions` and `pinned_kv_bytes_total`

The dashboard’s “sleep proof” panel displays these counters live. The benchmark records the raw samples.

---

## 9. Observability

### 9.1 Trace

Canonical JSONL. One line per event; no multiline records.

```json
{"ts":1735920000.001,"mono":12.3,"trace":"t_01","session":"s_1","type":"host_forward","tokens":1,"phase":"prefill"}
{"ts":1735920000.015,"mono":12.4,"trace":"t_01","session":"s_1","type":"control_parsed","kind":"submit","job_id":"job_vec_001"}
{"ts":1735920000.018,"mono":12.5,"trace":"t_01","session":"s_1","type":"job_state","job_id":"job_vec_001","from":"PENDING","to":"RUNNING"}
{"ts":1735920000.020,"mono":12.6,"trace":"t_01","session":"s_1","type":"park_enter","level":"S1","jobs":["job_vec_001"]}
{"ts":1735920002.113,"mono":14.7,"trace":"t_01","session":"s_1","type":"job_progress","job_id":"job_vec_001","progress":0.52}
{"ts":1735920004.001,"mono":16.6,"trace":"t_01","session":"s_1","type":"job_state","job_id":"job_vec_001","from":"RUNNING","to":"COMPLETED"}
{"ts":1735920004.002,"mono":16.7,"trace":"t_01","session":"s_1","type":"wake","reason":"job_done"}
{"ts":1735920004.003,"mono":16.7,"trace":"t_01","session":"s_1","type":"inject","event":"JobCompleted","job_id":"job_vec_001"}
```

Rules:

- Timestamps both wall-clock and monotonic; energy and latency use monotonic.
- Trace IDs propagate from session to job to worker.
- Large payloads are referenced by buffer ID, never inlined.
- Redact environment variables and secrets; worker stderr is bounded and marked `untrusted`.
- Rotation by size and age; `lmpu trace show` renders a timeline; `lmpu trace export` produces a portable archive.
- Token-level `host_forward` records are sampled and aggregated by default (`trace.token_events: sampled`, 1-in-N or per-segment). Full per-token tracing is a debug flag, because a long generation otherwise dominates the trace.
- Trace event taxonomy: `session_*`, `host_*`, `control_*`, `job_*`, `worker_*`, `buffer_*`, `sleep_*`, `wake_*`, `inject_*`, `quota_*`, `error_*`. The dashboard and replay tool consume only this taxonomy.
- Buffer access is auditable: `buffer_create`, `buffer_import`, `buffer_read`, `buffer_write`, `grant_granted`, and `grant_denied` carry buffer id, job id, and byte range.

### 9.2 Metrics

Prometheus-compatible names:

```text
lmpu_jobs_total{processor,status}
lmpu_job_duration_seconds{processor,op}
lmpu_queue_depth{processor}
lmpu_host_forwards_total{session,phase}
lmpu_host_tokens_total{session,kind}
lmpu_sleep_seconds{session,level}
lmpu_wake_latency_seconds{session}
lmpu_buffer_bytes{session,kind}
lmpu_pinned_kv_bytes{session}
lmpu_worker_restarts_total{processor}
lmpu_quota_rejections_total{session,reason}
```

OpenTelemetry export is optional and off by default; the JSONL trace is the source of truth.

### 9.3 Mission Control dashboard

The dashboard is the **flagship demo**, not a diagnostic afterthought. It must show the whole system at once and let a viewer launch different trusted demo tasks.

FastAPI + Server-Sent Events, bound to `127.0.0.1` by default. No authentication in 0.1; document the risk and refuse to bind to `0.0.0.0` unless `--allow-public`.

Panels:

- **Host:** model phase (`GENERATING`, `PARKED`, `WAKING`), sleep level, forward-pass counter, token counters, context bytes;
- **Sleep proof:** a live timeline showing zero `host_forward` events between `park_enter` and `wake`, plus GPU power and wake latency;
- **Jobs:** a task board with status, progress, duration, processor, priority, and cancel-request state;
- **Buffers:** dtype, shape, bytes, checksum, grants, and a slice preview;
- **Events:** the sequenced event stream with injection/observe markers;
- **Energy:** GPU power/energy, CPU energy when available, and idle baseline;
- **Scenarios:** a trusted launcher for matmul, long Python job, hash batch, symbolic search, interactive input, cancellation, and buffer-slice demos;
- **Trace:** download and timeline replay.

The dashboard is read-only for the model and for viewers. The scenario launcher runs in a trusted CLI/service process and uses the same public protocol as any host; the dashboard never executes arbitrary code. The model cannot trigger a scenario or mutate the runtime through the dashboard.

A separate `--admin` mode may expose cancel and close actions. It is off by default, binds to localhost, requires a bearer token, and is not part of the public demo. If the dashboard binds to a non-loopback address, authentication is mandatory regardless of mode.

`demos/12_mission_control.py` starts the runtime, the host session, the dashboard, and a multi-task scenario. `demos/scenarios/*.yaml` defines each task so the demo is reproducible. The killer demo is one screen with multiple processors active, one host asleep, buffers filling, progress moving, and the forward-pass counter frozen.

---

## 10. Testing strategy

### 10.1 Principles

The suite must stay fast, deterministic, and change-tolerant. It tests contracts and observable behavior, not private methods, log lines, or exact model prose.

1. **No wall-clock sleeps in tests.** Use `FakeClock`, explicit event barriers, and `asyncio.Event`.
2. **No exact model text.** Assert parsed control kinds, job outcomes, and correctness. Temperature 0 and constrained decoding reduce variance; they do not justify exact-string assertions.
3. **No tight timing assertions.** Performance is measured in the benchmark lane with generous budgets, not in unit tests.
4. **Real components over mocks.** Use the real kernel, real shm buffers, and real workers. Fake only the host and clock.
5. **Contract tests over implementation tests.** Every processor runs the same conformance suite.
6. **Seed everything.** Random buffers, job IDs, and graph fixtures use fixed seeds or factories.
7. **One behavior per test.** Names state the behavior and the expected outcome.
8. **Golden files only at boundaries.** Wire schema and trace schema may be snapshotted; full traces and model outputs may not.
9. **Never retry to green.** A flaky test is quarantined with an issue. Retries hide races and prompt fragility.
10. **Coverage is a guide.** Target 85% branch coverage for `kernel/` and `memory/`, 70% overall. UI and benchmark code are exempt.

### 10.2 Layers

| Layer | Scope | Marker | Runtime | Gate |
|---|---|---|---|---|
| Unit | Pure functions, protocol codec, state machine, buffer views | `unit` | <5 s | every commit |
| Property | Protocol round-trip, buffer slices, state transitions | `property` | <30 s | every PR |
| Contract | Every processor manifest, progress, cancel, timeout, limits | `contract` | <60 s | every PR |
| Integration | Kernel + real workers + scripted host, no LLM | `integration` | <2 min | every PR |
| Scenario (model-free) | Full session loop with scripted control blocks | `scenario` | <2 min | merge gate |
| E2E model | HF + GPU + constrained decoding + demos | `gpu`, `model` | 5–15 min | release gate and nightly |
| Failure injection | Kill worker, OOM, timeout, disconnect, duplicate submit, event-before-sleep | `failure` | <5 min | nightly and release |
| Security | Sandbox profiles, path escape, fd hygiene, network denial | `security` | <5 min | nightly; release for enabled profiles |
| Fuzz | Parser, framing, buffer slice bounds | `fuzz` | 5 min | nightly |
| Performance | Microbenchmarks and regression budgets | `perf` | 10 min | nightly; non-gating unless budget exceeded twice |
| Soak | 1-hour mixed workload, memory and fd leaks | `soak` | 1 h | weekly and release |

Default command: `pytest -m "unit or property or contract or integration or scenario"`. Nightly adds `failure`, `security`, `fuzz`, `perf`, and `soak`.

### 10.3 What each layer proves

- **Unit/property** proves the kernel's logic and memory safety boundaries.
- **Contract** proves processors tell the truth about their capabilities.
- **Integration** proves the kernel, buffers, workers, and scheduler compose without a model.
- **Scenario** proves the session loop, parser, injection, and sleep/wake races with a deterministic host.
- **E2E model** proves the model can actually emit usable control blocks and that forward-pass accounting is real.
- **Failure injection** proves the system fails closed and recovers.
- **Benchmarks** prove the efficiency claims; they are evidence, not correctness tests.

### 10.4 Fixtures

```text
FakeClock              deterministic time and timers
EventProbe             await a specific event with a timeout
TempShm                isolated /dev/shm directory per test
LocalRuntime           kernel + buffer manager on a UDS temp path
ScriptedHost           deterministic control-block scripts
BufferFactory          seeded typed buffers with known checksums
ProcessorHarness       launch a plugin worker in-process or subprocess
TinyModel              Qwen3.5-0.8B, lazily loaded, GPU-marked
```

### 10.5 Required failure cases

| Case | Expected |
|---|---|
| Malformed JSON in control block | `ControlError` event; job not created; session continues |
| Unknown processor/op | structured error; no job record |
| Buffer missing or no grant | `access_denied` / `buffer_not_found`; no partial write |
| Timeout | `FAILED/timeout`; worker killed; no later completion |
| OOM | `FAILED/oom`; other jobs unaffected |
| Worker crash | `FAILED/worker_crashed`; pool restarts; quarantine after repeat |
| Cancel running job | exactly one `JobCancelled`; no `JobCompleted` |
| Cancel completed job | `already_terminal`; state unchanged |
| Event before sleep | immediate `Woken(already_satisfied: true)` |
| Duplicate submit | same job returned; no duplicate execution |
| Duplicate event delivery | host dedupes by `event_id`; one injection |
| Session disconnect | jobs cancelled or detached per policy; no orphan shm |
| Kernel restart (Phase 6) | non-terminal jobs failed; event log replays |
| Context oversized result | `ResultTooLarge`; no silent truncation |
| Invalid `write_input` state | structured error; job state unchanged |
| Wake with no active sleep | idempotent no-op |
| Sleep with no conditions or deadline | rejected before parking |
| Batch item failure | per-item error; other items complete |
| Route has no capable processor | `capability_mismatch`; no job record |
| Unsupported vLLM version | adapter fails closed with a clear error |

Concurrency and isolation tests:

- many jobs from one session complete without state cross-talk;
- cancelling one job does not corrupt another job’s buffers or events;
- events from two sessions route only to their owner;
- batch results preserve input order, and per-item failures do not reorder successes;
- two jobs completing in one scheduler tick still wake a parked host exactly once;
- a worker crash during a batch leaves the other jobs intact.

### 10.6 Model E2E policy

- Use constrained decoding. The model's only job is to choose the right control kind and arguments.
- Temperature 0, fixed seed, fixed prompt version.
- Assert: at least one submit; correct processor/op; job completes; result is correct; control-block validity; context bytes under budget; forward passes during sleep equal zero.
- Run one canonical scenario per demo in the release gate; run the broader matrix nightly.
- If a scenario is flaky, fix the grammar, prompt, or state machine. Do not add retries to the pass criteria. Mark `xfail` only with a linked issue and an expiry date.
- Record the model, revision, quantization, dtype, GPU, driver, and library versions in the result artifact.

### 10.7 Security tests

Group tests by sandbox profile so a CPU-only contributor is not forced to install Docker or nsjail.

- `test_no_network_default`
- `test_filesystem_escape_blocked` (profile-dependent)
- `test_symlink_escape_blocked`
- `test_rlimit_cpu_enforced`
- `test_rlimit_memory_enforced`
- `test_fd_hygiene` (worker sees only granted fds)
- `test_path_traversal_rejected`
- `test_oversized_message_rejected`
- `test_buffer_bounds_checked`
- `test_secret_env_not_inherited`
- `test_public_mode_rejects_subprocess_profile`
- `test_unsupported_vllm_version_fails_closed`
- `test_dashboard_public_bind_requires_token`

The default subprocess profile is explicitly **not** a security boundary. Tests assert best-effort limits; the threat model states the gap.

### 10.8 CI

| Lane | Hardware | Commands | Trigger |
|---|---|---|---|
| `core` | CPU | `make check`, `pytest -m "unit or property or contract or integration or scenario"` | every push/PR |
| `gpu` | 1× GPU, 10 GB+ (reference RTX 3080); 16 GB recommended | `make test-gpu`, demo smoke | nightly and release |
| `bench` | fixed GPU runner | `make bench-smoke` | nightly |
| `release` | GPU runner | full demos, failure/security/fuzz suites, 1 h soak | tag/release |

GPU jobs are required for release, not for every PR. This honors the “GPU + small HF model required” decision for the official demo path while keeping day-to-day iteration fast.

---

## 11. Benchmarks and evidence

### 11.1 Claims to test

Register these before running the final benchmark; do not tune after seeing results.

**Primary launch claim:** C2 and C3. GPU energy is secondary; C1, C4, and C5 support the story.

- **C1 — Token reduction.** For long jobs (≥1 s) and large results (≥1 MiB), LMPU uses at least 10× fewer host-generated tokens than an async polling agent, and at least 5× fewer than a synchronous PAL-style tool call, at equal task correctness.
- **C2 — Sleep proof.** During S1 park, host forward passes and generated tokens are exactly zero. Report GPU power and energy against an idle baseline. S1 is “near idle,” not “zero energy”; measure S2 separately if it is available.
- **C3 — Context protection.** A 16 MiB result enters the host context as ≤64 tokens of metadata; context growth is O(1) in result size. A requested 4×4 slice is ≤256 tokens.
- **C4 — Batching.** 1,000 tiny pure-function jobs submitted as one batch use one control round-trip and at least 5× the throughput of 1,000 individual synchronous submits.
- **C5 — Wall time.** For exact arithmetic and matrix multiplication, LMPU wall time is within a small constant of native execution plus control overhead, and orders of magnitude below trace-token emulation.

### 11.2 Baselines

All baselines use the same model, prompt budget, hardware, and decoding settings unless an ablation says otherwise.

| Baseline | Description |
|---|---|
| Direct CoT | The model computes in natural language, no tools |
| Trace-token emulation | An interpreter-driven trace generator emits low-level operations as tokens; labeled simulation. Also run a small real-model trace where feasible. |
| Sync tool call | PAL-style: model emits a program, runtime executes, full or summarized result returns, model resumes |
| Async polling | Submit, then repeated status calls with small JSON results |
| LMPU sync | Submit with `mode: sync`, receive a summary; no park |
| LMPU sleep | Submit async, park, wake on completion (the treatment) |

The strong baseline is the async polling agent with a *summary-only* result, not a strawman that dumps 16 MB into context. Report where LMPU does **not** win: tiny lookups and single-turn low-latency calls may be equivalent or slower.

### 11.3 Workloads

| ID | Workload | Primary metric |
|---|---|---|
| B1 | `bigint_mul` with 30-digit operands | correctness, tokens, wall |
| B2 | `sha256` over 100 MB | throughput, tokens |
| B3 | `matmul` 2048×2048 float32 | wall, context bytes, energy |
| B4 | `fft` 2^20 | wall, correctness |
| B5 | `shortest_path` on 10k-node graph | wall, tokens |
| B6 | Python job sleeping 10 s | tokens, host energy, wake latency |
| B7 | Summarize 50k-character document | context bytes, tokens |
| B8 | Scan 1 GB (CI) / 5 GB (paper) synthetic dataset for a pattern | peak context, correctness, wall |
| B9 | 1,000 small adds in one batch | throughput, control messages |
| B10 | Mixed concurrent: matmul + Python + hash | makespan, event ordering, correctness |
| B11 | Iterative Python optimization, 100 iterations | progress, wall, CPU, correctness |
| B12 | Interactive planner with `NeedsMoreInput` | wake latency, tokens, correctness |

### 11.4 Metrics

```text
wall_time_seconds
host_forward_passes
host_generated_tokens
host_prefill_tokens
host_control_tokens
host_injected_tokens
host_total_tokens
peak_context_tokens
context_bytes_injected
result_bytes
gpu_energy_joules
cpu_energy_joules          # optional; GPU-only in 0.1
gpu_seconds
processor_wall_seconds
processor_cpu_seconds
peak_rss_bytes
peak_gpu_memory_bytes
latency_p50_ms
latency_p95_ms
correctness
wake_latency_seconds
sleep_duration_seconds
```

Derived metrics:

- **Token Tax (TT)** = host tokens spent on control, polling, waiting, and result ingestion per completed task.
- **Compute-to-Decision Ratio (CDR).** Define precisely in `docs/benchmarks.md`. Recommended: host FLOPs (estimated as `2 × params × tokens`, or measured where possible) divided by semantic decisions (submits + wakes + final answer). Report the formula and the estimate's assumptions.
- **Context amplification** = injected context bytes / result bytes.
- **Sleep duty cycle** = sleep duration / job wall time.
- **Token reduction** = 1 − (LMPU tokens / baseline tokens).

### 11.5 Energy methodology

- 0.1 reports GPU joules only. CPU RAPL is optional and requires a udev rule or privileged helper.
- Sample GPU power with NVML at 10 Hz; `power.draw.average` is available on the reference RTX 3080 and is accurate to ±5 W. Integrate power over time; do not treat a single sample as a joule measurement.
- CPU package energy via RAPL requires a udev rule or privileged helper on the reference host (`energy_uj` is root-only). If that is not set up, CPU joules are not measured and the benchmark says so.
- Measure a 60 s idle baseline immediately before each trial; subtract it.
- Same power cap, same clocks where possible; record driver, CUDA, and GPU model.
- Run n≥20 trials, discard 3 warmups, randomize baseline/treatment order, report median and bootstrap 95% CI.
- Report joules and joules per decision; do not report a single cherry-picked run.
- State the limitation: S1 retains KV memory and the GPU may still draw idle power. The claim is “near idle,” not “zero energy.”

### 11.6 Statistics and fairness

- Pre-register claims, workloads, and stopping rules in `docs/benchmarks.md` before final runs.
- Report failures, timeouts, and incorrect answers; do not filter them out.
- Use the same prompt and capability summary for tool-call and LMPU modes; count all host tokens, including schema and error-repair tokens.
- Report constrained-decoding overhead separately, then include it in the total.
- Publish raw JSONL, a hardware manifest, seeds, and the exact commit.
- Include an ablation: sleep vs no-sleep, S1 vs S2, summary vs full result, batching on/off, routing vs fixed processor, and constrained vs unconstrained decoding.

### 11.7 Context-bloat demonstration

- Synthetic 1 GB (CI) / 5 GB (paper) dataset with a known planted pattern; generated and scanned in chunks.
- Baseline A: chunk-and-prompt agent until the context limit; record failure point and peak context.
- Baseline B: retrieval/summarization pipeline, the strongest realistic alternative.
- LMPU: submit a `python.vm` or `vector.engine` scan job, park, wake to a ≤50-token summary, read a bounded slice.
- Report peak context, wall time, correctness, and cost; do not claim a win on a task the baseline was not designed for.
- Record a short screen capture of the dashboard during the run; it is communication, not evidence.

### 11.8 Artifacts

```text
benchmarks/results/<date>/<gpu>/<commit>/
  manifest.json        hardware, versions, seeds, config
  raw.jsonl            one record per trial
  summary.json         medians, CIs, derived metrics
  plots/               token reduction, energy timeline, context growth, latency CDF
  README.md            how to reproduce
```

---

## 12. Reliability, security, and performance budgets

### 12.1 SLOs for 0.1

| Property | Target |
|---|---|
| Control-plane submit→accepted, local | p99 < 10 ms |
| Wake latency after terminal event | p99 < 50 ms (S1) |
| Event loss within retention | zero |
| Orphan jobs after session close | zero after 5 s |
| Buffer integrity | checksum verified on read when enabled |
| Kernel event-loop stall | < 5 ms per operation |
| Crash containment | worker crash does not affect other jobs |
| Memory leak over 1 h soak | < 5% growth after warmup |

### 12.2 Failure matrix

| Failure | Detection | Response |
|---|---|---|
| Worker crash | pid wait / heartbeat | `FAILED/worker_crashed`; restart pool; quarantine after 3 |
| Worker hang | heartbeat + timeout | cancel, `SIGTERM`, `SIGKILL`; `FAILED/timeout` |
| OOM | rlimit/cgroup / exception | `FAILED/oom`; job isolated |
| Kernel crash | process exit | Phase 6 journal recovery; jobs failed on restart |
| Host disconnect | UDS close | cancel or detach per session policy; reaper cleans |
| Disk full | shm create error | `quota_exceeded`; no partial buffer |
| Corrupt buffer | checksum mismatch | `FAILED/checksum_mismatch`; quarantine buffer |
| Duplicate submit | job table | return existing or `idempotency_conflict` |
| Event gap on resume | sequence check | `gap` status; reconcile, do not guess |
| Clock jump | monotonic clock | timers use monotonic; wall clock only for display |

### 12.3 Threat model

Assets: host KV cache, session context, buffer contents, worker filesystem, runtime quotas.

Adversaries: untrusted model output, untrusted plugin code, untrusted job input, a local user on the same machine.

Controls in 0.1:

- strict schema and size validation at every boundary;
- path traversal and symlink checks;
- fd hygiene: workers receive only granted descriptors;
- no network by default (`unshare -n` where available);
- rlimits for CPU, memory, file size, open files, processes;
- temp cwd per job; minimal environment; no inherited secrets;
- per-session buffer grants; no cross-session reads; grants are advisory against malicious workers, so process isolation is the real boundary;
- control-plane UDS mode `0600`, peer credentials checked, no network listener;
- dashboard bound to localhost; public bind requires a bearer token;
- audit trace for every job and buffer access (`buffer_read`, `buffer_write`, `grant_denied`);
- public dashboard bind requires a bearer token; default is loopback-only.

Explicit non-goals for 0.1: hostile multi-tenant isolation, kernel-level sandboxing, side-channel resistance, remote authentication, supply-chain verification of plugins, hardware attestation. The default subprocess sandbox is **not** a security boundary; `lmpu doctor` and the README say so.

### 12.4 Performance budgets

| Path | Budget |
|---|---|
| Control message parse/validate | < 1 ms p99 |
| Buffer metadata creation | < 0.5 ms |
| shm map for 16 MiB buffer | < 2 ms |
| Inline slice read (4×4 float32) | < 1 ms |
| Event append + wake | < 1 ms |
| Context injection rendering | < 2 ms |
| Kernel memory overhead per job | < 8 KiB |
| Worker startup (warm pool) | < 100 ms |

---

## 13. Roadmap

Assumptions: one senior engineer, 12–16 weeks; two engineers can compress to 8–10 weeks by parallelizing host and kernel work. The estimate is optimistic; model integration, the vLLM spike, and benchmark debugging are the main variance. Each phase ends with a gate. Do not start the next phase with a red gate.

**Critical path:** P0 → P1 → P2 → P3 → P5 → P5.5 → P6 → P7. P4 can overlap with P3/P5. P5.5 (benchmark) starts after P3 and runs alongside P4/P6.

Schedule includes roughly 15% contingency. If the vLLM spike fails, Phase 6 drops the adapter, keeps the SGLang fallback, and the schedule does not extend past the 14-week target.

### Phase 0 — Foundations (week 1)

- [x] P0-01 Create `pyproject.toml`, `uv.lock`, `src/lmpu/`, `tests/`, `configs/`, `docs/` (S)
- [~] P0-02 Configure `ruff`, `mypy`, `pytest`, `pre-commit`, coverage, markers (S)
- [ ] P0-03 CI: CPU `core` lane; GPU lane stub with self-hosted runner docs (M)
- [ ] P0-04 Write `docs/rfcs/0001-lmpu-protocol.md` skeleton and JSON Schema location (M)
- [x] P0-05 Implement `lmpu doctor` checks: Python, shm, UDS, GPU, model cache, sandbox (S)
- [ ] P0-06 Record accepted ADRs for D-01/D-03/D-06/D-12; no decision freeze needed (S)
- [~] P0-07 Define trace schema and example trace fixture (S)
- [~] P0-08 Create a Python 3.12/3.13 environment if 3.14 wheels are missing; upgrade Transformers 5.x, vLLM 0.33, Outlines 0.1.14; verify Qwen3.5-0.8B/2B load, tokenizer, VRAM, and tokens/sec; record the baseline (M) — 0.8B verified on Python 3.14/Transformers 5.22.1; 2B, vLLM, Outlines remain

**Exit gate:** `make check` is green; the protocol RFC skeleton is reviewed; naming is frozen; no runtime behavior is claimed yet.

### Phase 1 — Kernel, buffers, pure functions (weeks 2–3)

- [x] P1-01 Protocol models, discriminated unions, JSON-RPC codec, limits (M)
- [x] P1-02 Job state machine table + exhaustive transition tests (M)
- [x] P1-03 JobTable, idempotency, terminal-event guarantee (M)
- [x] P1-04 EventLog: seq, ULID, ring buffer, ack, replay (M)
- [x] P1-05 BufferManager + shm backend + NumPy views + BLAKE3 (L)
- [~] P1-06 Grants, refcounts, TTL, reaper (M)
- [~] P1-07 CapabilityRegistry + routing + `query_capabilities`; advertisement is separate from session authorization (M)
- [x] P1-08 `pure.func` processor and contract tests (M)
- [x] P1-09 Scripted host + minimal session loop (M)
- [~] P1-10 Demos 1 and 2, headless assertion mode (S)
- [x] P1-11 Property tests: buffer slices, protocol round-trip (M)
- [x] P1-12 Runtime RPC handlers: hello, capabilities, submit, poll, wait, read, write, cancel, ack, configure, close; health reports liveness, readiness, and backend availability (M)
- [x] P1-13 TraceSink implementation and trace/replay schema tests; redact sensitive payloads by default (M)
- [~] P1-14 Buffer import from bytes/files plus provenance metadata (M)

**Exit gate:** `pytest -m "unit or property or integration"` green; all RPC methods are covered; trace replay works; buffer round-trip, checksum, import, and provenance verified; demo 1/2 pass CPU-only; a 16 MiB buffer can be created and read as a 4×4 slice without copying the whole payload into the process that reads it.

### Phase 2 — Isolation, long jobs, batching (weeks 3–5)

- [x] P2-01 Worker protocol, parent client, `SCM_RIGHTS` fd passing (L)
- [~] P2-02 Worker pool, health, restart, crash quarantine (M)
- [~] P2-03 Subprocess sandbox: rlimits, temp cwd, env scrubbing, no-network wrapper (M)
- [~] P2-04 `python.vm` processor + `lmpu` helper module (L)
- [~] P2-05 `vector.engine` processor + summaries; portable copy fallback; zero-copy only with `copy_count` evidence (L) — matmul/fallback/evidence done; remaining ops placeholders
- [x] P2-06 Timeouts, cooperative cancel, hard kill, terminal race tests (M)
- [~] P2-07 Quotas, admission control, bounded queues, and backpressure (M)
- [x] P2-08 Batching coalescer + batch API (M)
- [~] P2-09 Progress events, throttling, observe-only injection, bounded event injection (S)
- [~] P2-10 Demos 3, 4 (scripted), 6, 9, 10 (L)
- [x] P2-11 Failure-injection suite: crash, OOM, timeout, disconnect, duplicate, deterministic fake-worker cases (M)
- [ ] P2-12 Sandbox profiles: nsjail and Docker templates, public-mode guard, profile tests (M)

**Exit gate:** cancellation and timeout tests pass under race injection; a 16 MiB matmul completes with `copy_count == 0` on the data path; 1,000 batched adds complete in one control round-trip; nsjail/Docker profile tests pass; no orphan workers or shm after tests.

### Phase 3 — HF host, constrained decoding, sleep/wake (weeks 5–7)

- [x] P3-01 `HostEngine` ABC + `HostState` + metrics (M)
- [~] P3-02 HF engine: manual decode loop, cache, forward counter (L) — 0.8B baseline and session adapter done; cross-turn KV reuse and 2B remain
- [x] P3-03 Sentinel scanner with token-boundary handling (M)
- [~] P3-04 Two-phase constrained decoder spike: Outlines vs lm-format-enforcer (L)
- [~] P3-05 Parser, schema validation, `ControlError`, minimal repair policy (M)
- [~] P3-06 SessionController: generate/parse/dispatch/park/wake (L)
- [x] P3-07 Injection renderer + context budget + `ResultTooLarge` (M)
- [~] P3-08 S1 park/resume with zero-forward-pass assertion (M)
- [ ] P3-09 S2 offload/restore spike and cache-equality test (M)
- [ ] P3-10 Demos 4 and 5 with `Qwen/Qwen3.5-0.8B`; record a 2B run for the demo (L)
- [~] P3-11 Trace and metrics for host forwards/tokens/energy (M)
- [ ] P3-12 E2E GPU tests marked `gpu`/`model` (M)
- [ ] P3-13 Sentinel experiment: `<|p|>` vs `<|proc|>` with and without special-token registration; measure token cost and free-generation validity (M)
- [~] P3-14 Prompt templates with generated capability summary; version and regression tests (M)

**Exit gate:** flagship demo runs on GPU; trace shows zero `host_forward` between `park_enter` and `wake`; injected result metadata ≤64 tokens for a 16 MiB matrix; wake latency p99 <50 ms; prompt templates and capability summary are versioned; one canonical model scenario passes deterministically.

### Phase 4 — Processor ecosystem and interactivity (weeks 7–8)

- [~] P4-01 `symbolic.planner` with Dijkstra/BFS and deterministic tie-break (M)
- [ ] P4-02 `NeedsMoreInput`/`write_input` job state and resume path (M)
- [~] P4-03 `neural.summary` with CPU extractive backend; optional model backend (M)
- [~] P4-04 `hardware.stub` with simulated latency and capability flags (S)
- [ ] P4-05 Composition: child jobs, depth/budget/cycle limits (M)
- [ ] P4-06 Capability routing heuristics + explicit override (S)
- [ ] P4-07 Demos 7 and 8 (M)
- [ ] P4-08 Contract suite expanded to all processors; required for any advertised accelerator backend (M)

**Exit gate:** six processor types work through one interface; interactive input resumes correctly; composition depth limits are enforced; all processors pass the same contract suite.

### Phase 5 — Observability, dashboard, full demo suite (week 9)

- [ ] P5-01 Trace viewer and `lmpu trace` CLI (M)
- [ ] P5-02 FastAPI + SSE dashboard, read-only, localhost default (M)
- [ ] P5-03 Sleep-proof panel and event timeline (S)
- [ ] P5-04 `scripts/run_all_demos.sh` with assertion mode and artifacts (S)
- [ ] P5-05 Demo 11 skeleton and result capture (M)
- [ ] P5-06 Recorded walkthrough script and screenshots (S)
- [ ] P5-07 Mission Control scenario runner: multi-task YAML, trusted launcher, live task board (L)
- [ ] P5-08 Dashboard task filters, buffer preview, event injection markers, energy panel (M)
- [ ] P5-09 `demos/12_mission_control.py` and recorded multi-task walkthrough (M)

**Exit gate:** all demos run from a clean checkout with documented GPU prerequisites; Mission Control shows multiple concurrent processors, a parked host with frozen forward counters, live buffers, progress, and energy; `make demo` produces a trace artifact.

### Phase 5.5 — Benchmark and context-bloat evidence (weeks 9–10)

- [ ] P5.5-01 Benchmark harness: config, seeds, trials, artifacts (L)
- [ ] P5.5-02 Baseline adapters: direct CoT, trace-token, sync tool, async polling (L)
- [ ] P5.5-03 Energy sampling: NVML, optional RAPL, idle baseline, calibration (M)
- [ ] P5.5-04 Workloads B1–B12 and correctness checks (L)
- [ ] P5.5-05 Statistics and plots: medians, CIs, CDFs, timelines (M)
- [ ] P5.5-06 Context-bloat demo: 1 GB CI scan and 5 GB paper run (M)
- [ ] P5.5-07 Pre-registration document; final run; raw artifact (M)
- [ ] P5.5-08 Ablations: sleep on/off, S1/S2, summary/full, batching (M)

**Exit gate:** reproducible `make bench` on a fixed runner; report includes C1–C5 with raw data; strong baselines are included; honesty section lists where LMPU does not win.

### Phase 6 — vLLM adapter, durability, hardening (weeks 10–11)

- [ ] P6-01 vLLM feasibility spike: request parking, KV retention, and monkey-patch hook points (L)
- [ ] P6-02 Monkey-patch adapter and upstream design note; fork only if the gate proves it necessary (L)
- [ ] P6-03 Admission control for pinned/swapped KV; sleeping-session caps (M)
- [ ] P6-04 S2 swap path on vLLM, if supported (M)
- [ ] P6-05 SQLite journal for jobs/events; restart reconciliation (L)
- [ ] P6-06 Reconnect/resume/gap handling end-to-end (M)
- [ ] P6-07 1-hour soak: memory, fds, event retention, worker churn (M)
- [ ] P6-08 Security pass: path, fd, env, network, message-size tests (M)

**Exit gate:** vLLM demo parks a request without re-prefill and wakes within budget, or the plan records a documented fallback; soak passes; failure matrix green; no orphaned resources.

### Phase 7 — RFC, SDK, docs, release (week 12)

- [ ] P7-01 Finalize RFC 0001 and JSON Schema; publish examples (M)
- [ ] P7-02 Plugin SDK polish: `plugin new`, `plugin test`, 3-line example (M)
- [ ] P7-03 Architecture, threat model, and benchmark docs (M)
- [ ] P7-04 `README` quickstart that matches the code (S)
- [ ] P7-05 Package `lmpu==0.1.0` under Apache-2.0 (S)
- [ ] P7-06 Paper draft: Token Tax, CDR, sleep proof, context protection (L)
- [ ] P7-07 Manifesto/engineering post with trace comparison (M)
- [ ] P7-08 External-contributor test: a new processor in <10 minutes (S)
- [ ] P7-09 Release checklist: license, SBOM, dependency pins, signed artifact, security notes (S)
- [ ] P7-10 Publish two example processors (`jq`, `sqlite`) and their contract tests (S)

**Exit gate:** a new contributor can install, run a demo on a GPU, and write a processor using the docs; RFC and benchmark artifacts are public; release checklist, SBOM, license, and signed artifact are complete; release tag is reproducible.

### Phase 8 — Deferred research (post-0.1)

- [ ] P8-01 S3 checkpoint/migration
- [ ] P8-02 CUDA IPC / GPU buffers
- [ ] P8-03 WASM processor backend
- [ ] P8-04 Remote processors over gRPC/WebSocket
- [ ] P8-05 Multi-tenant scheduling and quotas
- [ ] P8-06 Fine-tuned control policy and synthetic trace training
- [ ] P8-07 Formal execution traces/proofs
- [ ] P8-08 MCP bridge and LangChain/LlamaIndex executor adapters
- [ ] P8-09 Second runtime adapter (llama.cpp, MLX, or Mamba) to validate the portability claim
- [ ] P8-10 Speculative jobs with cancel-on-stale results
- [ ] P8-11 Multimodal buffer adapter for image/audio tensors
- [ ] P8-12 Real accelerator backend (CUDA kernel or FPGA endpoint)
- [ ] P8-13 Redundant execution and independent result verification (operation-specific)
- [ ] P8-14 Persistent processor state and cross-job result cache

---

## 14. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model will not emit valid control blocks | Medium | High | Constrained decoding; scripted fallback for tests; fine-tune only if needed |
| vLLM parking patch is invasive or rejected upstream | High | High | HF reference is authoritative; version-pinned monkey-patch; SGLang fallback; keep the engine ABC narrow |
| Monkey-patch breaks on vLLM upgrade | High | Medium | Pin supported versions; `lmpu doctor` fail-closed; contract tests per version; upstream design note |
| “Sleep” does not save energy because KV stays resident | Medium | High | Define S1/S2/S3; measure against idle baseline; report honestly; offer S2 |
| Zero-copy claim is false on some path | Medium | Medium | Record `copy_count`; test fd mapping; state the fallback path |
| Cancellation/terminal-event races | High | High | Atomic state transition; race-injection tests; single terminal event invariant |
| Lost wake events | Medium | High | Atomic wait registration; seq/ack/replay; event-before-sleep tests |
| Benchmark criticized as strawman | High | High | Strong baselines; pre-registration; raw artifacts; state non-wins |
| Security of `python.vm` overclaimed | High | High | Pluggable sandbox; “not a boundary” warning; threat model; profile-gated tests |
| GPU CI cost and availability | Medium | Medium | CPU core lane; nightly GPU; self-hosted/release-only |
| Tokenizer lacks sentinel tokens | Medium | Medium | String sentinels + scanner; optional special-token addition later |
| Context bloat from verbose events | Medium | Medium | Observe-only default for progress; byte/token budgets; `ResultTooLarge` |
| Scope creep into a full OS | High | High | Phase gates; explicit non-goals; defer multi-tenant/remote |
| Energy measurement noise | High | Medium | Idle baseline, n≥20, CIs, fixed clocks, report methodology |
| Plugin ecosystem fragmentation | Medium | Medium | Contract suite; manifest schema; version negotiation; 3-line SDK |
| Newest model outruns tooling | High | Medium | Pin a known-good model for CI; upgrade Transformers/vLLM in P0-08; keep a text-only fallback |

---

## 15. Acceptance traceability

### 15.1 Functional criteria

| Criterion | Phase | Verification |
|---|---|---|
| F1 Submit job through abstract interface | P1 | `test_submit_sync`, demo 2 |
| F2 Route to appropriate processor | P1/P4 | capability/routing tests |
| F3 Execute outside host generation loop | P2/P3 | demo 3 trace; no host forwards during job |
| F4 Poll status | P1 | `test_poll_running` |
| F5 Wait with timeout | P1 | `test_wait_timeout`, demo 3 |
| F6 Sleep until job completion | P3 | demo 4; zero-forward-pass assertion |
| F7 Wake and inject event | P3 | `test_wake_injection`, demo 4 |
| F8 Read small result only | P1/P3 | demo 5; context byte budget |
| F9 Cancel running job | P2 | demo 6; terminal-event race tests |
| F10 Report failures | P2 | timeout/OOM/crash tests |
| F11 Three or more processor types | P4 | six processors, one contract suite |
| F12 Large buffers outside context | P1/P3/P5.5 | 16 MiB matmul; 1 GB/5 GB scan |
| F13 Capability discovery | P1 | demo 1; registry tests |
| F14 Progress events | P2 | demo 6; monotonic progress contract |
| F15 Interactive input | P4 | demo 7; `NeedsMoreInput` resume |
| F16 Configure session/resource limits | P1/P2 | admin API and quota tests |
| F17 Import buffer from file/bytes with provenance | P1 | import and provenance tests |
| F18 Batch per-item success and failure | P2 | batch contract tests |
| F19 Mission Control live activity | P5 | demo 12 and trace assertions |

### 15.2 Efficiency criteria

| Criterion | Phase | Verification |
|---|---|---|
| E1 Zero forwards during sleep | P3 | trace counter + benchmark C2 |
| E2 Zero generated tokens during sleep | P3 | trace counter + benchmark C2 |
| E3 Large results as small summaries | P3/P5.5 | C3; injected bytes/tokens |
| E4 Fewer tokens than direct computation | P5.5 | C1; direct CoT baseline |
| E5 Fewer tokens than trace-token | P5.5 | C1; trace-token baseline |
| E6 Lower wall time for exact/matrix tasks | P5.5 | C5; native vs trace-token |

### 15.3 Correctness criteria

| Criterion | Phase | Verification |
|---|---|---|
| C1 Deterministic processors exact | P1/P2 | bigint/hash vectors; repeat checksum |
| C2 Buffer checksums match | P1 | round-trip and corruption tests |
| C3 Valid state transitions | P1 | exhaustive transition table tests |
| C4 Cancelled jobs never complete | P2 | race-injection tests |
| C5 Timeouts produce failures | P2 | failure suite |
| C6 Events reach the correct session | P1/P6 | multi-session isolation tests |

### 15.4 Specification primitive coverage

| README primitive | Plan surface | Phase |
|---|---|---|
| `Submit(job_spec)` | `lmpu.submit` / model `submit` | P1 |
| `Poll(JobHandle)` | `lmpu.poll` / model `poll` | P1 |
| `Wait(JobHandle, timeout)` | `lmpu.wait` / model `wait` | P1 |
| `ReadResult(...)` | `lmpu.read` / model `read_result` | P1 |
| `WriteInput(...)` | `lmpu.write` / model `write_input` | P2/P4 |
| `Cancel(JobHandle)` | `lmpu.cancel` / model `cancel` | P2 |
| `Sleep(until_event)` | `lmpu.sleep` / model `sleep` | P3 |
| `Wake(JobHandle)` | `lmpu.wake` / model `wake` | P3 |
| `Configure(...)` | `lmpu.configure` (admin) | P1 |
| Capability query | `lmpu.capabilities` / model `query_capabilities` | P1 |
| Status `sleeping` | `WAITING_INPUT` / `WAITING_EVENT` mapping | P1/P2 |
| Events | `lmpu.event` notifications | P1/P3 |

### 15.5 Demo inventory

| Demo | Subject | Gate |
|---|---|---|
| 01 | Capability query and routing | P1 |
| 02 | Synchronous exact arithmetic | P1 |
| 03 | Asynchronous Python job, poll/wait | P2 |
| 04 | Sleep/wake matrix multiplication | P3 |
| 05 | Large-buffer partial read | P3 |
| 06 | Progress, polling, cancellation | P2 |
| 07 | Interactive `NeedsMoreInput` | P4 |
| 08 | Neural summarization | P4 |
| 09 | Concurrent processors | P2/P4 |
| 10 | Batched small jobs | P2 |
| 11 | Benchmark vs token compute | P5.5 |
| 12 | Mission Control dashboard | P5 |

---

## 16. Publication and adoption

The PoC is not enough. The plan must produce artifacts that let others verify, reproduce, and extend the claim.

### 16.1 Protocol and SDK

- RFC 0001: normative protocol, state machine, error codes, delivery semantics, versioning.
- JSON Schema checked into the repo; generated docs; conformance test suite.
- Python SDK and three-line processor decorator.
- Example processors: `ffmpeg`, `sqlite`, `duckdb`, `pinecone`-style retrieval, `jq`, `sympy`. These prove the abstraction is not just for matrices.
- MCP bridge: expose LMPU processors as MCP tools, and MCP tools as LMPU processors. Interop, not replacement.

### 16.2 Evidence package

- Reproducible benchmark harness with raw JSONL, hardware manifest, seeds, plots.
- Trace archive for the flagship demo; a script that replays the trace through the dashboard.
- Video/GIF of the sleep-proof panel and the context-bloat comparison.
- Honest limitations: single-node, single-tenant, best-effort sandbox, S1 retains KV.

### 16.3 Paper framing

- Working title: *Sleeping LLMs: Interrupt-Driven Execution for Decoupled Agentic Workflows*.
- Contributions: the protocol, per-request parking, zero-copy result references, measured Token Tax and CDR, and a reproducible benchmark.
- Related work: PAL/tool calling, ReAct/polling agents, Qwenputer/in-weight computation, vLLM sleep, DeepMind differentiable cache, MCP.
- Claims must be scoped to measured workloads; no universal “orders of magnitude” without data.

### 16.4 Adoption path

1. Open-source Apache-2.0 with a working GPU demo.
2. Upstream the vLLM parking patch or publish a maintained adapter.
3. Ship LangChain/LlamaIndex executor adapters that use LMPU as a backend.
4. Publish the RFC and manifesto; invite processor authors.
5. Approach inference providers with the parking primitive as an API feature.
6. Track adoption: time-to-first-processor, integrations, citations, benchmark reproductions.

---

## 17. Design decisions (accepted)

All design questions are closed for 0.1. A change requires an ADR; a protocol change also requires a version bump.

| Topic | Decision |
|---|---|
| Primary claim | Zero host forwards plus O(1) context; GPU energy is secondary |
| Tenancy | Single-tenant 0.1; admission hooks only |
| Naming | LMPU; `lmpu/1`; `<\|proc\|>` string; P3-13 tests `<\|p\|>` as a special token |
| Sleep level | S1 default; S2 under memory pressure |
| Sleep authority | Configurable: `model`, `controller`, `model_with_policy`; default `model_with_policy` |
| vLLM | Version-pinned monkey-patch adapter; no fork by default; SGLang fallback |
| Sandbox | Configurable; safe profile required for public mode |
| Event injection | Configurable per event and session; safe defaults |
| Buffer provenance | `buffer.import` from files and bytes; URLs and databases later |
| Portability | Interface portable; 0.1 implementation transformer-only |
| Decoding evidence | Report constrained and unconstrained validity; defer SFT |
| Context-bloat scale | 1 GB CI; 5 GB paper |
| Energy | GPU joules in 0.1; CPU RAPL optional |
| Model | Qwen3.5-0.8B CI; Qwen3.5-2B demos; 4B quantized optional |
| MCP | Separate layer with a bridge |
| License | Apache-2.0 |
| Killer demo | Mission Control dashboard with multiple concurrent tasks |

Change control: record any reversal in `docs/adr/`, update this table, and update the affected phase gate. No silent drift.

---

## 18. Definition of done for 0.1

- [ ] Phase 0–7 gates are green.
- [~] `pytest -m "unit or property or contract or integration or scenario"` passes on CPU in <5 minutes. (All current suites pass: 119 tests, 4 skipped in 68 s; two opt-in GPU tests skip unless `LMPU_RUN_GPU=1`.)
- [ ] `make test-gpu` passes on the reference GPU runner.
- [ ] Demos 1–12 run from a clean checkout and produce traces.
- [ ] Mission Control dashboard shows multiple concurrent processors, live buffers, progress, energy, and a parked host with frozen forward counters.
- [ ] Flagship demo shows zero host forwards and zero generated tokens during S1 park.
- [ ] A 16 MiB result is represented by ≤64 tokens of context metadata.
- [ ] Six processor types pass the contract suite.
- [ ] Failure matrix is covered by automated tests.
- [ ] 1-hour soak has no leak or orphan.
- [ ] RFC, JSON Schema, SDK, and threat model are published.
- [ ] Benchmark report contains C1–C5, raw data, strong baselines, and limitations.
- [ ] A new contributor can write and run a processor using only the docs.
- [ ] Release tag `v0.1.0` is reproducible.

---

## 19. Plan review record (2026-11-03)

**Scope.** Reviewed against `README.md`, `TODO.md`, and `compared.md`, plus internal consistency of protocol, buffers, kernel, processors, host integration, sleep/wake, observability, testing, benchmarks, security, roadmap, and appendices.

**Fixed in this review.**

- Completed the job state machine with `WAITING_EVENT` and `CANCEL_REQUESTED`; aligned cancellation semantics across protocol and kernel.
- Added explicit RPC response shapes, job-spec fields, batch schema, and `write_input` semantics.
- Aligned event-injection defaults between the event table, session controller, and config.
- Added the missing `lmpu.configure` surface and specification-primitive coverage table.
- Replaced the vLLM “patch or adapter” ambiguity with the accepted monkey-patch strategy and a fail-closed compatibility gate.
- Fixed the Qwen3.5 model-class assumption, Transformers 5.x requirement, and GPU-lane memory mismatch.
- Removed the read-only dashboard contradiction; cancellation is admin-only and public binds require a token.
- Added failure cases for invalid input, wake/no-sleep, empty sleep, batch item errors, unroutable jobs, and unsupported vLLM versions.
- Added dependency policy, demo/test runbook, demo inventory, release checklist, and second-adapter deferral.
- Tightened energy/context-bloat claims and added token/memory/CPU metrics.

**Residual risks.**

1. **vLLM parking may not be achievable by monkey-patch alone.** P6 is a go/no-go spike; the HF reference remains authoritative.
2. **Qwen3.5 tooling is new.** Transformers 4.57.6 rejects `qwen3_5`; P0-08 must prove the 5.x path, VRAM, and tokens/sec.
3. **The 10 GB GPU limits demo model size.** 2B is the practical demo default; 4B needs quantization.
4. **Energy is secondary.** S1 still retains KV and draws idle power; do not market it as zero energy.
5. **Benchmark credibility depends on strong baselines.** The trace-token and polling baselines must be implemented fairly, not as strawmen.
6. **The schedule is optimistic.** 12–16 weeks assumes focused work and a successful vLLM spike.

**Review outcome.** The plan is complete enough to start Phase 0. It is not evidence of production readiness. Do not start Phase 1 until P0-08 proves the toolchain and model path.

---

## 20. Revision reconciliation (2026-11-03)

A separate revision proposal was received after this plan was accepted. It is a delta,
not a replacement. This section reconciles it with the existing plan and with the code
that now exists. No existing phase number, decision ID, or gate was renumbered. Where the
proposal repeats existing work, the existing task is the source of truth; only genuinely
missing requirements were added.

### 20.1 Proposal priority mapping

| Proposal band | Existing plan home | Reconciliation |
|---|---|---|
| P0 protocol contract, lifecycle, errors, idempotency, timeout/cancel | §3, §5, P1-01…P1-03, P2-06 | Already covered; added acceptance detail in §20.4 |
| P0 CPU reference runtime, control/execution split, queues, shutdown, ownership | §5, §6, P1-08, P1-12, P2-02, P2-07 | Dispatch path and terminal ownership are now implemented; bounded queues and restart remain |
| P0 safe defaults, capability authorization, boundary validation | §6, §12.3, P2-03, P2-12 | Existing non-goal and threat model retained; added explicit acceptance checks |
| P1 buffer semantics, capability-scoped access, portable fallback, streaming, persistence | §4, P1-05, P1-06, P1-14 | Semantics exist; portable fallback and snapshot semantics need a decision/ADR |
| P1 sleep/wake as an integration layer, notification semantics, KV as optimization | §7, §8, P3-06…P3-09 | Existing decision retained; added the explicit “runtime works without sleep” rule |
| P1 scheduling, capability discovery, cost-aware dispatch | §5.2, §6.2, P1-07, P4-06 | Discovery and deterministic routing exist; cost model stays deferred |
| P1 observability, operational controls | §9, P1-13, P3-11 | Trace/metrics skeleton exists; health/readiness and sensitive-data redaction need tests |
| P1 conformance, fake worker, stress, fault injection, security tests, CI example | §10, P1-11, P2-11 | Integration tests added; property/contract/failure/security suites remain |
| P2 accelerators, zero-copy, checkpointing, verification, distributed execution | §8, P6, P8, D-23 | Confirmed deferred; no active-scope expansion |

### 20.2 Missing requirements accepted into active scope

These are additions to existing tasks, not new parallel tasks:

1. **First usable milestone is explicit** (§20.4). It precedes P3 and must stay green.
2. **Resource ownership is named per terminal path** (§20.5). P2-02, P2-06, and P2-07 use it as a checklist.
3. **Bounded queues and backpressure are gate items**, not later tuning. Add to P2-07 and P2-09.
4. **Graceful shutdown and restart behavior must be documented and tested** at the level the plan actually promises. P1–P5 remain in-memory; P6-05 owns durability.
5. **Capability authorization is separate from capability advertisement.** Discovery may say a processor supports an op; submission still validates session, grants, and quotas. Add to P1-07 and P2-07.
6. **Portable copy path is mandatory; zero-copy is optional.** Add to P2-05 and P5.5 evidence.
7. **Notification recovery by status query is mandatory.** A client must be able to recover from a missed event with `poll`/`wait`; add to P1-04 and P3-06.
8. **Sensitive payloads are not logged by default.** Add to P1-13 and P6-08.
9. **CI must contain a model-free end-to-end example.** Satisfied by `tests/integration/test_runtime_execution.py`; keep it model-free.
10. **The plan must not promise exactly-once execution or cross-device determinism.** Already in §5.6 and §6.5; restated in §20.6 guardrails.
11. **Health and readiness are distinct.** `lmpu.health` must report process liveness, admission readiness, and per-backend availability separately. Add to P1-12 and P5-02.
12. **Limits are configurable and inspectable.** Defaults for queues, payloads, timeouts, workers, retention, and quotas must be readable at runtime. Add to P1-12 (`lmpu.configure`, `lmpu.health`) and Appendix B.
13. **Deterministic fake-worker tests are required**, not only real workers: delayed results, duplicate delivery, malformed output, crash, and disconnect. Add to P2-11 and §10.4.
14. **Cost-aware dispatch is deferred, not dropped.** 0.1 keeps deterministic rules. P4-06 gains a cost model only if P5.5 workload data shows a benefit.
15. **Accelerator backends must pass the same contract suite** before they are advertised. Add to P4-08 and P8-12.
16. **Zero-copy is gated by measurement**, not claimed by default. P2-05 must record `copy_count` and retain the portable copy path.
17. **Redundant execution and independent result verification are deferred** to P8 as an optional, operation-specific capability.
18. **Persistent processor state and result caching are deferred**; 0.1 keeps the result cache in §5.5 only. Add to P8.

### 20.3 Additional decisions recorded

The proposal’s open questions are resolved here or explicitly deferred. A changed answer
requires an ADR.

| Question | 0.1 answer | Source |
|---|---|---|
| Protocol boundary | Local JSON-RPC 2.0 over a Unix domain socket. The model plane is separate. Network transports are later adapters. | D-03, §3.1 |
| What `wait` blocks | An async caller/runtime task. Host generation parking is a different operation. | §7.6 |
| What `sleep` means | Host generation suspension plus an optional job/event wake condition. Job `WAITING_INPUT`/`WAITING_EVENT` are distinct job states. | §3.4, §8.2 |
| Durability | In-memory through P5; non-terminal jobs fail with `runtime_restarted` after P6-05. No exactly-once execution. | §5.6 |
| Buffer consistency | Per-job grants; inputs are declared read-only references in 0.1. Producer mutation during a job is undefined and must not be relied on. | §4.4, needs ADR for snapshots |
| Result representation | Small values inline under `max_inline_bytes`; large values as `BufferRef` plus a bounded slice. | §4.3 |
| Client identity | Session ID plus local peer credentials. Single tenant; no remote authentication. | D-21, §12.3 |
| Portability | Linux-first implementation; the interface is portable, implementations are transformer-only. | D-23 |
| Capability versioning | `protocol`, `processor_version`, and `schema_hash` travel with capabilities; major mismatch is rejected. | §3.8 |
| Determinism | `strict` means reproducible on the same processor version, dtype, shape, host, BLAS build, and thread count. Not cross-machine bit-exactness. | §6.5 |

### 20.4 Acceptance criteria for the first usable milestone

The milestone is “CPU reference runtime works end to end.” It must be true before P3 GPU
work becomes the main track.

| Criterion | Status | Verification |
|---|---|---|
| Typed operation reaches a CPU backend and returns a stable job handle | Done | `tests/integration/test_runtime_execution.py::test_sync_submit_returns_result` |
| Client can await a result without busy-looping | Done | `JobTable.wait_terminal`; `test_async_wait_reports_one_terminal_event` |
| Duplicate submission has deterministic behavior | Done | `test_duplicate_submit_does_not_duplicate_execution` |
| Cancellation produces exactly one terminal event | Done | `test_cancel_running_job_emits_only_cancelled` |
| Timeout produces `FAILED/timeout` and stops the worker | Done | manual timeout run; promote to `failure` test |
| Explicit batch preserves order and reports per-item errors | Done | `test_explicit_batch_preserves_item_order` |
| Worker loss produces `FAILED/worker_crashed` and does not affect other jobs | Partial | crash path exists; no automated test yet |
| Queue depth, payload size, execution time, and resources are bounded | Partial | quotas and wall timeout exist; queue/backpressure test missing |
| Unauthorized operation or buffer access is rejected | Partial | grants exist; negative tests missing |
| Worker output and all input boundaries are validated | Partial | schema and buffer bounds exist; fuzz tests missing |
| Resources are released on every terminal path | Partial | grants and quota counter released; buffer refcount audit missing |
| Runtime restart behavior matches the documented promise | Partial | in-memory behavior documented; restart test is P6-05 |
| A model-free example runs in CI | Done | integration lane |
| A second independent client/backend can implement the protocol | Not yet | requires RFC + conformance fixtures |

### 20.5 Resource ownership matrix

| Resource | Owner | Released on success | Released on failure/cancel/timeout | Released on disconnect |
|---|---|---|---|---|
| Job record | `JobTable` | TTL expiry | TTL expiry | session close policy |
| Worker slot | `WorkerPool` | on result | on error or forced kill | reaped on session close |
| Grants | `GrantManager` | terminal callback | terminal callback | session close |
| Buffer bytes | `BufferManager` / session quota | TTL or explicit delete | TTL or explicit delete | session close, non-persistent only |
| Sleep registration | `SessionState` / `QuotaManager` | wake | wake or session close | session close |
| Event retention | `EventLog` ring | ring eviction | ring eviction | session removal |

### 20.6 Scope guardrails

These restate and tighten §0.2. They are not new promises.

- Do not make any LLM vendor, transformer class, or inference server part of the protocol core.
- Do not require KV freezing, sleep support, shared memory, or zero-copy for runtime correctness.
- Do not make arbitrary code execution the default; `python.vm` is an explicit, non-default processor.
- Do not promise exactly-once execution, durable results, deterministic scheduling, or cross-device reproducibility.
- Do not add multi-host scheduling, accelerator-specific handles, or learned scheduling before the single-host CPU path is correct and observable.
- Do not expand the active scope with P2 items until the §20.4 criteria are green.

### 20.7 Dependency-aware order

1. Reconcile plan and code. **Done in this revision.**
2. Close §20.4: worker-loss test, bounded queue/backpressure test, negative authorization tests, resource-release audit.
3. Close P1 exit gate: contract suite, property tests, trace/replay tests **(done)**; demo 1/2 assertion mode remains.
4. Complete the scripted session loop: injection, context budget, `ResultTooLarge`, parser failure, forced wake. **Done.**
5. Add `SCM_RIGHTS` and the portable copy fallback; then measure `copy_count` for `vector.engine`. **Done for matmul.**
6. Add the failure-injection suite. **Done: real and fake-worker cases.**
7. Verify P0-08 on the reference GPU. **0.8B done; 2B/vLLM/Outlines remain.**
8. Wire `HFEngine` into the session loop. **Done via `HFDecoder`; cross-turn KV reuse remains.**
6. Verify P0-08 on the reference GPU and start P3.
7. Keep P5.5 benchmarks behind a working P3 sleep/wake proof.

### 20.8 Proposal item disposition

Every bullet in the revision proposal is listed here exactly once. `Covered` means the
accepted plan already handled it; `Added` means this reconciliation added it to an
existing task or gate; `Deferred` means it is explicitly out of 0.1 scope.

| ID | Proposal item | Disposition | Plan location / task |
|---|---|---|---|
| R-01 | Versioned protocol specification with compatibility rules | Covered, open | §3, P0-04, D-01/D-03 |
| R-02 | Job lifecycle and legal transitions | Covered | §3.4, P1-02 |
| R-03 | Submission, retries, idempotency, duplicate behavior | Covered | §5.5, P1-03 |
| R-04 | Timeout, cancellation, cleanup semantics | Covered | §5.4, P2-06 |
| R-05 | Structured errors with retryability and origin | Covered | §3.7 |
| R-06 | Minimal CPU reference backend | Done | §0.3, integration suite |
| R-07 | Separate control plane and execution plane | Covered | §2, §6.3, P2-01/P2-02 |
| R-08 | Bounded queues and backpressure | Added | §20.2(3), P2-07/P2-09 |
| R-09 | Graceful shutdown and recovery | Covered | §5.6, P6-05 |
| R-10 | Resource ownership model | Added | §20.5 |
| R-11 | Registered operations as the default path | Covered | D-13, §3.3 |
| R-12 | Arbitrary code execution isolated and opt-in | Covered | D-09, §6.5 (`python.vm`), §20.6 |
| R-13 | Capability-based authorization, not only advertisement | Added | §20.2(5), P1-07/P2-07 |
| R-14 | Validate all input/output boundaries | Covered | §10.5, §10.7, P2-03/P6-08 |
| R-15 | Buffer semantics before zero-copy | Covered | §4.1–§4.4 |
| R-16 | Capability-scoped buffer access | Covered | §4.4, P1-06 |
| R-17 | Portable copy/serialization fallback | Added | §20.2(6), P2-05 |
| R-18 | Streaming and chunking semantics | Covered | §4.5 |
| R-19 | Persistence separated from volatile memory | Covered | §4.4, §5.6 |
| R-20 | Separate job waiting from host suspension | Covered | §3.4, §8.1, §20.3 |
| R-21 | Notification semantics: order, duplicates, reconnect, query fallback | Covered | §3.6, §20.2(7) |
| R-22 | Host suspension/resumption adapter requirements | Covered | §7.1, §7.3, §7.6 |
| R-23 | KV retention is an optimization, not a protocol guarantee | Added as explicit rule | §8.1, §20.6 |
| R-24 | Prototype an adapter only after the runtime contract is stable | Covered | §20.7 order, P3 after §20.4 |
| R-25 | Capability discovery fields and authorization distinction | Covered | §6.2, P1-07 |
| R-26 | Admission control and fair scheduling | Covered | §5.2, P2-07 |
| R-27 | Simple cost-aware dispatch | Deferred | P4-06, conditional on P5.5 data |
| R-28 | Pluggable policy, deterministic first | Covered | §5.2, §6.2 |
| R-29 | Correlated structured logs and tracing | Covered | §9.1, P1-13 |
| R-30 | Operational metrics | Covered | §9.2, P3-11 |
| R-31 | Health and readiness signals | Added | §20.2(11), P1-12/P5-02 |
| R-32 | Configurable and inspectable limits | Covered, added inspect | Appendix B, §20.2(12) |
| R-33 | Protocol conformance tests for independent clients/backends | Covered, open | §10.2, P1-11, P7-01 |
| R-34 | Deterministic fake-worker tests | Added | §20.2(13), P2-11 |
| R-35 | Concurrency and stress tests | Covered | §10.5 |
| R-36 | Fault injection and recovery tests | Covered | §10.5, P2-11 |
| R-37 | Security tests in CI | Covered | §10.7, P6-08 |
| R-38 | Model-free end-to-end CI example | Done | integration suite, §20.2(9) |
| R-39 | Accelerator backends behind the same conformance suite | Added | §20.2(15), P4-08/P8-12 |
| R-40 | Shared memory / zero-copy only after profiling | Added | §20.2(16), P2-05 |
| R-41 | Checkpointing and resumable jobs | Deferred | P8-01 |
| R-42 | Redundant execution or independent verification | Deferred | P8-13 |
| R-43 | Persistent processor state and caching | Deferred | P8-14 |
| R-44 | Multi-host / distributed execution | Deferred | P8-04, P8-05 |

**Audit result:** all 44 proposal items are dispositioned. No item is silently dropped.
The active-scope additions are limited to §20.2. Items R-27 and R-41…R-44 are deferred
with a named future task, not forgotten.

### 20.9 Ownership of active additions

| Addition | Responsible module | Test location | Done when |
|---|---|---|---|
| First usable milestone gate | `kernel/`, `tests/integration/` | `tests/integration/` | §20.4 green before P3 |
| Resource ownership matrix | `kernel/runtime.py`, `memory/` | `tests/failure/`, leak checks | every terminal path releases resources |
| Bounded queues/backpressure | `kernel/scheduler.py`, `kernel/quotas.py` | `tests/failure/`, `tests/perf/` | overload rejects predictably, memory bounded |
| Shutdown/restart promise | `kernel/runtime.py`, `kernel/jobs.py` | `tests/failure/` | documented behavior matches test |
| Capability authorization | `kernel/routing.py`, `memory/grants.py` | `tests/security/` | unauthorized op/buffer rejected |
| Portable copy fallback | `processors/builtin/vector_engine.py` | `tests/contract/` | works without shm fast path |
| Notification recovery | `kernel/events.py`, `host/session.py` | `tests/scenario/` | `poll`/`wait` recovers a missed event |
| Sensitive-payload redaction | `kernel/trace.py` | `tests/security/` | payloads absent by default |
| Model-free CI example | `tests/integration/` | CI `core` lane | passes without GPU or model |
| Health/readiness distinction | `kernel/runtime.py` | `tests/integration/` | fields reported separately |
| Inspectable limits | `kernel/runtime.py`, `configs/` | `tests/integration/` | defaults readable at runtime |
| Deterministic fake-worker tests | `kernel/worker_pool.py` | `tests/failure/` | delayed/duplicate/malformed/crash covered |
| Cost-aware dispatch (deferred) | `kernel/routing.py` | `tests/perf/` | only after P5.5 data justifies it |
| Accelerator contract gate | `processors/`, `tests/contract/` | `tests/contract/` | backend advertised only after suite |
| Zero-copy measurement gate | `processors/builtin/vector_engine.py` | `tests/contract/` | `copy_count` recorded per result |
| Redundant verification (deferred) | P8 | P8 | operation-specific design exists |
| Persistent state/cache (deferred) | P8 | P8 | isolation, keys, invalidation defined |

---

## Appendix A — Repository layout

```text
lpu/
├── README.md
├── TODO.1.md
├── TODO.md
├── compared.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── default.yaml
│   ├── dev.yaml
│   └── bench.yaml
├── src/lmpu/
│   ├── __init__.py
│   ├── cli.py
│   ├── protocol/
│   │   ├── messages.py
│   │   ├── codec.py
│   │   ├── errors.py
│   │   ├── sentinels.py
│   │   └── schema/
│   ├── kernel/
│   │   ├── runtime.py
│   │   ├── jobs.py
│   │   ├── scheduler.py
│   │   ├── events.py
│   │   ├── quotas.py
│   │   ├── reaper.py
│   │   └── routing.py
│   ├── memory/
│   │   ├── buffers.py
│   │   ├── shm.py
│   │   ├── checksum.py
│   │   └── grants.py
│   ├── processors/
│   │   ├── base.py
│   │   ├── manifest.py
│   │   ├── worker.py
│   │   ├── sandbox/
│   │   └── builtin/
│   ├── host/
│   │   ├── engine.py
│   │   ├── hf.py
│   │   ├── vllm.py
│   │   ├── scripted.py
│   │   ├── parser.py
│   │   ├── grammar.py
│   │   ├── session.py
│   │   ├── metrics.py
│   │   └── prompts/
│   ├── observability/
│   │   ├── trace.py
│   │   ├── metrics.py
│   │   └── dashboard/
│   ├── bench/
│   └── synth/
├── demos/
│   ├── 01_capability_query.py
│   ├── 02_sync_arithmetic.py
│   ├── 03_async_python_job.py
│   ├── 04_sleep_wake_matrix.py
│   ├── 05_large_buffer_read_slice.py
│   ├── 06_progress_and_cancel.py
│   ├── 07_needs_more_input.py
│   ├── 08_neural_summary.py
│   ├── 09_multi_processor_concurrent.py
│   ├── 10_batch_small_jobs.py
│   ├── 11_benchmark_vs_token_compute.py
│   ├── 12_mission_control.py
│   └── scenarios/
├── tests/
│   ├── unit/
│   ├── property/
│   ├── contract/
│   ├── integration/
│   ├── scenario/
│   ├── failure/
│   ├── security/
│   ├── fuzz/
│   ├── perf/
│   └── e2e/
├── benchmarks/
│   ├── run_benchmark.py
│   ├── metrics.py
│   └── plots.py
├── docs/
│   ├── rfcs/
│   ├── adr/
│   ├── architecture.md
│   ├── threat-model.md
│   ├── benchmarks.md
│   ├── dependencies.md
│   └── demo-runbook.md
└── scripts/
    ├── run_dashboard.py
    ├── run_all_demos.sh
    └── run_tests.sh
```

---

## Appendix B — Configuration sketch

```yaml
runtime:
  socket: /tmp/lmpu/run.sock
  shm_dir: /dev/shm/lmpu
  max_message_bytes: 4096
  max_inline_bytes: 65536
  event_ring: 10000
  max_sessions: 1
  trace_token_events: sampled   # sampled | full | off

session:
  max_outstanding_jobs: 32
  max_buffer_bytes: 2147483648
  max_injected_tokens_per_turn: 2048
  max_injected_bytes_per_event: 65536
  max_sleep_registrations: 1
  sleep_level: S1
  sleep_authority: model_with_policy   # model | controller | model_with_policy
  max_sleep_seconds: 300
  wake_on_failure: true
  injection:
    generating:
      JobFailed: interrupt
      NeedsMoreInput: interrupt
      JobCompleted: next_turn
      Progress: observe
    parked:
      JobFailed: wake
      NeedsMoreInput: wake
      JobCompleted: wake
      Progress: observe

buffers:
  checksum: blake3
  verify_on_read: true
  max_inline_bytes: 65536
  default_ttl_ms: 3600000
  import_roots: ["/data", "/tmp/lmpu/imports"]

host:
  engine: hf                 # hf | vllm | scripted | openai
  model: Qwen/Qwen3.5-2B
  model_class: auto          # auto | Qwen3_5ForConditionalGeneration
  trust_remote_code: false
  dtype: bfloat16
  device: cuda
  constrained_decoding: true
  max_control_tokens: 512
  temperature: 0.0
  seed: 0

kernel:
  batch_window_ms: 2
  max_batch: 64
  worker_heartbeat_ms: 1000
  cancel_grace_ms: 500
  reaper_interval_ms: 1000

sandbox:
  profile: subprocess        # subprocess | nsjail | docker | wasm
  public_mode: false
  allow_unsafe: false        # subprocess is not a security boundary
  allow_network: false
  cpu_seconds: 30
  memory_mb: 1024
  file_mb: 256

vllm:
  adapter: monkey_patch      # monkey_patch | sglang | none
  supported_versions: ["0.33.x"]
  fail_closed: true

observability:
  trace: true
  trace_dir: traces/
  prometheus: false
  dashboard: true
  mission_control: true
  dashboard_admin: false
  dashboard_bind: 127.0.0.1
  dashboard_token_env: LMPU_DASHBOARD_TOKEN
```

---

## Appendix C — Glossary

- **Park**: stop host decoding without destroying its state. Model-facing `sleep`.
- **Wake**: resume host decoding after a condition is satisfied.
- **Processor**: an execution backend registered with the LMPU Runtime (formerly co-processor).
- **Buffer**: typed shared-memory region referenced by ID, never by value in context.
- **Grant**: a per-job permission to read or write a buffer.
- **Capability descriptor**: the manifest that states what a processor can do and its limits.
- **Terminal event**: `JobCompleted`, `JobFailed`, or `JobCancelled`; exactly one per job.
- **Event cursor**: per-session acknowledgement position used for replay.
- **Token Tax**: host tokens spent controlling, waiting for, or ingesting a computation.
- **CDR**: Compute-to-Decision Ratio; host compute per semantic decision.
- **S1/S2/S3**: sleep levels with KV resident, offloaded, or checkpointed.

---

## Appendix D — Dependencies and compatibility policy

The plan is only buildable if the dependency surface is explicit. Use `uv` with a lockfile and split heavy or fragile packages into extras.

| Layer | Required | Optional / extras | Notes |
|---|---|---|---|
| Core runtime | Python 3.12/3.13, `pydantic` v2, `orjson`, `numpy`, `blake3` | `ml_dtypes` for bfloat16 | Python 3.14 only if every wheel exists |
| Vector processor | `numpy` | `scipy` for FFT/conv; `mkl`/OpenBLAS | Strict determinism pins BLAS threads |
| Host | `torch`, `transformers>=5.22,<6`, `tokenizers` | `accelerate` for device maps | Qwen3.5 requires a Transformers 5.x build |
| Decoding | One of `outlines` or `lm-format-enforcer` | Both during P3-04 spike | Pick one after the spike; pin it |
| Engine adapter | `vllm==0.33.*` | `lmpu-vllm-adapter` extra | Monkey-patch only; fail closed on version mismatch |
| Observability | `fastapi`, `uvicorn` | `sse-starlette`, `nvidia-ml-py` (`pynvml`) | NVML is required for GPU joules |
| Testing | `pytest`, `pytest-asyncio`, `hypothesis`, `pytest-benchmark` | `mutmut` for mutation tests | Deterministic clock and fixtures are in-repo |
| Tooling | `ruff`, `mypy`, `pre-commit`, `uv` | `pyright` optional | CI uses the lockfile |
| Sandbox | Linux namespaces, `resource` | `nsjail`, Docker/Podman, `wasmtime` | Subprocess is not a security boundary |

Policy:

- Pin versions in `uv.lock`; no “latest” in a release.
- Extras: `lmpu[host]`, `lmpu[vllm]`, `lmpu[decoder]`, `lmpu[dashboard]`, `lmpu[bench]`, `lmpu[dev]`.
- `lmpu doctor` reports missing optional dependencies and the active sandbox profile.
- A missing optional dependency must produce a clear error at startup, not a late `ImportError` inside a worker.
- The vLLM adapter checks the version and patch points before serving; unsupported versions fail closed.
- Qwen3.5 is multimodal; use the text-only path. Do not enable remote code unless explicitly configured.

---

## Appendix E — Demo and test runbook

Make targets:

```text
make setup            # uv sync with dev extras; create .env from .env.example
make doctor           # Python, shm, UDS, GPU, NVML, model cache, sandbox
make check            # ruff + mypy + unit/property/contract
make test-core        # integration + scenario, no GPU/model
make test-gpu         # E2E model tests on the reference GPU
make test-failure     # failure-injection suite
make test-security    # sandbox/profile security tests
make test-fuzz        # parser, framing, and buffer fuzz tests
make demo             # demos 01-11 with assertion mode
make mission-control  # demos/12 + dashboard on 127.0.0.1
make bench-smoke      # short benchmark sanity run
make bench            # full pre-registered benchmark
make soak             # 1-hour mixed workload
make release-check    # all gates, artifacts, and license/SBOM checks
```

Recommended first run:

1. `make setup`
2. Download `Qwen/Qwen3.5-0.8B`; confirm the cache.
3. `make doctor`
4. `make test-core`
5. `make test-gpu`
6. `make mission-control`
7. Open the dashboard and run the scenario board.
8. `make bench-smoke`, then `make bench`.

If a step fails, fix the gate before starting the next phase. Do not work around a red gate with a demo-only path.
