# LMPU — Abstract Computational Co-Processor for Language Models

> **Measured status** — preliminary paper: [`paper/preliminary.md`](paper/preliminary.md) ·
> development plan: [`TODO.5.md`](TODO.5.md) · claim artifacts: `benchmarks/results/claims/`.
> Headline: 93–95 control tokens from 1.2 KiB to 256 MiB; 3,200 records exceed a
> 32 k window in-context but answer in 94 tokens; a 2048³ matmul runs 14.4× faster
> and 16.4× more efficiently on the GPU than the CPU. A script-writing sandbox
> agent ties on tokens — see the paper's non-claims.

**LMPU is a working implementation** of the abstract co-processor architecture. It provides a typed, asynchronous interface between a host language model and external computational units (co-processors), with native sleep/wake, shared buffers, and resource accounting.

---

## Why This Exists: The Token Tax Problem

**Large language models are expensive at the margin.** Every forward pass costs GPU FLOPs, memory bandwidth, KV-cache memory, and energy. Yet current agent systems force models to:

- **Simulate computation** by generating trace tokens (Qwenputer-style: 1 CPU cycle ≈ 1 model token)
- **Poll for status** by repeatedly invoking the model to ask "are you done yet?"
- **Serialize large results** into the context window (16 MB matrix = millions of tokens)
- **Narrate execution** instead of delegating to native hardware

This is the **token tax**: the model pays inference cost for every computational step, every poll, every byte of output.

### The Co-Processor Alternative

LMPU implements a different architecture: **the model is a semantic supervisor, not a CPU.**

```
┌─────────────────────────────────────────────────────────────────┐
│  Host Model (LLM)                                               │
│  • Understands intent, chooses tools, interprets results        │
│  • Emits ~40 control tokens: "submit matmul, sleep until done"  │
│  • SLEEPS: 0 forward passes, 0 tokens generated, KV cache frozen│
└──────────────────────────────┬──────────────────────────────────┘
                               │ JSON-RPC (Unix socket)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  LMPU Kernel Runtime                                            │
│  • Schedules job on vector.engine (BLAS/GPU)                    │
│  • 2048³ matmul executes at native speed (~0.1s)                │
│  • Writes 16 MB result to shared buffer                         │
│  • Injects 40-token wake event: "job done, result in buf_C"     │
└─────────────────────────────────────────────────────────────────┘
```

**The model only pays for decisions, not steps.** Cost scales as `host_tokens ≈ a + b·decisions`, independent of data size, execution duration, or hardware instruction count.

---

## Measured Results (All Core Claims Validated)

| Claim | Description | Measured Result | Evidence |
|-------|-------------|-----------------|----------|
| **P1** | Control cost flat in data size | 93–95 tok, 1.2 KiB → 256 MiB (218,000×) | `baseline_law.py` |
| **P2** | Parked sessions cost nothing | 0 forwards, 0.696s vs 0.699s; 4 concurrent 4.04× | `parked_sessions.py` |
| **P3** | Capability beyond context window | 3,200 records → 55k in-context; LMPU 94 | `baseline_law.py` |
| **P5** | Buffer output composes | 168±0 tok, 3-stage, 1–256 MiB | `buffer_composition.py` |
| **P6** | GPU plane energy-positive | 2048³: 14.4× faster, 16.4× less energy | `gpu_plane.py` |
| **P7** | Protocol overhead flat | 37 tok, 64 KiB → 512 MiB | `dataplane.py` |
| **O0** | Horizon scaling (utility) | 66–68 tok/job constant N=1→40; 3.85× wall speedup | `horizon.py` |
| **O1** | Adaptive supervision | Oracle F1=0.976; One-shot 0.499 post-drift; Adaptive 0.976 (100% recovery, 3 dec/shard) | `adaptive_supervision.py` |
| **O2** | Decision-cost law | `host_tokens ≈ 17 + 93×decisions`; bytes slope 0.0000 | `decision_cost_law.py` |

**Non-claims (measured ties):**
- Script-writing sandbox agent ties LMPU on tokens (O(1) control cost)
- Token counts vs. a script agent: it is a tie, measured

All claim artifacts: `benchmarks/results/claims/*.json`

---

## Architecture (Implemented)

```
┌─────────────────────────────────────────────────────────────┐
│  Host Model (LLM)                                           │
│  - Generates control tokens / structured calls              │
│  - Sleeps while co-processor works (KV cache frozen)        │
└──────────────────────┬──────────────────────────────────────┘
                       │ JSON-RPC over Unix socket
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LMPU Kernel Runtime                                        │
│  - JobTable, Scheduler, QuotaManager, EventLog              │
│  - BufferManager (SHM, grants, checksums, partial reads)    │
│  - CapabilityRegistry (discovery, routing, limits)          │
│  - WorkerPool (process isolation, timeouts, retries)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬─────────────────┐
        ▼              ▼              ▼                 ▼
   pure.func      python.vm      vector.engine    symbolic.planner
   (exact math)   (Python VM)    (BLAS/NumPy)     (exact search)
        ▼              ▼              ▼                 ▼
   neural.summary  hardware.stub  ...              ...
```

---

## Core Primitives (Host → Runtime)

The host model emits structured control messages. The runtime intercepts, executes, and injects compact events back into the host's generation stream.

```json
// Submit an asynchronous job
{"method": "lmpu.submit", "params": {
  "processor": "vector.engine",
  "op": "matmul",
  "inputs": {"A": {"buffer": "buf_A"}, "B": {"buffer": "buf_B"}},
  "outputs": {"C": {"buffer": "buf_C"}},
  "config": {"callback_mode": "async"}
}}

// Sleep until job completes (host generates 0 tokens while waiting)
{"method": "lmpu.sleep", "params": {
  "until": {"any": ["job_001"]},
  "level": "S1"
}}

// Wake event injected into host context (resumes generation)
{"method": "lmpu.wake", "params": {}}

// Read bounded result slice (never dumps full matrix into context)
{"method": "lmpu.read", "params": {
  "buffer": "buf_C",
  "slices": [[0, 4], [0, 4]],
  "max_inline_bytes": 4096
}}
```

**Primitive operations:** `submit`, `poll`, `wait`, `read`, `write`, `cancel`, `sleep`, `wake`, `capabilities`, `configure`

**Events injected into host:** `JobCompleted`, `JobFailed`, `Progress`, `NeedsMoreInput`, `Woken`, `BufferUpdated`, `ControlError`

---

## Efficiency Gains (Measured)

| Dimension | Tool Calls / Chain-of-Thought | LMPU Co-Processor |
|-----------|------------------------------|-------------------|
| **Large data** | Serialized into context (millions of tokens) | Buffer reference (~40 tokens) |
| **Long jobs** | Polling / blocking model turns | Host sleeps: 0 forward passes |
| **Context growth** | O(data size + steps) | O(decisions) ≈ constant |
| **Energy while waiting** | High (generating tokens) | Near-zero (KV cache frozen) |
| **Concurrency** | Application-layer orchestration | Runtime-native scheduling |
| **Exact computation** | Unreliable / token-expensive | Native deterministic executors |
| **Hardware routing** | Hidden / manual | Capability-based, explicit |

**Key scaling law** (O2): `host_tokens ≈ 17 + 93×decisions` — independent of data size and execution duration.

---

## Co-Processor Types (Implemented)

| Processor | Operations | Deterministic | Latency | Use Case |
|-----------|------------|---------------|---------|----------|
| `pure.func` | bigint_add, mul, mod_pow, sha256, json_query | Yes | μs | Exact arithmetic, crypto, lookup |
| `python.vm` | run_program | No | ms–s | Multi-step programs, data processing |
| `vector.engine` | matmul, dot, fft, conv1d, softmax | Yes | ms | Linear algebra, signal processing |
| `symbolic.planner` | bfs_plan, constraint_solve, shortest_path | Yes | ms–s | Exact search, planning |
| `neural.summary` | summarize_text_buffer | No | s | Neural summarization / retrieval |
| `hardware.stub` | (simulated) | Yes | configurable | Hardware accelerator placeholder |

Capability descriptors queried at runtime via `lmpu.capabilities`.

---

## Integration with AIOS (Strategic Direction)

LMPU is designed to integrate as the **Compute Manager** subsystem of [AIOS](https://github.com/aios-foundation/aios):

| Layer | Responsibility |
|-------|----------------|
| **AIOS** | Agent lifecycle, LLM inference, context/memory, tool management, high-level scheduling |
| **LMPU** | Computational job lifecycle, worker/processor management, buffer management, job-level scheduling, resource accounting |

**Proposed Cerebrum API:**
```python
job = await aios.compute.submit(
    processor="python.vm",
    operation="analyze",
    inputs={"dataset": dataset_buffer},
    budget={"memory_mb": 4096, "timeout_seconds": 120},
)
await job.wait()
result = await job.result()
```

**Integration phases:**
1. LMPU as AIOS compute sidecar (IPC)
2. Native Cerebrum Compute API
3. Kernel integration (hierarchical scheduling, shared context events)
4. Durable workflows (checkpointing, replay, recovery)
5. Rust implementation (`aios-rs`)

---

## Quick Start

```bash
# Install dependencies
uv sync --dev --all-extras

# Run core measured experiments
uv run python benchmarks/experiments/baseline_law.py       # P1, P3
uv run python benchmarks/experiments/horizon.py            # O0
uv run python benchmarks/experiments/decision_cost_law.py  # O2
uv run python benchmarks/experiments/adaptive_supervision.py # O1

# Run full test suite (186 tests)
uv run python -m pytest tests/ -q

# View claim artifacts
cat benchmarks/results/claims/*.json
```

---

## Repository Structure

```
lmpu/
├── src/lmpu/
│   ├── kernel/           # Runtime, scheduler, quotas, jobs, events
│   ├── host/             # Session controller, HF engine, parser, metrics
│   ├── processors/       # pure.func, python.vm, vector.engine, ...
│   ├── memory/           # BufferManager, SHM, grants, checksums
│   ├── protocol/         # JSON-RPC messages, errors, sentinels
│   ├── bench/            # Benchmark harness, workloads, reports
│   └── observability/    # Tracing, dashboard
├── benchmarks/
│   ├── experiments/      # P1–P7, O0–O2 measurement scripts
│   └── results/claims/   # JSON claim artifacts
├── paper/
│   └── preliminary.md    # Living draft paper
├── tests/                # Unit, integration, scenario, property, failure
└── TODO.5.md             # Development plan (Revision 3)
```

---

## Key Design Principles (Validated by Measurement)

1. **Control plane ≠ data plane** — Model sees 40-token summaries; 16 MB matrices stay in buffers
2. **Sleep is a first-class primitive** — Host forwards = 0 while co-processor works
3. **Decisions, not steps** — Cost scales with model decisions, not computational steps
4. **Capability-based routing** — Host queries what's available; runtime dispatches
5. **Determinism flags** — Exact co-processors advertise `deterministic: true`; results checksummed
6. **Model-agnostic** — Works with any autoregressive model (tested: Qwen2.5-0.5B)
7. **No in-weight computation** — Co-processor is external; no token tax per instruction

---

## Status & Roadmap

| Milestone | Status |
|-----------|--------|
| Measured core (P1–P7) | ✅ Complete |
| O0 Horizon scaling | ✅ Complete |
| O1 Adaptive supervision | ✅ Complete |
| O2 Decision-cost law | ✅ Complete |
| O3 Replay / ledger / budgets | ⏳ Next |
| O4 Crash-resume | ⏳ Next |
| Paper draft (with O0/O1/O2) | ⏳ After O3/O4 |
| AIOS integration (Phase 1) | 📋 Planned |

See [`TODO.5.md`](TODO.5.md) for detailed development plan.

---

## License

Apache-2.0