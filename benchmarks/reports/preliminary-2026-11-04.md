# Preliminary conclusive results — 2026-11-04

**Hardware**: NVIDIA RTX 3080 (10 GB), CPU container
**Model**: Qwen/Qwen2.5-1.5B-Instruct (bf16) where a host model is used
**Protocol**: lmpu/1
**Scripts**: `benchmarks/experiments/dataplane.py`, `benchmarks/experiments/multisession.py`

These two experiments test the claims the architecture was built for. Both pass.
Neither is a token-count comparison against a strong agent; that comparison is a
*separate* boundary and is reported as such.

---

## 1. Data-plane invariance (README §5, §10)

Claim: *host cost does not scale with data size.*
Method: import a buffer of size S, reference it through the bounded-slice read,
count the control + summary tokens the host would ingest. Compare with the
tokens needed to put the same bytes into context (`S/4`, conservative).

| Buffer | Import s | Read s | Host tokens | Context tokens | Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 KiB | 0.000 | 0.000 | 29 | 256 | 9× |
| 64 KiB | 0.000 | 0.000 | 37 | 16,384 | 443× |
| 1 MiB | 0.001 | 0.000 | 37 | 262,144 | 7,085× |
| 16 MiB | 0.013 | 0.000 | 37 | 4,194,304 | 113,360× |
| 64 MiB | 0.054 | 0.000 | 37 | 16,777,216 | 453,438× |
| 256 MiB | 0.205 | 0.000 | 37 | 67,108,864 | 1,813,753× |
| 512 MiB | 0.404 | 0.000 | 37 | 134,217,728 | 3,627,506× |

**Result**: host tokens are constant at **37** across 8,000× growth in data
(64 KiB → 512 MiB). Ratio reaches **3.6 × 10⁶** at 512 MiB. Import cost is
linear and small (0.4 s for 512 MiB); read cost is ~0.

This is the asymptotic claim in the README, measured. It is not a claim about
beating a prefix-cached agent on a fixed-size task; it is a claim that the
scaling law in data size is flat.

---

## 2. Session overcommit (README §4, §22)

Claim: *a parked session frees the processor; N sessions overlap.*
Method: N sessions each submit a 3 s sleeping job, then wait. Measure wall time.
Serial execution gives `3N` s; full overlap gives `3` s.

| Sessions | Wall | Serial | Overlap |
|---:|---:|---:|---:|
| 1 | 3.37 s | 3 s | 0.89× |
| 2 | 3.64 s | 6 s | 1.65× |
| 3 | 3.95 s | 9 s | 2.28× |
| 4 | 4.22 s | 12 s | 2.84× |
| 8 | 7.30 s | 24 s | 3.29× |

**Result**: 4 sessions complete in 4.2 s instead of 12 s (2.84×). 8 sessions
complete in 7.3 s instead of 24 s (3.29×), i.e. two waves at the
`max_concurrent=4` limit.

---

## 3. Defects found and fixed during this measurement

Both defects made the architecture look worse than it is. Neither was a
limitation of the design.

**D1 — dispatcher serialised jobs.**
`Runtime._dispatch_loop` awaited `_run_job` inline, so one job ran per processor
regardless of `max_workers_per_processor` or the scheduler semaphore.
Fixed: dispatch spawns a task per job and releases the slot on completion.
Effect: N=3 wall 9.6 s → 3.95 s.

**D2 — spawned workers were immediately reusable.**
`WorkerPool._spawn_worker` added the new worker to `_available_workers` before
returning it to its caller, so the next `get_worker` handed the same busy worker
to a second job. Effective concurrency was 2 with 4 jobs sharing 2 workers.
Fixed: a worker becomes available only in `_release_worker`.
Effect: N=4 wall 6.8 s → 4.22 s; N=8 13.0 s → 7.30 s.

---

## 4. What these results do and do not establish

**Established**

- Context cost is invariant to data size over 4 orders of magnitude (C-data).
- A parked session does not block another session (C-overlap).
- The capability and negative-control results from Batch 3 stand.

**Not established**

- That LMPU beats a prefix-cached agent on tokens for small fixed tasks. It
  does not (E=NARROW, Batch 1). That is one boundary, not the verdict.
- That buffer outputs compose into another co-processor without host round-trip.
  No processor currently accepts a buffer reference as input; this is the next
  experiment and the strongest remaining claim.
- Any energy advantage below the NVML noise floor.

---

## 5. Reproduction

```bash
uv run python benchmarks/experiments/dataplane.py
uv run python benchmarks/experiments/multisession.py
```