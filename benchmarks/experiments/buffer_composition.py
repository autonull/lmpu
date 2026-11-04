"""Buffer composition (R1/P5): Host tokens must stay flat across a multi-stage pipeline.

Measures the host tokens ingested when chaining:
  import → reduce (sha256) → hash (text_stats) → read

The claim: host tokens flat regardless of pipe length and data size (≥ 64 MiB).
Only counts the *control path* tokens (submit requests + summaries + read), not
the one-time import setup.

Only the invariance property is tested: tokens must not scale with data size.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import numpy as np

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

SIZES = [
    ("1 MiB", 1_048_576),
    ("16 MiB", 16_777_216),
    ("64 MiB", 67_108_864),
    ("256 MiB", 268_435_456),
]


def submit_request_tokens(processor: str, op: str, input_buffer: str, output_buffer: str) -> int:
    """Estimate tokens for a submit request with buffer refs."""
    req = {
        "processor": processor,
        "op": op,
        "inputs": {"data": {"buffer": input_buffer}}
        if op == "sha256"
        else {"text": {"buffer": input_buffer}},
        "outputs": {"hash": {"buffer": output_buffer}}
        if op == "sha256"
        else {"text": {"buffer": output_buffer}},
        "config": {"mode": "sync"},
    }
    return estimate_tokens(req)


def submit_summary_tokens(
    job_id: str, status: str, result_checksum: str, buffer_refs: list[str]
) -> int:
    """Estimate tokens for a minimal submit response summary."""
    summary = {
        "job_id": job_id,
        "status": status,
        "result_checksum": result_checksum,
        "buffer_refs": buffer_refs,
    }
    return estimate_tokens(summary)


def read_request_tokens(buffer_id: str) -> int:
    """Estimate tokens for a read request."""
    return estimate_tokens({"buffer": buffer_id, "max_inline_bytes": 1024})


def read_summary_tokens(buffer: str, dtype: str, derived: bool, note: str | None) -> int:
    """Estimate tokens for a read summary."""
    return estimate_tokens({"buffer": buffer, "dtype": dtype, "derived": derived, "note": note})


async def run_pipeline(rt: Runtime, sid: str, size: int) -> tuple[int, dict]:
    """Run a 3-stage pipeline and return total control-path host tokens."""
    # Stage 0: Import (setup, not counted in control path)
    data = np.random.default_rng(42).bytes(size)
    ref = await rt._buffer_import(
        {"name": "source", "data": data, "dtype": "uint8", "shape": [size]}, sid
    )

    # Stage 1: Reduce (sha256) - output to a small buffer (64 bytes for hex digest)
    hash_ref = await rt._buffer_create({"name": "hash", "dtype": "uint8", "shape": [64]}, sid)
    t0 = time.perf_counter()
    sha_req_tok = submit_request_tokens("pure.func", "sha256", ref.buffer_id, hash_ref.buffer_id)
    sha_job = await rt._submit(
        {
            "processor": "pure.func",
            "op": "sha256",
            "inputs": {"data": {"buffer": ref.buffer_id}},
            "outputs": {"hash": {"buffer": hash_ref.buffer_id}},
            "config": {"mode": "sync", "timeout_ms": 60_000},
        },
        sid,
    )
    t1 = time.perf_counter()
    # Get buffer_refs from job record
    sha_job_rec = rt.job_table.get(sha_job.job_id)
    sha_buffer_refs = sha_job_rec.buffer_refs if sha_job_rec else []
    sha_checksum = (
        sha_job_rec.result.get("checksum", "") if sha_job_rec and sha_job_rec.result else ""
    )
    sha_resp_tok = submit_summary_tokens(
        sha_job.job_id, sha_job.status.value, sha_checksum, [b.buffer_id for b in sha_buffer_refs]
    )

    # Stage 2: Hash (text_stats on the hash) - output to another small buffer
    stats_ref = await rt._buffer_create({"name": "stats", "dtype": "uint8", "shape": [256]}, sid)
    t2 = time.perf_counter()
    stats_req_tok = submit_request_tokens(
        "pure.func", "text_stats", hash_ref.buffer_id, stats_ref.buffer_id
    )
    stats_job = await rt._submit(
        {
            "processor": "pure.func",
            "op": "text_stats",
            "inputs": {"text": {"buffer": hash_ref.buffer_id}},
            "outputs": {"text": {"buffer": stats_ref.buffer_id}},
            "config": {"mode": "sync", "timeout_ms": 60_000},
        },
        sid,
    )
    t3 = time.perf_counter()
    stats_job_rec = rt.job_table.get(stats_job.job_id)
    stats_buffer_refs = stats_job_rec.buffer_refs if stats_job_rec else []
    stats_checksum = (
        stats_job_rec.result.get("checksum", "") if stats_job_rec and stats_job_rec.result else ""
    )
    stats_resp_tok = submit_summary_tokens(
        stats_job.job_id,
        stats_job.status.value,
        stats_checksum,
        [b.buffer_id for b in stats_buffer_refs],
    )

    # Stage 3: Read final result
    t4 = time.perf_counter()
    read_req_tok = read_request_tokens(stats_ref.buffer_id)
    view = await rt._read({"buffer": stats_ref.buffer_id, "max_inline_bytes": 1024}, sid)
    t5 = time.perf_counter()
    read_resp_tok = read_summary_tokens(
        view.buffer, view.dtype, view.data is None and view.derived_buffer is not None, view.note
    )

    total_host_tok = (
        sha_req_tok + sha_resp_tok + stats_req_tok + stats_resp_tok + read_req_tok + read_resp_tok
    )

    timing = {
        "sha_s": t1 - t0,
        "stats_s": t3 - t2,
        "read_s": t5 - t4,
    }

    return total_host_tok, timing


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rt = Runtime(
            socket_path=str(Path(tmp) / "run.sock"),
            shm_dir=str(Path(tmp) / "shm"),
            config={"trace_dir": str(Path(tmp) / "traces")},
        )
        await rt.start()
        hello = await rt._hello({"protocol": "lmpu/1"})
        sid = hello.session_id

        header = (
            f"{'buffer':>9} {'host_tok':>9} {'sha_s':>7} "
            f"{'stats_s':>8} {'read_s':>7} {'invariant':>10}"
        )
        print(header)
        try:
            prev_tok = None
            for label, size in SIZES:
                host_tok, timing = await run_pipeline(rt, sid, size)
                invariant = "PASS" if prev_tok is None or abs(host_tok - prev_tok) <= 2 else "FAIL"
                row = (
                    f"{label:>9} {host_tok:>9} {timing['sha_s']:>7.3f} "
                    f"{timing['stats_s']:>8.3f} {timing['read_s']:>7.3f} {invariant:>10}"
                )
                print(row)
                prev_tok = host_tok
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()


if __name__ == "__main__":
    asyncio.run(main())
