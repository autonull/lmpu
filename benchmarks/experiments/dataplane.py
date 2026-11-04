"""Data-plane invariance: host tokens must not scale with buffer size.

Measures the control + summary tokens the host would ingest when a buffer is
referenced through the slice path, against the tokens required to ingest the
same data as context. The claim under test (README §5, §10) is that the first
is constant while the second is linear.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

SIZES = [
    ("1 KiB", 1_024),
    ("64 KiB", 65_536),
    ("1 MiB", 1_048_576),
    ("16 MiB", 16_777_216),
    ("64 MiB", 67_108_864),
    ("256 MiB", 268_435_456),
    ("512 MiB", 536_870_912),
]


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
        print(
            f"{'buffer':>9} {'import_s':>9} {'read_s':>7} {'host_tok':>9} "
            f"{'ingest_tok':>11} {'ratio':>10}"
        )
        try:
            for label, size in SIZES:
                data = bytes(size)
                t0 = time.perf_counter()
                ref = await rt._buffer_import(
                    {"name": "d", "data": data, "dtype": "uint8", "shape": [size]}, sid
                )
                t1 = time.perf_counter()
                request = {"buffer": ref.buffer_id, "max_inline_bytes": 1024}
                view = await rt._read(request, sid)
                t2 = time.perf_counter()
                summary = {
                    "buffer": view.buffer,
                    "dtype": view.dtype,
                    "derived": view.data is None and view.derived_buffer is not None,
                    "note": view.note,
                }
                host_tok = estimate_tokens(request) + estimate_tokens(summary)
                ingest = size // 4
                print(
                    f"{label:>9} {t1 - t0:>9.3f} {t2 - t1:>7.3f} {host_tok:>9} "
                    f"{ingest:>11} {ingest / host_tok:>9.0f}x"
                )
                del data
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()


if __name__ == "__main__":
    asyncio.run(main())