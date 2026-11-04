"""GPU compute plane: does the data plane actually use the GPU, and does the
energy difference survive measurement?

Part A (compute): the same matmul on CPU (numpy) and GPU (torch/CUDA), small
matrices, timed with NVML power sampled throughout. Reports GFLOPS, wall, mean
watts, and joules per matmul.

Part B (plane): import -> matmul on CUDA -> read, through the runtime. Confirms
the host sees a flat control payload while the GPU does the work.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lmpu.host.metrics import estimate_tokens
from lmpu.kernel.runtime import Runtime

DIMS = [1024, 2048]



@dataclass
class ComputeRow:
    device: str
    dim: int
    wall_s: float
    gflops: float
    mean_w: float
    joules: float
    joules_per_matmul: float


class PowerSampler:
    """Sample NVML power via nvidia-smi dmon for the duration of a window."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            if line.startswith(b"#"):
                continue
            # dmon -s p columns: idx, pwr(W), gtemp(C), mtemp(C)
            parts = line.decode().strip().split()
            if len(parts) >= 2:
                try:
                    self.samples.append(float(parts[1]))
                except ValueError:
                    pass

    def __enter__(self) -> PowerSampler:
        self._proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "p", "-d", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.3)  # let dmon emit its header and first sample
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        if self._thread:
            self._thread.join(timeout=1)

    @property
    def mean_w(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0


def matmul_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def matmul_gpu(a: np.ndarray, b: np.ndarray):
    import torch

    with torch.inference_mode():
        return torch.from_numpy(a).cuda() @ torch.from_numpy(b).cuda()


def run_compute(dim: int, device: str) -> ComputeRow:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((dim, dim), dtype=np.float32)
    b = rng.standard_normal((dim, dim), dtype=np.float32)
    flops = 2 * dim**3

    def once():
        return matmul_gpu(a, b) if device == "gpu" else matmul_cpu(a, b)

    if device == "gpu":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        matmul_gpu(a, b)  # warm up kernels, allocation, and clocks
        torch.cuda.synchronize()

    # Size the window so NVML's 100 ms cadence can resolve power: a 90 ms burst
    # reads as idle. One timed call, then enough repeats to reach ~2.5 s.
    t0 = time.perf_counter()
    once()
    if device == "gpu":
        import torch

        torch.cuda.synchronize()
    single = time.perf_counter() - t0
    repeats = max(3, min(200, int(2.5 / single) + 1))

    with PowerSampler() as power:
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = once()
        if device == "gpu":
            import torch

            torch.cuda.synchronize()
            _ = out.sum().item()  # force completion
        wall = time.perf_counter() - t0

    mean_w = power.mean_w
    return ComputeRow(
        device=device,
        dim=dim,
        wall_s=wall / repeats,
        gflops=(flops * repeats) / wall / 1e9,
        mean_w=mean_w,
        joules=mean_w * wall,
        joules_per_matmul=mean_w * wall / repeats,
    )


async def plane_check() -> dict[str, object]:
    """import -> matmul (CUDA) -> read, measuring the host payload."""
    dim = 2048
    rng = np.random.default_rng(1)
    a = rng.standard_normal((dim, dim), dtype=np.float32)
    b = rng.standard_normal((dim, dim), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        # Workers inherit the environment; the CUDA path is opt-in there.
        os.environ["LMPU_VECTOR_DEVICE"] = "cuda"
        rt = Runtime(
            socket_path=f"{tmp}/run.sock",
            shm_dir=f"{tmp}/shm",
            config={"trace_dir": f"{tmp}/traces"},
        )
        await rt.start()
        sid = (await rt._hello({"protocol": "lmpu/1"})).session_id
        try:
            ra = await rt._buffer_import(
                {"name": "A", "data": a.tobytes(), "dtype": "float32", "shape": [dim, dim]}, sid
            )
            rb = await rt._buffer_import(
                {"name": "B", "data": b.tobytes(), "dtype": "float32", "shape": [dim, dim]}, sid
            )
            rc = await rt._buffer_create(
                {"name": "C", "dtype": "float32", "shape": [dim, dim]}, sid
            )
            job = {
                "processor": "vector.engine",
                "op": "matmul",
                "inputs": {"A": {"buffer": ra.buffer_id}, "B": {"buffer": rb.buffer_id}},
                "outputs": {"C": {"buffer": rc.buffer_id}},
                "config": {"mode": "sync", "timeout_ms": 120000},
            }
            host_tokens = estimate_tokens(job)
            t0 = time.perf_counter()
            result = await rt._submit(job, sid)
            wall = time.perf_counter() - t0
            view = await rt._read({"buffer": rc.buffer_id, "max_inline_bytes": 0}, sid)
            host_tokens += estimate_tokens(
                {"buffer": rc.buffer_id, "dtype": view.dtype, "derived": True}
            )
            return {
                "status": str(result.status),
                "host_tokens": host_tokens,
                "wall_s": wall,
                "summary": result.result,
            }
        finally:
            await rt._close({"detach": False}, sid)
            await rt.stop()


def main() -> None:
    rows = []
    for dim in DIMS:
        for device in ("cpu", "gpu"):
            row = run_compute(dim, device)
            rows.append(row)
            print(
                f"{device:<4} {dim:>5}  {row.wall_s * 1000:>8.2f} ms  "
                f"{row.gflops:>7.1f} GFLOPS  {row.mean_w:>6.1f} W  "
                f"{row.joules_per_matmul:>8.2f} J/matmul"
            )

    print("\nplane check (import -> cuda matmul -> read):")
    plane = asyncio.run(plane_check())
    print(f"  status={plane['status']} host_tokens={plane['host_tokens']} "
          f"wall={plane['wall_s']:.3f}s")
    print(f"  summary={plane['summary']}")

    out = Path("benchmarks/results/gpu_plane")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps({"compute": [asdict(r) for r in rows], "plane": plane}, indent=2)
    )
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    main()
