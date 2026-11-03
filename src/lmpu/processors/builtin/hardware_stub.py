"""LMPU Processors - Hardware stub for simulating device latency."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from ...memory.checksum import compute_checksum
from ..base import JobContext, Manifest, Processor, ProcessorContext, Result


class HardwareStubProcessor(Processor):
    """Simulated hardware device for testing latency classes and async completion."""

    manifest = Manifest(
        name="hardware.stub",
        version="0.1.0",
        protocol="lmpu/1",
        ops=[
            {"name": "simulate_latency", "inputs": ["duration_ms", "payload"], "outputs": ["result"]},
            {"name": "deterministic_compute", "inputs": ["seed", "operation"], "outputs": ["result"]},
            {"name": "stream_data", "inputs": ["chunks", "delay_ms"], "outputs": ["stream"]},
        ],
        limits={
            "max_buffer_bytes": 1_000_000,
            "max_batch": 100,
        },
        determinism=["strict", "best_effort"],
        latency_class="ms",
        streaming=True,
        progress=True,
        interactive=False,
        persistent_state=False,
        batching=True,
        zero_copy=False,
        sandbox=["subprocess"],
        executor="process",
    )

    async def setup(self, ctx: ProcessorContext) -> None:
        pass

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        config = job.get("config", {})
        ctx.progress(0.0, f"{op} started")

        try:
            if op == "simulate_latency":
                return await self._simulate_latency(inputs, ctx, config)
            elif op == "deterministic_compute":
                return await self._deterministic_compute(inputs, ctx)
            elif op == "stream_data":
                return await self._stream_data(inputs, ctx, config)
            else:
                return Result(
                    success=False,
                    error_code="unknown_op",
                    error_message=f"Unknown operation: {op}",
                )
        except Exception as e:
            return Result(
                success=False,
                error_code="execution_error",
                error_message=str(e),
                retriable=False,
            )
        finally:
            ctx.progress(1.0, f"{op} finished")

    async def _simulate_latency(
        self,
        inputs: dict[str, Any],
        ctx: JobContext,
        config: dict[str, Any],
    ) -> Result:
        """Simulate a device with configurable latency."""
        duration_ms = inputs.get("duration_ms", 100)
        payload = inputs.get("payload", {})

        # Report progress
        ctx.progress(0.1, "Starting simulation")
        await asyncio.sleep(duration_ms / 1000 * 0.3)

        ctx.progress(0.5, "Processing")
        await asyncio.sleep(duration_ms / 1000 * 0.4)

        ctx.progress(0.9, "Finalizing")
        await asyncio.sleep(duration_ms / 1000 * 0.3)

        result_data = {
            "result": {"status": "completed", "input": payload},
            "actual_duration_ms": duration_ms,
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _deterministic_compute(
        self,
        inputs: dict[str, Any],
        ctx: JobContext,
    ) -> Result:
        """Deterministic computation based on seed."""
        seed = inputs.get("seed", 42)
        operation = inputs.get("operation", "hash")

        # Use seed for deterministic output
        np.random.seed(seed)

        if operation == "hash":
            data = np.random.bytes(32)
            result = compute_checksum(data)
        elif operation == "random_vector":
            size = inputs.get("size", 100)
            result = np.random.randn(size).astype(np.float32).tolist()
        elif operation == "random_matrix":
            rows = inputs.get("rows", 10)
            cols = inputs.get("cols", 10)
            result = np.random.randn(rows, cols).astype(np.float32).tolist()
        else:
            result = {"error": f"Unknown operation: {operation}"}

        result_data = {"result": result}
        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _stream_data(
        self,
        inputs: dict[str, Any],
        ctx: JobContext,
        config: dict[str, Any],
    ) -> Result:
        """Simulate streaming data from a device."""
        chunks = inputs.get("chunks", 10)
        delay_ms = inputs.get("delay_ms", 50)

        stream_data = []
        for i in range(chunks):
            await asyncio.sleep(delay_ms / 1000)
            chunk = {"index": i, "data": np.random.randn(10).astype(np.float32).tolist()}
            stream_data.append(chunk)
            ctx.progress((i + 1) / chunks, f"Chunk {i+1}/{chunks}")

        result_data = {
            "stream": stream_data,
            "chunks": chunks,
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def cancel(self, job_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass
