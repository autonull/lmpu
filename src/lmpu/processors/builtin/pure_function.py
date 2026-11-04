"""LMPU Processors - Pure function processor for deterministic operations."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..base import JobContext, Manifest, Processor, ProcessorContext, Result


class PureFunctionProcessor(Processor):
    """Pure function processor for deterministic, stateless operations."""

    manifest = Manifest(
        name="pure.func",
        version="0.1.0",
        protocol="lmpu/1",
        ops=[
            {"name": "bigint_add", "inputs": ["a", "b"], "outputs": ["result"]},
            {"name": "bigint_mul", "inputs": ["a", "b"], "outputs": ["result"]},
            {"name": "matmul", "inputs": ["size", "seed"], "outputs": ["shape", "checksum", "sum"]},
            {
                "name": "text_stats",
                "inputs": ["text"],
                "outputs": ["chars", "words", "checksum", "secret"],
            },
            {"name": "mod_pow", "inputs": ["base", "exp", "mod"], "outputs": ["result"]},
            {"name": "sha256", "inputs": ["data"], "outputs": ["hash"]},
            {"name": "blake3", "inputs": ["data"], "outputs": ["hash"]},
            {"name": "json_query", "inputs": ["data", "query"], "outputs": ["result"]},
            {"name": "lookup", "inputs": ["table", "key"], "outputs": ["value"]},
            {"name": "batch", "inputs": ["items"], "outputs": ["results"]},
        ],
        limits={
            "max_buffer_bytes": 10_000_000,
            "max_batch": 1000,
        },
        determinism=["strict"],
        latency_class="us",
        streaming=False,
        progress=False,
        interactive=False,
        persistent_state=False,
        batching=True,
        zero_copy=False,
        sandbox=["subprocess"],
        executor="inline",
    )

    async def setup(self, ctx: ProcessorContext) -> None:
        pass

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        # Batch jobs have items in inputs, not a separate "batch" key
        if inputs.get("items") is not None or op == "batch":
            return await self._batch(job, ctx)

        try:
            if op == "bigint_add":
                return await self._bigint_add(inputs)
            elif op == "bigint_mul":
                return await self._bigint_mul(inputs)
            elif op == "matmul":
                return await self._matmul(inputs)
            elif op == "text_stats":
                return await self._text_stats(inputs)
            elif op == "mod_pow":
                return await self._mod_pow(inputs)
            elif op == "sha256":
                return await self._sha256(inputs)
            elif op == "blake3":
                return await self._blake3(inputs)
            elif op == "json_query":
                return await self._json_query(inputs)
            elif op == "lookup":
                return await self._lookup(inputs)
            elif op == "batch":
                return await self._batch(job, ctx)
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

    async def _bigint_add(self, inputs: dict[str, Any]) -> Result:
        a = int(inputs.get("a", 0))
        b = int(inputs.get("b", 0))
        result = a + b
        return Result(
            success=True,
            data={"result": str(result)},
            checksum=hashlib.sha256(str(result).encode()).hexdigest()[:16],
        )

    async def _bigint_mul(self, inputs: dict[str, Any]) -> Result:
        a = int(inputs.get("a", 0))
        b = int(inputs.get("b", 0))
        result = a * b
        return Result(
            success=True,
            data={"result": str(result)},
            checksum=hashlib.sha256(str(result).encode()).hexdigest()[:16],
        )

    async def _matmul(self, inputs: dict[str, Any]) -> Result:
        import numpy as np

        size = int(inputs.get("size", 256))
        seed = int(inputs.get("seed", 0))
        rng = np.random.default_rng(seed)
        left = rng.standard_normal((size, size), dtype=np.float32)
        right = rng.standard_normal((size, size), dtype=np.float32)
        product = left @ right
        return Result(
            success=True,
            data={
                "shape": list(product.shape),
                "checksum": hashlib.sha256(product.tobytes()).hexdigest()[:16],
                "sum": float(product.sum(dtype=np.float64)),
            },
            checksum=hashlib.sha256(product.tobytes()).hexdigest()[:16],
        )

    async def _text_stats(self, inputs: dict[str, Any]) -> Result:
        text = str(inputs.get("text", ""))
        match = re.search(r"SECRET:\s*([A-Za-z0-9-]+)", text)
        return Result(
            success=True,
            data={
                "chars": len(text),
                "words": len(text.split()),
                "checksum": hashlib.sha256(text.encode()).hexdigest()[:16],
                "secret": match.group(1) if match else None,
            },
            checksum=hashlib.sha256(text.encode()).hexdigest()[:16],
        )

    async def _mod_pow(self, inputs: dict[str, Any]) -> Result:
        base = int(inputs.get("base", 0))
        exp = int(inputs.get("exp", 0))
        mod = int(inputs.get("mod", 1))
        result = pow(base, exp, mod)
        return Result(
            success=True,
            data={"result": str(result)},
            checksum=hashlib.sha256(str(result).encode()).hexdigest()[:16],
        )

    async def _sha256(self, inputs: dict[str, Any]) -> Result:
        data = inputs.get("data", b"")
        if isinstance(data, str):
            data = data.encode()
        elif isinstance(data, dict) and data.get("buffer"):
            # Buffer reference - would need buffer manager
            data = b""  # Placeholder

        hash_obj = hashlib.sha256(data)
        hex_digest = hash_obj.hexdigest()
        return Result(
            success=True,
            data={"hash": hex_digest},
            checksum=hex_digest[:16],
        )

    async def _blake3(self, inputs: dict[str, Any]) -> Result:
        try:
            import blake3

            data = inputs.get("data", b"")
            if isinstance(data, str):
                data = data.encode()
            hash_obj = blake3.blake3(data)
            hex_digest = hash_obj.hexdigest()
        except ImportError:
            # Fallback to sha256
            import hashlib

            data = inputs.get("data", b"")
            if isinstance(data, str):
                data = data.encode()
            hex_digest = hashlib.sha256(data).hexdigest()

        return Result(
            success=True,
            data={"hash": hex_digest},
            checksum=hex_digest[:16],
        )

    async def _json_query(self, inputs: dict[str, Any]) -> Result:
        data = inputs.get("data", {})
        query = inputs.get("query", "")

        # Simple JSONPath-like query (very basic)
        if query.startswith("$."):
            path = query[2:].split(".")
            result = data
            for key in path:
                if isinstance(result, dict):
                    result = result.get(key)
                elif isinstance(result, list) and key.isdigit():
                    result = result[int(key)]
                else:
                    result = None
                    break
        else:
            result = data

        return Result(
            success=True,
            data={"result": result},
            checksum=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16],
        )

    async def _lookup(self, inputs: dict[str, Any]) -> Result:
        table = inputs.get("table", {})
        key = inputs.get("key")

        if isinstance(table, dict):
            value = table.get(key)
        else:
            value = None

        return Result(
            success=True,
            data={"value": value},
            checksum=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
            if value
            else "",
        )

    async def _batch(self, job: dict[str, Any], ctx: JobContext) -> Result:
        """Execute batch of operations."""
        # Batch data is in job["inputs"]["items"], not job["batch"]
        inputs = job.get("inputs", {})
        items = inputs.get("items", [])
        default_op = job.get("op", "bigint_mul")

        results = []
        errors = []

        for i, item in enumerate(items):
            try:
                item_inputs = item.get("inputs", {})
                item_op = item.get("op", default_op)

                # Create sub-job
                sub_job = {"op": item_op, "inputs": item_inputs}
                sub_result = await self.execute(sub_job, ctx)

                if sub_result.success:
                    results.append(sub_result.data)
                else:
                    errors.append({"index": i, "error": sub_result.error_message})
                    results.append(None)
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
                results.append(None)

        return Result(
            success=True,
            data={"results": results, "errors": errors},
            checksum=hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16],
        )

    async def cancel(self, job_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass
