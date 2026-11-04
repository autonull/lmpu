"""LMPU Processors - Pure function processor for deterministic operations."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

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
        zero_copy=True,
        sandbox=["subprocess"],
        executor="inline",
    )

    async def setup(self, ctx: ProcessorContext) -> None:
        pass

    def _resolve_input(self, ctx: JobContext, value: Any) -> Any:
        """Resolve a job input value, handling buffer references."""
        if isinstance(value, dict) and value.get("buffer"):
            # Buffer reference: {"buffer": "buf_xxx"} or {"buffer": "buf_xxx", "view": ...}
            buffer_name = value["buffer"]
            if buffer_name in ctx._buffers:
                return ctx._buffers[buffer_name]
            # Fallback: return empty bytes if buffer not found
            return b""
        return value

    def _resolve_output(
        self, ctx: JobContext, output_name: str, data: Any, checksum: str, outputs_spec: dict | None = None
    ) -> tuple[dict[str, Any], list[str]]:
        """Resolve output: write to buffer if specified in outputs_spec, otherwise return inline."""
        buffer_refs: list[str] = []
        # Only treat as output buffer if the job's outputs spec maps this output_name to a buffer
        if outputs_spec and output_name in outputs_spec:
            output_buffer_spec = outputs_spec[output_name]
            if isinstance(output_buffer_spec, dict) and output_buffer_spec.get("buffer"):
                buffer_id = output_buffer_spec["buffer"]
                # Find the mapped buffer in ctx._buffers by buffer_id
                output_spec = None
                for k, v in ctx._buffers.items():
                    if isinstance(v, np.ndarray) and hasattr(v, 'buffer_id') and v.buffer_id == buffer_id:
                        output_spec = v
                        break
                if output_spec is None:
                    # Buffer might be mapped under the buffer_id key
                    output_spec = ctx._buffers.get(buffer_id)
                
                if isinstance(output_spec, np.ndarray):
                    # Output buffer is mapped; write to it
                    if isinstance(data, str):
                        data_bytes = data.encode()
                        data_arr = np.frombuffer(data_bytes, dtype=np.uint8)
                    elif isinstance(data, bytes):
                        data_arr = np.frombuffer(data, dtype=np.uint8)
                    else:
                        data_arr = np.asarray(data)

                    if data_arr.size != output_spec.size:
                        raise ValueError(f"Output size mismatch: {data_arr.size} != {output_spec.size}")
                    np.copyto(output_spec.reshape(data_arr.shape), data_arr, casting="same_kind")
                    buffer_refs = [buffer_id]
                    return {}, buffer_refs
        # No mapped output buffer; return data inline
        return data, []

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        outputs = job.get("outputs", {})
        
        # Resolve buffer references: inputs dict keys are logical names matching ctx._buffers
        resolved_inputs = {}
        for field_name, value in inputs.items():
            if isinstance(value, dict) and value.get("buffer"):
                # Buffer reference: {"buffer": "buf_xxx"} -> look up by logical field name
                if field_name in ctx._buffers:
                    resolved_inputs[field_name] = ctx._buffers[field_name]
                else:
                    resolved_inputs[field_name] = b""  # fallback
            else:
                resolved_inputs[field_name] = value

        # A batch is signalled either by the top-level `batch` field or by the
        # declared `batch` op, whose inputs carry `items`.
        if job.get("batch") is not None or op == "batch":
            return await self._batch(job, ctx)

        try:
            if op == "bigint_add":
                return await self._bigint_add(resolved_inputs)
            elif op == "bigint_mul":
                return await self._bigint_mul(resolved_inputs)
            elif op == "matmul":
                return await self._matmul(resolved_inputs)
            elif op == "text_stats":
                return await self._text_stats(resolved_inputs, ctx, outputs)
            elif op == "mod_pow":
                return await self._mod_pow(resolved_inputs)
            elif op == "sha256":
                return await self._sha256(resolved_inputs, ctx, outputs)
            elif op == "blake3":
                return await self._blake3(resolved_inputs, ctx, outputs)
            elif op == "json_query":
                return await self._json_query(resolved_inputs)
            elif op == "lookup":
                return await self._lookup(resolved_inputs)
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

    async def _text_stats(
        self, inputs: dict[str, Any], ctx: JobContext, outputs: dict[str, Any]
    ) -> Result:
        text_data = self._resolve_input(ctx, inputs.get("text", ""))
        if isinstance(text_data, np.ndarray):
            text = text_data.tobytes().decode("utf-8", errors="ignore")
        else:
            text = str(text_data)
        match = re.search(r"SECRET:\s*([A-Za-z0-9-]+)", text)
        checksum = hashlib.sha256(text.encode()).hexdigest()
        data = {
            "chars": len(text),
            "words": len(text.split()),
            "checksum": checksum[:16],
            "secret": match.group(1) if match else None,
        }
        inline_data, buffer_refs = self._resolve_output(ctx, "text", data, checksum, outputs)
        return Result(
            success=True,
            data=inline_data or data,
            buffer_refs=buffer_refs,
            checksum=checksum[:16],
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

    async def _sha256(
        self, inputs: dict[str, Any], ctx: JobContext, outputs: dict[str, Any]
    ) -> Result:
        data = self._resolve_input(ctx, inputs.get("data", b""))
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode()

        hash_obj = hashlib.sha256(data)
        hex_digest = hash_obj.hexdigest()

        inline_data, buffer_refs = self._resolve_output(ctx, "hash", hex_digest, hex_digest, outputs)
        # Wrap inline result in dict with output field name
        if not buffer_refs and not isinstance(inline_data, dict):
            inline_data = {"hash": inline_data}
        return Result(
            success=True,
            data=inline_data,
            buffer_refs=buffer_refs,
            checksum=hex_digest[:16],
        )

    async def _blake3(
        self, inputs: dict[str, Any], ctx: JobContext, outputs: dict[str, Any]
    ) -> Result:
        data = self._resolve_input(ctx, inputs.get("data", b""))
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode()

        try:
            import blake3

            hash_obj = blake3.blake3(data)
            hex_digest = hash_obj.hexdigest()
        except ImportError:
            # Fallback to sha256
            import hashlib as hl

            hex_digest = hl.sha256(data).hexdigest()

        inline_data, buffer_refs = self._resolve_output(ctx, "hash", hex_digest, hex_digest, outputs)
        # Wrap inline result in dict with output field name
        if not buffer_refs and not isinstance(inline_data, dict):
            inline_data = {"hash": inline_data}
        return Result(
            success=True,
            data=inline_data,
            buffer_refs=buffer_refs,
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
        items = (
            (job.get("batch") or {}).get("items") or (job.get("inputs") or {}).get("items") or []
        )
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
