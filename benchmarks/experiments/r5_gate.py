"""R5 Gate: Constrained decode throughput benchmark.

Measures decoders on JSON schema constrained generation.

Note: OutlinesDecoder requires `outlines` package with outlines-core (Rust build),
which fails in this environment. Code is implemented but untested.

Target: ≥45 tok/s at ≥95% valid JSON.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmpu.host.grammar import LMFormatEnforcerDecoder, FallbackDecoder

# OutlinesDecoder import (may fail if outlines not available)
try:
    from lmpu.host.grammar import OutlinesDecoder
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    OutlinesDecoder = None  # type: ignore


@dataclass
class BenchmarkResult:
    decoder: str
    schema_name: str
    total_tokens: int
    valid_json: int
    total_time_s: float
    tok_per_s: float
    validity_rate: float
    avg_latency_ms: float


# Test schemas of varying complexity
SCHEMAS = {
    "simple": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    },
    "nested": {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "email", "age"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "active": {"type": "boolean"},
        },
        "required": ["user", "tags", "active"],
    },
    "enum": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
            "metadata": {"type": "object"},
        },
        "required": ["status", "priority"],
    },
    "complex": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "price": {"type": "number"},
                    },
                    "required": ["name", "quantity", "price"],
                },
            },
            "total": {"type": "number"},
            "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        },
        "required": ["id", "items", "total", "currency"],
    },
}


PROMPT = """Generate a valid JSON object matching the schema.

{
"""


async def run_decoder_benchmark(
    decoder,
    schema: dict[str, Any],
    model,
    tokenizer,
    max_tokens: int = 256,
    num_trials: int = 10,
) -> BenchmarkResult:
    """Run multiple trials and aggregate."""
    valid_count = 0
    total_tokens = 0
    total_time = 0.0

    for i in range(num_trials):
        start = time.perf_counter()
        try:
            # For constrained decoding, don't use stop_sequence - let it complete naturally
            result = await decoder.decode_constrained(
                PROMPT,
                schema,
                max_tokens=max_tokens,
                stop_sequence="",
            )
            elapsed = time.perf_counter() - start

            # Validate JSON
            try:
                parsed = json.loads(result)
                # Check required fields
                required = schema.get("required", [])
                if all(field in parsed for field in required):
                    valid_count += 1
            except json.JSONDecodeError:
                pass

            # Count tokens
            tokens = len(tokenizer.encode(result))
            total_tokens += tokens
            total_time += elapsed

        except Exception as e:
            elapsed = time.perf_counter() - start
            total_time += elapsed
            continue

    validity_rate = valid_count / num_trials if num_trials > 0 else 0
    tok_per_s = total_tokens / total_time if total_time > 0 else 0
    avg_latency_ms = (total_time / num_trials) * 1000 if num_trials > 0 else 0

    return BenchmarkResult(
        decoder=decoder.__class__.__name__,
        schema_name="",
        total_tokens=total_tokens,
        valid_json=valid_count,
        total_time_s=total_time,
        tok_per_s=tok_per_s,
        validity_rate=validity_rate,
        avg_latency_ms=avg_latency_ms,
    )


async def main():
    # Load model
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for benchmarking
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.to("cuda")
    model.eval()

    # Test decoders
    decoders = [
        ("LMFormatEnforcerDecoder", LMFormatEnforcerDecoder(model, tokenizer)),
        ("FallbackDecoder", FallbackDecoder(model, tokenizer, max_retries=1)),
    ]

    if OUTLINES_AVAILABLE:
        decoders.insert(0, ("OutlinesDecoder", OutlinesDecoder(model, tokenizer)))

    all_results = []

    for schema_name, schema in SCHEMAS.items():
        print(f"\n=== Schema: {schema_name} ===")
        for decoder_name, decoder in decoders:
            print(f"  {decoder_name}...", end=" ", flush=True)
            result = await run_decoder_benchmark(decoder, schema, model, tokenizer)
            result.schema_name = schema_name
            all_results.append(result)
            print(f"tok/s={result.tok_per_s:.1f} valid={result.validity_rate:.0%} latency={result.avg_latency_ms:.1f}ms")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Decoder':<30} {'Schema':<10} {'tok/s':>8} {'Valid':>7} {'Latency':>10}")
    print("-" * 90)
    for r in all_results:
        print(f"{r.decoder:<30} {r.schema_name:<10} {r.tok_per_s:>8.1f} {r.validity_rate:>6.0%} {r.avg_latency_ms:>9.1f}ms")

    # Gate check
    print("\n" + "=" * 90)
    print("R5 GATE CHECK: ≥45 tok/s at ≥95% valid JSON")
    print("-" * 90)
    
    for decoder_name, _ in decoders:
        decoder_results = [r for r in all_results if r.decoder == decoder_name]
        all_pass = all(r.tok_per_s >= 45 and r.validity_rate >= 0.95 for r in decoder_results)
        status = "✅ PASS" if all_pass else "❌ FAIL"
        avg_tok = sum(r.tok_per_s for r in decoder_results) / len(decoder_results)
        avg_valid = sum(r.validity_rate for r in decoder_results) / len(decoder_results)
        note = ""
        if decoder_name == "OutlinesDecoder" and not OUTLINES_AVAILABLE:
            note = " (NOT TESTED - outlines-core build fails)"
        print(f"  {decoder_name}: {status} (avg tok/s={avg_tok:.1f}, avg valid={avg_valid:.0%}){note}")

    if not OUTLINES_AVAILABLE:
        print("\nNOTE: OutlinesDecoder code is implemented in src/lmpu/host/grammar.py")
        print("      but cannot be tested because outlines-core (Rust) fails to build in this environment.")
        print("      In a proper build environment, Outlines FSM-based decoding should achieve")
        print("      O(1) per-token vs lm-format-enforcer's O(n²) prefix parsing.")

    # Save
    output_dir = Path("benchmarks/results/r5_gate")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "throughput.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())