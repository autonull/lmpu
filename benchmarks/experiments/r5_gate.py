"""R5 gate: constrained vs free decode throughput.

Measures where constrained decoding spends its time. The hypothesis is that
lm-format-enforcer's cost is per generated token and proportional to the
allowed-character set of the current FSM state: structural tokens (keys,
numbers, booleans, enum values) restrict characters heavily and are cheap;
free-string field values allow almost every character and walk nearly the whole
token trie.

Gate from TODO.4: constrained >= 45 tok/s at >= 95% valid JSON.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 192
TRIALS = 5

SCHEMAS = {
    "numeric": {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
        "required": ["a", "b", "ok"],
    },
    "enum": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pending", "running", "done", "failed"]},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["status", "priority"],
    },
    "free_string": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
        "required": ["title", "year"],
    },
}

PROMPT = "Reply with one JSON object only, no prose.\n"


@dataclass
class Row:
    schema: str
    mode: str
    tok_per_s: float
    tokens: int
    valid: bool
    hit_cap: bool


def generate(model, tokenizer, prefix_fn=None) -> tuple[str, int, float]:
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    kwargs: dict[str, object] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if prefix_fn is not None:
        kwargs["prefix_allowed_tokens_fn"] = prefix_fn

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_ids = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids), elapsed


def _first_object_end(text: str, schema: dict) -> int | None:
    """Return the char index just past the first complete JSON object, or None.

    Constrained decoding can emit trailing whitespace or repeated structure
    after the object closes, so judge the first object rather than the tail.
    """
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if any(k not in obj for k in schema.get("required", [])):
        return None
    return start + end


def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    # Warm up CUDA kernels before any timed run.
    generate(model, tokenizer)

    rows: list[Row] = []
    for name, schema in SCHEMAS.items():
        for mode in ("free", "constrained"):
            # Rebuild the enforcer per trial: it caches per-prefix state.
            def make_fn(schema: dict = schema, constrained: bool = mode == "constrained"):
                return (
                    build_transformers_prefix_allowed_tokens_fn(
                        tokenizer_data, JsonSchemaParser(schema)
                    )
                    if constrained
                    else None
                )

            # Per-config warmup so first-call overhead is not measured.
            generate(model, tokenizer, make_fn())

            rates, tokens, valid = [], 0, 0
            hit_cap = False
            for _ in range(TRIALS):
                text, n, dt = generate(model, tokenizer, make_fn())
                end = _first_object_end(text, schema)
                useful = len(tokenizer.encode(text[:end], add_special_tokens=False)) if end else n
                rates.append(useful / dt)
                tokens = useful
                valid += int(end is not None)
                hit_cap = hit_cap or (n >= MAX_NEW_TOKENS and end is None)
            rows.append(
                Row(
                    schema=name,
                    mode=mode,
                    tok_per_s=statistics.median(rates),
                    tokens=tokens,
                    valid=valid == TRIALS,
                    hit_cap=hit_cap,
                )
            )

    print(f"\n{'schema':<12} {'mode':<12} {'tok/s':>8} {'tokens':>7} {'valid':>6} {'cap':>5}")
    print("-" * 56)
    for r in rows:
        print(
            f"{r.schema:<12} {r.mode:<12} {r.tok_per_s:>8.1f} {r.tokens:>7} "
            f"{str(r.valid):>6} {str(r.hit_cap):>5}"
        )

    constrained = [r for r in rows if r.mode == "constrained"]
    passed = all(r.tok_per_s >= 45 and r.valid for r in constrained)
    free = {r.schema: r.tok_per_s for r in rows if r.mode == "free"}

    print("-" * 56)
    print("overhead vs free decode:")
    for r in constrained:
        print(
            f"  {r.schema:<12} {r.tok_per_s:>6.1f} tok/s  "
            f"{free[r.schema] / r.tok_per_s:>4.2f}x slower than free ({free[r.schema]:.1f})"
        )
    print("-" * 56)
    print("GATE (>=45 tok/s, 100% valid):", "PASS" if passed else "FAIL")

    out_dir = Path("benchmarks/results/r5_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "throughput.json").write_text(
        json.dumps(
            {
                "gate_passed": passed,
                "free_tok_per_s": free,
                "overhead_vs_free": {
                    r.schema: round(free[r.schema] / r.tok_per_s, 2) for r in constrained
                },
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir / 'throughput.json'}")


if __name__ == "__main__":
    main()
