"""Measure the reference Qwen3.5-0.8B HF engine baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import torch

from lmpu.host.hf import HFEngine


async def run(model_name: str, prompt: str, max_tokens: int) -> dict[str, object]:
    engine = HFEngine(model_name=model_name, device="cuda", dtype="bfloat16")
    try:
        started = time.perf_counter()
        state = await engine.prefill(prompt)
        prefill_s = time.perf_counter() - started

        started = time.perf_counter()
        text = await engine.decode(state, max_tokens=max_tokens, stop_sequences=[])
        decode_s = time.perf_counter() - started

        generated = state.generated_tokens
        return {
            "model": model_name,
            "model_class": type(engine._model).__name__,
            "tokenizer_class": type(engine._tokenizer).__name__,
            "prompt_tokens": state.prefill_tokens,
            "generated_tokens": generated,
            "prefill_seconds": round(prefill_s, 3),
            "decode_seconds": round(decode_s, 3),
            "tokens_per_second": round(generated / decode_s, 2) if decode_s else 0.0,
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            "text": text,
        }
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Write a list of numbers: 1, 2, 3,")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.model, args.prompt, args.max_tokens)), indent=2))


if __name__ == "__main__":
    main()
