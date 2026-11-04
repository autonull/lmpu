"""Parked-session claim, measured against a real model.

Replaces the sleep-based multisession probe. One 0.5B model on one GPU, four
sessions sharing it. Three scenarios:

  solo      one session decodes; nothing else exists
  parked    one session decodes; three others are prefilled and parked (S1)
  active    four sessions decode concurrently

Claims under test:
  1. A parked session performs zero forward passes.
  2. Parked sessions do not slow the decoding session.
  3. Concurrent decoding does slow it (the contrast that makes (2) meaningful).
  4. S2 parking releases KV VRAM back toward the empty-model baseline.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmpu.host.engine import SleepLevel
from lmpu.host.hf import HFEngine

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 32
TRIALS = 3
PROMPT = "Count upward from 1, separated by commas. Start: 1,"


def make_engine(model, tokenizer, sid: str) -> HFEngine:
    return HFEngine(
        model_name=MODEL,
        model=model,
        tokenizer=tokenizer,
        session_id=sid,
        device="cuda",
    )


async def solo(engines: dict[str, HFEngine]) -> tuple[float, int]:
    a = engines["a"]
    state = await a.prefill(PROMPT)
    t0 = time.perf_counter()
    await a.decode(state, MAX_TOKENS, [])
    wall = time.perf_counter() - t0
    return wall, a.stats_.total_forward_passes


async def parked(engines: dict[str, HFEngine]) -> tuple[float, int, dict[str, int]]:
    others = {}
    for name in ("b", "c", "d"):
        eng = engines[name]
        state = await eng.prefill(PROMPT)
        await eng.park(state, SleepLevel.S1)
        others[name] = state  # keep alive so KV stays resident

    a = engines["a"]
    state = await a.prefill(PROMPT)
    before = {n: e.stats_.total_forward_passes for n, e in engines.items() if n != "a"}
    t0 = time.perf_counter()
    await a.decode(state, MAX_TOKENS, [])
    wall = time.perf_counter() - t0
    after = {n: e.stats_.total_forward_passes for n, e in engines.items() if n != "a"}
    forwarded = {n: after[n] - before[n] for n in after}
    del others
    return wall, a.stats_.total_forward_passes, forwarded


async def active(engines: dict[str, HFEngine]) -> tuple[float, dict[str, int]]:
    states = {}
    for name, eng in engines.items():
        states[name] = await eng.prefill(PROMPT)
    for eng in engines.values():
        eng.stats_.total_forward_passes = 0

    t0 = time.perf_counter()
    await asyncio.gather(*(eng.decode(states[n], MAX_TOKENS, []) for n, eng in engines.items()))
    wall = time.perf_counter() - t0
    return wall, {n: e.stats_.total_forward_passes for n, e in engines.items()}


async def main() -> None:
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    names = ("a", "b", "c", "d")

    def fresh() -> dict[str, HFEngine]:
        return {n: make_engine(model, tokenizer, n) for n in names}

    results: dict[str, object] = {}

    # Warm up kernels before timing anything.
    warm = make_engine(model, tokenizer, "warm")
    await warm.decode(await warm.prefill(PROMPT), 8, [])

    solo_walls, solo_forwards = [], []
    for _ in range(TRIALS):
        w, f = await solo(fresh())
        solo_walls.append(w)
        solo_forwards.append(f)

    parked_walls, parked_forwards, parked_others = [], [], []
    for _ in range(TRIALS):
        w, f, others = await parked(fresh())
        parked_walls.append(w)
        parked_forwards.append(f)
        parked_others.append(others)

    active_walls, active_forwards = [], []
    for _ in range(TRIALS):
        w, f = await active(fresh())
        active_walls.append(w)
        active_forwards.append(f)

    solo_w = statistics.median(solo_walls)
    parked_w = statistics.median(parked_walls)
    active_w = statistics.median(active_walls)
    results = {
        "solo": {"wall_s": solo_w, "forwards": statistics.median(solo_forwards)},
        "parked": {"wall_s": parked_w, "forwards": statistics.median(parked_forwards),
                   "other_forwards": parked_others[-1]},
        "active": {"wall_s": active_w, "forwards": active_forwards[-1]},
    }

    print(f"\n{'scenario':<10} {'wall (s)':>9} {'decoder fwd':>12} {'other fwd':>20}")
    print("-" * 56)
    print(f"{'solo':<10} {solo_w:>9.3f} {results['solo']['forwards']:>12}")
    print(f"{'parked':<10} {parked_w:>9.3f} {results['parked']['forwards']:>12} "
          f"{str(parked_others[-1]):>20}")
    print(f"{'active':<10} {active_w:>9.3f} {str(active_forwards[-1]):>12}")

    parked_zero = all(v == 0 for v in parked_others[-1].values())
    no_slowdown = parked_w <= solo_w * 1.15
    contention = active_w > solo_w * 1.5
    print("-" * 56)
    print(f"parked sessions performed zero forwards : {parked_zero}")
    print(f"parked sessions did not slow the decoder: {no_slowdown} "
          f"({solo_w:.3f}s -> {parked_w:.3f}s)")
    print(f"concurrent decoders contend for the GPU: {contention} "
          f"({solo_w:.3f}s -> {active_w:.3f}s, {active_w / solo_w:.2f}x)")

    # VRAM: S1 keeps KV resident, S2 offloads it. A short prompt has ~2 MiB of
    # KV at 0.5B, below measurement resolution, so use a 4k-token context.
    eng = make_engine(model, tokenizer, "vram")
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 400
    long_tokens = len(tokenizer(long_prompt).input_ids)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2**20
    states = [await eng.prefill(long_prompt) for _ in range(4)]
    torch.cuda.synchronize()
    with_kv = torch.cuda.memory_allocated() / 2**20
    for st in states:
        await eng.park(st, SleepLevel.S2)
    torch.cuda.synchronize()
    after_s2 = torch.cuda.memory_allocated() / 2**20

    kv_mib = with_kv - base
    print(
        f"\nVRAM (MiB, {long_tokens} tok/ctx): base={base:.1f}  +4 live KV={with_kv:.1f} "
        f"(+{kv_mib:.1f})  after S2 park={after_s2:.1f}"
    )
    print(f"S2 park returns VRAM to base: {after_s2 <= base + 8.0}")

    results["vram_mib"] = {
        "context_tokens": long_tokens,
        "base": base,
        "with_4_sessions": with_kv,
        "after_s2_park": after_s2,
    }

    import json

    out = Path("benchmarks/results/parked_sessions")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out / 'result.json'}")


if __name__ == "__main__":
    asyncio.run(main())
