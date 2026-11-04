"""Cross-turn KV reuse: injected text appends, it does not re-prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lmpu.host.hf import HFDecoder, HFEngine  # noqa: E402
from lmpu.host.metrics import EnergySample, MetricsCollector  # noqa: E402

pytestmark = pytest.mark.integration


class FakeTokenizer:
    """Word-level tokenizer with a reserved stop token."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"STOP": 1}
        self.reverse = {1: "STOP"}
        self.next_id = 10
        self.eos_token_id = None

    def _token(self, word: str) -> int:
        if word not in self.vocab:
            self.vocab[word] = self.next_id
            self.reverse[self.next_id] = word
            self.next_id += 1
        return self.vocab[word]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._token(word) for word in text.split()]

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        ids = self.encode(text)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.reverse.get(i, "") for i in ids)


class FakeModel:
    """Minimal causal model that always emits the stop token."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[int] = []
        self.vocab_size = 64

    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        logits_to_keep=None,
    ):
        fed = input_ids.shape[1]
        self.calls.append(fed)
        total = (past_key_values or 0) + fed
        logits = torch.zeros((1, 1, self.vocab_size))
        logits[0, 0, 1] = 1.0
        return SimpleNamespace(past_key_values=total, logits=logits)


class FakeEnergy:
    def sample(self) -> EnergySample:
        return EnergySample()


async def test_injected_text_appends_instead_of_reprefilling() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    engine = HFEngine(
        device="cpu",
        model=model,
        tokenizer=tokenizer,
        session_id="s_kv",
        metrics=MetricsCollector(),
        energy_sampler=FakeEnergy(),
    )
    decoder = HFDecoder(engine)

    prompt = "alpha beta"
    assert await decoder.decode_free(prompt, max_tokens=4, stop_sequence="STOP") == ""
    assert model.calls == [2, 1]

    injected = "\nGAMMA\n"
    await decoder.decode_free(prompt + injected, max_tokens=4, stop_sequence="STOP")

    # Prefill once, then one forward for the stop token, then only the
    # injected suffix. The full prompt is never fed again.
    assert model.calls == [2, 1, 1, 1]
    assert engine.stats_.total_prefill_tokens == 2
    assert engine.stats_.total_forward_passes == 4
    assert engine.stats_.total_generated_tokens == 2
    assert engine.stats_.total_injected_tokens == 1


async def test_non_prefix_prompt_forces_a_single_full_prefill() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    engine = HFEngine(
        device="cpu",
        model=model,
        tokenizer=tokenizer,
        session_id="s_kv",
        metrics=MetricsCollector(),
        energy_sampler=FakeEnergy(),
    )
    decoder = HFDecoder(engine)

    await decoder.decode_free("alpha beta", max_tokens=2, stop_sequence="STOP")
    await decoder.decode_free("a different prompt", max_tokens=2, stop_sequence="STOP")

    assert model.calls[0] == 2
    assert model.calls[2] == 3  # only the rewrite prefill
    assert engine.stats_.total_prefill_tokens == 5
