"""LMPU Host - Hugging Face reference engine implementation."""

from __future__ import annotations

from typing import Any

import torch

from .engine import HostEngine, HostState, HostStats, SleepLevel
from .grammar import ConstrainedDecoder


class HFEngine(HostEngine):
    """Hugging Face reference engine with manual decode loop."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_context: int = 32768,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.trust_remote_code = trust_remote_code
        self.max_context = max_context

        self._model = None
        self._tokenizer = None
        self._device_obj = torch.device(device)
        self.stats_ = HostStats()
        self._initialized = False

    async def _initialize(self) -> None:
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return

        # Import here to avoid dependency if not used
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Qwen3.5 is multimodal; AutoModelForCausalLM maps to the wrong class.
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        self._model = self._model.to(self._device_obj)

        self._model.eval()
        self._initialized = True

    async def prefill(self, prompt: str) -> HostState:
        await self._initialize()

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self._device_obj)
        attention_mask = inputs["attention_mask"].to(self._device_obj)

        # Prefill forward pass
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        self.stats_.total_prefill_tokens = input_ids.shape[1]
        self.stats_.total_forward_passes = 1

        return HostState(
            tokens=input_ids[0].tolist(),
            attention_mask=attention_mask[0],
            past_key_values=past_key_values,
            prefill_tokens=input_ids.shape[1],
            generated_tokens=0,
            forward_passes=1,
        )

    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        if state.parked:
            return ""

        await self._initialize()

        generated_text = ""

        # First decode after prefill: attend to the last prompt token using the
        # full cached attention mask (past + current), not a mask of length 1.
        input_ids = torch.tensor(
            [[state.tokens[-1]]],
            device=self._device_obj,
            dtype=torch.long,
        )
        if state.attention_mask is not None:
            attention_mask = state.attention_mask.to(self._device_obj).unsqueeze(0)
        else:
            attention_mask = torch.ones(
                (1, len(state.tokens)),
                device=self._device_obj,
                dtype=torch.long,
            )

        stop_token_ids = set()
        for seq in stop_sequences:
            stop_ids = self._tokenizer.encode(seq, add_special_tokens=False)
            stop_token_ids.update(stop_ids)

        # Also stop on EOS
        if self._tokenizer.eos_token_id is not None:
            stop_token_ids.add(self._tokenizer.eos_token_id)

        for _ in range(max_tokens):
            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).item()

            # Update state
            state.tokens.append(next_token_id)
            state.past_key_values = outputs.past_key_values
            state.generated_tokens += 1
            state.forward_passes += 1
            self.stats_.total_forward_passes += 1
            self.stats_.total_generated_tokens += 1

            # Check stop conditions
            if next_token_id in stop_token_ids:
                break

            # Decode new token
            token_text = self._tokenizer.decode([next_token_id], skip_special_tokens=True)
            generated_text += token_text

            # Check stop sequences in generated text
            for seq in stop_sequences:
                if seq in generated_text:
                    # Return up to the stop sequence
                    idx = generated_text.index(seq)
                    return generated_text[:idx]

            # Prepare next input
            input_ids = torch.tensor(
                [[next_token_id]],
                device=self._device_obj,
                dtype=torch.long,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(1, 1, device=self._device_obj, dtype=torch.long),
                ],
                dim=1,
            )
            state.attention_mask = attention_mask[0]

            # Context length check
            if len(state.tokens) >= self.max_context:
                break

        return generated_text

    async def park(self, state: HostState, level: str) -> None:
        """Park the host engine."""
        if level == SleepLevel.S1:
            # S1: Keep KV cache in VRAM, just stop decoding
            state.parked = True
            state.sleep_level = level
        elif level == SleepLevel.S2:
            # S2: Offload KV cache to CPU
            if state.past_key_values is not None:
                state.kv_cache_cpu = self._offload_kv_cache(state.past_key_values)
                state.past_key_values = None
                torch.cuda.empty_cache()
            state.parked = True
            state.sleep_level = level
        elif level == SleepLevel.S3:
            # S3: Full checkpoint (not implemented)
            state.parked = True
            state.sleep_level = level

    async def resume(self, state: HostState) -> None:
        """Resume the host engine."""
        if not state.parked:
            return

        if state.sleep_level == SleepLevel.S2 and state.kv_cache_cpu is not None:
            # Restore KV cache from CPU
            state.past_key_values = self._restore_kv_cache(state.kv_cache_cpu)
            state.kv_cache_cpu = None

        state.parked = False
        state.sleep_level = None

    def _offload_kv_cache(self, past_key_values) -> Any:
        """Offload KV cache to CPU memory."""
        # For DynamicCache or tuple format
        if hasattr(past_key_values, "to_legacy_cache"):
            # New DynamicCache format
            legacy = past_key_values.to_legacy_cache()
            return tuple((k.cpu(), v.cpu()) for k, v in legacy)
        elif isinstance(past_key_values, tuple):
            return tuple((k.cpu(), v.cpu()) for k, v in past_key_values)
        return past_key_values

    def _restore_kv_cache(self, cpu_kv_cache) -> Any:
        """Restore KV cache from CPU to GPU."""
        if isinstance(cpu_kv_cache, tuple):
            restored = tuple(
                (k.to(self._device_obj), v.to(self._device_obj)) for k, v in cpu_kv_cache
            )
            # Convert back to DynamicCache if needed
            from transformers.cache_utils import DynamicCache

            return DynamicCache.from_legacy_cache(restored)
        return cpu_kv_cache

    def stats(self) -> HostStats:
        return self.stats_

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()


class HFDecoder(ConstrainedDecoder):
    """Session decoder adapter for the reference HF engine.

    Each call regenerates from the supplied full prompt, so injected events are
    always visible to the model. KV reuse is within one generation call.
    """

    def __init__(self, engine: HFEngine, max_tokens: int = 1024):
        self.engine = engine
        self.max_tokens = max_tokens

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        state = await self.engine.prefill(prompt)
        return await self.engine.decode(
            state,
            max_tokens=min(max_tokens, self.max_tokens),
            stop_sequences=[stop_sequence] if stop_sequence else [],
        )

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        return await self.decode_free(prompt, max_tokens, stop_sequence)
