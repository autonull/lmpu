"""LMPU Host - Hugging Face reference engine implementation."""

from __future__ import annotations

from typing import Any

import torch

from .engine import HostEngine, HostState, HostStats, SleepLevel
from .grammar import ConstrainedDecoder, LMFormatEnforcerDecoder
from .metrics import EnergySampler, MetricsCollector, get_metrics_collector


class HFEngine(HostEngine):
    """Hugging Face reference engine with an incremental manual decode loop.

    Supports both multimodal models (Qwen3.5: AutoModelForImageTextToText)
    and pure causal LMs (Qwen2.5, etc.: AutoModelForCausalLM).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_context: int = 32768,
        model: Any = None,
        tokenizer: Any = None,
        session_id: str | None = None,
        metrics: MetricsCollector | None = None,
        energy_sampler: EnergySampler | None = None,
        model_class: str | None = None,  # "causal" | "multimodal" | None (auto)
        energy_sample_interval: int = 16,  # Sample energy every N forward passes
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.trust_remote_code = trust_remote_code
        self.max_context = max_context
        self._model_class = model_class
        self._energy_sample_interval = energy_sample_interval
        self._forward_since_sample = 0

        self._model = model
        self._tokenizer = tokenizer
        self._device_obj = torch.device(device)
        self.stats_ = HostStats()
        self._initialized = model is not None and tokenizer is not None
        self.session_id = session_id
        self.metrics = metrics or get_metrics_collector()
        self.energy_sampler = energy_sampler or EnergySampler()

    async def _initialize(self) -> None:
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return

        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )

        # Determine model class
        if self._model_class == "causal":
            model_loader = AutoModelForCausalLM
        elif self._model_class == "multimodal":
            model_loader = AutoModelForImageTextToText
        else:
            # Auto-detect: Qwen3.5 is multimodal, others assume causal
            if "Qwen3.5" in self.model_name or "Qwen2.5-VL" in self.model_name:
                model_loader = AutoModelForImageTextToText
            else:
                model_loader = AutoModelForCausalLM

        self._model = model_loader.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        self._model = self._model.to(self._device_obj)

        self._model.eval()
        self._initialized = True

    def _extend_mask(self, state: HostState, tokens: int) -> torch.Tensor:
        ones = torch.ones((1, tokens), device=self._device_obj, dtype=torch.long)
        if state.attention_mask is None:
            return ones
        return torch.cat([state.attention_mask.view(1, -1), ones], dim=1)

    def _forward(
        self,
        state: HostState,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        state.past_key_values = outputs.past_key_values
        state.logits = outputs.logits[:, -1, :]
        state.attention_mask = attention_mask[0]
        state.forward_passes += 1
        self.stats_.total_forward_passes += 1
        self.stats_.peak_context_tokens = max(self.stats_.peak_context_tokens, len(state.tokens))

    def _record(self, kind: str, tokens: int) -> None:
        """Feed one forward pass to the canonical metrics schema and trace."""
        if not self.session_id:
            return
        vram_peak_mb = 0.0
        if torch.cuda.is_available():
            vram_peak_mb = torch.cuda.max_memory_allocated(self._device_obj) / (1024.0**2)

        # Sample energy at intervals to avoid per-token syscall overhead
        self._forward_since_sample += 1
        energy = None
        if self._forward_since_sample >= self._energy_sample_interval:
            energy = self.energy_sampler.sample()
            self._forward_since_sample = 0

        self.metrics.record_forward(
            self.session_id,
            kind,
            tokens=tokens,
            energy=energy,
            vram_peak_mb=vram_peak_mb,
        )

    async def prefill(self, prompt: str) -> HostState:
        await self._initialize()

        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self._device_obj)
        attention_mask = inputs["attention_mask"].to(self._device_obj)

        state = HostState(tokens=input_ids[0].tolist())
        self._forward(state, input_ids, attention_mask)
        state.prefill_tokens = input_ids.shape[1]
        self.stats_.total_prefill_tokens += state.prefill_tokens
        self._record("prefill", state.prefill_tokens)
        return state

    async def append(self, state: HostState, text: str) -> int:
        """Append text to the live KV cache without re-prefilling the prompt."""
        if not text:
            return 0
        await self._initialize()

        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self._device_obj)
        tokens = input_ids.shape[1]
        if tokens == 0:
            return 0

        attention_mask = self._extend_mask(state, tokens)
        self._forward(state, input_ids, attention_mask)
        state.tokens.extend(input_ids[0].tolist())
        state.injected_tokens += tokens
        self.stats_.total_injected_tokens += tokens
        self._record("injected", tokens)
        return tokens

    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        if state.parked:
            return ""

        await self._initialize()

        stop_token_ids: set[int] = set()
        for seq in stop_sequences:
            stop_token_ids.update(self._tokenizer.encode(seq, add_special_tokens=False))
        if self._tokenizer.eos_token_id is not None:
            stop_token_ids.add(self._tokenizer.eos_token_id)

        generated_text = ""

        for _ in range(max_tokens):
            if state.logits is None:
                break
            next_token_id = int(torch.argmax(state.logits, dim=-1).item())
            is_stop = next_token_id in stop_token_ids

            # Forward every generated token, including a stop token, so the
            # next turn can append injected text directly to the cache.
            input_ids = torch.tensor(
                [[next_token_id]],
                device=self._device_obj,
                dtype=torch.long,
            )
            attention_mask = self._extend_mask(state, 1)
            self._forward(state, input_ids, attention_mask)
            state.tokens.append(next_token_id)
            state.generated_tokens += 1
            self.stats_.total_generated_tokens += 1
            self._record("generated", 1)

            if is_stop:
                break

            token_text = self._tokenizer.decode([next_token_id], skip_special_tokens=True)
            generated_text += token_text

            for seq in stop_sequences:
                if seq in generated_text:
                    return generated_text[: generated_text.index(seq)]

            if len(state.tokens) >= self.max_context:
                break

        return generated_text

    async def park(self, state: HostState, level: str) -> None:
        """Park the host engine."""
        if level == SleepLevel.S1:
            state.parked = True
            state.sleep_level = level
        elif level == SleepLevel.S2:
            if state.past_key_values is not None:
                state.kv_cache_cpu = self._offload_kv_cache(state.past_key_values)
                state.past_key_values = None
                torch.cuda.empty_cache()
            state.parked = True
            state.sleep_level = level
        elif level == SleepLevel.S3:
            state.parked = True
            state.sleep_level = level

    async def resume(self, state: HostState) -> None:
        """Resume the host engine."""
        if not state.parked:
            return

        if state.sleep_level == SleepLevel.S2 and state.kv_cache_cpu is not None:
            state.past_key_values = self._restore_kv_cache(state.kv_cache_cpu)
            state.kv_cache_cpu = None

        state.parked = False
        state.sleep_level = None

    def _offload_kv_cache(self, past_key_values) -> Any:
        """Offload KV cache to CPU memory."""
        if hasattr(past_key_values, "to_legacy_cache"):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFDecoder(ConstrainedDecoder):
    """Session decoder adapter with cross-turn KV reuse.

    The KV cache is the source of truth for the consumed prompt. Injected text
    is appended as a delta; a full re-prefill happens only when the prompt is
    not an extension of the text the cache has already seen.
    """

    def __init__(self, engine: HFEngine, max_tokens: int = 1024, use_constrained: bool = False):
        self.engine = engine
        self.max_tokens = max_tokens
        self._state: HostState | None = None
        self._known_text = ""
        self._use_constrained = use_constrained
        self._constrained_decoder = None

    def reset(self) -> None:
        """Discard the live cache, forcing a full prefill on the next call."""
        self._state = None
        self._known_text = ""
        self._constrained_decoder = None

    async def _advance(self, prompt: str) -> HostState:
        if self._state is None or not prompt.startswith(self._known_text):
            self._state = await self.engine.prefill(prompt)
            self._known_text = prompt
            return self._state

        suffix = prompt[len(self._known_text) :]
        if suffix:
            await self.engine.append(self._state, suffix)
            self._known_text = prompt
        return self._state

    async def _get_constrained_decoder(self):
        if self._constrained_decoder is None and self._use_constrained:
            self._constrained_decoder = LMFormatEnforcerDecoder(
                self.engine._model, self.engine._tokenizer
            )
        return self._constrained_decoder

    async def decode_free(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        state = await self._advance(prompt)
        text = await self.engine.decode(
            state,
            max_tokens=min(max_tokens, self.max_tokens),
            stop_sequences=[stop_sequence] if stop_sequence else [],
        )
        self._known_text += text
        return text

    async def decode_constrained(
        self,
        prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        stop_sequence: str,
    ) -> str:
        if self._use_constrained:
            decoder = await self._get_constrained_decoder()
            if decoder is not None:
                return await decoder.decode_constrained(prompt, schema, max_tokens, stop_sequence)
        # Fallback to free decoding
        return await self.decode_free(prompt, max_tokens, stop_sequence)
