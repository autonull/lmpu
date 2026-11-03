"""LMPU Host - Host engine abstraction and implementations."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class HostState:
    """Host engine state during generation."""
    tokens: list[int] = field(default_factory=list)
    attention_mask: Optional[np.ndarray] = None
    past_key_values: Any = None
    generated_tokens: int = 0
    forward_passes: int = 0
    prefill_tokens: int = 0
    control_tokens: int = 0
    injected_tokens: int = 0
    parked: bool = False
    sleep_level: Optional[str] = None
    kv_cache_cpu: Optional[Any] = None  # For S2 offload


@dataclass
class HostStats:
    """Host engine statistics."""
    total_forward_passes: int = 0
    total_generated_tokens: int = 0
    total_prefill_tokens: int = 0
    total_control_tokens: int = 0
    total_injected_tokens: int = 0
    peak_context_tokens: int = 0
    gpu_energy_joules: float = 0.0
    cpu_energy_joules: float = 0.0


class SleepLevel:
    """Sleep/park levels."""
    S1 = "S1"  # KV resident in VRAM
    S2 = "S2"  # KV offloaded to CPU
    S3 = "S3"  # Checkpointed to disk


class HostEngine(ABC):
    """Abstract host engine interface."""

    @abstractmethod
    async def prefill(self, prompt: str) -> HostState:
        """Prefill the model with initial prompt."""
        pass

    @abstractmethod
    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        """Generate tokens until stop sequence or max_tokens."""
        pass

    @abstractmethod
    async def park(self, state: HostState, level: str) -> None:
        """Park the host (stop generation, preserve state)."""
        pass

    @abstractmethod
    async def resume(self, state: HostState) -> None:
        """Resume the host from parked state."""
        pass

    @abstractmethod
    def stats(self) -> HostStats:
        """Get engine statistics."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class ScriptedEngine(HostEngine):
    """Deterministic scripted engine for testing."""
    
    def __init__(self, script: list[dict[str, Any]]):
        self.script = script
        self.position = 0
        self.stats_ = HostStats()
    
    async def prefill(self, prompt: str) -> HostState:
        self.stats_.total_prefill_tokens = len(prompt.split())
        return HostState(
            tokens=list(range(self.stats_.total_prefill_tokens)),
            prefill_tokens=self.stats_.total_prefill_tokens,
        )
    
    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        if self.position >= len(self.script):
            return ""
        
        segment = self.script[self.position]
        self.position += 1
        
        if segment.get("type") == "free":
            text = segment.get("text", "")
            self.stats_.total_generated_tokens += len(text.split())
            return text
        elif segment.get("type") == "control":
            text = segment.get("control", "")
            self.stats_.total_control_tokens += len(text.split())
            return text
        elif segment.get("type") == "injected":
            text = segment.get("event", "")
            self.stats_.total_injected_tokens += len(text.split())
            return text
        
        return ""
    
    async def park(self, state: HostState, level: str) -> None:
        state.parked = True
        state.sleep_level = level
    
    async def resume(self, state: HostState) -> None:
        state.parked = False
        state.sleep_level = None
    
    def stats(self) -> HostStats:
        return self.stats_
    
    async def close(self) -> None:
        pass


# Placeholder for HF and vLLM engines (implemented in separate files)
class HFEngine(HostEngine):
    """Hugging Face reference engine."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._device = kwargs.get("device", "cuda")
        self._dtype = kwargs.get("dtype", "bfloat16")
        self.stats_ = HostStats()
    
    async def prefill(self, prompt: str) -> HostState:
        raise NotImplementedError("HFEngine requires transformers and torch")
    
    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        raise NotImplementedError("HFEngine requires transformers and torch")
    
    async def park(self, state: HostState, level: str) -> None:
        state.parked = True
        state.sleep_level = level
    
    async def resume(self, state: HostState) -> None:
        state.parked = False
        state.sleep_level = None
    
    def stats(self) -> HostStats:
        return self.stats_
    
    async def close(self) -> None:
        pass


class VLLMEngine(HostEngine):
    """vLLM engine with parking adapter."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.stats_ = HostStats()
    
    async def prefill(self, prompt: str) -> HostState:
        raise NotImplementedError("VLLMEngine requires vLLM")
    
    async def decode(
        self,
        state: HostState,
        max_tokens: int,
        stop_sequences: list[str],
    ) -> str:
        raise NotImplementedError("VLLMEngine requires vLLM")
    
    async def park(self, state: HostState, level: str) -> None:
        state.parked = True
        state.sleep_level = level
    
    async def resume(self, state: HostState) -> None:
        state.parked = False
        state.sleep_level = None
    
    def stats(self) -> HostStats:
        return self.stats_
    
    async def close(self) -> None:
        pass