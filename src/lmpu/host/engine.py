"""LMPU Host - Host engine abstraction and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HostState:
    """Host engine state during generation."""

    tokens: list[int] = field(default_factory=list)
    attention_mask: Any = None
    past_key_values: Any = None
    logits: Any = None
    generated_tokens: int = 0
    forward_passes: int = 0
    prefill_tokens: int = 0
    control_tokens: int = 0
    injected_tokens: int = 0
    parked: bool = False
    sleep_level: str | None = None
    kv_cache_cpu: Any = None  # For S2 offload

    @property
    def context_tokens(self) -> int:
        """Tokens currently resident in the live KV cache."""
        return len(self.tokens)


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
        if state.parked:
            return ""
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
