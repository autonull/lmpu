"""LMPU Host - Canonical host metrics schema and collection.

One schema is the source of truth for host-side accounting. ``HostMetrics.to_dict``
is the payload emitted to the trace sink and consumed by the dashboard, energy
reporting, and benchmark harnesses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from lmpu.kernel.trace import get_trace_sink

FORWARD_KINDS = ("prefill", "generated", "injected", "control")


def estimate_tokens(payload: object) -> int:
    """Conservative token estimate for rendered payloads."""
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, separators=(",", ":"), default=str)
    )
    return max(1, len(text) // 4)


@dataclass
class EnergySample:
    """Joules accumulated between two sampler reads."""

    gpu_joules: float = 0.0
    cpu_joules: float = 0.0


class EnergySampler:
    """Optional NVML-backed energy accumulator.

    ``available`` is false when NVML or a GPU handle is missing; callers must
    treat a zero sample as "not measured" and report the limitation.
    """

    def __init__(self) -> None:
        self._pynvml: Any = None
        self._handle: Any = None
        self._last_power_w: float | None = None
        self._last_mono: float | None = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._handle = None
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _power_watts(self) -> float | None:
        if not self.available:
            return None
        try:
            milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return milliwatts / 1000.0
        except Exception:
            return None

    def sample(self) -> EnergySample:
        """Return energy accumulated since the previous call."""
        now = time.monotonic()
        power = self._power_watts()
        joules = 0.0
        if power is not None and self._last_power_w is not None and self._last_mono is not None:
            joules = power * (now - self._last_mono)
        self._last_power_w = power
        self._last_mono = now
        return EnergySample(gpu_joules=joules)

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._pynvml = None
        self._handle = None


@dataclass
class HostMetrics:
    """Canonical host metrics for one session."""

    session_id: str
    forward_passes: int = 0
    prefill_tokens: int = 0
    generated_tokens: int = 0
    control_tokens: int = 0
    injected_tokens: int = 0
    peak_context_tokens: int = 0
    gpu_energy_joules: float = 0.0
    cpu_energy_joules: float = 0.0
    vram_peak_mb: float = 0.0
    sleep_duration_seconds: float = 0.0
    wake_latency_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    sleep_start_time: float | None = None

    def _peak_context(self) -> int:
        return (
            self.prefill_tokens + self.generated_tokens + self.control_tokens + self.injected_tokens
        )

    def record_forward(self, kind: str, tokens: int = 1) -> None:
        """Record one forward pass under the given token kind."""
        if kind not in FORWARD_KINDS:
            raise ValueError(f"Unknown forward kind {kind!r}")
        self.forward_passes += 1
        if kind == "prefill":
            self.prefill_tokens += tokens
        elif kind == "generated":
            self.generated_tokens += tokens
        elif kind == "injected":
            self.injected_tokens += tokens
        else:
            self.control_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self._peak_context())

    def record_prefill(self, tokens: int) -> None:
        self.record_forward("prefill", tokens)

    def record_control_tokens(self, tokens: int) -> None:
        self.control_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self._peak_context())

    def record_injected_tokens(self, tokens: int) -> None:
        self.injected_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self._peak_context())

    def record_energy(self, gpu_joules: float = 0.0, cpu_joules: float = 0.0) -> None:
        self.gpu_energy_joules += gpu_joules
        self.cpu_energy_joules += cpu_joules

    def start_sleep(self) -> None:
        self.sleep_start_time = time.time()

    def end_sleep(self) -> None:
        if self.sleep_start_time:
            self.sleep_duration_seconds += time.time() - self.sleep_start_time
            self.sleep_start_time = None

    def record_wake_latency(self, latency: float) -> None:
        self.wake_latency_seconds = latency

    def to_dict(self) -> dict[str, Any]:
        """Emit the canonical schema. This is the only metrics payload."""
        return {
            "schema": "lmpu.host_metrics.v1",
            "session_id": self.session_id,
            "forward_passes": self.forward_passes,
            "prefill_tokens": self.prefill_tokens,
            "generated_tokens": self.generated_tokens,
            "control_tokens": self.control_tokens,
            "injected_tokens": self.injected_tokens,
            "total_tokens": self._peak_context(),
            "peak_context_tokens": self.peak_context_tokens,
            "vram_peak_mb": self.vram_peak_mb,
            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,
            "sleep_duration_seconds": self.sleep_duration_seconds,
            "wake_latency_seconds": self.wake_latency_seconds,
            "wall_time_seconds": time.time() - self.start_time,
        }


class MetricsCollector:
    """Collects host metrics and forwards each observation to the trace sink."""

    def __init__(self) -> None:
        self._metrics: dict[str, HostMetrics] = {}
        self._global_start = time.time()

    def get_or_create(self, session_id: str) -> HostMetrics:
        if session_id not in self._metrics:
            self._metrics[session_id] = HostMetrics(session_id)
        return self._metrics[session_id]

    def record_forward(
        self,
        session_id: str,
        kind: str,
        tokens: int = 1,
        phase: str = "",
        energy: EnergySample | None = None,
        vram_peak_mb: float = 0.0,
    ) -> HostMetrics:
        """Record a forward pass and emit one host_forward trace record."""
        metrics = self.get_or_create(session_id)
        metrics.record_forward(kind, tokens)
        metrics.vram_peak_mb = max(metrics.vram_peak_mb, vram_peak_mb)
        if energy is not None:
            metrics.record_energy(energy.gpu_joules, energy.cpu_joules)

        trace = get_trace_sink()
        if trace is not None:
            trace.emit_host_forward(session_id, tokens, phase or kind)
        return metrics

    def flush(self, session_id: str) -> dict[str, Any] | None:
        """Emit the canonical metrics payload once for a session."""
        metrics = self._metrics.get(session_id)
        if metrics is None:
            return None
        payload = metrics.to_dict()
        trace = get_trace_sink()
        if trace is not None:
            trace.emit_host_metrics(session_id, payload)
        return payload

    def remove(self, session_id: str) -> HostMetrics | None:
        return self._metrics.pop(session_id, None)

    def get_all(self) -> list[HostMetrics]:
        return list(self._metrics.values())

    def aggregate(self) -> dict[str, Any]:
        total_forward = sum(m.forward_passes for m in self._metrics.values())
        total_tokens = sum(m._peak_context() for m in self._metrics.values())
        total_sleep = sum(m.sleep_duration_seconds for m in self._metrics.values())
        total_gpu = sum(m.gpu_energy_joules for m in self._metrics.values())
        total_cpu = sum(m.cpu_energy_joules for m in self._metrics.values())

        return {
            "sessions": len(self._metrics),
            "total_forward_passes": total_forward,
            "total_tokens": total_tokens,
            "total_sleep_seconds": total_sleep,
            "total_gpu_energy_joules": total_gpu,
            "total_cpu_energy_joules": total_cpu,
            "uptime_seconds": time.time() - self._global_start,
        }


_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
