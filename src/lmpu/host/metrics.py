"""LMPU Host - Metrics collection for host engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class HostMetrics:
    """Host engine metrics."""
    session_id: str
    forward_passes: int = 0
    prefill_tokens: int = 0
    generated_tokens: int = 0
    control_tokens: int = 0
    injected_tokens: int = 0
    peak_context_tokens: int = 0
    gpu_energy_joules: float = 0.0
    cpu_energy_joules: float = 0.0
    sleep_duration_seconds: float = 0.0
    wake_latency_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    sleep_start_time: Optional[float] = None

    def record_forward_pass(self, tokens: int = 1) -> None:
        self.forward_passes += 1
        self.generated_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.generated_tokens + self.prefill_tokens)

    def record_prefill(self, tokens: int) -> None:
        self.prefill_tokens = tokens
        self.peak_context_tokens = max(self.peak_context_tokens, tokens)

    def record_control_tokens(self, tokens: int) -> None:
        self.control_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.generated_tokens + self.prefill_tokens + self.control_tokens)

    def record_injected_tokens(self, tokens: int) -> None:
        self.injected_tokens += tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.generated_tokens + self.prefill_tokens + self.control_tokens + self.injected_tokens)

    def start_sleep(self) -> None:
        self.sleep_start_time = time.time()

    def end_sleep(self) -> None:
        if self.sleep_start_time:
            self.sleep_duration_seconds += time.time() - self.sleep_start_time
            self.sleep_start_time = None

    def record_wake_latency(self, latency: float) -> None:
        self.wake_latency_seconds = latency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "forward_passes": self.forward_passes,
            "prefill_tokens": self.prefill_tokens,
            "generated_tokens": self.generated_tokens,
            "control_tokens": self.control_tokens,
            "injected_tokens": self.injected_tokens,
            "total_tokens": self.prefill_tokens + self.generated_tokens + self.control_tokens + self.injected_tokens,
            "peak_context_tokens": self.peak_context_tokens,
            "gpu_energy_joules": self.gpu_energy_joules,
            "cpu_energy_joules": self.cpu_energy_joules,
            "sleep_duration_seconds": self.sleep_duration_seconds,
            "wake_latency_seconds": self.wake_latency_seconds,
            "uptime_seconds": time.time() - self.start_time,
        }


class MetricsCollector:
    """Collects and aggregates host metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, HostMetrics] = {}
        self._global_start = time.time()

    def get_or_create(self, session_id: str) -> HostMetrics:
        if session_id not in self._metrics:
            self._metrics[session_id] = HostMetrics(session_id)
        return self._metrics[session_id]

    def remove(self, session_id: str) -> Optional[HostMetrics]:
        return self._metrics.pop(session_id, None)

    def get_all(self) -> List[HostMetrics]:
        return list(self._metrics.values())

    def aggregate(self) -> Dict[str, Any]:
        total_forward = sum(m.forward_passes for m in self._metrics.values())
        total_tokens = sum(m.prefill_tokens + m.generated_tokens + m.control_tokens + m.injected_tokens 
                          for m in self._metrics.values())
        total_sleep = sum(m.sleep_duration_seconds for m in self._metrics.values())
        
        return {
            "sessions": len(self._metrics),
            "total_forward_passes": total_forward,
            "total_tokens": total_tokens,
            "total_sleep_seconds": total_sleep,
            "uptime_seconds": time.time() - self._global_start,
        }


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector