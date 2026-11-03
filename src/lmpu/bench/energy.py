"""Idle GPU energy baseline for benchmark runs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from lmpu.host.metrics import EnergySampler


@dataclass(frozen=True, slots=True)
class IdleBaseline:
    """Power/energy observed while the GPU is idle."""

    available: bool
    seconds: float
    samples: int
    gpu_joules: float
    average_watts: float | None
    method: str = "nvml"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "seconds": self.seconds,
            "samples": self.samples,
            "gpu_joules": self.gpu_joules,
            "average_watts": self.average_watts,
            "method": self.method,
            "error": self.error,
        }


class EnergyProbe:
    """Integrate NVML power samples into an idle baseline."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self._sampler = EnergySampler()

    @property
    def available(self) -> bool:
        return self._sampler.available

    async def measure_idle(self, seconds: float) -> IdleBaseline:
        """Measure idle GPU energy for ``seconds``; zero seconds reports availability only."""
        if seconds <= 0:
            return IdleBaseline(
                available=self.available,
                seconds=0.0,
                samples=0,
                gpu_joules=0.0,
                average_watts=None,
            )
        if not self.available:
            return IdleBaseline(
                available=False,
                seconds=0.0,
                samples=0,
                gpu_joules=0.0,
                average_watts=None,
                error="NVML unavailable",
            )

        self._sampler.sample()
        started = time.monotonic()
        total = 0.0
        samples = 0
        while True:
            remaining = seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            await asyncio.sleep(min(self.interval_seconds, remaining))
            total += self._sampler.sample().gpu_joules
            samples += 1
        duration = time.monotonic() - started
        return IdleBaseline(
            available=True,
            seconds=duration,
            samples=samples,
            gpu_joules=total,
            average_watts=(total / duration) if duration > 0 else None,
        )

    def close(self) -> None:
        self._sampler.close()
