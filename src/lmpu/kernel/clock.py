"""LMPU Kernel - Clock abstraction for testing."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import heapq


class Clock:
    """Abstract clock interface."""

    def now(self) -> float:
        raise NotImplementedError

    def monotonic(self) -> float:
        raise NotImplementedError

    async def sleep(self, delay: float) -> None:
        raise NotImplementedError

    def call_later(self, delay: float, callback: Callable, *args: Any) -> Any:
        raise NotImplementedError

    def cancel(self, handle: Any) -> None:
        raise NotImplementedError


class RealClock(Clock):
    """Real system clock."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, delay: float) -> None:
        await asyncio.sleep(delay)

    def call_later(self, delay: float, callback: Callable, *args: Any) -> asyncio.TimerHandle:
        loop = asyncio.get_event_loop()
        return loop.call_later(delay, callback, *args)

    def cancel(self, handle: asyncio.TimerHandle) -> None:
        handle.cancel()


class FakeClock(Clock):
    """Deterministic fake clock for testing."""

    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time
        self._monotonic = initial_time
        self._timers: list[tuple[float, int, Callable, tuple]] = []
        self._timer_counter = 0
        self._callbacks: list[Callable] = []

    def now(self) -> float:
        return self._time

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, delta: float) -> None:
        """Advance time by delta seconds, firing timers."""
        target = self._monotonic + delta
        while self._timers and self._timers[0][0] <= target:
            due, _, callback, args, handle = heapq.heappop(self._timers)
            if handle.cancelled:
                continue
            self._monotonic = due
            self._time = due
            callback(*args)
        self._monotonic = target
        self._time = target

    async def sleep(self, delay: float) -> None:
        """Sleep by advancing fake time."""
        self.advance(delay)

    def call_later(self, delay: float, callback: Callable, *args: Any) -> "FakeTimerHandle":
        when = self._monotonic + delay
        self._timer_counter += 1
        handle = FakeTimerHandle(self, when, self._timer_counter, callback, args)
        heapq.heappush(self._timers, (when, self._timer_counter, callback, args, handle))
        return handle

    def cancel(self, handle: "FakeTimerHandle") -> None:
        handle.cancel()


@dataclass
class FakeTimerHandle:
    """Timer handle for fake clock."""

    clock: FakeClock
    when: float
    counter: int
    callback: Callable
    args: tuple
    cancelled: bool = False

    def cancel(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            # Remove from clock's timer heap (lazy removal)
            # The timer will be skipped when it fires


# Global clock instance (real by default)
_clock: Clock = RealClock()


def get_clock() -> Clock:
    return _clock


def set_clock(clock: Clock) -> None:
    global _clock
    _clock = clock


async def sleep(delay: float) -> None:
    await _clock.sleep(delay)


def call_later(delay: float, callback: Callable, *args: Any) -> Any:
    return _clock.call_later(delay, callback, *args)


def cancel_timer(handle: Any) -> None:
    _clock.cancel(handle)
