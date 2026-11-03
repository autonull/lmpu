"""LMPU Kernel - Quota management and admission control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..protocol.messages import ErrorCode
from ..protocol.errors import QuotaExceededError


@dataclass
class QuotaConfig:
    """Per-session quota configuration."""
    max_outstanding_jobs: int = 32
    max_buffer_bytes: int = 2_147_483_648  # 2 GiB
    max_sleep_seconds: int = 300
    max_injected_tokens_per_turn: int = 2048
    max_injected_bytes_per_event: int = 65536
    max_sleep_registrations: int = 1


@dataclass
class SessionQuotaState:
    """Current quota usage for a session."""
    session_id: str
    outstanding_jobs: int = 0
    buffer_bytes: int = 0
    sleep_registrations: int = 0
    cpu_seconds_used: float = 0.0
    gpu_seconds_used: float = 0.0
    wall_time_used: float = 0.0
    last_update: float = field(default_factory=time.time)


class QuotaManager:
    """Manages per-session and global quotas."""

    def __init__(self, global_quotas: Optional[QuotaConfig] = None):
        self.global_quotas = global_quotas or QuotaConfig()
        self._session_quotas: dict[str, QuotaConfig] = {}
        self._session_state: dict[str, SessionQuotaState] = {}

    def set_session_quotas(self, session_id: str, quotas: QuotaConfig) -> None:
        """Set custom quotas for a session (cannot exceed global)."""
        # Clamp to global limits
        quotas.max_outstanding_jobs = min(quotas.max_outstanding_jobs, self.global_quotas.max_outstanding_jobs)
        quotas.max_buffer_bytes = min(quotas.max_buffer_bytes, self.global_quotas.max_buffer_bytes)
        quotas.max_sleep_seconds = min(quotas.max_sleep_seconds, self.global_quotas.max_sleep_seconds)
        quotas.max_injected_tokens_per_turn = min(
            quotas.max_injected_tokens_per_turn,
            self.global_quotas.max_injected_tokens_per_turn,
        )
        quotas.max_injected_bytes_per_event = min(
            quotas.max_injected_bytes_per_event,
            self.global_quotas.max_injected_bytes_per_event,
        )
        quotas.max_sleep_registrations = min(
            quotas.max_sleep_registrations,
            self.global_quotas.max_sleep_registrations,
        )
        self._session_quotas[session_id] = quotas
        if session_id not in self._session_state:
            self._session_state[session_id] = SessionQuotaState(session_id)

    def get_quotas(self, session_id: str) -> QuotaConfig:
        """Get effective quotas for session."""
        return self._session_quotas.get(session_id, self.global_quotas)

    def get_state(self, session_id: str) -> SessionQuotaState:
        """Get current quota state for session."""
        if session_id not in self._session_state:
            self._session_state[session_id] = SessionQuotaState(session_id)
        return self._session_state[session_id]

    def check_job_submit(
        self,
        session_id: str,
        estimated_buffer_bytes: int = 0,
    ) -> None:
        """Check if job submission is allowed. Raises if quota exceeded."""
        quotas = self.get_quotas(session_id)
        state = self.get_state(session_id)

        if state.outstanding_jobs >= quotas.max_outstanding_jobs:
            raise QuotaExceededError(
                message="Session outstanding jobs quota exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={"limit": quotas.max_outstanding_jobs, "current": state.outstanding_jobs},
            )

        if state.buffer_bytes + estimated_buffer_bytes > quotas.max_buffer_bytes:
            raise QuotaExceededError(
                message="Session buffer quota exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={
                    "limit_bytes": quotas.max_buffer_bytes,
                    "current_bytes": state.buffer_bytes,
                    "requested_bytes": estimated_buffer_bytes,
                },
            )

    def check_sleep_register(self, session_id: str) -> None:
        """Check if sleep registration is allowed."""
        quotas = self.get_quotas(session_id)
        state = self.get_state(session_id)

        if state.sleep_registrations >= quotas.max_sleep_registrations:
            raise QuotaExceededError(
                message="Session sleep registrations quota exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={"limit": quotas.max_sleep_registrations, "current": state.sleep_registrations},
            )

    def check_buffer_create(
        self,
        session_id: str,
        buffer_bytes: int,
    ) -> None:
        """Check if buffer creation is allowed."""
        quotas = self.get_quotas(session_id)
        state = self.get_state(session_id)

        if state.buffer_bytes + buffer_bytes > quotas.max_buffer_bytes:
            raise QuotaExceededError(
                message="Session buffer quota exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={
                    "limit_bytes": quotas.max_buffer_bytes,
                    "current_bytes": state.buffer_bytes,
                    "requested_bytes": buffer_bytes,
                },
            )

    def check_injection(
        self,
        session_id: str,
        tokens: int,
        bytes_: int,
    ) -> None:
        """Check if event injection is within budget."""
        quotas = self.get_quotas(session_id)
        
        if tokens > quotas.max_injected_tokens_per_turn:
            raise QuotaExceededError(
                message="Injected tokens per turn limit exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={"limit": quotas.max_injected_tokens_per_turn, "requested": tokens},
            )

        if bytes_ > quotas.max_injected_bytes_per_event:
            raise QuotaExceededError(
                message="Injected bytes per event limit exceeded",
                code=ErrorCode.QUOTA_EXCEEDED,
                details={"limit": quotas.max_injected_bytes_per_event, "requested": bytes_},
            )

    def on_job_submit(self, session_id: str, buffer_bytes: int = 0) -> None:
        """Update state on job submission."""
        state = self.get_state(session_id)
        state.outstanding_jobs += 1
        state.buffer_bytes += buffer_bytes
        state.last_update = time.time()

    def on_job_complete(self, session_id: str, buffer_bytes: int = 0) -> None:
        """Update state on job completion."""
        state = self.get_state(session_id)
        state.outstanding_jobs = max(0, state.outstanding_jobs - 1)
        state.buffer_bytes = max(0, state.buffer_bytes - buffer_bytes)
        state.last_update = time.time()

    def on_buffer_create(self, session_id: str, buffer_bytes: int) -> None:
        state = self.get_state(session_id)
        state.buffer_bytes += buffer_bytes
        state.last_update = time.time()

    def on_buffer_delete(self, session_id: str, buffer_bytes: int) -> None:
        state = self.get_state(session_id)
        state.buffer_bytes = max(0, state.buffer_bytes - buffer_bytes)
        state.last_update = time.time()

    def on_sleep_register(self, session_id: str) -> None:
        state = self.get_state(session_id)
        state.sleep_registrations += 1
        state.last_update = time.time()

    def on_sleep_clear(self, session_id: str) -> None:
        state = self.get_state(session_id)
        state.sleep_registrations = max(0, state.sleep_registrations - 1)
        state.last_update = time.time()

    def on_cpu_time(self, session_id: str, seconds: float) -> None:
        state = self.get_state(session_id)
        state.cpu_seconds_used += seconds
        state.last_update = time.time()

    def on_gpu_time(self, session_id: str, seconds: float) -> None:
        state = self.get_state(session_id)
        state.gpu_seconds_used += seconds
        state.last_update = time.time()

    def on_wall_time(self, session_id: str, seconds: float) -> None:
        state = self.get_state(session_id)
        state.wall_time_used += seconds
        state.last_update = time.time()

    def get_usage(self, session_id: str) -> dict[str, Any]:
        """Get current usage snapshot."""
        state = self.get_state(session_id)
        quotas = self.get_quotas(session_id)
        return {
            "outstanding_jobs": {"current": state.outstanding_jobs, "limit": quotas.max_outstanding_jobs},
            "buffer_bytes": {"current": state.buffer_bytes, "limit": quotas.max_buffer_bytes},
            "sleep_registrations": {"current": state.sleep_registrations, "limit": quotas.max_sleep_registrations},
            "cpu_seconds_used": state.cpu_seconds_used,
            "gpu_seconds_used": state.gpu_seconds_used,
            "wall_time_used": state.wall_time_used,
        }

    def remove_session(self, session_id: str) -> None:
        self._session_quotas.pop(session_id, None)
        self._session_state.pop(session_id, None)