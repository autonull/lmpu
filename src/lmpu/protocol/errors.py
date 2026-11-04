"""LMPU Protocol v1 - Error definitions."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class LMPUError(Exception):
    """Base LMPU exception."""

    def __init__(
        self,
        message: str,
        code: str,
        retriable: bool = False,
        details: Optional[dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retriable = retriable
        self.details = details or {}
        self.job_id = job_id

    def to_rpc_error(self) -> "RPCError":
        from .messages import RPCError

        return RPCError(
            code=self.code,
            message=self.message,
            job_id=self.job_id,
            retriable=self.retriable,
            details=self.details,
        )


class ProtocolError(LMPUError):
    """Protocol-level errors."""

    pass


class ValidationError(LMPUError):
    """Schema validation errors."""

    pass


class QuotaExceededError(LMPUError):
    """Resource quota exceeded."""

    pass


class BufferError(LMPUError):
    """Buffer-related errors."""

    pass


class JobError(LMPUError):
    """Job execution errors."""

    pass


class ProcessorError(LMPUError):
    """Processor-specific errors."""

    pass


class SandboxError(LMPUError):
    """Sandbox/execution environment errors."""

    pass


class HostEngineError(LMPUError):
    """Host engine errors."""

    pass


# Error code to exception mapping
ERROR_CODE_MAP: dict[str, type[LMPUError]] = {
    "invalid_schema": ValidationError,
    "unknown_method": ProtocolError,
    "unknown_processor": ProcessorError,
    "unknown_op": ProcessorError,
    "capability_mismatch": ProcessorError,
    "buffer_not_found": BufferError,
    "buffer_too_large": BufferError,
    "access_denied": BufferError,
    "quota_exceeded": QuotaExceededError,
    "backpressure": QuotaExceededError,
    "timeout": JobError,
    "cancelled": JobError,
    "worker_crashed": ProcessorError,
    "oom": SandboxError,
    "checksum_mismatch": BufferError,
    "runtime_restarted": ProtocolError,
    "protocol_version": ProtocolError,
    "idempotency_conflict": ProtocolError,
    "already_terminal": JobError,
    "internal": LMPUError,
}


def create_error(code: str, message: str, **kwargs) -> LMPUError:
    """Factory function to create the appropriate error type."""
    error_cls = ERROR_CODE_MAP.get(code, LMPUError)
    return error_cls(message=message, code=code, **kwargs)
