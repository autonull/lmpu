"""LMPU Protocol v1 - Message definitions and JSON-RPC codec."""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ulid import ULID

from .sentinels import PROTOCOL_VERSION


# =============================================================================
# Enums and constants
# =============================================================================

class JobStatus(str, Enum):
    """Internal job states."""
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_EVENT = "waiting_event"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelJobStatus(str, Enum):
    """Model-facing job statuses (subset of internal states)."""
    PENDING = "pending"
    RUNNING = "running"
    SLEEPING = "sleeping"  # Maps to WAITING_INPUT, WAITING_EVENT
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventKind(str, Enum):
    """Event types for the event log."""
    JOB_SUBMITTED = "JobSubmitted"
    JOB_STARTED = "JobStarted"
    PROGRESS = "Progress"
    JOB_COMPLETED = "JobCompleted"
    JOB_FAILED = "JobFailed"
    JOB_CANCELLED = "JobCancelled"
    NEEDS_MORE_INPUT = "NeedsMoreInput"
    SLEEPING = "Sleeping"
    WOKEN = "Woken"
    BUFFER_CREATED = "BufferCreated"
    BUFFER_UPDATED = "BufferUpdated"
    RESOURCE_WARNING = "ResourceWarning"
    CONTROL_ERROR = "ControlError"
    RESULT_TOO_LARGE = "ResultTooLarge"
    HEARTBEAT = "Heartbeat"


class InjectionPolicy(str, Enum):
    """Event injection policies."""
    INTERRUPT = "interrupt"      # Interrupt generation at next token boundary
    NEXT_TURN = "next_turn"      # Inject at next generation turn
    WAKE = "wake"                # Wake host immediately
    INJECT = "inject"            # Inject immediately
    OBSERVE = "observe"          # Observe only, don't inject


class SleepLevel(str, Enum):
    """Sleep/park levels."""
    S1 = "S1"  # KV resident in VRAM
    S2 = "S2"  # KV offloaded to CPU
    S3 = "S3"  # Checkpointed to disk (deferred)


class ExecutionMode(str, Enum):
    """Job execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    INTERRUPT = "interrupt"


class DeterminismLevel(str, Enum):
    """Determinism requirements."""
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class ErrorCode(str, Enum):
    """Canonical error codes."""
    INVALID_SCHEMA = "invalid_schema"
    UNKNOWN_METHOD = "unknown_method"
    UNKNOWN_PROCESSOR = "unknown_processor"
    UNKNOWN_OP = "unknown_op"
    CAPABILITY_MISMATCH = "capability_mismatch"
    BUFFER_NOT_FOUND = "buffer_not_found"
    BUFFER_TOO_LARGE = "buffer_too_large"
    ACCESS_DENIED = "access_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    BACKPRESSURE = "backpressure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    WORKER_CRASHED = "worker_crashed"
    OOM = "oom"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    RUNTIME_RESTARTED = "runtime_restarted"
    PROTOCOL_VERSION = "protocol_version"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    ALREADY_TERMINAL = "already_terminal"
    INTERNAL = "internal"


# =============================================================================
# Base models
# =============================================================================

class LMPUBaseModel(BaseModel):
    """Base model with common configuration."""
    model_config = ConfigDict(
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
        validate_assignment=True,
        ser_json_timedelta="iso8601",
        ser_json_bytes="base64",
    )


class RPCError(LMPUBaseModel):
    """JSON-RPC 2.0 error object."""
    code: ErrorCode
    message: str
    job_id: Optional[str] = None
    retriable: bool = False
    details: Optional[dict[str, Any]] = None


class RPCRequest(LMPUBaseModel):
    """JSON-RPC 2.0 request."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None] = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class RPCResponse(LMPUBaseModel):
    """JSON-RPC 2.0 response."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Any] = None
    error: Optional[RPCError] = None


class RPCNotification(LMPUBaseModel):
    """JSON-RPC 2.0 notification (no id)."""
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Model-plane message kinds (inside <|proc|> sentinels)
# =============================================================================

class ModelMessageKind(str, Enum):
    """Model-plane message kinds."""
    SUBMIT = "submit"
    POLL = "poll"
    WAIT = "wait"
    READ_RESULT = "read_result"
    WRITE_INPUT = "write_input"
    CANCEL = "cancel"
    SLEEP = "sleep"
    WAKE = "wake"
    QUERY_CAPABILITIES = "query_capabilities"


class ModelMessageBase(LMPUBaseModel):
    """Base for all model-plane messages."""
    version: int = 1
    kind: ModelMessageKind


class SubmitMessage(ModelMessageBase):
    """Submit a job or batch."""
    kind: Literal[ModelMessageKind.SUBMIT] = ModelMessageKind.SUBMIT
    job_id: Optional[str] = None
    processor: Optional[str] = None
    op: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    program: Optional[str] = None
    batch: Optional[dict[str, Any]] = None
    config: Optional[dict[str, Any]] = None


class PollMessage(ModelMessageBase):
    """Poll job status."""
    kind: Literal[ModelMessageKind.POLL] = ModelMessageKind.POLL
    job_id: str


class WaitMessage(ModelMessageBase):
    """Wait for job completion with timeout."""
    kind: Literal[ModelMessageKind.WAIT] = ModelMessageKind.WAIT
    job_id: str
    timeout_ms: Optional[int] = None


class ReadResultMessage(ModelMessageBase):
    """Read a bounded buffer view."""
    kind: Literal[ModelMessageKind.READ_RESULT] = ModelMessageKind.READ_RESULT
    job_id: str
    buffer: str
    slices: Optional[list[list[int]]] = None
    max_inline_bytes: int = 65536


class WriteInputMessage(ModelMessageBase):
    """Write interactive input to a waiting job."""
    kind: Literal[ModelMessageKind.WRITE_INPUT] = ModelMessageKind.WRITE_INPUT
    job_id: str
    field: str
    value: Any


class CancelMessage(ModelMessageBase):
    """Request job cancellation."""
    kind: Literal[ModelMessageKind.CANCEL] = ModelMessageKind.CANCEL
    job_id: str
    reason: Optional[str] = None


class SleepCondition(LMPUBaseModel):
    """Sleep wake condition."""
    all: list[str] = Field(default_factory=list)
    any: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    deadline_ms: Optional[int] = None
    success_only: bool = False


class SleepMessage(ModelMessageBase):
    """Request host sleep/park."""
    kind: Literal[ModelMessageKind.SLEEP] = ModelMessageKind.SLEEP
    until: SleepCondition
    level: Optional[SleepLevel] = None


class WakeMessage(ModelMessageBase):
    """Force wake or cancel sleep."""
    kind: Literal[ModelMessageKind.WAKE] = ModelMessageKind.WAKE
    session_id: Optional[str] = None


class QueryCapabilitiesMessage(ModelMessageBase):
    """Query available processors and operations."""
    kind: Literal[ModelMessageKind.QUERY_CAPABILITIES] = ModelMessageKind.QUERY_CAPABILITIES
    op: Optional[str] = None
    processor: Optional[str] = None


ModelMessage = Union[
    SubmitMessage,
    PollMessage,
    WaitMessage,
    ReadResultMessage,
    WriteInputMessage,
    CancelMessage,
    SleepMessage,
    WakeMessage,
    QueryCapabilitiesMessage,
]


# =============================================================================
# Control-plane method parameters and results
# =============================================================================

class HelloParams(LMPUBaseModel):
    protocol: str = PROTOCOL_VERSION
    resume_from: Optional[int] = None
    session_config: Optional[dict[str, Any]] = None


class HelloResult(LMPUBaseModel):
    session_id: str
    protocol: str
    schema_hash: str
    resume_status: Literal["ok", "gap", "none"] = "none"
    server_time: float


class CapabilitiesParams(LMPUBaseModel):
    op: Optional[str] = None
    processor: Optional[str] = None


class CapabilityDescriptor(LMPUBaseModel):
    name: str
    version: str
    protocol: str
    ops: list[dict[str, Any]]
    limits: dict[str, Any]
    determinism: list[str]
    latency_class: str
    streaming: bool
    progress: bool
    interactive: bool
    persistent_state: bool
    batching: bool
    zero_copy: bool
    sandbox: list[str]


class CapabilitiesResult(LMPUBaseModel):
    processors: list[CapabilityDescriptor]
    matches: Optional[list[str]] = None
    recommended: Optional[str] = None


class SubmitParams(LMPUBaseModel):
    job_id: Optional[str] = None
    processor: Optional[str] = None
    op: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    program: Optional[str] = None
    batch: Optional[dict[str, Any]] = None
    config: Optional[dict[str, Any]] = None


class SubmitResult(LMPUBaseModel):
    job_id: str
    status: ModelJobStatus
    queue_position: Optional[int] = None
    seq: int


class PollParams(LMPUBaseModel):
    job_id: str


class PollResult(LMPUBaseModel):
    job_id: str
    status: ModelJobStatus
    progress: Optional[float] = None
    elapsed_ms: Optional[int] = None
    queue_position: Optional[int] = None


class WaitParams(LMPUBaseModel):
    job_id: str
    timeout_ms: Optional[int] = None


class WaitResult(LMPUBaseModel):
    status: ModelJobStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[RPCError] = None


class ReadParams(LMPUBaseModel):
    buffer: str
    slices: Optional[list[list[int]]] = None
    max_inline_bytes: int = 65536


class BufferView(LMPUBaseModel):
    buffer: str
    view: Optional[dict[str, Any]] = None
    dtype: str
    data: Optional[list[Any]] = None
    derived_buffer: Optional[str] = None
    note: Optional[str] = None


class WriteParams(LMPUBaseModel):
    buffer: str
    slices: list[list[int]]
    data: Any


class CancelParams(LMPUBaseModel):
    job_id: str
    reason: Optional[str] = None


class CancelResult(LMPUBaseModel):
    job_id: str
    status: Literal["cancel_requested", "already_terminal"]


class SleepParams(LMPUBaseModel):
    until: SleepCondition
    level: Optional[SleepLevel] = None


class SleepResult(LMPUBaseModel):
    sleep_id: str
    status: Literal["registered", "already_satisfied"]
    woken: Optional[dict[str, Any]] = None


class WakeParams(LMPUBaseModel):
    session_id: Optional[str] = None


class WakeResult(LMPUBaseModel):
    status: Literal["woken", "no_active_sleep"]


class AckParams(LMPUBaseModel):
    seq: int


class AckResult(LMPUBaseModel):
    acked: int


class BufferCreateParams(LMPUBaseModel):
    name: str
    dtype: str
    shape: list[int]
    persist: bool = False


class BufferImportParams(LMPUBaseModel):
    name: str
    data: Optional[bytes] = None
    path: Optional[str] = None
    dtype: str
    shape: list[int]


class BufferRef(LMPUBaseModel):
    buffer_id: str
    dtype: str
    shape: list[int]
    bytes: int
    checksum: str
    owner: str
    created_at: float
    ttl_ms: Optional[int] = None
    provenance: Optional[dict[str, Any]] = None


class BufferStatParams(LMPUBaseModel):
    buffer: str


class BufferChecksumParams(LMPUBaseModel):
    buffer: str
    algorithm: Literal["blake3", "sha256"] = "blake3"


class BufferDeleteParams(LMPUBaseModel):
    buffer: str


class JobsListParams(LMPUBaseModel):
    session_id: Optional[str] = None
    status: Optional[list[ModelJobStatus]] = None
    limit: int = 100


class ConfigureParams(LMPUBaseModel):
    max_outstanding_jobs: Optional[int] = None
    max_buffer_bytes: Optional[int] = None
    max_sleep_seconds: Optional[int] = None
    sleep_level: Optional[SleepLevel] = None
    injection: Optional[dict[str, Any]] = None


class HealthResult(LMPUBaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    queue_depth: int
    worker_health: dict[str, str]
    uptime_seconds: float


class CloseParams(LMPUBaseModel):
    detach: bool = False


class CloseResult(LMPUBaseModel):
    cancelled_jobs: list[str]
    detached_jobs: list[str]


# =============================================================================
# Event definitions
# =============================================================================

class EventData(LMPUBaseModel):
    """Base event data."""
    pass


class JobSubmittedData(EventData):
    job_id: str
    processor: str
    queue_position: int


class JobStartedData(EventData):
    job_id: str
    worker: str
    start_time: float


class ProgressData(EventData):
    job_id: str
    fraction: float
    checkpoint: Optional[dict[str, Any]] = None
    message: Optional[str] = None


class JobCompletedData(EventData):
    job_id: str
    status: Literal["completed"] = "completed"
    result: dict[str, Any]
    buffer_refs: list[str] = Field(default_factory=list)
    checksum: str
    cache_hit: bool = False


class JobFailedData(EventData):
    job_id: str
    error_code: ErrorCode
    retriable: bool
    detail: str


class JobCancelledData(EventData):
    job_id: str
    reason: str


class NeedsMoreInputData(EventData):
    job_id: str
    prompt: str
    fields: list[str]


class SleepingData(EventData):
    session: str
    sleep_id: str
    conditions: SleepCondition


class WokenData(EventData):
    session: str
    sleep_id: str
    reason: str
    satisfied_jobs: list[str]
    statuses: dict[str, ModelJobStatus]
    already_satisfied: bool = False


class BufferCreatedData(EventData):
    buffer_id: str


class BufferUpdatedData(EventData):
    buffer_id: str
    checksum: str


class ResourceWarningData(EventData):
    quota: str
    current: int
    limit: int


class ControlErrorData(EventData):
    position: int
    expected_schema: str
    message: str


class ResultTooLargeData(EventData):
    buffer: str
    reason: str


class HeartbeatData(EventData):
    worker: str
    timestamp: float


EventPayload = Union[
    JobSubmittedData,
    JobStartedData,
    ProgressData,
    JobCompletedData,
    JobFailedData,
    JobCancelledData,
    NeedsMoreInputData,
    SleepingData,
    WokenData,
    BufferCreatedData,
    BufferUpdatedData,
    ResourceWarningData,
    ControlErrorData,
    ResultTooLargeData,
    HeartbeatData,
]


class Event(LMPUBaseModel):
    """Event log entry."""
    event_id: str = Field(default_factory=lambda: str(ULID()))
    seq: int
    session: str
    job_id: Optional[str] = None
    type: EventKind
    ts_monotonic: float
    ts_wall: float = Field(default_factory=lambda: datetime.now().timestamp())
    data: EventPayload


# =============================================================================
# Model-plane event wrapper (for injection into model context)
# =============================================================================

class InjectedEvent(LMPUBaseModel):
    """Event as injected into model context via <|proc_event|>."""
    version: int = 1
    kind: Literal["event"] = "event"
    event: EventKind
    job_id: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[dict[str, Any]] = None


# =============================================================================
# Type discriminators for parsing
# =============================================================================

MODEL_MESSAGE_TYPES: dict[str, type[ModelMessageBase]] = {
    ModelMessageKind.SUBMIT: SubmitMessage,
    ModelMessageKind.POLL: PollMessage,
    ModelMessageKind.WAIT: WaitMessage,
    ModelMessageKind.READ_RESULT: ReadResultMessage,
    ModelMessageKind.WRITE_INPUT: WriteInputMessage,
    ModelMessageKind.CANCEL: CancelMessage,
    ModelMessageKind.SLEEP: SleepMessage,
    ModelMessageKind.WAKE: WakeMessage,
    ModelMessageKind.QUERY_CAPABILITIES: QueryCapabilitiesMessage,
}


def parse_model_message(data: dict[str, Any]) -> ModelMessage:
    """Parse a model-plane message from a dict."""
    kind = data.get("kind")
    if kind not in MODEL_MESSAGE_TYPES:
        raise ValueError(f"Unknown model message kind: {kind}")
    return MODEL_MESSAGE_TYPES[kind](**data)


# =============================================================================
# JSON-RPC codec
# =============================================================================

class JSONRPCCodec:
    """JSON-RPC 2.0 codec using orjson for performance."""

    @staticmethod
    def encode_request(request: RPCRequest) -> bytes:
        return orjson.dumps(request.model_dump(exclude_none=True), option=orjson.OPT_SERIALIZE_NUMPY)

    @staticmethod
    def encode_response(response: RPCResponse) -> bytes:
        return orjson.dumps(response.model_dump(exclude_none=True), option=orjson.OPT_SERIALIZE_NUMPY)

    @staticmethod
    def encode_notification(notification: RPCNotification) -> bytes:
        return orjson.dumps(notification.model_dump(exclude_none=True), option=orjson.OPT_SERIALIZE_NUMPY)

    @staticmethod
    def decode_request(data: bytes) -> RPCRequest:
        return RPCRequest.model_validate_json(data)

    @staticmethod
    def decode_response(data: bytes) -> RPCResponse:
        return RPCResponse.model_validate_json(data)

    @staticmethod
    def decode_notification(data: bytes) -> RPCNotification:
        return RPCNotification.model_validate_json(data)


# Import orjson at module level for the codec
import orjson