"""LMPU - Language Model Processing Unit.

A runtime for decoupled agentic workflows where the host LLM acts as a
semantic control plane and native processors execute computations.
"""

__version__ = "0.1.0"

from .kernel import (
    CapabilityRegistry,
    EventLog,
    JobRecord,
    JobTable,
    QuotaManager,
    Runtime,
    Scheduler,
)
from .memory import (
    BufferManager,
    Grant,
    GrantManager,
    GrantType,
    SHMBackend,
    TempSHMBackend,
)
from .observability import (
    TraceReplayer,
    TraceViewer,
)
from .processors import (
    HardwareStubProcessor,
    JobContext,
    Manifest,
    NeuralSummaryProcessor,
    Processor,
    ProcessorContext,
    PureFunctionProcessor,
    PythonVMProcessor,
    Result,
    SymbolicPlannerProcessor,
    VectorEngineProcessor,
)
from .protocol import (
    PROC_CLOSE,
    PROC_EVENT_CLOSE,
    PROC_EVENT_OPEN,
    PROC_OPEN,
    PROTOCOL_VERSION,
    BufferRef,
    CancelMessage,
    DeterminismLevel,
    ErrorCode,
    Event,
    EventKind,
    ExecutionMode,
    InjectionPolicy,
    JobStatus,
    ModelJobStatus,
    ModelMessageKind,
    PollMessage,
    QueryCapabilitiesMessage,
    ReadResultMessage,
    SleepLevel,
    SleepMessage,
    SubmitMessage,
    WaitMessage,
    WakeMessage,
    WriteInputMessage,
    parse_model_message,
)

__all__ = [
    # Protocol
    "PROC_OPEN",
    "PROC_CLOSE",
    "PROC_EVENT_OPEN",
    "PROC_EVENT_CLOSE",
    "PROTOCOL_VERSION",
    "JobStatus",
    "ModelJobStatus",
    "EventKind",
    "InjectionPolicy",
    "SleepLevel",
    "ExecutionMode",
    "DeterminismLevel",
    "ErrorCode",
    "ModelMessageKind",
    "SubmitMessage",
    "PollMessage",
    "WaitMessage",
    "ReadResultMessage",
    "WriteInputMessage",
    "CancelMessage",
    "SleepMessage",
    "WakeMessage",
    "QueryCapabilitiesMessage",
    "BufferRef",
    "Event",
    "parse_model_message",
    # Memory
    "BufferManager",
    "SHMBackend",
    "TempSHMBackend",
    "GrantManager",
    "GrantType",
    "Grant",
    # Kernel
    "Runtime",
    "JobTable",
    "JobRecord",
    "EventLog",
    "Scheduler",
    "QuotaManager",
    "CapabilityRegistry",
    # Processors
    "Processor",
    "ProcessorContext",
    "JobContext",
    "Result",
    "Manifest",
    "PureFunctionProcessor",
    "PythonVMProcessor",
    "VectorEngineProcessor",
    "SymbolicPlannerProcessor",
    "NeuralSummaryProcessor",
    "HardwareStubProcessor",
    # Host
    "HostEngine",
    "HostState",
    "HostStats",
    "HFEngine",
    "SessionController",
    "SessionConfig",
    # Observability
    "TraceViewer",
    "TraceReplayer",
]

_HOST_EXPORTS = frozenset(
    {
        "HostEngine",
        "HostState",
        "HostStats",
        "HFEngine",
        "SessionController",
        "SessionConfig",
    }
)


def __getattr__(name: str):
    """Load host symbols lazily: they depend on the optional `host` extra."""
    if name in _HOST_EXPORTS:
        from . import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
