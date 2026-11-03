"""LMPU - Language Model Processing Unit.

A runtime for decoupled agentic workflows where the host LLM acts as a 
semantic control plane and native processors execute computations.
"""

__version__ = "0.1.0"

from .protocol import (
    PROC_OPEN,
    PROC_CLOSE,
    PROC_EVENT_OPEN,
    PROC_EVENT_CLOSE,
    PROTOCOL_VERSION,
    JobStatus,
    ModelJobStatus,
    EventKind,
    InjectionPolicy,
    SleepLevel,
    ExecutionMode,
    DeterminismLevel,
    ErrorCode,
    ModelMessageKind,
    SubmitMessage,
    PollMessage,
    WaitMessage,
    ReadResultMessage,
    WriteInputMessage,
    CancelMessage,
    SleepMessage,
    WakeMessage,
    QueryCapabilitiesMessage,
    BufferRef,
    Event,
    parse_model_message,
)

from .memory import (
    BufferManager,
    SHMBackend,
    TempSHMBackend,
    GrantManager,
    GrantType,
    Grant,
)

from .kernel import (
    Runtime,
    JobTable,
    JobRecord,
    EventLog,
    Scheduler,
    QuotaManager,
    CapabilityRegistry,
)

from .processors import (
    Processor,
    ProcessorContext,
    JobContext,
    Result,
    Manifest,
    PureFunctionProcessor,
    PythonVMProcessor,
    VectorEngineProcessor,
    SymbolicPlannerProcessor,
    NeuralSummaryProcessor,
    HardwareStubProcessor,
)

from .host import (
    HostEngine,
    HostState,
    HostStats,
    HFEngine,
    SessionController,
    SessionConfig,
)

from .observability import (
    TraceViewer,
    TraceReplayer,
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