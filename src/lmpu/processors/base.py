"""LMPU Processors - Base classes and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from ..protocol.messages import CapabilityDescriptor


@dataclass
class Manifest:
    """Processor manifest / capability descriptor."""
    name: str
    version: str = "0.1.0"
    protocol: str = "lmpu/1"
    ops: list[dict[str, Any]] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    determinism: list[str] = field(default_factory=lambda: ["best_effort"])
    latency_class: str = "ms"
    streaming: bool = False
    progress: bool = False
    interactive: bool = False
    persistent_state: bool = False
    batching: bool = False
    zero_copy: bool = False
    sandbox: list[str] = field(default_factory=lambda: ["subprocess"])
    executor: str = "process"  # inline, thread, process, remote, gpu


class ProcessorContext:
    """Context provided to processor during setup."""
    
    def __init__(self, runtime_config: dict[str, Any]):
        self.runtime_config = runtime_config
        self._resources: dict[str, Any] = {}

    def get_resource(self, name: str) -> Any:
        return self._resources.get(name)

    def set_resource(self, name: str, value: Any) -> None:
        self._resources[name] = value


class JobContext:
    """Context provided to processor during job execution."""
    
    def __init__(
        self,
        job_id: str,
        session_id: str,
        buffers: dict[str, Any],  # buffer_id -> numpy array or mmap
        send_progress: callable,
        send_needs_input: callable,
        submit_child: callable,
        log: callable,
        trace: callable,
        cancelled: callable,
    ):
        self.job_id = job_id
        self.session_id = session_id
        self._buffers = buffers
        self._send_progress = send_progress
        self._send_needs_input = send_needs_input
        self._submit_child = submit_child
        self._log = log
        self._trace = trace
        self._cancelled = cancelled

    def read_buffer(self, name: str) -> Any:
        """Read a buffer by logical name."""
        buffer_id = self._buffers.get(name)
        if buffer_id is None:
            raise KeyError(f"Buffer {name} not found")
        return buffer_id

    def write_buffer(self, name: str, data: Any) -> None:
        """Write to a buffer by logical name."""
        buffer_id = self._buffers.get(name)
        if buffer_id is None:
            raise KeyError(f"Buffer {name} not found")
        # Implementation depends on buffer manager
        pass

    def progress(self, fraction: float, message: Optional[str] = None) -> None:
        """Report progress."""
        self._send_progress(fraction, message)

    def request_input(self, fields: list[str], prompt: str) -> None:
        """Request interactive input from host."""
        self._send_needs_input(fields, prompt)

    async def submit_child(
        self,
        processor: str,
        op: str,
        inputs: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Submit a child job."""
        return await self._submit_child(processor, op, inputs, config)

    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message."""
        self._log(level, message, **kwargs)

    def trace(self, event: str, data: dict[str, Any]) -> None:
        """Emit a trace event."""
        self._trace(event, data)

    def cancelled(self) -> bool:
        """Check if job has been cancelled."""
        return self._cancelled()


class Result(BaseModel):
    """Job execution result."""
    model_config = ConfigDict(extra="allow")
    
    success: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    buffer_refs: list[str] = Field(default_factory=list)
    checksum: str = ""
    cache_hit: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retriable: bool = False


class Processor(ABC):
    """Base processor class."""
    
    manifest: Manifest
    
    def __init__(self):
        if not hasattr(self, 'manifest') or self.manifest is None:
            raise NotImplementedError("Processor must define manifest class attribute")

    @abstractmethod
    async def setup(self, ctx: ProcessorContext) -> None:
        """Initialize processor (called once at worker startup)."""
        pass

    @abstractmethod
    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        """Execute a job."""
        pass

    async def cancel(self, job_id: str) -> None:
        """Cancel a running job (optional)."""
        pass

    async def shutdown(self) -> None:
        """Cleanup processor resources (optional)."""
        pass


# Worker protocol message types
WORKER_MESSAGES = {
    "ready": "Worker ready with manifest",
    "heartbeat": "Periodic heartbeat",
    "progress": "Job progress update",
    "needs_input": "Job needs interactive input",
    "result": "Job completed successfully",
    "error": "Job failed",
    "shutdown": "Shutdown request",
}

PARENT_COMMANDS = {
    "execute": "Execute a job",
    "cancel": "Cancel a job",
    "shutdown": "Shutdown worker",
}