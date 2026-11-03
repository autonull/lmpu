"""Thin async SDK over the LMPU JSON-RPC socket."""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Self

from lmpu.protocol.codec import RPCStream, open_rpc_connection
from lmpu.protocol.messages import (
    BufferRef,
    BufferView,
    CancelResult,
    HelloResult,
    PollResult,
    RPCRequest,
    RPCResponse,
    SleepResult,
    SubmitResult,
    WaitResult,
    WakeResult,
)
from lmpu.protocol.sentinels import PROTOCOL_VERSION

_DEFAULT_TIMEOUT = 30.0


class LMPUError(RuntimeError):
    """Remote LMPU error returned by the control plane."""

    def __init__(self, code: str, message: str, retriable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retriable = retriable


class LMPUClient:
    """Async LMPU client. Use ``async with await LMPUClient.connect(path)``."""

    def __init__(self, stream: RPCStream, session_id: str) -> None:
        self._stream = stream
        self.session_id = session_id
        self._next_id = 0
        self._closed = False

    @classmethod
    async def connect(
        cls,
        socket_path: str,
        *,
        protocol: str = PROTOCOL_VERSION,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Self:
        """Open a socket and perform the ``hello`` handshake."""
        stream = await open_rpc_connection(socket_path)
        client = cls(stream, session_id="")
        hello = await client.hello(protocol, timeout=timeout)
        client.session_id = hello.session_id
        return client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Client is closed")
        self._next_id += 1
        request = RPCRequest(id=self._next_id, method=method, params=params or {})
        await self._stream.send_request(request)
        while True:
            message = await asyncio.wait_for(self._stream.recv(), timeout)
            if not isinstance(message, RPCResponse) or message.id != request.id:
                continue
            if message.error is not None:
                raise LMPUError(
                    message.error.code.value,
                    message.error.message,
                    message.error.retriable,
                )
            return message.result

    async def hello(
        self, protocol: str = PROTOCOL_VERSION, *, timeout: float = _DEFAULT_TIMEOUT
    ) -> HelloResult:
        """Exchange protocol versions and create a session."""
        return HelloResult.model_validate(
            await self._call("lmpu.hello", {"protocol": protocol}, timeout=timeout)
        )

    async def submit(
        self,
        op: str,
        *,
        processor: str | None = None,
        inputs: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        batch: dict[str, Any] | None = None,
        job_id: str | None = None,
        mode: str = "sync",
    ) -> SubmitResult:
        """Submit a job."""
        params: dict[str, Any] = {
            "op": op,
            "inputs": inputs or {},
            "config": {**(config or {}), "mode": mode},
        }
        if processor is not None:
            params["processor"] = processor
        if batch is not None:
            params["batch"] = batch
        if job_id is not None:
            params["job_id"] = job_id
        return SubmitResult.model_validate(await self._call("lmpu.submit", params))

    async def poll(self, job_id: str) -> PollResult:
        """Poll a job."""
        return PollResult.model_validate(await self._call("lmpu.poll", {"job_id": job_id}))

    async def wait(self, job_id: str, timeout_ms: int | None = None) -> WaitResult:
        """Wait for a terminal job state."""
        params: dict[str, Any] = {"job_id": job_id}
        if timeout_ms is not None:
            params["timeout_ms"] = timeout_ms
        return WaitResult.model_validate(await self._call("lmpu.wait", params))

    async def cancel(self, job_id: str, reason: str | None = None) -> CancelResult:
        """Request job cancellation."""
        params: dict[str, Any] = {"job_id": job_id}
        if reason is not None:
            params["reason"] = reason
        return CancelResult.model_validate(await self._call("lmpu.cancel", params))

    async def read(
        self,
        buffer: str,
        *,
        max_inline_bytes: int = 65536,
        slices: list[list[int]] | None = None,
    ) -> BufferView:
        """Read a bounded buffer slice."""
        params: dict[str, Any] = {"buffer": buffer, "max_inline_bytes": max_inline_bytes}
        if slices is not None:
            params["slices"] = slices
        return BufferView.model_validate(await self._call("lmpu.read", params))

    async def import_buffer(
        self,
        name: str,
        dtype: str,
        shape: list[int],
        *,
        data: bytes | None = None,
        path: str | None = None,
    ) -> BufferRef:
        """Import a buffer by inline data or path."""
        params: dict[str, Any] = {"name": name, "dtype": dtype, "shape": shape}
        if data is not None:
            params["data"] = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
        if path is not None:
            params["path"] = path
        return BufferRef.model_validate(await self._call("lmpu.buffer.import", params))

    async def sleep(self, until: dict[str, Any], *, level: str = "S1") -> SleepResult:
        """Register a sleep condition."""
        return SleepResult.model_validate(
            await self._call("lmpu.sleep", {"until": until, "level": level})
        )

    async def wake(self, session_id: str | None = None) -> WakeResult:
        """Wake a parked session."""
        params: dict[str, Any] = {}
        if session_id is not None:
            params["session_id"] = session_id
        return WakeResult.model_validate(await self._call("lmpu.wake", params))

    async def close(self) -> None:
        """Close the session and the socket."""
        if self._closed:
            return
        try:
            await self._call("lmpu.close", {})
        finally:
            self._closed = True
            self._stream.close()
            await self._stream.wait_closed()
