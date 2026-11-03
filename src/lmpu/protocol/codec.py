"""LMPU Protocol v1 - JSON-RPC 2.0 codec over Unix domain sockets."""

from __future__ import annotations

import asyncio
import orjson
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Union

from .messages import (
    RPCError,
    RPCNotification,
    RPCRequest,
    RPCResponse,
    JSONRPCCodec,
)


@dataclass
class Frame:
    """Length-prefixed JSON-RPC frame."""
    data: bytes

    def encode(self) -> bytes:
        """Encode frame with 4-byte big-endian length prefix."""
        length = len(self.data)
        return struct.pack(">I", length) + self.data

    @classmethod
    def decode(cls, data: bytes) -> tuple["Frame", bytes]:
        """Decode a frame from the buffer, returning frame and remaining data."""
        if len(data) < 4:
            raise ValueError("Incomplete frame header")
        length = struct.unpack(">I", data[:4])[0]
        if len(data) < 4 + length:
            raise ValueError("Incomplete frame body")
        frame_data = data[4:4+length]
        remaining = data[4+length:]
        return cls(frame_data), remaining


class RPCCodec:
    """JSON-RPC 2.0 codec with length-prefixed framing."""

    @staticmethod
    def encode_request(request: RPCRequest) -> bytes:
        return Frame(JSONRPCCodec.encode_request(request)).encode()

    @staticmethod
    def encode_response(response: RPCResponse) -> bytes:
        return Frame(JSONRPCCodec.encode_response(response)).encode()

    @staticmethod
    def encode_notification(notification: RPCNotification) -> bytes:
        return Frame(JSONRPCCodec.encode_notification(notification)).encode()

    @staticmethod
    def decode_frame(frame: Frame) -> Union[RPCRequest, RPCResponse, RPCNotification]:
        """Decode a frame into a request, response, or notification."""
        try:
            obj = orjson.loads(frame.data)
        except orjson.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if "method" in obj and "id" not in obj:
            return RPCNotification.model_validate(obj)
        elif "method" in obj:
            return RPCRequest.model_validate(obj)
        else:
            return RPCResponse.model_validate(obj)


class RPCStream:
    """Async length-prefixed JSON-RPC stream reader/writer."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer
        self._buffer = bytearray()

    async def send_request(self, request: RPCRequest) -> None:
        self._writer.write(RPCCodec.encode_request(request))
        await self._writer.drain()

    async def send_response(self, response: RPCResponse) -> None:
        self._writer.write(RPCCodec.encode_response(response))
        await self._writer.drain()

    async def send_notification(self, notification: RPCNotification) -> None:
        self._writer.write(RPCCodec.encode_notification(notification))
        await self._writer.drain()

    async def recv(self) -> Union[RPCRequest, RPCResponse, RPCNotification]:
        """Receive and decode the next complete frame."""
        while True:
            if len(self._buffer) >= 4:
                length = struct.unpack(">I", self._buffer[:4])[0]
                if len(self._buffer) >= 4 + length:
                    frame_data = bytes(self._buffer[4:4+length])
                    del self._buffer[:4+length]
                    frame = Frame(frame_data)
                    return RPCCodec.decode_frame(frame)

            chunk = await self._reader.read(65536)
            if not chunk:
                raise EOFError("Connection closed")
            self._buffer.extend(chunk)

    async def __aiter__(self) -> AsyncIterator[Union[RPCRequest, RPCResponse, RPCNotification]]:
        while True:
            try:
                yield await self.recv()
            except EOFError:
                break

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        await self._writer.wait_closed()


async def open_rpc_connection(path: str) -> RPCStream:
    """Open a JSON-RPC connection to a Unix domain socket."""
    reader, writer = await asyncio.open_unix_connection(path)
    return RPCStream(reader, writer)