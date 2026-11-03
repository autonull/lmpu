"""Unix-domain socket file-descriptor passing."""

from __future__ import annotations

import json
import socket
from array import array
from typing import Any


class FDPassingError(Exception):
    """Raised when SCM_RIGHTS file-descriptor passing fails."""


def send_fds(
    sock: socket.socket,
    fds: list[int],
    header: dict[str, Any],
) -> None:
    """Send JSON header plus file descriptors over one SCM_RIGHTS message."""
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8") + b"\n"
    ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", fds))] if fds else []
    try:
        sent = sock.sendmsg([payload], ancillary)
    except OSError as exc:
        raise FDPassingError(f"sendmsg failed: {exc}") from exc
    if sent != len(payload):
        raise FDPassingError(f"Short fd-channel write: {sent}/{len(payload)}")


def recv_fds(sock: socket.socket, max_bytes: int = 1_048_576) -> tuple[dict[str, Any], list[int]]:
    """Receive one newline-delimited JSON header and its attached descriptors."""
    chunks: list[bytes] = []
    fds: list[int] = []
    total = 0
    while True:
        try:
            data, ancillary, _flags, _address = sock.recvmsg(
                65536,
                socket.CMSG_SPACE(4 * 1024 * 1024),
            )
        except OSError as exc:
            raise FDPassingError(f"recvmsg failed: {exc}") from exc

        if not data:
            raise FDPassingError("fd channel closed")

        for level, kind, payload in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array("i")
                received.frombytes(payload[: len(payload) - (len(payload) % array("i").itemsize)])
                fds.extend(received.tolist())

        chunks.append(data)
        total += len(data)
        if total > max_bytes:
            raise FDPassingError("fd-channel header exceeds maximum size")
        if b"\n" in data:
            break

    raw = b"".join(chunks)
    line = raw.split(b"\n", 1)[0]
    try:
        header = json.loads(line)
    except json.JSONDecodeError as exc:
        raise FDPassingError(f"invalid fd-channel header: {exc}") from exc
    if not isinstance(header, dict):
        raise FDPassingError("fd-channel header must be an object")
    return header, fds
