"""LMPU Memory - Buffer manager with typed views and provenance."""

from __future__ import annotations

import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..protocol.messages import BufferRef
from .checksum import ChecksumAlgorithm, compute_checksum
from .grants import Grant, GrantManager, GrantType
from .shm import SHMBackend


class BufferError(Exception):
    """Buffer operation errors."""

    pass


@dataclass
class BufferMetadata:
    """Complete buffer metadata."""

    buffer_id: str
    session_id: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    checksum: str
    owner: str
    created_at: float
    ttl_ms: int | None = None
    persist: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    grants: list[Grant] = field(default_factory=list)
    ref_count: int = 0

    def to_ref(self) -> BufferRef:
        """Convert to BufferRef for wire protocol."""
        return BufferRef(
            buffer_id=self.buffer_id,
            dtype=self.dtype,
            shape=list(self.shape),
            bytes=self.bytes,
            checksum=self.checksum,
            owner=self.owner,
            created_at=self.created_at,
            ttl_ms=self.ttl_ms,
        )


class BufferView:
    """A view into a buffer with optional slicing."""

    def __init__(
        self,
        buffer_id: str,
        dtype: str,
        shape: tuple[int, ...],
        slices: list[slice] | None = None,
        strides: tuple[int, ...] | None = None,
    ):
        self.buffer_id = buffer_id
        self.dtype = dtype
        self.shape = shape
        self.slices = slices
        self.strides = strides

    def to_dict(self) -> dict[str, Any]:
        result = {
            "buffer": self.buffer_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }
        if self.slices:
            result["view"] = {
                "slices": [[s.start or 0, s.stop or 0, s.step or 1] for s in self.slices]
            }
        return result


class BufferManager:
    """Manages typed shared memory buffers with provenance and grants."""

    def __init__(
        self,
        backend: SHMBackend | None = None,
        grant_manager: GrantManager | None = None,
        default_ttl_ms: int = 3600000,
        import_roots: list[str] | None = None,
        verify_on_read: bool = True,
    ):
        self.backend = backend or SHMBackend()
        self.grant_manager = grant_manager or GrantManager()
        self.default_ttl_ms = default_ttl_ms
        self.import_roots = [Path(p).resolve() for p in (import_roots or ["/tmp/lmpu/imports"])]
        self.verify_on_read = verify_on_read
        self._metadata: dict[str, BufferMetadata] = {}

    def _dtype_from_str(self, dtype_str: str) -> np.dtype:
        """Parse dtype string to numpy dtype."""
        try:
            return np.dtype(dtype_str)
        except TypeError as exc:
            raise BufferError(f"Invalid dtype: {dtype_str}") from exc

    def _validate_shape(self, shape: tuple[int, ...], dtype: np.dtype) -> int:
        """Validate shape and return total bytes."""
        if any(s < 0 for s in shape):
            raise BufferError("Negative dimensions not allowed")
        if len(shape) > 32:
            raise BufferError("Too many dimensions (max 32)")
        total_elements = 1
        for s in shape:
            total_elements *= s
        return total_elements * dtype.itemsize

    def _make_buffer_id(self, prefix: str = "buf") -> str:
        """Generate a unique buffer ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def create(
        self,
        session_id: str,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        persist: bool = False,
        ttl_ms: int | None = None,
    ) -> BufferRef:
        """Create a new zero-initialized buffer."""
        np_dtype = self._dtype_from_str(dtype)
        bytes_needed = self._validate_shape(shape, np_dtype)
        buffer_id = self._make_buffer_id(name)

        # Create in SHM
        mm, f = self.backend.create_buffer(session_id, buffer_id, bytes_needed)
        try:
            # Zero initialize
            arr = np.ndarray(shape, dtype=np_dtype, buffer=mm)
            arr.fill(0)
            mm.flush()
            checksum = compute_checksum(mm[:])
        finally:
            mm.close()
            f.close()

        metadata = BufferMetadata(
            buffer_id=buffer_id,
            session_id=session_id,
            dtype=dtype,
            shape=shape,
            bytes=bytes_needed,
            checksum=checksum,
            owner=session_id,
            created_at=time.time(),
            ttl_ms=ttl_ms or self.default_ttl_ms,
            persist=persist,
            provenance={"source": "create", "name": name},
        )
        self._metadata[buffer_id] = metadata
        return metadata.to_ref()

    def import_bytes(
        self,
        session_id: str,
        name: str,
        data: bytes,
        dtype: str,
        shape: tuple[int, ...],
        ttl_ms: int | None = None,
    ) -> BufferRef:
        """Import buffer from bytes."""
        np_dtype = self._dtype_from_str(dtype)
        expected_bytes = self._validate_shape(shape, np_dtype)
        if len(data) != expected_bytes:
            raise BufferError(f"Data size {len(data)} != expected {expected_bytes}")

        buffer_id = self._make_buffer_id(name)
        mm, f = self.backend.create_buffer(session_id, buffer_id, expected_bytes)
        try:
            mm[:] = data
            mm.flush()
            checksum = compute_checksum(data)
        finally:
            mm.close()
            f.close()

        metadata = BufferMetadata(
            buffer_id=buffer_id,
            session_id=session_id,
            dtype=dtype,
            shape=shape,
            bytes=expected_bytes,
            checksum=checksum,
            owner=session_id,
            created_at=time.time(),
            ttl_ms=ttl_ms or self.default_ttl_ms,
            provenance={"source": "inline", "name": name, "original_hash": checksum},
        )
        self._metadata[buffer_id] = metadata
        return metadata.to_ref()

    def import_file(
        self,
        session_id: str,
        name: str,
        path: str,
        dtype: str,
        shape: tuple[int, ...],
        ttl_ms: int | None = None,
    ) -> BufferRef:
        """Import buffer from file (path must be under import_roots)."""
        path_obj = Path(path).resolve()

        # Check path is under import roots
        allowed = False
        for root in self.import_roots:
            try:
                path_obj.relative_to(root)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise BufferError(f"Path {path} not under import roots: {self.import_roots}")

        if not path_obj.exists():
            raise BufferError(f"File not found: {path}")

        # Check file is not a symlink
        if path_obj.is_symlink():
            raise BufferError("Symlinks not allowed for import")

        np_dtype = self._dtype_from_str(dtype)
        expected_bytes = self._validate_shape(shape, np_dtype)
        actual_bytes = path_obj.stat().st_size
        if actual_bytes != expected_bytes:
            raise BufferError(f"File size {actual_bytes} != expected {expected_bytes}")

        buffer_id = self._make_buffer_id(name)
        mm, f = self.backend.create_buffer(session_id, buffer_id, expected_bytes)
        try:
            with path_obj.open("rb") as src:
                shutil.copyfileobj(src, f)
            f.flush()
            checksum = self.backend.compute_checksum(session_id, buffer_id)
        finally:
            mm.close()
            f.close()

        metadata = BufferMetadata(
            buffer_id=buffer_id,
            session_id=session_id,
            dtype=dtype,
            shape=shape,
            bytes=expected_bytes,
            checksum=checksum,
            owner=session_id,
            created_at=time.time(),
            ttl_ms=ttl_ms or self.default_ttl_ms,
            provenance={
                "source": "file",
                "name": name,
                "original_path": str(path_obj),
                "original_hash": checksum,
            },
        )
        self._metadata[buffer_id] = metadata
        return metadata.to_ref()

    def export_bytes(
        self,
        buffer_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Export buffer contents as bytes (bounded)."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")

        if length is None:
            length = meta.bytes - offset
        if offset + length > meta.bytes:
            raise BufferError("Export range exceeds buffer size")

        return self.backend.read_bytes(meta.session_id, buffer_id, offset, length)

    def export_file(
        self,
        buffer_id: str,
        path: str,
        offset: int = 0,
        length: int | None = None,
    ) -> None:
        """Export buffer to file (bounded)."""
        data = self.export_bytes(buffer_id, offset, length)
        Path(path).write_bytes(data)

    def read_view(
        self,
        buffer_id: str,
        slices: list[tuple[int, int, int]] | None = None,
        max_inline_bytes: int = 65536,
    ) -> tuple[np.ndarray | None, BufferRef | None]:
        """Read a buffer view. Returns (inline_data, derived_buffer_ref)."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")

        np_dtype = self._dtype_from_str(meta.dtype)
        mm, f = self.backend.open_buffer(meta.session_id, buffer_id, writable=False)
        try:
            arr = np.ndarray(meta.shape, dtype=np_dtype, buffer=mm)

            if slices:
                slice_objs = tuple(slice(s[0], s[1], s[2]) for s in slices)
                view = arr[slice_objs]
            else:
                view = arr

            if view.size * view.dtype.itemsize <= max_inline_bytes:
                # Return inline copy
                return view.copy(), None
            else:
                # Create derived buffer reference
                derived_id = self._make_buffer_id("view")
                # For now, just return the view info; derived buffer creation deferred
                return None, BufferRef(
                    buffer_id=derived_id,
                    dtype=meta.dtype,
                    shape=list(view.shape),
                    bytes=view.size * view.dtype.itemsize,
                    checksum="",
                    owner=meta.owner,
                    created_at=time.time(),
                )
        finally:
            mm.close()
            f.close()

    def write_view(
        self,
        buffer_id: str,
        slices: list[tuple[int, int, int]],
        data: np.ndarray,
        job_id: str,
        enforce_grant: bool = True,
    ) -> BufferRef:
        """Write to a buffer view (requires write grant unless disabled)."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")

        # Check grant
        slice_objs = tuple(slice(s[0], s[1], s[2]) for s in slices)
        grant = None
        if enforce_grant:
            grant = self.grant_manager.check_access(job_id, buffer_id, 0, meta.bytes, write=True)
            if not grant:
                raise BufferError(f"No write grant for job {job_id} on buffer {buffer_id}")

        mm, f = self.backend.open_buffer(meta.session_id, buffer_id, writable=True)
        try:
            np_dtype = self._dtype_from_str(meta.dtype)
            arr = np.ndarray(meta.shape, dtype=np_dtype, buffer=mm)
            arr[slice_objs] = data
            mm.flush()
            checksum = compute_checksum(mm[:])
            meta.checksum = checksum
            if grant is not None:
                meta.grants.append(grant)
        finally:
            mm.close()
            f.close()

        return meta.to_ref()

    def stat(self, buffer_id: str) -> BufferRef:
        """Get buffer metadata."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")
        return meta.to_ref()

    def get_buffer_path(self, buffer_id: str) -> str:
        """Return the shm path for a known buffer."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")
        return self.backend.get_buffer_path(meta.session_id, buffer_id)

    def checksum(
        self,
        buffer_id: str,
        algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
    ) -> str:
        """Compute buffer checksum."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")

        # Verify cached checksum matches
        if self.verify_on_read:
            computed = self.backend.compute_checksum(meta.session_id, buffer_id, algorithm)
            if computed != meta.checksum:
                raise BufferError(f"Checksum mismatch for {buffer_id}")
            return computed
        return meta.checksum

    def delete(self, buffer_id: str, force: bool = False) -> bool:
        """Delete a buffer. Returns True if deleted."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            return False

        if meta.ref_count > 0 and not force:
            raise BufferError(f"Buffer {buffer_id} has active references ({meta.ref_count})")

        # Revoke all grants
        self.grant_manager.revoke_buffer(buffer_id)

        # Delete from backend
        deleted = self.backend.delete_buffer(meta.session_id, buffer_id)
        if deleted:
            del self._metadata[buffer_id]
        return deleted

    def grant_read(
        self,
        job_id: str,
        buffer_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> Grant:
        """Grant read access to a job."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")
        length = length or (meta.bytes - offset)
        grant = self.grant_manager.grant(job_id, buffer_id, GrantType.READ, offset, length)
        meta.grants.append(grant)
        meta.ref_count += 1
        return grant

    def grant_write(
        self,
        job_id: str,
        buffer_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> Grant:
        """Grant write access to a job."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")
        length = length or (meta.bytes - offset)
        grant = self.grant_manager.grant(job_id, buffer_id, GrantType.WRITE, offset, length)
        meta.grants.append(grant)
        meta.ref_count += 1
        return grant

    def grant_read_write(
        self,
        job_id: str,
        buffer_id: str,
        offset: int = 0,
        length: int | None = None,
    ) -> Grant:
        """Grant read-write access to a job."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")
        length = length or (meta.bytes - offset)
        grant = self.grant_manager.grant(job_id, buffer_id, GrantType.READ_WRITE, offset, length)
        meta.grants.append(grant)
        meta.ref_count += 1
        return grant

    def release_grants(self, job_id: str) -> int:
        """Release all grants for a job. Returns count released."""
        grants = self.grant_manager.get_job_grants(job_id)
        count = self.grant_manager.revoke_job(job_id)
        per_buffer: dict[str, int] = {}
        for grant in grants:
            per_buffer[grant.buffer_id] = per_buffer.get(grant.buffer_id, 0) + 1
        for buffer_id, released in per_buffer.items():
            meta = self._metadata.get(buffer_id)
            if meta is None:
                continue
            meta.grants = [g for g in meta.grants if g.job_id != job_id]
            meta.ref_count = max(0, meta.ref_count - released)
        return count

    def reap_expired(self, now: float | None = None) -> list[str]:
        """Delete non-persistent buffers whose TTL has elapsed."""
        now = time.time() if now is None else now
        expired = [
            buffer_id
            for buffer_id, meta in self._metadata.items()
            if meta.ttl_ms is not None
            and not meta.persist
            and (now - meta.created_at) * 1000 >= meta.ttl_ms
        ]
        for buffer_id in expired:
            self.delete(buffer_id, force=True)
        return expired

    def list_buffers(self, session_id: str) -> list[BufferRef]:
        """List all buffers for a session."""
        return [m.to_ref() for m in self._metadata.values() if m.session_id == session_id]

    def cleanup_session(self, session_id: str) -> int:
        """Clean up all non-persistent buffers for a session."""
        to_delete = [
            bid
            for bid, meta in self._metadata.items()
            if meta.session_id == session_id and not meta.persist
        ]
        count = 0
        for bid in to_delete:
            if self.delete(bid, force=True):
                count += 1
        if not any(meta.session_id == session_id for meta in self._metadata.values()):
            self.backend.cleanup_session(session_id)
        return count

    @contextmanager
    def map_buffer(self, buffer_id: str, writable: bool = False):
        """Context manager for direct mmap access."""
        meta = self._metadata.get(buffer_id)
        if not meta:
            raise BufferError(f"Buffer not found: {buffer_id}")

        with self.backend.map_buffer(meta.session_id, buffer_id, writable) as mm:
            np_dtype = self._dtype_from_str(meta.dtype)
            arr = np.ndarray(meta.shape, dtype=np_dtype, buffer=mm)
            yield arr
