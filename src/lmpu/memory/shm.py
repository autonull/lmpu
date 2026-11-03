"""LMPU Memory - Shared memory backend using /dev/shm."""

from __future__ import annotations

import mmap
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np

from .checksum import ChecksumAlgorithm, compute_checksum_mmap


class SHMError(Exception):
    """Shared memory errors."""
    pass


class SHMBackend:
    """POSIX shared memory backend using /dev/shm."""

    def __init__(self, base_dir: str = "/dev/shm/lmpu"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _session_dir(self, session_id: str) -> Path:
        """Get session directory, creating if needed."""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return session_dir

    def _buffer_path(self, session_id: str, buffer_id: str) -> Path:
        """Get buffer file path."""
        return self._session_dir(session_id) / buffer_id

    def get_buffer_path(self, session_id: str, buffer_id: str) -> str:
        """Return the filesystem path for a buffer, without opening it."""
        return str(self._buffer_path(session_id, buffer_id))

    def create_buffer(
        self,
        session_id: str,
        buffer_id: str,
        size: int,
    ) -> tuple[mmap.mmap, BinaryIO]:
        """Create a new shared memory buffer.
        
        Returns:
            Tuple of (mmap object, file handle). Caller must close both.
        """
        path = self._buffer_path(session_id, buffer_id)
        
        # Create file with exact size
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            return mm, os.fdopen(fd, "r+b")
        except Exception:
            os.close(fd)
            try:
                path.unlink()
            except Exception:
                pass
            raise

    def open_buffer(
        self,
        session_id: str,
        buffer_id: str,
        writable: bool = False,
    ) -> tuple[mmap.mmap, BinaryIO]:
        """Open an existing shared memory buffer."""
        path = self._buffer_path(session_id, buffer_id)
        if not path.exists():
            raise SHMError(f"Buffer not found: {buffer_id}")

        flags = os.O_RDWR if writable else os.O_RDONLY
        fd = os.open(path, flags)
        try:
            access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
            mm = mmap.mmap(fd, 0, access=access)
            return mm, os.fdopen(fd, "r+b" if writable else "rb")
        except Exception:
            os.close(fd)
            raise

    def delete_buffer(self, session_id: str, buffer_id: str) -> bool:
        """Delete a buffer file. Returns True if deleted."""
        path = self._buffer_path(session_id, buffer_id)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception:
            raise

    def buffer_exists(self, session_id: str, buffer_id: str) -> bool:
        """Check if buffer exists."""
        return self._buffer_path(session_id, buffer_id).exists()

    def get_buffer_size(self, session_id: str, buffer_id: str) -> int:
        """Get buffer size in bytes."""
        path = self._buffer_path(session_id, buffer_id)
        return path.stat().st_size

    def compute_checksum(
        self,
        session_id: str,
        buffer_id: str,
        algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
    ) -> str:
        """Compute checksum of buffer contents."""
        mm, f = self.open_buffer(session_id, buffer_id, writable=False)
        try:
            return compute_checksum_mmap(mm, algorithm)
        finally:
            mm.close()
            f.close()

    def read_bytes(
        self,
        session_id: str,
        buffer_id: str,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> bytes:
        """Read bytes from buffer."""
        mm, f = self.open_buffer(session_id, buffer_id, writable=False)
        try:
            if length is None:
                length = len(mm) - offset
            return bytes(mm[offset:offset + length])
        finally:
            mm.close()
            f.close()

    def write_bytes(
        self,
        session_id: str,
        buffer_id: str,
        data: bytes,
        offset: int = 0,
    ) -> int:
        """Write bytes to buffer."""
        mm, f = self.open_buffer(session_id, buffer_id, writable=True)
        try:
            mm[offset:offset + len(data)] = data
            mm.flush()
            return len(data)
        finally:
            mm.close()
            f.close()

    @contextmanager
    def map_buffer(
        self,
        session_id: str,
        buffer_id: str,
        writable: bool = False,
    ):
        """Context manager for mapping a buffer."""
        mm, f = self.open_buffer(session_id, buffer_id, writable)
        try:
            yield mm
        finally:
            mm.close()
            f.close()

    def list_buffers(self, session_id: str) -> list[str]:
        """List all buffers for a session."""
        session_dir = self._session_dir(session_id)
        return [f.name for f in session_dir.iterdir() if f.is_file()]

    def cleanup_session(self, session_id: str) -> int:
        """Remove all buffers for a session. Returns count deleted."""
        session_dir = self.base_dir / session_id
        if not session_dir.exists():
            return 0
        count = 0
        for f in session_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        try:
            session_dir.rmdir()
        except Exception:
            pass
        return count


class TempSHMBackend(SHMBackend):
    """Temporary SHM backend for testing (uses temp directory)."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lmpu_test_"))
        super().__init__(str(self.temp_dir))

    def cleanup(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)