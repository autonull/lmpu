"""LMPU Memory - Checksum utilities."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

# Try to import blake3, fallback to blake2b if not available
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


class ChecksumAlgorithm(str, Enum):
    """Supported checksum algorithms."""
    BLAKE3 = "blake3"
    BLAKE2B = "blake2b"
    SHA256 = "sha256"


def _get_blake3_hasher():
    """Get a BLAKE3 hasher, using built-in hashlib if available, otherwise blake3 module."""
    if hasattr(hashlib, 'blake3'):
        return hashlib.blake3()
    elif HAS_BLAKE3:
        return blake3.blake3()
    else:
        # Fallback to BLAKE2b
        return hashlib.blake2b(digest_size=32)


def compute_checksum(
    data: bytes,
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
) -> str:
    """Compute checksum of bytes."""
    if algorithm == ChecksumAlgorithm.BLAKE3:
        hasher = _get_blake3_hasher()
        hasher.update(data)
        return f"blake3:{hasher.hexdigest()}"
    elif algorithm == ChecksumAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b(digest_size=32)
        hasher.update(data)
        return f"blake2b:{hasher.hexdigest()}"
    elif algorithm == ChecksumAlgorithm.SHA256:
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_checksum_file(
    path: str,
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute checksum of a file incrementally."""
    if algorithm == ChecksumAlgorithm.BLAKE3:
        hasher = _get_blake3_hasher()
    elif algorithm == ChecksumAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b(digest_size=32)
    elif algorithm == ChecksumAlgorithm.SHA256:
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    prefix = "blake3" if algorithm == ChecksumAlgorithm.BLAKE3 else \
             "blake2b" if algorithm == ChecksumAlgorithm.BLAKE2B else "sha256"
    return f"{prefix}:{hasher.hexdigest()}"


def compute_checksum_mmap(
    mmap_obj,
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Compute checksum of an mmap object incrementally."""
    if algorithm == ChecksumAlgorithm.BLAKE3:
        hasher = _get_blake3_hasher()
    elif algorithm == ChecksumAlgorithm.BLAKE2B:
        hasher = hashlib.blake2b(digest_size=32)
    elif algorithm == ChecksumAlgorithm.SHA256:
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    size = len(mmap_obj)
    for offset in range(0, size, chunk_size):
        end = min(offset + chunk_size, size)
        hasher.update(mmap_obj[offset:end])

    prefix = "blake3" if algorithm == ChecksumAlgorithm.BLAKE3 else \
             "blake2b" if algorithm == ChecksumAlgorithm.BLAKE2B else "sha256"
    return f"{prefix}:{hasher.hexdigest()}"


def verify_checksum(
    data: bytes,
    expected: str,
    algorithm: ChecksumAlgorithm = ChecksumAlgorithm.BLAKE3,
) -> bool:
    """Verify data matches expected checksum."""
    computed = compute_checksum(data, algorithm)
    return computed == expected


def parse_checksum(checksum: str) -> tuple[ChecksumAlgorithm, str]:
    """Parse a checksum string into algorithm and digest."""
    if checksum.startswith("blake3:"):
        return ChecksumAlgorithm.BLAKE3, checksum[7:]
    elif checksum.startswith("blake2b:"):
        return ChecksumAlgorithm.BLAKE2B, checksum[8:]
    elif checksum.startswith("sha256:"):
        return ChecksumAlgorithm.SHA256, checksum[7:]
    else:
        # Default to blake3 for backward compatibility
        return ChecksumAlgorithm.BLAKE3, checksum