"""LMPU Memory - Buffer grants and access control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import ulid


class GrantType(str, Enum):
    """Grant permission types."""

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


def _new_ulid() -> str:
    """Generate a new ULID."""
    return str(ulid.new())


@dataclass(frozen=True)
class Grant:
    """A grant allowing a job to access a buffer."""

    grant_id: str = field(default_factory=_new_ulid)
    job_id: str = ""
    buffer_id: str = ""
    grant_type: GrantType = GrantType.READ
    offset: int = 0
    length: Optional[int] = None  # None means full buffer

    def covers(self, offset: int, length: int) -> bool:
        """Check if this grant covers the requested range."""
        if self.grant_type == GrantType.READ and offset >= self.offset:
            end = self.offset + (self.length or float("inf"))
            return offset + length <= end
        if self.grant_type in (GrantType.WRITE, GrantType.READ_WRITE) and offset >= self.offset:
            end = self.offset + (self.length or float("inf"))
            return offset + length <= end
        return False

    def can_read(self) -> bool:
        return self.grant_type in (GrantType.READ, GrantType.READ_WRITE)

    def can_write(self) -> bool:
        return self.grant_type in (GrantType.WRITE, GrantType.READ_WRITE)


class GrantManager:
    """Manages buffer grants for jobs."""

    def __init__(self):
        self._grants: dict[str, list[Grant]] = {}  # job_id -> list of grants
        self._buffer_grants: dict[str, set[str]] = {}  # buffer_id -> set of grant_ids

    def grant(
        self,
        job_id: str,
        buffer_id: str,
        grant_type: GrantType,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> Grant:
        """Create a new grant."""
        grant = Grant(
            job_id=job_id,
            buffer_id=buffer_id,
            grant_type=grant_type,
            offset=offset,
            length=length,
        )
        self._grants.setdefault(job_id, []).append(grant)
        self._buffer_grants.setdefault(buffer_id, set()).add(grant.grant_id)
        return grant

    def revoke(self, grant_id: str) -> bool:
        """Revoke a specific grant."""
        for job_id, grants in self._grants.items():
            for i, grant in enumerate(grants):
                if grant.grant_id == grant_id:
                    grants.pop(i)
                    self._buffer_grants.get(grant.buffer_id, set()).discard(grant_id)
                    if not grants:
                        del self._grants[job_id]
                    return True
        return False

    def revoke_job(self, job_id: str) -> int:
        """Revoke all grants for a job. Returns count revoked."""
        grants = self._grants.pop(job_id, [])
        for grant in grants:
            self._buffer_grants.get(grant.buffer_id, set()).discard(grant.grant_id)
        return len(grants)

    def revoke_buffer(self, buffer_id: str) -> int:
        """Revoke all grants for a buffer. Returns count revoked."""
        grant_ids = self._buffer_grants.pop(buffer_id, set())
        count = 0
        for job_id, grants in list(self._grants.items()):
            for grant in grants[:]:
                if grant.grant_id in grant_ids:
                    grants.remove(grant)
                    count += 1
            if not grants:
                del self._grants[job_id]
        return count

    def get_job_grants(self, job_id: str) -> list[Grant]:
        """Get all grants for a job."""
        return self._grants.get(job_id, [])

    def get_buffer_grants(self, buffer_id: str) -> list[Grant]:
        """Get all grants for a buffer."""
        grant_ids = self._buffer_grants.get(buffer_id, set())
        result = []
        for grants in self._grants.values():
            for grant in grants:
                if grant.grant_id in grant_ids:
                    result.append(grant)
        return result

    def check_access(
        self,
        job_id: str,
        buffer_id: str,
        offset: int,
        length: int,
        write: bool = False,
    ) -> Optional[Grant]:
        """Check if job has access to buffer range. Returns grant if allowed."""
        grants = self._grants.get(job_id, [])
        for grant in grants:
            if grant.buffer_id != buffer_id:
                continue
            if write and not grant.can_write():
                continue
            if not write and not grant.can_read():
                continue
            if grant.covers(offset, length):
                return grant
        return None

    def has_active_grants(self, buffer_id: str) -> bool:
        """Check if buffer has any active grants."""
        return buffer_id in self._buffer_grants and len(self._buffer_grants[buffer_id]) > 0

    def clear(self) -> None:
        """Clear all grants."""
        self._grants.clear()
        self._buffer_grants.clear()
