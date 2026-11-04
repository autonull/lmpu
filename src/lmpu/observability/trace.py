"""LMPU Observability - Trace viewer and replay."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from lmpu.protocol.messages import Event, EventKind


class TraceViewer:
    """View and analyze trace files."""

    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = Path(trace_dir)

    def list_traces(self) -> list[Path]:
        """List all trace files."""
        return sorted(self.trace_dir.glob("*.jsonl"))

    def load_trace(self, path: Path) -> list[dict[str, Any]]:
        """Load trace file as list of records."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def filter_by_type(self, records: list[dict], event_type: str) -> list[dict]:
        """Filter records by type."""
        return [r for r in records if r.get("type") == event_type]

    def filter_by_session(self, records: list[dict], session_id: str) -> list[dict]:
        """Filter records by session."""
        return [r for r in records if r.get("session") == session_id]

    def get_timeline(self, records: list[dict]) -> list[dict]:
        """Get sorted timeline of events."""
        return sorted(records, key=lambda r: r.get("mono", 0))

    def compute_stats(self, records: list[dict]) -> dict[str, Any]:
        """Compute trace statistics."""
        if not records:
            return {}

        mono_times = [r.get("mono", 0) for r in records if "mono" in r]
        wall_times = [r.get("ts", 0) for r in records if "ts" in r]

        forwards = [r for r in records if r.get("type") == "host_forward"]
        events = [r for r in records if r.get("type", "").startswith("event_")]

        return {
            "total_records": len(records),
            "duration_mono": max(mono_times) - min(mono_times) if mono_times else 0,
            "duration_wall": max(wall_times) - min(wall_times) if wall_times else 0,
            "host_forwards": len(forwards),
            "total_tokens_generated": sum(r.get("tokens", 0) for r in forwards),
            "events": len(events),
            "sessions": len(set(r.get("session") for r in records if r.get("session"))),
            "jobs": len(set(r.get("job_id") for r in records if r.get("job_id"))),
        }


class TraceReplayer:
    """Replay trace through dashboard."""

    def __init__(self, trace_path: Path):
        self.trace_path = trace_path
        self.records = []
        self._load()

    def _load(self) -> None:
        with open(self.trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def replay(self, speed: float = 1.0) -> Iterator[dict]:
        """Replay trace events with timing."""
        if not self.records:
            return

        start_mono = self.records[0].get("mono", 0)
        last_mono = start_mono

        for record in self.records:
            mono = record.get("mono", last_mono)
            delay = (mono - last_mono) / speed
            if delay > 0:
                time.sleep(delay)
            last_mono = mono
            yield record


def export_trace_archive(trace_dir: str, output_path: str) -> None:
    """Export trace directory as compressed archive."""
    import tarfile

    trace_path = Path(trace_dir)
    with tarfile.open(output_path, "w:gz") as tar:
        for trace_file in trace_path.glob("*.jsonl"):
            tar.add(trace_file, arcname=trace_file.name)


def import_trace_archive(archive_path: str, trace_dir: str) -> None:
    """Import trace archive."""
    import tarfile

    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(trace_path)
