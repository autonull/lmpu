"""LMPU Kernel - Capability registry and routing."""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.messages import CapabilityDescriptor


@dataclass
class ProcessorRegistration:
    """Registered processor with manifest."""

    manifest: CapabilityDescriptor
    executor: str  # inline, thread, process, remote, gpu
    entry_point: str | None = None  # For external processors
    is_builtin: bool = True


class CapabilityRegistry:
    """Registry of processor capabilities."""

    def __init__(self):
        self._processors: dict[str, ProcessorRegistration] = {}

    def register(self, registration: ProcessorRegistration) -> None:
        """Register a processor."""
        self._processors[registration.manifest.name] = registration

    def unregister(self, name: str) -> bool:
        """Unregister a processor."""
        return self._processors.pop(name, None) is not None

    def get(self, name: str) -> ProcessorRegistration | None:
        return self._processors.get(name)

    def list_processors(self) -> list[CapabilityDescriptor]:
        return [r.manifest for r in self._processors.values()]

    def find_by_op(
        self,
        op: str,
        memory_limit: int | None = None,
        determinism: str | None = None,
        sandbox: str | None = None,
    ) -> list[ProcessorRegistration]:
        """Find processors supporting an operation with optional filters."""
        results = []
        for reg in self._processors.values():
            # Check op support
            op_supported = False
            for op_def in reg.manifest.ops:
                if op_def.get("name") == op:
                    op_supported = True
                    break
            if not op_supported:
                continue

            # Check memory limit
            if memory_limit is not None:
                max_mem = reg.manifest.limits.get("max_buffer_bytes", 0)
                if max_mem < memory_limit:
                    continue

            # Check determinism
            if determinism and determinism not in reg.manifest.determinism:
                continue

            # Check sandbox
            if sandbox and sandbox not in reg.manifest.sandbox:
                continue

            results.append(reg)

        return results

    def get_all_capabilities(self) -> list[CapabilityDescriptor]:
        return self.list_processors()


def score_processor(
    reg: ProcessorRegistration,
    explicit_preference: str | None = None,
    current_load: float = 0.0,
    latency_class: str = "ms",
) -> float:
    """Score a processor for routing (higher is better)."""
    score = 0.0

    # Explicit preference
    if explicit_preference and reg.manifest.name == explicit_preference:
        score += 1000

    # Latency class preference (lower is better for latency)
    latency_scores = {"us": 100, "ms": 10, "s": 1}
    score += latency_scores.get(reg.manifest.latency_class, 1)

    # Load factor (prefer less loaded)
    score -= current_load * 10

    # Batching support
    if reg.manifest.batching:
        score += 5

    # Zero-copy
    if reg.manifest.zero_copy:
        score += 5

    return score


def route_job(
    registry: CapabilityRegistry,
    op: str,
    explicit_processor: str | None = None,
    memory_limit: int | None = None,
    determinism: str | None = None,
    sandbox: str | None = None,
    current_loads: dict[str, float] | None = None,
    allowed_processors: set[str] | None = None,
) -> tuple[ProcessorRegistration | None, list[ProcessorRegistration], str]:
    """Route a job to the best authorized processor.

    Capability discovery is registry-wide; ``allowed_processors`` is the
    session authorization boundary and never changes the advertised set.
    """
    current_loads = current_loads or {}
    candidates = registry.find_by_op(op, memory_limit, determinism, sandbox)
    if allowed_processors is not None:
        candidates = [c for c in candidates if c.manifest.name in allowed_processors]

    if explicit_processor:
        reg = registry.get(explicit_processor)
        if not reg:
            return None, [], f"unknown_processor: {explicit_processor}"
        if reg not in candidates:
            if allowed_processors is not None and explicit_processor not in allowed_processors:
                return None, candidates, f"access_denied: {explicit_processor}"
            return (
                None,
                candidates,
                (f"capability_mismatch: {explicit_processor} does not support {op}"),
            )
        return reg, candidates, "explicit"

    if not candidates:
        reason = "access_denied" if allowed_processors is not None else "capability_mismatch"
        return None, [], reason

    best = None
    best_score = float("-inf")
    for reg in candidates:
        load = current_loads.get(reg.manifest.name, 0.0)
        score = score_processor(reg, current_load=load, latency_class=reg.manifest.latency_class)
        if score > best_score:
            best_score = score
            best = reg

    reason = "scored" if len(candidates) > 1 else "sole_candidate"
    return best, candidates, reason
