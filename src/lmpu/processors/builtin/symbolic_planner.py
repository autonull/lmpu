"""LMPU Processors - Symbolic planner for graph algorithms."""

from __future__ import annotations

import heapq
import json
from collections import deque
from typing import Any

from ...memory.checksum import compute_checksum
from ..base import JobContext, Manifest, Processor, ProcessorContext, Result


class SymbolicPlannerProcessor(Processor):
    """Symbolic planner for graph algorithms and constraint solving."""

    manifest = Manifest(
        name="symbolic.planner",
        version="0.1.0",
        protocol="lmpu/1",
        ops=[
            {"name": "shortest_path", "inputs": ["graph", "source", "target"], "outputs": ["path", "distance"]},
            {"name": "bfs_plan", "inputs": ["graph", "start", "goals"], "outputs": ["plan"]},
            {"name": "constraint_solve", "inputs": ["variables", "constraints"], "outputs": ["solution"]},
            {"name": "schedule", "inputs": ["tasks", "resources", "constraints"], "outputs": ["schedule"]},
        ],
        limits={
            "max_buffer_bytes": 10_000_000,
            "max_batch": 1,
        },
        determinism=["strict"],
        latency_class="ms",
        streaming=False,
        progress=True,
        interactive=True,
        persistent_state=False,
        batching=False,
        zero_copy=False,
        sandbox=["subprocess"],
        executor="process",
    )

    async def setup(self, ctx: ProcessorContext) -> None:
        pass

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        ctx.progress(0.0, f"{op} started")

        try:
            if op == "shortest_path":
                return await self._shortest_path(inputs, ctx)
            elif op == "bfs_plan":
                return await self._bfs_plan(inputs, ctx)
            elif op == "constraint_solve":
                return await self._constraint_solve(inputs, ctx)
            elif op == "schedule":
                return await self._schedule(inputs, ctx)
            else:
                return Result(
                    success=False,
                    error_code="unknown_op",
                    error_message=f"Unknown operation: {op}",
                )
        except Exception as e:
            return Result(
                success=False,
                error_code="execution_error",
                error_message=str(e),
                retriable=False,
            )
        finally:
            ctx.progress(1.0, f"{op} finished")

    async def _shortest_path(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """Dijkstra's algorithm for shortest path."""
        graph = inputs.get("graph", {})
        source = inputs.get("source")
        target = inputs.get("target")

        # Graph format: {"nodes": [...], "edges": [{"from": u, "to": v, "weight": w}, ...]}
        # Or adjacency: {u: {v: weight, ...}, ...}

        if isinstance(graph, dict) and "edges" in graph:
            # Convert to adjacency
            adj = {}
            for edge in graph["edges"]:
                u, v, w = edge["from"], edge["to"], edge.get("weight", 1)
                if u not in adj:
                    adj[u] = {}
                adj[u][v] = w
        else:
            adj = graph

        # Dijkstra
        dist = {source: 0}
        prev = {}
        pq = [(0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == target:
                break

            for v, w in adj.get(u, {}).items():
                alt = d + w
                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

        # Reconstruct path
        if target not in dist:
            return Result(
                success=True,
                data={"path": [], "distance": None, "found": False},
                checksum="",
            )

        path = []
        curr = target
        while curr in prev:
            path.append(curr)
            curr = prev[curr]
        path.append(source)
        path.reverse()

        result_data = {
            "path": path,
            "distance": dist[target],
            "found": True,
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _bfs_plan(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """BFS planning for goal reaching."""
        graph = inputs.get("graph", {})
        start = inputs.get("start")
        goals = inputs.get("goals", [])

        if isinstance(graph, dict) and "edges" in graph:
            adj = {}
            for edge in graph["edges"]:
                u, v = edge["from"], edge["to"]
                if u not in adj:
                    adj[u] = []
                adj[u].append(v)
        else:
            adj = graph

        # Multi-goal BFS
        queue = deque([(start, [start])])
        visited = {start}
        found_goals = {}

        while queue and len(found_goals) < len(goals):
            node, path = queue.popleft()

            if node in goals and node not in found_goals:
                found_goals[node] = path

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        result_data = {
            "plans": {goal: found_goals.get(goal) for goal in goals},
            "found": list(found_goals.keys()),
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _constraint_solve(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """Simple constraint satisfaction (placeholder)."""
        variables = inputs.get("variables", {})
        constraints = inputs.get("constraints", [])

        # Placeholder: return variables as-is
        result_data = {
            "solution": variables,
            "satisfied": True,
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _schedule(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """Simple task scheduling (placeholder)."""
        tasks = inputs.get("tasks", [])
        resources = inputs.get("resources", {})
        constraints = inputs.get("constraints", [])

        # Placeholder: return tasks in order
        schedule = [{"task": t, "start": i, "end": i+1} for i, t in enumerate(tasks)]

        result_data = {
            "schedule": schedule,
            "makespan": len(tasks),
        }

        checksum = compute_checksum(json.dumps(result_data, sort_keys=True).encode())

        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def cancel(self, job_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass
