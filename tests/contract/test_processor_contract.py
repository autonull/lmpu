"""Processor contract suite.

Every built-in processor must tell the truth about its manifest and must return
a structured result for a representative operation and for an unknown op.
These tests are model-free and run in-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from lmpu.processors.base import (
    JobContext,
    Manifest,
    Processor,
    ProcessorContext,
    Result,
)
from lmpu.processors.builtin import (
    HardwareStubProcessor,
    NeuralSummaryProcessor,
    PureFunctionProcessor,
    PythonVMProcessor,
    SymbolicPlannerProcessor,
    VectorEngineProcessor,
)

pytestmark = pytest.mark.contract


def _noop(*args: Any, **kwargs: Any) -> None:
    return None


class RecordingJobContext(JobContext):
    """JobContext that records callbacks instead of talking to a kernel."""

    def __init__(
        self,
        job_id: str = "job_contract",
        session_id: str = "s_contract",
        buffers: dict[str, Any] | None = None,
    ):
        self.progress_calls: list[tuple[float, str | None]] = []
        self.input_requests: list[tuple[list[str], str]] = []
        super().__init__(
            job_id=job_id,
            session_id=session_id,
            buffers=buffers or {},
            send_progress=self._on_progress,
            send_needs_input=self._on_needs_input,
            submit_child=self._on_submit_child,
            log=_noop,
            trace=_noop,
            cancelled=lambda: False,
        )

    def _on_progress(self, fraction: float, message: str | None = None) -> None:
        self.progress_calls.append((fraction, message))

    def _on_needs_input(self, fields: list[str], prompt: str) -> None:
        self.input_requests.append((fields, prompt))

    async def _on_submit_child(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("contract case must not submit child jobs")


@dataclass(frozen=True)
class Case:
    processor_cls: type[Processor]
    op: str
    inputs: dict[str, Any]
    config: dict[str, Any]
    buffers: dict[str, Any] | None = None


CASES = [
    Case(
        PureFunctionProcessor,
        "bigint_add",
        {"a": "12345678901234567890", "b": "1"},
        {},
    ),
    Case(
        PythonVMProcessor,
        "run_program",
        {"program": "print('{\"ok\": true}')"},
        {"timeout_ms": 10000},
    ),
    Case(
        VectorEngineProcessor,
        "matmul",
        {},
        {"determinism": "strict"},
        buffers={"A": "buf_A", "B": "buf_B", "C": "buf_C"},
    ),
    Case(
        SymbolicPlannerProcessor,
        "shortest_path",
        {
            "graph": {
                "edges": [
                    {"from": "a", "to": "b", "weight": 1},
                    {"from": "b", "to": "c", "weight": 2},
                ]
            },
            "source": "a",
            "target": "c",
        },
        {},
    ),
    Case(
        NeuralSummaryProcessor,
        "summarize_text_buffer",
        {"text": "LMPU runs native jobs outside the host generation loop." * 10},
        {},
    ),
    Case(
        HardwareStubProcessor,
        "deterministic_compute",
        {"seed": 7, "operation": "hash"},
        {},
    ),
]

CASE_IDS = [case.processor_cls.__name__ for case in CASES]


async def _execute(case: Case) -> tuple[Result, RecordingJobContext]:
    processor = case.processor_cls()
    await processor.setup(ProcessorContext(runtime_config={}))
    ctx = RecordingJobContext(buffers=case.buffers)
    job = {"job_id": ctx.job_id, "op": case.op, "inputs": case.inputs, "config": case.config}
    try:
        result = await processor.execute(job, ctx)
    finally:
        await processor.shutdown()
    return result, ctx


def _assert_manifest(processor_cls: type[Processor]) -> Manifest:
    manifest = processor_cls.manifest
    assert manifest.name
    assert manifest.version
    assert manifest.protocol == "lmpu/1"
    assert manifest.ops, f"{manifest.name} advertises no ops"
    assert manifest.determinism
    assert manifest.executor in {"inline", "thread", "process"}

    op_names = [op.get("name") for op in manifest.ops]
    assert len(op_names) == len(set(op_names)), "duplicate op names in manifest"
    for op in manifest.ops:
        assert op.get("name")
        assert isinstance(op.get("inputs"), list)
        assert isinstance(op.get("outputs"), list)
    return manifest


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_manifest_is_well_formed(case: Case) -> None:
    _assert_manifest(case.processor_cls)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_representative_op_succeeds_with_checksum(case: Case) -> None:
    result, _ = await _execute(case)

    assert result.success is True, result.error_message
    assert isinstance(result.data, dict) and result.data
    assert result.checksum, f"{case.processor_cls.manifest.name} returned no checksum"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_progress_claim_is_backed_by_events(case: Case) -> None:
    manifest = case.processor_cls.manifest
    if not manifest.progress:
        pytest.skip(f"{manifest.name} does not claim progress")

    _, ctx = await _execute(case)
    assert ctx.progress_calls, f"{manifest.name} claims progress but emitted none"
    fractions = [fraction for fraction, _ in ctx.progress_calls]
    assert fractions == sorted(fractions)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
async def test_unknown_op_returns_structured_error(case: Case) -> None:
    processor = case.processor_cls()
    await processor.setup(ProcessorContext(runtime_config={}))
    try:
        result = await processor.execute(
            {"job_id": "job_bad", "op": "does_not_exist", "inputs": {}, "config": {}},
            RecordingJobContext("job_bad"),
        )
    finally:
        await processor.shutdown()

    assert result.success is False
    assert result.error_code == "unknown_op"


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "strict" in case.processor_cls.manifest.determinism],
    ids=[
        case.processor_cls.__name__
        for case in CASES
        if "strict" in case.processor_cls.manifest.determinism
    ],
)
async def test_strict_processors_are_reproducible(case: Case) -> None:
    first, _ = await _execute(case)
    second, _ = await _execute(case)

    assert first.data == second.data
    assert first.checksum == second.checksum
