"""Property tests for typed buffers and bounded slices."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lmpu.memory import BufferManager, TempSHMBackend

pytestmark = pytest.mark.property

DTYPES = [
    np.dtype("int8"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("float32"),
    np.dtype("float64"),
]


@pytest.fixture
def manager():
    backend = TempSHMBackend()
    buffer_manager = BufferManager(backend=backend)
    yield buffer_manager
    backend.cleanup()


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    dtype=st.sampled_from(DTYPES),
    shape=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=3).map(tuple),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_import_and_full_read_round_trip(
    manager: BufferManager,
    dtype: np.dtype,
    shape: tuple[int, ...],
    seed: int,
) -> None:
    data = np.random.default_rng(seed).standard_normal(shape).astype(dtype)
    ref = manager.import_bytes(
        session_id="s_prop",
        name="prop",
        data=data.tobytes(),
        dtype=dtype.name,
        shape=shape,
    )

    inline, derived = manager.read_view(ref.buffer_id, max_inline_bytes=10_000_000)

    assert derived is None
    assert inline is not None
    np.testing.assert_array_equal(inline, data)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    rows=st.integers(min_value=1, max_value=12),
    cols=st.integers(min_value=1, max_value=12),
    row_start=st.integers(min_value=0, max_value=12),
    row_stop=st.integers(min_value=0, max_value=12),
    col_start=st.integers(min_value=0, max_value=12),
    col_stop=st.integers(min_value=0, max_value=12),
)
def test_2d_slice_matches_numpy(
    manager: BufferManager,
    rows: int,
    cols: int,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> None:
    data = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    ref = manager.import_bytes(
        session_id="s_prop",
        name="slice",
        data=data.tobytes(),
        dtype="int64",
        shape=(rows, cols),
    )
    row_start, row_stop = sorted((row_start % rows, row_stop % rows))
    col_start, col_stop = sorted((col_start % cols, col_stop % cols))
    slices = [[row_start, row_stop, 1], [col_start, col_stop, 1]]

    inline, derived = manager.read_view(
        ref.buffer_id,
        slices=slices,
        max_inline_bytes=10_000_000,
    )

    assert derived is None
    assert inline is not None
    np.testing.assert_array_equal(
        inline,
        data[row_start:row_stop, col_start:col_stop],
    )


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    rows=st.integers(min_value=2, max_value=8),
    cols=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_write_view_updates_only_the_requested_slice(
    manager: BufferManager,
    rows: int,
    cols: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    original = rng.integers(-100, 100, size=(rows, cols), dtype=np.int64)
    ref = manager.import_bytes(
        session_id="s_prop",
        name="write",
        data=original.tobytes(),
        dtype="int64",
        shape=(rows, cols),
    )
    manager.grant_write("job_prop", ref.buffer_id)

    replacement = rng.integers(-100, 100, size=(1, cols), dtype=np.int64)
    updated = manager.write_view(
        ref.buffer_id,
        slices=[[1, 2, 1], [0, cols, 1]],
        data=replacement,
        job_id="job_prop",
    )

    inline, _ = manager.read_view(ref.buffer_id, max_inline_bytes=10_000_000)
    assert inline is not None
    expected = original.copy()
    expected[1:2, :] = replacement
    np.testing.assert_array_equal(inline, expected)
    assert updated.checksum == manager.stat(ref.buffer_id).checksum


def test_oversized_view_returns_derived_reference(manager: BufferManager) -> None:
    data = np.arange(64, dtype=np.int64)
    ref = manager.import_bytes(
        session_id="s_prop",
        name="large",
        data=data.tobytes(),
        dtype="int64",
        shape=(64,),
    )

    inline, derived = manager.read_view(ref.buffer_id, max_inline_bytes=8)

    assert inline is None
    assert derived is not None
    assert derived.bytes == data.nbytes
    assert derived.shape == [64]
