"""Canonical demos must pass with executable assertions."""

from __future__ import annotations

import pytest

from lmpu.demos.canonical import CANONICAL_DEMOS, run_canonical

pytestmark = pytest.mark.integration


async def test_all_canonical_demos_pass() -> None:
    results = await run_canonical()

    assert [result.name for result in results] == list(CANONICAL_DEMOS)
    failures = [result.details for result in results if not result.passed]
    assert not failures, failures


async def test_sleep_proof_reports_frozen_forwards() -> None:
    from lmpu.demos.canonical import sleep_proof

    result = await sleep_proof()

    assert result.passed is True
    assert result.details["during_sleep"] == ""
    assert result.details["frozen"] == result.details["after_sleep"]


async def test_context_protection_reports_bounded_tokens() -> None:
    from lmpu.demos.canonical import context_protection

    result = await context_protection()

    assert result.passed is True
    assert result.details["result_bytes"] == 16 * 1024 * 1024
    assert result.details["derived"] is True
    assert result.details["context_tokens"] <= 64
