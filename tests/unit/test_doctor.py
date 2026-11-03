"""Doctor diagnostics stay usable without optional sandbox backends."""

from __future__ import annotations

import pytest

from lmpu.doctor import check_sandbox

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend_present", [False, True])
def test_sandbox_check_ignores_optional_backends(
    monkeypatch: pytest.MonkeyPatch, backend_present: bool
) -> None:
    monkeypatch.setattr(
        "lmpu.doctor.shutil.which", lambda _: "/usr/bin/tool" if backend_present else None
    )

    result = check_sandbox()

    assert result["ok"] is True
    assert "optional backends" in result["details"]
