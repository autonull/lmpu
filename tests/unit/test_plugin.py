"""Plugin scaffolding generates an importable processor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lmpu.plugin import create_plugin


def test_create_plugin_writes_importable_package(tmp_path: Path) -> None:
    target = create_plugin("hello_world", template="basic", root=tmp_path / "plugins")

    assert (target / "__init__.py").exists()
    assert (target / "processor.py").exists()
    assert (target / "README.md").read_text(encoding="utf-8").startswith("# hello_world")

    spec = importlib.util.spec_from_file_location("generated_plugin", target / "processor.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    processor = module.HelloWorldProcessor()
    assert processor.manifest.name == "hello_world"


def test_create_plugin_rejects_duplicate_and_bad_template(tmp_path: Path) -> None:
    create_plugin("hello", root=tmp_path / "plugins")

    with pytest.raises(FileExistsError):
        create_plugin("hello", root=tmp_path / "plugins")
    with pytest.raises(ValueError):
        create_plugin("other", template="unknown", root=tmp_path / "plugins")
    with pytest.raises(ValueError):
        create_plugin("9bad", root=tmp_path / "plugins")
