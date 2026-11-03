"""Processor plugin scaffolding."""

from __future__ import annotations

import re
from pathlib import Path

_TEMPLATES = {
    "basic": "Return the input value unchanged.",
    "python": "Run a bounded Python computation in the worker.",
    "vector": "Operate on a vector buffer.",
}

_PROCESSOR_TEMPLATE = '''"""Generated LMPU processor plugin: {name}."""

from __future__ import annotations

from typing import Any

from lmpu.processors.base import (
    JobContext,
    Manifest,
    Processor,
    ProcessorContext,
    Result,
)


class {class_name}Processor(Processor):
    """{name} processor ({template} template)."""

    manifest = Manifest(
        name="{name}",
        ops=[{{"name": "run", "inputs": {{"value": "any"}}, "outputs": {{"value": "any"}}}}],
        latency_class="ms",
    )

    async def setup(self, ctx: ProcessorContext) -> None:
        self.config = ctx.runtime_config

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        value = job.get("inputs", {{}}).get("value")
        return Result(data={{"value": value}})
'''

_README_TEMPLATE = """# {name}

Generated LMPU processor plugin ({template} template).

{description}

Install the package next to your LMPU deployment and register the processor in
the worker configuration. See `docs/quickstart.md`.
"""


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    if not slug or slug[0].isdigit():
        raise ValueError(f"Invalid plugin name: {name!r}")
    return slug


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def create_plugin(
    name: str,
    template: str = "basic",
    *,
    root: Path | str = "plugins",
) -> Path:
    """Create a processor plugin package and return its directory."""
    if template not in _TEMPLATES:
        raise ValueError(f"Unknown template {template!r}; choose {tuple(_TEMPLATES)}")
    slug = _slug(name)
    target = Path(root) / slug
    if target.exists():
        raise FileExistsError(f"Plugin already exists: {target}")
    target.mkdir(parents=True)

    class_name = _class_name(slug)
    replacement = {"name": slug, "class_name": class_name, "template": template}
    (target / "__init__.py").write_text(
        f'"""LMPU plugin {slug}."""\n\nfrom .processor import {class_name}Processor\n\n'
        f'__all__ = ["{class_name}Processor"]\n',
        encoding="utf-8",
    )
    (target / "processor.py").write_text(
        _PROCESSOR_TEMPLATE.format(**replacement), encoding="utf-8"
    )
    (target / "README.md").write_text(
        _README_TEMPLATE.format(**replacement, description=_TEMPLATES[template]),
        encoding="utf-8",
    )
    return target
