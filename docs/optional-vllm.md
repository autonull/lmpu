# Optional vLLM and constrained-decoder environments

vLLM 0.33 pins `transformers` below 5.21 and its own `outlines-core`; the
dashboard extra pins `sse-starlette` below 3. Those pins cannot coexist in one
universal lock split. `pyproject.toml` declares them as conflicting extras, so
each environment resolves separately.

## Host (default reference path)

```bash
uv sync --dev --extra host
```

## vLLM spike

```bash
uv sync --dev --extra vllm
```

Do not combine `vllm` with `host`, `decoder`, `dashboard`, or `bench` in the
same environment. The lock file is the source of truth; `uv lock` resolves all
conflicting splits.

## Constrained decoding

```bash
uv sync --dev --extra host --extra decoder
```

## Verification

```bash
uv run python -c "import optuna, scipy, torchvision, pytest"
uv run python -m pytest tests/unit tests/integration -q
```
