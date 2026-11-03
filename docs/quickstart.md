# LMPU Quickstart

## 1. Install

```bash
uv sync --dev --extra host --extra dashboard --extra bench
uv run python -c "import lmpu; print(lmpu.__version__)"
```

`vllm` and `decoder` extras are isolated; see `docs/optional-vllm.md`.

## 2. Check the environment

```bash
uv run lmpu doctor
```

## 3. Run the canonical demos

```bash
uv run lmpu demo canonical
```

Expected output ends with three `PASS` lines:

- `offload-recovery` — a worker crashes and the next job succeeds.
- `sleep-proof` — S1 park freezes forwards and resume matches the baseline.
- `context-protection` — a 16 MiB buffer stays under the 64-token context bound.

## 4. Start the runtime

In one terminal:

```bash
uv run lmpu serve --socket /tmp/lmpu/run.sock --trace-dir traces
```

## 5. Call it from Python

```python
import asyncio

from lmpu.sdk import LMPUClient


async def main() -> None:
    async with await LMPUClient.connect("/tmp/lmpu/run.sock") as client:
        result = await client.submit(
            "bigint_add",
            processor="pure.func",
            inputs={"a": 40, "b": 2},
            mode="sync",
        )
        print(result.status, result.result)


asyncio.run(main())
```

The SDK also exposes `poll`, `wait`, `cancel`, `read`, `import_buffer`,
`sleep`, and `wake`.

## 6. Inspect a trace

```bash
uv run lmpu trace latest --dir traces --timeline
uv run lmpu trace latest --dir traces --gaps --redact
uv run lmpu trace latest --dir traces --replay --speed 1000
```

`--gaps` exits `2` when a session sequence is incomplete. `--json` emits the
records for tooling.

## 7. Open the dashboard

In a second terminal:

```bash
uv run lmpu mission-control --socket /tmp/lmpu/run.sock --port 8080
```

Open `http://127.0.0.1:8080` for the parked-host state, jobs, buffers, and live
events. The dashboard is read-only.

## 8. Scaffold a processor plugin

```bash
uv run lmpu plugin hello --template basic
```

This writes `plugins/hello/` with a `Processor` subclass, a package
`__init__.py`, and a README. Register the generated processor with your worker
configuration.

## 9. Run a benchmark smoke test

```bash
uv run python -m lmpu.bench.run_benchmark --config configs/benchmarks.toml \
  --workload bigint_mul --baseline lmpu.sync --trials 1
```

Reports render with `make report`.

## 10. Frozen contracts

- Protocol RFC: `docs/rfcs/0001-protocol-v1.md`
- Protocol schema: `src/lmpu/schema/protocol-v1.schema.json`
- Trace schema: `src/lmpu/schema/trace-v1.schema.json`

Regenerate schemas after protocol changes:

```bash
uv run python scripts/export_schema.py
```
