# LMPU Makefile

.PHONY: help setup check schema test-core test-gpu test-failure test-security test-fuzz demo mission-control bench bench-smoke soak release-check release-build release-sbom release-checksums clean

# Default target
help:
	@echo "LMPU Make targets:"
	@echo "  setup           - Install dependencies with uv"
	@echo "  check           - Run linting and type checking"
	@echo "  test-core       - Run CPU-only tests (unit, property, contract, integration, scenario)"
	@echo "  test-gpu        - Run GPU/model tests"
	@echo "  test-failure    - Run failure injection tests"
	@echo "  test-security   - Run security tests"
	@echo "  test-fuzz       - Run fuzzing tests"
	@echo "  demo            - Run demos 01-11"
	@echo "  mission-control - Start Mission Control dashboard"
	@echo "  bench-smoke     - Quick benchmark sanity check"
	@echo "  bench           - Full benchmark suite"
	@echo "  soak            - 1-hour soak test"
	@echo "  release-check   - All release gates"
	@echo "  clean           - Clean build artifacts"

setup:
	uv sync --dev --extra host --extra dashboard --extra bench
	uv pip install -e ".[dev,host]"

schema:
	uv run python scripts/export_schema.py

check:
	ruff check src tests
	ruff format --check src tests
	mypy src/lmpu

test-core:
	pytest -m "unit or property or contract or integration or scenario" -v

test-gpu:
	pytest -m "gpu or model" -v

test-failure:
	pytest -m "failure" -v

test-security:
	pytest -m "security" -v

test-fuzz:
	pytest -m "fuzz" -v

demo:
	uv run lmpu demo canonical

mission-control:
	uv run lmpu mission-control

bench-smoke:
	uv run python -m lmpu.bench.run_benchmark --workload smoke --trials 3

bench:
	uv run python -m lmpu.bench.run_benchmark --config configs/benchmarks.toml --workload all

report:
	uv run python -m lmpu.bench.report --latest

soak:
	pytest -m "soak" -v --timeout=3600

release-check: check test-core test-gpu test-failure test-security bench

release-build:
	uv build
	$(MAKE) release-checksums

release-sbom:
	uvx pip-audit -f cyclonedx-json -o dist/sbom.json

release-checksums:
	cd dist && sha256sum *.whl *.tar.gz > SHA256SUMS

clean:
	rm -rf .venv __pycache__ src/lmpu/__pycache__ .mypy_cache .ruff_cache .pytest_cache
	rm -rf traces benchmarks/results/*.jsonl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

doctor:
	python -m lmpu.doctor