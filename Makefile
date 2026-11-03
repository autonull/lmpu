# LMPU Makefile

.PHONY: help setup check test-core test-gpu test-failure test-security test-fuzz demo mission-control bench bench-smoke soak release-check clean

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
	uv sync --all-extras
	uv pip install -e ".[dev,host]"

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
	python -m lmpu.demos 01
	python -m lmpu.demos 02
	python -m lmpu.demos 03
	python -m lmpu.demos 10

mission-control:
	python -m lmpu.demos 12

bench-smoke:
	uv run python -m lmpu.bench.run_benchmark --workload smoke --trials 3

bench:
	uv run python -m lmpu.bench.run_benchmark --config configs/benchmarks.toml --workload all --trials 20

soak:
	pytest -m "soak" -v --timeout=3600

release-check: check test-core test-gpu test-failure test-security bench

clean:
	rm -rf .venv __pycache__ src/lmpu/__pycache__ .mypy_cache .ruff_cache .pytest_cache
	rm -rf traces benchmarks/results/*.jsonl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

doctor:
	python -m lmpu.doctor