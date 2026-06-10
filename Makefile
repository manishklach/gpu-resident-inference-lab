# Makefile for XL-Persistent-Kernel
#
# Targets:
#   make install      - Install package in development mode with dev dependencies
#   make test         - Run pytest with coverage
#   make lint         - Run ruff (lint + format check) and mypy
#   make format       - Auto-fix with ruff and black
#   make bench        - Run benchmark harness
#   make demo         - Run demo script
#   make cuda-stub    - Build CUDA persistent-kernel stub (optional, requires nvcc)

.PHONY: install test lint format bench demo cuda-stub clean help

# Default target
help:
	@echo "XL-Persistent-Kernel - Make targets:"
	@echo "  install     - Install package in development mode with dev dependencies"
	@echo "  test        - Run pytest with coverage"
	@echo "  lint        - Run ruff (lint + format check) and mypy"
	@echo "  format      - Auto-fix with ruff and black"
	@echo "  bench       - Run benchmark harness"
	@echo "  demo        - Run demo script"
	@echo "  cuda-stub   - Build CUDA persistent-kernel stub (requires nvcc)"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help"

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

bench:
	python -c "from megakernel_lab.bench import BenchmarkRunner; print(BenchmarkRunner().run())"

demo:
	python -m megakernel_lab.demo

cuda-stub:
	@echo "Building CUDA persistent-kernel stub..."
	@if command -v nvcc >/dev/null 2>&1; then \
		mkdir -p cuda/build && cd cuda/build && cmake .. && make -j$$(nproc); \
		echo "CUDA stub built successfully at cuda/build/persistent_decode_stub"; \
	else \
		echo "nvcc not found - skipping CUDA build. Install CUDA toolkit to enable."; \
		exit 1; \
	fi

clean:
	rm -rf build dist *.egg-info
	rm -rf src/megakernel_lab/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf cuda/build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
