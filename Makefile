# Makefile for XL-Persistent-Kernel
#
# Targets:
#   make install      - Install package in development mode with dev dependencies
#   make test         - Run pytest with coverage
#   make lint         - Run ruff (lint + format check) and mypy
#   make format       - Auto-fix with ruff and black
#   make bench        - Run benchmark harness
#   make demo         - Run demo script
#   make cuda-stub    - Legacy stub (redirects to cuda-smoke)
#   make cuda-smoke   - Build and run CUDA smoke test
#   make cuda-bench   - Build and run CUDA measurement harness
#   make cuda-bench-large - Larger CUDA measurement run
#   make clean        - Remove build artifacts

.PHONY: install test lint format bench demo cuda-stub cuda-smoke cuda-bench cuda-bench-large clean help

# Default target
help:
	@echo "XL-Persistent-Kernel - Make targets:"
	@echo "  install     - Install package in development mode with dev dependencies"
	@echo "  test        - Run pytest with coverage"
	@echo "  lint        - Run ruff (lint + format check) and mypy"
	@echo "  format      - Auto-fix with ruff and black"
	@echo "  bench       - Run benchmark harness"
	@echo "  demo        - Run demo script"
	@echo "  cuda-smoke  - Build and run CUDA staging smoke tests (requires nvcc)"
	@echo "  cuda-bench  - Build and run CUDA measurement harness (requires nvcc)"
	@echo "  cuda-bench-large - Larger CUDA measurement run (requires nvcc)"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help"

install:
	pip install -e ".[dev]"

test: install
	python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ cuda/src/
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
	@echo "The legacy persistent_decode_stub has been replaced by the mega-kernel smoke test."
	@echo "Run 'make cuda-smoke' instead to build and run the full smoke test suite."
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "("; \
		make cuda-smoke; \
		echo ")"; \
	else \
		echo "nvcc not found - skipping CUDA build. Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads) to build."; \
	fi

cuda-smoke:
	@echo "Building and running CUDA staging smoke tests..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		echo "" && \
		echo "=== Running CUDA smoke tests ===" && \
		./build/cuda/xlpk_cuda_smoke --mode both --requests 4 --tokens 8; \
	else \
		echo "nvcc not found - skipping CUDA smoke tests."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the CUDA staging layer."; \
	fi

cuda-bench:
	@echo "Building and running CUDA measurement harness..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		echo "" && \
		echo "=== Running CUDA measurement harness ===" && \
		mkdir -p build/cuda && \
		./build/cuda/xlpk_cuda_smoke --mode both --requests 8 --tokens 128 --draft-len 4 --csv build/cuda/cuda_results.csv; \
	else \
		echo "nvcc not found - skipping CUDA measurement harness."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the CUDA measurement harness."; \
	fi

cuda-bench-large:
	@echo "Building and running large CUDA measurement harness..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		echo "" && \
		echo "=== Running large CUDA measurement harness ===" && \
		mkdir -p build/cuda && \
		./build/cuda/xlpk_cuda_smoke --mode both --requests 32 --tokens 512 --draft-len 4 --csv build/cuda/cuda_results_large.csv; \
	else \
		echo "nvcc not found - skipping large CUDA measurement harness."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the large CUDA measurement harness."; \
	fi

clean:
	rm -rf build dist *.egg-info
	rm -rf src/megakernel_lab/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf cuda/build build/cuda
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
