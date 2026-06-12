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
#   make cuda-research-bench - Run standalone research-kernel benchmarks
#   make clean        - Remove build artifacts

.PHONY: install test lint format bench demo compare check-claims cuda-stub cuda-smoke cuda-bench cuda-bench-large cuda-research-bench clean help

# Default target
help:
	@echo "XL-Persistent-Kernel - Make targets:"
	@echo "  install     - Install package in development mode with dev dependencies"
	@echo "  test        - Run pytest with coverage"
	@echo "  lint        - Run ruff (lint + format check) and mypy"
	@echo "  format      - Auto-fix with ruff and black"
	@echo "  bench       - Run benchmark harness"
	@echo "  demo        - Run demo script"
	@echo "  check-claims - Scan docs for overclaiming phrases (advisory)"
	@echo "  compare     - Run Python bench + CUDA sweep and compare side-by-side"
	@echo "  cuda-smoke  - Build and run CUDA staging smoke tests (requires nvcc)"
	@echo "  cuda-bench  - Build and run CUDA measurement sweep (requires nvcc)"
	@echo "  cuda-bench-large - Larger CUDA measurement sweep (requires nvcc)"
	@echo "  cuda-research-bench - Run standalone research-kernel benchmarks (requires nvcc)"
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

compare:
	@echo "Running Python CPU bench and CUDA measurement comparison..."
	@mkdir -p results
	@echo ""
	@echo "=== Python CPU Simulator ==="
	@python -c "from megakernel_lab.bench import BenchmarkRunner; BenchmarkRunner(batch_sizes=[1,2,4,8], block_sizes=[1,2,4]).run()" && \
	 echo "" && \
	 PY_CSV=$$(ls -t results/bench_*.csv 2>/dev/null | head -1) && \
	 if [ -z "$$PY_CSV" ]; then \
	   echo "Error: Python bench did not produce a CSV"; \
	   exit 1; \
	 fi && \
	 echo "Python CSV: $$PY_CSV" && \
	 if command -v nvcc >/dev/null 2>&1; then \
	   echo "" && \
	   echo "=== CUDA Measurement ===" && \
	   cmake -S cuda -B build/cuda && \
	   cmake --build build/cuda && \
	   ./build/cuda/xlpk_cuda_smoke --mode sweep --csv results/cuda_compare_$$(date +%Y%m%d_%H%M%S).csv && \
	   CU_CSV=$$(ls -t results/cuda_compare_*.csv 2>/dev/null | head -1) && \
	   echo "" && \
	   echo "=== Side-by-Side Comparison ===" && \
	   python scripts/compare_metrics.py "$$PY_CSV" "$$CU_CSV"; \
	 else \
	   echo "" && \
	   echo "=== Python-Only (CUDA not available) ===" && \
	   echo "Python CSV: $$PY_CSV"; \
	   echo "Install the CUDA toolkit to include CUDA measurements."; \
	 fi

check-claims:
	@python scripts/check_claims.py docs/ README.md; \
	 echo "  (advisory — does not block CI)"

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
	@echo "Building and running CUDA measurement harness (sweep)..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		mkdir -p results && \
		echo "" && \
		echo "=== Sweep: requests [2,4,8,16] x tokens [32,64,128] x draft_len [1,4,8] ===" && \
		./build/cuda/xlpk_cuda_smoke --mode sweep --csv results/bench_$$(date +%Y%m%d_%H%M%S).csv && \
		echo "" && \
		echo "=== Summary ===" && \
		python scripts/summarize_cuda_results.py results/bench_*.csv; \
	else \
		echo "nvcc not found - skipping CUDA measurement harness."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the CUDA measurement harness."; \
	fi

cuda-bench-large:
	@echo "Building and running large CUDA measurement harness (sweep)..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		mkdir -p results && \
		echo "" && \
		echo "=== Sweep: requests [4,8,16,32] x tokens [128,256,512] x draft_len [1,4,8] ===" && \
		./build/cuda/xlpk_cuda_smoke --mode sweep --requests 32 --tokens 512 --draft-len 8 --csv results/bench_large_$$(date +%Y%m%d_%H%M%S).csv && \
		echo "" && \
		echo "=== Summary ===" && \
		python scripts/summarize_cuda_results.py results/bench_large_*.csv; \
	else \
		echo "nvcc not found - skipping large CUDA measurement harness."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the large CUDA measurement harness."; \
	fi

cuda-research-bench:
	@echo "Building and running standalone CUDA research-kernel benchmarks..."
	@if command -v nvcc >/dev/null 2>&1; then \
		cmake -S cuda -B build/cuda && \
		cmake --build build/cuda && \
		echo "" && \
		echo "=== Sparse KV Gather ===" && \
		./build/cuda/xlpk_cuda_smoke --mode sparse-gather --requests 8 --draft-len 4 && \
		echo "" && \
		echo "=== Verify + Commit ===" && \
		./build/cuda/xlpk_cuda_smoke --mode verify-commit --requests 8 --draft-len 4 && \
		echo "" && \
		echo "=== DMA-Aware KV Movement ===" && \
		./build/cuda/xlpk_cuda_smoke --mode dma-movement --requests 8 --draft-len 4 && \
		echo "" && \
		echo "=== Resident Research Pipeline ===" && \
		./build/cuda/xlpk_cuda_smoke --mode research-pipeline --requests 8 --draft-len 4 --iterations 8; \
	else \
		echo "nvcc not found - skipping CUDA research benchmarks."; \
		echo "Install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)"; \
		echo "to build and run the research-kernel benchmarks."; \
	fi

clean:
	rm -rf build dist *.egg-info
	rm -rf src/megakernel_lab/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf cuda/build build/cuda
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
