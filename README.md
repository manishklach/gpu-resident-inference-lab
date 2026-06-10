# XL-Persistent-Kernel

**CPU-first control-plane simulator and CUDA staging ground for persistent-kernel LLM decode.**

This repository is not a production inference stack. It is a research scaffold for building the control flow, scheduling, and memory-management infrastructure that a persistent CUDA decode kernel will eventually need.

The simulator runs entirely on CPU today, but it is structured so that every abstractions (backend, KV-cache, request descriptors) can be swapped for real device implementations without rewriting the runtime.

## Architecture

```mermaid
graph LR
    A[Request Submit] --> B[Prefill Worker]
    B --> C[KV Page Planner]
    C --> D[Decode Worker]
    D --> E[Speculative Proposer]
    E --> F[Verifier]
    F -->|Accepted| G[Commit Tokens]
    F -->|Rejected| H[Discard Draft Pages]
    G --> I[Request Complete]
    H --> D

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#e8f5e9
    style F fill:#fce4ec
    style G fill:#c8e6c9
    style H fill:#ffcdd2
    style I fill:#c8e6c9
```

**Request lifecycle:**

1. A request is submitted with prompt tokens and a target output sequence.
2. The **prefill worker** processes the prompt and builds the initial KV-cache pages.
3. The **KV page planner** allocates physical pages across all layers.
4. The **decode worker** pins the active pages and runs the decode loop.
5. The **speculative proposer** drafts a block of candidate tokens.
6. The **verifier** checks the draft against the target and the backend.
7. Accepted tokens are committed; rejected draft pages are released.
8. The request completes when EOS is hit, the token budget is exhausted, or the target is fully matched.

## Mega-Kernel Design Philosophy

XL-Persistent-Kernel is built around one idea: the inference-serving pipeline should not be a long chain of CPU-launched GPU kernels. Prefill, decode, speculative verification, commit, and KV-cache updates are modeled as logical stages inside one persistent GPU mega-kernel.

The goal is to reduce:
- repeated kernel launches
- CPU scheduling overhead
- CPU–GPU synchronization
- fragmented GPU execution
- host-controlled token loops
- avoidable KV-cache movement

The result is a serving model where more request state and scheduling progress stays on the GPU.

This means:

- **Prefill**, **decode**, **speculative verification**, **commit**, and **KV page updates** are modeled as **logical stages inside one resident kernel** — not separate GPU kernel launches.
- The device-side loop (in `xl_persistent_megakernel.cu`) calls `__forceinline__ __device__` stage helpers from `.cuh` files, all fused into a single compiled device function.
- There is exactly **one** GPU kernel that stays resident for the entire request lifetime. Host involvement is limited to submission and completion polling.

The only separate CUDA kernel is `baseline_host_decode_kernel.cu`, which exists **solely as a baseline for comparison** — it represents the conventional host-launched model that this project aims to supersede.

### Why This Matters for 1T-Class Models

For 1T-class models, especially sparse/MoE systems, throughput is limited not only by FLOPs but by orchestration: token-by-token launch overhead, fragmented decode stages, KV-cache residency, inter-GPU communication, and speculative verification/commit overhead. A persistent mega-kernel is one execution technique for pushing such systems toward 1K+ tokens/sec when combined with:

- MoE or sparsity
- Quantization
- Speculative decoding
- Paged KV cache
- Continuous batching
- Multi-GPU parallelism
- Communication overlap
- GPU-resident scheduling

**Important:** A mega-kernel alone does not make dense 1T models exceed 1K TPS. It is one key component in the overall serving architecture. This repo does not serve a real 1T model — it is a control-flow scaffold for the persistent mega-kernel execution model.

### Component Table

| Component | Role | Today | Future |
|-----------|------|-------|--------|
| `xl_persistent_megakernel` | Fused resident GPU control loop | Deterministic control-flow scaffold | Real fused inference pipeline |
| `stage_prefill` | Logical prefill stage | Metadata only | Real prefill attention |
| `stage_decode` | Logical decode stage | Deterministic token generation | Real decode kernel path |
| `stage_spec_verify` | Speculative verifier | Deterministic accept/reject | Target-model verification |
| `stage_commit` | Accept/commit stage | Metadata transition | Fused token/KV commit |
| `stage_kv` | KV lifecycle helpers | Metadata only | Real paged KV movement |
| `stage_scheduler` | Device-side request picker | Linear scan + priority | GPU-resident scheduler |
| `baseline_host_decode_kernel` | Comparison baseline | One step per host launch | Performance baseline |

## What Is Implemented Today

- **Runtime simulator** with specialized prefill and decode workers
- **Speculative block proposal and verification** with configurable acceptance policy
- **Paged KV-cache planner** with LRU eviction, pinning, and memory accounting
- **Backend interface** (`AbstractKernelBackend`) + deterministic CPU stub backend
- **Benchmark harness** exporting TTFT, ITL, acceptance rate, KV hit rate, live/pinned KV bytes, fragmentation
- **Speculative KV distinction** between committed and draft pages
- **CUDA staging layer** with one `xl_persistent_megakernel` + one baseline comparison kernel + six device-side stage helpers (optional build)
- **CI pipeline** (pytest + ruff + mypy on Python 3.10/3.11/3.12)
- **Tests** covering runtime, KV cache, speculative KV lifecycle, and benchmark schema

## What Is Not Implemented Yet

- Real CUDA attention / projection / sampling kernels
- Fused speculative-verify kernels
- Device-resident request descriptors and work queues
- Multi-GPU / NVLink communication overlap
- Continuous batching with dynamic request admission
- Real transformer math on device
- Quantized weight and KV support
- Memory-mapped model loading

These are planned phases (see [docs/ROADMAP.md](docs/ROADMAP.md)).

## Why Persistent Kernels Matter

Traditional LLM decode launches one kernel per token per request:

```
Host: launch attention kernel -> wait -> launch projection -> wait -> sample -> wait -> repeat
```

Each launch incurs host-device synchronization, kernel launch overhead, and memory fence costs. For small batch sizes, this dominates runtime.

A persistent kernel keeps one long-lived GPU loop running:

```
Device: loop {
  read request descriptor from global memory
  load KV pages
  run attention + projection + sample
  write new token and update decode position
  if done, write completion status
}
```

The kernel never returns to the host until all requests complete. The host only manages request admission, KV page allocation, and completion callbacks.

This is the execution model behind production systems like vLLM's persistent batch, Xiaomi's Mirage/TileRT, and SGLang's RadixAttention. This repository builds the control-plane simulator that such a kernel requires.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the demo
python -m megakernel_lab.demo

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python -c "from megakernel_lab.bench import BenchmarkRunner; print(BenchmarkRunner().run())"

# Or use Make targets
make help
make demo
make test
make bench
```

## Benchmark Modes

| Mode | Description |
|------|-------------|
| `serial_decode` | Block size 1, no speculation (CPU simulates host-launched decode) |
| `speculative_decode` | Configurable block size with draft/verify/commit loop |
| `forced_rejection` | Mismatch stride forces periodic draft rejections |
| `kv_pressure` | Undersized KV cache to trigger eviction stress |
| `mega_kernel_sim` | Models the fused mega-kernel control path (draft → verify → commit loop on CPU) — **not a CUDA measurement** |

All Python benchmarks are control-flow simulations. The CUDA smoke test (`make cuda-smoke`) validates the staging path on real hardware.

## Benchmark Example Output

```
   batch_size  block_size  mean_ttft_ms  mean_itl_ms  acceptance_rate  kv_hit_rate  live_kv_bytes  pinned_kv_bytes  eviction_count  fragmentation_ratio
0           1           1          0.75         0.25              1.0          0.0           320              256               0                  0.0
1           1           2          0.75         0.25              1.0          0.0           320              256               0                  0.0
2           4           1          0.75         0.25              1.0          0.0          1280             1024               0                  0.0
3           4           4          0.75         0.25              1.0          0.0          1280             1024               0                  0.0
4           8           1          0.75         0.25              1.0          0.0          2560             2048               0                  0.0
5           8           4          0.75         0.25              1.0          0.0          2560             2048               0                  0.0
```

## Repository Structure

```
src/megakernel_lab/
    config.py           - Runtime configuration (block size, layers, KV dimensions)
    state.py            - Request, worker, and backend state objects
    runtime.py          - Persistent decode runtime with worker pool
    kv_cache.py         - Paged KV-cache with LRU eviction, pinning, memory accounting
    spec_decode.py      - Speculative block proposer and verifier
    backend.py          - Abstract kernel backend + CPU stub
    bench.py            - Benchmark harness with CSV export
    demo.py             - Runnable demo comparing decode modes

cuda/
    include/            - CUDA headers + stage helpers (.cuh)
        request_desc.h                 - Request descriptor struct
        kv_page_table.h                - KV page entry/table structs
        queue_desc.h                   - Ring queue descriptor
        kernel_status.h                - Request lifecycle states
        stage_scheduler.cuh            - Device-side request scheduler (inline)
        stage_prefill.cuh              - Prefill stage helper (inline)
        stage_decode.cuh               - Decode stage helper (inline)
        stage_spec_verify.cuh          - Verify stage helper (inline)
        stage_commit.cuh               - Commit stage helper (inline)
        stage_kv.cuh                   - KV page lifecycle helpers (inline)
    src/
        xl_persistent_megakernel.cu    - THE fused persistent mega-kernel
        baseline_host_decode_kernel.cu - Baseline comparison kernel
        host_launcher.cpp              - Host launcher + smoke tests
    CMakeLists.txt      - Builds xlpk_cuda_smoke executable

tests/
    test_runtime.py     - Runtime and worker tests
    test_kv_cache.py    - KV cache allocation, eviction, memory accounting
    test_spec_kv.py     - Speculative KV page lifecycle tests
    test_bench.py       - Benchmark schema validation

docs/
    ARCHITECTURE.md     - Design intent and core concepts
    CUDA_STAGING.md     - Kernel inventory, lifecycle, and queue design
    ROADMAP.md          - Development phases
```

## CUDA Staging Layer

The `cuda/` directory contains the **one** persistent mega-kernel (`xl_persistent_megakernel`) and a baseline comparison kernel (`baseline_host_decode_kernel`).

The mega-kernel is the centerpiece: a single persistent GPU kernel that fuses all pipeline stages (prefill, decode, speculative verify, commit, KV update, scheduling) into a device-resident loop. Stage helpers are `__forceinline__ __device__` functions in `.cuh` files — they are **not** separately launched kernels.

### Mega-Kernel Stages (device-side inline helpers)

| Stage | File | Purpose | Real today? | Future |
|-------|------|---------|-------------|--------|
| Scheduler | `stage_scheduler.cuh` | Pick next request | Linear scan + priority | GPU-resident scheduler |
| Prefill | `stage_prefill.cuh` | Mark prompt/KV initialized | Fake math | Real prefill attention |
| Decode | `stage_decode.cuh` | Produce token + draft block | Deterministic tokens | Fused decode kernel |
| Verify | `stage_spec_verify.cuh` | Accept/reject draft | Deterministic rule | Rejection sampling |
| Commit | `stage_commit.cuh` | Commit tokens | Metadata update | Fused commit |
| KV helpers | `stage_kv.cuh` | Page flag/lifecycle utilities | Flag toggle only | Real KV copy |

### Only Two Kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `xl_persistent_megakernel` | `src/xl_persistent_megakernel.cu` | **The** persistent fused mega-kernel |
| `baseline_host_decode_kernel` | `src/baseline_host_decode_kernel.cu` | Baseline comparison (conventional host-launched) |

### Build and Run (requires CUDA toolkit)

```bash
make cuda-smoke     # Builds and runs xlpk_cuda_smoke (tests both paths)
```

Without CUDA, this target prints a friendly message and skips. See [docs/CUDA_STAGING.md](docs/CUDA_STAGING.md) for the full design document.

## Development

```bash
make install     # Install with dev dependencies
make lint        # Run linters
make format      # Auto-fix formatting
make test        # Run test suite
```

## License

Research use only. See LICENSE for details.
