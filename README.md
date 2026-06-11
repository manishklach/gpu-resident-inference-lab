# XL-Persistent-Kernel

[![CI](https://github.com/manishklach/XL-Persistent-Kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/manishklach/XL-Persistent-Kernel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: Research](https://img.shields.io/badge/license-Research%20Use-yellow)](LICENSE)
[![Blog](https://img.shields.io/badge/GitHub%20Pages-blog-green)](https://manishklach.github.io/XL-Persistent-Kernel/)

**XL-Persistent-Kernel explores a persistent GPU mega-kernel execution model for LLM serving.** The goal is to move the decode-serving control loop from CPU-orchestrated kernel launches into one GPU-resident execution loop. Prefill, sparse KV selection, decode, speculative verification, commit, and KV lifecycle management are modeled as logical stages inside one persistent kernel. · [Read the blog](https://manishklach.github.io/XL-Persistent-Kernel/)

This repository is not a production inference stack. It is a research scaffold for building the control flow, scheduling, and memory-management infrastructure that a persistent CUDA decode kernel will eventually need.

The simulator runs entirely on CPU today, but it is structured so that every abstractions (backend, KV-cache, request descriptors) can be swapped for real device implementations without rewriting the runtime.

**Important: The current implementation uses fake deterministic math. It is a control-flow and lifecycle scaffold, not a production transformer runtime.**

## Architecture

```mermaid
graph LR
    A[Request Submit] --> B[Prefill Worker]
    B --> C[KV Page Planner]
    C --> D[Sparse KV Selector]
    D --> E[Decode Worker]
    E --> F[Speculative Proposer]
    F --> G[Verifier]
    G -->|Accepted| H[Commit Tokens]
    G -->|Rejected| I[Discard Draft Pages]
    H --> J[Request Complete]
    I --> D

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#e3f2fd
    style E fill:#e8f5e9
    style F fill:#e8f5e9
    style G fill:#fce4ec
    style H fill:#c8e6c9
    style I fill:#ffcdd2
    style J fill:#c8e6c9
```

**Request lifecycle:**

1. A request is submitted with prompt tokens and a target output sequence.
2. The **prefill worker** processes the prompt and builds the initial KV-cache pages.
3. The **KV page planner** allocates physical pages across all layers.
4. The **sparse KV selector** scores committed pages, picks a deterministic top-k subset, and marks those blocks as selected for the next decode step.
5. The **decode worker** pins the selected pages and runs the decode loop against that logical subset.
6. The **speculative proposer** drafts a block of candidate tokens.
7. The **verifier** checks the draft against the target and the backend.
8. Accepted tokens are committed; rejected draft pages are released.
9. The request completes when EOS is hit, the token budget is exhausted, or the target is fully matched.

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

- **Prefill**, **sparse KV selection**, **decode**, **speculative verification**, **commit**, and **KV page updates** are modeled as **logical stages inside one resident kernel** — not separate GPU kernel launches.
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
| `stage_sparse_kv_select` | Sparse KV block picker | Deterministic top-k metadata selection | Real sparse attention routing |
| `stage_decode` | Logical decode stage | Deterministic token generation | Real decode kernel path |
| `stage_spec_verify` | Speculative verifier | Deterministic accept/reject | Target-model verification |
| `stage_commit` | Accept/commit stage | Metadata transition | Fused token/KV commit |
| `stage_kv` | KV lifecycle helpers | Metadata only | Real paged KV movement |
| `stage_scheduler` | Device-side request picker | Linear scan + priority | GPU-resident scheduler |
| `baseline_host_decode_kernel` | Comparison baseline | One step per host launch | Performance baseline |

## What This Measures Today

The current CUDA scaffold does not measure real transformer math, model quality, or production LLM throughput. It measures orchestration structure: host launch count, host synchronization count, request lifecycle progress, and the difference between a CPU-driven token loop and one GPU-resident mega-kernel launch.

## What Is Implemented Today

- **Runtime simulator** with specialized prefill and decode workers
- **Speculative block proposal and verification** with configurable acceptance policy
- **Paged KV-cache planner** with LRU eviction, pinning, and memory accounting
- **MSA-inspired sparse KV selection scaffold** over logical KV pages with deterministic top-k selection
- **Backend interface** (`AbstractKernelBackend`) + deterministic CPU stub backend
- **Benchmark harness** exporting TTFT, ITL, acceptance rate, KV hit rate, live/pinned KV bytes, fragmentation, and sparse KV traffic estimates
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

## Many Logical Stages, One Resident Kernel

XL-Persistent-Kernel is not a bag of independent CUDA kernels. The opposite is the point. The repo models prefill, decode, speculative verification, commit, and KV lifecycle management as logical stages inside one persistent GPU mega-kernel. The stage helper files exist for readability, but the execution model is one resident kernel that keeps request state and control flow on the GPU.

This repo is not trying to build many independent CUDA kernels. It is trying to show how many logical serving stages can be fused into one resident GPU mega-kernel.

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
| `sparse_kv_megakernel` | Adds deterministic top-k KV block selection before decode to model reduced KV traffic — **not real sparse attention math** |

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
    sparse_kv.py        - Deterministic sparse KV block selector scaffold
    spec_decode.py      - Speculative block proposer and verifier
    backend.py          - Abstract kernel backend + CPU stub
    bench.py            - Benchmark harness with CSV export
    demo.py             - Runnable demo comparing decode modes
    block_spec_decode.py - DFlash-style block drafter and verifier
    block_runtime.py    - Block speculative runtime loop
    swa_state.py        - Sliding-window attention state model
    token_state.py      - Token lifecycle: draft, accept, commit, reject

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
        stage_sparse_kv_select.cuh     - Sparse KV top-k selection helper (inline)
    src/
        xl_persistent_megakernel.cu    - THE fused persistent mega-kernel
        baseline_host_decode_kernel.cu - Baseline comparison kernel
        host_launcher.cpp              - Host launcher + smoke tests
    CMakeLists.txt      - Builds xlpk_cuda_smoke executable

cuda/examples/
    diffusion_refinement_megakernel_sketch.cu     - Diffusion-style persistent kernel sketch
    warp_specialized_block_pipeline_sketch.cu      - Warp-specialized block speculative sketch

examples/
    block_speculative_demo.py                      - Block speculative decode comparison demo

tests/
    test_runtime.py                - Runtime and worker tests
    test_kv_cache.py               - KV cache allocation, eviction, memory accounting
    test_block_spec_decode.py      - DFlash-style block drafter and verifier
    test_token_state.py            - Token lifecycle: draft, accept, commit, reject
    test_swa_state.py              - Sliding-window state with read/write counters
    test_block_runtime.py          - Block speculative runtime loop
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
| Sparse KV select | `stage_sparse_kv_select.cuh` | Score resident KV blocks and pick top-k | Deterministic metadata only | Real sparse attention routing |
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

## Diffusion-Style Refinement Loop Sketch

Diffusion-style language models (e.g., DiffusionGemma, MDLM, SSD-LM) reduce sequential decode by refining many tokens in parallel through a series of denoising steps. A persistent mega-kernel is a complementary runtime idea for this setting: keep the entire denoise → refine → verify → commit → state-update loop resident on GPU, instead of bouncing through CPU orchestration between each diffusion step.

The file [`cuda/examples/diffusion_refinement_megakernel_sketch.cu`](cuda/examples/diffusion_refinement_megakernel_sketch.cu) is a conceptual sketch showing this mapping:

```
Autoregressive (main repo):  prefill → decode → spec_verify → commit → KV update
Diffusion-style (sketch):    denoise → update_confidence → verify_or_resample → commit → state update
```

**Common thesis:** Many logical stages, one resident GPU kernel. The stage names differ, but the control-flow architecture is the same — minimize CPU round-trips by keeping the iteration loop on device.

```cuda
while (!*shutdown && !r->done) {
    denoise_canvas_step(r, canvas);
    update_confidence_mask(r, canvas);
    verify_or_resample(r, canvas);
    commit_ready_tokens(r, canvas);
    update_resident_state(r, state);
}
```

> **This sketch is not an implementation of DiffusionGemma. It is a systems-level mapping of the persistent mega-kernel idea to diffusion-style token refinement. All math is fake/deterministic.**

📖 [**Full blog post: Diffusion-Style Token Refinement on a Persistent Mega-Kernel**](https://manishklach.github.io/XL-Persistent-Kernel/diffusion-sketch.html) — stage-by-stage breakdown, autoregressive comparison table, and design rationale.

## Why Speculative Decoding Makes Persistent Kernels More Valuable

If decode is strictly one token at a time, a persistent kernel mainly reduces host launch and synchronization overhead. Useful, but limited.

Once the runtime proposes a **block** of draft tokens in parallel, the persistent kernel has much more useful work to keep resident:

- load next tile (weights, activations, KV window)
- dequantize FP4 tiles
- compute current block
- verify draft tokens
- commit accepted tokens
- update KV-or-state metadata
- prefetch next block

**Key equation:**

| Technique | What it solves |
|-----------|---------------|
| FP4 quantization | reduces model weight bytes and bandwidth pressure |
| DFlash-style drafting | proposes multiple candidate tokens in parallel |
| Sliding-window attention (SWA) | limits drafter state dependency to a fixed-size window |
| Persistent mega-kernel | keeps draft/verify/commit/state-update pipeline resident on GPU |

**Relationship:**

> Speculative decoding creates block-level parallel work.  
> The persistent kernel keeps that block-level workflow resident and flowing.

This repo models these ideas with fake deterministic math and lifecycle counters. It does not implement Xiaomi DFlash, TileRT, or real transformer inference.

## MSA-Inspired Sparse KV Selection

This repo now includes an **MSA-inspired sparse KV selection scaffold**. For each decode iteration, the runtime scores logical KV pages using deterministic metadata, selects a top-k subset, pins those selected blocks, and passes only that subset into the logical decode and verify stages.

This is intentionally precise about what it is and is not:

- It is **not** MiniMax MSA.
- It is **not** FlashAttention.
- It is **not** production sparse attention math.
- It **is** a research scaffold for control flow and memory scheduling inspired by sparse attention systems.

What this demonstrates is the convergence of:

- speculative/token-parallel decode
- sparse KV block selection
- persistent GPU-resident execution
- KV lifecycle management

What it does **not** claim is model-quality preservation or production throughput. The current metrics estimate reduced KV traffic and orchestration structure only.

Additional files:

- [`cuda/examples/warp_specialized_block_pipeline_sketch.cu`](cuda/examples/warp_specialized_block_pipeline_sketch.cu) — conceptual sketch: warp-group roles for load, dequantize, compute, verify, commit, schedule
- [`src/megakernel_lab/block_spec_decode.py`](src/megakernel_lab/block_spec_decode.py) — DFlash-style drafter simulator (fake math)
- [`src/megakernel_lab/block_runtime.py`](src/megakernel_lab/block_runtime.py) — block speculative runtime: draft → verify → commit → update loop
- [Adaptive Speculative Block Sizing (ASBS) for XL-Persistent-Kernel](https://manishklach.github.io/writings/adaptive-speculative-block-sizing-xl-persistent-kernel.html) — blog post explaining the adaptive block sizing path and why it matters for persistent speculative decode
- [`src/megakernel_lab/swa_state.py`](src/megakernel_lab/swa_state.py) — SWA window state model with read/write counters
- [`src/megakernel_lab/token_state.py`](src/megakernel_lab/token_state.py) — token lifecycle: draft, accept, commit, reject, resample
- [`examples/block_speculative_demo.py`](examples/block_speculative_demo.py) — runnable comparison of serial, block-spec, and persistent control

| Benchmark mode | Description |
|---------------|-------------|
| `autoregressive_serial` | one token committed per iteration |
| `block_speculative` | DFlash-style block drafting and verification |
| `block_speculative_persistent_sim` | block spec with `host_kernel_launches = 1` |
| `block_speculative_host_orchestrated` | block spec with launch/sync per stage |

## Measurement: Host-Launched Decode vs Persistent Mega-Kernel

The CUDA measurement harness (`xlpk_cuda_smoke`) compares two execution-control paths. It does **not** measure transformer math — the math remains fake/deterministic. It measures **orchestration overhead**: host kernel launches, host synchronizations, and elapsed time for the control-flow scaffold.

### What the Numbers Show

| Path | Host launches | Host syncs | Control owner |
|------|--------------|------------|---------------|
| Baseline host-launched | O(tokens) | O(tokens) | CPU |
| Persistent mega-kernel | 1 | 1 | GPU |

The baseline path launches a kernel for every decode step. The mega-kernel launches once and the GPU advances requests internally.

### Commands

```bash
# Quick smoke test (4 requests, 8 tokens each)
make cuda-smoke

# Measurement run (8 requests, 128 tokens each, CSV output)
make cuda-bench

# Larger run (32 requests, 512 tokens each)
make cuda-bench-large

# Summarize results
python scripts/summarize_cuda_results.py build/cuda/cuda_results.csv
```

### Example Output

```
Baseline host-launched decode:
  host_kernel_launches: 128
  host_synchronizations: 128

Persistent mega-kernel:
  host_kernel_launches: 1
  host_synchronizations: 1

Relative:
  launch_reduction: 128:1
  sync_reduction: 128:1
```

**Important:** This is not claiming real 1T inference performance. This demonstrates the execution-control advantage that a real 1T-class serving system would need when combined with MoE, sparsity, quantization, paged KV cache, speculative decoding, continuous batching, and multi-GPU communication overlap.

## Development

```bash
make install     # Install with dev dependencies
make lint        # Run linters
make format      # Auto-fix formatting
make test        # Run test suite
```

## License

Research use only. See LICENSE for details.
