# GPU Resident Inference Lab

Research lab for GPU-resident LLM inference loops: persistent kernels, sparse KV selection, tiered residency, speculative decode, and trace-driven scheduling.

[![CI](https://github.com/manishklach/gpu-resident-inference-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/manishklach/gpu-resident-inference-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: Research](https://img.shields.io/badge/license-Research%20Use-yellow)](LICENSE)
[![Blog](https://img.shields.io/badge/GitHub%20Pages-blog-green)](https://manishklach.github.io/gpu-resident-inference-lab/)

A research scaffold for future LLM inference loops where decode control flow,
KV movement, speculative verification, and request scheduling stay resident
on the GPU as much as possible.

The project explores five interacting ideas:

1. Persistent GPU-resident execution to reduce host launch/sync overhead
2. Sparse KV block selection to reduce memory touched per decode step
3. Speculative/token-parallel decode to create more useful work per loop
4. Tiered KV residency across HBM, DRAM, and SSD-like tiers
5. Trace-driven admission and scheduling for multi-request serving

This is not a production LLM runtime. It is a control-flow, memory-scheduling,
and CUDA-staging research platform.

The core thesis is that once inference becomes quantized, sparse, and latency-sensitive, the bottleneck shifts from raw compute to orchestration and data movement. The next runtime layer should keep more of the decode/refine/verify/KV-update loop resident on GPU, while using sparse KV selection and token/block parallelism to ensure the persistent loop has enough useful work to execute.

## Why This Repo Exists

Traditional autoregressive decode is a skinny loop:

```text
CPU launch -> GPU decode -> CPU sync -> CPU launch -> GPU verify -> CPU sync
```

That shape creates orchestration gaps. Even if individual kernels are efficient, the overall serving loop can underutilize the GPU when each step is too narrow and too host-driven.

This repo studies a wider GPU-resident loop instead:

```text
submit once
   |
   v
GPU resident loop:
  sparse KV select
  -> draft / token-block decode
  -> expert route
  -> attention / verify
  -> commit accepted tokens
  -> KV update
  -> schedule next block
```

The point is not to overclaim throughput. The point is to make orchestration, residency, and memory-movement bottlenecks visible and to prototype how a future GPU-resident runtime might be structured.

## Decode Loop Shapes

Diagram 1: CPU-driven decode today

```text
CPU
 | launch decode
 v
GPU: decode one token/block
 |
CPU sync / schedule / launch again
 |
 v
GPU: verify / update
 |
repeat
```

Diagram 2: GPU-resident loop thesis

```text
CPU submits work once
 |
 v
GPU persistent loop:
  [select KV blocks]
  [draft token block]
  [route experts]
  [attention/verify]
  [commit accepted tokens]
  [update KV/state]
  [prefetch next block]
 |
 v
CPU receives coarse-grained completions
```

## Persistent Kernels Are Not Enough

Persistent kernel alone:
- reduces launch/sync overhead

Persistent kernel + token/block parallelism:
- creates enough useful resident work

Persistent kernel + sparse KV:
- reduces memory touched per iteration

Persistent kernel + tiered residency:
- controls where KV lives under pressure

Persistent kernel + trace-driven admission:
- decides which requests/blocks deserve GPU residency

A persistent kernel only removes orchestration gaps. It does not magically make autoregressive decoding parallel. The decode loop becomes interesting when persistence is combined with token/block parallelism, sparse KV selection, and resident scheduling.

## Modern Inference Stack

This repo studies the runtime/kernel side of a broader inference stack:

- FP4 / NVFP4-style quantization: reduce weight bandwidth
- MoE sparsity: reduce active parameters per token
- SWA / local attention / sparse KV: bound KV and context movement
- MTP / speculative / block decoding: make decode wider than one token at a time
- Persistent GPU-resident mega-kernels: keep the hot loop on device
- Tiered KV residency: decide what stays in HBM, what spills, and what is prefetched

The repo is mostly focused on the last three layers: token/block parallel decode, GPU-resident execution, and KV residency/scheduling.

## What Runs Today vs Future Work

| Area | Today | Future |
|---|---|---|
| Persistent loop | CUDA scaffold / control-flow prototype | real fused decode loop |
| Transformer math | deterministic placeholder math | attention/projection/sampling kernels |
| Sparse KV | metadata/top-k scaffold | real sparse KV gather |
| Tiered residency | planning model / simulator | async HBM/DRAM/SSD movement integration |
| Speculative decode | block workflow scaffold | real draft/verify model path |
| Metrics | launch/sync/memory estimates | real TTFT/ITL/tok/s under load |
| Scheduling | trace-driven admission ideas | multi-request GPU-resident scheduler |

## Research Themes

### Persistent GPU-Resident Execution

Instead of launching many short-lived kernels from the CPU, the project explores keeping the decode/verify/KV-update loop resident on the device.

### Sparse KV Selection

Instead of touching all KV blocks, a runtime can select a smaller relevant subset. In this repo, that path is deterministic and lightweight by design. It is not MiniMax MSA or production sparse attention.

### Speculative / Token-Parallel Decode

Speculative and block-style decode creates more useful work per resident iteration. This is what makes a persistent loop more valuable than a one-token-at-a-time device loop.

### Tiered KV Residency

The repo models conceptual HBM, DRAM, and SSD-like tiers, plus promotion, demotion, staging, and pressure handling. These are residency and scheduling scaffolds, not real migration engines.

### Trace-Driven Scheduling

The repo also models arrival, admission, active-set limits, and completion ordering so multi-request serving behavior can be studied as a control-flow problem, not just a single-request kernel problem.

## What This Repo Is

- A research scaffold for GPU-resident inference control flow
- A way to study persistent execution, sparse KV selection, and token/block parallel decode
- A place to prototype scheduling ideas before integrating with real serving stacks

## Non-goals

This repo is not:
- a production inference server
- a replacement for vLLM, SGLang, TensorRT-LLM, or FlashAttention
- a complete transformer implementation today
- a benchmark claiming state-of-the-art throughput
- a hardware proposal requiring a new SRAM chip

This repo is:
- a research scaffold for GPU-resident inference control flow
- a way to study persistent execution, sparse KV selection, and token/block parallel decode
- a place to prototype scheduling ideas before integrating with real serving stacks

## How to Evaluate This Repo

This repo should be evaluated on whether it makes the control-flow bottlenecks visible, not on whether it currently serves a real frontier model.

Good questions:
- How many CPU launches are removed per token/block?
- How much KV traffic is avoided by sparse selection?
- How many accepted tokens are produced per verify step?
- How much useful work happens inside one resident loop?
- Where do orchestration gaps still appear?
- Which parts would need to become real CUDA kernels next?

## Implemented Today

- Persistent execution scaffold in both Python simulation and CUDA staging
- Sparse KV block selection path with deterministic top-k selection
- DMA-aware KV movement planning over sparse-selected pages
- Tier-aware KV staging plan that orders selected pages for resident decode
- KV pressure and draft-first eviction scaffold for resident memory reclamation
- Hierarchical KV tier rebalance across HBM, DRAM, and SSD budgets
- Trace-driven request admission and completion replay on device queues
- Speculative/token-parallel workflow scaffolding
- KV lifecycle tracking across committed, draft, selected, and released states
- Benchmark harness for control-flow and memory-accounting experiments
- CUDA resident-loop scaffold plus host-launched baseline comparison
- Tests covering runtime behavior, sparse KV selection, benchmark schema, and KV lifecycle rules

## Metrics

The repo currently emits some metrics directly and models others as future benchmark surfaces.

| Metric | Status | Meaning |
|---|---|---|
| `tokens_per_resident_loop` | Implemented | Useful committed tokens per resident-loop iteration |
| `kv_blocks_total` | Implemented | Total logical KV blocks considered |
| `kv_blocks_selected` | Implemented | KV blocks selected for the sparse path |
| `kv_sparsity_ratio` | Implemented | Fraction of blocks not touched by sparse selection |
| `estimated_kv_bytes_read` | Implemented | Approximate KV bytes read under selected-block access |
| `estimated_kv_bytes_saved` | Implemented | Approximate KV bytes not read because of sparsity |
| `accepted_tokens` | Partially implemented | Accepted speculative tokens tracked in runtime traces/results |
| `rejected_tokens` | Implemented in block workflow benchmarks | Rejected speculative tail tokens |
| `trace queue metrics` | Implemented | Admission, completion, queue depth, and active-set watermarks |
| `TTFT / ITL / tok/s` | Future real-model mode | Real serving metrics once placeholder math is replaced |

## CUDA Staging Layer

The `cuda/` directory contains the resident-loop scaffold and the host-launched baseline.

- `src/xl_persistent_megakernel.cu` models a fused GPU-resident loop
- `src/baseline_host_decode_kernel.cu` models repeated host-launched decode steps
- `include/stage_sparse_kv_select.cuh` models sparse KV block selection in the loop
- `src/sparse_kv_gather_kernel.cu` models page scoring, top-k selection, and compacted sparse KV gather
- `src/verify_commit_kernel.cu` models fused speculative verify plus accepted-prefix commit and rejected-page release
- `src/tiered_kv_staging_kernel.cu`, `src/kv_pressure_eviction_kernel.cu`, `src/kv_tier_residency_kernel.cu`, and `src/trace_replay_admission_kernel.cu` model staging, pressure, tier rebalance, and trace-driven admission

These are research kernels and stage helpers for a persistent GPU-resident loop. They are not a production CUDA serving stack.

## Benchmark Modes

The Python side provides control-flow simulations such as:

- `serial_decode`
- `speculative_decode`
- `forced_rejection`
- `kv_pressure`
- `mega_kernel_sim`
- `sparse_kv_megakernel`
- `block_speculative`
- `block_speculative_persistent_sim`

The CUDA host launcher exposes standalone research-kernel modes for:

- resident scheduler ordering
- sparse KV gather and score
- KV prefetch planning
- fused verify plus commit
- DMA-aware KV movement planning
- tiered KV staging
- KV pressure eviction
- KV tier residency rebalance
- trace replay admission
- resident sparse decode pipeline

## Repository Structure

```text
src/megakernel_lab/
    config.py             - Runtime and benchmark configuration
    state.py              - Request, trace, and result state objects
    runtime.py            - Persistent runtime and worker model
    kv_cache.py           - Paged KV planner, pinning, eviction, accounting
    sparse_kv.py          - Sparse KV top-k selection scaffold
    spec_decode.py        - Draft/verify control-flow logic
    block_runtime.py      - Block speculative runtime model
    block_spec_decode.py  - Block drafting scaffold
    bench.py              - Benchmark harness and metrics
    demo.py               - Runnable runtime demo

cuda/
    include/              - Device-side stage helpers and metadata structs
    src/                  - Resident-loop scaffold and host baseline
    examples/             - Conceptual CUDA sketches

docs/
    ARCHITECTURE.md       - Core concepts and design intent
    CUDA_STAGING.md       - CUDA staging notes
    ROADMAP.md            - Development roadmap
    BLOG.md               - Draft long-form framing
    REASONING_PIPELINE.md - Decode-to-reasoning north-star framing
    REASONING_METRICS_STUB.md - Metric vocabulary for future branch verification

tools/
    reasoning_metrics_stub.py - A toy simulator for future verified-decisions/sec metrics. It does not run a model and should not be interpreted as a real reasoning benchmark.
```

## Quick Start

```bash
pip install -e ".[dev]"

python -m megakernel_lab.demo
python -m pytest tests/ -v
python -c "from megakernel_lab.bench import BenchmarkRunner; print(BenchmarkRunner().run())"

make help
make demo
make test
make bench
```

If CUDA is available:

```bash
make cuda-smoke
make cuda-bench
make cuda-bench-large
make cuda-research-bench
```

## Suggested GitHub Description

Research lab for GPU-resident LLM inference loops: persistent kernels, sparse KV selection, tiered residency, speculative decode, and trace-driven scheduling.

## Suggested GitHub Topics

- `gpu-inference`
- `llm-inference`
- `persistent-kernels`
- `cuda`
- `kv-cache`
- `speculative-decoding`
- `multi-token-prediction`
- `sparse-kv`
- `gpu-runtime`
- `inference-systems`

## Further Reading

- [Project blog](https://manishklach.github.io/gpu-resident-inference-lab/)
- [Adaptive Speculative Block Sizing (ASBS) blog, historical naming](https://manishklach.github.io/writings/adaptive-speculative-block-sizing-xl-persistent-kernel.html)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/BLOG.md](docs/BLOG.md)
- [From GPU-Resident Decode to GPU-Resident Reasoning](docs/REASONING_PIPELINE.md)
- [Reasoning Metrics Stub](docs/REASONING_METRICS_STUB.md)

## Where This Is Going: From Tokens/sec to Verified Decisions/sec

The near-term focus of this repo is GPU-resident inference control flow:
persistent loops, sparse KV selection, token/block parallel decode, tiered
residency, and trace-driven scheduling.

The longer-term direction is broader:

> not just faster tokens/sec, but faster verified decisions/sec.

Future inference systems will likely combine:

- continuous batching for many concurrent agents and reasoning branches
- speculative decoding evolving into speculative reasoning
- adaptive precision routing across FP4, FP8, and BF16
- memory-aware model architecture across HBM, L2, shared memory, registers, and KV tiers
- GPU-resident persistent loops that fuse draft, verify, commit, KV update, and scheduling

In this framing, the winning system is not only the fastest token generator.

It is the fastest correct reasoner.

This repo does not implement that full system today. It starts with the lower-level
runtime pieces needed to study that direction: resident execution, sparse memory
selection, block-level decode structure, and scheduling visibility.

### Scope guardrail

Today, this repo is not a reasoning engine.

It does not yet implement:
- real multi-agent orchestration
- real verifier models
- real tool-use loops
- real retrieval-integrated generation
- real symbolic checking
- real correctness scoring

Those are future research directions.

The current repo focuses on the systems substrate:
GPU-resident control flow, memory selection, token/block decode structure,
and orchestration-gap measurement.

## License

Research use only. See [LICENSE](LICENSE).
