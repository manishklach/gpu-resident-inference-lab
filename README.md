# XL-Persistent-Kernel

Persistent GPU-Resident Inference Research Platform

[![CI](https://github.com/manishklach/XL-Persistent-Kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/manishklach/XL-Persistent-Kernel/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: Research](https://img.shields.io/badge/license-Research%20Use-yellow)](LICENSE)
[![Blog](https://img.shields.io/badge/GitHub%20Pages-blog-green)](https://manishklach.github.io/XL-Persistent-Kernel/)

XL-Persistent-Kernel explores the convergence of persistent GPU-resident execution, sparse KV block selection, speculative/token-parallel decode workflows, and KV cache lifecycle management.

> This is a research and educational platform. It models inference control flow and memory scheduling. It is not a production LLM runtime.

Future inference performance may depend as much on moving fewer KV blocks and keeping execution resident as on making matrix multiplication faster.

The project is intentionally precise about scope. It models the control-flow and memory-scheduling shape of future inference systems. It does not claim compatibility with MiniMax MSA, FlashAttention, vLLM, SGLang, TensorRT-LLM, Mirage MPK, or any production serving stack.

## Why This Exists

Modern inference systems run into three different bottlenecks at once.

### A. Autoregressive Decode Dependency

Traditional decoding is sequential:

```text
token N -> token N+1 -> token N+2
```

That dependency chain is fundamental. A persistent kernel does not remove it by itself. Higher tokens/sec usually needs additional useful work inside the loop, such as batching, speculative decoding, token-parallel verification, or other multi-token structure.

### B. KV Cache Bandwidth

As context grows to 128K, 1M, and beyond, KV cache movement becomes a dominant memory-system problem. Even when matrix math is efficient, reading and updating ever-larger KV state can become the thing that limits throughput. Sparse-attention-inspired KV block selection is one way to explore how future runtimes might touch less memory per decode step.

### C. CPU-GPU Orchestration

Many inference systems involve repeated CPU scheduling, synchronization, and kernel launches. Persistent execution explores what happens when more of the loop remains resident on GPU and less progress depends on host round-trips.

## Core Idea

XL-Persistent-Kernel combines three complementary ideas:

1. Sparse KV selection reduces how much memory is touched.
2. Speculative/token-parallel execution increases useful work per iteration.
3. Persistent GPU-resident execution reduces host orchestration overhead.

These are complementary, not substitutes. Sparse KV selection does not replace speculation. Persistent execution does not replace useful token-level work. Speculative workflows do not automatically solve memory movement. The repo exists to model how these levers interact.

## Architecture

```text
Request
  |
  v
Scheduler
  |
  v
Sparse KV Block Selector
  |
  v
Persistent GPU-Resident Loop
  |-- Decode
  |-- Verify
  |-- Commit
  |-- KV Update
  |
  v
Response Stream
```

```text
while (!done) {
    select_sparse_kv_blocks();
    decode_candidates();
    verify_candidates();
    commit_accepted_tokens();
    update_kv_lifecycle();
}
```

At a high level, the repo models a request moving through scheduling, sparse KV page selection, decode, verification, commit, and KV lifecycle updates, all shaped around a GPU-resident loop rather than a host-driven per-step orchestration path.

## Research Themes

### Persistent Execution

Instead of launching many short-lived kernels from the CPU, the project explores keeping the decode/verify/commit/update loop resident.

Persistent execution reduces launch overhead and enables tighter scheduling, but it does not by itself parallelize autoregressive token generation.

### Sparse KV Block Selection

Instead of touching all KV blocks, a runtime can select a smaller relevant subset. This models sparse-attention-inspired memory behavior.

In this repo, the sparse selector is deliberately lightweight and deterministic. It is best described as MSA-inspired or sparse-attention-inspired control flow. It does not implement MiniMax MSA.

### Speculative / Token-Parallel Workflows

Speculative/token-parallel workflows create more useful work per decode iteration. This is the layer that can improve effective tokens/sec when combined with verification and commit logic.

The repo includes speculative draft/verify/commit scaffolding, block-style workflows, and adaptive block sizing experiments to show how multi-token work can make a resident loop more valuable.

### KV Lifecycle Management

KV blocks move through lifecycle states:

```text
allocated -> active -> selected -> verified -> released
```

This matters because future inference systems are not only compute pipelines. They are also memory-state machines that need to decide what remains resident, what is pinned, what is sparse-selected, and what is safe to discard.

## What This Repo Is / Is Not

| This repo is | This repo is not |
|---|---|
| A research scaffold for persistent inference control flow | A production LLM serving engine |
| A sparse KV selection demo | A full MiniMax MSA implementation |
| A speculative decode workflow simulator | A model-quality benchmark |
| A memory-scheduling playground | A replacement for FlashAttention/vLLM/SGLang |
| A way to reason about future inference kernels | A drop-in CUDA attention library |

## Why Persistent Kernels Need Useful Work

A persistent kernel is most valuable when the resident loop has enough work to amortize residency:

- speculative candidate generation
- multi-token verification
- sparse KV selection
- KV prefetch
- multi-request scheduling
- decode/commit/update overlap

If decode remains strictly one-token-at-a-time with no batching, speculation, or internal pipeline, persistent kernels mostly reduce orchestration overhead; they do not remove the fundamental autoregressive dependency.

That is the central systems point behind this repo. Persistent execution matters most when paired with useful work inside the resident loop.

## Implemented Today

- Persistent execution scaffold in both Python simulation and CUDA staging
- Sparse KV block selection path with deterministic top-k selection
- DMA-aware KV movement planning over sparse-selected pages
- Speculative/token-parallel workflow scaffolding
- KV lifecycle tracking across committed, draft, selected, and released states
- Benchmark harness for control-flow and memory-accounting experiments
- CUDA resident-loop scaffold plus host-launched baseline comparison
- Tests covering runtime behavior, sparse KV selection, benchmark schema, and KV lifecycle rules

## Metrics

The repo currently emits some metrics directly and models others as planned or expected fields for future benchmark surfaces.

| Metric | Status | Meaning |
|---|---|---|
| `tokens_per_resident_loop` | Implemented | Useful committed tokens per resident-loop iteration |
| `kv_blocks_total` | Implemented | Total logical KV blocks considered |
| `kv_blocks_selected` | Implemented | KV blocks selected for the sparse path |
| `kv_sparsity_ratio` | Implemented | Fraction of blocks not touched by sparse selection |
| `estimated_kv_bytes_read` | Implemented | Approximate KV bytes read under selected-block access |
| `estimated_kv_bytes_saved` | Implemented | Approximate KV bytes not read because of sparsity |
| `accepted_tokens` | Partially implemented | Accepted speculative tokens are tracked in runtime traces/results |
| `rejected_tokens` | Implemented in block workflow benchmarks | Rejected speculative tail tokens |
| `speculative_candidates` | Planned naming alignment | Candidate draft tokens proposed before verification |
| `commit_rate` | Planned | Accepted-to-proposed or accepted-to-iterated ratio depending on benchmark surface |
| `loop_iterations` | Planned naming alignment | Resident-loop iterations per request or batch |
| `orchestration_events_avoided` | Planned | Host launch/sync reductions relative to host-driven decode |

The emphasis today is on control-flow structure and estimated memory movement, not on model quality or production throughput claims.

## Benchmark Modes

| Mode | Description |
|---|---|
| `serial_decode` | Block size 1, no speculation; CPU simulates a host-launched decode path |
| `speculative_decode` | Configurable draft/verify/commit workflow |
| `forced_rejection` | Periodic draft rejection stress |
| `kv_pressure` | Undersized KV cache to trigger eviction pressure |
| `mega_kernel_sim` | CPU model of the fused resident-loop control path |
| `sparse_kv_megakernel` | Resident-loop model with deterministic sparse KV block selection |
| `autoregressive_serial` | One token committed per iteration in the block workflow model |
| `block_speculative` | DFlash-style block drafting and verification scaffold |
| `block_speculative_persistent_sim` | Block speculation plus persistent control model |
| `block_speculative_host_orchestrated` | Block speculation plus repeated host launch/sync model |

All Python benchmarks are control-flow simulations. The CUDA path is a staging scaffold and orchestration comparison harness, not a production inference benchmark.

## CUDA Staging Layer

The `cuda/` directory contains the resident-loop scaffold and the host-launched baseline:

- `src/xl_persistent_megakernel.cu` models a fused GPU-resident loop
- `src/baseline_host_decode_kernel.cu` models repeated host-launched decode steps
- `include/stage_sparse_kv_select.cuh` models sparse KV block selection in the loop
- `src/sparse_kv_gather_kernel.cu` models page scoring, top-k selection, and compacted sparse KV gather
- `src/verify_commit_kernel.cu` models fused speculative verify plus accepted-prefix commit and rejected-page release
- `include/stage_decode.cuh`, `include/stage_spec_verify.cuh`, `include/stage_commit.cuh`, and `include/stage_kv.cuh` model the rest of the logical inference path

These are stage helpers for one resident control-flow prototype. They are not a collection of production CUDA kernels.

The host launcher also exposes standalone research-kernel benchmark modes for:

- sparse KV gather and score
- fused verify plus commit
- DMA-aware KV movement planning
- resident sparse decode pipeline

The DMA-aware planner is still deliberately narrow in scope: it models how sparse-selected pages would be classified as HBM hits or DRAM/SSD fetches before decode consumes the compact working set. It does not implement real async copy, TMA, tier allocators, or production paging logic.

## Measurement: Host-Launched Decode vs Persistent Loop

The CUDA measurement harness focuses on orchestration structure:

- host kernel launches
- host synchronizations
- elapsed control-path time
- relative reduction between host-driven and resident-loop execution

This is intentionally narrower than claiming model throughput. The current scaffold uses deterministic fake math. What it can show credibly today is the difference between repeatedly launching work from the CPU and keeping the loop resident on GPU.

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
    block_spec_decode.py  - DFlash-style drafting scaffold
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

tests/
    test_runtime.py
    test_sparse_kv.py
    test_spec_kv.py
    test_bench.py
    ...
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

## Roadmap

Current:

- [x] Persistent execution scaffold
- [x] Sparse KV block selection
- [x] Speculative/token-parallel workflow
- [x] KV lifecycle tracking

Next:

- [ ] KV prefetch planning
- [ ] Multi-request scheduling
- [ ] Hierarchical KV tiers
- [ ] HBM / DRAM / SSD simulation
- [x] DMA-aware KV movement model
- [ ] Trace-driven replay
- [ ] Memory pressure simulation
- [ ] Visualization of KV block selection
- [ ] Benchmark report generation

More detailed phase notes live in [docs/ROADMAP.md](docs/ROADMAP.md).

## Related Ideas

This repo is conceptually adjacent to several important inference-system ideas:

- sparse attention
- MSA-style KV block sparsity
- persistent kernels
- speculative decoding
- memory-centric inference
- GPU-resident scheduling
- mega-kernel inference systems

The project references these ideas as systems inspiration only. It does not claim implementation compatibility with any specific production stack or published kernel library.

## Suggested GitHub Description

Option A:
Persistent GPU-resident inference research platform combining sparse KV selection, speculative decode, and KV lifecycle scheduling.

Option B:
Research scaffold for future LLM inference loops: persistent kernels, sparse KV blocks, token-parallel decode, and memory-centric scheduling.

Option C:
Experimental persistent inference kernel lab for reducing KV traffic, CPU orchestration, and decode-loop overhead.

## Suggested GitHub Topics

- `persistent-kernel`
- `cuda`
- `gpu-kernels`
- `llm-inference`
- `kv-cache`
- `sparse-attention`
- `speculative-decoding`
- `inference-systems`
- `memory-scheduling`
- `ai-infrastructure`

## Further Reading

- [Project blog](https://manishklach.github.io/XL-Persistent-Kernel/)
- [Adaptive Speculative Block Sizing (ASBS) for XL-Persistent-Kernel](https://manishklach.github.io/writings/adaptive-speculative-block-sizing-xl-persistent-kernel.html)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ROADMAP.md](docs/ROADMAP.md)

## Development

```bash
make install
make lint
make format
make test
```

## License

Research use only. See [LICENSE](LICENSE).
