# Roadmap

Development phases for the XL-Persistent-Kernel project.

## Phase 1: CPU Control-Flow Simulator

**Status: Complete**

The foundation: a Python simulator that models the exact control flow a persistent CUDA kernel will need.

- Persistent runtime with specialized prefill and decode workers
- Paged KV-cache planner with LRU eviction and pinning
- Speculative block proposal and verification
- MSA-inspired sparse KV page selection over logical blocks
- Backend interface (`AbstractKernelBackend`) with CPU stub
- Memory accounting (live bytes, pinned bytes, evicted bytes, fragmentation)
- Speculative KV distinction (committed vs draft pages)
- Benchmark harness with TTFT, ITL, acceptance rate, KV metrics
- Full test coverage for runtime and KV cache behavior

This phase ensures we get the state machine right before touching CUDA.

## Phase 2A: Mega-Kernel CUDA Control-Flow Scaffold

**Status: In progress**

The fused persistent mega-kernel as a device-side control-flow scaffold. No real transformer math yet.

- Request descriptor with lifecycle states and flags
- Device-side stage helpers: prefill, decode, verify, commit, KV, scheduler
- Device-side sparse KV block selection stage between scheduling and decode
- Fake KV page metadata and lifecycle transitions
- `xl_persistent_megakernel.cu` — the fused resident kernel
- `baseline_host_decode_kernel.cu` — host-launched baseline for comparison
- CUDA smoke test (`make cuda-smoke`) comparing baseline launches vs one mega-kernel launch
- Stage helpers are `__forceinline__ __device__` functions, not separate kernels

**Key principle:** Many logical inference stages, one persistent mega-kernel.

## Phase 2A.1: Sparse KV Control-Flow Scaffold

**Status: In progress**

Extend the persistent decode scaffold with deterministic sparse-KV selection.

- `stage_sparse_kv_select.cuh` device helper for top-k KV block selection
- Per-page score / selected / last_selected_step / sparse_rank metadata
- Python `SparseKVSelector` mirroring the CUDA control path
- Benchmark metrics for total vs selected KV blocks and estimated bytes saved
- Tests that verify selected pages stay pinned during decode and released draft pages are not re-selected

**Important:** This phase does not implement MiniMax MSA, FlashAttention, or production sparse attention. It models control flow and memory scheduling only.

## Phase 2B: Measured Orchestration Overhead

**Status: In progress**

Instrument the measurement harness to quantify the execution-control difference between host-launched decode and the persistent mega-kernel.

- CUDA event timing (elapsed_ms for each path)
- Host launch count (baseline: O(tokens), mega-kernel: 1)
- Host synchronization count (baseline: O(tokens), mega-kernel: 1)
- CSV export with RunMetrics columns: launch_reduction, sync_reduction, speedup_vs_baseline
- `--mode sweep` Cartesian product over requests × tokens × draft_len
- Repeatable `make cuda-bench` and `make cuda-bench-large` targets
- `scripts/summarize_cuda_results.py` for compact summary + optional chart
- `scripts/compare_metrics.py` for Python-CUDA side-by-side
- `make compare` target for unified comparison
- README and blog measurement sections

**Key insight:** The first measurable win is not model quality or FLOPs. It is reduced orchestration overhead and less fragmented GPU execution.

## Phase 2C: NVTX / Profiler Visibility

**Planned**

Add NVTX annotations for profiler-based visualization of the control-flow difference.

- NVTX ranges around baseline loop (`baseline_host_decode_loop`)
- NVTX range around mega-kernel launch (`persistent_megakernel`)
- Guard with `#ifdef XLPK_ENABLE_NVTX` so builds without NVTX still work
- Nsight Systems instructions in docs
- Document expected trace shape: many small ranges (baseline) vs one large range (mega-kernel)
- Profiler-backed latency analysis to complement CUDA event timing

## Phase 2D: Block Speculative Decode Model

**In progress**

A Python model of DFlash-style block speculative decoding, showing how block-level generation creates the internal pipeline work that makes persistent kernels more valuable.

- `DFlashStyleDrafter` — proposes blocks of draft tokens using deterministic fake math
- `TokenState` — explicit token lifecycle: draft, accept, commit, reject, resample
- `SlidingWindowState` — SWA-inspired KV/state window with read/write counters
- `BlockSpeculativeRuntime` — block-level runtime loop: draft → verify → commit → update
- Benchmark modes: `autoregressive_serial`, `block_speculative`, `block_speculative_persistent_sim`, `block_speculative_host_orchestrated`
- CSV columns: iterations, draft_blocks, accepted/rejected tokens, acceptance rate, state reads/writes, host launches, host syncs
- `examples/block_speculative_demo.py` — runnable comparison

**Key insight:** Speculative decoding creates block-level work; the persistent mega-kernel keeps that work resident and flowing on GPU.

## Phase 2E: Warp-Specialized Persistent Pipeline Sketch

**In progress**

A conceptual CUDA sketch showing how warp specialization maps onto the persistent mega-kernel for block speculative decode.

- `cuda/examples/warp_specialized_block_pipeline_sketch.cu`
- Warp group roles: load/prefetch, dequantize (FP4), compute block, verify/commit, schedule
- Declared as conceptual only — not real TileRT, not real DFlash
- Does not compile by default

## Phase 3: Real Fused Decode/Verify Kernels

**Planned**

Replace the stub with actual transformer operations.

- Fused attention kernel for decode
- Projection + sampling kernel
- KV page loading from paged cache
- Speculative token proposal (small draft model)
- Fused verification kernel
- Page table updates on device
- Memory-efficient attention (FlashAttention-style)
- Continuous batching with dynamic request admission

Key challenge: keeping the persistent loop efficient while adding real math.

## Phase 4: Multi-Request Scheduling and Admission

**Planned**

Scale from single-request decode to multi-request continuous batching.

- Dynamic request admission (add new requests mid-batch)
- Request preemption and resumption
- Priority-based scheduling
- KV cache pressure management
- Memory-aware batch sizing
- Speculative verification across requests

This phase enables realistic throughput measurements.

## Phase 5: Multi-GPU / NVLink / Communication Overlap

**Planned**

Distribute the persistent kernel across multiple GPUs.

- Tensor parallelism across NVLink-connected GPUs
- Pipeline parallelism for deep models
- Communication overlap with compute
- KV cache distribution and synchronization
- Load balancing across devices
- Fault tolerance and checkpointing

This phase targets production-scale LLM serving.

## Success Criteria

The project succeeds when:

1. The CPU simulator accurately models the control flow of the CUDA implementation
2. The CUDA mega-kernel runs on real hardware without host-device synchronization during decode
3. Fused kernels demonstrate measurable launch-overhead reduction vs per-token launch
4. Multi-GPU scaling demonstrates near-linear speedup on NVLink-connected nodes
5. The benchmark harness measures realistic metrics (TTFT, ITL, throughput)
