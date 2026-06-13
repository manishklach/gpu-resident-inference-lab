# Roadmap

This roadmap organizes the project as a staged GPU-resident inference loop experiment rather than a single-kernel demo.

## Phase 0: CPU Simulator and Trace Model

Goal:
Build a CPU-side model for decode control flow, trace collection, KV accounting, and scheduling behavior before real CUDA math is introduced.

Expected artifacts:
- Python runtime and worker model
- Request/trace/result state objects
- KV planner with eviction and pinning
- Trace and benchmark harness

Validation metric:
- Trace integrity and reproducible control-flow behavior in tests

## Phase 1: CUDA Persistent Loop Scaffold

Goal:
Create a persistent GPU-resident loop scaffold that removes repeated host-driven orchestration from the critical path.

Expected artifacts:
- `xl_persistent_megakernel.cu`
- Stage helpers for decode, verify, commit, scheduler, and KV state
- Baseline host-launched comparison path

Validation metric:
- Host launches per generated token/block

## Phase 2: NVTX / Profiler Visibility and Orchestration-Gap Measurement

Goal:
Make orchestration gaps visible and measurable with timing and profiler annotations.

Expected artifacts:
- CUDA timing harness
- launch/sync counters
- CSV sweep mode
- profiler and NVTX integration

Validation metric:
- Host launches and host synchronizations removed per decode path

## Phase 3: Sparse KV Selection and Residency Metadata

Goal:
Model how a GPU-resident runtime touches less KV state and tracks which pages are hot, selected, pinned, or evictable.

Expected artifacts:
- Sparse KV selection helper
- sparse KV gather kernel
- residency flags and metadata
- selected-vs-total KV metrics

Validation metric:
- Selected KV blocks vs total KV blocks

## Phase 4: Speculative / Token-Block Decode Workflow

Goal:
Widen the resident loop so persistence has enough useful work to execute.

Expected artifacts:
- speculative draft/verify/commit control flow
- block decode simulator and CUDA staging pieces
- accepted/rejected token accounting

Validation metric:
- Accepted tokens per verification step

## Phase 5: Real Fused Decode / Verify Kernels

Goal:
Replace deterministic placeholder math with real decode, attention, projection, sampling, and verification kernels.

Expected artifacts:
- real attention/projection/sampling kernels
- fused decode/verify path
- real model-backed staging mode

Validation metric:
- Tokens per resident loop iteration with real math

## Phase 6: Tiered KV Movement and Async Prefetch / Spill

Goal:
Model and later integrate tier-aware KV movement across HBM, DRAM, and SSD-like tiers.

Expected artifacts:
- DMA-aware movement planner
- tiered staging order
- pressure / eviction kernel
- tier residency rebalance kernel

Validation metric:
- Estimated KV bytes read/saved and bytes promoted/demoted/reclaimed

## Phase 7: Multi-Request Scheduling and Fairness

Goal:
Move from single-shot per-request passes to realistic multi-request GPU-resident scheduling, admission, and fairness decisions.

Expected artifacts:
- resident scheduler kernel
- trace replay admission kernel
- queue-driven active-set and completion models
- fairness and backpressure experiments

Validation metric:
- Queue depth, active-set watermark, and resident work completed per replay step

## Phase 8: Multi-GPU / NVLink or Fabric-Aware Residency

Goal:
Extend the residency and scheduling thesis beyond one device.

Expected artifacts:
- multi-GPU residency model
- NVLink/fabric-aware placement ideas
- remote KV movement experiments

Validation metric:
- TTFT and inter-token latency in future multi-device real-model mode

## Current Artifacts

Today’s repo already includes:

- CPU simulator and trace model
- CUDA persistent loop scaffold
- sparse KV selection
- speculative/block decode scaffolding
- DMA-aware movement planning
- tiered staging
- KV pressure and eviction
- KV tier residency rebalance
- trace replay admission

## Evaluation Questions

Good ways to judge progress:

- How many host launches are removed per token/block?
- How many selected KV blocks are touched relative to total KV blocks?
- How many accepted tokens are produced per verify step?
- How much useful work happens inside one resident loop?
- Where do orchestration gaps still remain?
- Which parts are still placeholder math and which are real kernels?
