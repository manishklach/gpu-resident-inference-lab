# CUDA Staging: Fused Persistent Mega-Kernel

This document describes the CUDA staging layer for the persistent decode runtime. The entire pipeline is fused into **one** persistent GPU kernel — the `xl_persistent_megakernel`. All pipeline stages are device-side inline helpers, not separately launched kernels.

The repo contains multiple stage helper files, but they are not independent launched kernels. They are device-side helpers intended to be inlined into the persistent mega-kernel. This preserves the central design: many logical stages, one resident GPU kernel.

## What Is Fake Today

All math in the current CUDA scaffold is fake and deterministic. No real transformer operations are performed:

- **No real transformer math**: All token generation is `(last_token + 1 + request_id) % 32000`.
- **No real KV data**: `stage_kv.cuh` only toggles flag bits; no key/value tensors are copied.
- **No real page allocation**: KV pages are pre-allocated at init time.
- **No real verification**: The verifier uses a modulo-based rule, not rejection sampling.
- **No real draft model**: Draft tokens are deterministic offsets from the commit position.
- **No real queues**: Work and completion queues are declared in `queue_desc.h` but not yet wired into the mega-kernel.
- **No real scheduling**: Thread blocks are statically assigned to requests (one block = one request).
- **No continuous batching**: Requests are pre-loaded; no admission mid-batch.

The repo is a control-flow scaffold, not a production LLM runtime.

---

## Why One Mega-Kernel?

The traditional LLM serving approach launches many small kernels per token:

```
for each token:
    launch prefill kernel      (CPU -> GPU)
    synchronize                (GPU -> CPU)
    launch attention kernel    (CPU -> GPU)
    synchronize                (GPU -> CPU)
    launch projection kernel   (CPU -> GPU)
    synchronize                (GPU -> CPU)
    launch sampling kernel     (CPU -> GPU)
    synchronize                (GPU -> CPU)
    ...
```

Each launch incurs ~5–10 µs overhead, host-device synchronization latency, and memory fence costs. For small batch sizes, this overhead can dominate the runtime.

A persistent mega-kernel eliminates this overhead:

```
host: launch xl_persistent_megakernel once
device: while (requests remain) {
    prefill  (inline, no new launch)
    decode   (inline, no new launch)
    verify   (inline, no new launch)
    commit   (inline, no new launch)
}
host: poll completion queue
```

The kernel stays resident on the GPU. There is one launch, one synchronization point (at completion), and zero host–GPU communication during the decode loop.

This is the core thesis of XL-Persistent-Kernel.

---

## Why This Matters for 1T-Class Models

For 1T-class models, especially sparse/MoE systems, throughput is limited not only by FLOPs but by orchestration: token-by-token launch overhead, fragmented decode stages, KV-cache residency, inter-GPU communication, and speculative verification/commit overhead. A persistent mega-kernel is one execution technique for pushing such systems toward 1K+ tokens/sec when combined with:

- MoE or sparsity
- Quantization
- Speculative decoding
- Paged KV cache
- Continuous batching
- Multi-GPU parallelism
- Communication overlap
- GPU-resident scheduling

A mega-kernel alone does not make dense 1T models exceed 1K TPS. It is one key component in the overall serving architecture.

---

## Logical Stages vs. Launched Kernels

The pipeline is divided into logical stages, but these are **not separate CUDA kernels**. They are `__forceinline__ __device__` functions (defined in `.cuh` files) that are called inline from the mega-kernel's resident loop.

| Stage | File | Purpose |
|-------|------|---------|
| Scheduler | `stage_scheduler.cuh` | Pick next request to process |
| Prefill | `stage_prefill.cuh` | Mark prompt/KV initialized |
| Decode | `stage_decode.cuh` | Produce token or speculative draft |
| Verify | `stage_spec_verify.cuh` | Accept/reject draft tokens |
| Commit | `stage_commit.cuh` | Commit tokens, update KV metadata |
| KV helpers | `stage_kv.cuh` | Page flag and lifecycle utilities |

The only exception is `baseline_host_decode_kernel.cu`, which exists **solely as a baseline comparison** representing the conventional host-launched model. It is not part of the mega-kernel design.

---

## Device-Side Stage Helpers

All stage helpers are in `cuda/include/*.cuh` and are included by `xl_persistent_megakernel.cu` at compile time.

### stage_scheduler.cuh

```
pick_next_request(requests, num_requests, start_index):
    scan for high-priority non-complete request
    fallback to first non-complete request
    return -1 if all done
```

### stage_prefill.cuh

```
if (request is PREFILL_READY) {
    decode_pos = 0
    output_token_count = 0
    last_token = deterministic seed based on request_id
    init fake KV pages as COMMITTED + PINNED + RESIDENT
    request -> DECODE_READY
}
```

### stage_decode.cuh

```
if (request is DECODE_READY) {
    if (speculative mode enabled) {
        set draft_len (capped by budget)
        write deterministic draft tokens to shared buffer
        request -> DRAFT_READY
    } else {
        next_token = (last_token + 1 + request_id) % 32000
        update last_token, output_token_count, decode_pos
        if EOS or budget exhausted -> COMPLETE
        else remain DECODE_READY
    }
}
```

### stage_spec_verify.cuh

```
if (request is DRAFT_READY or VERIFY_READY) {
    for each draft token:
        if token % 4 == 0: reject (break)
        else: accept
    accepted_prefix_len = count of accepted tokens
    request -> COMMIT_READY
}
```

### stage_commit.cuh

```
if (request is COMMIT_READY) {
    if (accepted_prefix_len == 0) {
        fall back: commit one deterministic token
    } else {
        advance decode_pos by accepted_prefix_len
        advance output_token_count by accepted_prefix_len
        mark accepted KV pages as committed
        mark rejected KV pages as evictable
    }
    if budget exhausted -> COMPLETE
    else -> DECODE_READY
}
```

### stage_kv.cuh

Device helper functions for KV page lifecycle:
- `allocate_fake_kv_pages_for_request` — init pages for a request
- `mark_draft_kv_region` — mark page range as DRAFT
- `commit_draft_kv_region` — mark page range as COMMITTED
- `discard_rejected_kv_region` — mark draft pages as EVICTABLE
- `touch_kv_pages` — refresh residency on touched pages

No real KV tensors are moved. All operations are metadata-only.

---

## Baseline Host-Launched Decode Path

The file `baseline_host_decode_kernel.cu` implements the conventional approach for comparison:

```
for each iteration:
    baseline_host_decode_step_kernel<<<grid, block>>>(requests, N)
    cudaDeviceSynchronize()
    inspect results on host
    if all done: break
```

This path is exercised by `run_baseline_path()` in `host_launcher.cpp`.

**Why it exists:** To measure the overhead that the mega-kernel eliminates. Every launch in the baseline path incurs dispatch + synchronization costs that the mega-kernel avoids.

**Comparison metrics (future work):**
- Total wall time for N tokens
- Host CPU utilization during decode loop
- Kernel launch overhead as fraction of total time

---

## Persistent Mega-Kernel Path

The file `xl_persistent_megakernel.cu` is the centerpiece:

```
xl_persistent_megakernel<<<N, block>>>(
    requests, N, kv_table, draft_tokens, &shutdown, max_iterations, block_size
)
// No return to host until all requests complete or host sets shutdown.
```

Inside the kernel loop, each thread block calls stage helpers inline:

```
while (!shutdown && iteration < max_iterations) {
    if (!req.is_done()) {
        stage_prefill(req, &kv_table);     // PREFILL_READY -> DECODE_READY
        stage_decode(req, draft, bs);      // DECODE_READY -> DRAFT_READY | COMPLETE
        stage_spec_verify(req, draft);     // DRAFT_READY -> COMMIT_READY
        stage_commit(req, &kv_table);      // COMMIT_READY -> DECODE_READY | COMPLETE
    }
    // monitor thread checks all-done, sets shutdown flag
    iteration++
}
```

This path is exercised by `run_megakernel_path()` in `host_launcher.cpp`.

---

## Request Lifecycle

```
EMPTY
  |  (host submits request)
  v
PREFILL_READY  -->  mega-kernel calls stage_prefill() inline
  |
  v
PREFILL_DONE   -->  (transitions automatically to DECODE_READY)
  |
  v
DECODE_READY   -->  stage_decode() inline
  |               ┌─ non-speculative: one token per call, stays DECODE_READY or done
  |               └─ speculative:     sets draft_len, goes to DRAFT_READY
  v
DRAFT_READY    -->  stage_spec_verify() inline
  |
  v
VERIFY_READY   -->  (transitions automatically to COMMIT_READY)
  |
  v
COMMIT_READY   -->  stage_commit() inline
  |                 (calls stage_kv() for metadata)
  v
DECODE_READY   (if budget remains, repeat from DECODE_READY)
COMPLETE       (if budget exhausted or EOS)
FAILED         (if error_code set)
```

---

## KV Page Lifecycle

```
Free --> DRAFT (speculative tokens)
           |
           |  verify accepts
           v
       COMMITTED + RESIDENT + PINNED (active decode)
           |
           |  request completes or evicted
           v
       EVICTABLE / Free (returned to pool)
```

The `stage_kv.cuh` helper provides device functions for each transition.

| State/Flag | Meaning |
|------------|---------|
| KV_PAGE_FREE | Slot available for allocation |
| KV_PAGE_DRAFT | Speculative, discardable on rejection |
| KV_PAGE_COMMITTED | Token data is finalized |
| KV_PAGE_EVICTABLE | Eligible for LRU eviction |
| KV_PAGE_PINNED | Protected from eviction |
| KV_FLAG_RESIDENT | Data present in device memory |

---

## CUDA Smoke Test

Build and run:

```bash
make cuda-smoke
```

Output format:

```
XL-Persistent-Kernel CUDA smoke test

Baseline host-launched decode:
  launches: N
  completed_requests: M / M

Persistent mega-kernel:
  launches: 1
  completed_requests: M / M
```

Without nvcc, the target prints a skip message and succeeds.

---

## Measured Control-Plane Difference

The measurement harness in `host_launcher.cpp` (`xlpk_cuda_smoke`) quantifies the orchestration gap between two execution-control models:

### 1. Baseline Host-Launched Path

```
CPU owns the token loop.
for each iteration:
    launch baseline_host_decode_step_kernel   // host_kernel_launches++
    synchronize                                // host_synchronizations++
    copy request descriptors back
CPU advances one decode step per launch.
```

### 2. Persistent Mega-Kernel Path

```
CPU launches once.
GPU owns the token loop.
xl_persistent_megakernel<<<N, block>>>(...)
synchronize once at end.                     // host_synchronizations = 1
CPU copies descriptors back.
```

### Comparison Table

| Path | Host launches | Host syncs | Control owner |
|------|--------------|------------|---------------|
| Baseline | O(tokens) | O(tokens) | CPU |
| Mega-kernel | 1 | 1 | GPU |

### What This Measures

The current CUDA code does not measure transformer math (the math is fake). It measures:

- **Host kernel launch count** — each baseline launch incurs ~5–10 µs overhead
- **Host synchronization count** — each sync incurs latency and memory fence costs
- **Elapsed wall time** — total time for the control-flow scaffold to complete
- **Tokens per second** — throughput through the deterministic stub

### Expected Results

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

**Key insight:** The first measurable win is not model quality or FLOPs. It is reduced orchestration overhead and less fragmented GPU execution. A real 1T-class serving system would build on this pattern, combined with MoE, quantization, paged KV cache, speculative decoding, continuous batching, and multi-GPU communication overlap.

---

## Measurement Disclaimer

The current CUDA harness does not measure real transformer math, model quality, or production LLM throughput. It measures orchestration structure:

- **Host kernel launch count** — each baseline launch incurs ~5–10 µs overhead
- **Host synchronization count** — each sync incurs latency and memory fence costs
- **Request lifecycle progress** — do requests complete correctly on both paths?
- **Elapsed wall time** — total time for the control-flow scaffold to complete
- **Tokens per second** — throughput through the deterministic stub

**Important:** This is a measurement of orchestration structure, not LLM inference performance.

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

Key upcoming phases:

- **Phase 2B**: Measured orchestration overhead (CUDA event timing, launch/sync counters, CSV export, make cuda-bench) — **in progress**.
- **Phase 2C**: NVTX / profiler visibility (NVTX ranges around baseline loop and mega-kernel launch, Nsight Systems trace documentation) — **planned**.
- **Phase 3**: Real fused decode/verify path with real attention, projection, sampling, KV tensors, block verification, continuous batching — **planned**.
- **Phase 4**: Dynamic request admission via device queues, continuous batching — **planned**.
- **Phase 5**: Multi-GPU tensor parallelism, NVLink communication overlap — **planned**.

---

## Mapping to Diffusion-Style Generation

The persistent mega-kernel pattern is not limited to autoregressive decode. Diffusion-style language models (DiffusionGemma, MDLM, SSD-LM) refine tokens in parallel through a denoising loop. The same runtime idea applies: keep the loop on GPU, avoid CPU round-trips between refinement steps.

### Stage Mapping

| Autoregressive (main repo) | Diffusion-style (sketch) | Purpose |
|---------------------------|--------------------------|---------|
| `stage_prefill` | `denoise_canvas_step` | Initialize or perturb the token canvas |
| `stage_decode` | `update_confidence_mask` | Score each token's confidence |
| `stage_spec_verify` | `verify_or_resample` | Identify tokens that need resampling |
| `stage_commit` | `commit_ready_tokens` | Mark stable tokens as committed |
| `stage_kv` | `update_resident_state` | Track per-request refinement progress |

### Common Runtime Idea

**Fewer CPU round-trips, more GPU-resident control flow.** Whether the pipeline is autoregressive decode or diffusion refinement, the architectural bottleneck is the same: orchestrating many small GPU operations from the CPU. A persistent mega-kernel fuses the pipeline into one resident loop and removes the host from the critical path.

### File

See [`cuda/examples/diffusion_refinement_megakernel_sketch.cu`](../cuda/examples/diffusion_refinement_megakernel_sketch.cu) for the conceptual sketch.

---

## Directory Structure

```
cuda/
  include/
    kernel_status.h        - Request lifecycle states (9 states, 6 flags)
    request_desc.h         - Request descriptor struct + helpers
    kv_page_table.h        - KV page entry/table structs + state/flags + helper fns
    queue_desc.h           - Ring queue descriptor + push/pop helpers
    stage_scheduler.cuh    - Device-side request scheduler (inline helper)
    stage_prefill.cuh      - Prefill stage helper (inline, not separate kernel)
    stage_decode.cuh       - Decode stage helper (inline)
    stage_spec_verify.cuh  - Speculative verify stage helper (inline)
    stage_commit.cuh       - Commit stage helper (inline)
    stage_kv.cuh           - KV page lifecycle helper functions (inline)

  src/
    xl_persistent_megakernel.cu     - ONE fused persistent mega-kernel
    baseline_host_decode_kernel.cu  - Baseline comparison kernel (NOT part of mega-kernel)
    host_launcher.cpp               - Host launcher with baseline + mega-kernel paths

  examples/
    diffusion_refinement_megakernel_sketch.cu - Conceptual sketch: diffusion-style loop

  CMakeLists.txt           - Builds xlpk_cuda_smoke executable
```
