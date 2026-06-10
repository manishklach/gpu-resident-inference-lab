# Roadmap

Development phases for the XL-Persistent-Kernel project.

## Phase 1: CPU Control-Flow Simulator

**Status: Complete**

The foundation: a Python simulator that models the exact control flow a persistent CUDA kernel will need.

- Persistent runtime with specialized prefill and decode workers
- Paged KV-cache planner with LRU eviction and pinning
- Speculative block proposal and verification
- Backend interface (`AbstractKernelBackend`) with CPU stub
- Memory accounting (live bytes, pinned bytes, evicted bytes, fragmentation)
- Speculative KV distinction (committed vs draft pages)
- Benchmark harness with TTFT, ITL, acceptance rate, KV metrics
- Full test coverage for runtime and KV cache behavior

This phase ensures we get the state machine right before touching CUDA.

## Phase 2: CUDA Persistent-Kernel Stub

**Status: Scaffolded**

A minimal CUDA kernel that demonstrates the control flow without real transformer math.

- Request descriptor struct (`request_desc.h`)
- Persistent decode loop (`persistent_decode_stub.cu`)
- Host submission queue and device work queue design
- Completion queue for host consumption
- Shutdown protocol
- Optional build target (`make cuda-stub`)

This phase proves the host/device interface works before adding real kernels.

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

Key challenge: keeping the persistent loop efficient while adding real math.

## Phase 4: Continuous Batching and Multi-Request Scheduling

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
2. The CUDA stub runs on real hardware without host-device synchronization during decode
3. Fused kernels achieve >10x throughput improvement over per-token launch
4. Multi-GPU scaling demonstrates near-linear speedup on NVLink-connected nodes
5. The benchmark harness measures realistic metrics (TTFT, ITL, throughput)
