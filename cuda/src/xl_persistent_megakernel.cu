/**
 * xl_persistent_megakernel.cu — Fused persistent mega-kernel.
 *
 * Role:
 *   One resident GPU kernel launch. The GPU advances the request lifecycle
 *   internally without returning control to the host. All pipeline stages
 *   are device-side inline helpers called from this kernel's loop.
 *
 * Key design:
 *   - Host launches this kernel once (host_kernel_launches = 1).
 *   - Host synchronizes once at completion (host_synchronizations = 1).
 *   - GPU owns the token loop: prefill → decode → verify → commit → repeat.
 *   - Shutdown uses atomic done_counter: each block increments when its
 *     request reaches is_done(); block 0 polls the aggregate counter and
 *     sets shutdown_flag when all requests complete. This avoids the race
 *     from a block-0 linear scan without a memory fence.
 *
 * Current status:
 *   - All math is fake/deterministic (control-flow scaffold only).
 *   - No real transformer attention, projection, or sampling.
 *   - The measurement target is orchestration overhead, not model FLOPs.
 *
 * Thread utilization:
 *   - BLOCK_SIZE = 1 (one thread per block, one request per block).
 *   - All stage helpers are called from a single thread; no warp-level
 *     parallelism yet. Real parallelism comes in Phase 3 when stage
 *     helpers become real fused kernels with warp-level work distribution.
 *
 * Future:
 *   - Real fused inference pipeline with real attention/decode kernels,
 *     KV tensors, and speculative verification.
 *
 * @param requests        Array of request descriptors (device memory)
 * @param num_requests    Number of request descriptors
 * @param kv_table        KV page table with device-side page entries
 * @param draft_tokens    Shared draft token buffer (device memory)
 * @param shutdown_flag   Host/device flag; kernel exits when set
 * @param done_counter    Aggregate counter; each block incs once when done
 * @param max_iterations  Safety bound on the internal loop
 * @param block_size      Speculative block size for draft proposal
 */

#include <cuda_runtime.h>
#include "request_desc.h"
#include "kv_page_table.h"
#include "stage_scheduler.cuh"
#include "stage_prefill.cuh"
#include "stage_decode.cuh"
#include "stage_spec_verify.cuh"
#include "stage_commit.cuh"

__global__ void xl_persistent_megakernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    int* draft_tokens,
    int* shutdown_flag,
    int* done_counter,
    int max_iterations,
    int block_size
) {
    if (blockIdx.x >= num_requests) return;

    RequestDescriptor* req = &requests[blockIdx.x];
    bool already_counted = false;

    for (int iteration = 0; !(*shutdown_flag) && iteration < max_iterations; iteration++) {
        if (!req->is_done()) {
            stage_prefill(req, &kv_table);
            stage_decode(req, draft_tokens, block_size);
            stage_spec_verify(req, draft_tokens);
            stage_commit(req, &kv_table);
        }

        if (!already_counted && req->is_done()) {
            __threadfence();
            atomicAdd(done_counter, 1);
            already_counted = true;
        }

        if (blockIdx.x == 0) {
            if (*done_counter >= num_requests) {
                __threadfence();
                *shutdown_flag = 1;
            }
        }

        __syncthreads();
    }
    // TODO Phase 3: expand to warp-level parallelism when stage helpers
    // become real fused kernels with shared-memory cooperative work.
