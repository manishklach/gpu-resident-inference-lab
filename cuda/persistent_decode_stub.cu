/**
 * persistent_decode_stub.cu - Minimal CUDA persistent-kernel stub.
 *
 * This is a scaffold for a real persistent decode kernel. It does NOT
 * implement any transformer math (attention, projection, sampling).
 * Instead, it demonstrates the control flow that a real persistent
 * kernel would use:
 *
 * 1. Read request descriptors from global memory
 * 2. For each active request:
 *    a. Load the current decode position
 *    b. Simulate a decode step (no real math yet)
 *    c. Advance decode_pos
 *    d. Write completion status when done
 * 3. Loop until all requests signal shutdown
 *
 * The key insight: the kernel never returns to the host mid-request.
 * It stays resident on the GPU and processes requests entirely in device code.
 *
 * How this maps to real persistent decode:
 * - The loop body would call fused attention + projection + sampling kernels
 * - KV pages would be loaded from paged KV cache in global memory
 * - Draft tokens would be proposed by a lightweight draft model
 * - Verification would compare draft vs target in a single pass
 * - Commit/reject would update the page table on device
 *
 * Build: nvcc -arch=sm_80 -o persistent_decode_stub persistent_decode_stub.cu
 * Or use: make cuda-stub (from repo root)
 */

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "request_desc.h"

// ---------------------------------------------------------------------------
// Configuration constants
// ---------------------------------------------------------------------------

/** Maximum requests the scheduler loop will process per wave. */
constexpr int MAX_REQUESTS = 64;

/** Block size for the persistent kernel grid. */
constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Device kernel: persistent decode loop
// ---------------------------------------------------------------------------

/**
 * persistent_decode_kernel - Device-side persistent decode loop.
 *
 * This kernel runs indefinitely on the GPU. Each thread block handles
 * one request (or multiple blocks can share via cooperative groups in
 * a real implementation).
 *
 * The kernel reads RequestDescriptors from global memory, processes
 * decode steps, and writes back status. It never returns to host
 * control until all requests complete or a shutdown flag is set.
 *
 * @param requests    Pointer to array of request descriptors in global memory
 * @param num_requests Number of active requests
 * @param shutdown    Host-writable flag to signal kernel termination
 */
__global__ void persistent_decode_kernel(
    RequestDescriptor* requests,
    int num_requests,
    volatile int* shutdown
) {
    // Each block handles one request. In a real implementation, blocks
    // would be assigned dynamically via work queues.
    int block_id = blockIdx.x;
    if (block_id >= num_requests) return;

    RequestDescriptor& req = requests[block_id];

    // Persistent loop: keep running until shutdown or all requests done.
    // In a real kernel, this loop would contain:
    //   - KV page loading from paged cache
    //   - Fused attention computation
    //   - Projection and sampling
    //   - Draft token proposal (small model)
    //   - Verification against target
    //   - Page table updates on commit/reject
    while (!(*shutdown) && req.is_active()) {
        // --- Simulate decode step (no real math) ---
        // In production, this is where attention + projection runs.
        // The kernel reads KV pages for the current decode position,
        // computes attention over committed tokens, and samples a new token.

        // Advance decode position (simulates accepting one token)
        bool still_active = req.advance();

        if (!still_active) {
            // Budget exhausted or EOS reached
            req.mark_finished();
        }

        // --- Sync point ---
        // In a real kernel, __syncthreads() would ensure all threads
        // in the block see the updated decode_pos before next iteration.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helper
// ---------------------------------------------------------------------------

/**
 * launch_persistent_decode - Host function to launch the persistent kernel.
 *
 * In a real runtime, this would be called once at startup and the kernel
 * would remain resident. New requests would be submitted by writing to
 * the request descriptor array in pinned host memory (mapped to device
 * address space via cudaHostAlloc / cudaHostGetDevicePointer).
 *
 * @param requests    Host-side array of request descriptors
 * @param num_requests Number of requests to process
 */
void launch_persistent_decode(RequestDescriptor* requests, int num_requests) {
    if (num_requests <= 0 || num_requests > MAX_REQUESTS) {
        fprintf(stderr, "Error: num_requests must be in [1, %d]\n", MAX_REQUESTS);
        return;
    }

    // Allocate request descriptors on device
    RequestDescriptor* d_requests = nullptr;
    cudaError_t err = cudaMalloc(&d_requests, num_requests * sizeof(RequestDescriptor));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy request descriptors to device
    err = cudaMemcpy(d_requests, requests, num_requests * sizeof(RequestDescriptor), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_requests);
        return;
    }

    // Allocate shutdown flag on device (initialized to 0 = not shutting down)
    int* d_shutdown = nullptr;
    err = cudaMalloc(&d_shutdown, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc shutdown flag failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_requests);
        return;
    }
    int zero = 0;
    cudaMemcpy(d_shutdown, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: one block per request
    // In a real implementation, this would be a single persistent block
    // that processes requests via cooperative groups or work-stealing.
    printf("Launching persistent decode kernel with %d blocks...\n", num_requests);
    persistent_decode_kernel<<<num_requests, BLOCK_SIZE>>>(
        d_requests, num_requests, d_shutdown
    );

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy results back
    err = cudaMemcpy(requests, d_requests, num_requests * sizeof(RequestDescriptor), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy result back failed: %s\n", cudaGetErrorString(err));
    }

    // Print results
    for (int i = 0; i < num_requests; i++) {
        printf("Request %u: decode_pos=%u committed_len=%u status=0x%x\n",
               requests[i].request_id,
               requests[i].decode_pos,
               requests[i].committed_len,
               requests[i].status);
    }

    // Cleanup
    cudaFree(d_requests);
    cudaFree(d_shutdown);
}

// ---------------------------------------------------------------------------
// Main: demo entry point
// ---------------------------------------------------------------------------

/**
 * main - Demo entry point for the persistent decode stub.
 *
 * Creates a few synthetic request descriptors and launches the kernel.
 * In a real system, request descriptors would be managed by the host
 * runtime and submitted via pinned memory queues.
 */
int main() {
    printf("XL-Persistent-Kernel: CUDA persistent decode stub\n");
    printf("This is a control-flow scaffold, not a real transformer kernel.\n\n");

    // Create synthetic requests
    constexpr int NUM_REQUESTS = 4;
    RequestDescriptor requests[NUM_REQUESTS] = {};

    uint32_t dummy_tokens[] = {100, 101, 102, 103};
    uint32_t target_tokens[] = {200, 201, 202, 203};

    for (int i = 0; i < NUM_REQUESTS; i++) {
        requests[i].request_id = i + 1;
        requests[i].priority = i;
        requests[i].prompt_tokens = dummy_tokens;
        requests[i].prompt_len = 4;
        requests[i].committed_tokens = dummy_tokens;
        requests[i].committed_len = 0;
        requests[i].target_tokens = target_tokens;
        requests[i].target_len = 4;
        requests[i].decode_pos = 0;
        requests[i].max_new_tokens = 4;
        requests[i].eos_token_id = 0;
        requests[i].kv_page_table = nullptr;
        requests[i].kv_page_table_entries = 0;
        requests[i].status = REQ_STATE_ACTIVE;
    }

    launch_persistent_decode(requests, NUM_REQUESTS);

    printf("\nDone. All requests completed.\n");
    return 0;
}
