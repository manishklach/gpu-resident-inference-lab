/**
 * baseline_host_decode_kernel.cu — Baseline comparison kernel.
 *
 * Role:
 *   Conventional host-driven decode. The CPU controls the token loop:
 *   launch once per step, synchronize after each step, inspect results
 *   on the host, repeat. One host kernel launch = one decode step.
 *
 * This exists ONLY for comparison against the persistent mega-kernel.
 * The whole point of this GPU resident inference loop experiment is to eliminate this pattern.
 *
 * Key design:
 *   - Host launches repeatedly (host_kernel_launches = O(tokens)).
 *   - Host synchronizes after every launch (host_synchronizations = O(tokens)).
 *   - CPU owns the token loop.
 *   - Intentionally shows the orchestration overhead the mega-kernel avoids.
 *
 * Current status:
 *   - One step per launch. No speculative decode. No device-resident loop.
 *   - Intentionally the anti-pattern / baseline.
 *
 * Launch pattern:
 *   for each iteration:
 *     baseline_host_decode_step_kernel<<<grid, block>>>(requests, N)
 *     cudaDeviceSynchronize()
 *     inspect results on host
 */

#include <cuda_runtime.h>
#include "request_desc.h"

__global__ void baseline_host_decode_step_kernel(
    RequestDescriptor* requests,
    int num_requests
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_requests) return;

    RequestDescriptor* req = &requests[idx];
    if (req->is_done()) return;
    if (!req->is_state(REQUEST_DECODE_READY) && !req->is_state(REQUEST_PREFILL_READY)) return;

    if (req->is_state(REQUEST_PREFILL_READY)) {
        req->decode_pos = 0;
        req->output_token_count = 0;
        req->last_token = (req->request_id * 7 + 3) % 32000;
        req->set_state(REQUEST_DECODE_READY);
    }

    if (req->is_state(REQUEST_DECODE_READY)) {
        int next_token = (req->last_token + 1 + req->request_id) % 32000;
        req->last_token = next_token;
        req->output_token_count++;
        req->decode_pos++;

        if (next_token == req->eos_token_id) {
            req->set_flag(REQUEST_FLAG_EOS_SEEN);
            req->set_state(REQUEST_COMPLETE);
        } else if (req->output_token_count >= req->max_new_tokens) {
            req->set_state(REQUEST_COMPLETE);
        }
    }
}
