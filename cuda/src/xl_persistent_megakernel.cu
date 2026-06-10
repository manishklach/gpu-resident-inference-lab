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
    int max_iterations,
    int block_size
) {
    if (blockIdx.x >= num_requests) return;

    RequestDescriptor* req = &requests[blockIdx.x];
    int iteration = 0;

    while (!(*shutdown_flag) && iteration < max_iterations) {
        if (!req->is_done()) {
            stage_prefill(req, &kv_table);
            stage_decode(req, draft_tokens, block_size);
            stage_spec_verify(req, draft_tokens);
            stage_commit(req, &kv_table);
        }

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            bool all_done = true;
            for (int i = 0; i < num_requests; i++) {
                if (!requests[i].is_done()) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) *shutdown_flag = 1;
        }

        iteration++;
        __syncthreads();
    }
}
