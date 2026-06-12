/**
 * compacted_decode_kernel.cu — Research-stage decode over compact sparse tiles.
 *
 * Role:
 *   Consumes compacted sparse KV payloads and emits deterministic speculative
 *   candidate tokens. This is a stand-in for a future decode kernel that works
 *   over sparse, compacted KV working sets rather than the full page table.
 */

#include <cuda_runtime.h>

#include "kernel_status.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"

__global__ void compacted_sparse_decode_kernel(
    RequestDescriptor* requests,
    int num_requests,
    const int* selected_page_ids,
    const int* compacted_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* draft_tokens,
    CompactedDecodeMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    RequestDescriptor* req = &requests[blockIdx.x];
    CompactedDecodeMetrics metric = {};
    metric.request_id = req->request_id;

    if (!req->is_state(REQUEST_DECODE_READY)) {
        metrics[blockIdx.x] = metric;
        return;
    }

    const int request_slot = blockIdx.x;
    const int selected_base = request_slot * max_selected_pages;
    const int compacted_base = request_slot * max_selected_pages * ints_per_page;

    int payload_sum = 0;
    for (int i = 0; i < max_selected_pages; ++i) {
        if (selected_page_ids[selected_base + i] < 0) {
            continue;
        }
        metric.pages_consumed += 1;
        payload_sum += compacted_payloads[compacted_base + i * ints_per_page];
    }

    int count = req->current_block_size > 0 ? req->current_block_size : 1;
    int budget_left = req->max_new_tokens - req->output_token_count;
    if (count > budget_left) {
        count = budget_left > 0 ? budget_left : 0;
    }

    req->draft_len = count;
    for (int i = 0; i < count; ++i) {
        draft_tokens[req->draft_offset + i] =
            (req->last_token + payload_sum + req->request_id + i + 1) % 32000;
    }

    if (count > 0) {
        req->set_state(REQUEST_DRAFT_READY);
    }
    metric.generated_tokens = count;
    metrics[blockIdx.x] = metric;
}
