/**
 * sparse_kv_gather_kernel.cu — Research-stage sparse KV gather kernel.
 *
 * Role:
 *   A standalone next-step kernel that turns sparse KV selection metadata into
 *   real GPU memory traffic. For each request, it:
 *     1. scores resident KV pages,
 *     2. selects a top-k subset,
 *     3. gathers those selected pages into a compact working set,
 *     4. emits per-request memory-traffic metrics.
 *
 * Scope:
 *   - deterministic scoring only
 *   - fake per-page payloads (int arrays), not real KV tensors
 *   - intended as a memory-scheduling benchmark surface
 *   - not production sparse attention math
 */

#include <cuda_runtime.h>

#include "kv_page_table.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"
#include "stage_sparse_kv_select.cuh"

__global__ void sparse_kv_gather_and_score_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    const int* page_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* selected_page_ids,
    int* compacted_payloads,
    SparseKVGatherMetrics* metrics
) {
    if (blockIdx.x >= num_requests) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    RequestDescriptor* req = &requests[blockIdx.x];
    const int request_slot = blockIdx.x;
    const int request_selected_offset = request_slot * max_selected_pages;
    const int compacted_offset = request_slot * max_selected_pages * ints_per_page;

    for (int i = 0; i < max_selected_pages; ++i) {
        selected_page_ids[request_selected_offset + i] = -1;
    }

    SparseKVGatherMetrics metric = {};
    metric.request_id = req->request_id;

    if (req->kv_num_pages <= 0) {
        metrics[request_slot] = metric;
        return;
    }

    const int top_k = req->current_block_size > 0 ? req->current_block_size : max_selected_pages;
    select_sparse_kv_blocks(req, &kv_table, top_k);

    const int request_start = req->kv_table_offset;
    const int request_end = request_start + req->kv_num_pages;
    const int bytes_per_page = ints_per_page * static_cast<int>(sizeof(int));
    int selected_count = 0;

    for (int idx = request_start; idx < request_end && idx < kv_table.num_entries; ++idx) {
        KVPageEntry* entry = &kv_table.entries[idx];
        if (entry->request_id != req->request_id) {
            continue;
        }
        if (!entry->has_flag(KV_FLAG_RESIDENT) || entry->has_flag(KV_FLAG_DRAFT)) {
            continue;
        }

        metric.blocks_examined += 1;

        if (!entry->selected || entry->sparse_rank < 0 || entry->sparse_rank >= max_selected_pages) {
            continue;
        }

        const int rank = entry->sparse_rank;
        selected_page_ids[request_selected_offset + rank] = entry->page_id;

        if (page_payloads != nullptr && compacted_payloads != nullptr) {
            const int src_base = entry->page_id * ints_per_page;
            const int dst_base = compacted_offset + rank * ints_per_page;
            for (int j = 0; j < ints_per_page; ++j) {
                compacted_payloads[dst_base + j] = page_payloads[src_base + j];
            }
        }

        selected_count += 1;
    }

    metric.blocks_selected = selected_count;
    metric.bytes_read = selected_count * bytes_per_page;
    metric.bytes_saved = (metric.blocks_examined - selected_count) * bytes_per_page;
    metrics[request_slot] = metric;
}
