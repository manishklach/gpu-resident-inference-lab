/**
 * kv_prefetch_planner_kernel.cu — Research-stage KV prefetch planning kernel.
 *
 * Role:
 *   Converts selected sparse KV page ids into a simple prefetch plan with
 *   double-buffer slot assignments.
 *
 * Scope:
 *   - metadata-only prefetch planning
 *   - no real DMA engine or async copy implementation
 *   - intended as a bridge toward DMA-aware KV movement models
 */

#include <cuda_runtime.h>

#include "research_kernel_metrics.h"

__global__ void kv_prefetch_planner_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* prefetched_page_ids,
    int* buffer_slots,
    KVPrefetchMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    const int request_slot = blockIdx.x;
    const int base = request_slot * max_selected_pages;

    KVPrefetchMetrics metric = {};
    metric.request_id = request_slot + 1;

    for (int i = 0; i < max_selected_pages; ++i) {
        const int page_id = selected_page_ids[base + i];
        prefetched_page_ids[base + i] = page_id;
        buffer_slots[base + i] = (page_id >= 0) ? (i & 1) : -1;
        if (page_id >= 0) {
            metric.prefetch_pages += 1;
        }
    }

    metric.prefetch_bytes = metric.prefetch_pages * bytes_per_page;
    metrics[request_slot] = metric;
}
