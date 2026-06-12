/**
 * dma_aware_kv_movement_planner_kernel.cu — Research-stage tiered KV planner.
 *
 * Role:
 *   Converts sparse-selected KV page ids into a deterministic movement plan
 *   across conceptual HBM, DRAM, and SSD tiers.
 *
 * Scope:
 *   - metadata-only movement planning
 *   - deterministic tier classification for repeatable benchmarks
 *   - no real async copy, TMA, cp.async, DMA engine, or tiered allocator
 */

#include <cuda_runtime.h>

#include "research_kernel_metrics.h"

namespace {

enum MovementTier : int {
    MOVEMENT_TIER_NONE = -1,
    MOVEMENT_TIER_HBM = 0,
    MOVEMENT_TIER_DRAM = 1,
    MOVEMENT_TIER_SSD = 2,
};

enum MovementOp : int {
    MOVEMENT_OP_NONE = 0,
    MOVEMENT_OP_DRAM_TO_HBM = 1,
    MOVEMENT_OP_SSD_TO_HBM = 2,
};

__device__ __forceinline__ int classify_page_tier(int page_id) {
    if (page_id < 0) {
        return MOVEMENT_TIER_NONE;
    }

    const int bucket = page_id % 5;
    if (bucket <= 1) {
        return MOVEMENT_TIER_HBM;
    }
    if (bucket <= 3) {
        return MOVEMENT_TIER_DRAM;
    }
    return MOVEMENT_TIER_SSD;
}

}  // namespace

__global__ void dma_aware_kv_movement_planner_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* source_tiers,
    int* dma_ops,
    DMAMovementPlanMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    const int request_slot = blockIdx.x;
    const int base = request_slot * max_selected_pages;

    DMAMovementPlanMetrics metric = {};
    metric.request_id = request_slot + 1;

    for (int i = 0; i < max_selected_pages; ++i) {
        const int page_id = selected_page_ids[base + i];
        const int out_index = base + i;

        if (page_id < 0) {
            source_tiers[out_index] = MOVEMENT_TIER_NONE;
            dma_ops[out_index] = MOVEMENT_OP_NONE;
            continue;
        }

        const int tier = classify_page_tier(page_id);
        source_tiers[out_index] = tier;
        metric.pages_planned += 1;

        if (tier == MOVEMENT_TIER_HBM) {
            dma_ops[out_index] = MOVEMENT_OP_NONE;
            metric.hbm_hits += 1;
            metric.bytes_from_hbm += bytes_per_page;
        } else if (tier == MOVEMENT_TIER_DRAM) {
            dma_ops[out_index] = MOVEMENT_OP_DRAM_TO_HBM;
            metric.dram_fetches += 1;
            metric.dma_ops += 1;
            metric.bytes_moved += bytes_per_page;
            metric.bytes_from_dram += bytes_per_page;
        } else {
            dma_ops[out_index] = MOVEMENT_OP_SSD_TO_HBM;
            metric.ssd_fetches += 1;
            metric.dma_ops += 1;
            metric.bytes_moved += bytes_per_page;
            metric.bytes_from_ssd += bytes_per_page;
        }
    }

    metrics[request_slot] = metric;
}
