/**
 * tiered_kv_staging_kernel.cu — Research-stage tier-aware KV staging kernel.
 *
 * Role:
 *   Consumes sparse-selected KV pages plus tier metadata and emits an ordered
 *   staging plan for the resident decode path.
 *
 * Scope:
 *   - metadata-only staging order and buffer-slot assignment
 *   - prioritizes HBM-resident pages before DRAM and SSD fetches
 *   - no real async copy, queueing fabric, or overlapping execution
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

}  // namespace

__global__ void tiered_kv_staging_kernel(
    const int* selected_page_ids,
    const int* source_tiers,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* staged_page_ids,
    int* staged_source_tiers,
    int* staged_buffer_slots,
    TieredKVStagingMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    const int request_slot = blockIdx.x;
    const int base = request_slot * max_selected_pages;

    TieredKVStagingMetrics metric = {};
    metric.request_id = request_slot + 1;

    for (int i = 0; i < max_selected_pages; ++i) {
        staged_page_ids[base + i] = -1;
        staged_source_tiers[base + i] = MOVEMENT_TIER_NONE;
        staged_buffer_slots[base + i] = -1;
    }

    int out = 0;
    int previous_buffer_slot = -1;
    for (int wanted_tier = MOVEMENT_TIER_HBM; wanted_tier <= MOVEMENT_TIER_SSD; ++wanted_tier) {
        for (int i = 0; i < max_selected_pages; ++i) {
            const int page_id = selected_page_ids[base + i];
            const int tier = source_tiers[base + i];
            if (page_id < 0 || tier != wanted_tier) {
                continue;
            }

            const int out_index = base + out;
            const int buffer_slot = out & 1;
            staged_page_ids[out_index] = page_id;
            staged_source_tiers[out_index] = tier;
            staged_buffer_slots[out_index] = buffer_slot;

            metric.staged_pages += 1;
            metric.staging_bytes += bytes_per_page;
            if (tier == MOVEMENT_TIER_HBM) {
                metric.hbm_pages_staged += 1;
            } else {
                metric.dma_pages_staged += 1;
            }
            if (previous_buffer_slot >= 0 && previous_buffer_slot != buffer_slot) {
                metric.buffer_slot_switches += 1;
            }
            previous_buffer_slot = buffer_slot;
            out += 1;
        }
    }

    metrics[request_slot] = metric;
}
