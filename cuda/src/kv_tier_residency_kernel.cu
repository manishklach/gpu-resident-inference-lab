/**
 * kv_tier_residency_kernel.cu — Research-stage hierarchical tier rebalance.
 *
 * Role:
 *   Models promote/demote decisions across conceptual HBM, DRAM, and SSD
 *   tiers once sparse selection identifies a hot working set.
 *
 * Scope:
 *   - metadata-only residency management
 *   - one-step promotions for selected pages
 *   - capacity-driven demotions for non-selected pages
 *   - no real allocator, migration engine, or shared global budgets
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

__device__ __forceinline__ bool is_selected_page(
    int page_id,
    const int* selected_page_ids,
    int base,
    int max_selected_pages
) {
    for (int i = 0; i < max_selected_pages; ++i) {
        if (selected_page_ids[base + i] == page_id) {
            return true;
        }
    }
    return false;
}

}  // namespace

__global__ void kv_tier_residency_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int pages_per_request,
    int hbm_capacity_pages,
    int dram_capacity_pages,
    int bytes_per_page,
    int* page_tiers,
    KVTierResidencyMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    const int request_slot = blockIdx.x;
    const int selected_base = request_slot * max_selected_pages;
    const int tier_base = request_slot * pages_per_request;

    KVTierResidencyMetrics metric = {};
    metric.request_id = request_slot + 1;

    for (int i = 0; i < max_selected_pages; ++i) {
        const int page_id = selected_page_ids[selected_base + i];
        if (page_id < 0) {
            continue;
        }

        const int local_index = page_id - tier_base;
        if (local_index < 0 || local_index >= pages_per_request) {
            continue;
        }

        int& tier = page_tiers[tier_base + local_index];
        if (tier == MOVEMENT_TIER_DRAM) {
            tier = MOVEMENT_TIER_HBM;
            metric.promotions_to_hbm += 1;
            metric.pages_rebalanced += 1;
            metric.bytes_promoted += bytes_per_page;
        } else if (tier == MOVEMENT_TIER_SSD) {
            tier = MOVEMENT_TIER_DRAM;
            metric.promotions_to_dram += 1;
            metric.pages_rebalanced += 1;
            metric.bytes_promoted += bytes_per_page;
        }
    }

    int hbm_count = 0;
    int dram_count = 0;
    int ssd_count = 0;
    for (int i = 0; i < pages_per_request; ++i) {
        const int tier = page_tiers[tier_base + i];
        if (tier == MOVEMENT_TIER_HBM) {
            hbm_count += 1;
        } else if (tier == MOVEMENT_TIER_DRAM) {
            dram_count += 1;
        } else {
            ssd_count += 1;
        }
    }

    for (int i = pages_per_request - 1; i >= 0 && hbm_count > hbm_capacity_pages; --i) {
        const int page_id = tier_base + i;
        int& tier = page_tiers[tier_base + i];
        if (tier != MOVEMENT_TIER_HBM) {
            continue;
        }
        if (is_selected_page(page_id, selected_page_ids, selected_base, max_selected_pages)) {
            continue;
        }
        tier = MOVEMENT_TIER_DRAM;
        hbm_count -= 1;
        dram_count += 1;
        metric.demotions_to_dram += 1;
        metric.pages_rebalanced += 1;
        metric.bytes_demoted += bytes_per_page;
    }

    for (int i = pages_per_request - 1; i >= 0 && dram_count > dram_capacity_pages; --i) {
        const int page_id = tier_base + i;
        int& tier = page_tiers[tier_base + i];
        if (tier != MOVEMENT_TIER_DRAM) {
            continue;
        }
        if (is_selected_page(page_id, selected_page_ids, selected_base, max_selected_pages)) {
            continue;
        }
        tier = MOVEMENT_TIER_SSD;
        dram_count -= 1;
        ssd_count += 1;
        metric.demotions_to_ssd += 1;
        metric.pages_rebalanced += 1;
        metric.bytes_demoted += bytes_per_page;
    }

    metric.final_hbm_pages = hbm_count;
    metric.final_dram_pages = dram_count;
    metric.final_ssd_pages = ssd_count;
    metrics[request_slot] = metric;
}
