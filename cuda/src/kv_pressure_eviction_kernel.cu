/**
 * kv_pressure_eviction_kernel.cu — Research-stage KV pressure + eviction pass.
 *
 * Role:
 *   Models how a resident runtime could scan a per-request KV working set
 *   under memory pressure, skip protected pages, and reclaim bytes by evicting
 *   draft pages before committed pages.
 *
 * Scope:
 *   - metadata-only pressure handling
 *   - deterministic draft-first eviction policy
 *   - no global allocator, queue, or real page migration
 */

#include <cuda_runtime.h>

#include "kv_page_table.h"
#include "research_kernel_metrics.h"

__global__ void kv_pressure_eviction_kernel(
    KVPageTable kv_table,
    int num_requests,
    int pages_per_request,
    int eviction_budget_pages,
    KVEvictionMetrics* metrics
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    const int request_slot = blockIdx.x;
    const int request_id = request_slot + 1;
    const int base = request_slot * pages_per_request;

    KVEvictionMetrics metric = {};
    metric.request_id = request_id;

    int remaining_budget = eviction_budget_pages;

    for (int pass = 0; pass < 2 && remaining_budget > 0; ++pass) {
        for (int i = 0; i < pages_per_request && remaining_budget > 0; ++i) {
            KVPageEntry* entry = &kv_table.entries[base + i];
            metric.pages_scanned += 1;

            if (entry->request_id != request_id) {
                continue;
            }
            if (!entry->has_flag(KV_FLAG_RESIDENT)) {
                continue;
            }
            if (entry->has_flag(KV_FLAG_PINNED)) {
                metric.pinned_pages_skipped += 1;
                continue;
            }
            if (entry->has_flag(KV_FLAG_SELECTED)) {
                metric.selected_pages_skipped += 1;
                continue;
            }

            const bool wants_draft = (pass == 0);
            const bool is_draft = entry->has_flag(KV_FLAG_DRAFT) || entry->is_state(KV_PAGE_DRAFT);
            const bool is_committed =
                entry->has_flag(KV_FLAG_COMMITTED) || entry->is_state(KV_PAGE_COMMITTED);

            if (wants_draft && !is_draft) {
                continue;
            }
            if (!wants_draft && (is_draft || !is_committed)) {
                continue;
            }

            mark_page_evictable(entry);
            entry->clear_flag(KV_FLAG_RESIDENT);
            entry->clear_flag(KV_FLAG_DRAFT);
            entry->clear_flag(KV_FLAG_COMMITTED);
            clear_page_selected(entry);

            metric.pages_evicted += 1;
            metric.reclaimed_bytes += kv_table.bytes_per_page;
            if (wants_draft) {
                metric.draft_pages_evicted += 1;
            } else {
                metric.committed_pages_evicted += 1;
            }
            remaining_budget -= 1;
        }
    }

    metrics[request_slot] = metric;
}
