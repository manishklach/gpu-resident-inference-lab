#ifndef STAGE_SPARSE_KV_SELECT_CUH
#define STAGE_SPARSE_KV_SELECT_CUH

#include "kv_page_table.h"
#include "request_desc.h"

__forceinline__ __device__ float sparse_kv_page_score(
    const KVPageEntry* entry,
    int decode_step
) {
    int distance = entry->start_token + entry->token_count - decode_step;
    if (distance < 0) {
        distance = -distance;
    }
    return static_cast<float>((entry->token_count * 10000) - (distance * 100) - entry->layer_id);
}

__forceinline__ __device__ void select_sparse_kv_blocks(
    RequestDescriptor* req,
    KVPageTable* kv_table,
    int top_k
) {
    if (req == nullptr || kv_table == nullptr || kv_table->entries == nullptr) {
        return;
    }

    const int decode_step = req->prompt_len + req->decode_pos;
    const int start = req->kv_table_offset;
    const int end = start + req->kv_num_pages;
    const int clamped_top_k = top_k < 0 ? 0 : top_k;

    for (int idx = start; idx < end && idx < kv_table->num_entries; ++idx) {
        KVPageEntry* entry = &kv_table->entries[idx];
        clear_page_selected(entry);
        entry->score = sparse_kv_page_score(entry, decode_step);
    }

    for (int rank = 0; rank < clamped_top_k; ++rank) {
        int best_index = -1;
        float best_score = -1.0e30f;

        for (int idx = start; idx < end && idx < kv_table->num_entries; ++idx) {
            KVPageEntry* entry = &kv_table->entries[idx];
            if (entry->request_id != req->request_id) {
                continue;
            }
            if (!entry->has_flag(KV_FLAG_RESIDENT) || entry->has_flag(KV_FLAG_DRAFT)) {
                continue;
            }
            if (entry->selected) {
                continue;
            }
            if (best_index < 0 || entry->score > best_score) {
                best_index = idx;
                best_score = entry->score;
            }
        }

        if (best_index < 0) {
            break;
        }

        mark_page_selected(&kv_table->entries[best_index], best_score, decode_step, rank);
    }
}

#endif
