#ifndef STAGE_KV_CUH
#define STAGE_KV_CUH

#include "request_desc.h"
#include "kv_page_table.h"

__device__ __forceinline__ void allocate_fake_kv_pages_for_request(
    KVPageTable* kv_table,
    int offset,
    int num_pages,
    int request_id,
    int start_token,
    int page_size
) {
    if (kv_table == nullptr || kv_table->entries == nullptr) return;
    for (int i = 0; i < num_pages; i++) {
        KVPageEntry* entry = &kv_table->entries[offset + i];
        entry->page_id = offset + i;
        entry->request_id = request_id;
        entry->start_token = start_token + i * page_size;
        entry->token_count = 0;
        entry->state = KV_PAGE_FREE;
        entry->flags = 0;
    }
}

__device__ __forceinline__ void mark_draft_kv_region(
    KVPageTable* kv_table,
    int offset,
    int num_pages
) {
    if (kv_table == nullptr) return;
    for (int i = 0; i < num_pages; i++) {
        KVPageEntry* entry = &kv_table->entries[offset + i];
        if (entry->state != KV_PAGE_FREE) continue;
        mark_page_draft(entry);
    }
}

__device__ __forceinline__ void commit_draft_kv_region(
    KVPageTable* kv_table,
    int offset,
    int num_pages
) {
    if (kv_table == nullptr) return;
    for (int i = 0; i < num_pages; i++) {
        KVPageEntry* entry = &kv_table->entries[offset + i];
        mark_page_committed(entry);
    }
}

__device__ __forceinline__ void discard_rejected_kv_region(
    KVPageTable* kv_table,
    int offset,
    int num_pages
) {
    if (kv_table == nullptr) return;
    for (int i = 0; i < num_pages; i++) {
        KVPageEntry* entry = &kv_table->entries[offset + i];
        if (entry->is_state(KV_PAGE_DRAFT)) {
            mark_page_evictable(entry);
        }
    }
}

__device__ __forceinline__ void touch_kv_pages(
    KVPageTable* kv_table,
    int offset,
    int num_pages
) {
    if (kv_table == nullptr) return;
    for (int i = 0; i < num_pages; i++) {
        KVPageEntry* entry = &kv_table->entries[offset + i];
        if (entry->has_flag(KV_FLAG_EVICTABLE)) {
            entry->clear_flag(KV_FLAG_EVICTABLE);
            mark_page_resident(entry);
        }
    }
}

#endif
