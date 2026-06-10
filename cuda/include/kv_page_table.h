#ifndef KV_PAGE_TABLE_H
#define KV_PAGE_TABLE_H

#include <cstdint>

enum KVPageState : int {
    KV_PAGE_FREE       = 0,
    KV_PAGE_DRAFT      = 1,
    KV_PAGE_COMMITTED  = 2,
    KV_PAGE_EVICTABLE  = 3,
    KV_PAGE_PINNED     = 4,
};

enum KVPageFlags : int {
    KV_FLAG_RESIDENT   = 1 << 0,
    KV_FLAG_DRAFT      = 1 << 1,
    KV_FLAG_COMMITTED  = 1 << 2,
    KV_FLAG_PINNED     = 1 << 3,
    KV_FLAG_EVICTABLE  = 1 << 4,
};

struct KVPageEntry {
    int page_id;
    int request_id;
    int layer_id;
    int start_token;
    int token_count;
    int state;
    int flags;

    __host__ __device__ bool has_flag(KVPageFlags f) const {
        return (flags & static_cast<int>(f)) != 0;
    }

    __host__ __device__ void set_flag(KVPageFlags f) {
        flags |= static_cast<int>(f);
    }

    __host__ __device__ void clear_flag(KVPageFlags f) {
        flags &= ~static_cast<int>(f);
    }

    __host__ __device__ bool is_state(KVPageState s) const {
        return state == static_cast<int>(s);
    }

    __host__ __device__ void set_state(KVPageState s) {
        state = static_cast<int>(s);
    }
};

struct KVPageTable {
    KVPageEntry* entries;
    int num_entries;
    int page_size_tokens;
    int bytes_per_page;

    __device__ KVPageEntry* find_entry(int layer_id, int token_pos) {
        for (int i = 0; i < num_entries; i++) {
            KVPageEntry& e = entries[i];
            if (e.layer_id == layer_id &&
                token_pos >= e.start_token &&
                token_pos < e.start_token + e.token_count) {
                return &e;
            }
        }
        return nullptr;
    }
};

__device__ __forceinline__ void mark_page_draft(KVPageEntry* entry) {
    entry->set_state(KV_PAGE_DRAFT);
    entry->set_flag(KV_FLAG_DRAFT);
    entry->clear_flag(KV_FLAG_COMMITTED);
    entry->clear_flag(KV_FLAG_PINNED);
}

__device__ __forceinline__ void mark_page_committed(KVPageEntry* entry) {
    entry->set_state(KV_PAGE_COMMITTED);
    entry->set_flag(KV_FLAG_COMMITTED);
    entry->clear_flag(KV_FLAG_DRAFT);
    entry->set_flag(KV_FLAG_RESIDENT);
}

__device__ __forceinline__ void mark_page_evictable(KVPageEntry* entry) {
    entry->set_state(KV_PAGE_EVICTABLE);
    entry->set_flag(KV_FLAG_EVICTABLE);
    entry->clear_flag(KV_FLAG_PINNED);
}

__device__ __forceinline__ void mark_page_pinned(KVPageEntry* entry) {
    entry->set_state(KV_PAGE_PINNED);
    entry->set_flag(KV_FLAG_PINNED);
}

__device__ __forceinline__ void mark_page_resident(KVPageEntry* entry) {
    entry->set_flag(KV_FLAG_RESIDENT);
}

#endif
