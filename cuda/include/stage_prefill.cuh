#ifndef STAGE_PREFILL_CUH
#define STAGE_PREFILL_CUH

#include "request_desc.h"
#include "kv_page_table.h"

__device__ __forceinline__ void stage_prefill(
    RequestDescriptor* req,
    KVPageTable* kv_table
) {
    if (!req->is_state(REQUEST_PREFILL_READY)) return;

    req->decode_pos = 0;
    req->output_token_count = 0;
    req->last_token = (req->request_id * 7 + 3) % 32000;

    if (req->kv_num_pages > 0 && kv_table != nullptr) {
        for (int i = 0; i < req->kv_num_pages; i++) {
            KVPageEntry* entry = &kv_table->entries[req->kv_table_offset + i];
            mark_page_committed(entry);
            mark_page_pinned(entry);
            mark_page_resident(entry);
            entry->request_id = req->request_id;
        }
    }

    req->set_state(REQUEST_DECODE_READY);
}

#endif
