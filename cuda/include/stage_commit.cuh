#ifndef STAGE_COMMIT_CUH
#define STAGE_COMMIT_CUH

#include "request_desc.h"
#include "kv_page_table.h"
#include "stage_kv.cuh"

__device__ __forceinline__ void stage_commit(
    RequestDescriptor* req,
    KVPageTable* kv_table
) {
    if (!req->is_state(REQUEST_COMMIT_READY)) return;

    int accepted = req->accepted_prefix_len;

    if (accepted == 0) {
        int token = (req->last_token + 1 + req->request_id) % 32000;
        req->last_token = token;
        req->output_token_count++;
        req->decode_pos++;

        if (token == req->eos_token_id) {
            req->set_flag(REQUEST_FLAG_EOS_SEEN);
            req->set_state(REQUEST_COMPLETE);
            return;
        }
        if (req->output_token_count >= req->max_new_tokens) {
            req->set_state(REQUEST_COMPLETE);
            return;
        }
        req->set_state(REQUEST_DECODE_READY);
        return;
    }

    req->last_token = 0;
    req->output_token_count += accepted;
    req->decode_pos += accepted;

    if (kv_table != nullptr && req->kv_num_pages > 0) {
        int hit_pages = (accepted + kv_table->page_size_tokens - 1) / kv_table->page_size_tokens;
        int total_pages_for_request = req->kv_num_pages;
        int draft_pages = total_pages_for_request - hit_pages;

        if (hit_pages > total_pages_for_request) hit_pages = total_pages_for_request;
        if (draft_pages < 0) draft_pages = 0;

        commit_draft_kv_region(kv_table, req->kv_table_offset, hit_pages);
        if (draft_pages > 0) {
            discard_rejected_kv_region(kv_table, req->kv_table_offset + hit_pages, draft_pages);
        }
    }

    if (req->output_token_count >= req->max_new_tokens) {
        req->set_state(REQUEST_COMPLETE);
        return;
    }

    req->set_state(REQUEST_DECODE_READY);
}

#endif
