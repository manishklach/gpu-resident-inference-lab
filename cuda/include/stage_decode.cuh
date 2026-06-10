#ifndef STAGE_DECODE_CUH
#define STAGE_DECODE_CUH

#include "request_desc.h"

__device__ __forceinline__ void stage_decode(
    RequestDescriptor* req,
    int* draft_tokens,
    int block_size
) {
    if (!req->is_state(REQUEST_DECODE_READY)) return;

    if (req->has_flag(REQUEST_FLAG_SPECULATIVE_ENABLED)) {
        int count = block_size;
        int budget_left = req->max_new_tokens - req->output_token_count;
        if (count > budget_left) count = (budget_left > 0) ? budget_left : 0;
        req->draft_len = count;

        if (draft_tokens != nullptr) {
            for (int i = 0; i < count; i++) {
                draft_tokens[req->draft_offset + i] =
                    (req->last_token + 1 + i + req->request_id) % 32000;
            }
        }

        req->set_state(REQUEST_DRAFT_READY);
    } else {
        int next_token = (req->last_token + 1 + req->request_id) % 32000;
        req->last_token = next_token;
        req->output_token_count++;
        req->decode_pos++;

        if (next_token == req->eos_token_id) {
            req->set_flag(REQUEST_FLAG_EOS_SEEN);
            req->set_state(REQUEST_COMPLETE);
        } else if (req->output_token_count >= req->max_new_tokens) {
            req->set_state(REQUEST_COMPLETE);
        }
    }
}

#endif
