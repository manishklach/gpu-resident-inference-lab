#ifndef STAGE_SPEC_VERIFY_CUH
#define STAGE_SPEC_VERIFY_CUH

#include "request_desc.h"

__device__ __forceinline__ void stage_spec_verify(
    RequestDescriptor* req,
    int* draft_tokens
) {
    if (!req->is_state(REQUEST_DRAFT_READY) && !req->is_state(REQUEST_VERIFY_READY)) return;

    int accepted = 0;
    int limit = req->draft_len;
    int budget_left = req->max_new_tokens - req->output_token_count;
    if (limit > budget_left) limit = budget_left;

    if (draft_tokens != nullptr) {
        for (int i = 0; i < limit; i++) {
            int token = draft_tokens[req->draft_offset + i];
            if (req->eos_token_id > 0 && token == req->eos_token_id) break;
            if (token % 4 == 0) break;
            accepted++;
        }
    }

    req->accepted_prefix_len = accepted;
    req->set_state(REQUEST_COMMIT_READY);
}

#endif
