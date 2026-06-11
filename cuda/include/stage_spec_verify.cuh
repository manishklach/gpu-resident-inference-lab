#ifndef STAGE_SPEC_VERIFY_CUH
#define STAGE_SPEC_VERIFY_CUH

#include "request_desc.h"

// ─── Device-side adaptive block sizing ──────────────────────────
// Uses the same thresholds as the Python spec_decode.AdaptiveBlockPolicy
// so the Python simulator remains a valid correctness specification.
//   high_accept_threshold = 0.80
//   low_accept_threshold  = 0.50
//   alpha                 = 0.20
//   pressure_cap          = 2
//   min_block             = 1
//   max_block             = 8

__device__ __forceinline__ void update_block_size_device(
    RequestDescriptor* req,
    int accepted_count,
    int proposed_count,
    int kv_pressure
) {
    float observed = (proposed_count > 0)
        ? (float)accepted_count / (float)proposed_count
        : 0.0f;

    float alpha = 0.20f;
    req->ema_acceptance_rate = alpha * observed
        + (1.0f - alpha) * req->ema_acceptance_rate;

    float rate = req->ema_acceptance_rate;
    int next_block;
    if (rate >= 0.80f) {
        next_block = 8;
    } else if (rate >= 0.50f) {
        next_block = 4;
    } else {
        next_block = 1;
    }

    if (kv_pressure && next_block > 2) {
        next_block = 2;
    }

    if (next_block < 1) next_block = 1;
    req->current_block_size = next_block;
}

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

    // Adaptive block sizing: update ema_acceptance_rate and current_block_size
    int kv_pressure = req->has_flag(REQUEST_FLAG_KV_PRESSURE) ? 1 : 0;
    update_block_size_device(req, accepted, limit, kv_pressure);

    req->set_state(REQUEST_COMMIT_READY);
}

#endif
