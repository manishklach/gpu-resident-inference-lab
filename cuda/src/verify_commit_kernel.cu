/**
 * verify_commit_kernel.cu — Fused speculative verify + commit kernel.
 *
 * Role:
 *   A standalone next-step kernel that fuses:
 *     1. speculative draft verification,
 *     2. accepted-prefix commit,
 *     3. rejected-page release,
 *     4. request-state advancement.
 *
 * Scope:
 *   - deterministic verification rule, same spirit as stage_spec_verify.cuh
 *   - metadata-only KV transitions through stage_kv helpers
 *   - designed as a future resident-loop building block
 *   - not production sampling or attention math
 */

#include <cuda_runtime.h>

#include "kv_page_table.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"
#include "stage_kv.cuh"
#include "stage_spec_verify.cuh"

__global__ void fused_verify_and_commit_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    int* draft_tokens,
    VerifyCommitMetrics* metrics
) {
    if (blockIdx.x >= num_requests) {
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }

    RequestDescriptor* req = &requests[blockIdx.x];
    VerifyCommitMetrics metric = {};
    metric.request_id = req->request_id;

    if (!req->is_state(REQUEST_DRAFT_READY) && !req->is_state(REQUEST_VERIFY_READY)) {
        metrics[blockIdx.x] = metric;
        return;
    }

    int accepted = 0;
    int limit = req->draft_len;
    int budget_left = req->max_new_tokens - req->output_token_count;
    if (limit > budget_left) {
        limit = budget_left;
    }

    metric.speculative_candidates = limit;

    if (draft_tokens != nullptr) {
        for (int i = 0; i < limit; ++i) {
            const int token = draft_tokens[req->draft_offset + i];
            if (req->eos_token_id > 0 && token == req->eos_token_id) {
                break;
            }
            if (token % 4 == 0) {
                break;
            }
            accepted += 1;
        }
    }

    req->accepted_prefix_len = accepted;
    update_block_size_device(
        req,
        accepted,
        limit,
        req->has_flag(REQUEST_FLAG_KV_PRESSURE) ? 1 : 0
    );

    metric.accepted_tokens = accepted;
    metric.rejected_tokens = limit - accepted;

    if (accepted == 0) {
        const int fallback_token = (req->last_token + 1 + req->request_id) % 32000;
        req->last_token = fallback_token;
        req->output_token_count += 1;
        req->decode_pos += 1;

        if (fallback_token == req->eos_token_id) {
            req->set_flag(REQUEST_FLAG_EOS_SEEN);
            req->set_state(REQUEST_COMPLETE);
        } else if (req->output_token_count >= req->max_new_tokens) {
            req->set_state(REQUEST_COMPLETE);
        } else {
            req->set_state(REQUEST_DECODE_READY);
        }

        metrics[blockIdx.x] = metric;
        return;
    }

    req->last_token = 0;
    req->output_token_count += accepted;
    req->decode_pos += accepted;

    if (kv_table.entries != nullptr && req->kv_num_pages > 0) {
        int committed_pages = (accepted + kv_table.page_size_tokens - 1) / kv_table.page_size_tokens;
        if (committed_pages > req->kv_num_pages) {
            committed_pages = req->kv_num_pages;
        }
        int released_pages = req->kv_num_pages - committed_pages;
        if (released_pages < 0) {
            released_pages = 0;
        }

        commit_draft_kv_region(&kv_table, req->kv_table_offset, committed_pages);
        if (released_pages > 0) {
            discard_rejected_kv_region(
                &kv_table,
                req->kv_table_offset + committed_pages,
                released_pages
            );
        }

        metric.committed_pages = committed_pages;
        metric.released_pages = released_pages;
    }

    if (req->output_token_count >= req->max_new_tokens) {
        req->set_state(REQUEST_COMPLETE);
    } else {
        req->set_state(REQUEST_DECODE_READY);
    }

    metrics[blockIdx.x] = metric;
}
