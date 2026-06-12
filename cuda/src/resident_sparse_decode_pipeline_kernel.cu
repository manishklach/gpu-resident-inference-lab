/**
 * resident_sparse_decode_pipeline_kernel.cu — Research-stage end-to-end loop.
 *
 * Role:
 *   A small resident pipeline kernel that combines:
 *     1. sparse KV selection,
 *     2. compact sparse gather,
 *     3. deterministic decode over compacted tiles,
 *     4. speculative verify + commit-style lifecycle mutation.
 *
 * Scope:
 *   - still fake math
 *   - real control flow and memory traffic over compacted payloads
 *   - intended as the first end-to-end resident research kernel beyond the
 *     metadata-only mega-kernel scaffold
 */

#include <cuda_runtime.h>

#include "kv_page_table.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"
#include "stage_kv.cuh"
#include "stage_sparse_kv_select.cuh"
#include "stage_spec_verify.cuh"

__global__ void resident_sparse_decode_pipeline_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    const int* page_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* draft_tokens,
    ResidentPipelineMetrics* metrics,
    int max_iterations
) {
    if (blockIdx.x >= num_requests || threadIdx.x != 0) {
        return;
    }

    RequestDescriptor* req = &requests[blockIdx.x];
    ResidentPipelineMetrics metric = {};
    metric.request_id = req->request_id;

    for (int iteration = 0; iteration < max_iterations && !req->is_done(); ++iteration) {
        metric.loop_iterations += 1;
        select_sparse_kv_blocks(req, &kv_table, max_selected_pages);

        int payload_sum = 0;
        int selected_count = 0;
        for (int idx = req->kv_table_offset;
             idx < req->kv_table_offset + req->kv_num_pages && idx < kv_table.num_entries;
             ++idx) {
            KVPageEntry* entry = &kv_table.entries[idx];
            if (entry->request_id != req->request_id || !entry->selected) {
                continue;
            }
            selected_count += 1;
            payload_sum += page_payloads[entry->page_id * ints_per_page];
        }

        metric.blocks_selected += selected_count;
        metric.bytes_read += selected_count * ints_per_page * static_cast<int>(sizeof(int));

        int count = req->current_block_size > 0 ? req->current_block_size : 1;
        int budget_left = req->max_new_tokens - req->output_token_count;
        if (count > budget_left) {
            count = budget_left > 0 ? budget_left : 0;
        }
        req->draft_len = count;

        for (int i = 0; i < count; ++i) {
            draft_tokens[req->draft_offset + i] =
                (req->last_token + payload_sum + req->request_id + i + 1) % 32000;
        }

        int accepted = 0;
        for (int i = 0; i < count; ++i) {
            const int token = draft_tokens[req->draft_offset + i];
            if (req->eos_token_id > 0 && token == req->eos_token_id) {
                break;
            }
            if (token % 4 == 0) {
                break;
            }
            accepted += 1;
        }

        req->accepted_prefix_len = accepted;
        update_block_size_device(
            req,
            accepted,
            count,
            req->has_flag(REQUEST_FLAG_KV_PRESSURE) ? 1 : 0
        );

        metric.accepted_tokens += accepted;

        if (accepted == 0) {
            const int fallback_token = (req->last_token + 1 + req->request_id) % 32000;
            req->last_token = fallback_token;
            req->output_token_count += 1;
            req->decode_pos += 1;
            if (fallback_token == req->eos_token_id || req->output_token_count >= req->max_new_tokens) {
                req->set_state(REQUEST_COMPLETE);
            } else {
                req->set_state(REQUEST_DECODE_READY);
            }
            continue;
        }

        req->last_token = 0;
        req->output_token_count += accepted;
        req->decode_pos += accepted;

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
            discard_rejected_kv_region(&kv_table, req->kv_table_offset + committed_pages, released_pages);
        }
        metric.committed_pages += committed_pages;

        if (req->output_token_count >= req->max_new_tokens) {
            req->set_state(REQUEST_COMPLETE);
        } else {
            req->set_state(REQUEST_DECODE_READY);
        }
    }

    metrics[blockIdx.x] = metric;
}
