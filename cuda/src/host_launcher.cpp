/**
 * host_launcher.cpp — CUDA measurement harness for orchestration overhead.
 *
 * Role:
 *   Compares two execution-control paths:
 *   A) Baseline host-launched decode — CPU launches one kernel per step.
 *   B) Persistent mega-kernel — CPU launches once, GPU advances requests.
 *
 * The comparison measures orchestration overhead:
 *   - host_kernel_launches: how many times the CPU launches a GPU kernel
 *   - host_synchronizations: how many times the CPU synchronizes with the GPU
 *   - elapsed_ms: wall time for the control-flow scaffold
 *   - tokens_per_second: throughput through deterministic stub math
 *   - launch_reduction / sync_reduction: ratio of baseline to mega-kernel
 *   - speedup_vs_baseline: elapsed_ms ratio (baseline / mega-kernel)
 *
 * All math is fake/deterministic. No real transformer operations.
 * The measurement target is launch/sync reduction, not model FLOPs.
 *
 * CLI usage:
 *   ./xlpk_cuda_smoke --mode both --requests 8 --tokens 128 --draft-len 4
 *   ./xlpk_cuda_smoke --mode sweep --csv results.csv
 *
 * CSV output (--csv):
 *   header: mode,requests,tokens_per_request,draft_len,
 *           host_kernel_launches,host_synchronizations,
 *           completed_requests,target_requests,
 *           tokens_generated,elapsed_ms,tokens_per_second,
 *           launch_reduction,sync_reduction,speedup_vs_baseline
 *
 * Sweep mode (--mode sweep):
 *   Runs Cartesian product of requests [2,4,8,16],
 *   tokens_per_request [32,64,128], draft_len [1,4,8]
 *   writing one row per configuration and path.
 *
 * NVTX (optional):
 *   Compile with -DXLPK_ENABLE_NVTX to add NVTX range annotations
 *   around the baseline loop and the mega-kernel launch.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#ifdef XLPK_ENABLE_NVTX
#include <nvToolsExt.h>
#endif

#include "request_desc.h"
#include "kv_page_table.h"
#include "kernel_status.h"
#include "research_kernel_metrics.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), #call); \
        exit(1); \
    } \
} while(0)

extern __global__ void xl_persistent_megakernel(
    RequestDescriptor* requests, int num_requests,
    KVPageTable kv_table, int* draft_tokens,
    int* shutdown_flag, int* done_counter,
    int max_iterations, int block_size
);

extern __global__ void baseline_host_decode_step_kernel(
    RequestDescriptor* requests, int num_requests
);

extern __global__ void sparse_kv_gather_and_score_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    const int* page_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* selected_page_ids,
    int* compacted_payloads,
    SparseKVGatherMetrics* metrics
);

extern __global__ void fused_verify_and_commit_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    int* draft_tokens,
    VerifyCommitMetrics* metrics
);

extern __global__ void resident_schedule_requests_kernel(
    RequestDescriptor* requests,
    int num_requests,
    int* scheduled_request_ids,
    int* scheduled_priorities,
    SchedulerKernelMetrics* metrics
);

extern __global__ void kv_prefetch_planner_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* prefetched_page_ids,
    int* buffer_slots,
    KVPrefetchMetrics* metrics
);

extern __global__ void dma_aware_kv_movement_planner_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* source_tiers,
    int* dma_ops,
    DMAMovementPlanMetrics* metrics
);

extern __global__ void tiered_kv_staging_kernel(
    const int* selected_page_ids,
    const int* source_tiers,
    int num_requests,
    int max_selected_pages,
    int bytes_per_page,
    int* staged_page_ids,
    int* staged_source_tiers,
    int* staged_buffer_slots,
    TieredKVStagingMetrics* metrics
);

extern __global__ void kv_pressure_eviction_kernel(
    KVPageTable kv_table,
    int num_requests,
    int pages_per_request,
    int eviction_budget_pages,
    KVEvictionMetrics* metrics
);

extern __global__ void kv_tier_residency_kernel(
    const int* selected_page_ids,
    int num_requests,
    int max_selected_pages,
    int pages_per_request,
    int hbm_capacity_pages,
    int dram_capacity_pages,
    int bytes_per_page,
    int* page_tiers,
    KVTierResidencyMetrics* metrics
);

extern __global__ void compacted_sparse_decode_kernel(
    RequestDescriptor* requests,
    int num_requests,
    const int* selected_page_ids,
    const int* compacted_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* draft_tokens,
    CompactedDecodeMetrics* metrics
);

extern __global__ void resident_sparse_decode_pipeline_kernel(
    RequestDescriptor* requests,
    int num_requests,
    KVPageTable kv_table,
    const int* page_payloads,
    int ints_per_page,
    int max_selected_pages,
    int* draft_tokens,
    ResidentPipelineMetrics* metrics,
    int max_iterations
);

constexpr int BLOCK_SIZE = 256;
constexpr int MEGAKERNEL_BLOCK_SIZE = 1;
constexpr int EOS_TOKEN_ID = 0;

struct RunMetrics {
    const char* mode;
    int requests;
    int tokens_per_request;
    int draft_len;
    int host_kernel_launches;
    int host_synchronizations;
    int completed_requests;
    int target_requests;
    int tokens_generated;
    float elapsed_ms;
    float tokens_per_second;
    int launch_reduction;
    int sync_reduction;
    float speedup_vs_baseline;
};

struct ResearchRunMetrics {
    const char* mode;
    int requests;
    int draft_len;
    float elapsed_ms;
    int blocks_examined;
    int blocks_selected;
    int bytes_read;
    int bytes_saved;
    int speculative_candidates;
    int accepted_tokens;
    int rejected_tokens;
    int committed_pages;
    int released_pages;
    int scheduler_requests_examined;
    int scheduler_requests_scheduled;
    int prefetch_pages;
    int prefetch_bytes;
    int dma_pages_planned;
    int dma_hbm_hits;
    int dma_dram_fetches;
    int dma_ssd_fetches;
    int dma_ops;
    int dma_bytes_moved;
    int dma_bytes_from_hbm;
    int dma_bytes_from_dram;
    int dma_bytes_from_ssd;
    int staged_pages;
    int staged_hbm_pages;
    int staged_dma_pages;
    int buffer_slot_switches;
    int staging_bytes;
    int eviction_pages_scanned;
    int eviction_pages_evicted;
    int eviction_draft_pages;
    int eviction_committed_pages;
    int eviction_pinned_skipped;
    int eviction_selected_skipped;
    int eviction_reclaimed_bytes;
    int tier_pages_rebalanced;
    int tier_promotions_to_hbm;
    int tier_promotions_to_dram;
    int tier_demotions_to_dram;
    int tier_demotions_to_ssd;
    int tier_final_hbm_pages;
    int tier_final_dram_pages;
    int tier_final_ssd_pages;
    int tier_bytes_promoted;
    int tier_bytes_demoted;
    int generated_tokens;
    int loop_iterations;
};

static RequestDescriptor* alloc_and_copy_to_device(RequestDescriptor* host, int n) {
    size_t bytes = n * sizeof(RequestDescriptor);
    RequestDescriptor* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice));
    return d;
}

static void copy_back(RequestDescriptor* host, RequestDescriptor* dev, int n) {
    size_t bytes = n * sizeof(RequestDescriptor);
    CUDA_CHECK(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
}

static void print_requests(const char* label, RequestDescriptor* reqs, int n) {
    printf("--- %s ---\n", label);
    for (int i = 0; i < n; i++) {
        printf("  req %d: state=%s decode_pos=%d output_count=%d last_token=%d flags=0x%x\n",
               reqs[i].request_id,
               request_state_name(static_cast<RequestState>(reqs[i].state)),
               reqs[i].decode_pos, reqs[i].output_token_count,
               reqs[i].last_token, reqs[i].flags);
    }
    printf("\n");
}

static int count_completed(RequestDescriptor* reqs, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (reqs[i].is_state(REQUEST_COMPLETE)) count++;
    }
    return count;
}

static int total_tokens_generated(RequestDescriptor* reqs, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += reqs[i].output_token_count;
    }
    return total;
}

static void init_requests(RequestDescriptor* reqs, int n, int max_tokens, bool speculative, int draft_offset_stride) {
    for (int i = 0; i < n; i++) {
        reqs[i].request_id = i + 1;
        reqs[i].state = REQUEST_DECODE_READY;
        reqs[i].flags = speculative ? REQUEST_FLAG_SPECULATIVE_ENABLED : 0;
        reqs[i].priority = i;
        reqs[i].prompt_len = 0;
        reqs[i].decode_pos = 0;
        reqs[i].max_new_tokens = max_tokens;
        reqs[i].eos_token_id = EOS_TOKEN_ID;
        reqs[i].last_token = (reqs[i].request_id * 7 + 3) % 32000;
        reqs[i].output_token_count = 0;
        reqs[i].draft_offset = i * draft_offset_stride;
        reqs[i].draft_len = 0;
        reqs[i].accepted_prefix_len = 0;
        reqs[i].kv_table_offset = i * 4;
        reqs[i].kv_num_pages = 4;
        reqs[i].error_code = 0;
        reqs[i].ema_acceptance_rate = 0.80f;
        reqs[i].current_block_size = 4;
    }
}

static void init_kv_entries(
    KVPageEntry* kv_entries,
    int num_requests,
    int pages_per_request,
    int page_size_tokens,
    int state,
    int flags
) {
    int total_kv_entries = num_requests * pages_per_request;
    memset(kv_entries, 0, total_kv_entries * sizeof(KVPageEntry));
    for (int i = 0; i < total_kv_entries; ++i) {
        kv_entries[i].page_id = i;
        kv_entries[i].request_id = (i / pages_per_request) + 1;
        kv_entries[i].layer_id = i % 2;
        kv_entries[i].start_token = (i % pages_per_request) * page_size_tokens;
        kv_entries[i].token_count = page_size_tokens;
        kv_entries[i].state = state;
        kv_entries[i].flags = flags;
        kv_entries[i].score = 0.0f;
        kv_entries[i].selected = 0;
        kv_entries[i].last_selected_step = -1;
        kv_entries[i].sparse_rank = -1;
    }
}

static KVPageTable make_device_kv_table(KVPageEntry* host_entries, int total_entries, int page_size_tokens, int bytes_per_page, KVPageEntry** d_kv_entries_out) {
    KVPageEntry* d_kv_entries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kv_entries, total_entries * sizeof(KVPageEntry)));
    CUDA_CHECK(cudaMemcpy(
        d_kv_entries,
        host_entries,
        total_entries * sizeof(KVPageEntry),
        cudaMemcpyHostToDevice
    ));

    KVPageTable table;
    table.entries = d_kv_entries;
    table.num_entries = total_entries;
    table.page_size_tokens = page_size_tokens;
    table.bytes_per_page = bytes_per_page;
    *d_kv_entries_out = d_kv_entries;
    return table;
}

static RunMetrics run_baseline_path(
    int N, int tokens_per_request, int draft_len, int max_iterations
) {
    (void)draft_len;
    RunMetrics m;
    m.mode = "baseline";
    m.requests = N;
    m.tokens_per_request = tokens_per_request;
    m.draft_len = 0;
    m.host_kernel_launches = 0;
    m.host_synchronizations = 0;
    m.completed_requests = 0;
    m.target_requests = N;
    m.tokens_generated = 0;
    m.elapsed_ms = 0.0f;
    m.tokens_per_second = 0.0f;
    m.launch_reduction = 0;
    m.sync_reduction = 0;
    m.speedup_vs_baseline = 1.0f;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, tokens_per_request, false, 0);
    print_requests("Baseline initial state", requests, N);

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < max_iterations; iter++) {
        bool all_done = true;
        for (int i = 0; i < N; i++) {
            if (!requests[i].is_done()) { all_done = false; break; }
        }
        if (all_done) break;

        baseline_host_decode_step_kernel<<<1, BLOCK_SIZE>>>(d_reqs, N);
        CUDA_CHECK(cudaGetLastError());
        m.host_kernel_launches++;

        CUDA_CHECK(cudaDeviceSynchronize());
        m.host_synchronizations++;

        copy_back(requests, d_reqs, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    print_requests("After baseline decode", requests, N);
    m.completed_requests = count_completed(requests, N);
    m.tokens_generated = total_tokens_generated(requests, N);
    m.tokens_per_second = (m.elapsed_ms > 0.0f) ? (m.tokens_generated / (m.elapsed_ms / 1000.0f)) : 0.0f;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    delete[] requests;
    return m;
}

static RunMetrics run_megakernel_path(
    int N, int tokens_per_request, int draft_len, int max_iterations
) {
    RunMetrics m;
    m.mode = "mega_kernel";
    m.requests = N;
    m.tokens_per_request = tokens_per_request;
    m.draft_len = draft_len;
    m.host_kernel_launches = 0;
    m.host_synchronizations = 0;
    m.completed_requests = 0;
    m.target_requests = N;
    m.tokens_generated = 0;
    m.elapsed_ms = 0.0f;
    m.tokens_per_second = 0.0f;
    m.launch_reduction = 0;
    m.sync_reduction = 0;
    m.speedup_vs_baseline = 1.0f;

    int draft_offset_stride = draft_len * 4;
    int total_kv_entries = N * 4;
    int draft_buffer_size = N * draft_offset_stride;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, tokens_per_request, true, draft_offset_stride);

    // Set per-request adaptive block size from the configured draft_len
    int init_block_size = draft_len > 0 ? draft_len : 4;
    if (init_block_size > 8) init_block_size = 8;
    if (init_block_size < 1) init_block_size = 1;
    for (int i = 0; i < N; i++) {
        requests[i].current_block_size = init_block_size;
    }

    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    memset(kv_entries, 0, total_kv_entries * sizeof(KVPageEntry));
    for (int i = 0; i < total_kv_entries; i++) {
        kv_entries[i].page_id = i;
        kv_entries[i].request_id = (i / 4) + 1;
        kv_entries[i].layer_id = i % 2;
        kv_entries[i].start_token = (i % 4) * 4;
        kv_entries[i].token_count = 4;
        kv_entries[i].state = KV_PAGE_FREE;
        kv_entries[i].flags = 0;
        kv_entries[i].score = 0.0f;
        kv_entries[i].selected = 0;
        kv_entries[i].last_selected_step = -1;
        kv_entries[i].sparse_rank = -1;
    }

    KVPageEntry* d_kv_entries = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kv_entries, total_kv_entries * sizeof(KVPageEntry)));
    CUDA_CHECK(cudaMemcpy(d_kv_entries, kv_entries, total_kv_entries * sizeof(KVPageEntry), cudaMemcpyHostToDevice));

    KVPageTable host_kv_table;
    host_kv_table.entries = d_kv_entries;
    host_kv_table.num_entries = total_kv_entries;
    host_kv_table.page_size_tokens = 4;
    host_kv_table.bytes_per_page = 4096;

    int* d_draft_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_draft_tokens, draft_buffer_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_draft_tokens, 0, draft_buffer_size * sizeof(int)));

    int* d_shutdown = nullptr;
    CUDA_CHECK(cudaMalloc(&d_shutdown, sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_shutdown, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int* d_done_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_done_counter, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_done_counter, &zero, sizeof(int), cudaMemcpyHostToDevice));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    print_requests("Mega-kernel initial state", requests, N);

#ifdef XLPK_ENABLE_NVTX
    nvtxRangePushA("persistent_megakernel");
#endif

    CUDA_CHECK(cudaEventRecord(start));

    xl_persistent_megakernel<<<N, MEGAKERNEL_BLOCK_SIZE>>>(
        d_reqs, N, host_kv_table, d_draft_tokens, d_shutdown, d_done_counter,
        max_iterations, draft_len
    );
    CUDA_CHECK(cudaGetLastError());
    m.host_kernel_launches = 1;

    CUDA_CHECK(cudaDeviceSynchronize());
    m.host_synchronizations = 1;

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

#ifdef XLPK_ENABLE_NVTX
    nvtxRangePop();
#endif

    copy_back(requests, d_reqs, N);
    print_requests("After mega-kernel", requests, N);

    m.completed_requests = count_completed(requests, N);
    m.tokens_generated = total_tokens_generated(requests, N);
    m.tokens_per_second = (m.elapsed_ms > 0.0f) ? (m.tokens_generated / (m.elapsed_ms / 1000.0f)) : 0.0f;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_shutdown));
    CUDA_CHECK(cudaFree(d_done_counter));
    delete[] requests;
    delete[] kv_entries;
    return m;
}

static ResearchRunMetrics run_sparse_gather_bench(int N, int draft_len) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "sparse_gather";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        ints_per_page * static_cast<int>(sizeof(int)),
        &d_kv_entries
    );

    const int payload_count = total_kv_entries * ints_per_page;
    int* h_payloads = new int[payload_count];
    for (int i = 0; i < payload_count; ++i) {
        h_payloads[i] = i + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, payload_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads, payload_count * sizeof(int), cudaMemcpyHostToDevice));

    const int max_selected_pages = draft_len > 0 ? draft_len : 2;
    int* d_selected_page_ids = nullptr;
    int* d_compacted_payloads = nullptr;
    SparseKVGatherMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted_payloads, N * max_selected_pages * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_metrics, N * sizeof(SparseKVGatherMetrics)));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sparse_kv_gather_and_score_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        max_selected_pages,
        d_selected_page_ids,
        d_compacted_payloads,
        d_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    SparseKVGatherMetrics* h_metrics = new SparseKVGatherMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, N * sizeof(SparseKVGatherMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.blocks_examined += h_metrics[i].blocks_examined;
        m.blocks_selected += h_metrics[i].blocks_selected;
        m.bytes_read += h_metrics[i].bytes_read;
        m.bytes_saved += h_metrics[i].bytes_saved;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_selected_page_ids));
    CUDA_CHECK(cudaFree(d_compacted_payloads));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] h_metrics;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_scheduler_bench(int N, int draft_len) {
    ResearchRunMetrics m = {};
    m.mode = "resident_scheduler";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].priority = ((i * 17) + 3) % (N + 7);
        if ((i % 5) == 0) {
            requests[i].set_state(REQUEST_COMPLETE);
        }
    }

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);
    int* d_scheduled_ids = nullptr;
    int* d_scheduled_priorities = nullptr;
    SchedulerKernelMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scheduled_ids, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scheduled_priorities, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_metrics, sizeof(SchedulerKernelMetrics)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    resident_schedule_requests_kernel<<<1, 1>>>(
        d_reqs,
        N,
        d_scheduled_ids,
        d_scheduled_priorities,
        d_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    SchedulerKernelMetrics h_metric = {};
    CUDA_CHECK(cudaMemcpy(&h_metric, d_metrics, sizeof(SchedulerKernelMetrics), cudaMemcpyDeviceToHost));
    m.scheduler_requests_examined = h_metric.requests_examined;
    m.scheduler_requests_scheduled = h_metric.requests_scheduled;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_scheduled_ids));
    CUDA_CHECK(cudaFree(d_scheduled_priorities));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_prefetch_bench(int N, int draft_len) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "kv_prefetch";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    const int bytes_per_page = ints_per_page * static_cast<int>(sizeof(int));
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        bytes_per_page,
        &d_kv_entries
    );

    const int payload_count = total_kv_entries * ints_per_page;
    int* h_payloads = new int[payload_count];
    for (int i = 0; i < payload_count; ++i) {
        h_payloads[i] = i + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, payload_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads, payload_count * sizeof(int), cudaMemcpyHostToDevice));

    const int max_selected_pages = draft_len > 0 ? draft_len : 2;
    int* d_selected_page_ids = nullptr;
    int* d_compacted_payloads = nullptr;
    SparseKVGatherMetrics* d_gather_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted_payloads, N * max_selected_pages * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gather_metrics, N * sizeof(SparseKVGatherMetrics)));

    int* d_prefetched_page_ids = nullptr;
    int* d_buffer_slots = nullptr;
    KVPrefetchMetrics* d_prefetch_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prefetched_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_buffer_slots, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefetch_metrics, N * sizeof(KVPrefetchMetrics)));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sparse_kv_gather_and_score_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        max_selected_pages,
        d_selected_page_ids,
        d_compacted_payloads,
        d_gather_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    kv_prefetch_planner_kernel<<<N, 1>>>(
        d_selected_page_ids,
        N,
        max_selected_pages,
        bytes_per_page,
        d_prefetched_page_ids,
        d_buffer_slots,
        d_prefetch_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    KVPrefetchMetrics* h_metrics = new KVPrefetchMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_prefetch_metrics, N * sizeof(KVPrefetchMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.prefetch_pages += h_metrics[i].prefetch_pages;
        m.prefetch_bytes += h_metrics[i].prefetch_bytes;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_selected_page_ids));
    CUDA_CHECK(cudaFree(d_compacted_payloads));
    CUDA_CHECK(cudaFree(d_gather_metrics));
    CUDA_CHECK(cudaFree(d_prefetched_page_ids));
    CUDA_CHECK(cudaFree(d_buffer_slots));
    CUDA_CHECK(cudaFree(d_prefetch_metrics));
    delete[] h_metrics;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_verify_commit_bench(int N, int draft_len) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;

    ResearchRunMetrics m = {};
    m.mode = "verify_commit";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len);
    for (int i = 0; i < N; ++i) {
        requests[i].state = REQUEST_DRAFT_READY;
        requests[i].draft_len = draft_len;
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_DRAFT,
        KV_FLAG_RESIDENT | KV_FLAG_DRAFT | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        4096,
        &d_kv_entries
    );

    const int total_draft_tokens = N * draft_len;
    int* h_draft_tokens = new int[total_draft_tokens];
    for (int req = 0; req < N; ++req) {
        for (int j = 0; j < draft_len; ++j) {
            int token = (req + 1) * 10 + j + 1;
            if (j == draft_len - 1) {
                token = 4 * (req + 1);
            }
            h_draft_tokens[req * draft_len + j] = token;
        }
    }

    int* d_draft_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_draft_tokens, total_draft_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        d_draft_tokens,
        h_draft_tokens,
        total_draft_tokens * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    VerifyCommitMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metrics, N * sizeof(VerifyCommitMetrics)));
    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fused_verify_and_commit_kernel<<<N, 1>>>(d_reqs, N, kv_table, d_draft_tokens, d_metrics);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    VerifyCommitMetrics* h_metrics = new VerifyCommitMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, N * sizeof(VerifyCommitMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.speculative_candidates += h_metrics[i].speculative_candidates;
        m.accepted_tokens += h_metrics[i].accepted_tokens;
        m.rejected_tokens += h_metrics[i].rejected_tokens;
        m.committed_pages += h_metrics[i].committed_pages;
        m.released_pages += h_metrics[i].released_pages;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] h_metrics;
    delete[] h_draft_tokens;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_dma_movement_bench(int N, int draft_len) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "dma_movement";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    const int bytes_per_page = ints_per_page * static_cast<int>(sizeof(int));
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        bytes_per_page,
        &d_kv_entries
    );

    const int payload_count = total_kv_entries * ints_per_page;
    int* h_payloads = new int[payload_count];
    for (int i = 0; i < payload_count; ++i) {
        h_payloads[i] = i + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, payload_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads, payload_count * sizeof(int), cudaMemcpyHostToDevice));

    const int max_selected_pages = draft_len > 0 ? draft_len : 2;
    int* d_selected_page_ids = nullptr;
    int* d_compacted_payloads = nullptr;
    SparseKVGatherMetrics* d_gather_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted_payloads, N * max_selected_pages * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gather_metrics, N * sizeof(SparseKVGatherMetrics)));

    int* d_source_tiers = nullptr;
    int* d_dma_ops = nullptr;
    DMAMovementPlanMetrics* d_dma_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_source_tiers, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dma_ops, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dma_metrics, N * sizeof(DMAMovementPlanMetrics)));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sparse_kv_gather_and_score_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        max_selected_pages,
        d_selected_page_ids,
        d_compacted_payloads,
        d_gather_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    dma_aware_kv_movement_planner_kernel<<<N, 1>>>(
        d_selected_page_ids,
        N,
        max_selected_pages,
        bytes_per_page,
        d_source_tiers,
        d_dma_ops,
        d_dma_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    SparseKVGatherMetrics* h_gather_metrics = new SparseKVGatherMetrics[N];
    DMAMovementPlanMetrics* h_dma_metrics = new DMAMovementPlanMetrics[N];
    CUDA_CHECK(cudaMemcpy(
        h_gather_metrics,
        d_gather_metrics,
        N * sizeof(SparseKVGatherMetrics),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        h_dma_metrics,
        d_dma_metrics,
        N * sizeof(DMAMovementPlanMetrics),
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < N; ++i) {
        m.blocks_examined += h_gather_metrics[i].blocks_examined;
        m.blocks_selected += h_gather_metrics[i].blocks_selected;
        m.bytes_read += h_gather_metrics[i].bytes_read;
        m.bytes_saved += h_gather_metrics[i].bytes_saved;
        m.dma_pages_planned += h_dma_metrics[i].pages_planned;
        m.dma_hbm_hits += h_dma_metrics[i].hbm_hits;
        m.dma_dram_fetches += h_dma_metrics[i].dram_fetches;
        m.dma_ssd_fetches += h_dma_metrics[i].ssd_fetches;
        m.dma_ops += h_dma_metrics[i].dma_ops;
        m.dma_bytes_moved += h_dma_metrics[i].bytes_moved;
        m.dma_bytes_from_hbm += h_dma_metrics[i].bytes_from_hbm;
        m.dma_bytes_from_dram += h_dma_metrics[i].bytes_from_dram;
        m.dma_bytes_from_ssd += h_dma_metrics[i].bytes_from_ssd;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_selected_page_ids));
    CUDA_CHECK(cudaFree(d_compacted_payloads));
    CUDA_CHECK(cudaFree(d_gather_metrics));
    CUDA_CHECK(cudaFree(d_source_tiers));
    CUDA_CHECK(cudaFree(d_dma_ops));
    CUDA_CHECK(cudaFree(d_dma_metrics));
    delete[] h_gather_metrics;
    delete[] h_dma_metrics;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_tiered_staging_bench(int N, int draft_len) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "tiered_kv_staging";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    const int bytes_per_page = ints_per_page * static_cast<int>(sizeof(int));
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        bytes_per_page,
        &d_kv_entries
    );

    const int payload_count = total_kv_entries * ints_per_page;
    int* h_payloads = new int[payload_count];
    for (int i = 0; i < payload_count; ++i) {
        h_payloads[i] = i + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, payload_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads, payload_count * sizeof(int), cudaMemcpyHostToDevice));

    const int max_selected_pages = draft_len > 0 ? draft_len : 2;
    int* d_selected_page_ids = nullptr;
    int* d_compacted_payloads = nullptr;
    SparseKVGatherMetrics* d_gather_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted_payloads, N * max_selected_pages * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gather_metrics, N * sizeof(SparseKVGatherMetrics)));

    int* d_source_tiers = nullptr;
    int* d_dma_ops = nullptr;
    DMAMovementPlanMetrics* d_dma_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_source_tiers, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dma_ops, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dma_metrics, N * sizeof(DMAMovementPlanMetrics)));

    int* d_staged_page_ids = nullptr;
    int* d_staged_source_tiers = nullptr;
    int* d_staged_buffer_slots = nullptr;
    TieredKVStagingMetrics* d_staging_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_staged_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_staged_source_tiers, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_staged_buffer_slots, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_staging_metrics, N * sizeof(TieredKVStagingMetrics)));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sparse_kv_gather_and_score_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        max_selected_pages,
        d_selected_page_ids,
        d_compacted_payloads,
        d_gather_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    dma_aware_kv_movement_planner_kernel<<<N, 1>>>(
        d_selected_page_ids,
        N,
        max_selected_pages,
        bytes_per_page,
        d_source_tiers,
        d_dma_ops,
        d_dma_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    tiered_kv_staging_kernel<<<N, 1>>>(
        d_selected_page_ids,
        d_source_tiers,
        N,
        max_selected_pages,
        bytes_per_page,
        d_staged_page_ids,
        d_staged_source_tiers,
        d_staged_buffer_slots,
        d_staging_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    DMAMovementPlanMetrics* h_dma_metrics = new DMAMovementPlanMetrics[N];
    TieredKVStagingMetrics* h_staging_metrics = new TieredKVStagingMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_dma_metrics, d_dma_metrics, N * sizeof(DMAMovementPlanMetrics), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_staging_metrics, d_staging_metrics, N * sizeof(TieredKVStagingMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.dma_pages_planned += h_dma_metrics[i].pages_planned;
        m.dma_hbm_hits += h_dma_metrics[i].hbm_hits;
        m.dma_dram_fetches += h_dma_metrics[i].dram_fetches;
        m.dma_ssd_fetches += h_dma_metrics[i].ssd_fetches;
        m.dma_ops += h_dma_metrics[i].dma_ops;
        m.dma_bytes_moved += h_dma_metrics[i].bytes_moved;
        m.dma_bytes_from_hbm += h_dma_metrics[i].bytes_from_hbm;
        m.dma_bytes_from_dram += h_dma_metrics[i].bytes_from_dram;
        m.dma_bytes_from_ssd += h_dma_metrics[i].bytes_from_ssd;
        m.staged_pages += h_staging_metrics[i].staged_pages;
        m.staged_hbm_pages += h_staging_metrics[i].hbm_pages_staged;
        m.staged_dma_pages += h_staging_metrics[i].dma_pages_staged;
        m.buffer_slot_switches += h_staging_metrics[i].buffer_slot_switches;
        m.staging_bytes += h_staging_metrics[i].staging_bytes;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_selected_page_ids));
    CUDA_CHECK(cudaFree(d_compacted_payloads));
    CUDA_CHECK(cudaFree(d_gather_metrics));
    CUDA_CHECK(cudaFree(d_source_tiers));
    CUDA_CHECK(cudaFree(d_dma_ops));
    CUDA_CHECK(cudaFree(d_dma_metrics));
    CUDA_CHECK(cudaFree(d_staged_page_ids));
    CUDA_CHECK(cudaFree(d_staged_source_tiers));
    CUDA_CHECK(cudaFree(d_staged_buffer_slots));
    CUDA_CHECK(cudaFree(d_staging_metrics));
    delete[] h_dma_metrics;
    delete[] h_staging_metrics;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_kv_pressure_bench(int N, int draft_len) {
    constexpr int pages_per_request = 6;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "kv_pressure";
    m.requests = N;
    m.draft_len = draft_len;

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED
    );

    for (int req = 0; req < N; ++req) {
        const int base = req * pages_per_request;
        // Two pinned hot pages survive pressure.
        kv_entries[base + 0].set_state(KV_PAGE_PINNED);
        kv_entries[base + 0].set_flag(KV_FLAG_PINNED);
        kv_entries[base + 1].set_state(KV_PAGE_PINNED);
        kv_entries[base + 1].set_flag(KV_FLAG_PINNED);

        // Two draft pages should be evicted first.
        kv_entries[base + 2].set_state(KV_PAGE_DRAFT);
        kv_entries[base + 2].set_flag(KV_FLAG_DRAFT);
        kv_entries[base + 2].clear_flag(KV_FLAG_COMMITTED);
        kv_entries[base + 3].set_state(KV_PAGE_DRAFT);
        kv_entries[base + 3].set_flag(KV_FLAG_DRAFT);
        kv_entries[base + 3].clear_flag(KV_FLAG_COMMITTED);

        // One selected page should be skipped.
        kv_entries[base + 4].selected = 1;
        kv_entries[base + 4].sparse_rank = 0;
        kv_entries[base + 4].set_flag(KV_FLAG_SELECTED);

        // Final committed page is a fallback eviction target.
        kv_entries[base + 5].set_state(KV_PAGE_COMMITTED);
        kv_entries[base + 5].set_flag(KV_FLAG_COMMITTED);
    }

    KVPageEntry* d_kv_entries = nullptr;
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        ints_per_page * static_cast<int>(sizeof(int)),
        &d_kv_entries
    );

    KVEvictionMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metrics, N * sizeof(KVEvictionMetrics)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kv_pressure_eviction_kernel<<<N, 1>>>(
        kv_table,
        N,
        pages_per_request,
        3,
        d_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    KVEvictionMetrics* h_metrics = new KVEvictionMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, N * sizeof(KVEvictionMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.eviction_pages_scanned += h_metrics[i].pages_scanned;
        m.eviction_pages_evicted += h_metrics[i].pages_evicted;
        m.eviction_draft_pages += h_metrics[i].draft_pages_evicted;
        m.eviction_committed_pages += h_metrics[i].committed_pages_evicted;
        m.eviction_pinned_skipped += h_metrics[i].pinned_pages_skipped;
        m.eviction_selected_skipped += h_metrics[i].selected_pages_skipped;
        m.eviction_reclaimed_bytes += h_metrics[i].reclaimed_bytes;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] h_metrics;
    delete[] kv_entries;
    return m;
}

static ResearchRunMetrics run_tier_residency_bench(int N, int draft_len) {
    constexpr int pages_per_request = 6;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "kv_tier_residency";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
        requests[i].kv_num_pages = pages_per_request;
        requests[i].kv_table_offset = i * pages_per_request;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    const int bytes_per_page = ints_per_page * static_cast<int>(sizeof(int));
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        bytes_per_page,
        &d_kv_entries
    );

    const int payload_count = total_kv_entries * ints_per_page;
    int* h_payloads = new int[payload_count];
    for (int i = 0; i < payload_count; ++i) {
        h_payloads[i] = i + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, payload_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_payloads, h_payloads, payload_count * sizeof(int), cudaMemcpyHostToDevice));

    const int max_selected_pages = draft_len > 0 ? draft_len : 2;
    int* d_selected_page_ids = nullptr;
    int* d_compacted_payloads = nullptr;
    SparseKVGatherMetrics* d_gather_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_page_ids, N * max_selected_pages * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_compacted_payloads, N * max_selected_pages * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gather_metrics, N * sizeof(SparseKVGatherMetrics)));

    int* h_page_tiers = new int[total_kv_entries];
    for (int req = 0; req < N; ++req) {
        const int base = req * pages_per_request;
        h_page_tiers[base + 0] = 0;
        h_page_tiers[base + 1] = 0;
        h_page_tiers[base + 2] = 1;
        h_page_tiers[base + 3] = 1;
        h_page_tiers[base + 4] = 2;
        h_page_tiers[base + 5] = 2;
    }

    int* d_page_tiers = nullptr;
    CUDA_CHECK(cudaMalloc(&d_page_tiers, total_kv_entries * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_page_tiers, h_page_tiers, total_kv_entries * sizeof(int), cudaMemcpyHostToDevice));

    KVTierResidencyMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metrics, N * sizeof(KVTierResidencyMetrics)));

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    sparse_kv_gather_and_score_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        max_selected_pages,
        d_selected_page_ids,
        d_compacted_payloads,
        d_gather_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    kv_tier_residency_kernel<<<N, 1>>>(
        d_selected_page_ids,
        N,
        max_selected_pages,
        pages_per_request,
        2,
        2,
        bytes_per_page,
        d_page_tiers,
        d_metrics
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    KVTierResidencyMetrics* h_metrics = new KVTierResidencyMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, N * sizeof(KVTierResidencyMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.tier_pages_rebalanced += h_metrics[i].pages_rebalanced;
        m.tier_promotions_to_hbm += h_metrics[i].promotions_to_hbm;
        m.tier_promotions_to_dram += h_metrics[i].promotions_to_dram;
        m.tier_demotions_to_dram += h_metrics[i].demotions_to_dram;
        m.tier_demotions_to_ssd += h_metrics[i].demotions_to_ssd;
        m.tier_final_hbm_pages += h_metrics[i].final_hbm_pages;
        m.tier_final_dram_pages += h_metrics[i].final_dram_pages;
        m.tier_final_ssd_pages += h_metrics[i].final_ssd_pages;
        m.tier_bytes_promoted += h_metrics[i].bytes_promoted;
        m.tier_bytes_demoted += h_metrics[i].bytes_demoted;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_selected_page_ids));
    CUDA_CHECK(cudaFree(d_compacted_payloads));
    CUDA_CHECK(cudaFree(d_gather_metrics));
    CUDA_CHECK(cudaFree(d_page_tiers));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] h_metrics;
    delete[] h_page_tiers;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static ResearchRunMetrics run_research_pipeline_bench(int N, int draft_len, int max_iterations) {
    constexpr int pages_per_request = 4;
    constexpr int page_size_tokens = 4;
    constexpr int ints_per_page = 16;

    ResearchRunMetrics m = {};
    m.mode = "research_pipeline";
    m.requests = N;
    m.draft_len = draft_len;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, 8, true, draft_len * 4);
    for (int i = 0; i < N; ++i) {
        requests[i].current_block_size = draft_len > 0 ? draft_len : 2;
    }

    const int total_kv_entries = N * pages_per_request;
    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    init_kv_entries(
        kv_entries,
        N,
        pages_per_request,
        page_size_tokens,
        KV_PAGE_COMMITTED,
        KV_FLAG_RESIDENT | KV_FLAG_COMMITTED | KV_FLAG_PINNED
    );

    KVPageEntry* d_kv_entries = nullptr;
    KVPageTable kv_table = make_device_kv_table(
        kv_entries,
        total_kv_entries,
        page_size_tokens,
        ints_per_page * static_cast<int>(sizeof(int)),
        &d_kv_entries
    );

    int* h_payloads = new int[total_kv_entries * ints_per_page];
    for (int i = 0; i < total_kv_entries * ints_per_page; ++i) {
        h_payloads[i] = (i % ints_per_page) + 1;
    }

    int* d_payloads = nullptr;
    CUDA_CHECK(cudaMalloc(&d_payloads, total_kv_entries * ints_per_page * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(
        d_payloads,
        h_payloads,
        total_kv_entries * ints_per_page * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    int* d_draft_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_draft_tokens, N * draft_len * 4 * sizeof(int)));
    ResidentPipelineMetrics* d_metrics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_metrics, N * sizeof(ResidentPipelineMetrics)));
    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    resident_sparse_decode_pipeline_kernel<<<N, 1>>>(
        d_reqs,
        N,
        kv_table,
        d_payloads,
        ints_per_page,
        draft_len > 0 ? draft_len : 2,
        d_draft_tokens,
        d_metrics,
        max_iterations
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&m.elapsed_ms, start, stop));

    ResidentPipelineMetrics* h_metrics = new ResidentPipelineMetrics[N];
    CUDA_CHECK(cudaMemcpy(h_metrics, d_metrics, N * sizeof(ResidentPipelineMetrics), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        m.loop_iterations += h_metrics[i].loop_iterations;
        m.blocks_selected += h_metrics[i].blocks_selected;
        m.bytes_read += h_metrics[i].bytes_read;
        m.accepted_tokens += h_metrics[i].accepted_tokens;
        m.committed_pages += h_metrics[i].committed_pages;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_reqs));
    CUDA_CHECK(cudaFree(d_kv_entries));
    CUDA_CHECK(cudaFree(d_payloads));
    CUDA_CHECK(cudaFree(d_draft_tokens));
    CUDA_CHECK(cudaFree(d_metrics));
    delete[] h_metrics;
    delete[] h_payloads;
    delete[] kv_entries;
    delete[] requests;
    return m;
}

static void compute_reduction(RunMetrics& mega, const RunMetrics& baseline) {
    mega.launch_reduction = baseline.host_kernel_launches;
    mega.sync_reduction = baseline.host_synchronizations;
    mega.speedup_vs_baseline = (mega.elapsed_ms > 0.0f)
        ? (baseline.elapsed_ms / mega.elapsed_ms) : 1.0f;
}

static void print_metrics(const RunMetrics& m) {
    printf("  host_kernel_launches: %d\n", m.host_kernel_launches);
    printf("  host_synchronizations: %d\n", m.host_synchronizations);
    printf("  completed_requests: %d / %d\n", m.completed_requests, m.target_requests);
    printf("  elapsed_ms: %.3f\n", m.elapsed_ms);
    printf("  tokens_generated: %d\n", m.tokens_generated);
    printf("  tokens_per_second: %.0f\n", m.tokens_per_second);
    if (m.launch_reduction > 0) {
        printf("  launch_reduction: %d:1\n", m.launch_reduction);
        printf("  sync_reduction: %d:1\n", m.sync_reduction);
        printf("  speedup_vs_baseline: %.2fx\n", m.speedup_vs_baseline);
    }
}

static void print_research_metrics(const ResearchRunMetrics& m) {
    printf("  elapsed_ms: %.3f\n", m.elapsed_ms);
    if (m.blocks_examined > 0 || m.blocks_selected > 0) {
        printf("  blocks_examined: %d\n", m.blocks_examined);
        printf("  blocks_selected: %d\n", m.blocks_selected);
        printf("  bytes_read: %d\n", m.bytes_read);
        printf("  bytes_saved: %d\n", m.bytes_saved);
    }
    if (m.speculative_candidates > 0 || m.accepted_tokens > 0 || m.rejected_tokens > 0) {
        printf("  speculative_candidates: %d\n", m.speculative_candidates);
        printf("  accepted_tokens: %d\n", m.accepted_tokens);
        printf("  rejected_tokens: %d\n", m.rejected_tokens);
        printf("  committed_pages: %d\n", m.committed_pages);
        printf("  released_pages: %d\n", m.released_pages);
    }
    if (m.scheduler_requests_examined > 0 || m.scheduler_requests_scheduled > 0) {
        printf("  scheduler_requests_examined: %d\n", m.scheduler_requests_examined);
        printf("  scheduler_requests_scheduled: %d\n", m.scheduler_requests_scheduled);
    }
    if (m.prefetch_pages > 0 || m.prefetch_bytes > 0) {
        printf("  prefetch_pages: %d\n", m.prefetch_pages);
        printf("  prefetch_bytes: %d\n", m.prefetch_bytes);
    }
    if (m.dma_pages_planned > 0 || m.dma_ops > 0 || m.dma_bytes_moved > 0) {
        printf("  dma_pages_planned: %d\n", m.dma_pages_planned);
        printf("  dma_hbm_hits: %d\n", m.dma_hbm_hits);
        printf("  dma_dram_fetches: %d\n", m.dma_dram_fetches);
        printf("  dma_ssd_fetches: %d\n", m.dma_ssd_fetches);
        printf("  dma_ops: %d\n", m.dma_ops);
        printf("  dma_bytes_moved: %d\n", m.dma_bytes_moved);
        printf("  dma_bytes_from_hbm: %d\n", m.dma_bytes_from_hbm);
        printf("  dma_bytes_from_dram: %d\n", m.dma_bytes_from_dram);
        printf("  dma_bytes_from_ssd: %d\n", m.dma_bytes_from_ssd);
    }
    if (m.staged_pages > 0 || m.staging_bytes > 0) {
        printf("  staged_pages: %d\n", m.staged_pages);
        printf("  staged_hbm_pages: %d\n", m.staged_hbm_pages);
        printf("  staged_dma_pages: %d\n", m.staged_dma_pages);
        printf("  buffer_slot_switches: %d\n", m.buffer_slot_switches);
        printf("  staging_bytes: %d\n", m.staging_bytes);
    }
    if (m.eviction_pages_scanned > 0 || m.eviction_pages_evicted > 0 || m.eviction_reclaimed_bytes > 0) {
        printf("  eviction_pages_scanned: %d\n", m.eviction_pages_scanned);
        printf("  eviction_pages_evicted: %d\n", m.eviction_pages_evicted);
        printf("  eviction_draft_pages: %d\n", m.eviction_draft_pages);
        printf("  eviction_committed_pages: %d\n", m.eviction_committed_pages);
        printf("  eviction_pinned_skipped: %d\n", m.eviction_pinned_skipped);
        printf("  eviction_selected_skipped: %d\n", m.eviction_selected_skipped);
        printf("  eviction_reclaimed_bytes: %d\n", m.eviction_reclaimed_bytes);
    }
    if (m.tier_pages_rebalanced > 0 || m.tier_bytes_promoted > 0 || m.tier_bytes_demoted > 0) {
        printf("  tier_pages_rebalanced: %d\n", m.tier_pages_rebalanced);
        printf("  tier_promotions_to_hbm: %d\n", m.tier_promotions_to_hbm);
        printf("  tier_promotions_to_dram: %d\n", m.tier_promotions_to_dram);
        printf("  tier_demotions_to_dram: %d\n", m.tier_demotions_to_dram);
        printf("  tier_demotions_to_ssd: %d\n", m.tier_demotions_to_ssd);
        printf("  tier_final_hbm_pages: %d\n", m.tier_final_hbm_pages);
        printf("  tier_final_dram_pages: %d\n", m.tier_final_dram_pages);
        printf("  tier_final_ssd_pages: %d\n", m.tier_final_ssd_pages);
        printf("  tier_bytes_promoted: %d\n", m.tier_bytes_promoted);
        printf("  tier_bytes_demoted: %d\n", m.tier_bytes_demoted);
    }
    if (m.generated_tokens > 0) {
        printf("  generated_tokens: %d\n", m.generated_tokens);
    }
    if (m.loop_iterations > 0) {
        printf("  loop_iterations: %d\n", m.loop_iterations);
    }
}

static void write_csv_header(FILE* f) {
    fprintf(f, "mode,requests,tokens_per_request,draft_len,"
               "host_kernel_launches,host_synchronizations,"
               "completed_requests,target_requests,"
               "tokens_generated,elapsed_ms,tokens_per_second,"
               "launch_reduction,sync_reduction,speedup_vs_baseline\n");
}

static void write_csv(FILE* f, const RunMetrics& m) {
    fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.0f,%d,%d,%.2f\n",
            m.mode,
            m.requests,
            m.tokens_per_request,
            m.draft_len,
            m.host_kernel_launches,
            m.host_synchronizations,
            m.completed_requests,
            m.target_requests,
            m.tokens_generated,
            m.elapsed_ms,
            m.tokens_per_second,
            m.launch_reduction,
            m.sync_reduction,
            m.speedup_vs_baseline);
}

static void run_sweep_config(
    int requests, int tokens, int draft_len, int iterations,
    FILE* csv_file
) {
    RunMetrics baseline = run_baseline_path(requests, tokens, draft_len, iterations);
    write_csv(csv_file, baseline);

    RunMetrics mega = run_megakernel_path(requests, tokens, draft_len, iterations);
    compute_reduction(mega, baseline);
    write_csv(csv_file, mega);
}

static void run_sweep(const char* csv_path, int iterations,
                       int cli_requests, int cli_tokens, int cli_draft_len)
{
    printf("\n========================================\n");
    printf("Sweep Mode: Cartesian product over configs\n");
    printf("========================================\n");

    // Default sweep arrays
    int request_vals[] = {2, 4, 8, 16};
    int token_vals[] = {32, 64, 128};
    int draft_vals[] = {1, 4, 8};

    int n_default_requests = sizeof(request_vals) / sizeof(request_vals[0]);
    int n_default_tokens = sizeof(token_vals) / sizeof(token_vals[0]);
    int n_default_drafts = sizeof(draft_vals) / sizeof(draft_vals[0]);

    // If user overrode a CLI default, scale the sweep to match
    int max_req = (cli_requests != 8) ? cli_requests : request_vals[n_default_requests - 1];
    int max_tok = (cli_tokens != 128) ? cli_tokens : token_vals[n_default_tokens - 1];
    int max_drf = (cli_draft_len != 4) ? cli_draft_len : draft_vals[n_default_drafts - 1];

    // Build request array
    int req_arr[16];
    int n_req = 0;
    for (int v = 2; v <= max_req; v *= 2) req_arr[n_req++] = v;

    int tok_arr[16];
    int n_tok = 0;
    for (int v = 32; v <= max_tok; v *= 2) tok_arr[n_tok++] = v;

    int drf_arr[16];
    int n_drf = 0;
    for (int v = 1; v <= max_drf; v *= 2) drf_arr[n_drf++] = v;

    int total_runs = n_req * n_tok * n_drf;

    printf("Requests:        [");
    for (int i = 0; i < n_req; i++) printf("%s%d", (i ? ", " : ""), req_arr[i]);
    printf("]\nTokens/req:      [");
    for (int i = 0; i < n_tok; i++) printf("%s%d", (i ? ", " : ""), tok_arr[i]);
    printf("]\nDraft len:       [");
    for (int i = 0; i < n_drf; i++) printf("%s%d", (i ? ", " : ""), drf_arr[i]);
    printf("]\nTotal runs:      %d (%d rows in CSV)\n\n", total_runs, total_runs * 2);

    FILE* csv_file = nullptr;
    if (csv_path) {
        csv_file = fopen(csv_path, "w");
        if (!csv_file) {
            fprintf(stderr, "Error: cannot open %s for writing\n", csv_path);
            return;
        }
        write_csv_header(csv_file);
    }

    int run_count = 0;
    for (int ri = 0; ri < n_req; ri++) {
        for (int ti = 0; ti < n_tok; ti++) {
            for (int di = 0; di < n_drf; di++) {
                run_count++;
                printf("[%d/%d] requests=%d tokens=%d draft_len=%d\n",
                       run_count, total_runs, req_arr[ri], tok_arr[ti], drf_arr[di]);

                run_sweep_config(req_arr[ri], tok_arr[ti], drf_arr[di],
                                 iterations, csv_file);
            }
        }
    }

    if (csv_file) {
        fclose(csv_file);
        printf("\nCSV written to %s (%d rows)\n", csv_path, total_runs * 2);
    }

    printf("\nSweep complete.\n");
}

int main(int argc, char** argv) {
    int requests = 8;
    int tokens = 128;
    int draft_len = 4;
    int iterations = 100000;
    const char* mode = "both";
    const char* csv_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--requests") == 0 && i + 1 < argc) {
            requests = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--draft-len") == 0 && i + 1 < argc) {
            draft_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s [--mode baseline|mega|both|sweep|resident-scheduler|sparse-gather|kv-prefetch|verify-commit|dma-movement|tiered-kv-staging|kv-pressure|tier-residency|research-pipeline|research-all] [--requests N] [--tokens N] [--draft-len N] [--iterations N] [--csv path]\n", argv[0]);
            return 1;
        }
    }

    if (strcmp(mode, "sweep") == 0) {
        run_sweep(csv_path, iterations, requests, tokens, draft_len);
        return 0;
    }

    printf("========================================\n");
    printf("XL-Persistent-Kernel CUDA Measurement\n");
    printf("========================================\n\n");
    printf("Configuration:\n");
    printf("  requests: %d\n", requests);
    printf("  tokens_per_request: %d\n", tokens);
    printf("  draft_len: %d\n", draft_len);
    printf("  mode: %s\n\n", mode);

    bool do_baseline = (strcmp(mode, "baseline") == 0 || strcmp(mode, "both") == 0);
    bool do_mega = (strcmp(mode, "mega") == 0 || strcmp(mode, "both") == 0);
    bool do_scheduler = (strcmp(mode, "resident-scheduler") == 0 || strcmp(mode, "research-all") == 0);
    bool do_sparse_gather = (strcmp(mode, "sparse-gather") == 0 || strcmp(mode, "research-all") == 0);
    bool do_prefetch = (strcmp(mode, "kv-prefetch") == 0 || strcmp(mode, "research-all") == 0);
    bool do_verify_commit = (strcmp(mode, "verify-commit") == 0 || strcmp(mode, "research-all") == 0);
    bool do_dma_movement = (strcmp(mode, "dma-movement") == 0 || strcmp(mode, "research-all") == 0);
    bool do_tiered_staging = (strcmp(mode, "tiered-kv-staging") == 0 || strcmp(mode, "research-all") == 0);
    bool do_kv_pressure = (strcmp(mode, "kv-pressure") == 0 || strcmp(mode, "research-all") == 0);
    bool do_tier_residency = (strcmp(mode, "tier-residency") == 0 || strcmp(mode, "research-all") == 0);
    bool do_research_pipeline = (strcmp(mode, "research-pipeline") == 0 || strcmp(mode, "research-all") == 0);

    FILE* csv_file = nullptr;
    if (csv_path != nullptr && !do_scheduler && !do_sparse_gather && !do_prefetch && !do_verify_commit && !do_dma_movement && !do_tiered_staging && !do_kv_pressure && !do_tier_residency && !do_research_pipeline) {
        csv_file = fopen(csv_path, "w");
        if (csv_file == nullptr) {
            fprintf(stderr, "Error: cannot open %s for writing\n", csv_path);
            return 1;
        }
        write_csv_header(csv_file);
    }

    RunMetrics baseline_metrics = {};
    RunMetrics mega_metrics = {};
    ResearchRunMetrics scheduler_metrics = {};
    ResearchRunMetrics sparse_gather_metrics = {};
    ResearchRunMetrics prefetch_metrics = {};
    ResearchRunMetrics verify_commit_metrics = {};
    ResearchRunMetrics dma_movement_metrics = {};
    ResearchRunMetrics tiered_staging_metrics = {};
    ResearchRunMetrics kv_pressure_metrics = {};
    ResearchRunMetrics tier_residency_metrics = {};
    ResearchRunMetrics research_pipeline_metrics = {};

    if (do_baseline) {
        printf("\n========================================\n");
        printf("Baseline host-launched decode:\n");
        printf("  (CPU controls token loop, repeated launches)\n");
        printf("========================================\n\n");

        baseline_metrics = run_baseline_path(requests, tokens, draft_len, iterations);

        printf("\nBaseline host-launched decode:\n");
        print_metrics(baseline_metrics);
        if (csv_file) write_csv(csv_file, baseline_metrics);
    }

    if (do_mega) {
        printf("\n========================================\n");
        printf("Persistent mega-kernel:\n");
        printf("  (one launch, device-resident loop)\n");
        printf("========================================\n\n");

        mega_metrics = run_megakernel_path(requests, tokens, draft_len, iterations);
        if (do_baseline) {
            compute_reduction(mega_metrics, baseline_metrics);
        }

        printf("\nPersistent mega-kernel:\n");
        print_metrics(mega_metrics);
        if (csv_file) write_csv(csv_file, mega_metrics);
    }

    if (do_scheduler) {
        printf("\n========================================\n");
        printf("Resident scheduler kernel:\n");
        printf("  (GPU-side priority ordering of live requests)\n");
        printf("========================================\n\n");

        scheduler_metrics = run_scheduler_bench(requests, draft_len);
        print_research_metrics(scheduler_metrics);
    }

    if (do_sparse_gather) {
        printf("\n========================================\n");
        printf("Sparse KV gather + score kernel:\n");
        printf("  (page scoring, top-k selection, compact gather)\n");
        printf("========================================\n\n");

        sparse_gather_metrics = run_sparse_gather_bench(requests, draft_len);
        print_research_metrics(sparse_gather_metrics);
    }

    if (do_prefetch) {
        printf("\n========================================\n");
        printf("KV prefetch planner kernel:\n");
        printf("  (selected sparse pages mapped to double-buffer slots)\n");
        printf("========================================\n\n");

        prefetch_metrics = run_prefetch_bench(requests, draft_len);
        print_research_metrics(prefetch_metrics);
    }

    if (do_verify_commit) {
        printf("\n========================================\n");
        printf("Fused verify + commit kernel:\n");
        printf("  (speculative verify, commit, release)\n");
        printf("========================================\n\n");

        verify_commit_metrics = run_verify_commit_bench(requests, draft_len);
        print_research_metrics(verify_commit_metrics);
    }

    if (do_dma_movement) {
        printf("\n========================================\n");
        printf("DMA-aware KV movement planner kernel:\n");
        printf("  (sparse-selected pages mapped to tiered movement ops)\n");
        printf("========================================\n\n");

        dma_movement_metrics = run_dma_movement_bench(requests, draft_len);
        print_research_metrics(dma_movement_metrics);
    }

    if (do_tiered_staging) {
        printf("\n========================================\n");
        printf("Tiered KV staging kernel:\n");
        printf("  (tier-prioritized staging order for selected KV pages)\n");
        printf("========================================\n\n");

        tiered_staging_metrics = run_tiered_staging_bench(requests, draft_len);
        print_research_metrics(tiered_staging_metrics);
    }

    if (do_kv_pressure) {
        printf("\n========================================\n");
        printf("KV pressure eviction kernel:\n");
        printf("  (draft-first eviction under resident KV pressure)\n");
        printf("========================================\n\n");

        kv_pressure_metrics = run_kv_pressure_bench(requests, draft_len);
        print_research_metrics(kv_pressure_metrics);
    }

    if (do_tier_residency) {
        printf("\n========================================\n");
        printf("KV tier residency kernel:\n");
        printf("  (promote selected pages and demote cold pages under tier budgets)\n");
        printf("========================================\n\n");

        tier_residency_metrics = run_tier_residency_bench(requests, draft_len);
        print_research_metrics(tier_residency_metrics);
    }

    if (do_research_pipeline) {
        printf("\n========================================\n");
        printf("Resident sparse decode pipeline kernel:\n");
        printf("  (select, gather, decode, verify, commit)\n");
        printf("========================================\n\n");

        research_pipeline_metrics = run_research_pipeline_bench(requests, draft_len, iterations);
        print_research_metrics(research_pipeline_metrics);
    }

    if (csv_file) {
        fclose(csv_file);
        printf("\nCSV written to %s\n", csv_path);
    }

    if (do_baseline && do_mega && baseline_metrics.elapsed_ms > 0.0f) {
        printf("\n========================================\n");
        printf("Relative:\n");
        printf("========================================\n\n");

        int launch_reduction_num = baseline_metrics.host_kernel_launches;
        int sync_reduction_num = baseline_metrics.host_synchronizations;
        int launch_reduction_den = (mega_metrics.host_kernel_launches > 0) ? mega_metrics.host_kernel_launches : 1;
        int sync_reduction_den = (mega_metrics.host_synchronizations > 0) ? mega_metrics.host_synchronizations : 1;

        printf("  launch_reduction: %d:%d\n", launch_reduction_num, launch_reduction_den);
        printf("  sync_reduction: %d:%d\n", sync_reduction_num, sync_reduction_den);

        float speedup = baseline_metrics.elapsed_ms / mega_metrics.elapsed_ms;
        printf("  speedup_vs_baseline: %.2fx\n", speedup);
    }

    if (do_baseline && do_mega) {
        printf("\n=== Validation ===\n");
        bool baseline_ok = (baseline_metrics.completed_requests == baseline_metrics.target_requests);
        bool mega_ok = (mega_metrics.completed_requests == mega_metrics.target_requests);
        bool baseline_launches_gt1 = (baseline_metrics.host_kernel_launches > 1);
        bool mega_launches_eq1 = (mega_metrics.host_kernel_launches == 1);
        bool baseline_syncs_gt1 = (baseline_metrics.host_synchronizations > 1);
        bool mega_syncs_eq1 = (mega_metrics.host_synchronizations == 1);

        printf("  baseline completed all requests: %s\n", baseline_ok ? "PASS" : "FAIL");
        printf("  mega-kernel completed all requests: %s\n", mega_ok ? "PASS" : "FAIL");
        printf("  baseline launches > 1: %s (%d)\n", baseline_launches_gt1 ? "PASS" : "FAIL", baseline_metrics.host_kernel_launches);
        printf("  mega-kernel launches == 1: %s (%d)\n", mega_launches_eq1 ? "PASS" : "FAIL", mega_metrics.host_kernel_launches);
        printf("  baseline syncs > 1: %s (%d)\n", baseline_syncs_gt1 ? "PASS" : "FAIL", baseline_metrics.host_synchronizations);
        printf("  mega-kernel syncs == 1: %s (%d)\n", mega_syncs_eq1 ? "PASS" : "FAIL", mega_metrics.host_synchronizations);
    }

    printf("\n=== Measurement complete ===\n");
    return 0;
}
