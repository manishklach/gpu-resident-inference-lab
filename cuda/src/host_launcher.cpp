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
 *
 * All math is fake/deterministic. No real transformer operations.
 * The measurement target is launch/sync reduction, not model FLOPs.
 *
 * CLI usage:
 *   ./xlpk_cuda_smoke --mode both --requests 8 --tokens 128 --draft-len 4
 *
 * CSV output:
 *   --csv path.csv  (writes mode,requests,tokens_per_request,...,elapsed_ms,tokens_per_second)
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
    int* shutdown_flag, int max_iterations, int block_size
);

extern __global__ void baseline_host_decode_step_kernel(
    RequestDescriptor* requests, int num_requests
);

constexpr int BLOCK_SIZE = 256;
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
    }
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

    int draft_offset_stride = draft_len * 4;
    int total_kv_entries = N * 4;
    int draft_buffer_size = N * draft_offset_stride;

    RequestDescriptor* requests = new RequestDescriptor[N];
    init_requests(requests, N, tokens_per_request, true, draft_offset_stride);

    KVPageEntry* kv_entries = new KVPageEntry[total_kv_entries];
    memset(kv_entries, 0, total_kv_entries * sizeof(KVPageEntry));
    for (int i = 0; i < total_kv_entries; i++) {
        kv_entries[i].page_id = i;
        kv_entries[i].state = KV_PAGE_FREE;
        kv_entries[i].flags = 0;
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

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    print_requests("Mega-kernel initial state", requests, N);

#ifdef XLPK_ENABLE_NVTX
    nvtxRangePushA("persistent_megakernel");
#endif

    CUDA_CHECK(cudaEventRecord(start));

    xl_persistent_megakernel<<<N, BLOCK_SIZE>>>(
        d_reqs, N, host_kv_table, d_draft_tokens, d_shutdown,
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
    delete[] requests;
    delete[] kv_entries;
    return m;
}

static void print_metrics(const RunMetrics& m) {
    printf("  host_kernel_launches: %d\n", m.host_kernel_launches);
    printf("  host_synchronizations: %d\n", m.host_synchronizations);
    printf("  completed_requests: %d / %d\n", m.completed_requests, m.target_requests);
    printf("  elapsed_ms: %.3f\n", m.elapsed_ms);
    printf("  tokens_generated: %d\n", m.tokens_generated);
    printf("  tokens_per_second: %.0f\n", m.tokens_per_second);
}

static void write_csv(FILE* f, const RunMetrics& m) {
    fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.0f\n",
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
            m.tokens_per_second);
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
            fprintf(stderr, "Usage: %s [--mode baseline|mega|both] [--requests N] [--tokens N] [--draft-len N] [--iterations N] [--csv path]\n", argv[0]);
            return 1;
        }
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

    RunMetrics baseline_metrics = {};
    RunMetrics mega_metrics = {};

    if (do_baseline) {
        printf("\n========================================\n");
        printf("Baseline host-launched decode:\n");
        printf("  (CPU controls token loop, repeated launches)\n");
        printf("========================================\n\n");

        baseline_metrics = run_baseline_path(requests, tokens, draft_len, iterations);

        printf("\nBaseline host-launched decode:\n");
        print_metrics(baseline_metrics);
    }

    if (do_mega) {
        printf("\n========================================\n");
        printf("Persistent mega-kernel:\n");
        printf("  (one launch, device-resident loop)\n");
        printf("========================================\n\n");

        mega_metrics = run_megakernel_path(requests, tokens, draft_len, iterations);

        printf("\nPersistent mega-kernel:\n");
        print_metrics(mega_metrics);
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

    if (csv_path != nullptr) {
        FILE* f = fopen(csv_path, "w");
        if (f == nullptr) {
            fprintf(stderr, "Error: cannot open %s for writing\n", csv_path);
            return 1;
        }
        fprintf(f, "mode,requests,tokens_per_request,draft_len,"
                   "host_kernel_launches,host_synchronizations,"
                   "completed_requests,target_requests,"
                   "tokens_generated,elapsed_ms,tokens_per_second\n");
        if (do_baseline) write_csv(f, baseline_metrics);
        if (do_mega) write_csv(f, mega_metrics);
        fclose(f);
        printf("\nCSV written to %s\n", csv_path);
    }

    if (do_baseline && do_mega) {
        bool baseline_ok = (baseline_metrics.completed_requests == baseline_metrics.target_requests);
        bool mega_ok = (mega_metrics.completed_requests == mega_metrics.target_requests);
        bool baseline_launches_gt1 = (baseline_metrics.host_kernel_launches > 1);
        bool mega_launches_eq1 = (mega_metrics.host_kernel_launches == 1);
        bool baseline_syncs_gt1 = (baseline_metrics.host_synchronizations > 1);
        bool mega_syncs_eq1 = (mega_metrics.host_synchronizations == 1);

        printf("\n=== Validation ===\n");
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
