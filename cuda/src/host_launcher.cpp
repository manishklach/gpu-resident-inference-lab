#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "request_desc.h"
#include "kv_page_table.h"
#include "kernel_status.h"

extern __global__ void xl_persistent_megakernel(
    RequestDescriptor* requests, int num_requests,
    KVPageTable kv_table, int* draft_tokens,
    int* shutdown_flag, int max_iterations, int block_size
);

extern __global__ void baseline_host_decode_step_kernel(
    RequestDescriptor* requests, int num_requests
);

constexpr int MAX_REQUESTS = 8;
constexpr int MAX_ITERATIONS = 1000;
constexpr int BLOCK_SIZE = 256;
constexpr int SPEC_BLOCK_SIZE = 4;
constexpr int EOS_TOKEN_ID = 0;

static void check_cuda(const char* step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at '%s': %s\n", step, cudaGetErrorString(err));
        exit(1);
    }
}

static RequestDescriptor* alloc_and_copy_to_device(RequestDescriptor* host, int n) {
    size_t bytes = n * sizeof(RequestDescriptor);
    RequestDescriptor* d = nullptr;
    cudaMalloc(&d, bytes);
    check_cuda("cudaMalloc requests");
    cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice);
    check_cuda("cudaMemcpy requests to device");
    return d;
}

static void copy_back(RequestDescriptor* host, RequestDescriptor* dev, int n) {
    size_t bytes = n * sizeof(RequestDescriptor);
    cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost);
    check_cuda("cudaMemcpy requests back to host");
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

static void init_requests(RequestDescriptor* reqs, int n, int max_tokens, bool speculative) {
    for (int i = 0; i < n; i++) {
        reqs[i].request_id = i + 1;
        reqs[i].state = REQUEST_DECODE_READY;
        reqs[i].flags = speculative ? REQUEST_FLAG_SPECULATIVE_ENABLED : 0;
        reqs[i].priority = i;
        reqs[i].prompt_len = 4;
        reqs[i].decode_pos = 0;
        reqs[i].max_new_tokens = max_tokens;
        reqs[i].eos_token_id = EOS_TOKEN_ID;
        reqs[i].last_token = (reqs[i].request_id * 7 + 3) % 32000;
        reqs[i].output_token_count = 0;
        reqs[i].draft_offset = i * SPEC_BLOCK_SIZE * 4;
        reqs[i].draft_len = 0;
        reqs[i].accepted_prefix_len = 0;
        reqs[i].kv_table_offset = i * 4;
        reqs[i].kv_num_pages = 4;
        reqs[i].error_code = 0;
    }
}

static void run_baseline_path() {
    printf("\n========================================\n");
    printf("Baseline host-launched decode:\n");
    printf("  (conventional model: CPU launches one step at a time)\n");
    printf("========================================\n\n");

    constexpr int N = 4;
    RequestDescriptor requests[N];
    init_requests(requests, N, 4, false);
    print_requests("Initial state", requests, N);

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);
    int launch_count = 0;

    for (int iter = 0; iter < 20; iter++) {
        bool all_done = true;
        for (int i = 0; i < N; i++) {
            if (!requests[i].is_done()) { all_done = false; break; }
        }
        if (all_done) break;

        baseline_host_decode_step_kernel<<<1, BLOCK_SIZE>>>(d_reqs, N);
        check_cuda("baseline_host_decode_step_kernel");
        cudaDeviceSynchronize();
        launch_count++;
        copy_back(requests, d_reqs, N);
    }

    print_requests("After baseline decode", requests, N);
    printf("Baseline host-launched decode:\n");
    printf("  launches: %d\n", launch_count);
    printf("  completed_requests: %d / %d\n\n",
           count_completed(requests, N), N);

    cudaFree(d_reqs);
}

static void run_megakernel_path() {
    printf("\n========================================\n");
    printf("Persistent mega-kernel:\n");
    printf("  (one launch, device-resident loop)\n");
    printf("========================================\n\n");

    constexpr int N = 4;
    constexpr int TOTAL_KV_ENTRIES = N * 4;
    constexpr int DRAFT_BUFFER_SIZE = N * SPEC_BLOCK_SIZE * 4;

    RequestDescriptor requests[N];
    init_requests(requests, N, 6, true);
    print_requests("Initial state (speculative enabled)", requests, N);

    RequestDescriptor* d_reqs = alloc_and_copy_to_device(requests, N);

    KVPageEntry kv_entries[TOTAL_KV_ENTRIES];
    memset(kv_entries, 0, sizeof(kv_entries));
    for (int i = 0; i < TOTAL_KV_ENTRIES; i++) {
        kv_entries[i].page_id = i;
        kv_entries[i].state = KV_PAGE_FREE;
        kv_entries[i].flags = 0;
    }

    KVPageEntry* d_kv_entries = nullptr;
    cudaMalloc(&d_kv_entries, sizeof(kv_entries));
    check_cuda("cudaMalloc kv entries");
    cudaMemcpy(d_kv_entries, kv_entries, sizeof(kv_entries), cudaMemcpyHostToDevice);
    check_cuda("cudaMemcpy kv entries to device");

    KVPageTable host_kv_table;
    host_kv_table.entries = d_kv_entries;
    host_kv_table.num_entries = TOTAL_KV_ENTRIES;
    host_kv_table.page_size_tokens = 4;
    host_kv_table.bytes_per_page = 4096;

    int* d_draft_tokens = nullptr;
    cudaMalloc(&d_draft_tokens, DRAFT_BUFFER_SIZE * sizeof(int));
    check_cuda("cudaMalloc draft tokens");
    cudaMemset(d_draft_tokens, 0, DRAFT_BUFFER_SIZE * sizeof(int));

    int* d_shutdown = nullptr;
    cudaMalloc(&d_shutdown, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_shutdown, &zero, sizeof(int), cudaMemcpyHostToDevice);

    printf("Launching xl_persistent_megakernel (%d blocks)...\n", N);
    xl_persistent_megakernel<<<N, BLOCK_SIZE>>>(
        d_reqs, N, host_kv_table, d_draft_tokens, d_shutdown,
        MAX_ITERATIONS, SPEC_BLOCK_SIZE
    );
    check_cuda("xl_persistent_megakernel");
    cudaDeviceSynchronize();

    copy_back(requests, d_reqs, N);
    print_requests("After mega-kernel", requests, N);

    printf("Persistent mega-kernel:\n");
    printf("  launches: 1\n");
    printf("  completed_requests: %d / %d\n\n",
           count_completed(requests, N), N);

    cudaFree(d_reqs);
    cudaFree(d_kv_entries);
    cudaFree(d_draft_tokens);
    cudaFree(d_shutdown);
}

int main() {
    printf("========================================\n");
    printf("XL-Persistent-Kernel CUDA smoke test\n");
    printf("========================================\n\n");

    run_baseline_path();
    run_megakernel_path();

    printf("=== All smoke tests completed ===\n");
    return 0;
}
