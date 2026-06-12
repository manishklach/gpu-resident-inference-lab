#ifndef RESEARCH_KERNEL_METRICS_H
#define RESEARCH_KERNEL_METRICS_H

struct SparseKVGatherMetrics {
    int request_id;
    int blocks_examined;
    int blocks_selected;
    int bytes_read;
    int bytes_saved;
};

struct VerifyCommitMetrics {
    int request_id;
    int speculative_candidates;
    int accepted_tokens;
    int rejected_tokens;
    int committed_pages;
    int released_pages;
};

struct SchedulerKernelMetrics {
    int requests_examined;
    int requests_scheduled;
};

struct KVPrefetchMetrics {
    int request_id;
    int prefetch_pages;
    int prefetch_bytes;
};

struct CompactedDecodeMetrics {
    int request_id;
    int pages_consumed;
    int generated_tokens;
};

struct ResidentPipelineMetrics {
    int request_id;
    int loop_iterations;
    int blocks_selected;
    int bytes_read;
    int accepted_tokens;
    int committed_pages;
};

#endif
