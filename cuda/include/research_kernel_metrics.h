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

struct DMAMovementPlanMetrics {
    int request_id;
    int pages_planned;
    int hbm_hits;
    int dram_fetches;
    int ssd_fetches;
    int dma_ops;
    int bytes_moved;
    int bytes_from_hbm;
    int bytes_from_dram;
    int bytes_from_ssd;
};

struct TieredKVStagingMetrics {
    int request_id;
    int staged_pages;
    int hbm_pages_staged;
    int dma_pages_staged;
    int buffer_slot_switches;
    int staging_bytes;
};

struct KVEvictionMetrics {
    int request_id;
    int pages_scanned;
    int pages_evicted;
    int draft_pages_evicted;
    int committed_pages_evicted;
    int pinned_pages_skipped;
    int selected_pages_skipped;
    int reclaimed_bytes;
};

struct KVTierResidencyMetrics {
    int request_id;
    int pages_rebalanced;
    int promotions_to_hbm;
    int promotions_to_dram;
    int demotions_to_dram;
    int demotions_to_ssd;
    int final_hbm_pages;
    int final_dram_pages;
    int final_ssd_pages;
    int bytes_promoted;
    int bytes_demoted;
};

struct TraceReplayMetrics {
    int replay_steps;
    int arrival_events;
    int admission_events;
    int completion_events;
    int queue_high_watermark;
    int active_high_watermark;
    int total_service_quanta;
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
