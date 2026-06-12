/**
 * resident_scheduler_kernel.cu — Research-stage resident scheduler kernel.
 *
 * Role:
 *   A simple GPU-side scheduler that scans request descriptors, orders live
 *   requests by priority, and emits a compact schedule list.
 *
 * Scope:
 *   - deterministic, metadata-only scheduler
 *   - no queues, atomics, or continuous admission yet
 *   - intended as a future building block for GPU-resident scheduling
 */

#include <cuda_runtime.h>

#include "kernel_status.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"

__global__ void resident_schedule_requests_kernel(
    RequestDescriptor* requests,
    int num_requests,
    int* scheduled_request_ids,
    int* scheduled_priorities,
    SchedulerKernelMetrics* metrics
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    SchedulerKernelMetrics metric = {};
    for (int i = 0; i < num_requests; ++i) {
        scheduled_request_ids[i] = -1;
        scheduled_priorities[i] = -1;
    }

    int scheduled = 0;
    bool scheduled_flags[256];
    for (int i = 0; i < 256; ++i) {
        scheduled_flags[i] = false;
    }

    for (int slot = 0; slot < num_requests; ++slot) {
        int best_index = -1;
        int best_priority = -2147483647;

        for (int i = 0; i < num_requests && i < 256; ++i) {
            metric.requests_examined += 1;
            if (scheduled_flags[i]) {
                continue;
            }
            if (requests[i].is_done()) {
                continue;
            }
            if (requests[i].priority > best_priority) {
                best_priority = requests[i].priority;
                best_index = i;
            }
        }

        if (best_index < 0) {
            break;
        }

        scheduled_flags[best_index] = true;
        scheduled_request_ids[slot] = requests[best_index].request_id;
        scheduled_priorities[slot] = requests[best_index].priority;
        scheduled += 1;
    }

    metric.requests_scheduled = scheduled;
    metrics[0] = metric;
}
