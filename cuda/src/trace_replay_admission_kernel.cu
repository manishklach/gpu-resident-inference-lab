/**
 * trace_replay_admission_kernel.cu — Research-stage trace replay + admission.
 *
 * Role:
 *   Replays a deterministic arrival/service trace on device, admits requests
 *   into a bounded active set, and records admission/completion ordering.
 *
 * Scope:
 *   - metadata-only request lifecycle replay
 *   - single-threaded device queue model for deterministic benchmarks
 *   - no real continuous batching, atomics, or concurrent queue consumers
 */

#include <cuda_runtime.h>

#include "queue_desc.h"
#include "request_desc.h"
#include "research_kernel_metrics.h"

__global__ void trace_replay_admission_kernel(
    RequestDescriptor* requests,
    int num_requests,
    const int* arrival_steps,
    const int* service_steps,
    int max_trace_steps,
    int max_active_requests,
    DeviceQueue pending_queue,
    int* active_request_ids,
    int* active_remaining_steps,
    int* admission_order,
    int* completion_order,
    TraceReplayMetrics* metrics
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    TraceReplayMetrics metric = {};
    for (int i = 0; i < max_active_requests; ++i) {
        active_request_ids[i] = -1;
        active_remaining_steps[i] = 0;
    }
    for (int i = 0; i < num_requests; ++i) {
        admission_order[i] = -1;
        completion_order[i] = -1;
    }

    int completed = 0;
    int next_admission_slot = 0;
    int next_completion_slot = 0;

    for (int step = 0; step < max_trace_steps && completed < num_requests; ++step) {
        metric.replay_steps += 1;

        for (int req = 0; req < num_requests; ++req) {
            if (arrival_steps[req] != step) {
                continue;
            }
            if (queue_try_push(&pending_queue, req)) {
                requests[req].set_state(REQUEST_PREFILL_READY);
                metric.arrival_events += 1;
            }
        }

        int queue_depth = (*pending_queue.tail - *pending_queue.head + pending_queue.capacity) % pending_queue.capacity;
        if (queue_depth > metric.queue_high_watermark) {
            metric.queue_high_watermark = queue_depth;
        }

        int active_count = 0;
        for (int i = 0; i < max_active_requests; ++i) {
            if (active_request_ids[i] >= 0) {
                active_count += 1;
            }
        }

        while (active_count < max_active_requests) {
            int req_index = -1;
            if (!queue_try_pop(&pending_queue, &req_index)) {
                break;
            }

            int slot = -1;
            for (int i = 0; i < max_active_requests; ++i) {
                if (active_request_ids[i] < 0) {
                    slot = i;
                    break;
                }
            }
            if (slot < 0) {
                break;
            }

            active_request_ids[slot] = req_index;
            active_remaining_steps[slot] = service_steps[req_index];
            requests[req_index].set_state(REQUEST_DECODE_READY);
            admission_order[next_admission_slot++] = requests[req_index].request_id;
            metric.admission_events += 1;
            active_count += 1;
        }

        if (active_count > metric.active_high_watermark) {
            metric.active_high_watermark = active_count;
        }

        for (int i = 0; i < max_active_requests; ++i) {
            const int req_index = active_request_ids[i];
            if (req_index < 0) {
                continue;
            }

            active_remaining_steps[i] -= 1;
            metric.total_service_quanta += 1;
            if (active_remaining_steps[i] > 0) {
                continue;
            }

            requests[req_index].set_state(REQUEST_COMPLETE);
            completion_order[next_completion_slot++] = requests[req_index].request_id;
            active_request_ids[i] = -1;
            active_remaining_steps[i] = 0;
            metric.completion_events += 1;
            completed += 1;
        }
    }

    metrics[0] = metric;
}
