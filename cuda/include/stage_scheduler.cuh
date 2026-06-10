#ifndef STAGE_SCHEDULER_CUH
#define STAGE_SCHEDULER_CUH

#include "request_desc.h"

__device__ __forceinline__ int pick_next_request(
    RequestDescriptor* requests,
    int num_requests,
    int start_index
) {
    for (int i = 0; i < num_requests; i++) {
        int idx = (start_index + i) % num_requests;
        RequestDescriptor* r = &requests[idx];
        if (r->is_done()) continue;
        if (r->has_flag(REQUEST_FLAG_PRIORITY_HIGH)) return idx;
    }
    for (int i = 0; i < num_requests; i++) {
        int idx = (start_index + i) % num_requests;
        if (!requests[idx].is_done()) return idx;
    }
    return -1;
}

#endif
