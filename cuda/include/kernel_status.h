#ifndef KERNEL_STATUS_H
#define KERNEL_STATUS_H

#include <cstdint>

enum RequestState : int {
    REQUEST_EMPTY             = 0,
    REQUEST_PREFILL_READY     = 1,
    REQUEST_PREFILL_DONE      = 2,
    REQUEST_DECODE_READY      = 3,
    REQUEST_DRAFT_READY       = 4,
    REQUEST_VERIFY_READY      = 5,
    REQUEST_COMMIT_READY      = 6,
    REQUEST_COMPLETE          = 7,
    REQUEST_FAILED            = 8,
};

enum RequestFlags : int {
    REQUEST_FLAG_SPECULATIVE_ENABLED = 1 << 0,
    REQUEST_FLAG_PRIORITY_HIGH       = 1 << 1,
    REQUEST_FLAG_HAS_DRAFT           = 1 << 2,
    REQUEST_FLAG_EOS_SEEN            = 1 << 3,
    REQUEST_FLAG_KV_PINNED           = 1 << 4,
    REQUEST_FLAG_KV_PRESSURE         = 1 << 5,
};

__host__ __device__ inline const char* request_state_name(RequestState s) {
    switch (s) {
        case REQUEST_EMPTY:         return "EMPTY";
        case REQUEST_PREFILL_READY: return "PREFILL_READY";
        case REQUEST_PREFILL_DONE:  return "PREFILL_DONE";
        case REQUEST_DECODE_READY:  return "DECODE_READY";
        case REQUEST_DRAFT_READY:   return "DRAFT_READY";
        case REQUEST_VERIFY_READY:  return "VERIFY_READY";
        case REQUEST_COMMIT_READY:  return "COMMIT_READY";
        case REQUEST_COMPLETE:      return "COMPLETE";
        case REQUEST_FAILED:        return "FAILED";
        default:                    return "UNKNOWN";
    }
}

#endif
