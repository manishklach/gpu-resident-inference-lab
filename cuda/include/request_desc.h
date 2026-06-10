#ifndef REQUEST_DESC_H
#define REQUEST_DESC_H

#include "kernel_status.h"

struct RequestDescriptor {
    int request_id;
    int state;
    int flags;
    int priority;

    int prompt_len;
    int decode_pos;
    int max_new_tokens;
    int eos_token_id;

    int last_token;
    int output_token_count;

    int draft_offset;
    int draft_len;
    int accepted_prefix_len;

    int kv_table_offset;
    int kv_num_pages;

    int error_code;

    __host__ __device__ bool is_state(RequestState s) const {
        return state == static_cast<int>(s);
    }

    __host__ __device__ void set_state(RequestState s) {
        state = static_cast<int>(s);
    }

    __host__ __device__ bool is_done() const {
        return is_state(REQUEST_COMPLETE) || is_state(REQUEST_FAILED);
    }

    __host__ __device__ bool has_flag(RequestFlags f) const {
        return (flags & static_cast<int>(f)) != 0;
    }

    __host__ __device__ void set_flag(RequestFlags f) {
        flags |= static_cast<int>(f);
    }

    __host__ __device__ void clear_flag(RequestFlags f) {
        flags &= ~static_cast<int>(f);
    }
};

#endif
