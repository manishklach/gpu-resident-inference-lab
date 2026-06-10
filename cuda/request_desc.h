/**
 * request_desc.h - Request descriptor for persistent decode kernel.
 *
 * This header defines the device-side layout of a request descriptor.
 * In a real persistent-kernel runtime, the host writes these descriptors
 * into pinned global memory, and the device-side scheduler reads them
 * without returning control to the host.
 *
 * The descriptor tracks:
 * - prompt and output token buffers (pointers into device memory)
 * - decode position (how many tokens have been committed)
 * - max tokens budget
 * - EOS token ID
 * - completion status (written by device, read by host)
 * - request state flags (active, finished, preempted)
 *
 * This maps directly to the Python RequestState dataclass in state.py.
 */

#ifndef REQUEST_DESC_H
#define REQUEST_DESC_H

#include <cstdint>

/**
 * Request state flags as bitfield values.
 * The host sets REQ_STATE_ACTIVE; the device sets REQ_STATE_FINISHED
 * or REQ_STATE_PREEMPTED when appropriate.
 */
enum RequestStateFlags : uint32_t {
    REQ_STATE_NONE       = 0u,
    REQ_STATE_ACTIVE     = 1u << 0,  // Host: request is live
    REQ_STATE_FINISHED   = 1u << 1,  // Device: EOS or budget exhausted
    REQ_STATE_PREEMPTED  = 1u << 2,  // Device: request paused for another
};

/**
 * RequestDescriptor - device-resident request state.
 *
 * Layout is designed for coalesced reads by warp-sized scheduler loops.
 * All fields are aligned to 64-bit boundaries for efficient memory access.
 *
 * In a real implementation:
 * - prompt_tokens, committed_tokens, target_tokens live in device global memory
 * - kv_page_table is a pointer to the per-layer page table for this request
 * - decode_pos is advanced by the persistent kernel on each accepted token
 * - status is written atomically on completion
 */
struct RequestDescriptor {
    // --- Identity and metadata ---
    uint32_t request_id;         // Unique request identifier
    uint32_t priority;           // Scheduling priority (higher = sooner)

    // --- Token buffers (device pointers) ---
    const uint32_t* prompt_tokens;     // Pointer to prompt token IDs
    uint32_t prompt_len;                // Number of prompt tokens

    uint32_t* committed_tokens;         // Pointer to committed output buffer
    uint32_t committed_len;             // Number of committed output tokens

    const uint32_t* target_tokens;      // Pointer to target sequence (for verification)
    uint32_t target_len;                // Length of target sequence

    // --- Decode state ---
    uint32_t decode_pos;                // Current decode position (advanced by kernel)
    uint32_t max_new_tokens;            // Maximum tokens to generate
    uint32_t eos_token_id;              // End-of-sequence token ID

    // --- KV page table (per-layer) ---
    // In a real system this is a device pointer to a page table structure.
    // Each entry maps a logical token position to a physical KV page.
    uint32_t* kv_page_table;
    uint32_t kv_page_table_entries;     // Number of page table entries

    // --- Completion ---
    uint32_t status;                    // Bitfield of RequestStateFlags
    uint32_t pad;                       // Alignment padding

    /**
     * Check if this request is still active (not finished or preempted).
     */
    __host__ __device__ bool is_active() const {
        return (status & REQ_STATE_ACTIVE) != 0 &&
               (status & REQ_STATE_FINISHED) == 0;
    }

    /**
     * Mark this request as finished (EOS or budget exhausted).
     */
    __host__ __device__ void mark_finished() {
        status |= REQ_STATE_FINISHED;
        status &= ~REQ_STATE_ACTIVE;
    }

    /**
     * Advance decode position by one token.
     * Returns true if the request is still within budget.
     */
    __host__ __device__ bool advance() {
        decode_pos++;
        committed_len++;
        return decode_pos < max_new_tokens;
    }
};

#endif // REQUEST_DESC_H
