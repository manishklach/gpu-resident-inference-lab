# CUDA Staging: Host/Device Queue Design

This document describes the planned CUDA persistent-kernel architecture for replacing the CPU simulator with a real device-resident decode loop.

## RequestDescriptor Layout

The `RequestDescriptor` struct (defined in `cuda/request_desc.h`) is the device-side representation of one active request. It mirrors the Python `RequestState` dataclass but uses fixed-size device-compatible types.

```c++
struct RequestDescriptor {
    uint32_t request_id;           // Unique identifier
    uint32_t priority;             // Scheduling priority

    const uint32_t* prompt_tokens; // Device pointer to prompt
    uint32_t prompt_len;

    uint32_t* committed_tokens;    // Device pointer to output buffer
    uint32_t committed_len;

    const uint32_t* target_tokens; // For speculative verification
    uint32_t target_len;

    uint32_t decode_pos;           // Current position (kernel advances this)
    uint32_t max_new_tokens;
    uint32_t eos_token_id;

    uint32_t* kv_page_table;       // Per-layer KV page table pointer
    uint32_t kv_page_table_entries;

    uint32_t status;               // Bitfield: ACTIVE | FINISHED | PREEMPTED
    uint32_t pad;
};
```

## Host Submission Queue

The host maintains a submission queue in **pinned host memory** (allocated via `cudaHostAlloc`). Pinned memory is accessible from both host and device without explicit `cudaMemcpy`.

```
Host memory (pinned):
+------------------+
| submit_queue     |  <-- Host writes new requests here
| [req0, req1, ..] |  <-- Device reads via mapped pointer
+------------------+
```

The host:
1. Allocates a `RequestDescriptor` in pinned memory
2. Fills in prompt tokens, target tokens, KV page pointers
3. Sets `status = REQ_STATE_ACTIVE`
4. Increments the submit queue head (atomic on device)

The device scheduler reads the queue head to discover new requests.

## Device Work Queue

The persistent kernel maintains an internal work queue in **device global memory**. This is separate from the host submission queue because:

- Device must own the queue for lock-free scheduling
- Host cannot directly modify device memory without memcpy
- Atomic operations on device are faster than host-device synchronization

```
Device global memory:
+------------------+
| work_queue       |  <-- Device scheduler owns this
| [active_req_ids] |  <-- Read by persistent decode loop
+------------------+
```

The scheduler loop:
1. Reads new requests from host submission queue (mapped pinned memory)
2. Adds them to the device work queue
3. Assigns request to a thread block
4. Block processes decode steps until completion
5. Moves completed request to completion queue

## Completion Queue

Completed requests are written to a **completion queue** in pinned host memory:

```
Host memory (pinned):
+------------------+
| completion_queue |  <-- Device writes finished requests here
| [req_id, status] |  <-- Host reads to reclaim resources
+------------------+
```

The device writes:
- `request_id`
- Final `committed_len`
- Final `status` (FINISHED or PREEMPTED)

The host polls this queue to:
- Free KV pages for completed requests
- Return tokens to the application
- Reclaim request descriptor slots

## Lifecycle States

```
Host submit --> ACTIVE
                 |
                 v
            Device picks up
                 |
                 v
            DECODE loop
            /    |    \
           v     v     v
        FINISHED PREEMPTED  (still ACTIVE, will resume)
           |        |
           v        v
        completion_queue
           |
           v
        Host reclaims resources
```

## Shutdown Protocol

The host signals shutdown by setting a device-resident flag:

```
Host: *d_shutdown = 1
Device kernel: while (!(*shutdown) && req.is_active()) { ... }
```

The device checks this flag at the top of each decode iteration. If set:
1. Current decode step completes (no partial writes)
2. Request status is preserved (can be resumed later)
3. Kernel exits gracefully
4. Host can inspect final state via completion queue

For immediate termination (e.g., GPU reset), the host calls `cudaDeviceReset()`.

## Speculative Verify Fusion

In a real implementation, speculative verification would be fused into the decode kernel:

```
persistent_decode_kernel {
    1. Load KV pages for current decode position
    2. Run attention + projection (fused)
    3. Sample token from logits
    4. Propose K draft tokens (small model or heuristic)
    5. Verify draft tokens against target sequence
    6. Accept prefix, reject suffix
    7. Update committed_tokens and decode_pos
    8. Release rejected draft KV pages
    9. Loop to step 1
}
```

Steps 2-6 would be a single fused kernel launch. The verify step compares:
- Draft token IDs against target token IDs
- Backend mask (from a verification kernel) against acceptance policy

Accepted tokens have their KV pages marked committed; rejected pages are released.

## KV Page Table

The KV page table lives in device global memory, structured as:

```
Per request, per layer:
  page_table[layer_id][logical_page] -> physical_page_id

Physical pages are allocated from a global page pool.
Pinned pages (active decode) are protected from eviction.
```

The page table is updated by the device kernel:
- **On commit**: new logical pages map to freshly allocated physical pages
- **On rejection**: draft physical pages are returned to the free pool
- **On eviction**: LRU unpinned pages are freed, table entries cleared

The page table structure mirrors the Python `KVCache._page_table` dictionary but uses flat arrays for device efficiency.

## How This Differs From Per-Token Host Launches

**Traditional approach:**
```
for each token:
    host: launch attention_kernel()
    host: launch projection_kernel()
    host: launch sampling_kernel()
    host: synchronize()
```

Each launch incurs:
- ~10us kernel launch overhead
- Host-device synchronization latency
- Memory fence costs
- Scheduling overhead

**Persistent kernel approach:**
```
host: launch persistent_decode_kernel()
device: loop forever {
    read request descriptor
    load KV pages
    run attention + projection + sample (fused)
    write new token
    advance position
}
host: poll completion queue
```

Benefits:
- Zero kernel launch overhead per token
- No host-device synchronization during decode
- KV pages stay in device cache (no host-induced evictions)
- Warp-level scheduling across requests
- Overlaps compute and memory movement

The CPU simulator in this repository models the same control flow that the persistent kernel will implement in device code.
