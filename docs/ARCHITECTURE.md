# Architecture

## Objective

Build a persistent decode runtime that can eventually host a mega-kernel style execution model:

- one long-lived worker loop
- request state kept resident
- speculative block proposal and acceptance
- minimal host orchestration

This repo starts with a simulation of that architecture.

## Core Concepts

### 1. Persistent Runtime

Instead of treating decode as:

1. launch operator
2. return to host
3. launch next operator

we model decode as:

1. request enters runtime
2. runtime owns request state until completion
3. each iteration proposes and verifies a token block
4. accepted tokens are committed in place

In CUDA, this becomes a persistent kernel. In this MVP, it is a persistent software loop.

### 2. Request State

Each active request owns:

- prompt tokens
- committed output tokens
- temporary draft tokens
- finish flag
- decode position

This mirrors what a real on-device request descriptor would track.

### 3. KV-Cache Residency

The KV-cache abstraction tracks which committed positions belong to which request.

The current implementation is intentionally simple:

- one request id
- positions committed so far

Later, this can evolve into:

- paged KV caches
- packed per-head layouts
- speculative entries vs committed entries
- per-device placement

### 4. Speculative Block Flow

Each decode iteration is modeled as:

1. proposer generates a draft block
2. verifier decides how many positions are accepted
3. accepted prefix is committed
4. rejected suffix is discarded
5. EOS or max token limit ends the request

This matches the key control pattern behind MTP / DFlash-style decoding.

### 5. Backend Boundary

The simulator separates runtime logic from kernel logic.

Today:

- proposer and verifier are Python objects

Later:

- proposer can call fused CUDA kernels
- verifier can call a large-model verify kernel
- runtime can hand work to a persistent device scheduler

## Planned CUDA Evolution

### Phase 1

- implement a single-request persistent decode kernel
- keep token buffer and request metadata on device
- model one speculative block per iteration

### Phase 2

- add fused attention/projection/sampling interfaces
- move KV commit onto device
- add persistent work queues

### Phase 3

- add multi-request continuous batching
- add multi-GPU communication overlap
- specialize workers for compute, memory movement, and communication

## Non-Goals For This MVP

- reproducing Xiaomi's production stack
- reproducing exact TileRT internals
- pretending the Python simulator is a performance artifact

The MVP is about getting the state machine right.
