# From GPU-Resident Decode to GPU-Resident Reasoning

## 1. Motivation

Tokens/sec is useful but incomplete.

For agentic workloads, the more important metric is verified decisions/sec.
A decision may involve generation, retrieval, tool calls, branch verification, and commit/reject logic.
The repo begins at the lower-level runtime layer because those loops eventually need a fast substrate.

## 2. Decode Loop vs Reasoning Loop

Traditional decode loop:

```markdown
next token -> update KV -> next token -> update KV
```

Future reasoning loop:

```markdown
draft candidate tokens/branches
-> retrieve/tool call if needed
-> verify candidates
-> commit accepted path
-> reject/refine failed paths
-> update KV/state
-> schedule next branch/block
```

The repo currently models the lower-level decode/control-flow pieces, not the full future reasoning loop.

## 3. Why Parallel Token/Block Decode Matters

One-token-at-a-time decode is too narrow for modern GPUs.

Speculative decoding, MTP, block decoding, and draft/verify pipelines widen the work.
This creates enough useful work for persistent GPU-resident loops.
Without wider decode, a persistent kernel only makes a serial loop more efficient.

## 4. Speculative Decoding to Speculative Reasoning

Speculative decoding drafts future tokens.

Speculative reasoning drafts future reasoning paths or branches.

A verifier/commit step accepts useful branches and rejects others.
This is conceptually similar to draft/verify, but at a higher semantic level.
The repo does not implement this today; it only creates the systems scaffolding that could eventually support it.

## 5. Dynamic Precision Routing

FP4/NVFP4 reduces weight bandwidth.

But not all tokens/steps are equally important.
Easy continuation tokens may tolerate low precision.
Hard reasoning or high-uncertainty steps may require FP8/BF16.
Future systems may choose precision per token, per layer, or per reasoning segment.
This repo should treat this as a future scheduling policy, not a current capability.

## 6. Memory Hierarchy as a First-Class Design Variable

HBM, L2, shared memory, registers, KV cache, and expert routing cannot be optimized separately.

Attention windows, MoE routing, KV eviction, prefetch, and persistent kernels should be co-designed.
The repo’s sparse KV and tiered residency ideas are early pieces of this.

## 7. Proposed Future Metrics

| Metric | Meaning | Status |
|---|---|---|
| tokens_per_resident_loop | tokens produced or processed inside one GPU-resident loop | near-term |
| accepted_tokens_per_verify | number of draft tokens accepted per verification step | near-term/future |
| selected_kv_blocks_ratio | selected KV blocks divided by total candidate KV blocks | near-term |
| estimated_kv_bytes_saved | approximate memory traffic avoided through sparse KV selection | near-term |
| branches_verified_per_cycle | candidate reasoning branches verified per cycle | future |
| verified_decisions_per_sec | correct/accepted decisions per second | future |
| verifier_rejection_rate | fraction of branches/tokens rejected by verifier | future |
| latency_per_verified_answer | end-to-end latency for a verified answer | future |

## 8. Implementation Boundary

This repo should first make GPU-resident decode/control-flow measurable.
Only after that should it attempt higher-level reasoning pipeline experiments.
The right order is:

1. measure orchestration gaps
2. keep decode/block loops resident
3. add sparse KV and residency decisions
4. add draft/verify structure
5. then explore branch-level reasoning and verified decisions/sec
