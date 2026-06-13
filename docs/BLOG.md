# The Next Inference Bottleneck Is Not FLOPs. It Is the Shape of the Decode Loop.

The interesting inference problem is no longer just “how many FLOPs can the accelerator sustain?” or “can we build another SRAM chip?” GPUs already have fast on-chip SRAM and sophisticated memory hierarchies. The deeper systems question is whether we are using that local workspace well enough while decode remains latency-sensitive, sparse, and orchestration-heavy.

This is the lens behind GPU Resident Inference Lab.

## The Bottleneck Is Shifting

As inference becomes more quantized and sparse, raw compute stops being the whole story.

- FP4 / NVFP4-style quantization reduces weight bandwidth.
- MoE sparsity reduces active parameters per token.
- SWA, local attention, and sparse KV schemes bound how much context must be touched.
- MTP, speculative decode, and block decode make the loop wider than one token at a time.

Once those ingredients are in place, the remaining problem is often the shape of the decode loop itself: how much of it is still CPU-driven, how often the GPU must wait for launch/sync orchestration, and how much KV movement happens unnecessarily between narrow steps.

## No Need for Another SRAM Chip

The answer is not necessarily a new SRAM chip. SRAM is already the GPU’s fast local workspace.

The real win is using existing on-chip SRAM, registers, shared memory, and the broader GPU memory hierarchy more effectively:

- keep the hot inference loop resident on device
- avoid CPU launch/sync gaps when possible
- touch fewer KV blocks per iteration
- keep useful token/block work flowing through the resident loop
- stage and rebalance KV intelligently across tiers

## A Wider Resident Loop

Traditional decode often looks like a skinny host-driven sequence:

```text
CPU launch
GPU decode
CPU sync
CPU launch
GPU verify
CPU sync
repeat
```

A future runtime may look more like:

```text
CPU submits coarse work once
GPU persistent loop:
  select sparse KV blocks
  draft token block
  route experts
  verify / attend
  commit accepted tokens
  update KV / state
  prefetch or spill residency state
CPU receives coarse completions
```

This is where persistent GPU-resident mega-kernels become interesting. Their value is not mystical. They remove orchestration gaps. The loop becomes genuinely useful when it also has:

- token/block parallel work
- sparse KV selection
- tier-aware residency decisions
- enough scheduling logic to keep the GPU busy

## What This Repo Studies

GPU Resident Inference Lab studies the runtime/kernel side of that stack.

It is not a production serving engine. It is a research scaffold for:

- persistent GPU-resident loops
- sparse KV selection
- speculative and block decode workflows
- tiered KV movement and residency
- pressure handling and eviction
- trace-driven admission and scheduling

The repo intentionally uses honest placeholder math in many places today. That is a feature, not a bug. The goal is to make control-flow bottlenecks visible before pretending the entire transformer stack is already fused and solved.

## The Real Thesis

The next inference bottleneck is not only FLOPs. It is the shape of the decode loop.

Once quantization reduces weight bandwidth, sparsity reduces active compute, and wider decode strategies create more resident work, the remaining runtime challenge is to keep that work on GPU with fewer orchestration gaps and less unnecessary KV traffic.

That is the layer this repo is trying to make legible.
