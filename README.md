# XL-Persistent-Kernel

`XL-Persistent-Kernel` is a prototype repository for building a Mirage/TileRT-style persistent decode runtime for large language model inference.

The goal is not to pretend we already have Xiaomi's production stack. The goal is to build the right scaffolding:

- a persistent decode state machine
- GPU-oriented request and KV-cache abstractions
- a speculative block proposal and acceptance flow
- clear boundaries between model logic, runtime scheduling, and backend kernels

This first version is CPU-only and intentionally simple. It gives us a place to validate the control flow before we move hot paths into CUDA.

## What This Repo Contains

- `docs/ARCHITECTURE.md`
  - MVP architecture for a persistent mega-kernel runtime
- `src/megakernel_lab/runtime.py`
  - persistent decode runtime simulator
- `src/megakernel_lab/spec_decode.py`
  - speculative block proposal and acceptance policy
- `src/megakernel_lab/state.py`
  - request, token, and scheduler state
- `src/megakernel_lab/kv_cache.py`
  - simple KV-cache layout abstraction
- `src/megakernel_lab/demo.py`
  - runnable demo comparing serial decode vs block speculative decode
- `tests/test_runtime.py`
  - small tests for acceptance and state transitions

## Design Intent

We are modeling the same high-level split that shows up in Mirage and TileRT:

- model side
  - token proposal
  - speculative block behavior
  - quantized weight / KV assumptions
- runtime side
  - request queueing
  - persistent worker loop
  - token commit / rejection handling
  - KV residency and scheduler ownership
- backend side
  - future CUDA kernels
  - future fused operators
  - future multi-GPU communication overlap

## MVP Roadmap

1. CPU simulator for persistent decode loop
2. explicit request queue and worker specialization model
3. paged KV-cache planner
4. backend interface for pluggable kernels
5. single-GPU CUDA prototype
6. speculative verify path in CUDA
7. multi-GPU overlap and communication

## Quick Start

Create a virtual environment if you want one, then run:

```bash
python -m megakernel_lab.demo
```

Run tests with:

```bash
python -m pytest
```

## Why Start With a Simulator

Before writing a giant CUDA kernel, we need to be precise about:

- which buffers are authoritative
- when draft tokens become committed tokens
- what state lives with the request vs the scheduler
- what must remain on device in a real persistent kernel

That control flow is what this repo captures first.
