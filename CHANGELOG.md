# Changelog

## Unreleased

### Added

- Added future-facing documentation on GPU-resident reasoning pipelines.
- Added a clear tokens/sec to verified decisions/sec framing.
- Added Phase 9 roadmap for speculative reasoning, verifier/commit workflows, adaptive precision, and decision-quality metrics.
- Added `tools/reasoning_metrics_stub.py`, a toy simulator for branch acceptance/rejection metrics.

### Repositioned

- Renamed the project direction from a narrow persistent-kernel experiment to a GPU-resident inference systems lab.
- Added clearer framing around persistent loops, sparse KV selection, speculative/token-block decode, and tiered KV residency.

### Documentation

- Added modern inference stack explanation.
- Added CPU-driven vs GPU-resident decode diagrams.
- Added “Persistent Kernels Are Not Enough” section.
- Added transparent “What runs today vs future work” table.
- Added non-goals and evaluation criteria.
- Reorganized the roadmap around phased GPU-resident inference research.

### Clarified

- Clarified that the repo does not currently implement real reasoning, verifier models, retrieval/tool loops, or correctness scoring.
- Clarified that persistent kernels alone do not make autoregressive decoding parallel; wider token/block work is needed.
