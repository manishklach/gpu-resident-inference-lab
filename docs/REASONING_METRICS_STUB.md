# Reasoning Metrics Stub

This is a toy simulator.

It does not prove reasoning ability.

It exists to define metric vocabulary.

It helps separate token throughput from accepted/verified branch throughput.

It is a placeholder for future real verifier/commit experiments.

## Usage

```bash
python tools/reasoning_metrics_stub.py --branches 8 --cycles 100 --accept-prob 0.35 --cycle-latency-ms 10
```

## Metrics

- `cycles`: number of simulated verification cycles
- `branches_per_cycle`: candidate branches proposed in each cycle
- `total_candidate_branches`: total drafted branches across the run
- `accepted_branches`: candidate branches accepted by the simulated verifier
- `rejected_branches`: candidate branches rejected by the simulated verifier
- `verifier_rejection_rate`: rejected branches divided by total candidate branches
- `avg_accepted_branches_per_cycle`: accepted branches per simulated cycle
- `simulated_cycle_latency_ms`: fixed per-cycle latency assumption used by the stub
- `simulated_verified_decisions_per_sec`: accepted branches divided by total simulated wall-clock time

## Boundary

This script is deliberately fake.

It does not run a model, call tools, retrieve documents, score correctness, or benchmark real reasoning. Its purpose is to define metric vocabulary for future verifier/commit experiments.
