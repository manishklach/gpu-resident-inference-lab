"""Stub simulator for future verified-decisions/sec metrics.

This script does not perform real reasoning and should not be interpreted as a
real model benchmark. It simulates candidate branches, verifier accept/reject
rates, and clearly labeled placeholder metrics.
"""

from __future__ import annotations

import argparse
import random


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def unit_prob(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate branch acceptance/rejection metrics for future reasoning experiments."
    )
    parser.add_argument("--branches", type=positive_int, default=8)
    parser.add_argument("--cycles", type=positive_int, default=100)
    parser.add_argument("--accept-prob", type=unit_prob, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycle-latency-ms", type=positive_float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    accepted_branches = 0
    rejected_branches = 0

    for _ in range(args.cycles):
        for _ in range(args.branches):
            if rng.random() < args.accept_prob:
                accepted_branches += 1
            else:
                rejected_branches += 1

    total_candidate_branches = args.cycles * args.branches
    verifier_rejection_rate = rejected_branches / total_candidate_branches
    avg_accepted_branches_per_cycle = accepted_branches / args.cycles
    simulated_total_seconds = (args.cycles * args.cycle_latency_ms) / 1000.0
    simulated_verified_decisions_per_sec = accepted_branches / simulated_total_seconds

    print("SIMULATED / NOT A REAL MODEL BENCHMARK")
    print()
    print(f"cycles: {args.cycles}")
    print(f"branches_per_cycle: {args.branches}")
    print(f"total_candidate_branches: {total_candidate_branches}")
    print(f"accepted_branches: {accepted_branches}")
    print(f"rejected_branches: {rejected_branches}")
    print(f"verifier_rejection_rate: {verifier_rejection_rate * 100:.2f}%")
    print(f"avg_accepted_branches_per_cycle: {avg_accepted_branches_per_cycle:.2f}")
    print(f"simulated_cycle_latency_ms: {args.cycle_latency_ms:.2f}")
    print(
        "simulated_verified_decisions_per_sec: "
        f"{simulated_verified_decisions_per_sec:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
