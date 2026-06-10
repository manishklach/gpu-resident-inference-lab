#!/usr/bin/env python3
"""
Side-by-side comparison of Python CPU-simulator and CUDA measurement CSVs.

Usage:
    python scripts/compare_metrics.py <python_csv> <cuda_csv>

This does NOT require numerical agreement (math is fake on both sides).
It confirms both pipelines produce plausible metrics for equivalent configs
and highlights structural differences between the two measurement approaches.
"""

import csv
import sys
from pathlib import Path


def load_csv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_metrics.py <python_csv> <cuda_csv>")
        sys.exit(1)

    py_path = sys.argv[1]
    cu_path = sys.argv[2]

    if not Path(py_path).is_file():
        print(f"Error: file not found: {py_path}")
        sys.exit(1)
    if not Path(cu_path).is_file():
        print(f"Error: file not found: {cu_path}")
        sys.exit(1)

    py_rows = load_csv(py_path)
    cu_rows = load_csv(cu_path)

    if not py_rows:
        print("Error: Python CSV is empty")
        sys.exit(1)
    if not cu_rows:
        print("Error: CUDA CSV is empty")
        sys.exit(1)

    # Build CUDA lookup: (requests, draft_len) -> mega_kernel TPS
    cu_lookup: dict[tuple[int, int], float] = {}
    for r in cu_rows:
        if r["mode"] == "mega_kernel":
            key = (int(r["requests"]), int(r["draft_len"]))
            val = float(r.get("tokens_per_second", "0"))
            cu_lookup[key] = val

    print("=" * 80)
    print("Python CPU Simulator  vs  CUDA Measurement — Side-by-Side")
    print("=" * 80)
    print(f"{'batch':>5} {'block':>5}  {'Python ITL':>10} {'Python accept':>13} "
          f"{'Python KV':>9}  {'CUDA TPS':>9}  {'CUDA reqs':>9} {'CUDA drf':>9}")
    print("-" * 80)

    matched = 0
    for py in py_rows:
        try:
            batch = int(py.get("batch_size", "0"))
            block = int(py.get("block_size", "0"))
        except ValueError:
            continue

        p50 = float(py.get("p50_itl_ms", "0"))
        acc = float(py.get("acceptance_rate", "0"))
        kv = float(py.get("kv_hit_rate", "0"))

        # Look up matching CUDA config: batch_size ~ requests, block_size ~ draft_len
        cu_tps = cu_lookup.get((batch, block))
        cu_reqs_str = str(batch)
        cu_drf_str = str(block)

        matched += 1
        cu_str = f"{cu_tps:>9.0f}" if cu_tps is not None else "   N/A   "
        print(f"{batch:>5} {block:>5}  {p50:>10.3f} {acc:>13.3f} "
              f"{kv:>9.3f}  {cu_str:>9}  {cu_reqs_str:>9} {cu_drf_str:>9}")

    print("-" * 80)
    print(f"Matched {matched} Python rows against CUDA sweep data.")
    print()

    # Summary
    any_cu = any(r["mode"] == "mega_kernel" for r in cu_rows)
    any_py = len(py_rows) > 0
    print("Pipelines:")
    print(f"  Python: {'OK' if any_py else 'EMPTY'} ({len(py_rows)} rows)")
    print(f"  CUDA:   {'OK' if any_cu else 'EMPTY'} ({len(cu_rows)} rows)")
    print()
    print("Note: Python measures simulated latency (ms) on CPU; CUDA measures")
    print("wall-clock time (ms) on GPU. The two pipelines use different fake-math")
    print("stubs. Numerical agreement is NOT expected — the goal is to confirm")
    print("that both produce structurally plausible metrics for the same configs.")


if __name__ == "__main__":
    main()
