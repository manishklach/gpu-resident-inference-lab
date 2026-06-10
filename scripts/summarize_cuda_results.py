#!/usr/bin/env python3
"""
Summarize CUDA measurement CSV output from xlpk_cuda_smoke.

Usage:
    python scripts/summarize_cuda_results.py <csv_path>

If matplotlib is installed, also generates a bar chart.
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


def summarize(rows: list[dict[str, str]]) -> None:
    print("CUDA Measurement Summary")
    print("=" * 50)

    baseline = None
    mega = None

    for r in rows:
        if r["mode"] == "baseline":
            baseline = r
        elif r["mode"] == "mega_kernel":
            mega = r

    if baseline:
        print("\nbaseline:")
        print(f"  launches: {baseline['host_kernel_launches']}")
        print(f"  synchronizations: {baseline['host_synchronizations']}")
        print(f"  completed_requests: {baseline['completed_requests']} / {baseline['target_requests']}")
        print(f"  tokens_generated: {baseline['tokens_generated']}")
        print(f"  elapsed_ms: {baseline['elapsed_ms']}")
        print(f"  TPS: {float(baseline['tokens_per_second']):.0f}")

    if mega:
        print("\nmega_kernel:")
        print(f"  launches: {mega['host_kernel_launches']}")
        print(f"  synchronizations: {mega['host_synchronizations']}")
        print(f"  completed_requests: {mega['completed_requests']} / {mega['target_requests']}")
        print(f"  tokens_generated: {mega['tokens_generated']}")
        print(f"  elapsed_ms: {mega['elapsed_ms']}")
        print(f"  TPS: {float(mega['tokens_per_second']):.0f}")

    if baseline and mega:
        b_launches = int(baseline["host_kernel_launches"])
        m_launches = int(mega["host_kernel_launches"])
        b_syncs = int(baseline["host_synchronizations"])
        m_syncs = int(mega["host_synchronizations"])

        print("\nReduction:")
        launch_ratio = b_launches / m_launches if m_launches > 0 else 0
        sync_ratio = b_syncs / m_syncs if m_syncs > 0 else 0
        print(f"  launches: {launch_ratio:.0f}x ({b_launches} vs {m_launches})")
        print(f"  synchronizations: {sync_ratio:.0f}x ({b_syncs} vs {m_syncs})")

        b_tps = float(baseline["tokens_per_second"])
        m_tps = float(mega["tokens_per_second"])
        if b_tps > 0:
            print(f"  speedup: {m_tps / b_tps:.2f}x")


def try_chart(rows: list[dict[str, str]], csv_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available — skipping chart)")
        return

    baseline = None
    mega = None
    for r in rows:
        if r["mode"] == "baseline":
            baseline = r
        elif r["mode"] == "mega_kernel":
            mega = r

    if not baseline or not mega:
        return

    labels = ["host_kernel_launches", "host_synchronizations", "tokens_per_second"]
    b_vals = [int(baseline.get("host_kernel_launches", 0)),
              int(baseline.get("host_synchronizations", 0)),
              float(baseline.get("tokens_per_second", 0))]
    m_vals = [int(mega.get("host_kernel_launches", 0)),
              int(mega.get("host_synchronizations", 0)),
              float(mega.get("tokens_per_second", 0))]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], b_vals, width, label="baseline")
    ax.bar([i + width / 2 for i in x], m_vals, width, label="mega_kernel")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title("CUDA Measurement: Baseline vs Mega-Kernel")
    ax.legend()

    out = Path(csv_path).with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out}")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_cuda_results.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not Path(csv_path).is_file():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    rows = load_csv(csv_path)
    if not rows:
        print("Error: CSV file is empty or has no data rows")
        sys.exit(1)

    summarize(rows)
    try_chart(rows, csv_path)


if __name__ == "__main__":
    main()
