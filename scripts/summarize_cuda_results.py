#!/usr/bin/env python3
"""
Summarize CUDA measurement CSV output from xlpk_cuda_smoke.

Usage:
    python scripts/summarize_cuda_results.py <csv_path>

Expected CSV header (14 columns):
    mode,requests,tokens_per_request,draft_len,
    host_kernel_launches,host_synchronizations,
    completed_requests,target_requests,
    tokens_generated,elapsed_ms,tokens_per_second,
    launch_reduction,sync_reduction,speedup_vs_baseline

If matplotlib is installed, also generates a bar chart of launch_reduction
by (requests, tokens_per_request) configuration, saved to
results/launch_reduction.png.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> None:
    print("CUDA Sweep Summary")
    print("=" * 60)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[r["mode"]].append(r)

    for mode in ["baseline", "mega_kernel"]:
        if mode not in groups:
            continue
        r = groups[mode]
        elapsed = [float(x["elapsed_ms"]) for x in r]
        tps = [float(x["tokens_per_second"]) for x in r]
        launches = [int(x["host_kernel_launches"]) for x in r]
        syncs = [int(x["host_synchronizations"]) for x in r]

        print(f"\n{mode} ({len(r)} runs):")
        print(f"  elapsed_ms:       mean={_mean(elapsed):>10.3f}  min={min(elapsed):>10.3f}  max={max(elapsed):>10.3f}")
        print(f"  tokens_per_sec:   mean={_mean(tps):>10.0f}  min={min(tps):>10.0f}  max={max(tps):>10.0f}")
        print(f"  launches:         mean={_mean(launches):>10.1f}  min={min(launches):>10d}  max={max(launches):>10d}")
        print(f"  synchronizations: mean={_mean(syncs):>10.1f}  min={min(syncs):>10d}  max={max(syncs):>10d}")

    if "mega_kernel" in groups:
        mega = groups["mega_kernel"]
        lr = [int(x.get("launch_reduction", "0")) for x in mega]
        sr = [int(x.get("sync_reduction", "0")) for x in mega]
        sp = [float(x.get("speedup_vs_baseline", "1.0")) for x in mega]

        print(f"\nmega_kernel (reduction vs baseline, {len(mega)} runs):")
        print(f"  launch_reduction:    mean={_mean(lr):>10.1f}  min={min(lr):>10d}  max={max(lr):>10d}")
        print(f"  sync_reduction:      mean={_mean(sr):>10.1f}  min={min(sr):>10d}  max={max(sr):>10d}")
        print(f"  speedup_vs_baseline: mean={_mean(sp):>10.2f}x  min={min(sp):>10.2f}x  max={max(sp):>10.2f}x")


def _mean(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0


def try_chart(rows: list[dict[str, str]], csv_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping chart)")
        return

    mega = [r for r in rows if r["mode"] == "mega_kernel"]
    if not mega:
        return

    # Build a pivot: requests x tokens_per_request -> avg launch_reduction
    pivot: dict[tuple[int, int], list[float]] = {}
    for r in mega:
        key = (int(r["requests"]), int(r["tokens_per_request"]))
        val = float(r.get("launch_reduction", "0"))
        pivot.setdefault(key, []).append(val)

    if not pivot:
        return

    request_vals = sorted({k[0] for k in pivot})
    token_vals = sorted({k[1] for k in pivot})

    x = range(len(request_vals))
    width = 0.8 / max(len(token_vals), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for ti, tok in enumerate(token_vals):
        heights = [_mean(pivot.get((req, tok), [0.0])) for req in request_vals]
        offset = (ti - (len(token_vals) - 1) / 2) * width
        ax.bar([i + offset for i in x], heights, width, label=f"tokens={tok}")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(r) for r in request_vals])
    ax.set_xlabel("requests")
    ax.set_ylabel("launch_reduction (baseline launches : 1 mega launch)")
    ax.set_title("CUDA Sweep: Launch Reduction by Configuration")
    ax.legend(title="tokens/req")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "launch_reduction.png"
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
