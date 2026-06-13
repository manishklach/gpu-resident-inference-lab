from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare_benchmarks.py <benchmark_a.json> <benchmark_b.json>")

    left = _load(sys.argv[1])
    right = _load(sys.argv[2])
    table = Table(title="Benchmark Diff")
    table.add_column("Domain")
    table.add_column("Variant")
    table.add_column("Delta D/s")

    domains = sorted(set(left["results"]) | set(right["results"]))
    for domain in domains:
        left_map = {item["config_name"]: item["ds_score"] for item in left["results"].get(domain, [])}
        right_map = {item["config_name"]: item["ds_score"] for item in right["results"].get(domain, [])}
        for variant in sorted(set(left_map) | set(right_map)):
            delta = right_map.get(variant, 0.0) - left_map.get(variant, 0.0)
            color = "green" if delta >= 0 else "red"
            table.add_row(domain, variant, f"[{color}]{delta:.4f}[/{color}]")

    Console().print(table)


if __name__ == "__main__":
    main()
