from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from decisions_benchmark import VerificationResult
from ds_scorer import compute_ds


def _load_run(conn: sqlite3.Connection, run_id: str):
    rows = conn.execute(
        "SELECT domain, correct, confidence, latency_ms, difficulty FROM results WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    grouped = defaultdict(list)
    for domain, correct, confidence, latency_ms, difficulty in rows:
        grouped[domain].append(
            (
                VerificationResult(bool(correct), float(confidence), float(latency_ms)),
                float(difficulty),
            )
        )
    return grouped


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare_runs.py <run_id_1> <run_id_2>")

    run_a, run_b = sys.argv[1], sys.argv[2]
    conn = sqlite3.connect("results.db")
    data_a = _load_run(conn, run_a)
    data_b = _load_run(conn, run_b)

    table = Table(title="Run Comparison")
    table.add_column("Domain")
    table.add_column(run_a)
    table.add_column(run_b)
    table.add_column("Winner")

    for domain in sorted(set(data_a) | set(data_b)):
        results_a = [item[0] for item in data_a.get(domain, [])]
        diff_a = [item[1] for item in data_a.get(domain, [])]
        results_b = [item[0] for item in data_b.get(domain, [])]
        diff_b = [item[1] for item in data_b.get(domain, [])]
        report_a = compute_ds(results_a, diff_a, max(sum(r.latency_ms for r in results_a) / 1000.0, 1e-6))
        report_b = compute_ds(results_b, diff_b, max(sum(r.latency_ms for r in results_b) / 1000.0, 1e-6))
        delta = report_a.ds_score - report_b.ds_score
        winner = run_a if delta > 0 else run_b
        table.add_row(
            domain,
            f"{report_a.ds_score:.4f}",
            f"{report_b.ds_score:.4f}",
            f"{winner} ({abs(delta):.4f})",
        )

    Console().print(table)


if __name__ == "__main__":
    main()
