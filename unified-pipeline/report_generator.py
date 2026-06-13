from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmark_suite import BenchmarkReport


def _score_map(report: BenchmarkReport) -> dict[str, dict[str, float]]:
    mapped: dict[str, dict[str, float]] = {}
    for domain, items in report.results.items():
        mapped[domain] = {item.config_name: item.ds_score for item in items}
    return mapped


def generate_report(report: BenchmarkReport) -> None:
    console = Console()
    score_map = _score_map(report)
    variants = sorted({item.config_name for items in report.results.values() for item in items})
    domains = list(report.results.keys())

    overall_scores = {
        variant: sum(score_map.get(domain, {}).get(variant, 0.0) for domain in domains)
        for variant in variants
    }

    table1 = Table(title="Table 1 - Per-domain D/s Scores")
    table1.add_column("Variant")
    for domain in domains:
        table1.add_column(domain)
    table1.add_column("overall")

    for variant in variants:
        row = [variant]
        style_prefix = "[bold]" if variant == report.overall_winner else ""
        style_suffix = "[/bold]" if variant == report.overall_winner else ""
        for domain in domains:
            value = score_map[domain].get(variant, 0.0)
            domain_values = [score_map[domain].get(v, 0.0) for v in variants]
            color = "green" if value == max(domain_values) else "red" if value == min(domain_values) else "white"
            row.append(f"{style_prefix}[{color}]{value:.4f}[/{color}]{style_suffix}")
        overall_value = overall_scores[variant]
        overall_values = list(overall_scores.values())
        color = "green" if overall_value == max(overall_values) else "red" if overall_value == min(overall_values) else "white"
        row.append(f"{style_prefix}[{color}]{overall_value:.4f}[/{color}]{style_suffix}")
        table1.add_row(*row)

    full_overall = overall_scores.get("full", 0.0)
    table2 = Table(title="Table 2 - Component Impact")
    table2.add_column("Disabled Variant")
    table2.add_column("D/s Delta vs full")
    impacts: dict[str, float] = {}
    for variant in variants:
        if variant == "full":
            continue
        delta = full_overall - overall_scores.get(variant, 0.0)
        impacts[variant] = delta
        color = "green" if delta > 0 else "red"
        table2.add_row(variant, f"[{color}]{delta:.4f}[/{color}]")

    table3 = Table(title="Table 3 - Cost-efficiency")
    table3.add_column("Variant")
    table3.add_column("D/s per GFLOP proxy")
    efficiency_rows = []
    for variant in variants:
        matching = [item for items in report.results.values() for item in items if item.config_name == variant]
        mean_cost = sum(item.cost_per_decision for item in matching) / max(len(matching), 1)
        efficiency = overall_scores.get(variant, 0.0) / max(mean_cost, 1e-9)
        efficiency_rows.append((variant, efficiency))
    for variant, efficiency in sorted(efficiency_rows, key=lambda item: item[1], reverse=True):
        table3.add_row(variant, f"{efficiency:.4f}")

    console.print(table1)
    console.print(table2)
    console.print(table3)

    if impacts:
        if any(value >= 0 for value in impacts.values()):
            biggest_gain_variant = max(impacts, key=impacts.get)
        else:
            biggest_gain_variant = min(impacts, key=impacts.get)
    else:
        biggest_gain_variant = "no_spec"
    least_value_variant = min(impacts, key=lambda key: abs(impacts[key])) if impacts else "no_spec"
    tradeoff_variant = sorted(efficiency_rows, key=lambda item: item[1], reverse=True)[0][0] if efficiency_rows else "full"
    peak = max(overall_scores.values()) if overall_scores else 1.0
    tradeoff_score = overall_scores.get(tradeoff_variant, 0.0)
    tradeoff_pct = 100.0 * tradeoff_score / max(peak, 1e-9)
    tradeoff_cost = sum(
        item.cost_per_decision for items in report.results.values() for item in items if item.config_name == tradeoff_variant
    )
    winner_cost = sum(
        item.cost_per_decision for items in report.results.values() for item in items if item.config_name == report.overall_winner
    )
    tradeoff_cost_pct = 100.0 * (tradeoff_cost / max(winner_cost, 1e-9))

    gain_value = impacts.get(biggest_gain_variant, 0.0)
    least_value = impacts.get(least_value_variant, 0.0)
    if gain_value >= 0:
        sentence1 = f"The biggest single gain comes from {biggest_gain_variant} (+{gain_value:.4f} D/s versus removing it)."
    else:
        sentence1 = f"Removing {biggest_gain_variant} improves D/s the most (+{abs(gain_value):.4f} D/s over full)."
    if least_value >= 0:
        sentence2 = f"{least_value_variant} adds the least marginal value ({least_value:.4f} D/s drop when removed)."
    else:
        sentence2 = f"{least_value_variant} changes D/s the least ({abs(least_value):.4f} D/s improvement when removed)."

    console.print(sentence1)
    console.print(sentence2)
    console.print(
        f"For a speed/cost tradeoff, run config {tradeoff_variant} which achieves {tradeoff_pct:.1f}% of peak D/s at {tradeoff_cost_pct:.1f}% of compute."
    )

    Path("benchmark_results.json").write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
