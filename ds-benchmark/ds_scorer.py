from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from decisions_benchmark import VerificationResult


@dataclass
class DSReport:
    ds_score: float
    accuracy: float
    mean_latency_ms: float
    false_confidence_count: int
    weighted_correct: float


def compute_ds(
    results: list[VerificationResult], difficulties: list[float], total_wall_time_s: float
) -> DSReport:
    false_confidence_count = sum(
        1 for result in results if (not result.correct and result.confidence_score > 0.8)
    )
    weighted_correct = 0.0
    for result, difficulty in zip(results, difficulties):
        penalty = 2.0 if (not result.correct and result.confidence_score > 0.8) else 0.0
        weighted_correct += difficulty * int(result.correct) * max(0.0, 1.0 - penalty)
    accuracy = sum(int(result.correct) for result in results) / max(len(results), 1)
    mean_latency_ms = sum(result.latency_ms for result in results) / max(len(results), 1)
    ds_score = weighted_correct / max(total_wall_time_s, 1e-9)
    return DSReport(
        ds_score=ds_score,
        accuracy=accuracy,
        mean_latency_ms=mean_latency_ms,
        false_confidence_count=false_confidence_count,
        weighted_correct=weighted_correct,
    )


def print_report(report: DSReport) -> None:
    console = Console()
    table = Table(title="Decisions per Second Report")
    table.add_column("Metric")
    table.add_column("Value")
    style = "green" if report.ds_score > 1.0 else "yellow"
    table.add_row("D/s", f"[{style}]{report.ds_score:.4f}[/{style}]")
    table.add_row("Accuracy", f"{report.accuracy:.3f}")
    table.add_row("Mean latency (ms)", f"{report.mean_latency_ms:.3f}")
    table.add_row("False confidence count", str(report.false_confidence_count))
    table.add_row("Weighted correct", f"{report.weighted_correct:.3f}")
    console.print(table)
