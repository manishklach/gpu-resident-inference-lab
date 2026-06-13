from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from _bootstrap import configure_paths
from ablation import AblationResult, run_ablation
from pipeline_config import PipelineConfig


configure_paths()

from task_loader import load_tasks  # type: ignore  # noqa: E402


@dataclass
class BenchmarkReport:
    results: dict[str, list[AblationResult]]
    winner_per_domain: dict[str, str]
    overall_winner: str
    eta_seconds: float

    def to_json(self) -> dict:
        return {
            "results": {
                domain: [asdict(item) for item in items]
                for domain, items in self.results.items()
            },
            "winner_per_domain": self.winner_per_domain,
            "overall_winner": self.overall_winner,
            "eta_seconds": self.eta_seconds,
        }


async def run_full_benchmark(config: PipelineConfig, n_per_domain: int = 30, domains: list[str] | None = None) -> BenchmarkReport:
    domains = domains or ["math", "coding", "triage"]
    console = Console()
    results: dict[str, list[AblationResult]] = {}
    timings: dict[str, float] = {}

    async def _run_domain(domain: str) -> tuple[str, list[AblationResult], float]:
        started = time.perf_counter()
        tasks = load_tasks(domain, n_per_domain)
        domain_results = await asyncio.to_thread(run_ablation, config, tasks)
        elapsed = time.perf_counter() - started
        return domain, domain_results, elapsed

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running unified benchmark suite", total=len(domains))
        coroutines = [_run_domain(domain) for domain in domains]
        completed = []
        for future in asyncio.as_completed(coroutines):
            domain, domain_results, elapsed = await future
            results[domain] = domain_results
            timings[domain] = elapsed
            completed.append(domain)
            progress.advance(task_id)

    first_elapsed = next(iter(timings.values()), 0.0)
    eta_seconds = max(0.0, first_elapsed * max(len(domains) - 1, 0))
    winner_per_domain = {
        domain: max(items, key=lambda item: item.ds_score).config_name
        for domain, items in results.items()
    }

    aggregate_scores: dict[str, float] = {}
    for domain_results in results.values():
        for item in domain_results:
            aggregate_scores[item.config_name] = aggregate_scores.get(item.config_name, 0.0) + item.ds_score
    overall_winner = max(aggregate_scores, key=aggregate_scores.get) if aggregate_scores else "full"
    return BenchmarkReport(
        results=results,
        winner_per_domain=winner_per_domain,
        overall_winner=overall_winner,
        eta_seconds=eta_seconds,
    )

