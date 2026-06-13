from __future__ import annotations

import argparse
import uuid
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from pipeline import UnifiedPipeline, save_pipeline_result
from pipeline_config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--domain", default="math", choices=["math", "coding", "triage"])
    parser.add_argument("--config", default="balanced", choices=["speed", "balanced", "quality"])
    args = parser.parse_args()

    config = PipelineConfig.from_env()
    config.precision_budget = args.config
    pipeline = UnifiedPipeline(config)
    decision_task = pipeline._build_decision_task(args.task, args.domain)
    result = pipeline.run(decision_task, domain=args.domain)
    save_pipeline_result(result, config, decision_task, str(uuid.uuid4()), "../ds-benchmark/results.db")

    table = Table(title="Unified Pipeline Result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Answer", result.answer)
    table.add_row("Correct", str(result.ds_result.correct))
    table.add_row("Confidence", f"{result.ds_result.confidence_score:.2f}")
    table.add_row("Latency (ms)", f"{result.latency_ms:.2f}")
    table.add_row("Agents", str(result.n_agents))
    table.add_row("Precision stats", str(result.precision_stats))
    table.add_row("Cache stats", str(result.cache_stats))
    table.add_row("Config", str(asdict(config)))
    Console().print(table)


if __name__ == "__main__":
    main()
