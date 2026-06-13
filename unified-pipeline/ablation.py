from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from _bootstrap import configure_paths
from pipeline import UnifiedPipeline
from pipeline_config import PipelineConfig


configure_paths()

from decisions_benchmark import DecisionTask  # type: ignore  # noqa: E402
from ds_scorer import compute_ds  # type: ignore  # noqa: E402


@dataclass
class AblationResult:
    config_name: str
    ds_score: float
    accuracy: float
    mean_latency_ms: float
    cost_per_decision: float


def _variant_config(base: PipelineConfig, variant: str) -> PipelineConfig:
    config = copy.deepcopy(base)
    if variant == "no_spec":
        config.spec_n_drafts = 1
    elif variant == "no_router":
        config.precision_budget = "quality"
    elif variant == "no_kvcache":
        config.kv_cpu_gb = 0.0
        config.kv_nvme_path = str(base.kv_nvme_path) + "_gpu_only"
    elif variant == "no_multiagent":
        config.n_agents = 1
    return config


def run_ablation(config: PipelineConfig, tasks: list[DecisionTask]) -> list[AblationResult]:
    variants = ["full", "no_spec", "no_router", "no_kvcache", "no_multiagent"]
    results: list[AblationResult] = []

    for variant in variants:
        started = time.perf_counter()
        variant_config = _variant_config(config, variant)
        pipeline = UnifiedPipeline(variant_config)
        verification_results = []
        difficulties = []
        total_flops = 0.0
        total_latency = 0.0

        for task in tasks:
            if variant == "no_spec":
                pipeline.spec_wrapper.run = lambda prompt, _task=task: f"Direct answer path for: {_task.prompt}"
            result = pipeline.run(task, domain=task.__class__.__name__.replace("DecisionTask", "").lower())
            verification_results.append(result.ds_result)
            difficulties.append(task.difficulty)
            total_flops += pipeline.last_compute_flops
            total_latency += result.latency_ms

        wall_time = time.perf_counter() - started
        report = compute_ds(verification_results, difficulties, wall_time)
        cost_per_decision = total_flops / max(report.weighted_correct, 1e-6) / 1e9
        results.append(
            AblationResult(
                config_name=variant,
                ds_score=report.ds_score,
                accuracy=report.accuracy,
                mean_latency_ms=total_latency / max(len(tasks), 1),
                cost_per_decision=cost_per_decision,
            )
        )

    return results

