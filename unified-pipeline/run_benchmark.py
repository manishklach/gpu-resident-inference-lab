from __future__ import annotations

import argparse
import asyncio

from benchmark_suite import run_full_benchmark
from pipeline_config import PipelineConfig
from report_generator import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--domains", nargs="+", default=["math", "coding", "triage"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig.from_env()
    n = 5 if args.quick else args.n
    domains = args.domains[:1] if args.quick and len(args.domains) == 1 else args.domains
    report = asyncio.run(run_full_benchmark(config, n_per_domain=n, domains=domains))
    generate_report(report)


if __name__ == "__main__":
    main()
