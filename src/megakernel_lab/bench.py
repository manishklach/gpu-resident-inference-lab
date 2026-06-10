"""Benchmark harness for the CPU persistent runtime simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backend import BackendLatencyConfig, CPUStubBackend
from .config import RuntimeConfig
from .runtime import PersistentDecodeRuntime
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import RequestState


@dataclass(slots=True)
class BenchmarkRecord:
    """One benchmark result row."""

    batch_size: int
    block_size: int
    mean_ttft_ms: float
    mean_itl_ms: float
    acceptance_rate: float
    kv_hit_rate: float


class BenchmarkRunner:
    """Runs deterministic batch and block-size sweeps over the simulator."""

    def __init__(
        self,
        batch_sizes: list[int] | None = None,
        block_sizes: list[int] | None = None,
    ) -> None:
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        self.block_sizes = block_sizes or [1, 2, 4, 8]

    def _build_request(self, request_id: int, max_new_tokens: int, eos_token_id: int) -> RequestState:
        base = 100 + request_id
        target = [base + idx for idx in range(4)] + [eos_token_id]
        return RequestState(
            request_id=request_id,
            prompt_tokens=[1, 2, 3],
            target_tokens=target,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            priority=request_id % 3,
            layer_ids=[0, 1],
        )

    def run(self) -> pd.DataFrame:
        """Execute the sweep and return a dataframe."""
        records: list[BenchmarkRecord] = []
        for batch_size in self.batch_sizes:
            for block_size in self.block_sizes:
                config = RuntimeConfig(
                    block_size=block_size,
                    max_new_tokens=8,
                    eos_token_id=0,
                    page_size=4,
                    max_pages=max(64, batch_size * 8),
                    num_layers=2,
                    num_prefill_workers=2,
                    num_decode_workers=2,
                )
                backend = CPUStubBackend(
                    BackendLatencyConfig(
                        prefill_ms=0.7,
                        decode_ms=0.2,
                        verify_ms=0.1,
                        copy_ms=0.02,
                        sleep=False,
                    )
                )
                runtime = PersistentDecodeRuntime(
                    config=config,
                    proposer=DraftBlockProposer(block_size=block_size, eos_token_id=config.eos_token_id),
                    verifier=SpeculativeVerifier(AcceptancePolicy()),
                    backend=backend,
                )
                for request_id in range(batch_size):
                    runtime.submit(self._build_request(request_id, config.max_new_tokens, config.eos_token_id))
                results = runtime.run()
                report = runtime.kv_cache.residency_report()
                mean_ttft_ms = sum(result.ttft_ms for result in results) / len(results)
                itls = [value for result in results for value in result.inter_token_latency_ms]
                mean_itl_ms = 0.0 if not itls else sum(itls) / len(itls)
                acceptance_rate = sum(result.acceptance_rate for result in results) / len(results)
                records.append(
                    BenchmarkRecord(
                        batch_size=batch_size,
                        block_size=block_size,
                        mean_ttft_ms=mean_ttft_ms,
                        mean_itl_ms=mean_itl_ms,
                        acceptance_rate=acceptance_rate,
                        kv_hit_rate=float(report["hit_rate"]),
                    )
                )
        df = pd.DataFrame(asdict(record) for record in records)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(results_dir / f"bench_{timestamp}.csv", index=False)
        return df
