"""Benchmark harness for the CPU persistent runtime simulator.

Provides four benchmark modes:
- serial_decode: block_size=1, no speculation
- speculative_decode: configurable block size with speculation
- forced_rejection: forces periodic draft rejections via mismatch_stride
- kv_pressure: intentionally undersized KV cache to trigger evictions

Exports expanded CSV with memory metrics for analysis.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from .backend import BackendLatencyConfig, CPUStubBackend
from .config import RuntimeConfig
from .runtime import PersistentDecodeRuntime
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import RequestState


class BenchmarkMode(str, Enum):
    """Available benchmark modes.

    These modes model the control-flow paths that the persistent mega-kernel
    will fuse into a single GPU-resident loop. Python benchmarks are
    control-flow simulations, not CUDA measurements.
    """

    SERIAL_DECODE = "serial_decode"
    SPECULATIVE_DECODE = "speculative_decode"
    FORCED_REJECTION = "forced_rejection"
    KV_PRESSURE = "kv_pressure"
    MEGA_KERNEL_SIM = "mega_kernel_sim"


@dataclass(slots=True)
class BenchmarkRecord:
    """One benchmark result row with expanded metrics."""

    batch_size: int
    block_size: int
    mean_ttft_ms: float
    p50_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    acceptance_rate: float
    kv_hit_rate: float
    live_kv_bytes: int
    pinned_kv_bytes: int
    eviction_count: int
    fragmentation_ratio: float
    mode: str


class BenchmarkRunner:
    """Runs deterministic batch and block-size sweeps over the simulator.

    Supports five modes:
    - serial_decode: block_size=1, no speculation
    - speculative_decode: configurable block size
    - forced_rejection: mismatch_stride forces periodic rejections
    - kv_pressure: small max_pages to trigger evictions
    - mega_kernel_sim: models the fused mega-kernel control path (draft → verify → commit loop)
      This is NOT a CUDA measurement; it simulates the control-flow architecture.

    Python benchmarks are control-flow simulations. The CUDA smoke test
    (make cuda-smoke) validates the staging path on real hardware.
    """

    def __init__(
        self,
        batch_sizes: list[int] | None = None,
        block_sizes: list[int] | None = None,
        modes: list[BenchmarkMode] | None = None,
    ) -> None:
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32]
        self.block_sizes = block_sizes or [1, 2, 4, 8]
        self.modes = modes or [BenchmarkMode.SPECULATIVE_DECODE]

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

    def _percentile(self, values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * pct / 100.0
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[f]
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def _run_single(
        self,
        batch_size: int,
        block_size: int,
        mode: BenchmarkMode,
    ) -> BenchmarkRecord:
        # Configure based on mode
        mismatch_stride = 0
        reject_draft_blocks = False
        max_pages = max(64, batch_size * 8)

        if mode == BenchmarkMode.SERIAL_DECODE:
            block_size = 1
        elif mode == BenchmarkMode.FORCED_REJECTION:
            mismatch_stride = 2
        elif mode == BenchmarkMode.MEGA_KERNEL_SIM:
            # Models the persistent mega-kernel's fused control path:
            # decode → draft → verify → commit → decode loop.
            # This runs on CPU only; it simulates the control-flow architecture.
            pass
        elif mode == BenchmarkMode.KV_PRESSURE:
            # Intentionally tight: each request ~6 pages (3 per layer * 2 layers)
            # for prompt=3 + max_new_tokens=8 with page_size=4.
            # Use multiplier 3 to leave room for prefill+decode but still trigger evictions.
            max_pages = max(8, batch_size * 3)  # Intentionally small

        config = RuntimeConfig(
            block_size=block_size,
            max_new_tokens=8,
            eos_token_id=0,
            page_size=4,
            max_pages=max_pages,
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
            verifier=SpeculativeVerifier(
                AcceptancePolicy(
                    mismatch_stride=mismatch_stride,
                    reject_draft_blocks=reject_draft_blocks,
                )
            ),
            backend=backend,
        )
        for request_id in range(batch_size):
            runtime.submit(self._build_request(request_id, config.max_new_tokens, config.eos_token_id))
        results = runtime.run()
        report = runtime.kv_cache.residency_report()

        # Compute TTFT
        mean_ttft_ms = sum(result.ttft_ms for result in results) / len(results)

        # Compute ITL percentiles
        itls = [value for result in results for value in result.inter_token_latency_ms]
        p50 = self._percentile(itls, 50.0)
        p95 = self._percentile(itls, 95.0)
        p99 = self._percentile(itls, 99.0)

        acceptance_rate = sum(result.acceptance_rate for result in results) / len(results)

        return BenchmarkRecord(
            batch_size=batch_size,
            block_size=block_size,
            mean_ttft_ms=mean_ttft_ms,
            p50_itl_ms=p50,
            p95_itl_ms=p95,
            p99_itl_ms=p99,
            acceptance_rate=acceptance_rate,
            kv_hit_rate=float(report["hit_rate"]),
            live_kv_bytes=int(report["live_kv_bytes"]),
            pinned_kv_bytes=int(report["pinned_kv_bytes"]),
            eviction_count=int(report["eviction_count"]),
            fragmentation_ratio=float(report["fragmentation_ratio"]),
            mode=mode.value,
        )

    def run(self, modes: list[BenchmarkMode] | None = None) -> pd.DataFrame:
        """Execute the sweep and return a dataframe.

        Args:
            modes: Override modes for this run. Defaults to self.modes.
        """
        active_modes = modes or self.modes
        records: list[BenchmarkRecord] = []
        for mode in active_modes:
            for batch_size in self.batch_sizes:
                effective_block_sizes = [1] if mode == BenchmarkMode.SERIAL_DECODE else self.block_sizes
                for block_size in effective_block_sizes:
                    record = self._run_single(batch_size, block_size, mode)
                    records.append(record)

        df = pd.DataFrame(asdict(record) for record in records)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(results_dir / f"bench_{timestamp}.csv", index=False)
        return df
