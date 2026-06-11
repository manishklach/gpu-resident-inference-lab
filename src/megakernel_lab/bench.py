"""Benchmark harness for the CPU persistent runtime simulator.

Provides four benchmark modes:
- serial_decode: block_size=1, no speculation
- speculative_decode: configurable block size with speculation
- forced_rejection: forces periodic draft rejections via mismatch_stride
- kv_pressure: intentionally undersized KV cache to trigger evictions

Exports expanded CSV with memory metrics for analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from .backend import BackendLatencyConfig, CPUStubBackend
from .block_runtime import BlockSpeculativeRuntime
from .block_spec_decode import DFlashStyleDrafter
from .config import RuntimeConfig
from .runtime import PersistentDecodeRuntime
from .spec_decode import (
    AcceptancePolicy,
    AdaptiveBlockPolicy,
    DraftBlockProposer,
    SpeculativeVerifier,
)
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

    AUTOREGRESSIVE_SERIAL = "autoregressive_serial"
    BLOCK_SPECULATIVE = "block_speculative"
    BLOCK_SPECULATIVE_PERSISTENT_SIM = "block_speculative_persistent_sim"
    BLOCK_SPECULATIVE_HOST_ORCHESTRATED = "block_speculative_host_orchestrated"


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

    iterations: int = 0
    draft_blocks: int = 0
    total_draft_tokens: int = 0
    accepted_tokens_block: int = 0
    rejected_tokens: int = 0
    average_accepted_prefix_len: float = 0.0
    state_reads: int = 0
    state_writes: int = 0
    host_kernel_launches: int = 0
    host_synchronizations: int = 0
    simulated_tokens_per_iteration: float = 0.0
    mean_block_size: float = 0.0
    min_block_size: int = 0
    max_block_size: int = 0
    block_size_variance: float = 0.0


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

    def _build_request(
        self, request_id: int, max_new_tokens: int, eos_token_id: int
    ) -> RequestState:
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
        adaptive_policy = AdaptiveBlockPolicy(
            min_block=1,
            max_block=block_size,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
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
            adaptive_policy=adaptive_policy,
        )
        for request_id in range(batch_size):
            runtime.submit(
                self._build_request(request_id, config.max_new_tokens, config.eos_token_id)
            )
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

        block_sizes = [r.mean_block_size for r in results]
        mean_bs = sum(block_sizes) / len(block_sizes) if block_sizes else 0.0
        min_bs = min(r.min_block_size for r in results) if results else 0
        max_bs = max(r.max_block_size for r in results) if results else 0
        if block_sizes:
            variance = sum((x - mean_bs) ** 2 for x in block_sizes) / len(block_sizes)
        else:
            variance = 0.0

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
            mean_block_size=mean_bs,
            min_block_size=min_bs,
            max_block_size=max_bs,
            block_size_variance=variance,
        )

    def _run_block_single(
        self,
        batch_size: int,
        block_size: int,
        window_size: int,
        mode: BenchmarkMode,
        max_new_tokens: int = 64,
    ) -> BenchmarkRecord:
        drafter = DFlashStyleDrafter(
            block_size=block_size,
            window_size=window_size,
        )
        runtime = BlockSpeculativeRuntime(drafter=drafter)

        if mode == BenchmarkMode.AUTOREGRESSIVE_SERIAL:
            drafter.block_size = 1
        if mode == BenchmarkMode.BLOCK_SPECULATIVE_PERSISTENT_SIM:
            pass
        if mode == BenchmarkMode.BLOCK_SPECULATIVE_HOST_ORCHESTRATED:
            pass

        total_accepted = 0
        total_iterations = 0
        total_draft_blocks = 0
        total_draft_tokens = 0
        total_rejected = 0
        total_state_reads = 0
        total_state_writes = 0

        for _ in range(batch_size):
            runtime.token_state.committed_tokens = []
            runtime.token_state.current_position = 0
            runtime.swa.total_state_reads = 0
            runtime.swa.total_state_writes = 0
            metrics = runtime.run(max_new_tokens=max_new_tokens)
            total_accepted += metrics.accepted_tokens
            total_iterations += metrics.iterations
            total_draft_blocks += metrics.draft_blocks
            total_draft_tokens += metrics.total_draft_tokens
            total_rejected += metrics.rejected_tokens
            total_state_reads += metrics.state_reads
            total_state_writes += metrics.state_writes

        avg_acc_prefix = 0.0
        if total_draft_blocks > 0:
            avg_acc_prefix = total_accepted / total_draft_blocks
        num_total = total_accepted + total_rejected
        acceptance_rate = total_accepted / num_total if num_total > 0 else 0.0

        if mode == BenchmarkMode.BLOCK_SPECULATIVE_PERSISTENT_SIM:
            launches = 1
            syncs = 1
        elif mode == BenchmarkMode.BLOCK_SPECULATIVE_HOST_ORCHESTRATED:
            launches = total_iterations * 4
            syncs = total_iterations * 4
        else:
            launches = total_iterations
            syncs = total_iterations

        tokens_per_iter = total_accepted / total_iterations if total_iterations > 0 else 0.0

        return BenchmarkRecord(
            batch_size=batch_size,
            block_size=block_size,
            mean_ttft_ms=0.0,
            p50_itl_ms=0.0,
            p95_itl_ms=0.0,
            p99_itl_ms=0.0,
            acceptance_rate=acceptance_rate,
            kv_hit_rate=0.0,
            live_kv_bytes=0,
            pinned_kv_bytes=0,
            eviction_count=0,
            fragmentation_ratio=0.0,
            mode=mode.value,
            iterations=total_iterations,
            draft_blocks=total_draft_blocks,
            total_draft_tokens=total_draft_tokens,
            accepted_tokens_block=total_accepted,
            rejected_tokens=total_rejected,
            average_accepted_prefix_len=avg_acc_prefix,
            state_reads=total_state_reads,
            state_writes=total_state_writes,
            host_kernel_launches=launches,
            host_synchronizations=syncs,
            simulated_tokens_per_iteration=tokens_per_iter,
        )

    def run(self, modes: list[BenchmarkMode] | None = None) -> pd.DataFrame:
        """Execute the sweep and return a dataframe.

        Args:
            modes: Override modes for this run. Defaults to self.modes.
        """
        active_modes = modes or self.modes
        records: list[BenchmarkRecord] = []
        for mode in active_modes:
            if mode in (
                BenchmarkMode.AUTOREGRESSIVE_SERIAL,
                BenchmarkMode.BLOCK_SPECULATIVE,
                BenchmarkMode.BLOCK_SPECULATIVE_PERSISTENT_SIM,
                BenchmarkMode.BLOCK_SPECULATIVE_HOST_ORCHESTRATED,
            ):
                for batch_size in self.batch_sizes:
                    effective_block_sizes = (
                        [1] if mode == BenchmarkMode.AUTOREGRESSIVE_SERIAL else self.block_sizes
                    )
                    for block_size in effective_block_sizes:
                        record = self._run_block_single(batch_size, block_size, 256, mode)
                        records.append(record)
            else:
                for batch_size in self.batch_sizes:
                    effective_block_sizes = (
                        [1] if mode == BenchmarkMode.SERIAL_DECODE else self.block_sizes
                    )
                    for block_size in effective_block_sizes:
                        record = self._run_single(batch_size, block_size, mode)
                        records.append(record)

        df = pd.DataFrame(asdict(record) for record in records)
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(results_dir / f"bench_{timestamp}.csv", index=False)
        return df
