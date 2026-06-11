"""Tests for Adaptive Speculative Block Sizing (ASBS)."""

from megakernel_lab.backend import BackendLatencyConfig, CPUStubBackend
from megakernel_lab.config import RuntimeConfig
from megakernel_lab.runtime import PersistentDecodeRuntime
from megakernel_lab.spec_decode import (
    AcceptanceMask,
    AdaptiveBlockPolicy,
    update_block_size,
)
from megakernel_lab.state import RequestState


def _make_request(request_id: int = 0, max_new_tokens: int = 32) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_tokens=[1, 2, 3],
        target_tokens=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        max_new_tokens=max_new_tokens,
        eos_token_id=0,
        priority=0,
        layer_ids=[0, 1],
    )


class TestASBS:
    def test_asbs_rises_on_high_acceptance(self) -> None:
        req = _make_request()
        policy = AdaptiveBlockPolicy(
            min_block=1,
            max_block=8,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
        )
        for _ in range(10):
            mask = AcceptanceMask(accepted=[True, True, True, True])
            update_block_size(req, mask, policy, kv_pressure=False)

        assert req.current_block_size >= 6
        assert max(req.block_size_history) <= policy.max_block

    def test_asbs_falls_on_rejection(self) -> None:
        req = _make_request()
        policy = AdaptiveBlockPolicy(
            min_block=1,
            max_block=8,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
        )
        for _ in range(10):
            mask = AcceptanceMask(accepted=[False, False, False, False])
            update_block_size(req, mask, policy, kv_pressure=False)

        assert req.current_block_size == policy.min_block

    def test_asbs_capped_under_pressure(self) -> None:
        req = _make_request()
        policy = AdaptiveBlockPolicy(
            min_block=1,
            max_block=8,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
        )
        for _ in range(10):
            mask = AcceptanceMask(accepted=[True, True, True, True])
            update_block_size(req, mask, policy, kv_pressure=True)

        assert req.current_block_size <= policy.pressure_cap

    def test_asbs_ema_smoothing(self) -> None:
        req = _make_request()
        policy = AdaptiveBlockPolicy(
            min_block=1,
            max_block=8,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
        )
        for _ in range(10):
            mask = AcceptanceMask(accepted=[True, True, True, True])
            update_block_size(req, mask, policy, kv_pressure=False)

        high_block = req.current_block_size

        mask = AcceptanceMask(accepted=[False, False, False, False])
        update_block_size(req, mask, policy, kv_pressure=False)

        # A single rejection after many acceptances should not immediately drop to min_block
        assert req.current_block_size > policy.min_block
        assert req.current_block_size < high_block

    def test_asbs_integration_via_runtime(self) -> None:
        config = RuntimeConfig(
            block_size=4,
            max_new_tokens=16,
            eos_token_id=0,
            page_size=4,
            max_pages=64,
            num_layers=2,
            num_prefill_workers=1,
            num_decode_workers=1,
        )
        backend = CPUStubBackend(BackendLatencyConfig(sleep=False))
        adaptive = AdaptiveBlockPolicy(
            min_block=1,
            max_block=4,
            alpha=0.20,
            pressure_cap=2,
            high_accept_threshold=0.80,
            low_accept_threshold=0.50,
        )
        runtime = PersistentDecodeRuntime(
            config=config,
            backend=backend,
            adaptive_policy=adaptive,
        )
        req = _make_request(request_id=0, max_new_tokens=16)
        runtime.submit(req)
        results = runtime.run()

        assert len(results) == 1
        result = results[0]
        assert result.mean_block_size > 0
        assert result.min_block_size >= 1
        assert result.max_block_size <= 4
        assert len(result.traces) > 0
        for trace in result.traces:
            assert trace.block_size_used > 0
