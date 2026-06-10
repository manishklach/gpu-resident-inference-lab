"""Tests for the persistent decode runtime and worker model."""

from megakernel_lab.backend import CPUStubBackend
from megakernel_lab.config import RuntimeConfig
from megakernel_lab.runtime import PersistentDecodeRuntime
from megakernel_lab.spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from megakernel_lab.state import RequestState


def make_runtime(
    *,
    block_size: int = 4,
    mismatch_stride: int = 0,
    reject_draft_blocks: bool = False,
    max_pages: int = 32,
    num_prefill_workers: int = 1,
    num_decode_workers: int = 1,
) -> PersistentDecodeRuntime:
    """Build a test runtime."""
    config = RuntimeConfig(
        block_size=block_size,
        max_new_tokens=16,
        eos_token_id=0,
        page_size=2,
        max_pages=max_pages,
        num_layers=2,
        num_prefill_workers=num_prefill_workers,
        num_decode_workers=num_decode_workers,
    )
    return PersistentDecodeRuntime(
        config=config,
        proposer=DraftBlockProposer(block_size=block_size, eos_token_id=config.eos_token_id),
        verifier=SpeculativeVerifier(
            AcceptancePolicy(
                mismatch_stride=mismatch_stride,
                reject_draft_blocks=reject_draft_blocks,
            )
        ),
        backend=CPUStubBackend(),
    )


def test_full_acceptance_commits_complete_target() -> None:
    runtime = make_runtime(block_size=4)
    request = RequestState(
        request_id=7,
        prompt_tokens=[1, 2],
        target_tokens=[10, 11, 12, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0, 1],
    )
    runtime.submit(request)
    result = runtime.run()[0]

    assert result.committed_tokens == [10, 11, 12, 0]
    assert result.acceptance_rate > 0.9


def test_rejection_tail_still_commits_prefix() -> None:
    runtime = make_runtime(block_size=4, mismatch_stride=2)
    request = RequestState(
        request_id=8,
        prompt_tokens=[1],
        target_tokens=[20, 21, 22, 23, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0, 1],
    )
    runtime.submit(request)
    result = runtime.run()[0]

    assert result.committed_tokens[0] == 20
    assert len(result.traces) >= 2
    assert any(
        len(trace.accepted_tokens) < len(trace.proposed_tokens) for trace in result.traces[:-1]
    )


def test_draft_rejection_cascade_falls_back_to_serial() -> None:
    runtime = make_runtime(block_size=4, reject_draft_blocks=True)
    request = RequestState(
        request_id=9,
        prompt_tokens=[5, 6],
        target_tokens=[30, 31, 32, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0, 1],
    )
    runtime.submit(request)
    result = runtime.run()[0]

    assert result.committed_tokens == [30, 31, 32, 0]
    assert any(trace.used_fallback_serial for trace in result.traces)
    assert all(len(trace.accepted_tokens) <= 1 for trace in result.traces)


def test_concurrent_request_preemption_restores_state() -> None:
    runtime = make_runtime(block_size=2, num_prefill_workers=1, num_decode_workers=1)
    low_priority = RequestState(
        request_id=11,
        prompt_tokens=[1],
        target_tokens=[40, 41, 42, 43, 0],
        max_new_tokens=8,
        eos_token_id=0,
        priority=0,
        layer_ids=[0, 1],
    )
    high_priority = RequestState(
        request_id=12,
        prompt_tokens=[2],
        target_tokens=[50, 51, 0],
        max_new_tokens=8,
        eos_token_id=0,
        priority=10,
        layer_ids=[0, 1],
    )
    runtime.submit(low_priority)
    runtime.submit(high_priority)
    results = {result.request_id: result for result in runtime.run()}

    assert results[12].committed_tokens == [50, 51, 0]
    assert results[11].committed_tokens == [40, 41, 42, 43, 0]
    assert results[11].was_preempted or results[12].was_preempted


def test_worker_handoff_preserves_page_ownership() -> None:
    runtime = make_runtime(block_size=2, num_prefill_workers=1, num_decode_workers=1)
    request = RequestState(
        request_id=21,
        prompt_tokens=[7, 8, 9],
        target_tokens=[60, 61, 0],
        max_new_tokens=8,
        eos_token_id=0,
        priority=1,
        layer_ids=[0, 1],
    )
    runtime.submit(request)
    runtime.run()

    assert runtime.worker_pool.handoffs
    handoff = runtime.worker_pool.handoffs[0]
    assert handoff.request_id == 21
    assert handoff.prefill_worker_id.startswith("prefill-")
    assert handoff.decode_worker_id.startswith("decode-")
    assert handoff.page_ids
