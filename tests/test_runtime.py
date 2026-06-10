"""Tests for the persistent decode simulator."""

from megakernel_lab.config import RuntimeConfig
from megakernel_lab.runtime import PersistentDecodeRuntime
from megakernel_lab.spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from megakernel_lab.state import DecodeRequest


def make_runtime(block_size: int = 4, mismatch_stride: int = 0) -> PersistentDecodeRuntime:
    """Build a test runtime."""
    config = RuntimeConfig(block_size=block_size, max_new_tokens=16, eos_token_id=0)
    return PersistentDecodeRuntime(
        config=config,
        proposer=DraftBlockProposer(block_size=block_size, eos_token_id=config.eos_token_id),
        verifier=SpeculativeVerifier(AcceptancePolicy(mismatch_stride=mismatch_stride)),
    )


def test_full_acceptance_commits_complete_target() -> None:
    runtime = make_runtime(block_size=4, mismatch_stride=0)
    request = DecodeRequest(
        request_id=7,
        prompt_tokens=[1, 2],
        target_tokens=[10, 11, 12, 0],
        max_new_tokens=8,
        eos_token_id=0,
    )
    runtime.submit(request)
    result = runtime.run()[0]

    assert result.committed_tokens == [10, 11, 12, 0]
    assert runtime.kv_cache.positions_for(7) == [2, 3, 4, 5]


def test_rejection_tail_still_commits_prefix() -> None:
    runtime = make_runtime(block_size=4, mismatch_stride=2)
    request = DecodeRequest(
        request_id=8,
        prompt_tokens=[1],
        target_tokens=[20, 21, 22, 23, 0],
        max_new_tokens=8,
        eos_token_id=0,
    )
    runtime.submit(request)
    result = runtime.run()[0]

    assert result.committed_tokens[0] == 20
    assert len(result.traces) >= 2
    assert any(
        len(trace.accepted_tokens) < len(trace.proposed_tokens) for trace in result.traces[:-1]
    )
