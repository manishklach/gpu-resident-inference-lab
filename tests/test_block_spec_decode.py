"""Tests for DFlash-style block speculative decode."""

from megakernel_lab.block_spec_decode import DFlashStyleDrafter


class TestDFlashStyleDrafter:
    def test_drafter_returns_block_size_tokens(self) -> None:
        drafter = DFlashStyleDrafter(block_size=8)
        draft = drafter.draft_block([1, 2, 3, 4, 5], position=5)
        assert len(draft.draft_tokens) == 8

    def test_drafter_uses_only_sliding_window(self) -> None:
        drafter = DFlashStyleDrafter(block_size=4, window_size=8)
        long_prefix = list(range(100))
        draft = drafter.draft_block(long_prefix, position=100)
        assert len(draft.draft_tokens) == 4

    def test_drafter_window_size_respected(self) -> None:
        drafter = DFlashStyleDrafter(block_size=4, window_size=8)
        short_prefix = [1, 2, 3]
        draft = drafter.draft_block(short_prefix, position=3)
        assert len(draft.draft_tokens) == 4

    def test_verifier_accepts_deterministic_prefix(self) -> None:
        drafter = DFlashStyleDrafter(block_size=8)
        committed = [10, 20, 30, 40, 50]
        draft = drafter.draft_block(committed, position=5)
        result = drafter.verify_block(committed, draft)
        assert result.accepted_prefix_len > 0
        assert len(result.accepted_tokens) == result.accepted_prefix_len

    def test_verifier_tracks_rejected_tail(self) -> None:
        drafter = DFlashStyleDrafter(block_size=8)
        committed = [100, 200, 300]
        draft = drafter.draft_block(committed, position=3)
        result = drafter.verify_block(committed, draft)
        total = len(result.accepted_tokens) + len(result.rejected_tokens)
        assert total == len(draft.draft_tokens)

    def test_draft_confidence_decreases_with_offset(self) -> None:
        drafter = DFlashStyleDrafter(block_size=8)
        draft = drafter.draft_block([1, 2, 3], position=3)
        assert len(draft.confidence) == 8
        for i in range(len(draft.confidence) - 1):
            assert draft.confidence[i] >= draft.confidence[i + 1] - 0.01
