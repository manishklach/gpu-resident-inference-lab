"""Tests for token lifecycle model."""

from megakernel_lab.token_state import TokenState


class TestTokenState:
    def test_draft_tokens_are_temporary(self) -> None:
        state = TokenState()
        state.append_draft([100, 101, 102, 103])
        assert len(state.draft_tokens) == 4
        assert len(state.committed_tokens) == 0

    def test_commit_prefix_commits_only_accepted(self) -> None:
        state = TokenState()
        state.append_draft([100, 101, 102, 103])
        state.commit_prefix(2)
        assert state.committed_tokens == [100, 101]
        assert state.current_position == 2

    def test_rejected_tail_is_discarded(self) -> None:
        state = TokenState()
        state.append_draft([100, 101, 102, 103])
        state.commit_prefix(2)
        state.discard_rejected_tail()
        assert state.rejected_tokens == [102, 103]

    def test_current_position_advances_correctly(self) -> None:
        state = TokenState()
        state.append_draft([10, 20, 30, 40])
        state.commit_prefix(3)
        assert state.current_position == 3

    def test_position_advance_manual(self) -> None:
        state = TokenState()
        state.advance_position(5)
        assert state.current_position == 5

    def test_is_complete(self) -> None:
        state = TokenState()
        state.append_draft([1, 2, 3, 4])
        state.commit_prefix(4)
        assert state.is_complete(4)
        assert not state.is_complete(5)

    def test_reset_draft_clears_iteration_state(self) -> None:
        state = TokenState()
        state.append_draft([1, 2, 3])
        state.commit_prefix(2)
        state.discard_rejected_tail()
        state.reset_draft()
        assert state.draft_tokens == []
        assert state.accepted_tokens == []
        assert state.rejected_tokens == []
        assert state.committed_tokens == [1, 2]
