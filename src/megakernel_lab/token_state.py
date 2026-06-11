"""Token lifecycle model for block speculative decoding.

Tracks the state of tokens through the draft → verify → accept →
commit → reject/resample pipeline. This makes the token lifecycle
explicit at each stage rather than mixing all tokens into a single
committed list.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TokenState:
    """Tracks tokens through their lifecycle in the speculative pipeline.

    Lifecycle:
      draft (proposed) → verify → accepted (committed) or rejected (discarded/resampled)
    """

    committed_tokens: list[int] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    accepted_tokens: list[int] = field(default_factory=list)
    rejected_tokens: list[int] = field(default_factory=list)
    current_position: int = 0

    def append_draft(self, tokens: list[int]) -> None:
        """Store a new block of draft tokens."""
        self.draft_tokens = list(tokens)

    def commit_prefix(self, n: int) -> None:
        """Move the first n draft tokens into accepted + committed."""
        if n <= 0 or n > len(self.draft_tokens):
            return
        accepted = self.draft_tokens[:n]
        self.accepted_tokens.extend(accepted)
        self.committed_tokens.extend(accepted)
        self.current_position += n

    def discard_rejected_tail(self) -> None:
        """Move remaining (unaccepted) draft tokens into rejected."""
        if len(self.draft_tokens) <= len(self.accepted_tokens):
            return
        tail = self.draft_tokens[len(self.accepted_tokens) :]
        self.rejected_tokens.extend(tail)

    def resample_rejected_tail(self) -> None:
        """Placeholder: move rejected tokens back into draft pool.

        In a real system this would trigger a fresh draft proposal.
        """

    def advance_position(self, n: int) -> None:
        """Manually advance the current position by n committed tokens."""
        self.current_position += n

    def is_complete(self, max_new_tokens: int) -> bool:
        """Check if generation has reached the target token count."""
        return len(self.committed_tokens) >= max_new_tokens

    def reset_draft(self) -> None:
        """Clear per-iteration draft state after commit step."""
        self.draft_tokens = []
        self.accepted_tokens = []
        self.rejected_tokens = []
