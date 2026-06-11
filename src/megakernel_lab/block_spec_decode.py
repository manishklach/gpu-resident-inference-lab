"""DFlash-style block speculative decode building blocks.

This models the lifecycle of DFlash-style block drafting, not the actual
DFlash algorithm. All math is fake/deterministic. The purpose is to
demonstrate how block-level speculative decoding creates parallel work
that a persistent mega-kernel can keep resident.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BlockDraft:
    """A block of proposed draft tokens produced by the drafter."""

    request_id: int
    start_pos: int
    block_size: int
    draft_tokens: list[int] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)


@dataclass(slots=True)
class BlockVerifyResult:
    """Result of verifying a block of draft tokens against committed state."""

    request_id: int
    accepted_prefix_len: int
    accepted_tokens: list[int] = field(default_factory=list)
    rejected_tokens: list[int] = field(default_factory=list)
    resample_required: bool = False


class DFlashStyleDrafter:
    """Simulates a DFlash-style block drafter with SWA-inspired context window.

    This is a control-flow scaffold, not a real implementation of DFlash.
    The drafter uses deterministic fake math to produce block drafts and
    verification outcomes. All token values are derived from the committed
    prefix position and fixed seeds — no model inference occurs.
    """

    def __init__(
        self,
        block_size: int = 8,
        window_size: int = 256,
        vocab_size: int = 32000,
        deterministic_seed: int = 17,
    ) -> None:
        self.block_size = block_size
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.seed = deterministic_seed

    def _deterministic_token(self, context_hash: int, offset: int) -> int:
        h = (self.seed * 31 + context_hash * 7 + offset * 13) % self.vocab_size
        return h

    def _sliding_window_context(self, committed_tokens: list[int], position: int) -> list[int]:
        start = max(0, len(committed_tokens) - self.window_size)
        return committed_tokens[start:]

    def draft_block(self, committed_tokens: list[int], position: int) -> BlockDraft:
        context = self._sliding_window_context(committed_tokens, position)
        context_hash = sum(context) % self.vocab_size if context else 0
        draft_tokens: list[int] = []
        confidence: list[float] = []
        for offset in range(self.block_size):
            token = self._deterministic_token(context_hash, offset)
            draft_tokens.append(token)
            conf = max(0.1, 1.0 - (offset * 0.12))
            confidence.append(conf)
        return BlockDraft(
            request_id=0,
            start_pos=position,
            block_size=self.block_size,
            draft_tokens=draft_tokens,
            confidence=confidence,
        )

    def verify_block(self, committed_tokens: list[int], draft: BlockDraft) -> BlockVerifyResult:
        accepted: list[int] = []
        rejected: list[int] = []
        context = self._sliding_window_context(committed_tokens, draft.start_pos)
        context_hash = sum(context) % self.vocab_size if context else 0
        for offset, token in enumerate(draft.draft_tokens):
            expected = self._deterministic_token(context_hash, offset)
            if offset < len(draft.draft_tokens) // 2 or token == expected:
                accepted.append(token)
            else:
                rejected.append(token)
                break
        rejected.extend(draft.draft_tokens[len(accepted) + 1 :])
        return BlockVerifyResult(
            request_id=draft.request_id,
            accepted_prefix_len=len(accepted),
            accepted_tokens=accepted,
            rejected_tokens=rejected,
            resample_required=len(rejected) > 0,
        )
