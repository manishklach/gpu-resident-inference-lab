"""Block speculative runtime — simulates DFlash-style block decoding.

The runtime models the control loop:
  read SWA/KV-or-state window
  draft block of tokens in parallel
  verify block
  commit accepted prefix
  discard/resample rejected tail
  update token state
  update SWA/KV-or-state
  schedule next block

All math is fake/deterministic. This is a control-flow scaffold showing
how block-level speculation turns decode from one-token steps into
block-level iterations — which a persistent mega-kernel can keep resident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .block_spec_decode import DFlashStyleDrafter
from .swa_state import SlidingWindowState
from .token_state import TokenState


@dataclass(slots=True)
class BlockRuntimeMetrics:
    """Metrics collected during a block speculative runtime run."""

    draft_blocks: int = 0
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    average_accepted_prefix_len: float = 0.0
    acceptance_rate: float = 0.0
    state_reads: int = 0
    state_writes: int = 0
    committed_tokens: int = 0
    iterations: int = 0

    def finalize(self) -> None:
        if self.draft_blocks > 0:
            self.average_accepted_prefix_len = self.accepted_tokens / self.draft_blocks
        total = self.accepted_tokens + self.rejected_tokens
        if total > 0:
            self.acceptance_rate = self.accepted_tokens / total


@dataclass(slots=True)
class BlockSpeculativeRuntime:
    """Simulates a block speculative decode loop.

    One iteration == one complete block: draft → verify → commit → update.
    """

    drafter: DFlashStyleDrafter = field(default_factory=DFlashStyleDrafter)
    swa: SlidingWindowState = field(default_factory=SlidingWindowState)
    token_state: TokenState = field(default_factory=TokenState)
    metrics: BlockRuntimeMetrics = field(default_factory=BlockRuntimeMetrics)

    def run(
        self,
        max_new_tokens: int = 64,
    ) -> BlockRuntimeMetrics:
        while not self.token_state.is_complete(max_new_tokens):
            position = self.token_state.current_position
            committed = self.token_state.committed_tokens

            self.swa.get_window_tokens(committed, position)
            self.swa.read_window_state(0, position)
            self.metrics.state_reads += 1

            draft = self.drafter.draft_block(committed, position)
            self.token_state.append_draft(draft.draft_tokens)
            self.metrics.draft_blocks += 1
            self.metrics.total_draft_tokens += len(draft.draft_tokens)

            verify = self.drafter.verify_block(committed, draft)
            accepted = verify.accepted_tokens
            rejected = verify.rejected_tokens

            if accepted:
                self.token_state.commit_prefix(len(accepted))
                self.token_state.discard_rejected_tail()
                self.token_state.advance_position(len(accepted))
                self.swa.update_window_state(0, self.token_state.committed_tokens)
                self.metrics.state_writes += 1
                self.metrics.accepted_tokens += len(accepted)
                self.metrics.rejected_tokens += len(rejected)
                self.metrics.committed_tokens = len(self.token_state.committed_tokens)
            else:
                self.token_state.discard_rejected_tail()
                self.metrics.rejected_tokens += len(rejected)

            self.token_state.reset_draft()
            self.metrics.iterations += 1

        self.metrics.finalize()
        return self.metrics
