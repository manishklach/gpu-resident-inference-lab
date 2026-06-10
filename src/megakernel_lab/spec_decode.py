"""Speculative decode building blocks."""

from dataclasses import dataclass

from .state import DecodeRequest


@dataclass(slots=True)
class AcceptancePolicy:
    """A deterministic acceptance policy for the simulator.

    `mismatch_stride` lets us inject occasional rejection tails so we can
    exercise the speculative control flow without a real verifier model.
    """

    mismatch_stride: int = 0


class DraftBlockProposer:
    """Produces a speculative block for the next decode iteration."""

    def __init__(self, block_size: int, eos_token_id: int) -> None:
        self.block_size = block_size
        self.eos_token_id = eos_token_id

    def propose(self, request: DecodeRequest) -> list[int]:
        """Draft up to `block_size` next tokens."""
        remaining = request.remaining_target()
        budget = request.token_budget_left()
        if not remaining or budget == 0:
            return [self.eos_token_id]

        count = min(self.block_size, len(remaining), budget)
        return remaining[:count]


class SpeculativeVerifier:
    """Verifies how much of a proposed block may be committed."""

    def __init__(self, policy: AcceptancePolicy) -> None:
        self.policy = policy

    def verify(self, request: DecodeRequest, proposal: list[int]) -> int:
        """Return the accepted prefix length of `proposal`."""
        expected = request.remaining_target()
        max_accept = min(len(proposal), len(expected), request.token_budget_left())
        accepted = 0

        for idx in range(max_accept):
            if proposal[idx] != expected[idx]:
                break
            if self.policy.mismatch_stride > 0:
                absolute_pos = len(request.committed_tokens) + idx + 1
                if absolute_pos % self.policy.mismatch_stride == 0 and idx < len(proposal) - 1:
                    accepted += 1
                    break
            accepted += 1

        return accepted
