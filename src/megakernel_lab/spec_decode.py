"""Speculative decode building blocks."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import AbstractKernelBackend
from .state import AcceptanceMask, RequestState


@dataclass(slots=True)
class AcceptancePolicy:
    """Deterministic controls for simulated speculative verification."""

    mismatch_stride: int = 0
    reject_draft_blocks: bool = False


@dataclass(slots=True)
class AdaptiveBlockPolicy:
    """Controls for Adaptive Speculative Block Sizing (ASBS).

    Each request maintains an EMA of its observed acceptance rate.
    This policy maps that rate to a block size for the next decode iteration.
    """

    min_block: int = 1
    max_block: int = 8
    alpha: float = 0.20
    pressure_cap: int = 2
    high_accept_threshold: float = 0.80
    low_accept_threshold: float = 0.50


def update_block_size(
    request: RequestState,
    mask: AcceptanceMask,
    policy: AdaptiveBlockPolicy,
    kv_pressure: bool,
) -> None:
    """Update the request's EMA acceptance rate and next block size.

    Pure function — no side effects beyond mutating *request*.
    """
    proposed = max(len(mask.accepted), 1)
    observed = mask.accepted_count / proposed

    request.ema_acceptance_rate = (
        policy.alpha * observed + (1.0 - policy.alpha) * request.ema_acceptance_rate
    )

    rate = request.ema_acceptance_rate
    if rate >= policy.high_accept_threshold:
        next_block = policy.max_block
    elif rate >= policy.low_accept_threshold:
        next_block = (policy.min_block + policy.max_block) // 2
    else:
        next_block = policy.min_block

    if kv_pressure and next_block > policy.pressure_cap:
        next_block = policy.pressure_cap

    request.current_block_size = max(next_block, policy.min_block)
    request.block_size_history.append(request.current_block_size)


class DraftBlockProposer:
    """Produces a speculative block for the next decode iteration."""

    def __init__(self, block_size: int, eos_token_id: int) -> None:
        self.block_size = block_size
        self.eos_token_id = eos_token_id

    def propose(self, request: RequestState, block_size: int | None = None) -> list[int]:
        """Draft up to `block_size` next tokens."""
        remaining = request.remaining_target()
        budget = request.token_budget_left()
        if not remaining or budget == 0:
            return [self.eos_token_id]

        width = self.block_size if block_size is None else block_size
        count = min(width, len(remaining), budget)
        return remaining[:count]


class SpeculativeVerifier:
    """Verifies how much of a proposed block may be committed."""

    def __init__(self, policy: AcceptancePolicy) -> None:
        self.policy = policy

    def verify(
        self,
        request: RequestState,
        proposal: list[int],
        kv_pages: list[int],
        backend: AbstractKernelBackend,
    ) -> AcceptanceMask:
        """Return the accepted prefix mask for `proposal`."""
        if self.policy.reject_draft_blocks and len(proposal) > 1:
            return AcceptanceMask(accepted=[False for _ in proposal], latency_ms=0.0)

        backend_mask = backend.speculative_verify(proposal, kv_pages)
        expected = request.remaining_target()
        result: list[bool] = []
        for idx, token in enumerate(proposal):
            accepted = idx < len(expected) and token == expected[idx]
            if idx < len(backend_mask.accepted):
                accepted = accepted and backend_mask.accepted[idx]
            if self.policy.mismatch_stride > 0:
                absolute_pos = len(request.committed_tokens) + idx + 1
                if absolute_pos % self.policy.mismatch_stride == 0 and idx < len(proposal) - 1:
                    accepted = True
                    result.append(accepted)
                    break
            if not accepted:
                result.append(False)
                break
            result.append(True)

        if len(result) < len(proposal):
            result.extend([False] * (len(proposal) - len(result)))
        return AcceptanceMask(accepted=result, latency_ms=backend_mask.latency_ms)
