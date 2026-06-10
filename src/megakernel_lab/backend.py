"""Abstract backend interface for future CPU and CUDA kernel implementations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .state import AcceptanceMask, KVSnapshot, RequestState, TokenLogits


@dataclass(slots=True)
class BackendLatencyConfig:
    """Per-operation latency knobs for the CPU stub backend."""

    prefill_ms: float = 1.0
    decode_ms: float = 0.25
    verify_ms: float = 0.15
    copy_ms: float = 0.05
    sleep: bool = False


class AbstractKernelBackend(ABC):
    """Contract that future CUDA kernels must satisfy without runtime changes."""

    @abstractmethod
    def prefill(self, token_ids: list[int], kv_pages: list[int]) -> KVSnapshot:
        """Build an initial KV snapshot from the prompt tokens."""

    @abstractmethod
    def decode_step(self, request_state: RequestState, kv_pages: list[int]) -> TokenLogits:
        """Run one decode step for the active request."""

    @abstractmethod
    def speculative_verify(self, draft_tokens: list[int], kv_pages: list[int]) -> AcceptanceMask:
        """Verify a draft token block against the model state."""

    @abstractmethod
    def copy_kv_pages(self, src_pages: list[int], dst_pages: list[int]) -> None:
        """Copy KV data between page regions."""


class CPUStubBackend(AbstractKernelBackend):
    """A deterministic CPU backend used to simulate kernel contention and timing."""

    def __init__(self, latency: BackendLatencyConfig | None = None) -> None:
        self.latency = latency or BackendLatencyConfig()
        self._active_request: RequestState | None = None

    def _maybe_sleep(self, latency_ms: float) -> None:
        if self.latency.sleep and latency_ms > 0:
            time.sleep(latency_ms / 1000.0)

    def prefill(self, token_ids: list[int], kv_pages: list[int]) -> KVSnapshot:
        latency_ms = self.latency.prefill_ms + 0.01 * len(token_ids)
        self._maybe_sleep(latency_ms)
        return KVSnapshot(
            request_id=-1,
            layer_page_map={},
            prompt_length=len(token_ids),
            latency_ms=latency_ms,
        )

    def decode_step(self, request_state: RequestState, kv_pages: list[int]) -> TokenLogits:
        self._active_request = request_state
        remaining = request_state.remaining_target()
        width = max(1, min(request_state.token_budget_left(), len(remaining)))
        candidate_tokens = remaining[:width] or [request_state.eos_token_id]
        scores = [1.0 - (idx * 0.01) for idx in range(len(candidate_tokens))]
        latency_ms = self.latency.decode_ms + 0.01 * len(kv_pages)
        self._maybe_sleep(latency_ms)
        return TokenLogits(candidate_tokens=candidate_tokens, scores=scores, latency_ms=latency_ms)

    def speculative_verify(self, draft_tokens: list[int], kv_pages: list[int]) -> AcceptanceMask:
        if self._active_request is None:
            raise RuntimeError("decode_step must run before speculative_verify")
        expected = self._active_request.remaining_target()
        accepted = [
            idx < len(expected) and draft_tokens[idx] == expected[idx]
            for idx in range(len(draft_tokens))
        ]
        latency_ms = self.latency.verify_ms + 0.005 * len(kv_pages)
        self._maybe_sleep(latency_ms)
        return AcceptanceMask(accepted=accepted, latency_ms=latency_ms)

    def copy_kv_pages(self, src_pages: list[int], dst_pages: list[int]) -> None:
        if len(src_pages) != len(dst_pages):
            raise ValueError("source and destination page lists must align")
        latency_ms = self.latency.copy_ms + 0.001 * len(src_pages)
        self._maybe_sleep(latency_ms)
