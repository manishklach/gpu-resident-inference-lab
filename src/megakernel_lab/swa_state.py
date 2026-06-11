"""Sliding-window attention (SWA) state model for block speculative decoding.

Models the KV-or-state metadata that an SWA-style drafter would read as a
sliding window rather than the full context. This is a control-flow scaffold:
no real KV tensors, no real attention — just lifecycle counters and window
enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SlidingWindowState:
    """Tracks SWA-inspired window state for a block speculative runtime.

    The drafter reads only the last *window_size* tokens of committed context,
    modeling how SWA limits attention dependency to a fixed-size window.
    """

    window_size: int = 256
    kv_or_state_pages: dict[int, list[int]] = field(default_factory=dict)
    total_state_reads: int = 0
    total_state_writes: int = 0

    def get_window_tokens(self, committed_tokens: list[int], position: int) -> list[int]:
        """Return only the last *window_size* committed tokens."""
        if len(committed_tokens) <= self.window_size:
            return list(committed_tokens)
        return committed_tokens[-self.window_size :]

    def read_window_state(self, request_id: int, position: int) -> None:
        """Record a state read (increments counter, no real IO)."""
        self.total_state_reads += 1
        pages = self.kv_or_state_pages.setdefault(request_id, [])
        if not pages:
            pages.append(position)

    def update_window_state(self, request_id: int, committed_tokens: list[int]) -> None:
        """Update state after committing a block of tokens.

        Writes are only performed when tokens are actually committed,
        not when drafts are merely proposed.
        """
        self.total_state_writes += 1
        self.kv_or_state_pages[request_id] = list(committed_tokens)

    def report(self) -> dict[str, int]:
        """Return a summary of SWA state activity."""
        return {
            "window_size": self.window_size,
            "total_state_reads": self.total_state_reads,
            "total_state_writes": self.total_state_writes,
            "active_requests": len(self.kv_or_state_pages),
        }
