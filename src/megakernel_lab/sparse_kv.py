"""Sparse KV page selection scaffold for persistent decode experiments.

This module does not implement real sparse attention math. It models the
control-flow choice of scoring resident KV blocks, selecting a deterministic
top-k subset, and passing only that subset into the logical decode stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig
from .kv_cache import KVCache, KVPage
from .state import RequestState


@dataclass(slots=True)
class SparseKVSelection:
    """Selection result for one decode iteration."""

    page_ids: list[int]
    total_blocks: int
    selected_blocks: int
    estimated_kv_bytes_read: int
    estimated_kv_bytes_saved: int


class SparseKVSelector:
    """Deterministically choose a top-k subset of committed KV pages."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def _score_page(self, page: KVPage, decode_tokens: int) -> float:
        distance = abs((page.token_start + page.token_count) - decode_tokens)
        return float((page.token_count * 10_000) - (distance * 100) - page.layer_id)

    def select(
        self,
        request: RequestState,
        kv_cache: KVCache,
        decode_step: int,
    ) -> SparseKVSelection:
        all_pages = kv_cache.pages_for_request(request.request_id)
        active_pages = [page for page in all_pages if page.is_committed and not page.is_draft]

        for page in all_pages:
            page.selected = False
            page.sparse_rank = -1
            if page.last_selected_step != decode_step:
                page.score = 0.0

        decode_tokens = len(request.prompt_tokens) + len(request.committed_tokens)
        for page in active_pages:
            page.score = self._score_page(page, decode_tokens)

        ranked_pages = sorted(
            active_pages,
            key=lambda page: (-page.score, page.layer_id, page.token_start, page.page_id),
        )
        top_k = max(0, min(self.config.sparse_top_k, len(ranked_pages)))
        selected_pages = ranked_pages[:top_k]

        for rank, page in enumerate(selected_pages):
            page.selected = True
            page.last_selected_step = decode_step
            page.sparse_rank = rank

        bytes_per_page = self.config.kv_block_size * self.config.bytes_per_token_per_layer()
        total_blocks = len(active_pages)
        selected_blocks = len(selected_pages)
        estimated_read = selected_blocks * bytes_per_page
        estimated_saved = max(total_blocks - selected_blocks, 0) * bytes_per_page

        return SparseKVSelection(
            page_ids=[page.page_id for page in selected_pages],
            total_blocks=total_blocks,
            selected_blocks=selected_blocks,
            estimated_kv_bytes_read=estimated_read,
            estimated_kv_bytes_saved=estimated_saved,
        )
