"""Paged KV-cache planner for the persistent runtime simulation.

This module tracks paged KV residency with:
- Layer-aware page tables
- Pinned pages for active decode requests (protected from eviction)
- LRU eviction under memory pressure
- Explicit distinction between committed and speculative (draft) pages
- Memory accounting: total live bytes, pinned bytes, evicted bytes, fragmentation
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil

from .config import RuntimeConfig
from .state import RequestState


@dataclass(slots=True)
class KVPage:
    """Represents one physical KV page.

    A page covers a contiguous range of token positions for one layer of one request.
    Pages may be pinned (active decode), committed (finalized tokens), or draft
    (speculative, discardable on rejection).
    """

    page_id: int
    request_id: int
    layer_id: int
    token_start: int
    token_count: int
    pinned: bool = False
    is_committed: bool = False
    is_draft: bool = False


class KVCache:
    """Tracks paged KV residency with LRU eviction, pinning, and memory accounting.

    Supports explicit distinction between committed and draft (speculative) pages.
    Draft pages are released on rejection; committed pages are kept.

    Memory accounting tracks:
    - total_live_kv_bytes: bytes currently resident in the cache
    - pinned_kv_bytes: bytes protected from eviction (active decode)
    - evicted_kv_bytes: total bytes evicted across all requests
    - fragmentation_ratio: fraction of page capacity that is wasted (partial pages)
    """

    def __init__(
        self,
        page_size: int = 4,
        max_pages: int = 64,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_pages = max_pages
        self._config = config
        self._next_page_id = 0
        self._pages: dict[int, KVPage] = {}
        self._page_table: dict[tuple[int, int], list[int]] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._eviction_count = 0
        self._access_count = 0
        self._hit_count = 0
        self._total_live_kv_bytes = 0
        self._pinned_kv_bytes = 0
        self._evicted_kv_bytes = 0
        self._total_capacity_tokens = 0
        self._used_capacity_tokens = 0

    def _bytes_per_token_per_layer(self) -> int:
        """Bytes of KV per token per layer.

        Falls back to a default if no config is provided.
        Each physical page covers one layer, so this is the base unit for page accounting.
        """
        if self._config is not None:
            return self._config.bytes_per_token_per_layer()
        # Default: 12 heads * 64 dim * 2 bytes * 2 tensors = 3072 bytes/token/layer
        return 12 * 64 * 2 * 2

    def _bytes_per_page(self) -> int:
        """Bytes of KV per physical page.

        Each page stores page_size tokens for one layer.
        """
        return self.page_size * self._bytes_per_token_per_layer()

    def _required_pages(self, token_count: int) -> int:
        return 0 if token_count <= 0 else ceil(token_count / self.page_size)

    def _pages_needed_for_layer(
        self, request: RequestState, layer_id: int, num_new_tokens: int
    ) -> int:
        key = (request.request_id, layer_id)
        existing_pages = len(self._page_table.get(key, []))
        current_tokens = len(request.prompt_tokens) + len(request.committed_tokens)
        new_total = current_tokens + num_new_tokens
        required_pages = self._required_pages(new_total)
        return max(required_pages - existing_pages, 0)

    def can_allocate(self, request: RequestState, num_new_tokens: int) -> bool:
        """Return whether enough pages could be available for the allocation."""
        needed = sum(
            self._pages_needed_for_layer(request, layer_id, num_new_tokens)
            for layer_id in request.layer_ids
        )
        if needed == 0:
            return True
        free_pages = self.max_pages - len(self._pages)
        evictable = sum(1 for page in self._pages.values() if not page.pinned)
        return free_pages + evictable >= needed

    def evict_lru(self, n_pages: int) -> list[int]:
        """Evict up to `n_pages` least-recently-used unpinned pages.

        Draft (speculative) pages are evicted before committed pages when possible,
        since they can be safely discarded. Pinned pages are never evicted.
        """
        evicted: list[int] = []
        # First pass: evict draft pages
        for page_id in list(self._lru.keys()):
            if len(evicted) >= n_pages:
                break
            page = self._pages[page_id]
            if page.pinned or not page.is_draft:
                continue
            self._evict_single_page(page_id)
            evicted.append(page_id)
        # Second pass: evict committed pages if we still need more
        for page_id in list(self._lru.keys()):
            if len(evicted) >= n_pages:
                break
            page = self._pages[page_id]
            if page.pinned or page.is_draft:
                continue
            self._evict_single_page(page_id)
            evicted.append(page_id)
        return evicted

    def _evict_single_page(self, page_id: int) -> None:
        """Remove one page from the cache and update memory accounting."""
        page = self._pages.pop(page_id)
        self._lru.pop(page_id, None)
        key = (page.request_id, page.layer_id)
        page_ids = self._page_table.get(key, [])
        if page_id in page_ids:
            page_ids.remove(page_id)
            if not page_ids:
                self._page_table.pop(key, None)
        # Track evicted bytes
        page_bytes = page.token_count * self._bytes_per_token_per_layer()
        self._total_live_kv_bytes -= page_bytes
        self._evicted_kv_bytes += page_bytes
        self._used_capacity_tokens -= page.token_count
        self._total_capacity_tokens -= self.page_size
        if page.pinned:
            self._pinned_kv_bytes -= page_bytes
        self._eviction_count += 1

    def is_under_pressure(self) -> bool:
        """Return True if live page count exceeds 85% of max_pages.

        Used by the adaptive block sizing policy to cap draft block size
        when the KV cache is near capacity.
        """
        live = len(self._pages)
        return live > self.max_pages * 0.85

    def ensure_capacity(self, request: RequestState, num_new_tokens: int) -> None:
        """Evict as needed and fail if the allocation still cannot fit."""
        needed = sum(
            self._pages_needed_for_layer(request, layer_id, num_new_tokens)
            for layer_id in request.layer_ids
        )
        free_pages = self.max_pages - len(self._pages)
        deficit = max(needed - free_pages, 0)
        if deficit > 0:
            self.evict_lru(deficit)
        if not self.can_allocate(request, num_new_tokens):
            raise RuntimeError("KV cache cannot allocate pages for request")

    def _allocate_one_page(
        self,
        request_id: int,
        layer_id: int,
        token_start: int,
        token_count: int,
        pinned: bool,
        is_committed: bool = False,
        is_draft: bool = False,
    ) -> int:
        page_id = self._next_page_id
        self._next_page_id += 1
        page = KVPage(
            page_id=page_id,
            request_id=request_id,
            layer_id=layer_id,
            token_start=token_start,
            token_count=token_count,
            pinned=pinned,
            is_committed=is_committed,
            is_draft=is_draft,
        )
        self._pages[page_id] = page
        self._lru[page_id] = None
        self._lru.move_to_end(page_id)
        self._page_table.setdefault((request_id, layer_id), []).append(page_id)
        # Track memory accounting: add live bytes for this page
        # Each page covers one layer, so bytes = token_count * bytes_per_token_per_layer
        page_bytes = token_count * self._bytes_per_token_per_layer()
        self._total_live_kv_bytes += page_bytes
        self._used_capacity_tokens += token_count
        self._total_capacity_tokens += self.page_size
        if pinned:
            self._pinned_kv_bytes += page_bytes
        return page_id

    def allocate_for_request(
        self, request: RequestState, total_tokens: int, pinned: bool
    ) -> dict[int, list[int]]:
        """Ensure the request has enough pages for `total_tokens` across layers."""
        pages_needed = 0
        for layer_id in request.layer_ids:
            key = (request.request_id, layer_id)
            existing_pages = len(self._page_table.get(key, []))
            required_pages = self._required_pages(total_tokens)
            pages_needed += max(required_pages - existing_pages, 0)
        free_pages = self.max_pages - len(self._pages)
        deficit = max(pages_needed - free_pages, 0)
        if deficit > 0:
            self.evict_lru(deficit)
        if (self.max_pages - len(self._pages)) < pages_needed:
            raise RuntimeError("KV cache cannot allocate pages for request")
        layer_page_map: dict[int, list[int]] = {}
        for layer_id in request.layer_ids:
            key = (request.request_id, layer_id)
            page_ids = list(self._page_table.get(key, []))
            existing_capacity = len(page_ids) * self.page_size
            if total_tokens > existing_capacity:
                missing = total_tokens - existing_capacity
                pages_needed = ceil(missing / self.page_size)
                token_start = existing_capacity
                for _ in range(pages_needed):
                    token_count = min(self.page_size, total_tokens - token_start)
                    page_id = self._allocate_one_page(
                        request_id=request.request_id,
                        layer_id=layer_id,
                        token_start=token_start,
                        token_count=token_count,
                        pinned=pinned,
                    )
                    page_ids.append(page_id)
                    token_start += token_count
            layer_page_map[layer_id] = page_ids
        return layer_page_map

    def commit_tokens(
        self, request: RequestState, num_new_tokens: int, pin: bool = True
    ) -> dict[int, list[int]]:
        """Commit additional tokens by making sure pages exist for them.

        Mark allocated pages as committed (finalized) rather than draft.
        """
        total_tokens = len(request.prompt_tokens) + len(request.committed_tokens) + num_new_tokens
        layer_page_map = self.allocate_for_request(request, total_tokens=total_tokens, pinned=pin)
        # Mark all pages for this request as committed, clear draft flag
        for layer_id in request.layer_ids:
            for page_id in self._page_table.get((request.request_id, layer_id), []):
                page = self._pages.get(page_id)
                if page is not None:
                    page.is_committed = True
                    page.is_draft = False
        return layer_page_map

    def allocate_draft_pages(
        self, request: RequestState, num_new_tokens: int
    ) -> dict[int, list[int]]:
        """Allocate speculative (draft) pages that can be discarded on rejection.

        These pages are marked as draft and not committed until verify_accept is called.
        """
        total_tokens = len(request.prompt_tokens) + len(request.committed_tokens) + num_new_tokens
        layer_page_map = self.allocate_for_request(request, total_tokens=total_tokens, pinned=True)
        for layer_id in request.layer_ids:
            for page_id in self._page_table.get((request.request_id, layer_id), []):
                page = self._pages.get(page_id)
                if page is not None and not page.is_committed:
                    page.is_draft = True
        return layer_page_map

    def release_draft_pages(self, request_id: int, layer_ids: list[int]) -> None:
        """Release all draft (speculative) pages for a request on rejection.

        Committed pages are left intact. Only draft pages are freed.
        """
        for layer_id in layer_ids:
            key = (request_id, layer_id)
            page_ids = list(self._page_table.get(key, []))
            to_remove: list[int] = []
            for page_id in page_ids:
                page = self._pages.get(page_id)
                if page is not None and page.is_draft:
                    to_remove.append(page_id)
            for page_id in to_remove:
                page = self._pages.pop(page_id, None)
                self._lru.pop(page_id, None)
                if page_id in self._page_table.get(key, []):
                    self._page_table[key].remove(page_id)
                if page is not None:
                    page_bytes = page.token_count * self._bytes_per_token_per_layer()
                    self._total_live_kv_bytes -= page_bytes
                    self._used_capacity_tokens -= page.token_count
                    self._total_capacity_tokens -= self.page_size
                    if page.pinned:
                        self._pinned_kv_bytes -= page_bytes
            if not self._page_table.get(key):
                self._page_table.pop(key, None)

    def accept_draft_pages(self, request_id: int, layer_ids: list[int]) -> None:
        """Transition draft pages to committed status on verification acceptance.

        This promotes all draft pages for the given request to committed.
        """
        for layer_id in layer_ids:
            for page_id in self._page_table.get((request_id, layer_id), []):
                page = self._pages.get(page_id)
                if page is not None and page.is_draft:
                    page.is_committed = True
                    page.is_draft = False

    def pin_pages(self, page_ids: list[int]) -> None:
        """Pin physical pages so they are protected from eviction.

        Also updates pinned byte accounting for memory metrics.
        """
        for page_id in page_ids:
            page = self._pages.get(page_id)
            if page is not None and not page.pinned:
                page.pinned = True
                page_bytes = page.token_count * self._bytes_per_token_per_layer()
                self._pinned_kv_bytes += page_bytes

    def unpin_pages(self, page_ids: list[int]) -> None:
        """Unpin physical pages so they may be evicted.

        Also updates pinned byte accounting for memory metrics.
        """
        for page_id in page_ids:
            page = self._pages.get(page_id)
            if page is not None and page.pinned:
                page.pinned = False
                page_bytes = page.token_count * self._bytes_per_token_per_layer()
                self._pinned_kv_bytes -= page_bytes

    def touch_pages(self, page_ids: list[int]) -> None:
        """Mark pages as recently used and account for residency hits."""
        for page_id in page_ids:
            self._access_count += 1
            page = self._pages.get(page_id)
            if page is not None:
                self._hit_count += 1
                if page_id in self._lru:
                    self._lru.move_to_end(page_id)

    def touch_request(self, request: RequestState) -> None:
        """Touch every page currently associated with the request."""
        self.touch_pages(request.kv_page_ids)

    def page_ids_for(self, request_id: int, layer_id: int) -> list[int]:
        """Return the pages mapped to one request/layer pair."""
        return list(self._page_table.get((request_id, layer_id), []))

    def positions_for(self, request_id: int) -> list[int]:
        """Return token positions currently covered by the request's pages."""
        positions: list[int] = []
        for (req_id, _layer_id), page_ids in self._page_table.items():
            if req_id != request_id:
                continue
            for page_id in page_ids:
                page = self._pages.get(page_id)
                if page is None:
                    continue
                positions.extend(range(page.token_start, page.token_start + page.token_count))
            break
        return positions

    def release_request(self, request_id: int) -> None:
        """Drop all pages belonging to a request and update memory accounting."""
        keys = [key for key in self._page_table if key[0] == request_id]
        for key in keys:
            for page_id in self._page_table.pop(key, []):
                page = self._pages.pop(page_id, None)
                self._lru.pop(page_id, None)
                if page is not None:
                    page_bytes = page.token_count * self._bytes_per_token_per_layer()
                    self._total_live_kv_bytes -= page_bytes
                    self._used_capacity_tokens -= page.token_count
                    self._total_capacity_tokens -= self.page_size
                    if page.pinned:
                        self._pinned_kv_bytes -= page_bytes
        # Defensive clamp: counters should never go negative
        if self._total_live_kv_bytes < 0:
            self._total_live_kv_bytes = 0
        if self._pinned_kv_bytes < 0:
            self._pinned_kv_bytes = 0
        if self._used_capacity_tokens < 0:
            self._used_capacity_tokens = 0
        if self._total_capacity_tokens < 0:
            self._total_capacity_tokens = 0

    def residency_report(self) -> dict[str, float | int]:
        """Return a cache residency summary with memory accounting.

        Includes:
        - hit_rate: fraction of page accesses that hit resident pages
        - eviction_count: total number of pages evicted
        - live_pages: number of pages currently in cache
        - pinned_pages: number of pages protected from eviction
        - live_kv_bytes: total bytes of KV data currently resident
        - pinned_kv_bytes: bytes protected from eviction (active decode)
        - evicted_kv_bytes: total bytes evicted across all requests
        - fragmentation_ratio: fraction of page capacity wasted (0.0 = no waste)
        """
        pinned = sum(1 for page in self._pages.values() if page.pinned)
        hit_rate = 0.0 if self._access_count == 0 else self._hit_count / self._access_count
        # Fragmentation: ratio of unused token slots in allocated pages
        if self._total_capacity_tokens > 0:
            fragmentation_ratio = 1.0 - (self._used_capacity_tokens / self._total_capacity_tokens)
        else:
            fragmentation_ratio = 0.0
        return {
            "hit_rate": hit_rate,
            "eviction_count": self._eviction_count,
            "live_pages": len(self._pages),
            "pinned_pages": pinned,
            "live_kv_bytes": self._total_live_kv_bytes,
            "pinned_kv_bytes": self._pinned_kv_bytes,
            "evicted_kv_bytes": self._evicted_kv_bytes,
            "fragmentation_ratio": fragmentation_ratio,
        }
