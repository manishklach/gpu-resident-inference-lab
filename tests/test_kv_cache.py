"""Tests for paged KV-cache allocation and eviction."""

from megakernel_lab.kv_cache import KVCache
from megakernel_lab.state import RequestState


def make_request(request_id: int, prompt_len: int = 2) -> RequestState:
    """Build a request with two logical layers."""
    return RequestState(
        request_id=request_id,
        prompt_tokens=[100 + idx for idx in range(prompt_len)],
        target_tokens=[1, 2, 3, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0, 1],
    )


def test_kv_page_eviction_under_pressure_respects_pins() -> None:
    cache = KVCache(page_size=2, max_pages=4)
    pinned_request = make_request(1)
    cold_request = make_request(2)

    pinned_map = cache.allocate_for_request(pinned_request, total_tokens=2, pinned=True)
    cold_map = cache.allocate_for_request(cold_request, total_tokens=2, pinned=False)
    cache.touch_pages([page for pages in cold_map.values() for page in pages])
    cache.touch_pages([page for pages in pinned_map.values() for page in pages])

    evicted = cache.evict_lru(2)
    pinned_pages = {page for pages in pinned_map.values() for page in pages}
    cold_pages = {page for pages in cold_map.values() for page in pages}

    assert evicted
    assert pinned_pages.isdisjoint(evicted)
    assert any(page in cold_pages for page in evicted)


def test_residency_report_tracks_hits_and_evictions() -> None:
    cache = KVCache(page_size=2, max_pages=2)
    request = make_request(3)
    page_map = cache.allocate_for_request(request, total_tokens=2, pinned=False)
    page_ids = [page for pages in page_map.values() for page in pages]
    cache.touch_pages(page_ids)
    cache.evict_lru(1)
    report = cache.residency_report()

    assert report["hit_rate"] > 0.0
    assert report["eviction_count"] == 1
    assert report["live_pages"] == 1
