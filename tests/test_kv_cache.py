"""Tests for paged KV-cache allocation, eviction, and memory accounting."""

from megakernel_lab.config import RuntimeConfig
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


def test_memory_accounting_live_bytes_match_config() -> None:
    """Verify that live_kv_bytes equals expected token count * bytes_per_token_per_layer."""
    config = RuntimeConfig(
        num_heads=12, head_dim=64, dtype_bytes=2, kv_tensors_per_token=2, num_layers=2
    )
    cache = KVCache(page_size=2, max_pages=16, config=config)
    request = make_request(10, prompt_len=4)
    cache.allocate_for_request(request, total_tokens=4, pinned=True)
    report = cache.residency_report()

    # Each layer gets its own pages. For 4 tokens with page_size=2, we need 2 pages per layer.
    # Total pages = 2 pages * 2 layers = 4 pages.
    # Each page covers 2 tokens for 1 layer.
    # Bytes per page = 2 tokens * bytes_per_token_per_layer = 2 * 3072 = 6144
    # Total bytes = 4 pages * 6144 = 24576
    # OR: 4 tokens * 2 layers * bytes_per_token_per_layer = 4 * 2 * 3072 = 24576
    expected_bytes = 4 * config.num_layers * config.bytes_per_token_per_layer()
    assert report["live_kv_bytes"] == expected_bytes
    assert report["pinned_kv_bytes"] == expected_bytes


def test_memory_accounting_evicted_bytes() -> None:
    """Verify that evicted_kv_bytes accumulates correctly."""
    config = RuntimeConfig(
        num_heads=4, head_dim=32, dtype_bytes=2, kv_tensors_per_token=2, num_layers=1
    )
    cache = KVCache(page_size=2, max_pages=4, config=config)
    req1 = make_request(20, prompt_len=2)
    req2 = make_request(21, prompt_len=2)

    cache.allocate_for_request(req1, total_tokens=2, pinned=False)
    cache.allocate_for_request(req2, total_tokens=2, pinned=False)
    cache.touch_pages(req1.kv_page_ids)
    cache.touch_pages(req2.kv_page_ids)

    report_before = cache.residency_report()
    assert report_before["evicted_kv_bytes"] == 0

    cache.evict_lru(1)
    report_after = cache.residency_report()

    # Evicted 1 page of 2 tokens for 1 layer
    # Bytes = 2 tokens * bytes_per_token_per_layer = 2 * (4 * 32 * 2 * 2) = 2 * 512 = 1024
    expected_evicted = 2 * config.bytes_per_token_per_layer()
    assert report_after["evicted_kv_bytes"] == expected_evicted


def test_memory_accounting_fragmentation_ratio() -> None:
    """Fragmentation should be 0 when pages are fully used, >0 for partial pages."""
    config = RuntimeConfig(
        num_heads=4, head_dim=32, dtype_bytes=2, kv_tensors_per_token=2, num_layers=1
    )
    # page_size=3: allocating 3 tokens fills one page exactly -> no fragmentation
    cache = KVCache(page_size=3, max_pages=8, config=config)
    request = make_request(30, prompt_len=3)
    cache.allocate_for_request(request, total_tokens=3, pinned=False)
    report = cache.residency_report()
    assert report["fragmentation_ratio"] == 0.0

    # page_size=4: allocating 3 tokens -> 1 page with 1 unused slot -> 25% fragmentation
    cache2 = KVCache(page_size=4, max_pages=8, config=config)
    request2 = make_request(31, prompt_len=3)
    cache2.allocate_for_request(request2, total_tokens=3, pinned=False)
    report2 = cache2.residency_report()
    assert report2["fragmentation_ratio"] == 0.25


def test_pinning_updates_pinned_bytes() -> None:
    """Pin/unpin should correctly adjust pinned_kv_bytes."""
    config = RuntimeConfig(
        num_heads=4, head_dim=32, dtype_bytes=2, kv_tensors_per_token=2, num_layers=1
    )
    cache = KVCache(page_size=2, max_pages=8, config=config)
    # Use layer_ids=[0] to match num_layers=1 in config
    request = RequestState(
        request_id=40,
        prompt_tokens=[100, 101],
        target_tokens=[1, 2, 3, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0],
    )
    page_map = cache.allocate_for_request(request, total_tokens=2, pinned=False)
    page_ids = [p for pages in page_map.values() for p in pages]

    report_unpinned = cache.residency_report()
    assert report_unpinned["pinned_kv_bytes"] == 0

    cache.pin_pages(page_ids)
    report_pinned = cache.residency_report()
    # 1 page (for 1 layer) * 2 tokens * bytes_per_token_per_layer = 2 * 512 = 1024
    expected = 2 * config.bytes_per_token_per_layer()
    assert report_pinned["pinned_kv_bytes"] == expected

    cache.unpin_pages(page_ids)
    report_unpinned2 = cache.residency_report()
    assert report_unpinned2["pinned_kv_bytes"] == 0
