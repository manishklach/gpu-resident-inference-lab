"""Tests for speculative KV page lifecycle (committed vs draft pages)."""

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


def test_rejected_draft_pages_are_released() -> None:
    """Draft pages should be freed when verification rejects them."""
    cache = KVCache(page_size=2, max_pages=16)
    request = make_request(50, prompt_len=2)

    # Allocate initial committed pages
    cache.allocate_for_request(request, total_tokens=2, pinned=True)
    cache.commit_tokens(request, num_new_tokens=0, pin=True)
    cache.accept_draft_pages(request.request_id, request.layer_ids)
    committed_report = cache.residency_report()

    # Now allocate draft pages for speculative tokens
    cache.allocate_draft_pages(request, num_new_tokens=4)
    draft_report = cache.residency_report()
    assert draft_report["live_pages"] > committed_report["live_pages"]

    # Release draft pages (simulating rejection)
    cache.release_draft_pages(request.request_id, request.layer_ids)
    post_release_report = cache.residency_report()
    assert post_release_report["live_pages"] == committed_report["live_pages"]


def test_accepted_draft_pages_become_committed() -> None:
    """Draft pages should transition to committed when accept_draft_pages is called."""
    cache = KVCache(page_size=2, max_pages=16)
    request = make_request(51, prompt_len=2)

    # Allocate initial pages (these will be committed)
    cache.allocate_for_request(request, total_tokens=2, pinned=True)
    cache.commit_tokens(request, num_new_tokens=0, pin=True)
    committed_before = set(cache.page_ids_for(request.request_id, request.layer_ids[0]))

    # Allocate draft pages (separate from committed)
    cache.allocate_draft_pages(request, num_new_tokens=2)
    all_pages = cache.page_ids_for(request.request_id, 0)
    draft_pages = [p for p in all_pages if p not in committed_before]
    assert draft_pages

    # Check draft pages are marked as draft
    for page_id in draft_pages:
        page = cache._pages[page_id]
        assert page.is_draft
        assert not page.is_committed

    # Accept draft pages
    cache.accept_draft_pages(request.request_id, request.layer_ids)

    # Now draft pages should be committed
    for page_id in draft_pages:
        page = cache._pages[page_id]
        assert page.is_committed
        assert not page.is_draft


def test_committed_pages_not_discarded_during_rejection() -> None:
    """release_draft_pages should not touch committed pages."""
    cache = KVCache(page_size=2, max_pages=16)
    request = make_request(52, prompt_len=4)

    # Allocate committed pages
    cache.allocate_for_request(request, total_tokens=4, pinned=True)
    cache.commit_tokens(request, num_new_tokens=0, pin=True)
    cache.accept_draft_pages(request.request_id, request.layer_ids)
    committed_pages = set(cache.page_ids_for(request.request_id, 0))
    committed_count_before = len(committed_pages)

    # Release draft pages (there are none, but the call should not remove committed)
    cache.release_draft_pages(request.request_id, request.layer_ids)
    committed_pages_after = set(cache.page_ids_for(request.request_id, 0))

    assert committed_pages_after == committed_pages
    assert len(committed_pages_after) == committed_count_before


def test_pinned_active_decode_pages_are_not_evicted() -> None:
    """Pinned pages used for active decode should survive eviction pressure."""
    cache = KVCache(page_size=2, max_pages=6)
    active_request = make_request(53, prompt_len=2)
    cold_request = make_request(54, prompt_len=2)

    # Active request: allocate + commit + pin
    cache.allocate_for_request(active_request, total_tokens=2, pinned=True)
    cache.commit_tokens(active_request, num_new_tokens=0, pin=True)
    cache.accept_draft_pages(active_request.request_id, active_request.layer_ids)
    cache.touch_request(active_request)

    # Cold request: allocate but don't pin
    cache.allocate_for_request(cold_request, total_tokens=2, pinned=False)
    cache.touch_request(cold_request)

    # Fill remaining capacity
    filler = make_request(55, prompt_len=2)
    cache.allocate_for_request(filler, total_tokens=2, pinned=False)
    cache.touch_request(filler)

    # Request eviction of 3 pages (more than cold pages available)
    evicted = cache.evict_lru(3)

    # Active request's pages should never appear in evicted list
    active_page_ids = set(active_request.kv_page_ids)
    assert active_page_ids.isdisjoint(evicted)

    # Active pages should still be live
    report = cache.residency_report()
    assert report["pinned_pages"] >= 1


def test_draft_evicted_before_committed() -> None:
    """When both draft and committed pages exist, drafts should be evicted first."""
    cache = KVCache(page_size=2, max_pages=4)
    req1 = make_request(60, prompt_len=2)
    req2 = make_request(61, prompt_len=2)

    # Commit pages for req1 (2 tokens -> 1 page per layer)
    cache.allocate_for_request(req1, total_tokens=2, pinned=True)
    cache.commit_tokens(req1, num_new_tokens=0, pin=True)
    cache.accept_draft_pages(req1.request_id, req1.layer_ids)

    # Allocate separate draft pages for req2 (not committed)
    cache.allocate_for_request(req2, total_tokens=2, pinned=False)
    # Manually mark these as draft (allocate_draft_pages would also allocate more)
    for layer_id in req2.layer_ids:
        for page_id in cache._page_table.get((req2.request_id, layer_id), []):
            page = cache._pages[page_id]
            if page is not None:
                page.is_draft = True

    # Evict 1 page - should prefer draft pages over committed
    evicted = cache.evict_lru(1)
    assert len(evicted) == 1
    # Committed pages for req1 should still be intact
    req1_pages = set(cache.page_ids_for(req1.request_id, req1.layer_ids[0]))
    assert req1_pages


def test_multiple_requests_independent_lifecycle() -> None:
    """Draft lifecycle for one request should not affect another."""
    cache = KVCache(page_size=2, max_pages=16)
    req_a = make_request(70, prompt_len=2)
    req_b = make_request(71, prompt_len=2)

    # Commit req_a
    cache.allocate_for_request(req_a, total_tokens=2, pinned=True)
    cache.commit_tokens(req_a, num_new_tokens=0, pin=True)
    cache.accept_draft_pages(req_a.request_id, req_a.layer_ids)

    # Draft for req_b
    cache.allocate_for_request(req_b, total_tokens=2, pinned=True)
    cache.allocate_draft_pages(req_b, num_new_tokens=2)

    a_pages_before = set(cache.page_ids_for(req_a.request_id, req_a.layer_ids[0]))

    # Release req_b drafts - req_a should be unaffected
    cache.release_draft_pages(req_b.request_id, req_b.layer_ids)
    a_pages_after = set(cache.page_ids_for(req_a.request_id, req_a.layer_ids[0]))

    assert a_pages_after == a_pages_before
