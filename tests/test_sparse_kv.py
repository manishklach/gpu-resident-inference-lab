"""Tests for the sparse-KV control-flow scaffold."""

from megakernel_lab.backend import CPUStubBackend
from megakernel_lab.config import RuntimeConfig
from megakernel_lab.kv_cache import KVCache
from megakernel_lab.runtime import DecodeWorker
from megakernel_lab.spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from megakernel_lab.sparse_kv import SparseKVSelector
from megakernel_lab.state import KVSnapshot, RequestState


def make_request(request_id: int = 1) -> RequestState:
    return RequestState(
        request_id=request_id,
        prompt_tokens=[1, 2, 3],
        target_tokens=[10, 11, 12, 13, 0],
        max_new_tokens=8,
        eos_token_id=0,
        layer_ids=[0, 1],
    )


def make_config() -> RuntimeConfig:
    return RuntimeConfig(
        block_size=2,
        max_new_tokens=8,
        eos_token_id=0,
        page_size=2,
        max_pages=32,
        num_layers=2,
        enable_sparse_kv=True,
        sparse_top_k=2,
        kv_block_size=2,
    )


def test_sparse_selector_picks_top_k_active_pages() -> None:
    config = make_config()
    cache = KVCache(page_size=config.page_size, max_pages=config.max_pages, config=config)
    request = make_request(10)
    layer_page_map = cache.allocate_for_request(request, total_tokens=7, pinned=False)
    request.kv_snapshot = KVSnapshot(
        request_id=request.request_id,
        layer_page_map=layer_page_map,
        prompt_length=len(request.prompt_tokens),
    )
    cache.commit_tokens(request, num_new_tokens=4, pin=False)

    selector = SparseKVSelector(config)
    selection = selector.select(request, cache, decode_step=len(request.committed_tokens))

    assert selection.selected_blocks == 2
    selected_pages = [cache._pages[page_id] for page_id in selection.page_ids]
    assert all(page.selected for page in selected_pages)
    assert [page.sparse_rank for page in selected_pages] == [0, 1]
    assert selected_pages[0].score >= selected_pages[1].score


def test_selected_pages_are_pinned_during_decode() -> None:
    config = make_config()
    cache = KVCache(page_size=config.page_size, max_pages=config.max_pages, config=config)
    request = make_request(11)
    layer_page_map = cache.allocate_for_request(request, total_tokens=5, pinned=False)
    request.kv_snapshot = KVSnapshot(
        request_id=request.request_id,
        layer_page_map=layer_page_map,
        prompt_length=len(request.prompt_tokens),
    )
    cache.commit_tokens(request, num_new_tokens=2, pin=False)

    worker = DecodeWorker(
        worker_id="decode-0",
        backend=CPUStubBackend(),
        kv_cache=cache,
        proposer=DraftBlockProposer(block_size=2, eos_token_id=config.eos_token_id),
        verifier=SpeculativeVerifier(AcceptancePolicy()),
        block_size=2,
        keep_ready_pinned=True,
        sparse_selector=SparseKVSelector(config),
    )

    request, trace, _ = worker.process(request, now_ms=0.0)

    assert request.phase.value == "ready_decode"
    assert trace.selected_kv_page_ids
    assert all(cache._pages[page_id].pinned for page_id in trace.selected_kv_page_ids)


def test_released_draft_pages_are_not_selected() -> None:
    config = make_config()
    cache = KVCache(page_size=config.page_size, max_pages=config.max_pages, config=config)
    request = make_request(12)
    layer_page_map = cache.allocate_for_request(request, total_tokens=3, pinned=False)
    request.kv_snapshot = KVSnapshot(
        request_id=request.request_id,
        layer_page_map=layer_page_map,
        prompt_length=len(request.prompt_tokens),
    )
    cache.commit_tokens(request, num_new_tokens=0, pin=False)
    cache.allocate_draft_pages(request, num_new_tokens=4)
    draft_ids = {
        page.page_id
        for page in cache.pages_for_request(request.request_id)
        if page.is_draft
    }
    assert draft_ids

    cache.release_draft_pages(request.request_id, request.layer_ids)
    selection = SparseKVSelector(config).select(request, cache, decode_step=0)

    assert draft_ids.isdisjoint(selection.page_ids)
    assert all(not cache._pages[page_id].is_draft for page_id in selection.page_ids)


def test_sparse_kv_metrics_exist_in_benchmark_schema() -> None:
    from megakernel_lab.bench import BenchmarkMode, BenchmarkRunner

    runner = BenchmarkRunner(batch_sizes=[1], block_sizes=[2])
    df = runner.run(modes=[BenchmarkMode.SPARSE_KV_MEGAKERNEL])

    row = df.iloc[0]
    assert "kv_blocks_total" in df.columns
    assert "kv_blocks_selected" in df.columns
    assert "kv_sparsity_ratio" in df.columns
    assert row["tokens_per_resident_loop"] >= 0.0
