"""Persistent runtime with specialized prefill and decode workers."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .backend import AbstractKernelBackend, CPUStubBackend
from .config import RuntimeConfig
from .kv_cache import KVCache
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import (
    DecodeResult,
    DecodeStepTrace,
    KVSnapshot,
    RequestPhase,
    RequestState,
    TokenLogits,
    WorkerHandoff,
    WorkerStats,
)


@dataclass(order=True, slots=True)
class _QueueItem:
    """Priority queue wrapper for requests."""

    sort_key: tuple[int, int]
    request: RequestState = field(compare=False)


class _PriorityRequestQueue:
    """A tiny stable max-priority queue over requests."""

    def __init__(self) -> None:
        self._items: list[_QueueItem] = []
        self._counter = 0

    def push(self, request: RequestState) -> None:
        item = _QueueItem(sort_key=(-request.priority, self._counter), request=request)
        self._counter += 1
        heapq.heappush(self._items, item)

    def pop(self) -> RequestState:
        return heapq.heappop(self._items).request

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)


class PrefillWorker:
    """Consumes waiting requests and produces a KV snapshot."""

    def __init__(self, worker_id: str, backend: AbstractKernelBackend, kv_cache: KVCache) -> None:
        self.worker_id = worker_id
        self.backend = backend
        self.kv_cache = kv_cache
        self.stats = WorkerStats()

    def process(
        self, request: RequestState, now_ms: float
    ) -> tuple[RequestState, KVSnapshot, float]:
        request.phase = RequestPhase.PREFILL
        total_tokens = len(request.prompt_tokens)
        layer_page_map = self.kv_cache.allocate_for_request(
            request, total_tokens=total_tokens, pinned=False
        )
        page_ids = [page_id for pages in layer_page_map.values() for page_id in pages]
        self.kv_cache.touch_pages(page_ids)
        snapshot = self.backend.prefill(request.prompt_tokens, page_ids)
        snapshot = KVSnapshot(
            request_id=request.request_id,
            layer_page_map=layer_page_map,
            prompt_length=len(request.prompt_tokens),
            latency_ms=snapshot.latency_ms,
        )
        request.kv_snapshot = snapshot
        request.phase = RequestPhase.READY_DECODE
        request.prefill_worker_id = self.worker_id
        request.prefill_complete_ms = now_ms + snapshot.latency_ms
        self.stats.processed_requests += 1
        self.stats.backend_time_ms += snapshot.latency_ms
        return request, snapshot, snapshot.latency_ms


class DecodeWorker:
    """Consumes ready requests and advances them one speculative iteration at a time."""

    def __init__(
        self,
        worker_id: str,
        backend: AbstractKernelBackend,
        kv_cache: KVCache,
        proposer: DraftBlockProposer,
        verifier: SpeculativeVerifier,
        block_size: int,
    ) -> None:
        self.worker_id = worker_id
        self.backend = backend
        self.kv_cache = kv_cache
        self.proposer = proposer
        self.verifier = verifier
        self.block_size = block_size
        self.stats = WorkerStats()

    def process(
        self, request: RequestState, now_ms: float
    ) -> tuple[RequestState, DecodeStepTrace, float]:
        request.phase = RequestPhase.DECODE
        request.decode_worker_id = self.worker_id
        self.kv_cache.pin_pages(request.kv_page_ids)
        self.kv_cache.touch_request(request)

        logits: TokenLogits = self.backend.decode_step(request, request.kv_page_ids)
        proposal = self.proposer.propose(request, block_size=self.block_size)
        request.draft_tokens = list(proposal)
        mask = self.verifier.verify(request, proposal, request.kv_page_ids, self.backend)
        accepted_tokens = proposal[: mask.accepted_count]
        used_fallback = False

        if not accepted_tokens and len(proposal) > 1:
            used_fallback = True
            serial_proposal = self.proposer.propose(request, block_size=1)
            request.draft_tokens = list(serial_proposal)
            mask = self.verifier.verify(request, serial_proposal, request.kv_page_ids, self.backend)
            accepted_tokens = serial_proposal[: mask.accepted_count]
            proposal = serial_proposal

        if accepted_tokens:
            if not self.kv_cache.can_allocate(request, len(accepted_tokens)):
                needed_pages = sum(
                    self.kv_cache._pages_needed_for_layer(request, layer_id, len(accepted_tokens))
                    for layer_id in request.layer_ids
                )
                self.kv_cache.evict_lru(needed_pages)
            layer_page_map = self.kv_cache.commit_tokens(request, len(accepted_tokens), pin=True)
            request.committed_tokens.extend(accepted_tokens)
            request.kv_snapshot = KVSnapshot(
                request_id=request.request_id,
                layer_page_map=layer_page_map,
                prompt_length=len(request.prompt_tokens),
                latency_ms=request.kv_snapshot.latency_ms if request.kv_snapshot else 0.0,
            )

        backend_latency_ms = logits.latency_ms + mask.latency_ms
        finish_request = (
            request.token_budget_left() == 0
            or (accepted_tokens and accepted_tokens[-1] == request.eos_token_id)
            or len(request.committed_tokens) >= len(request.target_tokens)
            or (not accepted_tokens and len(proposal) == 1)
        )

        if finish_request:
            request.phase = RequestPhase.FINISHED
            request.finished = True
            self.kv_cache.unpin_pages(request.kv_page_ids)
        else:
            request.phase = RequestPhase.READY_DECODE
            request.was_preempted = True

        if request.first_decode_ms is None:
            request.first_decode_ms = now_ms + backend_latency_ms
        request.last_decode_ms = now_ms + backend_latency_ms

        trace = DecodeStepTrace(
            worker_id=self.worker_id,
            proposed_tokens=list(proposal),
            accepted_tokens=list(accepted_tokens),
            used_fallback_serial=used_fallback,
            backend_latency_ms=backend_latency_ms,
        )
        self.stats.processed_requests += 1
        self.stats.backend_time_ms += backend_latency_ms
        return request, trace, backend_latency_ms


class WorkerPool:
    """Manages prefill and decode workers with a simple priority dispatcher."""

    def __init__(
        self,
        config: RuntimeConfig,
        backend: AbstractKernelBackend,
        kv_cache: KVCache,
        proposer: DraftBlockProposer,
        verifier: SpeculativeVerifier,
    ) -> None:
        self.prefill_workers = [
            PrefillWorker(f"prefill-{idx}", backend, kv_cache)
            for idx in range(config.num_prefill_workers)
        ]
        self.decode_workers = [
            DecodeWorker(f"decode-{idx}", backend, kv_cache, proposer, verifier, config.block_size)
            for idx in range(config.num_decode_workers)
        ]
        self.prefill_queue = _PriorityRequestQueue()
        self.decode_queue = _PriorityRequestQueue()
        self.handoffs: list[WorkerHandoff] = []
        self._prefill_rr = 0
        self._decode_rr = 0

    def enqueue_prefill(self, request: RequestState) -> None:
        self.prefill_queue.push(request)

    def enqueue_decode(self, request: RequestState) -> None:
        self.decode_queue.push(request)

    def dispatch_prefill(self, now_ms: float) -> tuple[list[RequestState], float]:
        completed: list[RequestState] = []
        spent_ms = 0.0
        while self.prefill_queue:
            worker = self.prefill_workers[self._prefill_rr % len(self.prefill_workers)]
            self._prefill_rr += 1
            request = self.prefill_queue.pop()
            request, snapshot, latency_ms = worker.process(request, now_ms + spent_ms)
            completed.append(request)
            spent_ms += latency_ms
            decode_worker = self.decode_workers[self._decode_rr % len(self.decode_workers)]
            self.handoffs.append(
                WorkerHandoff(
                    request_id=request.request_id,
                    prefill_worker_id=worker.worker_id,
                    decode_worker_id=decode_worker.worker_id,
                    page_ids=[
                        page_id for pages in snapshot.layer_page_map.values() for page_id in pages
                    ],
                )
            )
        return completed, spent_ms

    def dispatch_decode(
        self, now_ms: float
    ) -> tuple[list[tuple[RequestState, DecodeStepTrace]], float]:
        results: list[tuple[RequestState, DecodeStepTrace]] = []
        spent_ms = 0.0
        workers = max(1, len(self.decode_workers))
        for _ in range(min(len(self.decode_queue), workers)):
            worker = self.decode_workers[self._decode_rr % len(self.decode_workers)]
            self._decode_rr += 1
            request = self.decode_queue.pop()
            request, trace, latency_ms = worker.process(request, now_ms + spent_ms)
            results.append((request, trace))
            spent_ms += latency_ms
        return results, spent_ms


class PersistentDecodeRuntime:
    """Owns requests and simulates a persistent decode loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        proposer: DraftBlockProposer | None = None,
        verifier: SpeculativeVerifier | None = None,
        backend: AbstractKernelBackend | None = None,
        kv_cache: KVCache | None = None,
    ) -> None:
        self.config = config
        self.proposer = proposer or DraftBlockProposer(
            block_size=config.block_size,
            eos_token_id=config.eos_token_id,
        )
        self.verifier = verifier or SpeculativeVerifier(AcceptancePolicy())
        self.backend = backend or CPUStubBackend()
        self.kv_cache = kv_cache or KVCache(
            page_size=config.page_size,
            max_pages=config.max_pages,
            config=config,
        )
        self.worker_pool = WorkerPool(
            config=config,
            backend=self.backend,
            kv_cache=self.kv_cache,
            proposer=self.proposer,
            verifier=self.verifier,
        )
        self._virtual_time_ms = 0.0
        self._results: dict[int, list[DecodeStepTrace]] = {}

    def submit(self, request: RequestState) -> None:
        """Submit a request to the runtime."""
        if not request.layer_ids:
            request.layer_ids = list(range(self.config.num_layers))
        self.worker_pool.enqueue_prefill(request)

    def run(self) -> list[DecodeResult]:
        """Run until all submitted requests finish."""
        final_results: list[DecodeResult] = []
        while self.worker_pool.prefill_queue or self.worker_pool.decode_queue:
            completed_prefill, prefill_ms = self.worker_pool.dispatch_prefill(self._virtual_time_ms)
            self._virtual_time_ms += prefill_ms
            for request in completed_prefill:
                self.worker_pool.enqueue_decode(request)

            decode_results, decode_ms = self.worker_pool.dispatch_decode(self._virtual_time_ms)
            self._virtual_time_ms += decode_ms
            for request, trace in decode_results:
                self._results.setdefault(request.request_id, []).append(trace)
                if request.finished:
                    final_results.append(self._build_result(request))
                    self.kv_cache.release_request(request.request_id)
                else:
                    self.worker_pool.enqueue_decode(request)
        return sorted(final_results, key=lambda result: result.request_id)

    def _build_result(self, request: RequestState) -> DecodeResult:
        traces = self._results.get(request.request_id, [])
        total_accepted = sum(len(trace.accepted_tokens) for trace in traces)
        total_proposed = sum(len(trace.proposed_tokens) for trace in traces)
        acceptance_rate = 0.0 if total_proposed == 0 else total_accepted / total_proposed
        ttft_ms = (request.first_decode_ms or self._virtual_time_ms) - (
            request.prefill_complete_ms or 0.0
        )
        inter_token_latency_ms = [
            trace.backend_latency_ms for trace in traces if trace.accepted_tokens
        ]
        return DecodeResult(
            request_id=request.request_id,
            prompt_tokens=list(request.prompt_tokens),
            committed_tokens=list(request.committed_tokens),
            traces=traces,
            ttft_ms=ttft_ms,
            inter_token_latency_ms=inter_token_latency_ms,
            acceptance_rate=acceptance_rate,
            was_preempted=request.was_preempted,
        )
