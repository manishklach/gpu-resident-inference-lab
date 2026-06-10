"""Core request, worker, and backend state for the persistent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestPhase(str, Enum):
    """Lifecycle phases for a request inside the runtime."""

    WAITING_PREFILL = "waiting_prefill"
    PREFILL = "prefill"
    READY_DECODE = "ready_decode"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass(slots=True)
class KVSnapshot:
    """Represents the KV state produced by a prefill stage."""

    request_id: int
    layer_page_map: dict[int, list[int]]
    prompt_length: int
    latency_ms: float = 0.0


@dataclass(slots=True)
class TokenLogits:
    """A lightweight stand-in for backend decode output."""

    candidate_tokens: list[int]
    scores: list[float]
    latency_ms: float = 0.0


@dataclass(slots=True)
class AcceptanceMask:
    """Boolean acceptance mask for speculative verification."""

    accepted: list[bool]
    latency_ms: float = 0.0

    @property
    def accepted_count(self) -> int:
        """Return the accepted prefix length."""
        count = 0
        for accepted in self.accepted:
            if not accepted:
                break
            count += 1
        return count


@dataclass(slots=True)
class DecodeStepTrace:
    """Records one decode iteration."""

    worker_id: str
    proposed_tokens: list[int]
    accepted_tokens: list[int]
    used_fallback_serial: bool = False
    backend_latency_ms: float = 0.0


@dataclass(slots=True)
class RequestState:
    """Represents one active request owned by the runtime."""

    request_id: int
    prompt_tokens: list[int]
    target_tokens: list[int]
    max_new_tokens: int
    eos_token_id: int
    priority: int = 0
    layer_ids: list[int] = field(default_factory=lambda: [0])
    committed_tokens: list[int] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    phase: RequestPhase = RequestPhase.WAITING_PREFILL
    kv_snapshot: KVSnapshot | None = None
    finished: bool = False
    was_preempted: bool = False
    prefill_worker_id: str | None = None
    decode_worker_id: str | None = None
    prefill_complete_ms: float | None = None
    first_decode_ms: float | None = None
    last_decode_ms: float | None = None

    def remaining_target(self) -> list[int]:
        """Return the uncommitted portion of the target sequence."""
        offset = len(self.committed_tokens)
        return self.target_tokens[offset:]

    def token_budget_left(self) -> int:
        """Number of output positions still allowed."""
        return max(self.max_new_tokens - len(self.committed_tokens), 0)

    def full_sequence(self) -> list[int]:
        """Prompt plus committed decode output."""
        return [*self.prompt_tokens, *self.committed_tokens]

    @property
    def kv_page_ids(self) -> list[int]:
        """Flatten the KV page ids across layers."""
        if self.kv_snapshot is None:
            return []
        page_ids: list[int] = []
        for layer_id in self.layer_ids:
            page_ids.extend(self.kv_snapshot.layer_page_map.get(layer_id, []))
        return page_ids


@dataclass(slots=True)
class WorkerStats:
    """Tracks how much work a worker processed."""

    processed_requests: int = 0
    backend_time_ms: float = 0.0


@dataclass(slots=True)
class WorkerHandoff:
    """Captures the prefill-to-decode transfer event."""

    request_id: int
    prefill_worker_id: str
    decode_worker_id: str
    page_ids: list[int]


@dataclass(slots=True)
class DecodeResult:
    """Final output of one decoded request."""

    request_id: int
    prompt_tokens: list[int]
    committed_tokens: list[int]
    traces: list[DecodeStepTrace]
    ttft_ms: float
    inter_token_latency_ms: list[float]
    acceptance_rate: float
    was_preempted: bool

    @property
    def full_sequence(self) -> list[int]:
        """Prompt plus committed decode output."""
        return [*self.prompt_tokens, *self.committed_tokens]


DecodeRequest = RequestState
