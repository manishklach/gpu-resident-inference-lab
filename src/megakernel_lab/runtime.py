"""Persistent decode runtime simulator."""

from collections import deque

from .config import RuntimeConfig
from .kv_cache import KVCache
from .spec_decode import AcceptancePolicy, DraftBlockProposer, SpeculativeVerifier
from .state import DecodeRequest, DecodeResult, DecodeStepTrace


class PersistentDecodeRuntime:
    """Owns requests and simulates a persistent decode loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        proposer: DraftBlockProposer | None = None,
        verifier: SpeculativeVerifier | None = None,
    ) -> None:
        self.config = config
        self.proposer = proposer or DraftBlockProposer(
            block_size=config.block_size,
            eos_token_id=config.eos_token_id,
        )
        self.verifier = verifier or SpeculativeVerifier(AcceptancePolicy())
        self.kv_cache = KVCache()
        self._queue: deque[DecodeRequest] = deque()

    def submit(self, request: DecodeRequest) -> None:
        """Submit a request to the runtime."""
        self._queue.append(request)

    def run(self) -> list[DecodeResult]:
        """Run until all submitted requests finish."""
        results: list[DecodeResult] = []
        while self._queue:
            request = self._queue.popleft()
            results.append(self._run_request(request))
        return results

    def _run_request(self, request: DecodeRequest) -> DecodeResult:
        traces: list[DecodeStepTrace] = []

        while not request.finished:
            proposal = self.proposer.propose(request)
            request.draft_tokens = proposal
            accepted_count = self.verifier.verify(request, proposal)
            accepted_tokens = proposal[:accepted_count]

            if accepted_tokens:
                start_pos = len(request.prompt_tokens) + len(request.committed_tokens)
                request.committed_tokens.extend(accepted_tokens)
                self.kv_cache.commit(
                    request_id=request.request_id,
                    num_tokens=len(accepted_tokens),
                    starting_pos=start_pos,
                )

            traces.append(
                DecodeStepTrace(
                    proposed_tokens=list(proposal),
                    accepted_tokens=list(accepted_tokens),
                )
            )

            if (
                accepted_count == 0
                or request.token_budget_left() == 0
                or (accepted_tokens and accepted_tokens[-1] == request.eos_token_id)
                or len(request.committed_tokens) >= len(request.target_tokens)
            ):
                request.finished = True

        return DecodeResult(
            request_id=request.request_id,
            prompt_tokens=list(request.prompt_tokens),
            committed_tokens=list(request.committed_tokens),
            traces=traces,
        )
