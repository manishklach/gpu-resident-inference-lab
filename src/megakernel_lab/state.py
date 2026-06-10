"""Core request and result state for persistent decode."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class DecodeRequest:
    """Represents one active request owned by the runtime."""

    request_id: int
    prompt_tokens: list[int]
    target_tokens: list[int]
    max_new_tokens: int
    eos_token_id: int
    committed_tokens: list[int] = field(default_factory=list)
    draft_tokens: list[int] = field(default_factory=list)
    finished: bool = False

    def remaining_target(self) -> list[int]:
        """Return the uncommitted portion of the target sequence."""
        offset = len(self.committed_tokens)
        return self.target_tokens[offset:]

    def token_budget_left(self) -> int:
        """Number of output positions still allowed."""
        return max(self.max_new_tokens - len(self.committed_tokens), 0)


@dataclass(slots=True)
class DecodeStepTrace:
    """Records one decode iteration."""

    proposed_tokens: list[int]
    accepted_tokens: list[int]


@dataclass(slots=True)
class DecodeResult:
    """Final output of one decoded request."""

    request_id: int
    prompt_tokens: list[int]
    committed_tokens: list[int]
    traces: list[DecodeStepTrace]

    @property
    def full_sequence(self) -> list[int]:
        """Prompt plus committed decode output."""
        return [*self.prompt_tokens, *self.committed_tokens]
