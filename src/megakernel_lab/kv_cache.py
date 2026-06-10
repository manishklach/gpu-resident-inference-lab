"""KV-cache abstractions for the runtime simulation."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class KVPage:
    """A tiny placeholder for a committed KV region."""

    request_id: int
    token_positions: list[int] = field(default_factory=list)


class KVCache:
    """Tracks committed positions as if they were device-resident KV entries."""

    def __init__(self) -> None:
        self._pages: dict[int, KVPage] = {}

    def commit(self, request_id: int, num_tokens: int, starting_pos: int) -> None:
        """Mark positions as committed for one request."""
        page = self._pages.setdefault(request_id, KVPage(request_id=request_id))
        page.token_positions.extend(range(starting_pos, starting_pos + num_tokens))

    def positions_for(self, request_id: int) -> list[int]:
        """Return committed positions for the request."""
        page = self._pages.get(request_id)
        return [] if page is None else list(page.token_positions)
