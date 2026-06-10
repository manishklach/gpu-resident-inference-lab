"""Configuration objects for the runtime."""

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    """Controls the persistent runtime simulation."""

    block_size: int = 4
    max_new_tokens: int = 32
    eos_token_id: int = 0
    page_size: int = 4
    max_pages: int = 64
    num_layers: int = 2
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
