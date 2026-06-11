"""Configuration objects for the runtime."""

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    """Controls the persistent runtime simulation.

    Attributes:
        block_size: Number of tokens per speculative block proposal.
        max_new_tokens: Maximum tokens to generate per request.
        eos_token_id: Token ID that signals end of sequence.
        page_size: Number of tokens per KV-cache page.
        max_pages: Total physical pages available in the KV cache.
        num_layers: Number of transformer layers (each gets its own KV pages).
        num_prefill_workers: Number of parallel prefill workers.
        num_decode_workers: Number of parallel decode workers.
        pin_ready_decode_pages: Keep READY_DECODE request pages pinned between
            decode iterations. If False, requests may need KV rehydration after
            eviction before they resume.
        num_heads: Number of KV attention heads (for memory accounting).
        head_dim: Dimension per attention head (for memory accounting).
        dtype_bytes: Bytes per KV element (2 for fp16, 4 for fp32).
        kv_tensors_per_token: Number of KV tensors stored per token per layer.
            Typically 2 (key + value). Some architectures use 4 (key, value,
            plus gate/up projections for GQA).
    """

    block_size: int = 4
    max_new_tokens: int = 32
    eos_token_id: int = 0
    page_size: int = 4
    max_pages: int = 64
    num_layers: int = 2
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
    pin_ready_decode_pages: bool = True
    num_heads: int = 12
    head_dim: int = 64
    dtype_bytes: int = 2
    kv_tensors_per_token: int = 2
    enable_sparse_kv: bool = False
    sparse_top_k: int = 4
    kv_block_size: int = 4

    def bytes_per_token_per_layer(self) -> int:
        """Bytes of KV storage for one token across one layer.

        Computed as: num_heads * head_dim * dtype_bytes * kv_tensors_per_token.
        For a standard multi-head attention layer this is 2 * num_heads * head_dim * dtype_bytes.
        """
        return self.num_heads * self.head_dim * self.dtype_bytes * self.kv_tensors_per_token

    def bytes_per_page(self) -> int:
        """Bytes of KV storage for one physical page across all layers.

        Each page holds page_size tokens across num_layers layers.
        """
        return self.page_size * self.num_layers * self.bytes_per_token_per_layer()
