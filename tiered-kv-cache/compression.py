from __future__ import annotations

import pickle
import time

import numpy as np
import torch
import zstandard as zstd


def compress_block(kv: torch.Tensor) -> bytes:
    cpu = kv.detach().to("cpu", dtype=torch.float32)
    array = cpu.numpy()
    scale = np.max(np.abs(array), axis=-1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale / 127.0).astype(np.float32)
    quantized = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
    payload = {
        "shape": array.shape,
        "dtype": str(kv.dtype),
        "scale": scale,
        "quantized": quantized,
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return zstd.ZstdCompressor(level=7).compress(raw)


def decompress_block(data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    raw = zstd.ZstdDecompressor().decompress(data)
    payload = pickle.loads(raw)
    quantized = payload["quantized"].astype(np.float32)
    scale = payload["scale"].astype(np.float32)
    restored = quantized * scale
    tensor = torch.from_numpy(restored.reshape(shape))
    return tensor.to(dtype=dtype)


def benchmark() -> None:
    kv = torch.randn(32, 128, 128, dtype=torch.float16)
    start = time.perf_counter()
    encoded = compress_block(kv)
    mid = time.perf_counter()
    decoded = decompress_block(encoded, shape=tuple(kv.shape), dtype=kv.dtype)
    end = time.perf_counter()

    ratio = (kv.numel() * kv.element_size()) / max(len(encoded), 1)
    compress_ms = (mid - start) * 1000.0
    decompress_ms = (end - mid) * 1000.0
    max_error = (kv.float() - decoded.float()).abs().max().item()

    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Compress latency: {compress_ms:.2f} ms")
    print(f"Decompress latency: {decompress_ms:.2f} ms")
    print(f"Max reconstruction error: {max_error:.5f}")


if __name__ == "__main__":
    benchmark()
