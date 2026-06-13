from __future__ import annotations

import mmap
import os
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from compression import compress_block, decompress_block


@dataclass
class BlockLocation:
    tier: str
    offset: int
    size_bytes: int


class TieredKVCache:
    def __init__(self, gpu_budget_gb: float, cpu_budget_gb: float, nvme_path: str) -> None:
        self.gpu_budget_bytes = int(gpu_budget_gb * (1024**3))
        self.cpu_budget_bytes = int(cpu_budget_gb * (1024**3))
        self.nvme_root = Path(nvme_path)
        self.nvme_root.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.locations: Dict[Tuple[int, int], BlockLocation] = {}
        self.metadata: Dict[Tuple[int, int], dict] = {}
        self.gpu_store: Dict[Tuple[int, int], torch.Tensor] = {}
        self.cpu_store: Dict[Tuple[int, int], torch.Tensor] = {}
        self.lru = {
            "gpu": OrderedDict(),
            "cpu": OrderedDict(),
            "nvme": OrderedDict(),
        }
        self.usage = {"gpu": 0, "cpu": 0, "nvme": 0}
        self.hits = {"gpu": 0, "cpu": 0, "nvme": 0}
        self.read_counts = {"gpu": 0, "cpu": 0, "nvme": 0}
        self.read_latency_ms = {"gpu": [], "cpu": [], "nvme": []}
        self._offset = 0

    def _key(self, layer_id: int, block_id: int) -> Tuple[int, int]:
        return (layer_id, block_id)

    def _touch(self, tier: str, key: Tuple[int, int]) -> None:
        bucket = self.lru[tier]
        if key in bucket:
            bucket.move_to_end(key)
        else:
            bucket[key] = time.time()

    def _promote_cpu_to_gpu(self, key: Tuple[int, int]) -> torch.Tensor:
        cpu_tensor = self.cpu_store[key]
        size_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        self._ensure_capacity("gpu", size_bytes)
        if torch.cuda.is_available():
            pinned = cpu_tensor.pin_memory()
            with torch.cuda.stream(self.stream):
                gpu_tensor = pinned.to("cuda", non_blocking=True)
            self.stream.synchronize()
        else:
            gpu_tensor = cpu_tensor.clone()
        self.gpu_store[key] = gpu_tensor
        self.locations[key] = BlockLocation("gpu", 0, size_bytes)
        self.usage["gpu"] += size_bytes
        self._touch("gpu", key)
        return gpu_tensor

    def _nvme_file(self, key: Tuple[int, int]) -> Path:
        layer_id, block_id = key
        return self.nvme_root / f"layer_{layer_id}_block_{block_id}.bin"

    def _ensure_capacity(self, tier: str, needed_bytes: int) -> None:
        budget = self.gpu_budget_bytes if tier == "gpu" else self.cpu_budget_bytes
        while budget > 0 and self.usage[tier] + needed_bytes > budget and self.lru[tier]:
            self.evict(strategy="lru", tier=tier)

    def allocate_block(self, layer_id: int, block_id: int) -> BlockLocation:
        with self.lock:
            key = self._key(layer_id, block_id)
            if key in self.locations:
                location = self.locations[key]
                self._touch(location.tier, key)
                return location

            placeholder_size = 4 * 1024 * 1024
            if self.usage["gpu"] + placeholder_size <= self.gpu_budget_bytes:
                location = BlockLocation("gpu", 0, placeholder_size)
                self.usage["gpu"] += placeholder_size
            elif self.usage["cpu"] + placeholder_size <= self.cpu_budget_bytes:
                location = BlockLocation("cpu", 0, placeholder_size)
                self.usage["cpu"] += placeholder_size
            else:
                file_path = self._nvme_file(key)
                file_path.touch(exist_ok=True)
                location = BlockLocation("nvme", self._offset, placeholder_size)
                self.usage["nvme"] += placeholder_size
                self._offset += placeholder_size

            self.locations[key] = location
            self.metadata[key] = {"shape": None, "dtype": None, "compressed": False}
            self._touch(location.tier, key)
            return location

    def write(self, layer_id: int, block_id: int, kv: torch.Tensor) -> None:
        with self.lock:
            key = self._key(layer_id, block_id)
            size_bytes = kv.numel() * kv.element_size()
            metadata = {"shape": tuple(kv.shape), "dtype": str(kv.dtype), "compressed": False}

            target_tier = "gpu"
            if self.gpu_budget_bytes <= 0 or self.usage["gpu"] + size_bytes > self.gpu_budget_bytes:
                target_tier = "cpu"
            if target_tier == "cpu" and (self.cpu_budget_bytes <= 0 or self.usage["cpu"] + size_bytes > self.cpu_budget_bytes):
                target_tier = "nvme"

            if target_tier == "gpu":
                self._ensure_capacity("gpu", size_bytes)
                tensor = kv.detach().to(self.device)
                self.gpu_store[key] = tensor
                self.locations[key] = BlockLocation("gpu", 0, size_bytes)
                self.usage["gpu"] += size_bytes
                self._touch("gpu", key)
            elif target_tier == "cpu":
                self._ensure_capacity("cpu", size_bytes)
                tensor = kv.detach().to("cpu")
                self.cpu_store[key] = tensor
                self.locations[key] = BlockLocation("cpu", 0, size_bytes)
                self.usage["cpu"] += size_bytes
                self._touch("cpu", key)
            else:
                cpu_tensor = kv.detach().to("cpu")
                payload = compress_block(cpu_tensor)
                file_path = self._nvme_file(key)
                file_path.write_bytes(payload)
                metadata["compressed"] = True
                self.locations[key] = BlockLocation("nvme", 0, len(payload))
                self.usage["nvme"] += len(payload)
                self._touch("nvme", key)

            self.metadata[key] = metadata

    def _read_nvme_to_cpu(self, key: Tuple[int, int]) -> torch.Tensor:
        metadata = self.metadata[key]
        shape = tuple(metadata["shape"])
        dtype = getattr(torch, metadata["dtype"].split(".")[-1], torch.float32)
        file_path = self._nvme_file(key)

        def _read_bytes() -> bytes:
            with open(file_path, "rb") as handle:
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm[:]

        data = self.executor.submit(_read_bytes).result()
        return decompress_block(data, shape=shape, dtype=dtype)

    def read(self, layer_id: int, block_id: int) -> torch.Tensor:
        started = time.perf_counter()
        with self.lock:
            key = self._key(layer_id, block_id)
            if key in self.gpu_store:
                self.hits["gpu"] += 1
                self.read_counts["gpu"] += 1
                self._touch("gpu", key)
                tensor = self.gpu_store[key]
                self.read_latency_ms["gpu"].append((time.perf_counter() - started) * 1000.0)
                return tensor

            if key in self.cpu_store:
                self.hits["cpu"] += 1
                self.read_counts["cpu"] += 1
                tensor = self._promote_cpu_to_gpu(key)
                self.read_latency_ms["cpu"].append((time.perf_counter() - started) * 1000.0)
                return tensor

            if key in self.locations and self.locations[key].tier == "nvme":
                self.hits["nvme"] += 1
                self.read_counts["nvme"] += 1
                cpu_tensor = self._read_nvme_to_cpu(key)
                size_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
                self._ensure_capacity("cpu", size_bytes)
                self.cpu_store[key] = cpu_tensor
                self.usage["cpu"] += size_bytes
                self.locations[key] = BlockLocation("cpu", 0, size_bytes)
                self._touch("cpu", key)
                tensor = self._promote_cpu_to_gpu(key)
                self.read_latency_ms["nvme"].append((time.perf_counter() - started) * 1000.0)
                return tensor

        raise KeyError(f"Missing KV block {(layer_id, block_id)}")

    def evict(self, strategy: str = "lru", tier: str | None = None) -> None:
        with self.lock:
            if strategy != "lru":
                raise ValueError("Only lru eviction is implemented.")
            target_tier = tier or "gpu"
            bucket = self.lru[target_tier]
            if not bucket:
                return
            key, _ = bucket.popitem(last=False)
            location = self.locations.get(key)
            if location is None:
                return

            if target_tier == "gpu" and key in self.gpu_store:
                tensor = self.gpu_store.pop(key)
                size_bytes = tensor.numel() * tensor.element_size()
                self.usage["gpu"] = max(0, self.usage["gpu"] - size_bytes)
                self.write(key[0], key[1], tensor.to("cpu"))
            elif target_tier == "cpu" and key in self.cpu_store:
                tensor = self.cpu_store.pop(key)
                size_bytes = tensor.numel() * tensor.element_size()
                self.usage["cpu"] = max(0, self.usage["cpu"] - size_bytes)
                payload = compress_block(tensor)
                self._nvme_file(key).write_bytes(payload)
                self.locations[key] = BlockLocation("nvme", 0, len(payload))
                self.usage["nvme"] += len(payload)
                self._touch("nvme", key)

    def stats(self) -> dict:
        total_reads = sum(self.read_counts.values()) or 1
        hit_rate = {tier: self.hits[tier] / total_reads for tier in self.hits}
        mean_latency = {
            tier: (sum(values) / len(values) if values else 0.0)
            for tier, values in self.read_latency_ms.items()
        }
        return {
            "hit_rate": hit_rate,
            "usage_bytes": dict(self.usage),
            "mean_read_latency_ms": mean_latency,
        }
