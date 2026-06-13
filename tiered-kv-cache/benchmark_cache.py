from __future__ import annotations

import statistics
import threading
import time

import torch
from rich.console import Console
from rich.table import Table

from tiered_kv_cache import TieredKVCache


console = Console()


class GPUOnlyBaseline:
    def __init__(self, gpu_budget_gb: float) -> None:
        self.gpu_budget_bytes = int(gpu_budget_gb * (1024**3))
        self.usage = 0
        self.store = {}

    def write(self, key: tuple[int, int], kv: torch.Tensor) -> None:
        size = kv.numel() * kv.element_size()
        if self.usage + size > self.gpu_budget_bytes:
            raise MemoryError("GPU-only baseline OOM")
        self.store[key] = kv
        self.usage += size


def main() -> None:
    cache = TieredKVCache(gpu_budget_gb=0.01, cpu_budget_gb=0.02, nvme_path="./kv_nvme_store")
    baseline = GPUOnlyBaseline(gpu_budget_gb=0.01)
    latencies = []
    baseline_oom = False

    def _session(session_id: int) -> None:
        for block_id in range(1000):
            kv = torch.randn(32, 8, 64, dtype=torch.float16)
            cache.write(session_id, block_id, kv)
            started = time.perf_counter()
            cache.read(session_id, block_id)
            latencies.append((time.perf_counter() - started) * 1000.0)
            if not baseline_oom:
                try:
                    baseline.write((session_id, block_id), kv)
                except MemoryError:
                    nonlocal_baseline_oom[0] = True

    nonlocal_baseline_oom = [False]
    threads = [threading.Thread(target=_session, args=(session_id,)) for session_id in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stats = cache.stats()
    table = Table(title="Tiered KV Cache Benchmark")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("GPU hit rate", f"{stats['hit_rate']['gpu']:.3f}")
    table.add_row("CPU hit rate", f"{stats['hit_rate']['cpu']:.3f}")
    table.add_row("NVMe hit rate", f"{stats['hit_rate']['nvme']:.3f}")
    table.add_row("Mean read latency (ms)", f"{statistics.mean(latencies):.3f}")
    table.add_row("GPU mean tier latency (ms)", f"{stats['mean_read_latency_ms']['gpu']:.3f}")
    table.add_row("CPU mean tier latency (ms)", f"{stats['mean_read_latency_ms']['cpu']:.3f}")
    table.add_row("NVMe mean tier latency (ms)", f"{stats['mean_read_latency_ms']['nvme']:.3f}")
    table.add_row("Peak GPU usage (MB)", f"{stats['usage_bytes']['gpu'] / (1024**2):.2f}")
    table.add_row("Peak CPU usage (MB)", f"{stats['usage_bytes']['cpu'] / (1024**2):.2f}")
    table.add_row("NVMe usage (MB)", f"{stats['usage_bytes']['nvme'] / (1024**2):.2f}")
    table.add_row("GPU-only baseline", "OOM" if nonlocal_baseline_oom[0] else "Within budget")
    console.print(table)


if __name__ == "__main__":
    main()
