from __future__ import annotations

import heapq
import threading
import time
from typing import List

from tiered_kv_cache import TieredKVCache


class PrefetchScheduler:
    def __init__(self, tiered_cache: TieredKVCache, layer_id: int = 0) -> None:
        self.tiered_cache = tiered_cache
        self.layer_id = layer_id
        self.cv = threading.Condition()
        self.heap: List[tuple[int, int]] = []
        self._stop = False
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def hint(self, upcoming_block_ids: list[int]) -> None:
        with self.cv:
            now = int(time.time() * 1000)
            for step, block_id in enumerate(upcoming_block_ids):
                heapq.heappush(self.heap, (now + step, block_id))
            self.cv.notify_all()

    def _consume(self) -> None:
        while True:
            with self.cv:
                while not self.heap and not self._stop:
                    self.cv.wait(timeout=0.1)
                if self._stop:
                    return
                _, block_id = heapq.heappop(self.heap)
            try:
                self.tiered_cache.read(self.layer_id, block_id)
            except Exception:
                continue

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        self.thread.join(timeout=1.0)
