from __future__ import annotations

import sys
from pathlib import Path


def configure_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    playground = root.parent
    paths = {
        "repo_root": root,
        "playground": playground,
        "orchestrator": playground / "multi-agent-orchestrator",
        "spec_reasoner": playground / "spec-reasoner",
        "precision_router": playground / "precision-router",
        "tiered_kv": root / "tiered-kv-cache",
        "ds_benchmark": root / "ds-benchmark",
    }
    for path in paths.values():
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return paths

