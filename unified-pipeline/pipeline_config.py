from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass
class PipelineConfig:
    mimo_base_url: str = "https://api.platform.xiaomimimo.com/v1"
    mimo_api_key: str = ""
    n_agents: int = 3
    precision_budget: str = "balanced"
    kv_gpu_gb: float = 4.0
    kv_cpu_gb: float = 16.0
    kv_nvme_path: str = "/tmp/kv_cache"
    spec_n_drafts: int = 3
    spec_strategy: str = "threshold"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        if load_dotenv is not None:
            load_dotenv()
        return cls(
            mimo_base_url=os.getenv("MIMO_BASE_URL", cls.mimo_base_url),
            mimo_api_key=os.getenv("MIMO_API_KEY", cls.mimo_api_key),
            n_agents=int(os.getenv("PIPELINE_N_AGENTS", cls.n_agents)),
            precision_budget=os.getenv("PIPELINE_PRECISION_BUDGET", cls.precision_budget),
            kv_gpu_gb=float(os.getenv("PIPELINE_KV_GPU_GB", cls.kv_gpu_gb)),
            kv_cpu_gb=float(os.getenv("PIPELINE_KV_CPU_GB", cls.kv_cpu_gb)),
            kv_nvme_path=os.getenv("PIPELINE_KV_NVME_PATH", cls.kv_nvme_path),
            spec_n_drafts=int(os.getenv("PIPELINE_SPEC_N_DRAFTS", cls.spec_n_drafts)),
            spec_strategy=os.getenv("PIPELINE_SPEC_STRATEGY", cls.spec_strategy),
        )

