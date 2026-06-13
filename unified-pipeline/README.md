# Unified Pipeline

```text
task
  |
  v
+-------------------+
| Precision Router  |
+-------------------+
  |
  v
+-------------------+
| Tiered KV Cache   |
+-------------------+
  |
  v
+-------------------+
| Spec Reasoner     |
+-------------------+
  |
  v
+-------------------+
| Multi-Agent Fanout|
+-------------------+
  |
  v
+-------------------+
| Consensus + D/s   |
+-------------------+
```

```bash
pip install -r requirements.txt
cp .env.example .env  # or set MIMO_BASE_URL / MIMO_API_KEY manually
python run_pipeline.py --task "What is 12 * 8?" --domain math
```

- `precision-router/`: chooses a lightweight FP4/FP8/BF16 budget proxy for each task.
- `tiered-kv-cache/`: models KV placement and promotion across GPU, CPU, and NVMe tiers.
- `spec-reasoner/`: drafts and verifies candidate reasoning chains before full generation.
- `multi-agent-orchestrator/`: fans the task out to multiple agents in parallel.
- `ds-benchmark/`: verifies the final answer and converts throughput into D/s-style evaluation.
