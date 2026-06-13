# D/s Benchmark

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=... OPENAI_BASE_URL=...
python runner.py --domain math --n 50
python runner.py --domain coding --n 50
python compare_runs.py <run_id_1> <run_id_2>
```
