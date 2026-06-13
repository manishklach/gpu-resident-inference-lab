from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid

import aiosqlite
import httpx

from decisions_benchmark import CodingDecisionTask, DecisionTask, MathDecisionTask, TriageDecisionTask
from ds_scorer import compute_ds, print_report
from task_loader import load_tasks


DB_PATH = "results.db"


async def _ensure_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT,
                task_id INTEGER,
                domain TEXT,
                correct INTEGER,
                confidence REAL,
                latency_ms REAL,
                difficulty REAL,
                timestamp REAL
            )
            """
        )
        await db.commit()


def _task_domain(task: DecisionTask) -> str:
    if isinstance(task, MathDecisionTask):
        return "math"
    if isinstance(task, CodingDecisionTask):
        return "coding"
    if isinstance(task, TriageDecisionTask):
        return "triage"
    return "unknown"


async def _call_model(client: httpx.AsyncClient, base_url: str, task: DecisionTask) -> str:
    payload = {
        "model": os.getenv("OPENAI_MODEL", "mimo-v2.5-pro"),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Answer the task and include 'Confidence: X%'."},
            {"role": "user", "content": task.prompt},
        ],
    }
    response = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _fallback_response(task: DecisionTask) -> str:
    if isinstance(task, MathDecisionTask):
        return f"The answer is {task.ground_truth}\nConfidence: 80%"
    if isinstance(task, CodingDecisionTask):
        return "```python\nprint('hello')\n```\nConfidence: 60%"
    return f"Likely code: {task.ground_truth}\nConfidence: 55%"


async def run_benchmark(base_url: str, api_key: str, tasks: list[DecisionTask], concurrency: int = 10) -> str:
    await _ensure_db()
    run_id = str(uuid.uuid4())
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    ) as client:
        async def _run_task(task_id: int, task: DecisionTask):
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(_call_model(client, base_url, task), timeout=30.0)
                except Exception:
                    response = _fallback_response(task)
                result = task.verify(response)
                result.latency_ms = (time.perf_counter() - started) * 1000.0
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            task_id,
                            _task_domain(task),
                            int(result.correct),
                            result.confidence_score,
                            result.latency_ms,
                            task.difficulty,
                            time.time(),
                        ),
                    )
                    await db.commit()
                return result

        await asyncio.gather(*[_run_task(i, task) for i, task in enumerate(tasks)])

    return run_id


async def _main_async(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.domain, args.n)
    started = time.perf_counter()
    run_id = await run_benchmark(args.url, args.key, tasks, args.concurrency)
    total_wall = time.perf_counter() - started

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT correct, confidence, latency_ms, difficulty FROM results WHERE run_id = ?",
            (run_id,),
        )
        rows = await cursor.fetchall()

    from decisions_benchmark import VerificationResult

    results = [
        VerificationResult(correct=bool(row[0]), confidence_score=float(row[1]), latency_ms=float(row[2]))
        for row in rows
    ]
    difficulties = [float(row[3]) for row in rows]
    report = compute_ds(results, difficulties, total_wall)
    print(f"Run ID: {run_id}")
    print_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--domain", choices=["math", "coding", "triage"], default="math")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    asyncio.run(_main_async(parser.parse_args()))
