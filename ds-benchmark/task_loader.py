from __future__ import annotations

import random

from decisions_benchmark import CodingDecisionTask, DecisionTask, MathDecisionTask, TriageDecisionTask

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


def _fallback_tasks(domain: str, n: int) -> list[DecisionTask]:
    if domain == "math":
        tasks = [
            MathDecisionTask("What is 2 + 2?", 0.2, "4"),
            MathDecisionTask("What is 17 * 34?", 0.4, "578"),
        ]
    elif domain == "coding":
        tasks = [
            CodingDecisionTask("Write Python that prints hello", 0.3, "hello"),
            CodingDecisionTask("Write Python that prints 4", 0.2, "4"),
        ]
    else:
        tasks = [
            TriageDecisionTask("Chest pain with shortness of breath", 0.8, "R07.9,I20.9"),
            TriageDecisionTask("Fever and cough", 0.8, "R50.9,J06.9"),
        ]
    return tasks[:n]


def load_tasks(domain: str, n: int = 100) -> list[DecisionTask]:
    random.seed(42)
    if load_dataset is None:
        return _fallback_tasks(domain, n)

    tasks: list[DecisionTask] = []
    try:
        if domain == "math":
            dataset = load_dataset("gsm8k", "main", split=f"test[:{n}]")
            for row in dataset:
                answer = str(row["answer"]).split("####")[-1].strip()
                digits = len("".join(ch for ch in answer if ch.isdigit()))
                difficulty = min(1.0, max(0.1, 1 - digits / 10))
                tasks.append(MathDecisionTask(row["question"], difficulty, answer))
        elif domain == "coding":
            dataset = load_dataset("openai_humaneval", split=f"test[:{n}]")
            for row in dataset:
                prompt = row["prompt"]
                difficulty = min(1.0, max(0.1, len(prompt) / 1000.0))
                tasks.append(CodingDecisionTask(prompt, difficulty, ""))
        elif domain == "triage":
            dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=f"train[:{n}]")
            for row in dataset:
                prompt = row["question"]
                answer = str(row.get("answer", "R69"))
                tasks.append(TriageDecisionTask(prompt, 0.8, answer))
    except Exception:
        return _fallback_tasks(domain, n)

    random.shuffle(tasks)
    return tasks[:n]
