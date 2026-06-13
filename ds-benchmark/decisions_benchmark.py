from __future__ import annotations

import abc
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

try:
    import resource
except Exception:  # pragma: no cover
    resource = None


@dataclass
class VerificationResult:
    correct: bool
    confidence_score: float
    latency_ms: float


@dataclass
class DecisionTask(abc.ABC):
    prompt: str
    difficulty: float
    ground_truth: str

    @abc.abstractmethod
    def verify(self, response: str) -> VerificationResult:
        raise NotImplementedError

    def _parse_confidence(self, response: str) -> float:
        match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)%", response, re.IGNORECASE)
        if not match:
            return 0.5
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))


@dataclass
class CodingDecisionTask(DecisionTask):
    def verify(self, response: str) -> VerificationResult:
        started = time.perf_counter()
        confidence = self._parse_confidence(response)
        blocks = re.findall(r"```python\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
        code = blocks[0].strip() if blocks else response.strip()
        correct = False

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            script_path = handle.name

        def _limit() -> None:
            if resource is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
                resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

        try:
            completed = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5,
                preexec_fn=_limit if os.name != "nt" else None,
            )
            output = completed.stdout.strip()
            correct = output == self.ground_truth.strip()
        except Exception:
            correct = False
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

        latency_ms = (time.perf_counter() - started) * 1000.0
        return VerificationResult(correct=correct, confidence_score=confidence, latency_ms=latency_ms)


@dataclass
class MathDecisionTask(DecisionTask):
    def verify(self, response: str) -> VerificationResult:
        started = time.perf_counter()
        confidence = self._parse_confidence(response)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
        guess = float(numbers[-1]) if numbers else math.inf
        correct = abs(guess - float(self.ground_truth)) <= 0.01
        latency_ms = (time.perf_counter() - started) * 1000.0
        return VerificationResult(correct=correct, confidence_score=confidence, latency_ms=latency_ms)


@dataclass
class TriageDecisionTask(DecisionTask):
    def verify(self, response: str) -> VerificationResult:
        started = time.perf_counter()
        confidence = self._parse_confidence(response)
        valid_codes = {code.strip().upper() for code in self.ground_truth.split(",")}
        found_codes = {code.upper() for code in re.findall(r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?", response)}
        correct = bool(valid_codes & found_codes)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return VerificationResult(correct=correct, confidence_score=confidence, latency_ms=latency_ms)
