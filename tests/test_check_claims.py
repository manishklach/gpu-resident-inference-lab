"""Tests for the claim checker guardrail script."""

from scripts.check_claims import line_contains_disclaimer

# Lines that should NOT be flagged because they contain disclaimers
SAFE_LINES = [
    "This repo models these ideas with fake deterministic math and "
    "lifecycle counters. It does not implement Xiaomi DFlash, TileRT, "
    "or real transformer inference.",
    "It does not implement Xiaomi DFlash or real transformer inference.",
    "This is a conceptual sketch, not real TileRT.",
    "This is not a real DFlash implementation.",
    "Not production LLM inference — control-flow scaffold only.",
    "This is not true 1T inference; it is a measurement scaffold.",
    "This is not an implementation of DiffusionGemma.",
    "All math is fake deterministic.",
    "This is a scaffold, not production transformer runtime.",
    "Not compatible with Google's implementation.",
    "Does not claim 1000 TPS or real 1T inference.",
    "This is not a working diffusion model.",
    "1T inference is not the current target — this measures orchestration.",
]

# Lines that SHOULD be flagged (no disclaimer)
RISKY_LINES = [
    "This repo implements Xiaomi DFlash and achieves 1000 TPS.",
    "Our system serves real 1T models at 1000 TPS.",
    "Implements real TileRT for production LLM inference.",
    "Achieves 6.7x speedup on real transformer inference.",
    "Can dwarf the first-order compute cost for 1T parameter models.",
    "This is a production LLM inference runtime.",
    "30-60% of total decode time is eliminated.",
]


class TestClaimChecker:
    def test_safe_lines_have_disclaimer(self) -> None:
        for line in SAFE_LINES:
            msg = f"'{line[:60]}...' should have a disclaimer"
            assert line_contains_disclaimer(line), msg

    def test_risky_lines_lack_disclaimer(self) -> None:
        for line in RISKY_LINES:
            msg = f"'{line[:60]}...' should lack disclaimer"
            assert not line_contains_disclaimer(line), msg
