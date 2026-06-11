#!/usr/bin/env python3
"""
Lightweight guardrail: scan repo documents for overclaiming phrases.

Flags phrases that imply real 1T inference, real 1K TPS, or unqualified
performance numbers. If a line also contains a disclaimer phrase, the
line is not flagged. Designed to be advisory — does not fail CI.

Usage:
    python scripts/check_claims.py [--strict]
    python scripts/check_claims.py [files...]
"""

import re
import sys
from pathlib import Path


DANGER_PATTERNS = [
    (r"(?i)achieves?\s+1K\s+TPS", "Implies 1K TPS achievement (use 'toward 1K+ TPS')"),
    (r"(?i)serves?\s+(real\s+)?1T\s+(model|parameter)", "Implies 1T model serving (use '1T-class')"),
    (r"(?i)production\s+(LLM|transformer)?\s*(inference|runtime)", "May imply production readiness (use 'control-flow scaffold')"),
    (r"(?i)real\s+transformer\s+inference", "May overclaim (use 'future real fused inference path')"),
    (r"(?i)6\.7x", "Unqualified 6.7x number — must cite source and context"),
    (r"(?i)30[-–]60%\s*%?\s*of\s+total\s+decode", "Unqualified 30-60% claim"),
    (r"(?i)can\s+dwarf\s+the\s+(first|compute)", "Overly strong language — use 'first-order latency term'"),
    (r"(?i)Xiaomi\s+DFlash", "May imply real DFlash implementation (use 'DFlash-style')"),
    (r"(?i)TileRT", "May imply real TileRT implementation (use 'conceptual CUDA scaffold')"),
    (r"(?i)1T\s+inference", "May overclaim (qualify with '1T-class' or 'future')"),
    (r"(?i)1000\s+TPS", "Implies 1000 TPS achievement (use 'toward 1K+ TPS')"),
]

DISCLAIMER_PHRASES = [
    "does not implement",
    "not real",
    "not a real",
    "not an implementation",
    "not compatible",
    "fake deterministic",
    "conceptual",
    "scaffold",
    "does not claim",
    "not production",
    "control-flow",
    "measurement disclaimer",
    "what is fake",
    "fake math",
    "deterministic fake",
    "not a working",
    "not true",
    "not the",
    "not a ",
    "not claiming",
    "reproducing",
    "inspired",
    "related work",
    "demonstrates",
]


def line_contains_disclaimer(line: str) -> bool:
    lowered = line.lower()
    for phrase in DISCLAIMER_PHRASES:
        if phrase in lowered:
            return True
    return False


def scan_file(path: Path, strict: bool = False) -> list[str]:
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [f"  Cannot read {path}: {e}"]

    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, msg in DANGER_PATTERNS:
            for match in re.finditer(pattern, line):
                if line_contains_disclaimer(line):
                    continue
                issues.append(f"  {path.name}:{lineno}: {msg}")
                issues.append(f"    -> {line.strip()[:120]}")
    return issues


def main() -> None:
    strict = "--strict" in sys.argv
    targets: list[Path] = []

    # If specific files given, use those
    file_args = [a for a in sys.argv[1:] if a != "--strict"]
    if file_args:
        for arg in file_args:
            p = Path(arg)
            if p.is_file():
                targets.append(p)
    else:
        default_dirs = [Path("docs"), Path(".")]
        for d in default_dirs:
            if d.is_dir():
                for f in d.rglob("*.md"):
                    targets.append(f)
        blog = Path("docs/index.html")
        if blog.is_file():
            targets.append(blog)

    all_issues: list[str] = []
    for t in sorted(set(targets)):
        issues = scan_file(t, strict=strict)
        if issues:
            all_issues.append(f"\n{t}:")
            all_issues.extend(issues)

    if all_issues:
        print("=" * 60)
        print("Claim Check — Advisory Report")
        print("=" * 60)
        for line in all_issues:
            print(line)
        print()
        print(f"Found {len(all_issues)} potential issues across {len(set(targets))} files.")
        print()
        print("Advisory only. Pass --strict to fail on findings." if not strict else "Strict mode: EXIT with warnings.")
        print()
        sys.exit(1 if strict else 0)
    else:
        print("Claim check: no overclaiming phrases found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
