"""The loop share: a hypothesis loops when a whitespace 3-gram occurs at least 3 times (overlaps counted).

This is the definition behind ``reports/post_s2_plan.md`` section 1 (real 28, blank 23, a1 54, a4 47 on D20). It reads raw
whitespace tokens, not ``motion_to_text.words``, which splits "you've" into three tokens and reads a1 59.8, a4 52.0, real 33.0,
blank 30.0 on the same hypotheses. ``scripts/comp_a_lib.py`` has the same rule as ``is_loop``; ``tests/test_loop_share.py`` holds the two in step.
"""
from __future__ import annotations

from collections import Counter


def is_loop(hypothesis: str) -> bool:
    """A whitespace 3-gram repeated at least 3 times."""
    t = hypothesis.split()
    counts = Counter(tuple(t[i:i + 3]) for i in range(len(t) - 2))
    return bool(counts) and max(counts.values()) >= 3


def loop_flags(hypotheses) -> list[float]:
    return [float(is_loop(h)) for h in hypotheses]


__all__ = ["is_loop", "loop_flags"]
