"""Blinded comprehension + preference dissociation (Doc-12 §6).

For a generated answer g and an intended proposition set P, blinded raters recover a
proposition set R(g); the adequacy endpoint is proposition precision/recall/F1 —
recovered MEANING, not preference. Preference and comprehension can dissociate: a
system may be visually preferred yet convey less meaning. Inter-rater reliability
reuses the Doc-03 kappa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from ..grammar.signbleu import cohens_kappa


def proposition_prf(recovered: Set, intended: Set) -> Dict[str, float]:
    """Precision/recall/F1 of recovered propositions against the intended set."""
    inter = len(recovered & intended)
    precision = inter / len(recovered) if recovered else (1.0 if not intended else 0.0)
    recall = inter / len(intended) if intended else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def comprehension_f1(recovered: Set, intended: Set) -> float:
    return proposition_prf(recovered, intended)["f1"]


def mean_comprehension_f1(items: Sequence) -> float:
    """Mean proposition-F1 over (recovered, intended) items."""
    if not items:
        raise ValueError("no items")
    return sum(comprehension_f1(r, p) for r, p in items) / len(items)


def proposition_agreement(rater_a: Sequence[Set], rater_b: Sequence[Set],
                          universe: Sequence) -> float:
    """Cohen's kappa on per-proposition present/absent judgements (reuse Doc-03).

    Each rater's recovered set becomes a binary vector over the proposition
    ``universe``; kappa measures inter-rater reliability of comprehension coding.
    """
    a_bits: List[int] = []
    b_bits: List[int] = []
    for ra, rb in zip(rater_a, rater_b):
        for prop in universe:
            a_bits.append(1 if prop in ra else 0)
            b_bits.append(1 if prop in rb else 0)
    return cohens_kappa(a_bits, b_bits)


# ---------------------------------------------------------------------------
# preference / comprehension dissociation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemScores:
    system: str
    preference: float          # visual preference (e.g. Bradley-Terry / mean rank)
    comprehension_f1: float    # recovered-meaning adequacy


def preference_comprehension_dissociate(a: SystemScores, b: SystemScores) -> bool:
    """True iff preference and comprehension order the two systems OPPOSITELY.

    Demonstrates the document's separation: a preferred system can convey LESS
    meaning, so preference cannot substitute for comprehension.
    """
    dp = a.preference - b.preference
    dc = a.comprehension_f1 - b.comprehension_f1
    return dp * dc < 0


# ---------------------------------------------------------------------------
# blinded human trial scaffolding
# ---------------------------------------------------------------------------
@dataclass
class BlindedTrial:
    """A blinded comprehension trial: system identity hidden, attention-checked."""

    show_text_before_signing: bool = False        # must be False (no text priming)
    rater_language_backgrounds: tuple = ()         # captured per rater

    def __post_init__(self):
        if self.show_text_before_signing:
            raise ValueError("text must not be shown before signing (priming)")

    def attention_check_pass_rate(self, responses: Sequence, gold: Sequence) -> float:
        if len(responses) != len(gold) or not gold:
            raise ValueError("responses and gold must be parallel and non-empty")
        return sum(1 for r, g in zip(responses, gold) if r == g) / len(gold)
