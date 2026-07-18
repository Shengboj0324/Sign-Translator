"""Fail-closed output policy.

The source specification's safety rule: *"Fail closed on low confidence: pause
or fingerspell a verified item instead of hallucinating a sign."*

The failure mode this prevents is specific and serious. A sign is a confident,
fluent assertion; a Deaf viewer has no way to tell a guessed sign from a correct
one. Emitting a plausible-looking sign for a word the recogniser is unsure about
is therefore *worse* than emitting nothing, because it destroys the viewer's
ability to detect the error. Silence and fingerspelling are both legible as
"the system is unsure" -- a wrong sign is not.

Two axes drive the decision, not one:

* **confidence** -- calibrated, from :mod:`calibration`. An uncalibrated score
  makes every threshold here meaningless, which is why Stage 3 pairs them.
* **lexicon coverage** -- whether a token has a known sign at all, and whether
  it is a *verified* word. Fingerspelling an unverified token would just move
  the hallucination from the sign channel to the spelling channel, so it is
  forbidden.

Actions are ordered ``PAUSE < FINGERSPELL < EMIT`` by how much they assert. The
policy is monotone in confidence with respect to that order: more evidence can
never produce a more conservative action, and less evidence can never produce a
bolder one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch


class Action(IntEnum):
    """Ordered by assertiveness, so monotonicity is expressible as <=."""

    PAUSE = 0
    FINGERSPELL = 1
    EMIT = 2


@dataclass
class PolicyDecision:
    token: int
    confidence: float
    action: Action
    reason: str

    @property
    def asserts_a_sign(self) -> bool:
        return self.action is Action.EMIT


@dataclass
class PolicyOutcome:
    decisions: List[PolicyDecision] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.decisions)

    def count(self, action: Action) -> int:
        return sum(1 for d in self.decisions if d.action is action)

    @property
    def coverage(self) -> float:
        """Fraction of tokens rendered as signs (the system's assertion rate)."""
        return self.count(Action.EMIT) / len(self.decisions) if self.decisions else 0.0

    @property
    def abstention_rate(self) -> float:
        return 1.0 - self.coverage

    def emitted_tokens(self) -> List[int]:
        return [d.token for d in self.decisions if d.action is Action.EMIT]

    def report(self) -> str:
        return (f"emit={self.count(Action.EMIT)} "
                f"fingerspell={self.count(Action.FINGERSPELL)} "
                f"pause={self.count(Action.PAUSE)} "
                f"coverage={self.coverage:.3f}")


class FailClosedPolicy:
    """Maps (token, calibrated confidence) to an output action.

    Args:
        emit_threshold: minimum confidence to render a sign.
        fingerspell_threshold: minimum confidence to spell a verified word.
            Must not exceed ``emit_threshold`` -- otherwise the policy would
            demand *more* evidence for the safer action, which is incoherent.
        sign_lexicon: tokens that have a known sign. ``None`` means "all tokens
            have signs", which is only appropriate for a closed vocabulary.
        verified_lexicon: tokens known to be real words. Defaults to the sign
            lexicon. A token outside it can never be fingerspelled.
    """

    def __init__(self, emit_threshold: float = 0.8,
                 fingerspell_threshold: float = 0.4,
                 sign_lexicon: Optional[Iterable[int]] = None,
                 verified_lexicon: Optional[Iterable[int]] = None) -> None:
        if not 0.0 <= fingerspell_threshold <= emit_threshold <= 1.0:
            raise ValueError(
                "require 0 <= fingerspell_threshold <= emit_threshold <= 1; "
                "the safer action must not need more evidence than the bolder one")
        self.emit_threshold = emit_threshold
        self.fingerspell_threshold = fingerspell_threshold
        self.sign_lexicon: Optional[Set[int]] = (
            None if sign_lexicon is None else {int(t) for t in sign_lexicon})
        if verified_lexicon is not None:
            self.verified_lexicon: Optional[Set[int]] = {int(t) for t in verified_lexicon}
        elif self.sign_lexicon is not None:
            self.verified_lexicon = set(self.sign_lexicon)
        else:
            self.verified_lexicon = None

    def _has_sign(self, token: int) -> bool:
        return self.sign_lexicon is None or token in self.sign_lexicon

    def _is_verified(self, token: int) -> bool:
        return self.verified_lexicon is None or token in self.verified_lexicon

    # Posteriors are produced by summing floats, so a legitimate value can land
    # an ulp outside [0, 1] (e.g. 1.0000000000000002). Rejecting that would make
    # the policy crash on valid input; silently accepting 1.5 would hide a real
    # upstream bug. So: tolerate rounding, clamp, and reject anything beyond.
    _CONFIDENCE_TOLERANCE = 1e-6

    def decide(self, token: int, confidence: float) -> PolicyDecision:
        token = int(token)
        c = float(confidence)
        tol = self._CONFIDENCE_TOLERANCE
        if not (-tol <= c <= 1.0 + tol):
            raise ValueError(
                f"confidence must be a probability in [0, 1], got {c}")
        c = min(1.0, max(0.0, c))

        if c >= self.emit_threshold and self._has_sign(token):
            return PolicyDecision(token, c, Action.EMIT, "confident, sign known")
        if c >= self.fingerspell_threshold and self._is_verified(token):
            # Either the confidence is too low for a sign, or the word has no
            # sign at all (names, technical terms). Spelling is the honest
            # fallback in both cases.
            reason = ("no sign in lexicon" if not self._has_sign(token)
                      else "below emit threshold")
            return PolicyDecision(token, c, Action.FINGERSPELL, reason)
        reason = ("unverified token" if not self._is_verified(token)
                  else "below fingerspell threshold")
        return PolicyDecision(token, c, Action.PAUSE, reason)

    def decide_sequence(self, tokens: Sequence[int],
                        confidences: Sequence[float]) -> PolicyOutcome:
        if len(tokens) != len(confidences):
            raise ValueError("tokens and confidences must have equal length")
        return PolicyOutcome([self.decide(t, c)
                              for t, c in zip(tokens, confidences)])


# ---------------------------------------------------------------------------
# Selective prediction (risk vs. coverage)
# ---------------------------------------------------------------------------
@dataclass
class SelectivePoint:
    threshold: float
    coverage: float
    selective_accuracy: float

    @property
    def risk(self) -> float:
        return 1.0 - self.selective_accuracy


def selective_metrics(confidences, correct, threshold: float) -> SelectivePoint:
    """Coverage and accuracy among predictions at or above ``threshold``.

    This is the quantity a fail-closed system is actually optimising: not raw
    accuracy, but accuracy *on what it chose to assert*, traded against how
    often it asserts anything.
    """
    c = torch.as_tensor(confidences, dtype=torch.float64).flatten()
    y = torch.as_tensor(correct).flatten().bool()
    if c.shape != y.shape:
        raise ValueError("confidences and correct must have equal length")
    if c.numel() == 0:
        raise ValueError("empty input")
    kept = c >= threshold
    n_kept = int(kept.sum())
    coverage = n_kept / c.numel()
    acc = float(y[kept].double().mean()) if n_kept else 1.0
    return SelectivePoint(threshold=threshold, coverage=coverage,
                          selective_accuracy=acc)


def risk_coverage_curve(confidences, correct,
                        thresholds: Optional[Sequence[float]] = None
                        ) -> List[SelectivePoint]:
    """Sweep thresholds to trace the risk/coverage tradeoff."""
    if thresholds is None:
        thresholds = [i / 20.0 for i in range(21)]
    return [selective_metrics(confidences, correct, t) for t in thresholds]


def area_under_risk_coverage(confidences, correct,
                             thresholds: Optional[Sequence[float]] = None) -> float:
    """Mean risk across the coverage sweep (lower is better).

    Summarises the whole curve, so two policies can be compared without picking
    an operating point first.
    """
    pts = risk_coverage_curve(confidences, correct, thresholds)
    return sum(p.risk for p in pts) / len(pts)
