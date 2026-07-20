"""Falsifiable contract chain (Doc-12 §1).

A contract passes iff its metric meets a threshold in the required direction, and it
carries a MANDATORY caveat (constructing one without a caveat raises). A chain over
the metric stack is adequate iff EVERY contract passes — so a single passing metric
(BLEU/WER/FID/MPJPE) is necessary but never sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Direction(Enum):
    GE = "ge"   # higher is better: pass iff value >= threshold
    LE = "le"   # lower is better:  pass iff value <= threshold


@dataclass(frozen=True)
class Contract:
    name: str
    layer: str
    value: float
    threshold: float
    direction: Direction
    caveat: str

    def __post_init__(self):
        if not self.caveat or not self.caveat.strip():
            raise ValueError(
                f"contract {self.name!r} must carry its mandatory caveat")
        if self.direction not in (Direction.GE, Direction.LE):
            raise ValueError("direction must be Direction.GE or Direction.LE")

    @property
    def passed(self) -> bool:
        if self.direction is Direction.GE:
            return self.value >= self.threshold
        return self.value <= self.threshold


@dataclass
class EvaluationChain:
    """A conjunction of falsifiable contracts across the metric stack."""

    contracts: List[Contract]

    @property
    def adequate(self) -> bool:
        """Adequate iff EVERY contract passes (a single metric is insufficient)."""
        return bool(self.contracts) and all(c.passed for c in self.contracts)

    @property
    def failures(self) -> List[Contract]:
        return [c for c in self.contracts if not c.passed]

    def layers(self) -> List[str]:
        seen: List[str] = []
        for c in self.contracts:
            if c.layer not in seen:
                seen.append(c.layer)
        return seen

    def report(self) -> str:
        lines = [f"adequate={self.adequate}"]
        for c in self.contracts:
            arrow = ">=" if c.direction is Direction.GE else "<="
            status = "PASS" if c.passed else "FAIL"
            lines.append(
                f"  [{status}] {c.layer}/{c.name}: {c.value:.4g} {arrow} "
                f"{c.threshold:.4g}  ({c.caveat})")
        return "\n".join(lines)
