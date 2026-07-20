"""Baselines, stratification, and model card (Doc-12 §7).

A system is credible only relative to retrieval/stitching, deterministic seq2seq, and
a human-recorded upper reference; endpoints are sliced (never only aggregate); and a
Model Card (Mitchell et al.) documents the system with mandatory failure reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from .contracts import Direction


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
class BaselineType(Enum):
    RETRIEVAL_STITCHING = "retrieval_stitching"
    DETERMINISTIC_SEQ2SEQ = "deterministic_seq2seq"
    HUMAN_UPPER_REFERENCE = "human_upper_reference"


REQUIRED_BASELINES = (
    BaselineType.RETRIEVAL_STITCHING,
    BaselineType.DETERMINISTIC_SEQ2SEQ,
    BaselineType.HUMAN_UPPER_REFERENCE,
)


def has_required_baselines(present: Dict[BaselineType, float]) -> bool:
    """Every required comparison condition must be present."""
    return all(b in present for b in REQUIRED_BASELINES)


def exceeds_human_upper_reference(system_ci: Tuple[float, float],
                                  human_ci: Tuple[float, float]) -> bool:
    """True iff the system's CI lies strictly ABOVE the human upper reference.

    Such a claim is extraordinary and (per the document) requires independent
    replication before it is believed — the framework flags it.
    """
    return system_ci[0] > human_ci[1]


# ---------------------------------------------------------------------------
# stratification
# ---------------------------------------------------------------------------
def worst_slice(slices: Dict[str, float], direction: Direction) -> Tuple[str, float]:
    """The weakest slice: min value for GE metrics, max value for LE metrics."""
    if not slices:
        raise ValueError("no slices")
    if direction is Direction.GE:
        name = min(slices, key=slices.get)
    else:
        name = max(slices, key=slices.get)
    return name, slices[name]


def aggregate_hides_worst_slice(slices: Dict[str, float], aggregate: float,
                                threshold: float, direction: Direction) -> bool:
    """True iff the aggregate passes the threshold but the worst slice fails it.

    Demonstrates why per-slice reporting is mandatory (Doc-10/Doc-11 discipline).
    """
    _, wv = worst_slice(slices, direction)
    if direction is Direction.GE:
        return aggregate >= threshold and wv < threshold
    return aggregate <= threshold and wv > threshold


# ---------------------------------------------------------------------------
# model card (Mitchell et al. 2019)
# ---------------------------------------------------------------------------
MODEL_CARD_SECTIONS = (
    "model_details", "intended_use", "factors", "metrics", "evaluation_data",
    "training_data", "quantitative_analyses", "ethical_considerations",
    "caveats_and_recommendations",
)


@dataclass
class ModelCard:
    model_details: str = ""
    intended_use: str = ""
    factors: str = ""
    metrics: str = ""
    evaluation_data: str = ""
    training_data: str = ""
    quantitative_analyses: str = ""
    ethical_considerations: str = ""
    caveats_and_recommendations: str = ""
    model_size: str = ""
    train_compute: str = ""
    inference_compute: str = ""
    latency: str = ""
    failure_modes: tuple = ()          # mandatory: report failures

    def missing_sections(self) -> List[str]:
        missing = [s for s in MODEL_CARD_SECTIONS if not getattr(self, s)]
        for extra in ("model_size", "train_compute", "inference_compute", "latency"):
            if not getattr(self, extra):
                missing.append(extra)
        if not self.failure_modes:
            missing.append("failure_modes")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_sections()
