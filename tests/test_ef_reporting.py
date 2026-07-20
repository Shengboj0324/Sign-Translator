"""Adversarial tests for baselines + stratification + model card (Doc-12, 12g)."""

import pytest

from signtranslator.eval_framework.contracts import Direction
from signtranslator.eval_framework.reporting import (
    BaselineType, REQUIRED_BASELINES, has_required_baselines,
    exceeds_human_upper_reference, worst_slice, aggregate_hides_worst_slice,
    MODEL_CARD_SECTIONS, ModelCard,
)


def test_required_baselines_enforced():
    present = {b: 0.5 for b in REQUIRED_BASELINES}
    assert has_required_baselines(present)
    del present[BaselineType.HUMAN_UPPER_REFERENCE]
    assert not has_required_baselines(present)


def test_exceeding_human_upper_reference_is_flagged():
    # system CI strictly above human CI -> extraordinary, needs replication.
    assert exceeds_human_upper_reference((0.92, 0.96), (0.85, 0.90))
    # overlapping CIs -> not flagged.
    assert not exceeds_human_upper_reference((0.86, 0.94), (0.85, 0.90))


def test_worst_slice_direction_aware():
    slices = {"short": 0.9, "long": 0.6, "occluded": 0.4}
    assert worst_slice(slices, Direction.GE) == ("occluded", 0.4)
    # for a LE metric (error), worst == largest.
    errs = {"clean": 0.05, "noisy": 0.30}
    assert worst_slice(errs, Direction.LE) == ("noisy", 0.30)


def test_aggregate_can_hide_a_failing_slice():
    # aggregate 0.82 passes >=0.8, but the occluded slice (0.4) fails it.
    slices = {"short": 0.95, "long": 0.9, "occluded": 0.4}
    assert aggregate_hides_worst_slice(slices, aggregate=0.82, threshold=0.8,
                                       direction=Direction.GE)


def test_model_card_incomplete_until_all_sections_and_failures():
    card = ModelCard(model_details="x")
    assert "intended_use" in card.missing_sections()
    assert "failure_modes" in card.missing_sections()
    assert not card.is_complete()


def test_complete_model_card():
    kw = {s: "filled" for s in MODEL_CARD_SECTIONS}
    card = ModelCard(model_size="120M", train_compute="10 GPU-h",
                     inference_compute="0.1 GFLOP", latency="90 ms",
                     failure_modes=("fails on heavy occlusion",), **kw)
    assert card.missing_sections() == []
    assert card.is_complete()


def test_model_card_requires_failure_reporting():
    kw = {s: "filled" for s in MODEL_CARD_SECTIONS}
    card = ModelCard(model_size="1M", train_compute="1h", inference_compute="1",
                     latency="10ms", failure_modes=(), **kw)
    assert not card.is_complete()                       # no failures reported
    assert "failure_modes" in card.missing_sections()
