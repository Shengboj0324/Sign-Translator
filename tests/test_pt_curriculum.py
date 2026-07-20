"""Adversarial tests for curriculum + frozen baselines (Doc-11, stage 11g)."""

import pytest
import torch
import torch.nn as nn

from signtranslator.pretraining.curriculum import (
    Stage, CURRICULUM, is_monotone_unlock, stage_objective, FrozenBaseline,
)


def test_curriculum_has_five_ordered_stages():
    assert [c.stage for c in CURRICULUM] == [
        Stage.UNIMODAL_MASKED, Stage.CROSSVIEW_ALIGN, Stage.MOTION_TEXT_CONTRAST,
        Stage.SUPERVISED, Stage.LIMITED_E2E,
    ]


def test_unlock_is_monotone():
    assert is_monotone_unlock()
    # final stage unlocks everything the earlier stages did (superset chain).
    assert CURRICULUM[-1].unlocked.issuperset(CURRICULUM[0].unlocked)


def test_non_monotone_curriculum_is_rejected():
    bad = [CURRICULUM[2], CURRICULUM[0]]        # capacity shrinks
    assert not is_monotone_unlock(bad)


def test_stage_objectives_present():
    assert stage_objective(Stage.UNIMODAL_MASKED) == "masked_motion_nll"
    assert stage_objective(Stage.LIMITED_E2E) == "end_to_end_finetune"


def test_frozen_baseline_snapshot_is_bit_identical_when_unchanged():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    fb = FrozenBaseline()
    fb.snapshot("stage1", model)
    assert fb.has("stage1")
    assert fb.matches("stage1", model)               # unchanged -> identical
    assert fb.max_param_drift("stage1", model) == 0.0


def test_frozen_baseline_detects_regression():
    torch.manual_seed(1)
    model = nn.Linear(4, 3)
    fb = FrozenBaseline()
    fb.snapshot("stage1", model)
    with torch.no_grad():
        model.weight += 0.01                          # a later stage changes it
    assert not fb.matches("stage1", model)
    assert fb.max_param_drift("stage1", model) == pytest.approx(0.01, abs=1e-6)


def test_missing_baseline_raises():
    fb = FrozenBaseline()
    with pytest.raises(KeyError):
        fb.max_param_drift("nope", nn.Linear(2, 2))
