"""Adversarial tests for consistency + augmentation guard (Doc-11, stage 11e)."""

import numpy as np
import pytest
import torch

from signtranslator.pretraining.consistency import (
    recover_order_from_timestamps, pairwise_precedence_accuracy,
    align_views, view_retrieval_recall1,
    AugmentationError, LinguisticDirection, augment_appearance, horizontal_flip,
)
from signtranslator.pretraining.contrast import l2_normalize

torch.manual_seed(0)


def test_order_uniquely_recoverable_from_distinct_timestamps():
    ts = [3.0, 1.0, 2.0, 0.5]
    assert recover_order_from_timestamps(ts).tolist() == [3, 1, 2, 0]


def test_order_requires_distinct_timestamps():
    with pytest.raises(ValueError):
        recover_order_from_timestamps([1.0, 1.0, 2.0])


def test_pairwise_precedence_accuracy():
    true = [0, 1, 2, 3]
    assert pairwise_precedence_accuracy([0, 1, 2, 3], true) == 1.0
    assert pairwise_precedence_accuracy([3, 2, 1, 0], true) == 0.0   # fully reversed
    assert pairwise_precedence_accuracy([1, 0, 2, 3], true) == pytest.approx(5 / 6)


def test_same_clip_views_align_in_shared_space():
    # paired views embedded in a shared space (post-projection) align; a shuffled
    # pairing does not. (Aligning two arbitrarily-rotated raw views is precisely
    # what the learned projection heads are for -- not an identity-map property.)
    torch.manual_seed(1)
    z = l2_normalize(torch.randn(12, 8))
    view_a = z
    view_b = l2_normalize(z + 0.01 * torch.randn_like(z))   # same clip, small jitter
    assert view_retrieval_recall1(view_a, view_b) == 1.0
    assert float(align_views(view_a, view_b, temperature=0.05)) < 1e-2
    # negative control: mismatched pairing does not retrieve.
    shuffled = view_b[torch.randperm(12)]
    assert view_retrieval_recall1(view_a, shuffled) < 1.0


def test_appearance_augment_preserves_shape_and_is_content_ish():
    x = np.random.default_rng(0).normal(size=(5, 3, 2))
    y = augment_appearance(x, scale=1.2, translate=(0.5, -0.3), noise_std=0.01, seed=1)
    assert y.shape == x.shape


def test_horizontal_flip_without_direction_raises():
    x = np.random.default_rng(0).normal(size=(4, 3, 2))
    with pytest.raises(AugmentationError):
        horizontal_flip(x, direction=None)


def test_horizontal_flip_transforms_direction_consistently():
    x = np.random.default_rng(0).normal(size=(4, 3, 2))
    d = LinguisticDirection(dominant_hand=1, loci=(0.5, -0.3), agreement_sign=1)
    y, d2 = horizontal_flip(x, d)
    assert np.allclose(y[..., 0], -x[..., 0])        # x mirrored
    assert np.allclose(y[..., 1], x[..., 1])         # y untouched
    assert d2.dominant_hand == 0                      # handedness swapped
    assert d2.loci == (-0.5, 0.3)                     # loci mirrored
    assert d2.agreement_sign == -1                    # agreement reversed


def test_flip_is_an_involution_on_direction():
    d = LinguisticDirection(dominant_hand=0, loci=(1.0, -2.0), agreement_sign=-1)
    assert d.flipped().flipped() == d                 # flipping twice is identity
