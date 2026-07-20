"""Verification of linguistically-aware LOD and consent gating.

The decisive property: fingers and facial cues are kept at EVERY LOD level, while
lower tiers decimate. Also proves budget monotonicity and consent/non-impersonation
gating.
"""

import pytest
import torch

from signtranslator.avatar_render.lod import (
    ImportanceTier, lod_keep_mask, fingers_face_always_kept, budget_curve,
    AppearanceConsent, can_render_identity, requires_synthetic_marker,
)


def _tiers():
    # a small avatar: background, torso, arms, palm, face, fingers vertices
    return torch.tensor([
        int(ImportanceTier.BACKGROUND), int(ImportanceTier.BACKGROUND),
        int(ImportanceTier.TORSO), int(ImportanceTier.TORSO),
        int(ImportanceTier.ARMS),
        int(ImportanceTier.HANDS_PALM),
        int(ImportanceTier.FACE), int(ImportanceTier.FACE),
        int(ImportanceTier.FINGERS), int(ImportanceTier.FINGERS), int(ImportanceTier.FINGERS),
    ])


# ---------------------------------------------------------------------------
# the guarantee
# ---------------------------------------------------------------------------
def test_fingers_and_face_kept_at_every_level():
    tiers = _tiers()
    assert fingers_face_always_kept(tiers)
    # explicit per-level check
    finger_face = (tiers == int(ImportanceTier.FINGERS)) | (tiers == int(ImportanceTier.FACE))
    for level in range(6):
        keep = lod_keep_mask(tiers, level)
        assert bool(torch.all(keep[finger_face]))            # never dropped


def test_lower_tiers_are_decimated_at_high_levels():
    tiers = _tiers()
    keep0 = lod_keep_mask(tiers, 0)                          # keep everything
    keep5 = lod_keep_mask(tiers, 5)                          # most aggressive
    assert bool(torch.all(keep0))                            # level 0 keeps all
    # background/torso/arms are dropped at the highest level, fingers/face remain
    assert not bool(keep5[0])                                # a background vertex dropped
    finger_idx = (tiers == int(ImportanceTier.FINGERS)).nonzero().flatten()
    assert bool(torch.all(keep5[finger_idx]))


def test_budget_curve_is_monotone_non_increasing():
    curve = budget_curve(_tiers())
    assert curve == sorted(curve, reverse=True)              # non-increasing
    assert curve[0] == len(_tiers())                         # level 0 keeps all
    # never drops below the protected (fingers+face) count
    tiers = _tiers()
    protected = int(((tiers == int(ImportanceTier.FINGERS))
                     | (tiers == int(ImportanceTier.FACE))).sum())
    assert curve[-1] >= protected


# ---------------------------------------------------------------------------
# consent / non-impersonation
# ---------------------------------------------------------------------------
def test_identity_render_requires_consent():
    assert not can_render_identity(AppearanceConsent("s1", consented=False))
    assert can_render_identity(AppearanceConsent("s1", consented=True))


def test_synthetic_marker_default_and_opt_out():
    # no consent -> must mark synthetic
    assert requires_synthetic_marker(AppearanceConsent("s1", consented=False))
    # consent but non-impersonation still on -> still marked
    assert requires_synthetic_marker(AppearanceConsent("s1", True, non_impersonation_mode=True))
    # consent AND explicitly opted out of non-impersonation -> may drop the marker
    assert not requires_synthetic_marker(
        AppearanceConsent("s1", True, non_impersonation_mode=False))
