"""Tests for external keypoint-layout adapters."""

import pytest
import torch

from signtranslator.data.adapters import (
    KeypointAdapter, mediapipe_holistic_adapter, openpose_adapter,
    MEDIAPIPE_POSE_MAP, RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET,
)
from signtranslator.data.quality import clean_pose
from signtranslator.skeleton import NUM_DEFAULT_JOINTS


def _mp_inputs(t=8):
    body = torch.randn(3, t, 33)          # MediaPipe pose: 33 landmarks
    rh = torch.randn(3, t, 21)            # 21 landmarks per hand
    lh = torch.randn(3, t, 21)
    return body, rh, lh


def test_mediapipe_adapter_output_shape():
    ad = mediapipe_holistic_adapter()
    body, rh, lh = _mp_inputs()
    res = ad(body, rh, lh)
    assert res.pose.shape == (3, 8, NUM_DEFAULT_JOINTS)
    assert res.missing.shape == res.pose.shape


def test_direct_index_mapping_is_exact():
    """A 1:1 mapped joint must copy the source coordinates verbatim."""
    ad = mediapipe_holistic_adapter()
    body, rh, lh = _mp_inputs()
    res = ad(body, rh, lh)
    # target 3 (r_shoulder) <- source 12 ; target 0 (head) <- source 0 (nose)
    assert torch.allclose(res.pose[:, :, 3], body[:, :, 12], atol=1e-6)
    assert torch.allclose(res.pose[:, :, 0], body[:, :, 0], atol=1e-6)


def test_midpoint_mapping_is_the_mean():
    """The neck is defined as the shoulder midpoint."""
    ad = mediapipe_holistic_adapter()
    body, rh, lh = _mp_inputs()
    res = ad(body, rh, lh)
    expected = (body[:, :, 11] + body[:, :, 12]) / 2
    assert torch.allclose(res.pose[:, :, 1], expected, atol=1e-6)


def test_hand_offsets_are_applied():
    ad = mediapipe_holistic_adapter()
    body, rh, lh = _mp_inputs()
    res = ad(body, rh, lh)
    # right palm (target 9) <- right-hand landmark 0
    assert torch.allclose(res.pose[:, :, RIGHT_HAND_OFFSET], rh[:, :, 0], atol=1e-6)
    assert torch.allclose(res.pose[:, :, LEFT_HAND_OFFSET], lh[:, :, 0], atol=1e-6)


def test_absent_hand_is_marked_missing_not_zero_filled_silently():
    ad = mediapipe_holistic_adapter()
    body, rh, _ = _mp_inputs()
    res = ad(body, right_hand=rh, left_hand=None)
    left_slice = res.missing[:, :, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 9]
    assert bool(left_slice.all())                     # every left-hand joint flagged
    right_slice = res.missing[:, :, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 9]
    assert not bool(right_slice.any())


def test_low_confidence_keypoints_marked_missing():
    ad = mediapipe_holistic_adapter(conf_threshold=0.5)
    body, rh, lh = _mp_inputs(t=8)
    conf = torch.ones(8, 33)
    conf[:, 12] = 0.1                                 # r_shoulder unreliable
    res = ad(body, rh, lh, body_conf=conf)
    assert bool(res.missing[:, :, 3].all())           # target 3 <- source 12
    assert not bool(res.missing[:, :, 0].any())       # nose still confident


def test_midpoint_confidence_is_the_minimum():
    """A midpoint joint is only as trustworthy as its worst contributor."""
    ad = mediapipe_holistic_adapter(conf_threshold=0.5)
    body, rh, lh = _mp_inputs(t=8)
    conf = torch.ones(8, 33)
    conf[:, 11] = 0.2                                 # one shoulder unreliable
    res = ad(body, rh, lh, body_conf=conf)
    assert bool(res.missing[:, :, 1].all())           # neck = midpoint(11,12)


def test_adapter_output_feeds_cleaning_pipeline():
    """Adapter -> cleaning integration: flagged joints get interpolated away."""
    ad = mediapipe_holistic_adapter()
    body, rh, _ = _mp_inputs(t=12)
    res = ad(body, right_hand=rh, left_hand=None)     # left hand missing
    pose = res.pose.clone()
    pose[res.missing] = float("nan")                  # mark for the cleaner
    cleaned, kept, rep = clean_pose(pose.unsqueeze(0), drop_if_missing_rate_above=0.9)
    assert torch.isfinite(cleaned).all()
    assert rep.filled_missing > 0


def test_openpose_adapter_maps_body25():
    ad = openpose_adapter()
    body = torch.randn(3, 6, 25)
    res = ad(body)
    assert res.pose.shape == (3, 6, NUM_DEFAULT_JOINTS)
    assert torch.allclose(res.pose[:, :, 1], body[:, :, 1], atol=1e-6)  # neck 1:1


def test_mismatched_hand_shape_rejected():
    ad = mediapipe_holistic_adapter()
    body, _, _ = _mp_inputs(t=8)
    with pytest.raises(ValueError):
        ad(body, right_hand=torch.randn(3, 5, 21))    # wrong T


def test_bad_body_rank_rejected():
    ad = mediapipe_holistic_adapter()
    with pytest.raises(ValueError):
        ad(torch.randn(3, 33))
