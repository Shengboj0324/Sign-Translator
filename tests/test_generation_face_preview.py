"""Schematic face panel: brows, eyelids, mouth and head turn are drawn from the state's coefficients."""
from types import SimpleNamespace

import numpy as np
import pytest

from signtranslator.generation.face_preview import render_face_frames, with_face_panel

NAMES = ("browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "jawOpen",
         "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight")


def state(frames=3, coefficients=None, yaw=0., face_valid=True):
    values = np.zeros((frames, len(NAMES), 1))
    for name, v in (coefficients or {}).items():
        values[:, NAMES.index(name), 0] = v
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    six = np.tile(np.concatenate([r[:, 0], r[:, 1]])[None, None], (frames, 1, 1))   # 6D: first two columns
    return SimpleNamespace(timestamps=np.arange(frames) / 24., channels=dict(
        face=SimpleNamespace(values=values, valid=np.full((frames, 1), face_valid), labels=NAMES),
        head_rotation=SimpleNamespace(values=six, valid=np.ones((frames, 1), bool), labels=("Head",))))


def ink(frame):
    return (frame != 255).any(axis=-1)


def test_neutral_face_draws_features_and_is_deterministic():
    a = render_face_frames(state(), width=160, height=200)
    assert len(a) == 3 and a[0].shape == (200, 160, 3)
    assert ink(a[0]).sum() > 300
    assert np.array_equal(a[0], render_face_frames(state(), width=160, height=200)[0])


@pytest.mark.parametrize("name", ["browInnerUp", "browDownLeft", "jawOpen", "eyeBlinkLeft", "mouthSmileLeft"])
def test_each_grammatical_coefficient_changes_the_drawing(name):
    neutral = render_face_frames(state(), width=160, height=200)[0]
    moved = render_face_frames(state(coefficients={name: 1.}), width=160, height=200)[0]
    assert (ink(neutral) != ink(moved)).sum() > 10


def test_raised_brows_sit_higher_than_furrowed_brows():
    brown = np.array([90, 55, 30])

    def brow_rows(frame):
        return np.nonzero((frame == brown).all(-1))[0].mean()
    raised = render_face_frames(state(coefficients={"browInnerUp": 1., "browOuterUpLeft": 1., "browOuterUpRight": 1.}),
                                width=160, height=200)[0]
    furrowed = render_face_frames(state(coefficients={"browDownLeft": 1., "browDownRight": 1.}), width=160, height=200)[0]
    assert brow_rows(raised) < brow_rows(furrowed)                  # image rows grow downwards


def test_head_turn_moves_features_and_invalid_face_draws_outline_only():
    straight = render_face_frames(state(), width=160, height=200)[0]
    turned = render_face_frames(state(yaw=.5), width=160, height=200)[0]
    assert (ink(straight) != ink(turned)).sum() > 50
    blank = render_face_frames(state(face_valid=False), width=160, height=200)[0]
    drawn = ink(blank)
    drawn[:, 0] = False                                             # the panel's separator line
    assert 0 < drawn.sum() < ink(straight).sum()
    assert (blank[drawn] == 185).all()                              # outline only, in grey


def test_panel_is_appended_beside_the_skeleton_and_sizes_must_be_even():
    skeleton = [np.full((240, 240, 3), 255, np.uint8)] * 3
    joined = with_face_panel(skeleton, state())
    assert joined[0].shape == (240, 360, 3)
    with pytest.raises(ValueError):
        render_face_frames(state(), width=161, height=200)
