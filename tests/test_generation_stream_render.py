"""Timestamp-exact encoding and bundle provenance; no Blender required."""
from fractions import Fraction

import numpy as np
import pytest

from signtranslator.generation.stream_render import (
    DEFAULT_TIME_BASE, encode_timestamped_video, presentation_timestamps,
    timestamp_error_bound, verify_timestamped_video, write_frame_bundle,
)
from test_generation_state_rig import binding_for, rig, state_for  # noqa: F401

NONUNIFORM = [10.0, 10.0333, 10.07, 10.1, 10.19, 10.2, 10.35, 10.3501]


def frames(count, size=(16, 16)):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (*size, 3), dtype=np.uint8) for _ in range(count)]


def test_ticks_are_exact_rounded_and_within_half_a_tick():
    ticks = presentation_timestamps(NONUNIFORM)
    assert ticks[0] == 0
    for tick, value in zip(ticks, NONUNIFORM):
        exact = (Fraction(value) - Fraction(NONUNIFORM[0])) / DEFAULT_TIME_BASE
        assert tick == round(exact)
    assert timestamp_error_bound(NONUNIFORM, ticks) <= DEFAULT_TIME_BASE / 2


def test_frames_closer_than_one_tick_are_refused_not_merged():
    with pytest.raises(ValueError, match="same or earlier tick"):
        presentation_timestamps([0., 1e-7, 1.])
    # A finer declared time base represents them.
    assert presentation_timestamps([0., 1e-7, 1.], time_base=Fraction(1, 10**9))[1] == 100


def test_encoded_video_reproduces_a_nonuniform_clock(tmp_path):
    path = tmp_path / "stream.mp4"
    record = encode_timestamped_video(frames(len(NONUNIFORM)), NONUNIFORM, path)
    assert record["frames"] == len(NONUNIFORM)
    assert record["clock_origin_seconds"] == NONUNIFORM[0]
    verified = verify_timestamped_video(path, NONUNIFORM)
    assert verified["frames"] == len(NONUNIFORM)
    assert verified["max_clock_error_seconds"] <= verified["half_tick_bound_seconds"]
    assert verified["max_pts_drift_seconds"] == 0.
    # The stream is genuinely variable rate: consecutive gaps differ.
    gaps = np.diff(record["ticks"])
    assert len(set(gaps.tolist())) > 1


def test_encoder_rejects_inconsistent_inputs(tmp_path):
    with pytest.raises(ValueError, match="frame count"):
        encode_timestamped_video(frames(3), NONUNIFORM, tmp_path / "a.mp4")
    with pytest.raises(ValueError, match="even frame dimensions"):
        encode_timestamped_video(frames(len(NONUNIFORM), (15, 16)), NONUNIFORM, tmp_path / "b.mp4")


def test_verifier_detects_a_wrong_clock(tmp_path):
    path = tmp_path / "stream.mp4"
    encode_timestamped_video(frames(len(NONUNIFORM)), NONUNIFORM, path)
    shifted = list(NONUNIFORM)
    shifted[3] += .01
    with pytest.raises(ValueError, match="half-tick"):
        verify_timestamped_video(path, shifted)
    with pytest.raises(ValueError, match="decoded"):
        verify_timestamped_video(path, NONUNIFORM[:-1])


def test_frame_bundle_binds_provenance_and_refuses_overwrite(tmp_path, rig):  # noqa: F811
    state = state_for(rig, observed=False)
    path = tmp_path / "bundle.npz"
    bundle = write_frame_bundle(rig, state, binding_for(rig), path, allow_inferred=True)
    assert bundle.frames == len(state.timestamps)
    assert bundle.provenance["posed_inferred_values"] is True
    assert bundle.provenance["accepted_provenance"] is False
    with np.load(path, allow_pickle=False) as archive:
        assert archive["vertices"].shape == (bundle.frames, rig.rest_vertices.shape[0], 3)
        np.testing.assert_array_equal(archive["clock"], state.timestamps)
        assert archive["renderable"].any() and not archive["renderable"].all()
    with pytest.raises(FileExistsError):
        write_frame_bundle(rig, state, binding_for(rig), path, allow_inferred=True)
    with pytest.raises(ValueError, match="explicit opt-in"):
        write_frame_bundle(rig, state, binding_for(rig), tmp_path / "other.npz",
                           allow_inferred=False)


def test_review_quality_is_playable_420_with_the_same_timestamps(tmp_path):
    path = tmp_path / "review.mp4"
    record = encode_timestamped_video(frames(len(NONUNIFORM)), NONUNIFORM, path, quality="review")
    assert record["pixel_format"] == "yuv420p"
    assert verify_timestamped_video(path, NONUNIFORM)["max_pts_drift_seconds"] == 0.
    with pytest.raises(ValueError, match="quality"):
        encode_timestamped_video(frames(len(NONUNIFORM)), NONUNIFORM, tmp_path / "x.mp4", quality="draft")
