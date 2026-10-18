"""Masked-reconstruction protocol: exact geodesic baselines and common support."""
import math

import numpy as np
import pytest
import torch

from signtranslator.generation.representation_comparison import (
    compare_reconstructions, hidden_spans, hold_out_rest_baseline, interpolation_baseline,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

# A deliberately nonuniform clock: a frame-index surrogate is wrong on it.
CLOCK = np.array([0., .02, .05, .09, .14, .2, .27, .35, .44, .54, .65, .77], dtype=np.float64)
AXIS = np.array([1., 2., -2.]) / 3.
OMEGA = 1.7  # radians per second


def state(*, body=None, gaze=None, translation=None, observed=None):
    frames = len(CLOCK)
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        values = np.zeros((frames, 1, width), dtype=np.float64)
        if kind == "rotation_6d_columns":
            values[:] = [1, 0, 0, 0, 1, 0]
        elif kind == "unit_vector":
            values[:] = [1, 0, 0]
        if name == "body" and body is not None:
            values = body
        if name == "left_gaze" and gaze is not None:
            values = gaze
        if name == "root_translation" and translation is not None:
            values = translation
        seen = np.ones((frames, 1), dtype=bool) if observed is None or name != "body" else observed
        channels[name] = MotionChannel(
            values=values, valid=np.ones((frames, 1), dtype=bool), observed=seen,
            confidence=np.ones((frames, 1), dtype=np.float64), labels=(name,),
            coordinate_frame="fixture", units=units, source_id="fixture", source_sha256="c" * 64,
            convention="fixture-v1", inference_method=None if seen.all() else "fixture-inference")
    return MultichannelMotion(CLOCK.copy(), "fixture-clock", "fixture", channels)


def constant_rotation():
    angles = torch.from_numpy(CLOCK * OMEGA)
    matrices = axis_angle_to_matrix(torch.from_numpy(AXIS)[None] * angles[:, None])
    return matrix_to_rotation_6d(matrices).numpy().reshape(len(CLOCK), 1, 6)


def great_circle():
    start, end = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    angles = CLOCK * 1.1
    return (np.cos(angles)[:, None] * start + np.sin(angles)[:, None] * end).reshape(-1, 1, 3)


def interior_mask(name, frames=(3, 4, 5, 8)):
    mask = np.zeros((len(CLOCK), 1), dtype=bool)
    mask[list(frames)] = True
    return {name: mask}


def test_geodesic_interpolation_is_exact_for_constant_angular_velocity_on_a_nonuniform_clock():
    s = state(body=constant_rotation())
    hidden = interior_mask("body")
    report = compare_reconstructions(s, hidden, {
        "interpolation": interpolation_baseline, "rest": hold_out_rest_baseline}).to_dict()
    record = report["channels"]["body"]
    assert record["methods"]["interpolation"]["max_error"] < 1e-9
    assert record["methods"]["rest"]["mean_error"] > 1.
    assert record["methods"]["interpolation"]["coverage"] == 1.


def test_frame_index_interpolation_would_be_wrong_on_the_same_clock():
    """Pins that the baseline uses seconds: a frame-index tau gives a real error."""
    rotation = constant_rotation()
    a, frame, b = 2, 4, 6
    tau_frames = (frame - a) / (b - a)
    tau_seconds = (CLOCK[frame] - CLOCK[a]) / (CLOCK[b] - CLOCK[a])
    assert abs(tau_frames - tau_seconds) > .05
    reconstructed = interpolation_baseline(state(body=rotation), interior_mask("body", (4, 5)))
    values, covered = reconstructed["body"]
    assert covered[frame, 0]
    np.testing.assert_allclose(values[frame, 0], rotation[frame, 0], atol=1e-10)


def test_great_circle_slerp_is_exact_and_magnitude_free():
    s = state(gaze=great_circle())
    report = compare_reconstructions(s, interior_mask("left_gaze"), {
        "interpolation": interpolation_baseline, "rest": hold_out_rest_baseline}).to_dict()
    record = report["channels"]["left_gaze"]
    # The rest baseline declines gaze, so the common support is empty and no
    # method is scored there; coverage still reports what each method did.
    assert record["common_support_cells"] == 0
    assert record["methods"]["rest"]["coverage"] == 0.
    assert record["methods"]["interpolation"]["coverage"] == 1.
    alone = interpolation_baseline(s, interior_mask("left_gaze"))["left_gaze"][0]
    np.testing.assert_allclose(alone, great_circle(), atol=1e-12)


def test_linear_translation_is_reconstructed_exactly():
    path = (CLOCK[:, None] * np.array([.3, -.1, .05])).reshape(-1, 1, 3)
    s = state(translation=path)
    report = compare_reconstructions(s, interior_mask("root_translation"), {
        "interpolation": interpolation_baseline, "rest": hold_out_rest_baseline}).to_dict()
    record = report["channels"]["root_translation"]
    assert record["unit"] == "metres"
    assert record["methods"]["interpolation"]["max_error"] < 1e-12


def test_common_support_prevents_winning_by_declining_hard_cells():
    s = state(body=constant_rotation())
    hidden = interior_mask("body")

    def selective(state_, hidden_):
        values, covered = hold_out_rest_baseline(state_, hidden_)["body"]
        covered = covered.copy()
        covered[3, 0] = False  # decline one cell
        return {"body": (state_.channels["body"].values.copy(), covered)}

    report = compare_reconstructions(s, hidden, {
        "interpolation": interpolation_baseline, "selective": selective}).to_dict()
    record = report["channels"]["body"]
    assert record["eligible_cells"] == 4 and record["common_support_cells"] == 3
    assert record["methods"]["selective"]["coverage"] == pytest.approx(.75)
    assert record["methods"]["interpolation"]["coverage"] == 1.


def test_unobserved_hidden_cells_are_never_scored():
    observed = np.ones((len(CLOCK), 1), dtype=bool)
    observed[4] = False
    s = state(body=constant_rotation(), observed=observed)
    report = compare_reconstructions(s, interior_mask("body"), {
        "interpolation": interpolation_baseline, "rest": hold_out_rest_baseline}).to_dict()
    assert report["channels"]["body"]["eligible_cells"] == 3


def test_half_turn_ambiguity_is_uncovered_rather_than_guessed():
    rotation = np.tile(np.array([1., 0, 0, 0, 1, 0]), (len(CLOCK), 1, 1))
    half_turn = matrix_to_rotation_6d(axis_angle_to_matrix(
        torch.tensor([0., 0., math.pi], dtype=torch.float64))).numpy()
    rotation[6:] = half_turn
    values, covered = interpolation_baseline(state(body=rotation), interior_mask("body", (5,)))["body"]
    assert not covered[5, 0]
    antipodal = np.tile(np.array([1., 0., 0.]), (len(CLOCK), 1, 1))
    antipodal[6:] = [-1., 0., 0.]
    _, covered = interpolation_baseline(state(gaze=antipodal), interior_mask("left_gaze", (5,)))["left_gaze"]
    assert not covered[5, 0]


def test_spans_stay_interior_disjoint_and_seeded():
    s = state()
    first = hidden_spans(s, channels=("body", "face"), span_seconds=.1, spans=2, seed=3)
    second = hidden_spans(s, channels=("body", "face"), span_seconds=.1, spans=2, seed=3)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    frames = first["body"][:, 0]
    assert not frames[0] and not frames[-1] and frames.any()
    with pytest.raises(ValueError, match="unknown or empty"):
        hidden_spans(s, channels=("wings",), span_seconds=.1, spans=1, seed=0)


def test_methods_cannot_claim_visible_cells_or_omit_channels():
    s = state(body=constant_rotation())
    hidden = interior_mask("body")

    def cheating(state_, hidden_):
        values = state_.channels["body"].values.copy()
        return {"body": (values, np.ones_like(hidden_["body"]))}

    with pytest.raises(ValueError, match="visible cell"):
        compare_reconstructions(s, hidden, {"interpolation": interpolation_baseline,
                                            "cheating": cheating})
    with pytest.raises(ValueError, match="exactly the hidden channels"):
        compare_reconstructions(s, hidden, {"interpolation": interpolation_baseline,
                                            "empty": lambda st, h: {}})
    with pytest.raises(ValueError, match="at least two"):
        compare_reconstructions(s, hidden, {"interpolation": interpolation_baseline})


def test_inferred_reference_requires_an_explicit_recorded_opt_in():
    observed = np.zeros((len(CLOCK), 1), dtype=bool)  # a solved, never-observed track
    s = state(body=constant_rotation(), observed=observed)
    hidden = interior_mask("body")
    methods = {"interpolation": lambda st, h: interpolation_baseline(st, h, allow_inferred=True),
               "rest": hold_out_rest_baseline}
    refused = compare_reconstructions(s, hidden, methods).to_dict()["channels"]["body"]
    assert refused["eligible_cells"] == 0 and refused["reference"] == "observed_only"
    scored = compare_reconstructions(s, hidden, methods, allow_inferred=True).to_dict()["channels"]["body"]
    assert scored["reference"] == "valid_including_inferred"
    assert scored["eligible_cells"] == 4
    assert scored["methods"]["interpolation"]["max_error"] < 1e-9
    with pytest.raises(ValueError, match="explicit boolean"):
        compare_reconstructions(s, hidden, methods, allow_inferred=1)


def test_zero_order_hold_repeats_the_last_reference_value():
    from signtranslator.generation.representation_comparison import zero_order_hold_baseline
    rotation = constant_rotation()
    values, covered = zero_order_hold_baseline(state(body=rotation), interior_mask("body", (4, 5)))["body"]
    assert covered[4, 0] and covered[5, 0]
    np.testing.assert_array_equal(values[4, 0], rotation[3, 0])
    np.testing.assert_array_equal(values[5, 0], rotation[3, 0])
