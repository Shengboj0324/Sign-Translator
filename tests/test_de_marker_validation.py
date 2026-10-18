"""Cross-modal elbow validation: exact geometry and recovered clocks."""
import numpy as np
import pytest

from signtranslator.data_engineering.marker_validation import (
    align_series, joint_angle_degrees, marker_elbow_flexion, parse_v3d_markers,
)


def test_joint_angle_is_exact_and_undefined_for_degenerate_segments():
    origin = np.zeros((3, 3))
    proximal = np.array([[1., 0, 0], [1., 0, 0], [0., 0, 0]])
    distal = np.array([[0., 1, 0], [-1., 0, 0], [1., 0, 0]])
    angles = joint_angle_degrees(proximal, origin, distal)
    assert angles[0] == pytest.approx(90.) and angles[1] == pytest.approx(180.)
    assert np.isnan(angles[2])


def test_alignment_recovers_a_hidden_rate_and_offset_exactly():
    rate, offset = 120., 37
    marker_time = np.arange(600) / rate
    markers = 60 + 40 * np.sin(2 * np.pi * .7 * marker_time) + 10 * np.sin(2 * np.pi * 1.9 * marker_time)
    skeleton_times = np.arange(48) / 24.
    reference = np.interp(skeleton_times * rate + offset, np.arange(600), markers) + 3.  # constant bias
    result = align_series(skeleton_times, reference, markers)
    assert result.rate_hz == rate and result.offset_frames == offset
    assert result.correlation == pytest.approx(1., abs=1e-12)
    assert result.centred_mae_degrees == pytest.approx(0., abs=1e-9)
    assert result.passes(min_correlation=.95, max_centred_mae_degrees=10.)


def test_alignment_never_bridges_missing_markers():
    markers = np.linspace(0, 90, 400)
    markers[100:300] = np.nan
    times = np.arange(30) / 24.
    reference = np.linspace(0, 90, 30)
    result = align_series(times, reference, markers, rates=(120.,))
    if result is not None:
        index = (times - times[0]) * 120. + result.offset_frames
        assert not np.any((index > 99) & (index < 300)) or result.samples < 30


def test_v3d_parser_discards_the_label_line_and_reads_missing_as_nan():
    text = ("Performer Name\nFrame,LFSH<T-X>,LFSH<T-Y>,LFSH<T-Z>,LBSH<T-X>,LBSH<T-Y>,LBSH<T-Z>,"
            "LELB<T-X>,LELB<T-Y>,LELB<T-Z>,LIEL<T-X>,LIEL<T-Y>,LIEL<T-Z>,"
            "LIWR<T-X>,LIWR<T-Y>,LIWR<T-Z>,LOWR<T-X>,LOWR<T-Y>,LOWR<T-Z>,\n"
            "1,0,0,10,0,0,10,0,0,0,0,0,0,10,0,0,10,0,0\n"
            "2,0,0,10,0,0,10,0,0,0,0,0,0,,0,0,10,0,0\n")
    names, data = parse_v3d_markers(text)
    assert names == ["LFSH", "LBSH", "LELB", "LIEL", "LIWR", "LOWR"]
    assert "Performer" not in repr(names)
    flexion = marker_elbow_flexion(names, data, "L")
    assert flexion[0] == pytest.approx(90.) and np.isnan(flexion[1])
    with pytest.raises(ValueError, match="lacks"):
        marker_elbow_flexion(names, data, "R")


def test_joint_alignment_uses_one_clock_and_judges_static_series_by_magnitude():
    from signtranslator.data_engineering.marker_validation import align_joint
    rate, offset = 120., 25
    t_marker = np.arange(500) / rate
    # Incommensurate components plus a chirp: no shift reproduces the signal,
    # so the true alignment is unique (a pure sinusoid repeats every period).
    moving = (70 + 30 * np.sin(2 * np.pi * .8 * t_marker) + 9 * np.sin(2 * np.pi * 1.93 * t_marker)
              + 6 * np.sin(2 * np.pi * .21 * t_marker ** 2))
    rng = np.random.default_rng(1)
    static = 160 + .05 * rng.standard_normal(500)  # a resting elbow: noise only
    times = np.arange(40) / 24.
    ref_moving = np.interp(times * rate + offset, np.arange(500), moving) + 4.
    ref_static = 160 + .05 * rng.standard_normal(40)
    result = align_joint(times, [ref_moving, ref_static], [moving, static])
    assert result.rate_hz == rate and result.offset_frames == offset
    assert result.correlations[0] == pytest.approx(1., abs=1e-9)
    assert result.reference_ranges_degrees[1] < 1.
    # Correlation of the static series is noise; magnitude agreement decides.
    passed = result.passes(min_correlation=.9, max_centred_mae_degrees=10., static_range_degrees=5.)
    assert passed == (True, True)
    strict = result.passes(min_correlation=.9, max_centred_mae_degrees=10., static_range_degrees=0.)
    assert strict[0] is True
