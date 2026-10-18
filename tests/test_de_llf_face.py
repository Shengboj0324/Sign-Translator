"""Live Link Face adapter: exact clock, truncated tails, refusals, channel contracts."""
from fractions import Fraction

import numpy as np
import pytest

from signtranslator.data_engineering.llf_face import (
    ARKIT_BLENDSHAPES, LLF_COLUMNS, adapt_llf, gaze_vectors, parse_llf_csv,
)

HEADER = ",".join(LLF_COLUMNS)


def row(frame, sub="000", *, second=3, blend=None, rot=None, count="61"):
    blend = [0.25] * 52 if blend is None else blend
    rot = [0.] * 3 + [0.1, -0.2, 0.01, -0.1, 0.05, 0.] if rot is None else rot
    return ",".join([f"15:50:{second:02d}:{frame:02d}.{sub}", count,
                     *(f"{v:.10f}" for v in blend), *(f"{v:.10f}" for v in rot)])


def csv(rows, *, terminated=True):
    text = "\n".join([HEADER, *rows])
    return (text + "\n" if terminated else text).encode()


def adapt(payload):
    return adapt_llf(payload, sample_id="s", source_id="3dlex_llf", clock_id="llf:s",
                     inference_method="arkit-truedepth-solve")


def test_clock_is_exact_rational_timecode_at_the_declared_rate():
    rec = parse_llf_csv(csv([row(49, "599"), row(50, "602"), row(0, "617", second=4)]))
    expected = [Fraction(3600 * 15 + 50 * 60 + 3) + Fraction(49_599, 60_000),
                Fraction(3600 * 15 + 50 * 60 + 3) + Fraction(50_602, 60_000),
                Fraction(3600 * 15 + 50 * 60 + 4) + Fraction(617, 60_000)]
    np.testing.assert_array_equal(rec.times, np.array([float(t) for t in expected]))
    assert not rec.dropped_tail


def test_unterminated_last_line_is_always_discarded_even_if_it_parses():
    rows = [row(1), row(2), row(3)]
    rec = parse_llf_csv(csv(rows, terminated=False))
    assert rec.dropped_tail and len(rec.times) == 2
    truncated = rows[:2] + [row(3)[:40]]  # cut inside a number, as measured
    assert len(parse_llf_csv(csv(truncated, terminated=False)).times) == 2


@pytest.mark.parametrize("rows,message", [
    ([row(1), row(1)], "strictly increase"),
    ([row(2), row(1)], "strictly increase"),
    ([row(1), row(60)], "out of range"),
    ([row(1), row(2, count="6")], "blendshape count"),
    ([row(1), row(2)[:60]], "fields"),
    ([row(1, blend=[1.5] + [0.] * 51)], r"outside \[0, 1\]"),
    ([row(1, rot=[0.] * 3 + [2.] + [0.] * 5)], "quarter turn"),
    ([row(1).replace("0.2500000000", "nan", 1)], "non-numeric"),
])
def test_malformed_interior_rows_refuse_the_file(rows, message):
    with pytest.raises(ValueError, match=message):
        parse_llf_csv(csv(rows))


def test_corrupted_header_and_recorded_head_rotation_are_refused():
    state, report = adapt(b"FILE!oline_asl_four\n" + csv([row(1)]))
    assert state is None and report.status == "refused" and "header" in report.reason
    state, report = adapt(csv([row(1, rot=[0.01] + [0.] * 8)]))
    assert state is None and "head rotation" in report.reason


def test_converted_state_channels_and_contracts():
    blend = [i / 100 for i in range(52)]
    state, report = adapt(csv([row(1, blend=blend), row(2, blend=blend), row(4, blend=blend)]))
    assert report.status == "converted" and report.frames == 3 and report.dropped_frames == 1
    face = state.channels["face"]
    assert face.labels == ARKIT_BLENDSHAPES and face.values.shape == (3, 52, 1)
    np.testing.assert_array_equal(face.values[0, :, 0], blend)
    assert not face.observed.any() and face.inference_method == "arkit-truedepth-solve"
    blink = state.channels["blink"].values[0, :, 0]
    assert blink.tolist() == [blend[ARKIT_BLENDSHAPES.index("EyeBlinkLeft")],
                              blend[ARKIT_BLENDSHAPES.index("EyeBlinkRight")]]
    for name in ("head_rotation", "body", "left_hand", "right_hand", "root_translation", "contact"):
        assert not state.channels[name].valid.any(), name
    assert set(report.unavailable_channels) >= {"head_rotation", "body"}


def test_gaze_is_unit_norm_and_inverts_to_yaw_and_pitch():
    rng = np.random.default_rng(0)
    yaw, pitch = rng.uniform(-1.5, 1.5, 200), rng.uniform(-1.5, 1.5, 200)
    vectors = gaze_vectors(yaw, pitch)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-15)
    np.testing.assert_allclose(np.arcsin(vectors[:, 1]), pitch, atol=1e-12)
    np.testing.assert_allclose(np.arctan2(vectors[:, 0], vectors[:, 2]), yaw, atol=1e-12)
    np.testing.assert_allclose(gaze_vectors(np.zeros(1), np.zeros(1)), [[0., 0., 1.]])
