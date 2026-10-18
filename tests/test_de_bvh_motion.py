"""BVH adapter: exact listed-order Euler rotations, forward kinematics, strict refusals."""
from fractions import Fraction
import math

import numpy as np
import pytest
import torch

from signtranslator.data_engineering.bvh_motion import (
    BVHMapping, adapt_bvh, elementary_rotation, keyframe_seconds, parse_bvh, world_positions,
)
from signtranslator.pose.rotations import rotation_6d_to_matrix

HIERARCHY = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Head
    {
      OFFSET 0 20 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 5 0
      }
    }
  }
  JOINT LeftHand
  {
    OFFSET 5 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
  JOINT RightHand
  {
    OFFSET -5 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }
}
"""


def bvh(rows, frame_time="0.0083333", declared=None):
    body = "\n".join(" ".join(f"{v:g}" for v in row) for row in rows)
    return (HIERARCHY + f"MOTION\nFrames: {declared if declared is not None else len(rows)}\n"
            f"Frame Time: {frame_time}\n{body}\n").encode()


ROWS = [[1, 2, 3, 30, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 0, 0, 0, 0, 0, 0, 45, 30, 15, 0, 0, 0, 0, 0, 0]]
MAPPING = BVHMapping("fixture-v1", "Hips", ("Hips", "Spine"), ("LeftHand",), ("RightHand",), "Head", (), .01,
                     "fixture-y-up")


def axis(axis_name, degrees):
    """Independent Rodrigues construction."""
    k = np.eye(3)["XYZ".index(axis_name)]
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = math.radians(degrees)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def test_elementary_rotations_match_rodrigues():
    for a in "XYZ":
        for d in (-170., -30., 0., 12.5, 90., 179.):
            np.testing.assert_allclose(elementary_rotation(a, np.array([d]))[0], axis(a, d), atol=1e-14)


def test_listed_order_composition_and_forward_kinematics_are_exact():
    doc = parse_bvh(bvh(ROWS))
    p = world_positions(doc)
    np.testing.assert_allclose(p["Hips"][0], [1, 2, 3])
    # Frame 0: hips Zrot 30 rotates the spine offset (0,10,0); spine Xrot 90 rotates the head offset.
    hips = axis("Z", 30)
    np.testing.assert_allclose(p["Spine"][0], np.array([1, 2, 3]) + hips @ [0, 10, 0], atol=1e-12)
    np.testing.assert_allclose(p["Head"][0], p["Spine"][0] + hips @ axis("X", 90) @ [0, 20, 0], atol=1e-12)
    # Frame 1: head rotation Z45 X30 Y15 composes in listed order R = Rz Rx Ry.
    state, report = adapt_bvh(bvh(ROWS), mapping=MAPPING, sample_id="s", source_id="src", clock_id="c",
                              inference_method="mocap solve")
    assert report.status == "converted"
    head = rotation_6d_to_matrix(torch.from_numpy(state.channels["head_rotation"].values[1, 0])).numpy()
    np.testing.assert_allclose(head, axis("Z", 45) @ axis("X", 30) @ axis("Y", 15), atol=1e-12)
    np.testing.assert_allclose(state.channels["root_translation"].values[0, 0], [.01, .02, .03], atol=1e-15)
    assert not state.channels["face"].valid.any() and not state.channels["body"].observed.any()


def test_clock_is_exact_in_the_declared_frame_time():
    doc = parse_bvh(bvh(ROWS * 3, frame_time="0.0083333"))
    assert doc.frame_time == Fraction(83333, 10_000_000)
    np.testing.assert_array_equal(doc.times, [float(i * Fraction(83333, 10_000_000)) for i in range(6)])
    assert keyframe_seconds(120, Fraction(1, 120)) == 1.0
    with pytest.raises(ValueError):
        keyframe_seconds(-1, Fraction(1, 120))


@pytest.mark.parametrize("payload,message", [
    (bvh(ROWS, declared=3), "declares 3 rows"),
    (bvh([ROWS[0][:-1]]), "expected 18 values"),
    (bvh(ROWS).replace(b"Zrotation Xrotation Yrotation\n    JOINT Head", b"Zrotation Zrotation Yrotation\n    JOINT Head"), "repeated"),
    (bvh(ROWS).replace(b"JOINT RightHand", b"JOINT LeftHand"), "duplicate"),
    (bvh(ROWS).replace(b"  }\n}\nMOTION", b"  }\nMOTION"), "unbalanced"),
    (bvh(ROWS).replace(b"Frame Time: 0.0083333", b"Frame Time: 0"), "positive"),
    (bvh([[float("nan")] + ROWS[0][1:]]), "non-finite"),
])
def test_malformed_files_are_refused(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_bvh(payload)


def test_mapping_must_cover_every_joint_exactly():
    incomplete = BVHMapping("m", "Hips", ("Hips",), ("LeftHand",), ("RightHand",), "Head", (), .01, "f")
    state, report = adapt_bvh(bvh(ROWS), mapping=incomplete, sample_id="s", source_id="src", clock_id="c",
                              inference_method="solve")
    assert state is None and "undeclared ['Spine']" in report.reason
    wrong_root = BVHMapping("m", "Spine", ("Hips", "Spine"), ("LeftHand",), ("RightHand",), "Head", (), .01, "f")
    state, report = adapt_bvh(bvh(ROWS), mapping=wrong_root, sample_id="s", source_id="src", clock_id="c",
                              inference_method="solve")
    assert state is None and "root" in report.reason
