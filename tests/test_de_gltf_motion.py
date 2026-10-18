"""glTF adapter: strict container parsing, exact kinematics and refusal rules."""
import json
import math
import struct

import numpy as np
import pytest
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.data_engineering.gltf_motion import (
    THREE_D_LEX_MIXAMO68, adapt_glb, parse_glb,
)
from signtranslator.pose.rotations import (
    axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix,
)

INFERENCE = "fixture-solve"


def quaternion_xyzw(axis_angle):
    """Independent unit quaternion of an axis-angle vector, glTF (x, y, z, w) order."""
    angle = float(np.linalg.norm(axis_angle))
    if angle == 0:
        return [0., 0., 0., 1.]
    axis = np.asarray(axis_angle, dtype=np.float64) / angle
    return [*(axis * math.sin(angle / 2)).tolist(), math.cos(angle / 2)]


def build_glb(*, frames=6, fps=24., root_path=None, rotations=None, extra_joint=None,
              drop_joint=None, second_clock=False, interpolation="LINEAR", animation=True,
              scale_track=None, duplicate_key=False):
    joints = [*THREE_D_LEX_MIXAMO68.joints(), *THREE_D_LEX_MIXAMO68.excluded]
    joints = [j for j in dict.fromkeys(joints) if j != "Armature" and j != drop_joint]
    if extra_joint:
        joints.append(extra_joint)
    nodes = [dict(name="RootNode", children=[1]),
             dict(name="actorBP", translation=[.2, .3, .1], rotation=[1., 0., 0., 0.], children=[2]),
             dict(name="Armature", rotation=[math.sqrt(.5), 0., 0., math.sqrt(.5)], children=[3])]
    hips = joints.index("Hips")
    order = [hips] + [i for i in range(len(joints)) if i != hips]
    names = [joints[i] for i in order]
    for position, name in enumerate(names):
        node = dict(name=name)
        if position == 0:
            node["children"] = list(range(4, 3 + len(names)))
        nodes.append(node)
    index = {name: i for i, name in enumerate(n["name"] for n in nodes)}
    times = (np.arange(frames, dtype=np.float32) / np.float32(fps))
    blobs, accessors, views = [], [], []

    def add(array, kind):
        array = np.ascontiguousarray(array, dtype="<f4")
        offset = sum(len(b) for b in blobs)
        blobs.append(array.tobytes())
        views.append(dict(buffer=0, byteOffset=offset, byteLength=array.nbytes))
        accessors.append(dict(bufferView=len(views) - 1, componentType=5126, count=len(array),
                              type=kind))
        return len(accessors) - 1

    clock = add(times, "SCALAR")
    other_clock = add(times + np.float32(1e-3), "SCALAR") if second_clock else clock
    samplers, channels = [], []
    path = root_path if root_path is not None else np.tile([0., .1, .95], (frames, 1))
    samplers.append(dict(input=clock, output=add(path, "VEC3"), interpolation=interpolation))
    channels.append(dict(sampler=0, target=dict(node=index["Hips"], path="translation")))
    for joint, values in (rotations or {}).items():
        samplers.append(dict(input=other_clock, output=add(values, "VEC4"), interpolation=interpolation))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=index[joint], path="rotation")))
    if scale_track is not None:
        samplers.append(dict(input=clock, output=add(scale_track, "VEC3")))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=index["Hips"], path="scale")))
    document = dict(asset=dict(version="2.0", generator="test"), scene=0, scenes=[dict(nodes=[0])],
                    nodes=nodes, skins=[dict(joints=[index[n] for n in names])],
                    accessors=accessors, bufferViews=views,
                    buffers=[dict(byteLength=sum(len(b) for b in blobs))])
    if animation:
        document["animations"] = [dict(samplers=samplers, channels=channels)]
    payload = json.dumps(document).encode()
    if duplicate_key:
        payload = payload.replace(b'"scene": 0', b'"scene": 0, "scene": 0')
    payload += b" " * (-len(payload) % 4)
    binary = b"".join(blobs)
    binary += b"\0" * (-len(binary) % 4)
    body = (struct.pack("<II", len(payload), 0x4E4F534A) + payload
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def adapt(data, **kwargs):
    return adapt_glb(data, sample_id="fixture", source_id="fixture-source", clock_id="fixture-clock",
                     inference_method=INFERENCE, **kwargs)


def test_quaternion_order_matches_an_independent_axis_angle_construction():
    rotation = np.array([.3, -.4, .5])
    track = np.tile(quaternion_xyzw(rotation), (6, 1))
    state, report = adapt(build_glb(rotations={"LeftForeArm": track}))
    assert report.status == "converted", report.reason
    element = state.channels["body"].labels.index("LeftForeArm")
    expected = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(rotation))).numpy()
    np.testing.assert_allclose(state.channels["body"].values[0, element], expected, atol=1e-6)
    # Unanimated joints keep their static node rotation, here the identity.
    neutral = state.channels["body"].labels.index("Spine")
    np.testing.assert_allclose(rotation_6d_to_matrix(torch.from_numpy(
        state.channels["body"].values[0, neutral])).numpy(), np.eye(3), atol=1e-12)


def test_root_position_composes_the_static_parent_chain_exactly():
    local = np.tile([.01, .1, .95], (6, 1))
    state, _ = adapt(build_glb(root_path=local))
    flip = np.diag([1., -1., -1.])                      # actorBP: 180 degrees about x
    up = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])  # Armature: +90 degrees about x
    expected = flip @ (up @ local[0]) + np.array([.2, .3, .1])
    np.testing.assert_allclose(state.channels["root_translation"].values[0, 0], expected, atol=1e-6)
    assert state.channels["root_translation"].units == "m"


def test_trailing_reset_key_is_invalidated_and_reported_never_kept():
    path = np.tile([0., .1, .95], (40, 1))
    path[-1] = [0., 1.3, .01]  # the 3D-LEX reset pattern: about 33 m/s at 24 fps
    state, report = adapt(build_glb(frames=40, root_path=path))
    assert report.status == "converted"
    assert [f["frame"] for f in report.invalid_frames] == [39]
    for name in ("body", "left_hand", "root_translation"):
        assert not state.channels[name].valid[39].any()
        assert not state.channels[name].values[39].any()
    assert state.channels["body"].valid[:39].all()


def test_isolated_interior_spike_is_invalidated_but_a_discontinuity_is_refused():
    spike = np.tile([0., .1, .95], (40, 1))
    spike[20] = [0., 1.5, .95]
    _, report = adapt(build_glb(frames=40, root_path=spike))
    assert report.status == "converted" and [f["frame"] for f in report.invalid_frames] == [20]
    step = np.tile([0., .1, .95], (6, 1))
    step[3:] = [0., 1.5, .95]
    state, report = adapt(build_glb(root_path=step))
    assert state is None and "ambiguous frame set" in report.reason


def test_generated_values_are_solved_not_observed_and_unavailable_channels_explicit():
    state, report = adapt(build_glb())
    for name in ("body", "left_hand", "right_hand", "head_rotation", "root_translation"):
        channel = state.channels[name]
        assert channel.valid.all() and not channel.observed.any()
        assert channel.inference_method == INFERENCE
        assert not channel.supervision_weights().any()
        assert channel.supervision_weights(allow_inferred=True).all()
    for name in ("face", "left_gaze", "right_gaze", "blink", "contact", "head_translation"):
        assert not state.channels[name].valid.any()
    assert set(report.unavailable_channels) == {"face", "left_gaze", "right_gaze", "blink",
                                                "contact", "head_translation"}
    assert len(state.channels["left_hand"].labels) == 15


def test_states_from_one_mapping_collate_without_an_adapter():
    first, _ = adapt(build_glb(frames=6))
    second, _ = adapt(build_glb(frames=9))
    batch = collate_multichannel([first, second])
    assert batch.lengths.tolist() == [6, 9]


def test_unrepresented_scale_is_reported_with_its_range():
    scale = np.tile([1.04, 1.04, 1.04], (6, 1))
    scale[3] = 1.0411
    _, report = adapt(build_glb(scale_track=scale))
    assert report.unrepresented["Hips.scale"]["max"][0] == pytest.approx(1.0411, abs=1e-6)


@pytest.mark.parametrize("kwargs,reason", [
    (dict(animation=False), "no animation"),
    (dict(second_clock=True, rotations={"Neck": np.tile([0., 0., 0., 1.], (6, 1))}), "share one input clock"),
    (dict(interpolation="CUBICSPLINE"), "not supported"),
    (dict(drop_joint="LeftHandIndex2"), "missing"),
    (dict(extra_joint="Tail"), "undeclared joints"),
    (dict(rotations={"Neck": np.tile([0., 0., 0., 1.01], (6, 1))}), "norm error"),
])
def test_refusals_are_explicit(kwargs, reason):
    state, report = adapt(build_glb(**kwargs))
    assert state is None and report.status == "refused" and reason in report.reason


@pytest.mark.parametrize("mutate,message", [
    (lambda b: b"gltf" + b[4:], "glTF 2.0"),
    (lambda b: b[:4] + struct.pack("<I", 1) + b[8:], "glTF 2.0"),
    (lambda b: b[:-4], "glTF 2.0"),
])
def test_container_header_is_strict(mutate, message):
    with pytest.raises(ValueError, match=message):
        parse_glb(mutate(build_glb()))


def test_duplicate_json_keys_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        parse_glb(build_glb(duplicate_key=True))


def test_accessor_reads_are_bounds_checked():
    glb = parse_glb(build_glb())
    glb.document["accessors"][0]["count"] = 10 ** 6
    with pytest.raises(ValueError, match="beyond"):
        glb.accessor(0)
    glb.document["accessors"][0]["count"] = 6
    glb.document["accessors"][0]["componentType"] = 5123
    with pytest.raises(ValueError, match="float32"):
        glb.accessor(0)


def test_animated_axis_correction_above_the_root_is_refused_with_its_angle():
    data = build_glb()
    glb = parse_glb(data)
    document = glb.document
    armature = next(i for i, n in enumerate(document["nodes"]) if n["name"] == "Armature")
    frames = 6
    angles = np.linspace(0., math.radians(80.), frames)
    track = np.array([quaternion_xyzw([a, 0., 0.]) for a in angles], dtype="<f4")
    binary = glb.binary + track.tobytes()
    document["bufferViews"].append(dict(buffer=0, byteOffset=len(glb.binary), byteLength=track.nbytes))
    document["accessors"].append(dict(bufferView=len(document["bufferViews"]) - 1,
                                      componentType=5126, count=frames, type="VEC4"))
    animation = document["animations"][0]
    animation["samplers"].append(dict(input=animation["samplers"][0]["input"],
                                      output=len(document["accessors"]) - 1))
    animation["channels"].append(dict(sampler=len(animation["samplers"]) - 1,
                                      target=dict(node=armature, path="rotation")))
    document["buffers"][0]["byteLength"] = len(binary)
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = (struct.pack("<II", len(payload), 0x4E4F534A) + payload
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    state, report = adapt(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    assert state is None
    assert "re-orients the whole skeleton" in report.reason and "80.0 degrees" in report.reason


def test_two_frame_trailing_reset_is_excluded_by_the_longest_plausible_path():
    path = np.tile([0., .1, .95], (40, 1))
    path[-2:] = [0., 1.3, .01]  # the 176-file pattern: a two-key reset block
    state, report = adapt(build_glb(frames=40, root_path=path))
    assert report.status == "converted", report.reason
    assert [f["frame"] for f in report.invalid_frames] == [38, 39]
    assert state.channels["body"].valid[:38].all() and not state.channels["body"].valid[38:].any()


def test_longest_plausible_subsequence_is_exact_and_bounded():
    from signtranslator.data_engineering.gltf_motion import longest_plausible_subsequence
    times = np.arange(10) / 24.
    smooth = np.cumsum(np.full((10, 3), .001), axis=0)
    assert longest_plausible_subsequence(times, smooth, 5.) == list(range(10))
    spiky = smooth.copy()
    spiky[4] += 2.
    assert longest_plausible_subsequence(times, spiky, 5.) == [0, 1, 2, 3, 5, 6, 7, 8, 9]
    # Exhaustive check against brute force on a small case.
    import itertools
    rng = np.random.default_rng(0)
    points = rng.normal(scale=.3, size=(7, 3))
    t = np.arange(7) / 24.
    chain = longest_plausible_subsequence(t, points, 5.)
    def ok(seq):
        return all(np.linalg.norm(points[b] - points[a]) <= 5. * (t[b] - t[a]) for a, b in zip(seq, seq[1:]))
    best = max(len(c) for r in range(1, 8) for c in itertools.combinations(range(7), r) if ok(c))
    assert len(chain) == best and ok(chain)


def test_majority_ambiguity_is_refused_rather_than_guessed():
    path = np.tile([0., .1, .95], (8, 1))
    path[4:] = [0., 1.5, .95]  # half the frames are a teleported block
    state, report = adapt(build_glb(frames=8, root_path=path))
    assert state is None and "ambiguous frame set" in report.reason


def keyed_axis_glb(frames=40):
    data = build_glb(frames=frames)
    glb = parse_glb(data)
    document = glb.document
    armature = next(i for i, n in enumerate(document["nodes"]) if n["name"] == "Armature")
    angles = np.linspace(0., math.radians(85.), frames)
    track = np.array([quaternion_xyzw([a, 0., 0.]) for a in angles], dtype="<f4")
    binary = glb.binary + track.tobytes()
    document["bufferViews"].append(dict(buffer=0, byteOffset=len(glb.binary), byteLength=track.nbytes))
    document["accessors"].append(dict(bufferView=len(document["bufferViews"]) - 1,
                                      componentType=5126, count=frames, type="VEC4"))
    animation = document["animations"][0]
    animation["samplers"].append(dict(input=animation["samplers"][0]["input"],
                                      output=len(document["accessors"]) - 1))
    animation["channels"].append(dict(sampler=len(animation["samplers"]) - 1,
                                      target=dict(node=armature, path="rotation")))
    document["buffers"][0]["byteLength"] = len(binary)
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = (struct.pack("<II", len(payload), 0x4E4F534A) + payload
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_salvage_keeps_parent_relative_joints_and_masks_the_global_orientation():
    rotation = np.tile(quaternion_xyzw([.2, .1, 0.]), (40, 1))
    data = keyed_axis_glb()
    refused, report = adapt(data)
    assert refused is None and "re-orients" in report.reason
    state, report = adapt(data, salvage_reoriented_root=True)
    assert report.status == "converted_partial"
    body = state.channels["body"]
    hips = body.labels.index("Hips")
    assert not body.valid[:, hips].any() and not body.values[:, hips].any()
    assert body.valid[:, body.labels.index("Spine")].all()
    assert not state.channels["root_translation"].valid.any()
    assert state.channels["left_hand"].valid.all()
    assert "salvaged_global_orientation" in report.unrepresented
    del rotation
    with pytest.raises(ValueError, match="explicit boolean"):
        adapt(data, salvage_reoriented_root=1)


def test_skeleton_elbow_flexion_is_invariant_to_the_keyed_axis_correction():
    from signtranslator.data_engineering.gltf_motion import skeleton_elbow_flexion
    clean = skeleton_elbow_flexion(build_glb(frames=40), "Left")[1]
    keyed = skeleton_elbow_flexion(keyed_axis_glb(), "Left")[1]
    np.testing.assert_allclose(clean, keyed, atol=1e-9)


def _posed_tracks(frames, *, rest_frames=()):
    """Several animated joints well away from rest, snapped to rest on given frames."""
    tracks = {}
    for i, joint in enumerate(("LeftForeArm", "RightForeArm", "LeftHandIndex2", "RightHandIndex2", "Neck")):
        rows = [quaternion_xyzw(np.array([.4 + .01 * t, .1 * i, .2])) for t in range(frames)]
        for t in rest_frames:
            rows[t] = [0., 0., 0., 1.]  # node rest rotation is the identity here
        tracks[joint] = np.array(rows)
    return tracks


def test_rest_pose_end_frames_are_excluded_and_reported():
    frames = 40
    state, report = adapt(build_glb(frames=frames, rotations=_posed_tracks(frames, rest_frames=(0, 38, 39))))
    assert report.status == "converted", report.reason
    excluded = {f["frame"]: f["reason"] for f in report.invalid_frames}
    assert set(excluded) == {0, 38, 39} and all("rest pose" in r for r in excluded.values())
    assert not state.channels["body"].valid[[0, 38, 39]].any() and state.channels["body"].valid[1:38].all()


def test_interior_rest_frames_and_posed_ends_are_kept():
    frames = 40
    _, report = adapt(build_glb(frames=frames, rotations=_posed_tracks(frames, rest_frames=(20,))))
    assert report.invalid_frames == []  # only end frames can be resets
    _, report = adapt(build_glb(frames=frames, rotations=_posed_tracks(frames, rest_frames=(39,))),
                      rest_pose_tolerance_degrees=None)
    assert report.invalid_frames == []  # the rule is explicit and can be disabled


def test_rest_pose_rule_counts_toward_the_exclusion_limit():
    frames = 40
    _, report = adapt(build_glb(frames=frames, rotations=_posed_tracks(frames, rest_frames=range(34, 40))))
    assert report.status == "refused" and "exclusion limit" in report.reason
    with pytest.raises(ValueError):
        adapt(build_glb(frames=frames), rest_pose_tolerance_degrees=-1.)
