"""The render filter, the collision correction, the CC0 wardrobe, and that no render setting can differ by arm or mark one."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from signtranslator.avatar_render.collision import (BodyField, _body_triangles, _part_of_bones, _signed_distance, _skin, apply_correction,
                                                    build_field, penetration)
from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_wardrobe import load_wardrobe
from signtranslator.avatar_render.motion_filter import (FILTER_V1, FilterSettings, OneEuro, SavitzkyGolay, _from_6d, _to_6d, fill_gaps,
                                                        filter_rotations, filter_state, filter_values)
from signtranslator.avatar_render.motion_quality import (articulation, forward_kinematics, geodesic_degrees, hf_share, position_jitter_mm,
                                                         preservation, rotation_jitter_degrees, shake)
from signtranslator.avatar_render.retarget import build_retarget_binding, retarget_state
from signtranslator.avatar_render.retarget_check import raised_arm_points, state_from_points

ROOT = Path(__file__).resolve().parents[1]
INTAKE = ROOT / "Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
FPS = 24.
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def rig():
    return load_makehuman_rig(INTAKE / "makehuman")


@pytest.fixture(scope="module")
def binding(rig):
    return build_retarget_binding(rig)


@pytest.fixture(scope="module")
def field(rig):
    return build_field(rig)


def about_x(angles):
    return Rotation.from_rotvec(np.outer(angles, [1., 0., 0.])).as_matrix()[:, None]


def angle_of(rotations):
    return Rotation.from_matrix(rotations[:, 0]).as_rotvec()[:, 0]


def handshape_changes(rate, amplitude=1., ramp=.1, seconds=6.):
    """Plateaus joined by ramps, ``rate`` changes a second: the fastest articulation a filter must keep (fingerspelling runs near 6)."""
    fine = np.arange(0., seconds, 1. / 960.)
    value = np.zeros_like(fine)
    period = 1. / rate
    for k in range(1, int(seconds * rate) + 1):
        start = k * period
        before, after = amplitude * ((k - 1) % 2), amplitude * (k % 2)
        inside = (fine >= start) & (fine < start + ramp)
        value[inside] = before + (after - before) * (fine[inside] - start) / ramp
        value[fine >= start + ramp] = after
    return value[::40], period


def plateau_reach(output, rate, period, amplitude=1., ramp=.1):
    index = [int(round((k * period + ramp + (period - ramp) / 2) * FPS)) for k in range(1, int(6 * rate) - 1)]
    return np.abs(output[index[1::2]] - output[index[0::2][:len(index[1::2])]]).mean() / amplitude


# ---------------------------------------------------------------------- filter: what it must keep and what it must remove

@pytest.mark.parametrize("spec", [SavitzkyGolay(5, 2), SavitzkyGolay(9, 2), OneEuro(3., 3.)])
def test_filter_keeps_a_constant_rotation_exactly(spec):
    rotation = Rotation.from_rotvec([.3, -.2, .9]).as_matrix()
    rotations = np.tile(rotation, (40, 1, 1, 1))
    assert np.allclose(filter_rotations(rotations, spec, FPS), rotations, atol=1e-12)


def test_savitzky_golay_keeps_a_ramp_exactly_and_has_no_lag():
    angles = np.linspace(0., 2., 48)
    output = angle_of(filter_rotations(about_x(angles), SavitzkyGolay(5, 2), FPS))
    assert np.allclose(output, angles, atol=1e-10)                               # no lag, ends included


def test_one_euro_both_ways_has_no_lag():
    t = np.arange(96) / FPS
    angles = np.sin(2 * np.pi * 1.5 * t)
    output = angle_of(filter_rotations(about_x(angles), OneEuro(3., 3.), FPS))
    lags = [np.mean(angles[12:-12 - abs(k) or None] * np.roll(output, -k)[12:-12 - abs(k) or None]) for k in range(-3, 4)]
    assert int(np.argmax(lags)) == 3


def test_light_filter_removes_white_noise_and_keeps_two_hertz_motion():
    t = np.arange(240) / FPS
    signal = .5 * np.sin(2 * np.pi * 2. * t)
    noise = np.random.default_rng(0).normal(0., np.radians(2.), len(t))
    clean = angle_of(filter_rotations(about_x(signal), SavitzkyGolay(5, 2), FPS))
    assert np.abs(clean[12:-12]).max() >= .98 * .5
    noisy = angle_of(filter_rotations(about_x(signal + noise), SavitzkyGolay(5, 2), FPS))
    assert np.std(noisy[12:-12] - signal[12:-12]) < .75 * np.std(noise)


def test_light_filter_keeps_handshape_changes_at_fingerspelling_rate_where_a_wider_window_loses_them():
    for rate in (2, 4, 6, 7):
        x, period = handshape_changes(rate)
        kept = angle_of(filter_rotations(about_x(x), SavitzkyGolay(5, 2), FPS))
        assert plateau_reach(kept, rate, period) > .92, rate
    x, period = handshape_changes(7)
    wide = angle_of(filter_rotations(about_x(x), SavitzkyGolay(9, 3), FPS))
    assert plateau_reach(wide, 7, period) < .6                                   # the wrong frequency does change what a sign looks like


def test_filter_table_keeps_the_finger_filter_light_and_is_fixed():
    assert FILTER_V1.finger == SavitzkyGolay(5, 2) and FILTER_V1.arm == SavitzkyGolay(5, 2) and FILTER_V1.hand == SavitzkyGolay(5, 2)
    assert FILTER_V1.torso == SavitzkyGolay(9, 2) and FILTER_V1.face == SavitzkyGolay(5, 2) and FILTER_V1.head == SavitzkyGolay(5, 2)
    assert all(getattr(FILTER_V1, name) is not None for name in FILTER_V1.__dataclass_fields__)
    assert FILTER_V1.sha256() == FilterSettings(**FILTER_V1.__dict__).sha256()


def test_filter_values_clips_nothing_and_keeps_a_parabola():
    t = np.arange(60, dtype=float)
    values = (.002 * t ** 2)[:, None]
    assert np.allclose(filter_values(values, SavitzkyGolay(5, 2), FPS), values, atol=1e-9)


def noisy_state(rig, frames=96, seed=3, *, noise_chest=np.radians(1.)):
    """A state from the rig's own rest landmarks, with chest noise that the stored arm rotations cancel, as estimated states carry."""
    rng = np.random.default_rng(seed)
    world, visible, hands = raised_arm_points(rig, "Left", frames=frames)
    state = state_from_points(np.arange(frames) / FPS, world, visible, hands, sample_id="noisy")
    body = state.channels["body"]
    values = body.values.astype(np.float64).copy()
    chest = body.labels.index("Chest")
    wiggle = Rotation.from_rotvec(rng.normal(0., noise_chest, (frames, 3))).as_matrix()
    matrices = _from_6d(values[:, chest])
    values[:, chest] = _to_6d(matrices @ wiggle)
    for name in ("LeftUpperArm", "RightUpperArm"):
        k = body.labels.index(name)
        values[:, k] = _to_6d(np.swapaxes(wiggle, -1, -2) @ _from_6d(values[:, k]))   # the arm's world direction is unchanged
    from dataclasses import replace
    channels = dict(state.channels, body=replace(body, values=values.astype(body.values.dtype)))
    return replace(state, channels=channels)


def test_filtering_the_chest_does_not_push_its_noise_into_the_arm(rig, binding):
    state = noisy_state(rig)
    before = retarget_state(rig, binding, state, allow_inferred=True).local_rotations
    after = retarget_state(rig, binding, filter_state(state, FilterSettings(torso=SavitzkyGolay(9, 2))), allow_inferred=True).local_rotations
    g0, g1 = forward_kinematics(rig, before), forward_kinematics(rig, after)
    i = rig.bone_names.index("lowerarm01.L")
    a0 = rotation_jitter_degrees(g0[:, i, :3, :3], FPS)
    a1 = rotation_jitter_degrees(g1[:, i, :3, :3], FPS)
    assert a1 <= a0 + 1e-6                                # world-frame filtering: the forearm's direction gains no noise from the chest's filter
    h = rig.bone_names.index("head")
    assert position_jitter_mm(g1[:, h, :3, 3], FPS) < .5 * position_jitter_mm(g0[:, h, :3, 3], FPS)


def test_filter_state_keeps_clock_shapes_validity_and_unit_gaze(rig):
    state = noisy_state(rig, frames=48)
    out = filter_state(state, FILTER_V1)
    out.validate()
    assert np.array_equal(out.timestamps, state.timestamps)
    for name, channel in state.channels.items():
        assert out.channels[name].values.shape == channel.values.shape
        assert np.array_equal(out.channels[name].valid, channel.valid)


# ---------------------------------------------------------------------- gaps

def test_fill_gaps_interpolates_counts_holds_the_ends_and_leaves_never_valid_alone(rig):
    from dataclasses import replace
    state = noisy_state(rig, frames=24)
    body = state.channels["body"]
    k = body.labels.index("LeftForearm")
    values, valid = body.values.astype(np.float64).copy(), body.valid.copy()
    start, end = Rotation.from_rotvec([0., 0., .2]).as_matrix(), Rotation.from_rotvec([0., 0., 1.]).as_matrix()
    values[5, k], values[15, k] = _to_6d(start), _to_6d(end)
    valid[6:15, k] = False
    valid[:3, k] = False
    never = body.labels.index("RightForearm")
    valid[:, never] = False
    values[~valid] = 0.
    confidence = np.where(valid, body.confidence, 0.).astype(body.confidence.dtype)
    state = replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype), valid=valid, confidence=confidence)))
    filled, record = fill_gaps(state)
    middle = _from_6d(filled.channels["body"].values[10, k].astype(np.float64))
    assert geodesic_degrees(middle, Rotation.from_rotvec([0., 0., .6]).as_matrix()) < 1e-2
    assert np.allclose(filled.channels["body"].values[0, k], filled.channels["body"].values[3, k])
    assert not filled.channels["body"].valid[:, never].any()
    row = record["body/LeftForearm"]
    assert row["filled_frames"] == 12 and row["gaps"] == 2
    filled.validate()


# ---------------------------------------------------------------------- measures

def test_jitter_measures_read_millimetres_and_shares():
    t = np.arange(240) / FPS
    still = np.zeros((240, 3))
    assert position_jitter_mm(still, FPS) == 0.
    rng = np.random.default_rng(0)
    noisy = rng.normal(0., .05, (240, 3))                 # 5 mm per axis in source units (decimetres)
    assert 4. < position_jitter_mm(noisy, FPS) < 9.
    slow = np.stack((np.sin(2 * np.pi * t), np.zeros(240), np.zeros(240)), axis=1)
    assert hf_share(slow, FPS) < .01 < .3 < hf_share(noisy, FPS)


def test_preservation_is_one_on_the_same_clip_and_holds_read_true(rig, binding):
    state = noisy_state(rig, frames=64)
    local = retarget_state(rig, binding, state, allow_inferred=True).local_rotations
    a = articulation(rig, local, FPS)
    kept = preservation(a, a, FPS)
    for side in "LR":
        assert kept[side]["hold_iou"] in (None, 1.) and kept[side]["finger_bend_lag_frames"] == 0
    record = shake(rig, local, FPS, screen_pixels_per_unit=768 / 8.4)
    assert record["points"]["head"]["jitter_mm"] >= 0. and "jitter_px" in record["points"]["wrist.L"]


# ---------------------------------------------------------------------- collision

def test_field_reads_inside_and_outside_at_rest_and_follows_the_bone(rig, field):
    rest_globals = rig.rest_globals.numpy()
    identity = forward_kinematics(rig, np.tile(np.eye(3), (1, len(rig.bone_names), 1, 1)))[0]
    assert np.allclose(identity, rest_globals, atol=1e-9)
    chest_inside = np.array([[0., 4.3, 0.2]])
    assert _signed_distance(field, identity, chest_inside)[0][0] < -.3
    assert _signed_distance(field, identity, np.array([[0., 4.3, 5.]]))[0][0] > .5
    head_inside = rest_globals[rig.bone_names.index("head"), :3, 3][None] + np.array([[0., .6, .2]])
    assert _signed_distance(field, identity, head_inside)[0][0] < 0.


def test_posed_torso_surface_reads_near_zero_in_the_field(rig, binding, field):
    """Caught a real bug: the part transform was composed in the wrong order, which is invisible at rest."""
    state = noisy_state(rig, frames=8, noise_chest=np.radians(8.))
    local = retarget_state(rig, binding, state, allow_inferred=True).local_rotations
    _, surface = _body_triangles(rig)
    dominant = rig.weights.numpy().argmax(1)
    chosen = surface[_part_of_bones(rig)[dominant[surface]] != ""]
    globals_ = forward_kinematics(rig, local)
    for t in (0, 4, 7):
        posed = rig.pose(torch.from_numpy(np.ascontiguousarray(local[t]))).vertices.numpy()
        distance, _ = _signed_distance(field, globals_[t], posed[chosen])
        assert np.percentile(np.abs(distance), 95) < .12            # 12 mm


def chest_pose(rig):
    """(T, B, 3, 3) locals with the left arm folded across the body, which puts the hand inside the torso on this rig."""
    rest = rig.rest_globals.numpy()[:, :3, :3]
    local = np.tile(np.eye(3), (48, len(rig.bone_names), 1, 1)).copy()
    upper, fore = rig.bone_names.index("upperarm01.L"), rig.bone_names.index("lowerarm01.L")
    local[:, upper] = rest[upper].T @ Rotation.from_euler("z", -35, degrees=True).as_matrix() @ rest[upper]
    local[:, fore] = rest[fore].T @ Rotation.from_euler("y", -115, degrees=True).as_matrix() @ rest[fore]
    return local, rig.bone_names.index("wrist.L")


def test_correction_takes_the_hand_out_along_depth_and_keeps_image_place_and_hand_rotation(rig, binding, field):
    local, wrist = chest_pose(rig)
    globals_ = forward_kinematics(rig, local)
    before = penetration(rig, field, local)
    assert before["hand.torso"]["frames_over_10mm"] > .5                 # the test pose does put the hand inside
    motion = retarget_state(rig, binding, noisy_state(rig, frames=48), allow_inferred=True)
    from dataclasses import replace
    pose = replace(motion, local_rotations=local, target_globals=globals_[:, :, :3, :3])
    fixed, record = apply_correction(rig, field, pose)
    after = forward_kinematics(rig, fixed.local_rotations)
    assert record["after"]["hand.torso"]["frames_over_0mm"] == 0.
    assert np.abs(after[:, wrist, :2, 3] - globals_[:, wrist, :2, 3]).max() < 1e-6      # same place in the front view
    assert after[:, wrist, 2, 3].min() > globals_[:, wrist, 2, 3].min()                 # moved toward the camera
    assert geodesic_degrees(after[:, wrist, :3, :3], globals_[:, wrist, :3, :3]).max() < 1e-5
    fingers = [i for i, n in enumerate(rig.bone_names) if n.startswith("finger") and n.endswith(".L")]
    assert np.array_equal(fixed.local_rotations[:, fingers], local[:, fingers])          # handshape untouched
    for parent, child in (("upperarm01.L", "lowerarm01.L"), ("lowerarm01.L", "wrist.L")):
        p, c = rig.bone_names.index(parent), rig.bone_names.index(child)
        assert np.allclose(np.linalg.norm(after[:, c, :3, 3] - after[:, p, :3, 3], axis=-1),
                           np.linalg.norm(globals_[:, c, :3, 3] - globals_[:, p, :3, 3], axis=-1), atol=1e-9)


def test_correction_leaves_a_clear_pose_exactly_alone(rig, binding, field):
    motion = retarget_state(rig, binding, state_from_points(np.arange(8) / FPS, *raised_arm_points(rig, "Left", frames=8), sample_id="clear"),
                            allow_inferred=True)
    fixed, record = apply_correction(rig, field, motion)
    assert record["before"]["hand.torso"]["frames_over_0mm"] == 0.
    assert np.array_equal(fixed.local_rotations[:, [rig.bone_names.index("wrist.L")]], motion.local_rotations[:, [rig.bone_names.index("wrist.L")]])


def test_correction_treats_a_clip_the_same_whatever_its_name(rig, binding, field):
    from dataclasses import replace
    local, _ = chest_pose(rig)
    motion = retarget_state(rig, binding, noisy_state(rig, frames=48), allow_inferred=True)
    a = replace(motion, local_rotations=local, target_globals=forward_kinematics(rig, local)[:, :, :3, :3], sample_id="real-clip")
    b = replace(a, sample_id="generated-clip")
    assert np.array_equal(apply_correction(rig, field, a)[0].local_rotations, apply_correction(rig, field, b)[0].local_rotations)


# ---------------------------------------------------------------------- assets

def test_wardrobe_files_are_pinned_and_every_text_asset_states_cc0(rig):
    root = INTAKE / "makehuman-system-assets"
    intake = json.loads((root / "intake.json").read_text())
    assert intake["source"]["sha256"] == "b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"
    for item in intake["files"]:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        if item["path"].endswith((".mhclo", ".mhmat")):
            assert "explicitly released as CC0" in (root / item["path"]).read_text(errors="ignore")
    names = " ".join(item["path"].lower() for item in intake["files"])
    assert "smpl" not in names and "mano" not in names
    assert "CC0" in (root / "LICENSE.ASSETS.md").read_text()
    wardrobe = load_wardrobe(root, rig)
    assert set(wardrobe.attachments) == {"hair", "brows", "lashes", "top"} and len(wardrobe.body_deleted_vertices) > 1000


def test_wardrobe_refuses_a_changed_file(rig, tmp_path):
    import shutil
    root = tmp_path / "assets"
    shutil.copytree(INTAKE / "makehuman-system-assets", root)
    target = root / "hair/short01/short01.mhmat"
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_wardrobe(root, rig)


def test_the_top_leaves_the_face_uncovered(rig):
    """Face must stay visible: nothing the wardrobe adds may cover the face's front, and the hair stays above the brows."""
    wardrobe = load_wardrobe(INTAKE / "makehuman-system-assets", rig)
    rest = rig.rest_globals.numpy()
    index = {name: i for i, name in enumerate(rig.bone_names)}
    eye_y = (rest[index["eye.L"], 1, 3] + rest[index["eye.R"], 1, 3]) / 2
    eye_x = abs(rest[index["eye.L"], 0, 3])
    for name in ("hair",):
        v = wardrobe.attachments[name].rest_vertices.numpy()
        front = v[(v[:, 2] > rest[index["eye.L"], 2, 3]) & (np.abs(v[:, 0]) < 1.3 * eye_x)]
        assert front.size == 0 or front[:, 1].min() > eye_y + .3    # 3 cm above the eye line at the front of the face
    top = wardrobe.attachments["top"].rest_vertices.numpy()
    assert top[:, 1].max() < rest[index["head"], 1, 3]              # the collar stops below the head joint


# ---------------------------------------------------------------------- fairness and blinding

def _source(name: str) -> str:
    return (ROOT / "scripts" / name).read_text()


def test_renderer_has_no_option_that_could_name_an_arm():
    tree = ast.parse(_source("render_avatar.py"))
    options = {node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"}
    assert options == {"--state", "--key", "--sha256", "--store", "--out-dir", "--blender", "--resolution", "--start-seconds", "--max-seconds",
                       "--face-baseline", "--no-hinge-twist", "--drive-legs", "--lossless", "--allow-inferred"}


def test_renderer_never_decides_on_a_clip_name_or_clock():
    tree = ast.parse(_source("render_avatar.py"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            for part in ast.walk(node.test):
                assert not (isinstance(part, ast.Attribute) and part.attr in ("sample_id", "clock_id"))
    filter_source = (ROOT / "signtranslator/avatar_render/motion_filter.py").read_text()
    collision_source = (ROOT / "signtranslator/avatar_render/collision.py").read_text()
    for source in (filter_source, collision_source):
        assert "sample_id" not in source and "clock_id" not in source


def test_settings_hash_depends_on_settings_only():
    import render_avatar
    assert render_avatar.settings_sha256(768) == render_avatar.settings_sha256(768)
    assert render_avatar.settings_sha256(768) != render_avatar.settings_sha256(384)
    assert render_avatar.settings_sha256(768) != render_avatar.settings_sha256(768, options=dict(render_avatar.DEFAULT_OPTIONS, lossless=True))


def test_frame_is_a_constant_of_the_renderer_not_of_the_clip():
    import render_avatar
    source = _source("render_avatar.py")
    assert "framing(" not in source
    centre, span = render_avatar.frame()
    assert span == render_avatar.RENDER_SETTINGS["frame_span"] and centre[1] == render_avatar.RENDER_SETTINGS["frame_centre"][1]


def test_packet_builder_passes_one_argument_list_for_every_system(monkeypatch, tmp_path):
    import build_human_eval_packet as packet
    import render_avatar
    seen = []

    def fake_main(argv):
        seen.append([a for a in argv if not a.startswith(str(Path(argv[1]).parents[0]))])
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir()
        (out / "avatar.mp4").write_bytes(b"x")
        (out / "verification.json").write_text(json.dumps(dict(render_settings_sha256=render_avatar.settings_sha256(768), acceptance=dict(mirror=True))))
        return 0

    monkeypatch.setattr(render_avatar, "main", fake_main)
    rig = load_makehuman_rig(INTAKE / "makehuman")
    for sample in ("reference-clip", "generated-clip"):
        state = state_from_points(np.arange(3) / FPS, *raised_arm_points(rig, "Left", frames=3), sample_id=sample)
        assert packet.render_avatar_video(state, tmp_path / f"{sample}.mp4", resolution=768)["frames"] == 3
    stripped = [[a for a in call if a.startswith("--") or a.isdigit()] for call in seen]
    assert stripped[0] == stripped[1] == ["--state", "--out-dir", "--allow-inferred", "--resolution", "768"]


def test_packet_builder_refuses_a_render_with_other_settings(monkeypatch, tmp_path):
    import build_human_eval_packet as packet
    import render_avatar

    def other_settings(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir()
        (out / "avatar.mp4").write_bytes(b"x")
        (out / "verification.json").write_text(json.dumps(dict(render_settings_sha256="0" * 64, acceptance=dict(mirror=True))))
        return 0

    monkeypatch.setattr(render_avatar, "main", other_settings)
    rig = load_makehuman_rig(INTAKE / "makehuman")
    state = state_from_points(np.arange(3) / FPS, *raised_arm_points(rig, "Left", frames=3), sample_id="x")
    with pytest.raises(RuntimeError, match="other settings"):
        packet.render_avatar_video(state, tmp_path / "x.mp4")


def test_packet_builder_calls_the_avatar_with_the_packets_one_resolution():
    tree = ast.parse(_source("build_human_eval_packet.py"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "render_avatar_video"]
    assert len(calls) == 2
    for call in calls:
        assert all(isinstance(a, ast.Name) for a in call.args)      # state and path: no label, no system
        names = [keyword.arg for keyword in call.keywords]
        assert set(names) <= {"resolution", "verification"} and "verification" in names       # the verification path holds the item id, private, under key/


def test_avatar_fixes_refuses_clips_outside_d20():
    source = _source("avatar_fixes.py")
    assert 'refused, not a D20 clip' in source and 'SIGNERS = ("c01", "c03", "c05")' in source


@pytest.mark.skipif(not BLENDER.exists(), reason="local Blender is not installed")
def test_two_names_and_clocks_on_one_motion_give_byte_identical_video_with_no_mark_in_it(tmp_path):
    import av
    import render_avatar
    rig = load_makehuman_rig(INTAKE / "makehuman")
    world, visible, hands = raised_arm_points(rig, "Left", frames=4)
    world[1:, 13:16:2] += [.0, -.3, .3]
    records, payloads = [], []
    for sample, clock_offset in (("how2sign-video-estimate-real-clip", .25), ("round15-eval-generated-clip", 0.)):
        state = state_from_points(clock_offset + np.arange(4) / FPS, world, visible, hands, sample_id=sample)
        path = tmp_path / f"{sample}.npz"
        state.save(path)
        out = tmp_path / sample
        assert render_avatar.main(["--state", str(path), "--out-dir", str(out), "--allow-inferred", "--resolution", "96", "--blender", str(BLENDER)]) == 0
        payloads.append((out / "avatar.mp4").read_bytes())
        records.append(json.loads((out / "verification.json").read_text()))
        assert sample.encode() not in payloads[-1] and b"real" not in payloads[-1] and b"generated" not in payloads[-1]
        with av.open(str(out / "avatar.mp4")) as container:
            tags = dict(container.metadata)
            assert set(tags) <= {"major_brand", "minor_version", "compatible_brands", "encoder"}
            assert not any(sample in str(value) for value in tags.values())
    assert payloads[0] == payloads[1]
    assert records[0]["render_settings_sha256"] == records[1]["render_settings_sha256"]
    assert records[0]["deterministic_rerender"] and records[0]["camera"] == records[1]["camera"]
    assert records[0]["acceptance"]["passed"]
