"""The render filter, the collision correction, the CC0 wardrobe, and that no render setting can differ by arm or mark one."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from signtranslator.avatar_render import collision
from signtranslator.avatar_render.collision import (CORRECTION_V2, CorrectionSettings, _body_triangles, _part_of_bones, _signed_distance,
                                                    apply_correction, arm_image_changes, build_field, correct, hidden_from_front, penetration)
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


@pytest.fixture(scope="module")
def wardrobe(rig):
    return load_wardrobe(INTAKE / "makehuman-system-assets", rig)


@pytest.fixture(scope="module")
def drawn_field(rig, wardrobe):
    """The field the renderer builds: the top and the hair are part of what covers a hand."""
    return build_field(rig, drawn=wardrobe.attachments)


def _triangles(faces):
    return np.array([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)], dtype=np.int64)


def drawn_triangles(rig, wardrobe, *, limbs=None):
    """{object: (F, 3) vertex indices} of what is drawn in front of the camera's ray: the body's torso and head where the bundle keeps them,
    the top without its sleeves, the hair. ``limbs`` ("arm.L", "hand.R"): one limb instead, sleeve included, for cover by the other arm."""
    parts = _part_of_bones(rig)
    dominant = rig.weights.numpy().argmax(1)
    deleted = set(int(v) for v in wardrobe.body_deleted_vertices)
    if limbs is None:
        keep_bone = parts != ""
    else:
        kind, side = limbs.split(".")
        keep_bone = np.array([n.endswith(f".{side}") and n.startswith(collision.LIMB_PREFIXES[kind]) for n in rig.bone_names])
    body = [f for i, f in enumerate(rig.faces) if "body" in rig.face_groups[i] and all(keep_bone[dominant[v]] for v in f)
            and (limbs is not None or not all(v in deleted for v in f))]
    out = {"body": _triangles(body)}
    for name in (("top", "hair") if limbs is None else ("top",) if limbs.startswith("arm") else ()):
        attachment = wardrobe.attachments[name]
        keep = keep_bone[attachment.weights.numpy().argmax(1)]
        faces = [f for f in attachment.faces if all(keep[v] for v in f)]
        if faces:
            out[name] = _triangles(faces)
    return out


def exact_gaps(rig, wardrobe, local, hand, triangles=None):
    """(T, V): how far each hand vertex sits in front of the frontmost drawn triangle on its ray to the camera (+z), in source units;
    negative where drawn body is in front of it, +inf where no triangle is on the ray. Exact: the rig and the attachments posed by their own
    skinning, every triangle tested, no cells and no samples. Independent of ``collision.front_maps``."""
    from signtranslator.avatar_render.rigged import apply_lbs
    triangles = drawn_triangles(rig, wardrobe) if triangles is None else triangles
    out = np.full((len(local), len(hand)), np.inf)
    for t in range(len(local)):
        posed = rig.pose(torch.from_numpy(np.ascontiguousarray(local[t])))
        vertices = {"body": posed.vertices.numpy().astype(np.float64)}
        for name in triangles:
            if name != "body":
                attachment = wardrobe.attachments[name]
                vertices[name] = apply_lbs(attachment.rest_vertices, attachment.weights, posed.skin_transforms).numpy().astype(np.float64)
        p = vertices["body"][hand]
        low, high = p[:, :2].min(0) - .01, p[:, :2].max(0) + .01
        for name, tri in triangles.items():
            c = vertices[name][tri]
            c = c[(c[..., :2].max(1) >= low).all(1) & (c[..., :2].min(1) <= high).all(1)]
            if not len(c):
                continue
            a, b, d = c[:, 0], c[:, 1], c[:, 2]
            det = (b[:, 1] - d[:, 1]) * (a[:, 0] - d[:, 0]) + (d[:, 0] - b[:, 0]) * (a[:, 1] - d[:, 1])
            ok = np.abs(det) > 1e-14
            a, b, d, det = a[ok], b[ok], d[ok], det[ok]
            x, y = p[:, None, 0], p[:, None, 1]
            u = ((b[:, 1] - d[:, 1]) * (x - d[:, 0]) + (d[:, 0] - b[:, 0]) * (y - d[:, 1])) / det
            w = ((d[:, 1] - a[:, 1]) * (x - d[:, 0]) + (a[:, 0] - d[:, 0]) * (y - d[:, 1])) / det
            z = u * a[:, 2] + w * b[:, 2] + (1. - u - w) * d[:, 2]
            front = np.where((u >= 0) & (w >= 0) & (1. - u - w >= 0), z, -np.inf).max(1)
            out[t] = np.minimum(out[t], np.where(np.isfinite(front), p[:, 2] - front, np.inf))
    return out


def arm_pose(rig, frames, side, upper, fore, base=None):
    """Locals with one arm swung by ``upper`` (z, x degrees) and bent by ``fore`` (y degrees) from rest."""
    rest = rig.rest_globals.numpy()[:, :3, :3]
    local = np.tile(np.eye(3), (frames, len(rig.bone_names), 1, 1)).copy() if base is None else base.copy()
    u, f = rig.bone_names.index(f"upperarm01.{side}"), rig.bone_names.index(f"lowerarm01.{side}")
    local[:, u] = rest[u].T @ Rotation.from_euler("zx", upper, degrees=True).as_matrix() @ rest[u]
    local[:, f] = rest[f].T @ Rotation.from_euler("y", fore, degrees=True).as_matrix() @ rest[f]
    return local


def two_arm_poses(rig, count, seed=3, frames=3):
    """Fixed random poses, both arms down to forward and bent into signing space: hands across, on and behind the body."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        local = None
        for side, sign in (("L", 1), ("R", -1)):
            local = arm_pose(rig, frames, side, [rng.uniform(-75, 20) * sign, rng.uniform(-10, 90)], [rng.uniform(-150, -40) * sign], base=local)
        out.append(local)
    return out


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


def chest_pose(rig, frames=48):
    """(T, B, 3, 3) locals with the left arm folded across the body, which puts the hand 4 cm inside the torso on this rig, within the cap."""
    return arm_pose(rig, frames, "L", [-35., 0.], [-95.]), rig.bone_names.index("wrist.L")


def test_correction_takes_the_hand_out_along_depth_and_keeps_image_place_and_hand_rotation(rig, binding, field, wardrobe, drawn_field):
    local, wrist = chest_pose(rig)
    globals_ = forward_kinematics(rig, local)
    before = penetration(rig, field, local)
    assert before["hand.torso"]["frames_over_10mm"] > .5                 # the test pose does put the hand inside
    hand = drawn_field.hand_vertices["L"]
    assert (exact_gaps(rig, wardrobe, local[:2], hand) < 0.).any(1).all()     # and an exact ray finds it behind the drawn body
    motion = retarget_state(rig, binding, noisy_state(rig, frames=48), allow_inferred=True)
    from dataclasses import replace
    pose = replace(motion, local_rotations=local, target_globals=globals_[:, :, :3, :3])
    fixed, record = apply_correction(rig, drawn_field, pose)
    after = forward_kinematics(rig, fixed.local_rotations)
    assert penetration(rig, field, fixed.local_rotations)["hand.torso"]["frames_over_0mm"] == 0.
    assert exact_gaps(rig, wardrobe, fixed.local_rotations[::12], hand).min() >= collision.CLEARANCE      # in front of every drawn triangle
    assert not record["depth_capped"] and record["moved"]["L"]["frames_at_depth_cap"] == 0
    assert np.abs(after[:, wrist, :2, 3] - globals_[:, wrist, :2, 3]).max() < 1e-6      # same place in the front view
    assert after[:, wrist, 2, 3].min() > globals_[:, wrist, 2, 3].min()                 # moved toward the camera
    assert geodesic_degrees(after[:, wrist, :3, :3], globals_[:, wrist, :3, :3]).max() < 1e-5
    fingers = [i for i, n in enumerate(rig.bone_names) if n.startswith("finger") and n.endswith(".L")]
    assert np.array_equal(fixed.local_rotations[:, fingers], local[:, fingers])          # handshape untouched
    for parent, child in (("upperarm01.L", "lowerarm01.L"), ("lowerarm01.L", "wrist.L")):
        p, c = rig.bone_names.index(parent), rig.bone_names.index(child)
        assert np.allclose(np.linalg.norm(after[:, c, :3, 3] - after[:, p, :3, 3], axis=-1),
                           np.linalg.norm(globals_[:, c, :3, 3] - globals_[:, p, :3, 3], axis=-1), atol=1e-9)


def behind_the_back_pose(rig, field):
    """Locals with the left hand behind the torso and not inside it, found by a fixed search; the first pose that fits."""
    rest = rig.rest_globals.numpy()[:, :3, :3]
    upper, fore = rig.bone_names.index("upperarm01.L"), rig.bone_names.index("lowerarm01.L")
    for down, back, across in ((-40., 40., -60.), (-50., 45., -70.), (-40., 60., -90.), (-55., 30., -100.), (-30., 70., -90.)):
        local = np.tile(np.eye(3), (24, len(rig.bone_names), 1, 1)).copy()
        local[:, upper] = rest[upper].T @ Rotation.from_euler("zx", [down, back], degrees=True).as_matrix() @ rest[upper]
        local[:, fore] = rest[fore].T @ Rotation.from_euler("y", across, degrees=True).as_matrix() @ rest[fore]
        inside = penetration(rig, field, local)["hand.torso"]["frames_over_0mm"]
        seen = hidden_from_front(rig, field, local)
        if inside == 0. and seen["frames_half_of_a_hand"] == 1.:
            return local
    raise AssertionError("no test pose has the hand behind the body")


def retarget_shell(rig, binding, local):
    from dataclasses import replace
    motion = retarget_state(rig, binding, noisy_state(rig, frames=len(local)), allow_inferred=True)
    return replace(motion, local_rotations=local, target_globals=forward_kinematics(rig, local)[:, :, :3, :3])


def test_a_hand_behind_the_body_moves_at_most_the_cap_stays_covered_and_is_flagged(rig, binding, field, wardrobe, drawn_field):
    """A hand behind the back would travel through the torso; the cap stops it at 15 cm, and the record says so."""
    local = behind_the_back_pose(rig, field)
    fixed, record = apply_correction(rig, drawn_field, retarget_shell(rig, binding, local))
    wrist = rig.bone_names.index("wrist.L")
    before, after = forward_kinematics(rig, local), forward_kinematics(rig, fixed.local_rotations)
    assert np.abs(before[:, wrist, :2, 3] - after[:, wrist, :2, 3]).max() < 1e-6                         # still only along depth
    move = after[:, wrist, 2, 3] - before[:, wrist, 2, 3]
    assert collision.DEPTH_CAP == 1.5 and np.abs(move - collision.DEPTH_CAP).max() < 1e-6                 # exactly the cap, every frame
    assert record["depth_capped"] and record["depth_capped_by_side"] == dict(L=True, R=False)
    moved = record["moved"]["L"]
    assert moved["depth_capped"] and moved["frames_at_depth_cap"] == len(local) and moved["wrist_depth_move_cm_max"] <= 15. + 1e-6
    assert record["hidden_after"]["frames_any_vertex"] == 1.                                              # left covered, and counted
    assert (exact_gaps(rig, wardrobe, fixed.local_rotations[:1], drawn_field.hand_vertices["L"]) < 0.).any()


def test_a_hand_just_inside_the_back_and_one_just_behind_it_end_together(rig, binding, field, drawn_field):
    """Before, one point past the back's surface jumped through the torso and the next stayed hidden, so a hand could vanish between frames.
    Both now stop at the cap, together."""
    wrist = rig.bone_names.index("wrist.L")
    local = behind_the_back_pose(rig, field)
    fixed, first = apply_correction(rig, drawn_field, retarget_shell(rig, binding, local))
    shallow = local.copy()
    upper = rig.bone_names.index("upperarm01.L")
    shallow[:, upper] = shallow[:, upper] @ Rotation.from_euler("x", -6, degrees=True).as_matrix()      # the same hand a little deeper, into the back
    fixed_shallow, second = apply_correction(rig, drawn_field, retarget_shell(rig, binding, shallow))
    a = forward_kinematics(rig, fixed.local_rotations)[0, wrist, :3, 3]
    b = forward_kinematics(rig, fixed_shallow.local_rotations)[0, wrist, :3, 3]
    assert first["depth_capped"] and second["depth_capped"]
    assert abs(a[2] - b[2]) < 1.5


def test_correction_does_not_move_an_arm_for_a_forearm_inside_while_the_hand_is_clear(rig, binding, field):
    rest = rig.rest_globals.numpy()[:, :3, :3]
    upper, fore = rig.bone_names.index("upperarm01.L"), rig.bone_names.index("lowerarm01.L")
    for down, across in ((-50., -40.), (-60., -30.), (-55., -50.), (-65., -45.), (-70., -35.)):
        local = np.tile(np.eye(3), (12, len(rig.bone_names), 1, 1)).copy()
        local[:, upper] = rest[upper].T @ Rotation.from_euler("z", down, degrees=True).as_matrix() @ rest[upper]
        local[:, fore] = rest[fore].T @ Rotation.from_euler("y", across, degrees=True).as_matrix() @ rest[fore]
        seen = penetration(rig, field, local)
        if seen["forearm.torso"]["frames_over_0mm"] == 1. and hidden_from_front(rig, field, local)["frames_any_vertex"] == 0.:
            break
    else:
        pytest.skip("no fixed pose has only the forearm inside")
    fixed, record = apply_correction(rig, field, retarget_shell(rig, binding, local))
    assert np.array_equal(fixed.local_rotations, local) and record["moved"]["L"]["frames_depth_moved"] == 0


def test_shipped_correction_is_pinned_and_keeps_the_arm_in_the_image(rig, binding, field):
    assert CORRECTION_V2 == CorrectionSettings(forearm_drives_wrist=False, swivel_cap_deg=0., elbow_nearest_in_image=True)
    assert collision.CLEARANCE == .03 and collision.DEPTH_CAP == 1.5      # 3 mm past the drawn surface, at most 15 cm forward
    local, wrist = chest_pose(rig)
    fixed, record = apply_correction(rig, field, retarget_shell(rig, binding, local))
    assert record["moved"]["L"]["swivel_deg_max"] == 0.
    changes = arm_image_changes(rig, forward_kinematics(rig, local), forward_kinematics(rig, fixed.local_rotations), 768 / 9.)["L"]
    old, _, _ = correct(rig, field, local, forward_kinematics(rig, local)[:, :, :3, :3], CorrectionSettings(True, 70., False))
    worse = arm_image_changes(rig, forward_kinematics(rig, local), forward_kinematics(rig, old), 768 / 9.)["L"]
    assert changes["elbow_image_shift_px"]["max"] < worse["elbow_image_shift_px"]["max"]
    assert changes["forearm_image_length_ratio"]["minimum"] > worse["forearm_image_length_ratio"]["minimum"]


def test_the_hand_ends_just_in_front_of_the_body_not_far_from_it(rig, binding, wardrobe, drawn_field):
    """A clearance of 5 cm would float a contact sign in front of the chest. Read by an exact ray, not by the correction's own maps."""
    local, _ = chest_pose(rig, frames=8)
    fixed, _ = apply_correction(rig, drawn_field, retarget_shell(rig, binding, local))
    gaps = exact_gaps(rig, wardrobe, fixed.local_rotations, drawn_field.hand_vertices["L"])
    nearest = np.where(np.isfinite(gaps), gaps, np.inf).min(1)                 # each frame's closest hand vertex to the surface over it
    assert (nearest >= collision.CLEARANCE).all() and np.median(nearest) < .2   # at least the clearance, under 2 cm of map cell and smoothing


def test_an_exact_ray_finds_every_corrected_hand_in_front_of_the_drawn_body_by_the_clearance(rig, wardrobe, drawn_field):
    """Fixed random poses, both arms: wherever a hand was covered and the cap wasn't reached, every hand vertex ends at least the clearance
    in front of every drawn triangle, the top and hair included. Catches a front map without the top, a map read without its edge margin,
    and a correction without its clearance."""
    checked = 0
    for local in two_arm_poses(rig, 6):
        fixed, _, record = correct(rig, drawn_field, local, forward_kinematics(rig, local)[:, :, :3, :3], CORRECTION_V2)
        for side in ("L", "R"):
            hand = drawn_field.hand_vertices[side]
            if record[side]["depth_capped"] or not (exact_gaps(rig, wardrobe, local[:1], hand) < 0.).any():
                continue
            assert exact_gaps(rig, wardrobe, fixed, hand).min() >= collision.CLEARANCE, side
            checked += 1
    assert checked >= 6


def test_the_elbow_goes_to_the_nearest_point_of_its_whole_circle(rig, drawn_field):
    """Against an independent search over the full circle of elbows the fixed bone lengths allow, at 0.1 degree: the shipped elbow is the
    nearest, in the image, to where it was drawn. Right arms here have their nearest point on the circle's other half."""
    index = {n: i for i, n in enumerate(rig.bone_names)}
    checked = 0
    for local in two_arm_poses(rig, 6):
        before = forward_kinematics(rig, local)
        fixed, _, record = correct(rig, drawn_field, local, before[:, :, :3, :3], CORRECTION_V2)
        after = forward_kinematics(rig, fixed)
        for side in ("L", "R"):
            if record[side]["depth_capped"] or record[side]["frames_depth_moved"] == 0:
                continue
            s, e, w = (index[f"upperarm01.{side}"], index[f"lowerarm01.{side}"], index[f"wrist.{side}"])
            shoulder, elbow, wrist = before[:, s, :3, 3], before[:, e, :3, 3], after[:, w, :3, 3]
            a, b = np.linalg.norm(elbow - shoulder, axis=-1), np.linalg.norm(before[:, w, :3, 3] - elbow, axis=-1)
            d = np.linalg.norm(wrist - shoulder, axis=-1)
            axis = (wrist - shoulder) / d[:, None]
            along = (a ** 2 - b ** 2 + d ** 2) / (2 * d)
            radius = np.sqrt(np.clip(a ** 2 - along ** 2, 0., None))
            first = np.cross(axis, [0., 0., 1.])
            first /= np.linalg.norm(first, axis=-1, keepdims=True)
            second = np.cross(axis, first)
            angles = np.radians(np.arange(0., 360., .1))
            circle = (shoulder + axis * along[:, None])[:, None] + radius[:, None, None] * (first[:, None] * np.cos(angles)[None, :, None] + second[:, None] * np.sin(angles)[None, :, None])
            best = np.linalg.norm((circle - elbow[:, None])[..., :2], axis=-1).min(1)
            got = np.linalg.norm((after[:, e, :3, 3] - elbow)[:, :2], axis=-1)
            assert (got <= best + .01).all(), side                                       # within 1 mm of the nearest
            checked += 1
    assert checked >= 6


def test_cover_by_the_other_arm_and_hand_is_counted_and_agrees_with_an_exact_ray(rig, wardrobe, drawn_field):
    from signtranslator.avatar_render.collision import covered_by_other_limb
    for local in two_arm_poses(rig, 3)[:3:2]:                                    # the right hand behind the left arm, then hands apart
        counted = covered_by_other_limb(rig, drawn_field, local)["other_arm"]["frames_any_by_side"]
        exact = {side: float((exact_gaps(rig, wardrobe, local[:1], drawn_field.hand_vertices[side], drawn_triangles(rig, wardrobe, limbs=f"arm.{other}")) < 0.).any())
                 for side, other in (("L", "R"), ("R", "L"))}
        assert counted == exact
    assert counted == dict(L=0., R=0.) and covered_by_other_limb(rig, drawn_field, two_arm_poses(rig, 1)[0])["other_arm"]["frames_any_by_side"]["R"] == 1.
    clear = arm_pose(rig, 2, "L", [-75., 10.], [-20.], base=arm_pose(rig, 2, "R", [75., 10.], [20.]))
    nothing = covered_by_other_limb(rig, drawn_field, clear)
    assert nothing["other_arm"]["frames_any_vertex"] == 0. and nothing["other_hand"]["frames_any_vertex"] == 0.


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


def fake_record(**extra):
    """What a real render's verification.json carries for the packet: the settings hash and code read now, acceptance and the correction's flags."""
    import render_avatar
    return dict(dict(render_settings_sha256=render_avatar.settings_sha256(768), code=render_avatar.code_commit(), acceptance=dict(mirror=True),
                     collision=dict(depth_capped=False, cover_by_other_limb_after=dict(other_arm=dict(frames_any_vertex=0.), other_hand=dict(frames_any_vertex=0.)))), **extra)


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
        (out / "verification.json").write_text(json.dumps(fake_record()))
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
        (out / "verification.json").write_text(json.dumps(fake_record(**changed)))
        return 0

    monkeypatch.setattr(render_avatar, "main", other_settings)
    rig = load_makehuman_rig(INTAKE / "makehuman")
    state = state_from_points(np.arange(3) / FPS, *raised_arm_points(rig, "Left", frames=3), sample_id="x")
    for changed, match in ((dict(render_settings_sha256="0" * 64), "other settings"), (dict(code=dict(render_avatar.code_commit(), digest="0" * 64)), "other code")):
        with pytest.raises(SystemExit, match=match):
            packet.render_avatar_video(state, tmp_path / "x.mp4")
        assert not (tmp_path / "x.mp4").exists()


def test_packet_builder_calls_the_avatar_with_the_packets_one_resolution():
    tree = ast.parse(_source("build_human_eval_packet.py"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "render_avatar_video"]
    assert len(calls) == 2
    for call in calls:
        assert all(isinstance(a, ast.Name) for a in call.args)      # state and path: no label, no system
        names = [keyword.arg for keyword in call.keywords]
        assert set(names) <= {"resolution", "verification", "archive", "expected"} and {"verification", "archive", "expected"} <= set(names)     # keyed by the item id, private


def edited_on_disk(monkeypatch, path: Path):
    """Every read of ``path`` from now on returns other bytes, as an edit to the file would."""
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: original(self) + b"\n# edited" if self.resolve() == path.resolve() else original(self))


def test_the_hash_moves_with_everything_that_shapes_a_video(monkeypatch):
    import numpy
    import render_avatar
    from signtranslator.avatar_render import motion_filter
    from signtranslator.generation import stream_render
    base = render_avatar.settings_sha256(768)
    assert render_avatar.settings_sha256(768) == base                         # nothing is cached that could hide a change: read again, equal

    def moved(target, name, value):
        with monkeypatch.context() as patch:
            patch.setattr(target, name, value)
            return render_avatar.settings_sha256(768) != base

    assert moved(render_avatar, "RENDER_SETTINGS", dict(render_avatar.RENDER_SETTINGS, samples=8))
    assert moved(motion_filter, "FILTER_V1", FilterSettings())
    assert moved(collision, "CLEARANCE", .5)
    assert moved(collision, "DEPTH_CAP", 2.)
    assert moved(collision, "CORRECTION_V2", CorrectionSettings(True, 70., False))
    assert moved(render_avatar, "ENCODER", dict(render_avatar.ENCODER, quality="review"))
    assert moved(stream_render, "CONSTANT_RATE", dict(stream_render.CONSTANT_RATE, bits_per_second=1))
    assert moved(render_avatar, "blender_version", lambda blender=None: "Blender 9")
    assert moved(render_avatar, "x264_build", lambda: "x264 - core 1")
    assert moved(numpy, "__version__", "0.0")
    payload = render_avatar.settings_payload(768)
    assert set(payload["libraries"]) == {"python", "numpy", "scipy", "torch", "pillow", "pyav", "ffmpeg", "x264", "blender"}
    assert payload["libraries"]["x264"].startswith("x264 - core") and any(k.startswith("makehuman-system-assets/") and k.endswith("intake.json") for k in payload["assets"])
    # an edit on disk moves it at once, to a render module, to any other repository module, and to an asset
    for name in ("signtranslator/avatar_render/collision.py", "scripts/render_avatar.py", "signtranslator/eval_framework/human_study.py",
                 "Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake/makehuman-system-assets/intake.json"):
        with monkeypatch.context() as patch:
            edited_on_disk(patch, ROOT / name)
            assert render_avatar.settings_sha256(768) != base, name


def test_a_render_refuses_code_that_changed_while_it_ran(tmp_path, monkeypatch, rig):
    import render_avatar
    from PIL import Image

    def blender_while_someone_edits(blender, bundle, frames_dir, resolution, log, *, first_only=False):
        Path(frames_dir).mkdir()
        for i in range(1 if first_only else np.load(bundle)["body.vertices"].shape[0]):
            Image.new("RGB", (resolution, resolution)).save(Path(frames_dir) / f"frame-{i:06d}.png")
        if not first_only:
            edited_on_disk(monkeypatch, ROOT / "signtranslator/avatar_render/collision.py")
    monkeypatch.setattr(render_avatar, "run_blender", blender_while_someone_edits)
    state = state_from_points(np.arange(4) / FPS, *raised_arm_points(rig, "Left", frames=4), sample_id="x")
    state.save(tmp_path / "s.npz")
    with pytest.raises(SystemExit, match="changed while this render ran"):
        render_avatar.main(["--state", str(tmp_path / "s.npz"), "--out-dir", str(tmp_path / "out"), "--allow-inferred", "--resolution", "32"])
    assert not (tmp_path / "out" / "verification.json").exists()


def test_the_code_is_named_by_commit_dirty_and_digest_and_by_the_digest_alone_without_git(tmp_path):
    import subprocess
    import render_avatar
    from signtranslator.eval import provenance
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "signtranslator").mkdir()
    (repo / "scripts" / "x.py").write_text("a = 1\n")
    (repo / "signtranslator" / "y.py").write_text("b = 1\n")
    (repo / "notes.txt").write_text("not code\n")
    no_git = render_avatar.code_commit(repo)
    assert no_git == dict(commit=None, dirty=None, digest=provenance.code_digest(root=repo))       # a source archive, as on a pod
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "x")
    clean = render_avatar.code_commit(repo)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False and clean["digest"] == no_git["digest"]
    (repo / "notes.txt").write_text("changed, not code\n")
    assert render_avatar.code_commit(repo)["dirty"] is False
    (repo / "signtranslator" / "y.py").write_text("b = 2\n")
    edited = render_avatar.code_commit(repo)
    assert edited["dirty"] is True and edited["commit"] == clean["commit"] and edited["digest"] != clean["digest"]


def test_a_packet_of_mixed_renders_is_refused():
    import build_human_eval_packet as packet
    one = dict(render_settings_sha256="a", code_commit="c", code_dirty=False, code_digest="d")
    assert packet.refuse_mixed_renders([one, dict(one)]) == dict(settings_sha256="a", code_commit="c", code_dirty=False, code_digest="d")
    assert packet.refuse_mixed_renders([dict(system="x")]) == {}
    for other in (dict(one, render_settings_sha256="b"), dict(one, code_commit="d"), dict(one, code_dirty=True), dict(one, code_digest="e"),
                  dict(one, code_commit=None, code_dirty=None)):
        with pytest.raises(SystemExit, match="mixes renders"):
            packet.refuse_mixed_renders([one, other])


def bundle_of(tmp_path, monkeypatch, state, name):
    """render_avatar.main with Blender stubbed: the bundle is exactly what Blender would draw."""
    import render_avatar
    from PIL import Image

    def fake_blender(blender, bundle, frames_dir, resolution, log, *, first_only=False):
        Path(frames_dir).mkdir()
        count = 1 if first_only else np.load(bundle)["body.vertices"].shape[0]
        for i in range(count):
            Image.new("RGB", (resolution, resolution)).save(Path(frames_dir) / f"frame-{i:06d}.png")
    monkeypatch.setattr(render_avatar, "run_blender", fake_blender)
    path = tmp_path / f"{name}.npz"
    state.save(path)
    out = tmp_path / name
    assert render_avatar.main(["--state", str(path), "--out-dir", str(out), "--allow-inferred", "--resolution", "32"]) == 0
    return np.load(out / "bundle.npz"), json.loads((out / "verification.json").read_text())


def noisy_hand_on_chest(rig, frames=48, seed=5):
    """A state through the project's own adapter: the left arm folded so its hand lands in the chest, with noise on the chest as estimated states carry."""
    from dataclasses import replace
    from signtranslator.data_engineering.video_landmarks import L_ELBOW, L_SHOULDER, L_WRIST
    from signtranslator.avatar_render.retarget import rest_landmarks
    world, visible, hands = rest_landmarks(rig)
    world = world.copy()
    shoulder, elbow = world[0, L_SHOULDER], world[0, L_ELBOW]
    hand_points, found = hands["Left"]
    points = np.vstack((world[0, [L_ELBOW, L_WRIST]], hand_points[0]))
    down = Rotation.from_euler("z", -25, degrees=True).as_matrix()
    points = (points - shoulder) @ down.T + shoulder
    new_elbow = points[0]
    across = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    points[1:] = (points[1:] - new_elbow) @ across.T + new_elbow
    world[0, L_ELBOW], world[0, L_WRIST] = points[0], points[1]
    hands = dict(hands, Left=(points[2:][None], found))
    tile = lambda a: np.repeat(a, frames, axis=0)                                                  # noqa: E731
    state = state_from_points(np.arange(frames) / FPS, tile(world), tile(visible), {k: (tile(p), np.repeat(f, frames)) for k, (p, f) in hands.items()}, sample_id="on-chest")
    body = state.channels["body"]
    values = body.values.astype(np.float64).copy()
    k = body.labels.index("Chest")
    wiggle = Rotation.from_rotvec(np.random.default_rng(seed).normal(0., np.radians(1.5), (frames, 3))).as_matrix()
    values[:, k] = _to_6d(_from_6d(values[:, k]) @ wiggle)
    return replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype))))


@pytest.fixture
def render_parts(rig, binding):
    from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
    from signtranslator.avatar_render.makehuman_wardrobe import load_wardrobe
    wardrobe = load_wardrobe(INTAKE / "makehuman-system-assets", rig)
    return load_makehuman_face_units(INTAKE / "makehuman-face", rig), wardrobe, build_field(rig, drawn=wardrobe.attachments)


def skinned(rig, local):
    out = []
    for t in range(len(local)):
        out.append(rig.pose(torch.from_numpy(np.ascontiguousarray(local[t]))).vertices.numpy())
    return np.stack(out)


def test_the_bundle_holds_the_filtered_and_corrected_motion(tmp_path, monkeypatch, rig, binding, render_parts):
    state = noisy_hand_on_chest(rig)
    bundle, record = bundle_of(tmp_path, monkeypatch, state, "drawn")
    face_units, wardrobe, field = render_parts
    options = dict(allow_inferred=True, face_units=face_units)
    raw = retarget_state(rig, binding, fill_gaps(state)[0], **options)
    filtered = retarget_state(rig, binding, filter_state(fill_gaps(state)[0], FILTER_V1), **options)
    final, _ = apply_correction(rig, field, filtered)
    drawn = bundle["body.vertices"]
    assert np.abs(drawn - skinned(rig, final.local_rotations)).max() < 1e-3                      # exactly this pipeline
    assert np.abs(drawn - skinned(rig, filtered.local_rotations)).max() > .1                     # the correction moved something
    head = rig.weights.numpy().argmax(1) == rig.bone_names.index("head")
    jitter = lambda v: position_jitter_mm(v[:, head].mean(1), FPS)                               # noqa: E731
    assert jitter(drawn) < .5 * jitter(skinned(rig, raw.local_rotations))                        # and the filter ran
    assert record["collision"]["hidden_before"]["frames_any_vertex"] > 0. and record["collision"]["hidden_after"]["frames_any_vertex"] == 0.


def test_acceptance_reads_the_pose_as_drawn(tmp_path, monkeypatch, rig):
    _, record = bundle_of(tmp_path, monkeypatch, noisy_hand_on_chest(rig), "accept")
    assert record["direction_errors"]["groups"]["forearm"]["max_deg"] > record["direction_errors_before_correction"]["groups"]["forearm"]["max_deg"] + 1.
    assert record["acceptance"]["groups"]["forearm"]["median_deg"] == record["direction_errors"]["groups"]["forearm"]["median_deg"]
    assert "L" in record["arm_in_image_after_correction"] and len(record["code"]["digest"]) == 64


def test_the_observed_flag_changes_nothing_that_is_drawn(tmp_path, monkeypatch, rig):
    from dataclasses import replace
    state = noisy_hand_on_chest(rig, frames=24)
    flipped = replace(state, channels={name: replace(c, observed=c.valid.copy()) for name, c in state.channels.items()})
    a, _ = bundle_of(tmp_path, monkeypatch, state, "inferred")
    b, _ = bundle_of(tmp_path, monkeypatch, flipped, "observed")
    assert np.array_equal(a["body.vertices"], b["body.vertices"]) and np.array_equal(a["top.vertices"], b["top.vertices"])
    for name in ("signtranslator/avatar_render/motion_filter.py", "signtranslator/avatar_render/collision.py"):
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            assert not (isinstance(node, ast.Attribute) and node.attr == "observed"), name
    for node in ast.walk(ast.parse(_source("render_avatar.py"))):                     # the windowing of a state copies the flag; no branch reads it
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            assert not any(isinstance(part, ast.Attribute) and part.attr == "observed" for part in ast.walk(node.test))


def test_the_camera_is_orthographic():
    tree = ast.parse(_source("render_avatar.py"))
    kinds = [node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
             and node.targets[0].attr == "type" and isinstance(node.value, ast.Constant)]
    assert kinds == ["ORTHO"]


def test_the_packet_hands_the_renderer_its_gaps(tmp_path, monkeypatch, rig):
    from dataclasses import replace
    from signtranslator.pose.multichannel import MultichannelMotion
    import build_human_eval_packet as packet
    import render_avatar
    state = noisy_state(rig, frames=24)
    body = state.channels["body"]
    k = body.labels.index("LeftForearm")
    valid = body.valid.copy()
    valid[6:12, k] = False
    values = body.values.copy()
    values[~valid] = 0.
    state = replace(state, channels=dict(state.channels, body=replace(body, values=values, valid=valid, confidence=np.where(valid, body.confidence, 0.).astype(body.confidence.dtype))))
    seen = {}

    def capture(argv):
        saved = Path(argv[argv.index("--state") + 1])
        loaded = MultichannelMotion.load(saved, expected_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
        seen["gaps"] = int((~loaded.channels["body"].valid[:, k]).sum())
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir()
        (out / "avatar.mp4").write_bytes(b"x")
        (out / "verification.json").write_text(json.dumps(fake_record()))
        return 0
    monkeypatch.setattr(render_avatar, "main", capture)
    packet.render_avatar_video(state, tmp_path / "v.mp4")
    assert seen["gaps"] == 6


def mp4_boxes(payload: bytes, start: int = 0, end: int | None = None) -> list[tuple[str, int, int]]:
    """(kind, offset, size) of each box from ``start`` to ``end``, read by their size fields; the chain must end exactly at ``end``."""
    import struct
    end = len(payload) if end is None else end
    out, i = [], start
    while i < end:
        size, kind = struct.unpack(">I4s", payload[i:i + 8])
        if size == 1:
            size = struct.unpack(">Q", payload[i + 8:i + 16])[0]
        assert size >= 8, (kind, size)
        out.append((kind.decode(), i, size))
        i += size
    assert i == end, (i, end)
    return out


def boxes_except_samples(payload: bytes) -> list:
    """Every box but mdat and free, with its bytes, recursing into the containers, leaving out the sample-size table (stsz)."""
    out = []

    def walk(start, end, depth):
        for kind, offset, size in mp4_boxes(payload, start, end):
            if kind in ("moov", "trak", "mdia", "minf", "stbl"):
                out.append((depth, kind, size))
                walk(offset + 8, offset + size, depth + 1)
            elif kind not in ("mdat", "free", "stsz"):
                out.append((depth, kind, payload[offset:offset + size]))
            else:
                out.append((depth, kind, size if kind == "stsz" else None))
    walk(0, len(payload), 0)
    return out


def nal_offsets(sample: bytes) -> list[int]:
    out, i = [], 0
    while i < len(sample):
        out.append(i)
        i += 4 + int.from_bytes(sample[i:i + 4], "big")
    assert i == len(sample)
    return out


def synthetic_clips(frames=30, size=64):
    rng = np.random.default_rng(0)
    return {name: [np.clip(np.full((size, size, 3), 100, np.int16) + rng.integers(0, amount, (size, size, 3)), 0, 255).astype(np.uint8) for _ in range(frames)]
            for name, amount in (("still", 2), ("busy", 200))}


def test_a_padded_video_decodes_to_the_same_frames_and_every_file_has_one_size_that_a_zip_cannot_shrink(tmp_path):
    import av
    import zlib
    import build_human_eval_packet as packet
    from signtranslator.generation.stream_render import encode_timestamped_video, verify_timestamped_video
    clock = np.arange(12) / FPS
    decoded = {}
    for name, frames in synthetic_clips(12, 32).items():
        path = tmp_path / f"{name}.mp4"
        encode_timestamped_video(frames, clock, path, quality="constant")
        before = path.read_bytes()
        with av.open(str(path)) as container:
            decoded[name] = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        assert packet.pad_video(path, f"item-{name}") == packet.UNIFORM_VIDEO_BYTES == path.stat().st_size
        payload = path.read_bytes()
        assert payload.startswith(before)
        boxes = mp4_boxes(payload)                                                   # the chain is whole, to the last byte
        kind, offset, size = boxes[-1]
        assert kind == "free" and offset == len(before) and size == len(payload) - len(before)      # its size field says exactly what it holds
        assert [b[0] for b in boxes[:-1]] == [b[0] for b in mp4_boxes(before)]
        pad = payload[offset + 8:]
        assert pad == packet.pad_bytes(f"item-{name}", len(pad)) and len(zlib.compress(pad, 9)) > .999 * len(pad)      # nothing for a zip to squeeze
        verify_timestamped_video(path, clock)
        with av.open(str(path)) as container:
            again = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        assert all(np.array_equal(a, b) for a, b in zip(decoded[name], again)) and len(again) == 12
    assert (tmp_path / "still.mp4").stat().st_size == (tmp_path / "busy.mp4").stat().st_size
    assert packet.pad_bytes("a", 64) != packet.pad_bytes("b", 64)
    zipped = tmp_path / "re.zip"
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:      # what anyone forwarding the files might do
        z.write(tmp_path / "still.mp4", "still.mp4")
    assert zipped.stat().st_size > .99 * packet.UNIFORM_VIDEO_BYTES


def test_constant_rate_videos_of_one_length_match_in_every_box_but_their_sample_sizes(tmp_path):
    """A still clip and a busy one of the same frame count: equal stream bytes, equal bit rates, equal frame types and timing tables, so only
    the per-frame size table differs. The filler changes no picture."""
    import av
    from signtranslator.generation.stream_render import CONSTANT_RATE, constant_rate_budget, encode_timestamped_video, verify_timestamped_video
    assert CONSTANT_RATE == dict(bits_per_second=2_000_000, buffer_bits=2_000_000, keyint=48, preset="slow")
    clock = 5. + np.arange(30) / FPS
    out = {}
    for name, frames in synthetic_clips(30).items():
        path = tmp_path / f"{name}.mp4"
        record = encode_timestamped_video(frames, clock, path, quality="constant")["constant_rate"]
        verify_timestamped_video(path, clock)
        assert record["stream_bytes"] == constant_rate_budget(30, 24.) and record["filler_bytes"] > 0
        assert not (tmp_path / f"{name}.mp4.cbr").exists()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            samples = [bytes(p) for p in container.demux(stream) if p.size]
        with av.open(str(path)) as container:
            pictures = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        assert sum(map(len, samples)) == record["stream_bytes"]
        kinds = [sample[i + 4] & 31 for sample in samples for i in nal_offsets(sample)]
        assert 12 not in kinds and kinds.count(6) >= len(samples)                     # no 0xFF filler units, SEI units in every sample
        error = np.mean([(a.astype(float) - b.astype(float)) ** 2 for a, b in zip(frames, pictures)], axis=(1, 2, 3))
        out[name] = dict(size=path.stat().st_size, boxes=boxes_except_samples(path.read_bytes()), record=record, error=error)
    assert out["still"]["size"] == out["busy"]["size"]
    assert out["still"]["boxes"] == out["busy"]["boxes"]
    assert out["still"]["record"]["stream_bytes_before_filler"] != out["busy"]["record"]["stream_bytes_before_filler"]    # the content still differs
    assert out["still"]["error"].max() < 2.                                                                            # and the picture is kept
    import zlib
    squeezed = {name: len(zlib.compress((tmp_path / f"{name}.mp4").read_bytes(), 9)) for name in out}
    assert abs(squeezed["still"] - squeezed["busy"]) < .001 * out["still"]["size"] and min(squeezed.values()) > .995 * out["still"]["size"]


def test_every_render_goes_to_the_store_with_its_index(tmp_path):
    import render_avatar
    from signtranslator.data.text_motion_dataset import DirectoryStore
    folder = tmp_path / "render"
    (folder / "frames").mkdir(parents=True)
    (folder / "avatar.mp4").write_bytes(b"video")
    (folder / "frames" / "frame-000000.png").write_bytes(b"png")
    store = DirectoryStore(tmp_path / "store")
    made = render_avatar.archive_render(store, "renders/x/item", folder, dict(item_id="item"))
    index = json.loads(store.get("renders/x/item/INDEX.json"))
    assert made["files"] == 2 and index["item_id"] == "item" and index["files"]["avatar.mp4"]["sha256"] == hashlib.sha256(b"video").hexdigest()
    assert store.get("renders/x/item/frames/frame-000000.png") == b"png" and made["resumed"] is False


class CountingStore:
    """A directory store that counts its writes."""

    def __init__(self, root):
        from signtranslator.data.text_motion_dataset import DirectoryStore
        self.inner, self.puts = DirectoryStore(root), []

    def put(self, key, payload):
        self.puts.append(key)
        return self.inner.put(key, payload)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_render_is_written_to_the_store_once_and_a_resumed_upload_skips_what_matches(tmp_path):
    import render_avatar
    folder = tmp_path / "render"
    (folder / "frames").mkdir(parents=True)
    (folder / "avatar.mp4").write_bytes(b"video")
    (folder / "frames" / "frame-000000.png").write_bytes(b"png")
    store = CountingStore(tmp_path / "store")
    render_avatar.archive_render(store, "renders/x/item", folder, dict(item_id="item"))
    assert len(store.puts) == 3
    again = render_avatar.archive_render(store, "renders/x/item", folder, dict(item_id="item"))        # the same render, stored: nothing written
    assert again["resumed"] is True and len(store.puts) == 3
    with pytest.raises(SystemExit, match="written once"):                                                 # another render under the same item
        render_avatar.archive_render(store, "renders/x/item", folder, dict(item_id="other"))
    (folder / "avatar.mp4").write_bytes(b"other video")
    with pytest.raises(SystemExit, match="written once"):
        render_avatar.archive_render(store, "renders/x/item", folder, dict(item_id="item"))
    assert store.get("renders/x/item/avatar.mp4") == b"video" and len(store.puts) == 3
    # an upload cut off before its index: what matches is kept, the rest written, then the index
    (folder / "avatar.mp4").write_bytes(b"video")
    store.put("renders/x/cut/avatar.mp4", b"video")
    done = render_avatar.archive_render(store, "renders/x/cut", folder, dict(item_id="cut"))
    assert done["files_already_stored"] == 1 and store.puts[-2:] == ["renders/x/cut/frames/frame-000000.png", "renders/x/cut/INDEX.json"]
    store.put("renders/x/bad/avatar.mp4", b"not this render")
    with pytest.raises(SystemExit, match="written once"):
        render_avatar.archive_render(store, "renders/x/bad", folder, dict(item_id="bad"))
    assert not store.exists("renders/x/bad/INDEX.json")


def test_arm_invalid_share_reads_any_arm_joint(rig):
    from dataclasses import replace
    import build_human_eval_packet as packet
    state = noisy_state(rig, frames=20)
    assert packet.arm_invalid_share(state) == 0.
    body = state.channels["body"]
    valid = body.valid.copy()
    valid[:5, body.labels.index("RightUpperArm")] = False
    values = body.values.copy()
    values[~valid] = 0.
    thinned = replace(body, values=values, valid=valid, confidence=np.where(valid, body.confidence, 0.).astype(body.confidence.dtype))
    assert packet.arm_invalid_share(replace(state, channels=dict(state.channels, body=thinned))) == .25


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
