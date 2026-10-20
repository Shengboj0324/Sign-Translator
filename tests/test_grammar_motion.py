"""Grammar measured on motion: every detector against synthetic ground truth."""
import json
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from signtranslator.grammar_motion import agreement, frames, gaze_locus, nmm, pointing, prosody, role_shift, text
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from tests.test_generation_signing_space import LABELS, signer_state

MP_NAMES = ("_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
            "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
            "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
            "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
            "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft",
            "noseSneerRight")


def channel(values, labels, units="unitless", frame="f"):
    valid = np.ones(values.shape[:2], bool)
    return MotionChannel(values, valid, np.zeros_like(valid), valid.astype(float), tuple(labels), frame, units, "src",
                         "a" * 64, "c", inference_method="estimate")


def six(rotvecs):
    from signtranslator.pose.rotations import matrix_to_rotation_6d
    return matrix_to_rotation_6d(torch.from_numpy(Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix())).numpy() \
        .reshape(rotvecs.shape[:-1] + (6,))


# ---------------------------------------------------------------- body frame
def test_body_frame_is_invariant_to_whole_body_rotation():
    state = signer_state(seed=3, frames=6, amplitude=.4)
    joints, base, valid = frames.body_frame_positions(state)
    body = state.channels["body"]
    turned = body.values.copy()
    turn = Rotation.from_euler("y", 35, degrees=True)
    k = LABELS["body"].index("Pelvis")
    from signtranslator.generation.motion_metrics import _matrices
    from signtranslator.pose.rotations import matrix_to_rotation_6d
    m = turn.as_matrix()[None] @ _matrices(body.values[:, k])
    turned[:, k] = matrix_to_rotation_6d(torch.from_numpy(m)).numpy()
    rotated = replace(state, channels={**state.channels, "body": replace(body, values=turned)})
    _, again, _ = frames.body_frame_positions(rotated)
    np.testing.assert_allclose(again, base, atol=1e-9)          # signing space moves with the body
    assert valid.all()
    np.testing.assert_allclose(frames.azimuth_elevation(np.array([[1., 0, 1.], [0, 1., 1.]])),
                               np.radians([[45, 0], [0, 45]]), atol=1e-12)


# ---------------------------------------------------------------- pointing and loci
def test_least_squares_locus_is_exact_and_refuses_parallel_rays():
    target = np.array([.4, .2, 1.1])
    origins = np.array([[0, 0, 0], [.3, -.1, .2], [-.2, .3, .1], [.1, .1, -.3]], float)
    x, cond = pointing.least_squares_locus(origins, target - origins)
    np.testing.assert_allclose(x, target, atol=1e-8)
    x, _ = pointing.least_squares_locus(origins, np.tile([0, 0, 1.], (4, 1)))
    assert x is None


def test_dp_vmf_means_infers_the_number_of_loci():
    rng = np.random.default_rng(0)
    centres = np.radians([-40, 0, 35])
    dirs = []
    for az in centres:
        for _ in range(12):
            a = az + rng.normal(scale=np.radians(3)); e = rng.normal(scale=np.radians(3))
            dirs.append([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    labels, cent, objective = pointing.dp_vmf_means(np.array(dirs))
    assert len(cent) == 3 and objective > 0
    found = sorted(np.degrees(np.arctan2(cent[:, 0], cent[:, 2])))
    np.testing.assert_allclose(found, [-40, 0, 35], atol=3.)
    assert len(set(labels[:12])) == 1 and len(set(labels)) == 3
    labels, cent, _ = pointing.dp_vmf_means(np.array(dirs), radius_deg=60.)   # coarser radius merges loci
    assert len(cent) < 3


def _hand(T, pointing_now, side="Right"):
    """Joint dict for one hand: index straight forward when pointing, all fingers curled otherwise."""
    joints = [f"{side}Hand"] + [f"{side}Hand{f}{k}" for f in ("Thumb", "Index", "Middle", "Ring", "Pinky") for k in (1, 2, 3, 4)]
    pos = np.zeros((T, len(joints), 3))
    for t in range(T):
        pos[t, 0] = [0., 0., .9]
        for f, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
            base = np.array([.05 * (f - 2), 0., .9 + .25])
            chain = [base]
            for k in range(3):
                if finger == "Index" and pointing_now[t]:
                    step = np.array([0., 0., .04])
                else:                                          # flexion of 75° per joint: the finger rolls under
                    angle = np.radians(75) * (k + 1)
                    step = .04 * np.array([0., -np.sin(angle), np.cos(angle)])
                chain.append(chain[-1] + step)
            for k in range(4):
                pos[t, joints.index(f"{side}Hand{finger}{k + 1}")] = chain[k]
    return joints, pos


def test_pointing_frames_need_an_extended_index_curled_fingers_and_a_hold():
    T = 30
    truth = np.zeros(T, bool); truth[8:16] = True; truth[22:24] = True     # second run too short to count
    joints, pos = _hand(T, truth)
    valid = np.ones(pos.shape[:2], bool)
    found = pointing.pointing_frames(joints, pos, valid, np.arange(T) / 30., "Right")
    assert found[8:16].all() and not found[22:24].any() and not found[:8].any()
    events = pointing.pointing_events(joints, pos, valid, np.arange(T) / 30., "Right")
    assert [(e.start, e.stop) for e in events] == [(8, 16)]
    np.testing.assert_allclose(events[0].direction, [0, 0, 1], atol=1e-9)


# ---------------------------------------------------------------- non-manual markers
def face_state(T=90, fps=30., raise_span=(30, 54), furrow_span=None, noise=.02, seed=0):
    rng = np.random.default_rng(seed)
    v = np.clip(.1 + rng.normal(scale=noise, size=(T, len(MP_NAMES))), 0, 1)
    if raise_span:
        a, b = raise_span
        for n in ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"):
            v[a:b, MP_NAMES.index(n)] += .5
    if furrow_span:
        a, b = furrow_span
        for n in ("browDownLeft", "browDownRight"):
            v[a:b, MP_NAMES.index(n)] += .5
    return channel(np.clip(v, 0, 1)[..., None], MP_NAMES)


def test_brow_raise_and_furrow_spans_are_found_where_they_are():
    tracks = nmm.nmm_tracks(face_state(raise_span=(30, 54)), 30.)
    (a, b), = tracks.spans["brow_raise"]
    assert abs(a - 30) <= 3 and abs(b - 54) <= 3 and not tracks.spans["brow_furrow"]
    tracks = nmm.nmm_tracks(face_state(raise_span=None, furrow_span=(10, 40)), 30.)
    assert tracks.spans["brow_furrow"] and not tracks.spans["brow_raise"]


def test_short_flicker_is_not_a_marker():
    tracks = nmm.nmm_tracks(face_state(raise_span=(40, 42)), 30.)      # 67 ms
    assert not tracks.spans["brow_raise"]


def head_channel(yaw_deg, pitch_deg=None):
    pitch = np.zeros_like(yaw_deg) if pitch_deg is None else pitch_deg
    rv = np.stack([Rotation.from_euler("yx", [y, -p], degrees=True).as_rotvec() for y, p in zip(yaw_deg, pitch)])
    return channel(six(rv)[:, None], ("Head",))


def test_headshake_is_a_fast_oscillation_and_a_turn_is_not():
    t = np.arange(60) / 30.
    shake = np.where((t > .5) & (t < 1.5), 10 * np.sin(2 * np.pi * 2.5 * (t - .5)), 0.)
    spans = nmm.head_gestures(head_channel(shake), 30.)["headshake"]
    assert len(spans) == 1 and 12 <= spans[0][0] <= 22 and 38 <= spans[0][1] <= 48
    turn = np.clip((t - .5) * 30, 0, 25)                               # one 25° turn
    assert not nmm.head_gestures(head_channel(turn), 30.)["headshake"]
    nod = nmm.head_gestures(head_channel(np.zeros(60), shake), 30.)
    assert nod["nod"] and not nod["headshake"]


# ---------------------------------------------------------------- role shift
def brute_force_partition(y, penalty, min_size):
    n = len(y)
    best = np.full(n + 1, np.inf); best[0] = -penalty; back = np.zeros(n + 1, int)
    def cost(a, b):
        s = y[a:b]; return float(((s - s.mean(0)) ** 2).sum())
    for t in range(min_size, n + 1):
        for s in range(0, t - min_size + 1):
            if s and s < min_size:
                continue
            v = best[s] + cost(s, t) + penalty
            if v < best[t]:
                best[t], back[t] = v, s
    pts, t = [], n
    while t > 0:
        if back[t]: pts.append(int(back[t]))
        t = back[t]
    return sorted(pts), best[n]


def test_pelt_is_exact():
    rng = np.random.default_rng(4)
    for trial in range(5):
        y = np.concatenate([rng.normal(m, 1, size=(rng.integers(6, 15), 2)) for m in rng.normal(0, 3, size=4)])
        cps = role_shift.pelt(y, penalty=8., min_size=4)
        ref, ref_cost = brute_force_partition(y, 8., 4)
        bounds = [0] + cps + [len(y)]
        total = sum(float(((y[a:b] - y[a:b].mean(0)) ** 2).sum()) + 8. for a, b in zip(bounds[:-1], bounds[1:])) - 8.
        assert abs(total - ref_cost) < 1e-8, (cps, ref)


def body_state_with_yaw(yaw_deg, gaze=None, fps=30.):
    T = len(yaw_deg)
    base = signer_state(seed=0, frames=T, amplitude=0.)
    body = base.channels["body"]
    values = body.values.copy()
    k = LABELS["body"].index("Pelvis")
    values[:, k] = six(np.stack([Rotation.from_euler("y", y, degrees=True).as_rotvec() for y in yaw_deg]))
    channels = {**base.channels, "body": replace(body, values=values), "head_rotation": head_channel(np.asarray(yaw_deg))}
    return replace(base, channels=channels, timestamps=np.arange(T) / fps)


def test_role_shift_is_found_as_a_departure_from_the_narrator_baseline():
    rng = np.random.default_rng(1)
    yaw = rng.normal(0, 1.5, 120)
    yaw[40:80] += 22.                                         # 1.3 s turned toward the signer's left
    shifts, report = role_shift.role_shifts(body_state_with_yaw(yaw), 30.)
    assert len(shifts) == 1 and abs(shifts[0].start - 40) <= 2 and abs(shifts[0].stop - 80) <= 2
    assert shifts[0].direction == "left"
    calm, _ = role_shift.role_shifts(body_state_with_yaw(rng.normal(0, 1.5, 120)), 30.)
    assert not calm


# ---------------------------------------------------------------- gaze and agreement
def test_gaze_targets_and_the_agreement_window():
    loci = np.array([[np.sin(np.radians(40)), 0, np.cos(np.radians(40))], [-np.sin(np.radians(40)), 0, np.cos(np.radians(40))]])
    gaze = np.tile([0., 0., 1.], (30, 1))
    gaze[10:20] = loci[1]
    targets = gaze_locus.gaze_targets(gaze, None, loci, np.ones(30, bool))
    assert (targets[10:20] == 1).all() and (targets[:10] == -1).all()
    assert gaze_locus.verb_gaze_marked(targets, 1, onset=14, stop=24, fps=30.)       # starts 4 frames (133 ms) early
    assert not gaze_locus.verb_gaze_marked(targets, 0, onset=14, stop=24, fps=30.)
    g = torch.tensor(gaze, requires_grad=True)
    loss = gaze_locus.gaze_locus_loss(g, torch.tensor(np.tile(loci[1], (30, 1))), torch.ones(30, dtype=torch.float64))
    loss.backward()
    assert float(loss) > 0 and g.grad.abs().sum() > 0


def test_agreement_errors_and_loss():
    src, goal = np.array([-.5, 0, 1.]), np.array([.5, 0, 1.])
    path = np.linspace(src * .6, goal * .6, 10)
    e = agreement.agreement_errors(path, goal - path[-1], src, goal, goal)
    assert e["direction_deg"] < 1e-6 and e["endpoint_deg"] < 1e-6 and e["facing_cos"] > .99
    backwards = agreement.agreement_errors(path[::-1], None, src, goal)
    assert abs(backwards["direction_deg"] - 180) < 1e-6
    assert agreement.verb_class_from_path(path, np.array([1., 0, 0])) == "regular"
    assert agreement.verb_class_from_path(path, np.array([-1., 0, 0])) == "backwards"
    h = torch.tensor(path[::-1].copy()[None], requires_grad=True)
    loss = agreement.agreement_loss(h, torch.tensor(src[None]), torch.tensor(goal[None]))
    loss.backward()
    assert float(loss) > 1 and h.grad.abs().sum() > 0
    good = agreement.agreement_loss(torch.tensor(path[None]), torch.tensor(src[None]), torch.tensor(goal[None]))
    assert float(good) < 1e-6


# ---------------------------------------------------------------- prosody
def test_movement_units_boundaries_and_final_lengthening():
    fps = 30.
    speed = np.concatenate([np.sin(np.linspace(0, np.pi, 9)) + .05 for _ in range(4)] + [np.sin(np.linspace(0, np.pi, 20)) + .05])
    valid = np.ones(len(speed), bool)
    units = prosody.movement_units(speed, valid, fps)
    assert len(units) >= 4
    assert prosody.final_lengthening(units) > 1.5
    blink = np.full(len(speed), .1) + np.random.default_rng(0).normal(scale=.01, size=len(speed))
    blink[len(speed) - 6:len(speed) - 3] = .45                            # a closure well above the open-eye level
    onsets = prosody.blink_onsets(blink, valid, fps)
    ev = prosody.boundary_evidence(units, onsets, np.zeros(len(speed)), np.zeros(len(speed)))
    assert ev[-1]["lengthened"] and ev[-1]["blink"] and ev[-1]["boundary"]
    assert not any(e["boundary"] for e in ev[:-1])
    heights = np.full((len(speed), 2), -1.5); heights[5:30, 0] = -.4      # the right hand signs in frames 5-29
    assert prosody.active_range(heights, valid) == (5, 30)
    assert prosody.movement_units(speed, valid, fps, span=(5, 30))[0][0] == 5


# ---------------------------------------------------------------- English features
@pytest.mark.parametrize("sentence,expected", [
    ("So what do you do next?", "wh_question"), ("Do you have the scissors?", "yes_no_question"),
    ("It's easy, right?", "other_question"), ("You don't want to overdo it.", "negation"),
    ("If it rains, cover the plants.", "conditional"), ('He said, "stop right there."', "reported_speech"),
    ("Never use bleach here.", "negation")])
def test_sentence_features(sentence, expected):
    f = text.sentence_features(sentence)
    assert f[expected]
    others = {"wh_question", "yes_no_question", "other_question"} - {expected}
    assert not any(f[k] for k in others)


def test_feature_vector_layout():
    v = text.feature_vector("What did they put there, and why not?")
    assert v.shape == (8,) and v[0] == 1 and v[3] == 1 and 0 < v[6] <= 1 and 0 < v[7] <= 1
    assert text.feature_vector("Cut the paper.").sum() == 0


# ---------------------------------------------------------------- whole record
def test_grammar_record_is_complete_and_json_serialisable():
    from signtranslator.grammar_motion.tracks import grammar_record
    state = signer_state(seed=2, frames=90, amplitude=.3)
    rng = np.random.default_rng(0)
    face = face_state(T=90, raise_span=(20, 50))
    yaw = np.zeros(90); yaw[50:75] = 10 * np.sin(2 * np.pi * 2.5 * np.arange(25) / 30)
    blink = channel(np.zeros((90, 2, 1)), ("eyeBlinkLeft", "eyeBlinkRight"))
    gaze = np.tile([0., 0., 1.], (90, 1))
    g = channel(gaze[:, None].copy(), ("LeftEye",)), channel(gaze[:, None].copy(), ("RightEye",))
    state = replace(state, timestamps=np.arange(90) / 30., channels={**state.channels, "face": face,
                    "head_rotation": head_channel(yaw), "blink": blink, "left_gaze": g[0], "right_gaze": g[1]})
    record = grammar_record(state)
    json.dumps(record, default=float)
    assert record["nmm"]["brow_raise"]["spans"] and record["head"]["headshake"]
    assert {"pointing", "role_shift", "prosody"} <= set(record)


# ---------------------------------------------------------------- generator conditioning
def test_grammar_features_condition_both_generators_and_start_as_no_ops():
    from signtranslator.generation.canonical_state import channel_bindings_from_state
    from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, encode_texts
    from signtranslator.grammar_motion.text import token_features
    tokens, mask = encode_texts(["What is it?", "Cut the paper."])
    f = token_features(tokens)
    assert f.shape == (2, 8) and f[0, 0] == 1 and f[1].sum() == 0
    empty = tokens.clone(); empty[:, 1:] = 256; empty[:, 1] = 258
    assert token_features(empty).abs().sum() == 0                     # the guidance-free text has no features
    state = signer_state(0, frames=8)
    base = dict(bindings=channel_bindings_from_state(state), trained_channels=("body",), d_model=32, heads=4,
                text_layers=1, motion_layers=1, feedforward=64, dropout=0.)
    torch.manual_seed(0); plain = TextToMotion(TextToMotionConfig(**base))
    torch.manual_seed(0); graded = TextToMotion(TextToMotionConfig(**base, grammar_features=True))
    graded.load_state_dict({**plain.state_dict(), **{k: v for k, v in graded.state_dict().items() if k.startswith("grammar")}})
    h1, d1 = plain.encode(tokens, mask); h2, d2 = graded.encode(tokens, mask)
    torch.testing.assert_close(h1, h2); torch.testing.assert_close(d1, d2)   # zero-initialised: identical at start
    with torch.no_grad():
        graded.grammar.weight.normal_()
    h3, _ = graded.encode(tokens, mask)
    assert not torch.allclose(h1[0], h3[0]) and torch.allclose(h1[1], h3[1], atol=1e-6)   # only the question moves


# ---------------------------------------------------------------- evaluation
def test_auc_and_wasserstein_are_exact():
    from signtranslator.grammar_motion.evaluation import auc, wasserstein1
    assert auc([3, 4], [1, 2]) == 1. and auc([1, 2], [3, 4]) == 0. and auc([1, 2], [1, 2]) == .5
    assert np.isnan(auc([], [1.]))
    assert abs(wasserstein1([0, 0, 1, 1], [1, 1, 2, 2]) - 1.) < 1e-12 and wasserstein1([1, 2], [1, 2]) == 0.


def test_grammar_comparison_recovers_a_marker_the_references_carry():
    from signtranslator.grammar_motion.evaluation import grammar_comparison
    texts = ["What is it?"] * 6 + ["Cut the paper."] * 6
    def with_face(furrow):
        s = signer_state(seed=1, frames=60, amplitude=.2)
        return replace(s, timestamps=np.arange(60) / 30., channels={**s.channels, "face": face_state(
            T=60, raise_span=None, furrow_span=(10, 50) if furrow else None, seed=len(texts))})
    references = [with_face(i < 6) for i in range(12)]
    flat = [with_face(False) for _ in range(12)]
    good = grammar_comparison(references, references, texts)
    bad = grammar_comparison(flat, references, texts)
    wh = lambda r: next(t for t in r["tests"] if t["feature"] == "wh_question")
    assert wh(good)["generated_auc"] == wh(good)["reference_auc"] == 1.
    assert wh(bad)["generated_auc"] == .5 and bad["distributions"]["brow_furrow_fraction"]["mean_ratio"] < .5


def test_smoothing_keeps_length_for_clips_shorter_than_the_window():
    out = nmm.smooth(np.arange(6.), np.ones(6, bool), 30.)
    assert out.shape == (6,) and np.isfinite(out).all()
    x = np.r_[np.zeros(20), np.ones(20)]
    y = nmm.smooth(x, np.ones(40, bool), 30.)
    assert y[:15].max() == 0 and y[25:].min() == 1 and np.all(np.diff(y) >= 0)


def test_role_shift_ignores_off_screen_legs():
    rng = np.random.default_rng(2)
    yaw = rng.normal(0, 1.5, 120); yaw[30:75] -= 20.
    state = body_state_with_yaw(yaw)
    body = state.channels["body"]
    valid = body.valid.copy()
    for leg in ("LeftThigh", "LeftShin", "RightThigh", "RightShin"):
        valid[:, LABELS["body"].index(leg)] = False
    state = replace(state, channels={**state.channels, "body": replace(
        body, valid=valid, values=np.where(valid[..., None], body.values, 0.), confidence=valid.astype(float))})
    shifts, _ = role_shift.role_shifts(state, 30.)
    assert len(shifts) == 1 and shifts[0].direction == "right"


# ---------------------------------------------------------------- locus conditioning
def test_locus_slots_planner_and_state_measurement():
    from signtranslator.grammar_motion import loci_conditioning as lc
    s = lc.slots([[0, 0, 2.], [1., 0, 0]])
    assert s.shape == (lc.SLOT_WIDTH,) and s[3] == 1 and s[7] == 1 and s[11] == 0
    np.testing.assert_allclose(s[:3], [0, 0, 1])
    assert lc.plan_loci("Cut the paper.").sum() == 0
    two = lc.plan_loci("Give it to him and then ask her.").reshape(lc.MAX_LOCI, 4)
    assert two[:, 3].tolist() == [1, 1, 0, 0] and two[0, 0] < 0 < two[1, 0]       # right side first, then contrast
    one = lc.plan_loci("Put it there.").reshape(lc.MAX_LOCI, 4)
    assert one[:, 3].tolist() == [1, 0, 0, 0]
    assert lc.state_loci(signer_state(0, frames=12)).shape == (lc.SLOT_WIDTH,)


def test_locus_conditioning_trains_and_generates_with_planned_loci():
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.data.text_motion_dataset import collate_text_motion
    from signtranslator.generation.canonical_state import channel_bindings_from_state
    from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, generate, training_loss
    states = [signer_state(s, frames=12) for s in range(2)]
    tokens, mask, motion, texts, loci = collate_text_motion([("Ask him.", states[0]), ("Put it there.", states[1])], loci=True)
    assert loci.shape == (2, 16)
    config = TextToMotionConfig(bindings=channel_bindings_from_state(states[0]), trained_channels=("body",), d_model=32,
                                heads=4, text_layers=1, motion_layers=1, feedforward=64, dropout=0., locus_conditioning=True)
    model = TextToMotion(config)
    loss, terms, _ = training_loss(model, tokens, mask, motion, loci)
    loss.backward()
    assert model.loci.weight.grad is not None
    with torch.no_grad():
        model.loci.weight.normal_()
        for head in model.heads_out.values():                  # output heads start at zero (the neutral pose)
            head.weight.normal_(std=.1)
    a = generate(model, ["Ask him."], fps=24., durations=[.5])[0]
    b = generate(model, ["Ask him."], fps=24., durations=[.5], loci=np.zeros((1, 16)))[0]
    assert not np.allclose(a.channels["body"].values, b.channels["body"].values)   # planned loci change the motion
