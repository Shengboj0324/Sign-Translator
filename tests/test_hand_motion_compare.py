"""Hand motion compare: the measures on synthetic states, including the ways they could mislead. Offline: nothing is read from R2."""
import json

import numpy as np
import pytest

import scripts.hand_motion_compare as hm
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion

SHA = "0" * 64
BODY = ("Pelvis", "Chest", "LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "LeftHand", "RightHand")
CAPTURE_BODY = ("Hips", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm", "LeftHand", "RightHand")
SWING = "sapiens-two-view-swing-ik-y-up-v1"
GLTF = "gltf-node-local-rotation-including-rest-v1;mapping=test"
IDENTITY = np.array([1., 0., 0., 0., 1., 0.])


def rot_x(angle):
    """6D columns of a rotation about +x, for any shape of angle."""
    a = np.asarray(angle, dtype=float)
    return np.stack((np.ones_like(a), 0 * a, 0 * a, 0 * a, np.cos(a), np.sin(a)), -1)


def rot_y(angle):
    a = np.asarray(angle, dtype=float)
    return np.stack((np.cos(a), 0 * a, -np.sin(a), 0 * a, np.ones_like(a), 0 * a), -1)


def channel(name, values, valid=None, observed=None, labels=None, convention=SWING):
    values = np.array(values, dtype=np.float32)
    shape = values.shape[:2]
    valid = np.ones(shape, bool) if valid is None else np.array(valid, bool)
    observed = valid.copy() if observed is None else np.array(observed, bool) & valid
    values = np.where(valid[..., None], values, 0.).astype(np.float32)
    return MotionChannel(values, valid, observed, valid.astype(np.float32), labels or tuple(f"{name}{k}" for k in range(shape[1])), "frame", CHANNELS[name][2], "test", SHA, convention, "test" if (valid & ~observed).any() else None)


def hand_labels(side):
    return tuple(f"{side}Hand{j}" for j in hm.JOINTS)


def build(right, left=None, *, fps=24., right_valid=None, left_valid=None, right_observed=None, left_observed=None, convention=SWING, arm_right=None, arm_left=None, body_labels=BODY):
    """A state from hand values (T, 15, 6); arms still unless given (T,) angles."""
    T = len(right)
    t = np.arange(T, dtype=np.float64) / fps
    left = np.tile(IDENTITY, (T, 15, 1)) if left is None else left
    body = np.tile(IDENTITY, (T, len(body_labels), 1))
    upper = "RightUpperArm" if "RightUpperArm" in body_labels else "RightArm"
    upper_left = "LeftUpperArm" if "LeftUpperArm" in body_labels else "LeftArm"
    if arm_right is not None:
        body[:, body_labels.index(upper)] = rot_x(arm_right)
    if arm_left is not None:
        body[:, body_labels.index(upper_left)] = rot_x(arm_left)
    channels = {}
    for name, (kind, width, _) in CHANNELS.items():
        if name == "right_hand":
            channels[name] = channel(name, right, right_valid, right_observed, hand_labels("Right"), convention)
        elif name == "left_hand":
            channels[name] = channel(name, left, left_valid, left_observed, hand_labels("Left"), convention)
        elif name == "body":
            channels[name] = channel(name, body, labels=body_labels, convention=convention)
        else:
            channels[name] = channel(name, np.zeros((T, 1, width)), np.zeros((T, 1), bool))
    return MultichannelMotion(t, "clock", "sample", channels)


def still(T=48):
    return np.tile(IDENTITY, (T, 15, 1))


def curling(T=48, hz=1., amplitude=.5):
    """Every joint flexing about x, smoothly."""
    t = np.arange(T) / 24.
    return rot_x(amplitude * (1 + np.sin(2 * np.pi * hz * t)))[:, None].repeat(15, axis=1)


# ---- all-invalid frames ----------------------------------------------------------------------------------------------

def test_all_invalid_frames_give_no_row_never_a_zero():
    state = build(still(), right_valid=np.zeros((48, 15), bool), left_valid=np.zeros((48, 15), bool))
    row = hm.clip_row(state, True)
    assert row["hands"] == dict(Right=None, Left=None)
    assert row["dominant"] is None
    assert hm.hand_of(row, "dominant") is None
    means = hm.word_means([dict(entry="a", **row)], lambda r: (hm.hand_of(r, "dominant") or {}).get("speed_raw"))
    assert means == {}
    assert hm.boot_mean([])["mean"] is None
    assert hm.boot_ratio([], [])["ratio"] is None


def test_too_few_valid_frames_give_no_row():
    valid = np.zeros((48, 15), bool)
    valid[:hm.MIN_FRAMES - 1] = True
    assert hm.clip_row(build(curling(), right_valid=valid), True)["hands"]["Right"] is None
    valid[:hm.MIN_FRAMES] = True
    assert hm.clip_row(build(curling(), right_valid=valid), True)["hands"]["Right"]["frames"] == hm.MIN_FRAMES


def test_valid_but_unobserved_frames_are_not_read_as_real():
    observed = np.zeros((48, 15), bool)
    state = build(curling(), right_observed=observed)
    assert hm.clip_row(state, True)["hands"]["Right"] is None
    assert hm.clip_row(state, False)["hands"]["Right"] is not None


def test_a_gap_is_no_motion_and_nothing_crosses_it():
    """Two still poses far apart with invalid frames between: any step or filter across the gap would read motion."""
    values = still()
    values[24:] = rot_x(1.)
    valid = np.ones((48, 15), bool)
    valid[20:28] = False
    row = hm.clip_row(build(values, right_valid=valid), True)["hands"]["Right"]
    assert row["speed_raw"] == pytest.approx(0., abs=1e-4)
    assert row["speed_6hz"] == pytest.approx(0., abs=1e-4)
    assert row["active_0_5"] == 0.


def test_a_joint_never_valid_leaves_no_handshape_so_no_row():
    values = curling()
    valid = np.ones((48, 15), bool)
    valid[:, 3] = False
    row = hm.clip_row(build(values, right_valid=valid), True)
    assert row["hands"]["Right"] is None      # no frame holds all 15 joints: no handshape, so no row


def test_a_source_read_on_valid_stops_if_it_holds_an_observed_frame():
    none = dict(entry="a", validity=dict(Right=dict(valid=1., observed=0., confidence=1.), Left=None))
    assert hm.check_observed(dict(x=[none]))["x"]["read_on"] == "valid"
    some = dict(entry="a", validity=dict(Right=dict(valid=1., observed=.5, confidence=1.), Left=None))
    with pytest.raises(SystemExit):
        hm.check_observed(dict(x=[none, some]))


# ---- speed, jitter ---------------------------------------------------------------------------------------------------

def test_speed_is_in_radians_per_second():
    omega = 1.2
    values = rot_x(omega * np.arange(48) / 24.)[:, None].repeat(15, axis=1)
    row = hm.clip_row(build(values), True)["hands"]["Right"]
    assert row["speed_raw"] == pytest.approx(omega, rel=1e-3)
    assert row["p90_raw"] == pytest.approx(omega, rel=1e-3)
    assert row["autocorr_raw"] == pytest.approx(1., abs=1e-3)
    assert row["active_0_5"] == 1. and row["active_1_0"] == 1.
    assert row["jitter_share"] == pytest.approx(0., abs=.02)


def test_jitter_only_motion_is_not_read_as_articulation():
    """White noise about a still pose has high raw speed, but it is not motion: low-pass drops it and the step's autocorrelation sits near -0.5."""
    rng = np.random.default_rng(0)
    T = 240
    noise = .05 * rng.standard_normal((T, 15))
    jitter = hm.clip_row(build(rot_x(noise)), True)["hands"]["Right"]
    smooth = hm.clip_row(build(curling(T)), True)["hands"]["Right"]
    assert jitter["speed_raw"] > 1.
    assert jitter["autocorr_raw"] < -.3
    assert jitter["jitter_share"] > .4
    assert jitter["speed_3hz"] < .5 * jitter["speed_raw"]
    assert smooth["autocorr_raw"] > .9
    assert smooth["jitter_share"] < .05
    assert smooth["speed_6hz"] == pytest.approx(smooth["speed_raw"], rel=.05)


def test_a_higher_rate_source_reads_the_same_speed_at_24_fps():
    t60 = np.arange(120) / 60.
    a = rot_x(.5 * (1 + np.sin(2 * np.pi * t60)))[:, None].repeat(15, axis=1)
    row60 = hm.clip_row(build(a, fps=60.), True)["hands"]["Right"]
    row24 = hm.clip_row(build(curling(48)), True)["hands"]["Right"]
    assert row60["speed_raw"] == pytest.approx(row24["speed_raw"], rel=.05)


def test_bend_ignores_twist_and_geodesic_does_not():
    """3D-LEX's solved twist must not flatter it against swing-IK states on the bend reading."""
    values = rot_y(np.arange(48) / 24.)[:, None].repeat(15, axis=1)
    values[:, :3] = IDENTITY          # the thumb's rest is not y: leave it out
    row = hm.clip_row(build(values, convention=GLTF, body_labels=CAPTURE_BODY), False)["hands"]["Right"]
    assert row["speed_raw"] > .5
    assert row["bend_raw"] == pytest.approx(0., abs=.01)


# ---- handshape features ----------------------------------------------------------------------------------------------

def test_features_are_the_nominal_skeleton_of_canonical_states():
    """The FK here against ``canonical_skeleton`` itself, for a posed canonical hand and a still body."""
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    rng = np.random.default_rng(1)
    values = rot_x(rng.uniform(0, 1.2, (5, 15)))
    full = np.tile(IDENTITY, (5, 15, 1)).copy()
    full[:] = values
    labels = ("Pelvis", "Chest", "LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "LeftHand", "RightHand")
    state = build(full, body_labels=labels)
    joints, positions, _ = skeleton_positions(state)
    p = {j: positions[:, k] for k, j in enumerate(joints)}
    R = hm.matrices(hm.hand_channel(state, "Right", False)[0], np.ones((5, 15), bool))
    mine = hm.hand_points(R, "Right", False)
    wrist = p["RightHand"]
    for name in ("Thumb4", "Index4", "Index1", "Pinky1"):
        assert np.allclose(mine[name], p[f"RightHand{name}"] - wrist, atol=1e-6), name


def test_features_do_not_change_under_mirroring():
    from signtranslator.pose.mirror import mirror_state
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1., (24, 15))
    b = rng.uniform(-.3, .3, (24, 15))
    R = hm.matrices(rot_x(a), np.ones((24, 15), bool)) @ hm.matrices(rot_y(b), np.ones((24, 15), bool))
    values = R[..., :2].transpose(0, 1, 3, 2).reshape(24, 15, 6)
    state = build(values)
    mirrored = mirror_state(state, sample_id="m")
    one, other = hm.hand_row(*hm.hand_channel(state, "Right", True), "Right", False), hm.hand_row(*hm.hand_channel(mirrored, "Left", True), "Left", False)
    assert np.allclose(one["core"], other["core"], atol=2e-3)


def test_capture_left_hand_reads_as_its_right_hand_mirror():
    """3D-LEX's left hand x is mirrored against its right (rest offsets), so its features need the sign handled."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1., (24, 15))
    R = hm.matrices(rot_x(a), np.ones((24, 15), bool))
    S = np.diag([-1., 1., 1.])
    left = S @ R @ S
    v = lambda M: M[..., :2].transpose(0, 1, 3, 2).reshape(24, 15, 6)     # noqa: E731
    state = build(v(R), v(left), convention=GLTF, body_labels=CAPTURE_BODY)
    one = hm.hand_row(*hm.hand_channel(state, "Right", True), "Right", True)
    other = hm.hand_row(*hm.hand_channel(state, "Left", True), "Left", True)
    assert np.allclose(one["core"], other["core"], atol=2e-3)


def test_features_separate_a_fist_from_an_open_hand():
    open_hand = hm.hand_row(*hm.hand_channel(build(still()), "Right", True), "Right", False)
    fist = hm.hand_row(*hm.hand_channel(build(rot_x(np.full((48, 15), 1.2))), "Right", True), "Right", False)
    names = list(hm.FEATURES)
    assert open_hand["core"][names.index("Index_mcp")] == pytest.approx(0., abs=1e-3)
    assert fist["core"][names.index("Index_mcp")] == pytest.approx(1.2, abs=1e-3)
    assert fist["core"][names.index("Index_pip_dip")] == pytest.approx(2.4, abs=1e-3)
    assert len(open_hand["core"]) == len(names)


def test_core_is_the_central_part_of_valid_frames_only():
    features = np.zeros((40, 1))
    features[15:25] = 1.
    whole = np.ones(40, bool)
    assert hm.core_features(features, whole, .25)[0] == 1.
    whole[:20] = False                      # the centre of what is valid moves
    assert hm.core_features(features, whole, .25)[0] == 0.
    assert hm.core_features(features, np.zeros(40, bool), .5) is None


def test_dominant_is_by_arm_never_by_fingers():
    arm = np.arange(48) / 24.
    state = build(curling(amplitude=.8), still(), arm_left=arm)     # the right fingers move, the left arm moves
    assert hm.clip_row(state, False)["dominant"] == "Left"
    state = build(still(), curling(amplitude=.8), arm_right=arm)
    assert hm.clip_row(state, False)["dominant"] == "Right"
    assert hm.clip_row(build(still(), right_valid=np.zeros((48, 15), bool)), True)["dominant"] == "Left"


def test_dominant_reads_capture_arm_labels():
    arm = np.arange(48) / 24.
    assert hm.clip_row(build(still(), arm_left=arm, convention=GLTF, body_labels=CAPTURE_BODY), False)["dominant"] == "Left"


# ---- a collapsed generator -------------------------------------------------------------------------------------------

def varied(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, len(hm.FEATURES)))


def test_a_collapsed_generator_shows_in_every_variety_read():
    rng = np.random.default_rng(0)
    real = varied()
    collapsed = np.tile(real[:1], (200, 1)) + .001 * rng.normal(size=real.shape)
    centres = hm.kmeans(real)
    assert hm.effective_clusters(real, centres)["exp_entropy"] > 15
    row = hm.effective_clusters(collapsed, centres)
    assert row["exp_entropy"] == pytest.approx(1., abs=.01) and row["largest_share"] == 1.
    assert hm.pairwise_distance(collapsed)["mean"] < .01 * hm.pairwise_distance(real)["mean"]
    assert hm.distance_ratio(collapsed, real)["far"] == "below"
    assert hm.cluster_ratio(collapsed, real, centres)["far"] == "below"


def test_a_generator_with_variety_but_no_word_signal_shows_in_separation():
    rng = np.random.default_rng(1)
    centres = varied(50, 1)
    faithful = {f"w{k}": [c + .05 * rng.normal(size=c.shape) for _ in range(3)] for k, c in enumerate(centres)}
    unrelated = {f"w{k}": [rng.normal(size=c.shape) for _ in range(3)] for k, c in enumerate(centres)}
    assert hm.separation(faithful)["mean"] > .95
    assert hm.separation(unrelated)["mean"] == pytest.approx(.5, abs=.05)


def test_collapsed_query_matches_only_at_chance_and_the_rule_says_uninformative():
    T = varied(100, 2)
    names = [f"w{k}" for k in range(100)]
    rng = np.random.default_rng(3)
    collapsed = np.tile(T[:1], (100, 1)) + 1e-3 * rng.normal(size=T.shape)
    row = hm.match(names, collapsed, names, T, shuffles=50)
    assert row["top5"] < .25
    assert row["median_rank"] == pytest.approx(50.5, abs=20)
    assert hm.informative(row)["informative"] is False


def test_a_perfect_match_is_top1_and_informative():
    T = varied(100, 4)
    names = [f"w{k}" for k in range(100)]
    row = hm.match(names, T.copy(), names, T, shuffles=50)
    assert row["top1"] == 1. and row["top5"] == 1. and row["median_rank"] == 1
    assert row["chance"]["top5"] == .05
    assert row["shuffled"]["top5"] == pytest.approx(.05, abs=.03)
    assert hm.informative(row)["informative"] is True
    assert hm.lift_share(row, row) == 1.


def test_scaling_absorbs_a_constant_path_offset():
    T = varied(100, 5)
    names = [f"w{k}" for k in range(100)]
    shifted = T + 5.
    assert hm.match(names, shifted, names, T, shuffles=10)["top1"] == 1.
    assert hm.match(names, shifted, names, T, scale=False, shuffles=10)["top1"] < .2


def test_ties_take_the_middle_rank():
    ranks = hm.ranks_of(np.zeros((1, 2)), np.array([0]), np.zeros((5, 2)))
    assert ranks[0] == 3.


def test_every_sample_counts_inside_its_word():
    row = hm.rank_summary(["a", "a", "b"], [1, 9, 1], 10)
    assert row["top1"] == pytest.approx(.75)      # a: 1 of 2; b: 1 of 1
    assert row["words"] == 2 and row["samples"] == 3


def test_the_far_rule_needs_the_whole_interval():
    low = hm.boot_ratio([.5, .5, .5, .5], [1., 1., 1., 1.])
    assert low["far"] == "below" and low["ratio"] == .5
    high = hm.boot_ratio([2.] * 4, [1.] * 4)
    assert high["far"] == "above"
    noisy = hm.boot_ratio([.1, 1.5, .1, 1.5], [1.] * 4)
    assert noisy["far"] is None


# ---- what may be read ------------------------------------------------------------------------------------------------

def record(**k):
    base = dict(corpus="asllvd-text-v1", partition="train", sign="s", text="word", performer_sha256="p",
                continuous=dict(status="converted", state_sha256=SHA, state_key="k", seconds=1., available_channels=["right_hand"]))
    return {**base, **k}


@pytest.mark.parametrize("bad", [dict(partition="test"), dict(partition="val"), dict(partition="Train"), dict(partition=None), dict(corpus=hm.H2S, signer_component="c00"), dict(corpus=hm.H2S, signer_component="c01"), dict(corpus=hm.H2S), dict(corpus="semlex-text-v2"), dict(corpus=None)])
def test_refuses_anything_but_named_train_records(bad):
    with pytest.raises(SystemExit):
        hm.refuse(record(**bad))


def test_refuses_c00_on_any_corpus():
    with pytest.raises(SystemExit):
        hm.refuse(record(signer_component="c00"))


def test_filters_keep_only_train_of_the_named_groups():
    records = [record(sign="ok", text="word"), record(sign="t", partition="test"), record(sign="v", partition="val"), record(sign="o", corpus="youtube-asl-v1"),
               record(sign="c00", corpus=hm.H2S, signer_component="c00", partition="train"), record(sign="c02t", corpus=hm.H2S, signer_component="c02", partition="test"),
               record(sign="c02", corpus=hm.H2S, signer_component="c02"), record(sign="c04", corpus=hm.H2S, signer_component="c04")]
    assert [r["sign"] for r in hm.word_records(records)["word"]] == ["ok"]
    assert [r["sign"] for r in hm.sentence_records(records)] == ["c02"]


def test_the_test_split_report_is_never_opened():
    class Store:
        def list(self, prefix):
            return [f"{prefix}test_0.json", f"{prefix}train_0.json", f"{prefix}val_0.json"]

        def get(self, key):
            assert "/train_" in key, key
            return json.dumps(dict(results=[dict(sample_id="a", state_key="k", state_sha256=SHA), dict(sample_id="b", state_key="k", state_sha256=SHA)])).encode()
    assert hm._monocular_index(Store(), {"a"}) == {"a": ("k", SHA)}


def test_no_score_leaves_an_index():
    payload = dict(summary=dict(top1=.9), entries=[dict(entry="a b", seconds=1., scorer="recogniser", label="AB", scores=[9., 1.], probabilities=[.9], log_probabilities=[-.1], ranks=[1], seeds=[1], chosen={"16": dict(index=1, seed=1, score=9., state_key="k", state_sha256=SHA)})])
    rows = hm.index_rows(payload)
    assert rows == [dict(entry="a b", seconds=1., scorer="recogniser", n=16, index=1, state_key="k", state_sha256=SHA)]


def test_stored_losers_are_the_other_stored_samples():
    chosen = lambda i: dict(index=i, seed=1, score=0., state_key=f"k{i}", state_sha256=SHA)    # noqa: E731
    payload = dict(entries=[dict(entry="a", seconds=1., scorer="prior_pmi", chosen={"4": chosen(0), "8": chosen(5), "16": chosen(5)}), dict(entry="b", seconds=1., scorer="recogniser", chosen={"4": chosen(2), "8": chosen(2), "16": chosen(2)}), dict(entry="c", seconds=1., scorer="recogniser", chosen={"4": chosen(1), "8": chosen(6), "16": chosen(12)})])
    rows = hm.phase_a_rows(payload)
    assert [(r["entry"], r["index"], r["role"]) for r in rows] == [("a", 5, "best16"), ("a", 0, "loser"), ("b", 2, "best16"), ("c", 12, "best16"), ("c", 6, "loser"), ("c", 1, "loser")]


def test_clip_draw_goes_round_the_performers():
    records = [record(sign=f"a{k}", performer_sha256="a") for k in range(10)] + [record(sign="b0", performer_sha256="b")]
    picked = hm.pick_clips(records, "word", 3)
    assert len(picked) == 3 and "b" in {r["performer_sha256"] for r in picked}
    assert hm.pick_clips(records, "word", 3) == picked


@pytest.mark.parametrize("entry,name", [("accept", "accept"), ("a lot", None), ("24", None), ("don't", None), ("accepts", "accepts")])
def test_word_match_rule_is_exact(entry, name):
    assert hm.capture_match(entry) == name


# ---- round trip ------------------------------------------------------------------------------------------------------

def test_round_trip_reads_only_the_originals_valid_frames():
    values = curling()
    valid = np.ones((48, 15), bool)
    valid[40:] = False
    state = build(values, right_valid=valid, left_valid=np.zeros((48, 15), bool))
    garbage = values.copy()
    garbage[40:] = rot_x(np.random.default_rng(0).uniform(0, 3, (8, 15)))
    decoded = build(garbage)
    row = hm.round_trip_pair(state, decoded)
    assert row["Left"] is None
    assert row["Right"]["error_degrees"] < .01
    assert row["Right"]["decoded"]["speed_raw"] == pytest.approx(row["Right"]["original"]["speed_raw"], rel=1e-3)


def test_round_trip_reads_a_tokenizer_that_freezes_fingers():
    state = build(curling(), left_valid=np.zeros((48, 15), bool))
    row = hm.round_trip_pair(state, build(np.tile(curling()[:1], (48, 1, 1))))
    assert row["Right"]["decoded"]["speed_raw"] == pytest.approx(0., abs=1e-4)
    assert row["Right"]["original"]["speed_raw"] > .5
    assert row["Right"]["error_degrees"] > 5


# ---- added after the first numbers ---------------------------------------------------------------------------------

def test_coupling_tells_fingers_moving_together_from_independent_noise():
    together = hm.added_hand_row(curling(120).astype(float), np.ones((120, 15), bool), False)
    rng = np.random.default_rng(0)
    alone = hm.added_hand_row(rot_x(.3 * np.cumsum(rng.standard_normal((120, 15)), axis=0) * .1), np.ones((120, 15), bool), False)
    assert together["coupling_6hz"] > .95
    assert abs(alone["coupling_6hz"]) < .4


def test_a_quiet_hand_shows_a_floor_near_zero_and_a_noisy_one_does_not():
    rng = np.random.default_rng(1)
    move = np.zeros((96, 15))
    move[40:56] = np.linspace(0, 1., 16)[:, None]
    move[56:] = 1.
    quiet = hm.added_hand_row(rot_x(move), np.ones((96, 15), bool), False)
    noisy = hm.added_hand_row(rot_x(move + .08 * rng.standard_normal((96, 15))), np.ones((96, 15), bool), False)
    assert quiet["floor_6hz"] < .01
    assert noisy["floor_6hz"] > 10 * max(quiet["floor_6hz"], .01)


def test_cross_path_finds_the_shared_signal_at_its_shift_and_a_null_for_unrelated_clips():
    rng = np.random.default_rng(2)
    signal = np.cumsum(rng.standard_normal((200, 15)), axis=0) * .02
    a = rot_x(signal + .002 * rng.standard_normal((200, 15)))
    b = rot_x(np.roll(signal, 3, axis=0) + .002 * rng.standard_normal((200, 15)))
    other = rot_x(np.cumsum(rng.standard_normal((200, 15)), axis=0) * .02)
    same = hm.cross_path_row(build(a), build(b))["two_Right_mono_Right_6hz"]
    null = hm.cross_path_row(build(a), build(other))["two_Right_mono_Right_6hz"]
    assert same["r"] > .8 and abs(same["lag"]) == 3
    assert null["r"] < .4


def test_cross_path_skips_a_hand_with_no_valid_frames():
    row = hm.cross_path_row(build(curling(), left_valid=np.zeros((48, 15), bool)), build(curling()))
    assert row["two_Left_mono_Right_6hz"] is None
    assert row["two_Right_mono_Right_6hz"]["r"] > .9


def test_traversals_count_crossings_of_the_range():
    row = dict(entry="a", dominant="Right", hands=dict(Right=dict(bend_6hz=2., frames=48, range_mean=.5)))
    assert hm.traversals([row], "dominant")["mean"] == 8.
    assert hm.traversals([dict(entry="a", dominant=None, hands=dict())], "dominant")["n"] == 0


# ---- the whole read on synthetic rows --------------------------------------------------------------------------------

def _row(entry, source, core, speed, **k):
    hand = {m: speed for m in hm.METRICS}
    hand.update(frames=48, core=list(map(float, core)), core_30=list(map(float, core)), speed_per_joint_6hz=[speed] * 15, range_per_joint=[speed] * 15)
    validity = dict(valid=1., observed=0., confidence=1.)
    return dict(id=f"{source}|{entry}|{k.get('index', 0)}", entry=entry, source=source, hands=dict(Right=hand, Left=None), dominant="Right", validity=dict(Right=validity, Left=validity), **k)


def test_analyse_end_to_end_on_a_collapsed_slow_generator(tmp_path):
    """Generated is half real's speed and one handshape: the report must say far below on speed and on variety, and give its match no lift."""
    T = varied(120, 7)
    rng = np.random.default_rng(8)
    words = [f"word{chr(97 + k // 26)}{chr(97 + k % 26)}" for k in range(120)]
    write = lambda name, rows: (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))    # noqa: E731
    gen, real, capture = [], [], []
    for k, w in enumerate(words):
        capture.append(_row(w, "capture", T[k], 2., performer="x"))
        for c in range(3):
            real.append(_row(w, "real_words", T[k] + .1 * rng.normal(size=T[k].shape), 2., index=c, corpus="asllvd-text-v1", performer=f"p{c}"))
        for d in hm.DRAWS:
            gen.append(_row(w, f"draw{d}", T[0] + .01 * rng.normal(size=T[0].shape), 1., scorer="recogniser", role="best16", index=d))
        gen.append(_row(w, "phase_a", T[0], 1., scorer="recogniser", role="best16", index=0))
        gen.append(_row(w, "phase_a", T[0], .5, scorer="recogniser", role="loser", index=1))
    write("generated", gen), write("real_words", real), write("capture", capture)
    report = hm.analyse(tmp_path)
    json.dumps(report)
    assert report["ratios"]["generated_over_real_words"]["dominant"]["speed_6hz"]["far"] == "below"
    assert report["variety"]["per_draw"]["1"]["pairwise_over_real"][0]["far"] == "below"
    assert report["variety"]["per_draw"]["1"]["clusters_over_real"][0]["far"] == "below"
    wm = report["word_match"]
    assert wm["real_words"]["common_words"]["z_scored"]["top1"] > .8
    assert wm["uninformative_rule"]["informative"] is True
    assert wm["generated_share_of_real_lift"]["share"] < .2
    assert report["best_of_16"]["recogniser"]["best_over_loser"]["speed_6hz"]["ratio"] == 2.
    assert report["word_match_exclusions"]["matched"] == 120
