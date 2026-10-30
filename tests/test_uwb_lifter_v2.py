"""UWB lifter v2 and gate v2: split, pairs, validity head, the ``head`` rule, trainer, gate options and the v2 verdict."""
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_uwb_lifter_pairs as pairs_job  # noqa: E402
import build_uwb_lifter_split as split_job  # noqa: E402
import train_uwb_lifter_v2 as trainer  # noqa: E402
import uwb_gate_verdict as verdict  # noqa: E402
import uwb_lifter_gate as gate  # noqa: E402
import uwb_lifter_v2_crops as crops  # noqa: E402
from semlex_synthetic import rotation, semlex_array  # noqa: E402
from test_uwb_lifter_gate import _state  # noqa: E402
from uwb_synthetic import TrueWorldXY, uwb_clip_json  # noqa: E402

from signtranslator.data_engineering import uwb_keypoints as uk  # noqa: E402
from signtranslator.data_engineering import world_regressor as wr  # noqa: E402
from signtranslator.data_engineering.video_landmarks import Y_UP  # noqa: E402

FPS = 25.


# ----------------------------------------------------------------------------------------------------------------------
# fixtures

def xy2_bundle(seed=0):
    torch.manual_seed(seed)
    hand = {s: (wr.MLP(40, 63, width=16, depth=1).eval(), np.zeros(40, np.float32), np.ones(40, np.float32))
            for s in ("Left", "Right")}
    pose = (wr.MLP(50, 75 + 33, width=16, depth=1).eval(), np.zeros(50, np.float32), np.ones(50, np.float32))
    return wr.WorldRegressors(hand=hand, pose=pose, features=wr.FEATURES_XY2)


def with_head(bundle, thresholds, seed=1):
    torch.manual_seed(seed)
    head = (wr.MLP(wr.VALIDITY_INPUTS, 6, width=8, depth=1).eval(), np.zeros(wr.VALIDITY_INPUTS, np.float32),
            np.ones(wr.VALIDITY_INPUTS, np.float32), dict(thresholds))
    return wr.WorldRegressors(hand=bundle.hand, pose=bundle.pose, features=bundle.features, validity=head)


def build_store(root: Path, videos=("vid0", "vid1", "vid2"), *, frames=110, starts=(2., 7., 12., 17.)):
    """A directory store: our reports (channel ch<k>, 25 fps) and states, the UWB translations and one zip. UWB clip
    k of a video shows the same motion as our cue k."""
    uwb_prefix, yt_prefix = "acq/uwb", "acq/yt"
    for d in (root / uwb_prefix, root / yt_prefix / "videos", root / yt_prefix / "states"):
        d.mkdir(parents=True, exist_ok=True)
    translations = {}
    with zipfile.ZipFile(root / uwb_prefix / "raw_keypoints_1.zip", "w") as z:
        for v, video in enumerate(videos):
            segments, order, tr = [], [], {}
            for k, start in enumerate(starts):
                first = math.floor(start * FPS)
                clip_id = f"{video}.{first:06d}-{first + frames - 1:06d}"
                data, body, hands = uwb_clip_json(size=(1280, 720), frames=frames, seed=10 * v + k + 1)
                z.writestr(f"{clip_id}.json", json.dumps(data))
                order.append(clip_id)
                tr[clip_id] = {"translation": f"Cue number {k}"}
                times = (first + np.arange(frames)) / FPS + .002
                state = _state(body, times, {s: (h - h[:, :1], np.ones(frames, bool)) for s, h in hands.items()})
                key = f"{yt_prefix}/states/{video}-{k:04d}.npz"
                digest = state.save(root / key)
                segments.append(dict(cue=k, sample_id=f"{video}-{k:04d}", start=start + .01,
                                     end=(first + frames - 1) / FPS, text=f"Cue number {k}.", state_key=key,
                                     state_sha256=digest))
            tr["clip_order"] = order
            translations[video] = tr
            (root / yt_prefix / "videos" / f"{video}.json").write_text(json.dumps(dict(
                video_id=video, status="converted", meta=dict(fps=25, channel_id=f"ch{v}"), segments=segments)))
    (root / uwb_prefix / "YT.translations.all.json").write_text(json.dumps(translations))
    return ["--store", f"dir:{root}", "--uwb-prefix", uwb_prefix, "--youtube-prefix", yt_prefix]


def crafted_split(path: Path, train=("vid0",), val=("vid1",), held=("vid2",)) -> str:
    channel = lambda v: f"ch{v[3:]}"                                                    # noqa: E731
    out = dict(schema=split_job.SCHEMA, **{p: dict(videos=sorted(vs), channels=sorted({channel(v) for v in vs}))
                                           for p, vs in (("train", train), ("lifter_val", val), ("held_out", held))})
    text = json.dumps(out, sort_keys=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# split

def test_split_is_deterministic_channel_disjoint_and_holds_out_monitor_channels():
    from build_youtube_asl_manifest import monitor_channel
    videos = {f"v{k:04d}": dict(channel=f"ch{k // 3}", uwb_clips=10 + k % 5, our_cues=8, candidate_pairs=k % 7)
              for k in range(2400)}
    videos["v9999"] = dict(channel=None, uwb_clips=3, our_cues=3, candidate_pairs=1)
    excluded = {"v0000", "v0001"}
    a, b = split_job.split_videos(videos, excluded), split_job.split_videos(dict(reversed(videos.items())), excluded)
    assert a == b
    parts = split_job.PARTS
    for kind in ("videos", "channels"):
        for x in range(3):
            for y in range(x + 1, 3):
                assert not set(a[parts[x]][kind]) & set(a[parts[y]][kind])
    assert a["dropped"] == dict(development=["v0000", "v0001"], no_channel=["v9999"])
    assert sum(len(a[p]["videos"]) for p in parts) == len(videos) - 3
    monitors = {f"ch{k}" for k in range(800) if monitor_channel(f"ch{k}", .01)}
    assert monitors and monitors <= set(a["held_out"]["channels"]) and set(a["monitor_channels"]) == monitors
    for ch in a["held_out"]["channels"]:
        assert ch in monitors or split_job.below(split_job.HELD_OUT_SALT, ch, "0.20")
    for ch in a["lifter_val"]["channels"]:
        assert split_job.below(split_job.VAL_SALT, ch, "0.10") and not split_job.below(split_job.HELD_OUT_SALT, ch, "0.20")
    share = len(a["held_out"]["channels"]) / 800
    assert .15 < share < .27                                     # 0.20 of the channels plus the 1% monitor channels
    assert .05 < len(a["lifter_val"]["channels"]) / (800 - len(a["held_out"]["channels"])) < .15
    assert split_job.below("s", "x", "1") and not split_job.below("s", "x", "0")
    comp = split_job.composition(a, videos)
    assert comp["held_out"]["videos"] == len(a["held_out"]["videos"]) and comp["train"]["largest_channel"]["videos"] == 3
    assert comp["held_out"]["monitor_channels"] == len(monitors)


def test_split_main_writes_identical_bytes_twice_with_composition(tmp_path):
    common = build_store(tmp_path / "s", frames=20)
    dev = tmp_path / "dev.txt"
    dev.write_text("# development\nvidX\n")
    outs = []
    for k in range(2):
        out = tmp_path / f"split{k}.json"
        assert split_job.main(common + ["--exclude-videos", str(dev), "--workers", "1", "--out", str(out)]) == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    split = json.loads(outs[0])
    assert split["schema"] == "uwb-lifter-split-v1" and split["population"]["uwb_and_ours_converted"] == 3
    assert sorted(sum((split[p]["videos"] for p in split_job.PARTS), [])) == ["vid0", "vid1", "vid2"]
    assert sum(split["composition"][p]["candidate_pairs"] for p in split_job.PARTS) == 12
    with pytest.raises(SystemExit, match="exists"):
        split_job.main(common + ["--exclude-videos", str(dev), "--out", str(tmp_path / "x.json"),
                                 "--output-key", "acq/uwb/YT.translations.all.json"])


def test_a_split_whose_parts_overlap_is_refused_and_the_gate_refuses_non_held_out_videos(tmp_path):
    crafted_split(tmp_path / "bad.json", train=("vid0",), val=("vid1",), held=("vid0", "vid2"))
    with pytest.raises(SystemExit, match="share"):
        gate.load_split(None, str(tmp_path / "bad.json"))
    crafted_split(tmp_path / "ok.json")
    split, _ = gate.load_split(None, str(tmp_path / "ok.json"))
    gate.check_population([dict(video_id="vid2")], split)
    with pytest.raises(SystemExit, match="not held out"):
        gate.check_population([dict(video_id="vid2"), dict(video_id="vid0")], split)


# ----------------------------------------------------------------------------------------------------------------------
# pairs

def _semlex_pair(frames=110, *, shift=0, our_body=None, hidden_arms=False, seed=3):
    """(our state, UWB clip) of one synthetic cue; ``shift``: our frame t shows UWB frame t + shift."""
    data, body, hands = uwb_clip_json(size=(1280, 720), frames=frames + 8, seed=seed)
    first = 300
    clip = uk.parse_clip(dict(data, keypoints=data["keypoints"][4:4 + frames]),
                         f"vid.{first:06d}-{first + frames - 1:06d}")
    ours_body = (body if our_body is None else our_body)[4 + shift:4 + shift + frames]
    packed = {s: (h[4 + shift:4 + shift + frames] - h[4 + shift:4 + shift + frames, :1], np.ones(frames, bool))
              for s, h in hands.items()}
    visible = None
    if hidden_arms:
        visible = np.ones((frames, 33), bool)
        visible[:, 11:17] = False
    times = (first + np.arange(frames)) / FPS + .002
    return _state(ours_body, times, packed, visible), clip, body, hands


def test_pair_filters_each_refuse_their_case():
    ours, clip, *_ = _semlex_pair()
    assert pairs_job.pair_arrays(ours, clip, FPS, 25)["status"] == "kept"
    assert pairs_job.pair_arrays(ours, clip, FPS, 30)["reason"] == "stream mismatch"
    assert pairs_job.pair_arrays(ours, clip, FPS, None)["reason"] == "stream mismatch"
    hidden, clip_h, *_ = _semlex_pair(hidden_arms=True)
    assert pairs_job.pair_arrays(hidden, clip_h, FPS, 25)["reason"] == "not arm-aligned"
    shifted, clip_s, *_ = _semlex_pair(shift=3)
    refused = pairs_job.pair_arrays(shifted, clip_s, FPS, 25)
    assert refused["reason"] == "lag outside {-1, 0, 1}" and abs(refused["alignment"]["best_lag"]) == 3
    short, clip_short, *_ = _semlex_pair(frames=60)
    refused = pairs_job.pair_arrays(short, clip_short, FPS, 25)
    assert refused["reason"] == "fewer than 400 bone-frames" and refused["alignment"]["frames"] < 400
    _, _, body, _ = _semlex_pair()
    turned = body @ rotation((0, 0, 1), 45.).T                     # every bone 45 deg off in the image plane
    off, clip_o, *_ = _semlex_pair(our_body=turned)
    loose = dict(pairs_job.FILTERS, lags=tuple(range(-6, 7)))
    refused = pairs_job.pair_arrays(off, clip_o, FPS, 25, loose)
    assert refused["reason"] == "lag-0 score over 20 deg" and refused["alignment"]["lag0_score"] > 20


def test_kept_pair_targets_are_our_directions_in_mediapipe_axes_with_labels_and_fingers():
    ours, clip, body, hands = _semlex_pair()
    kept = pairs_job.pair_arrays(ours, clip, FPS, 25)
    a = kept["alignment"]
    assert a["lag"] == 0 and a["best_lag"] == 0 and a["frames"] >= 400 and a["lag0_score"] < 1e-3
    n = len(kept["our_index"])
    assert n == 110 and (kept["our_index"] == kept["uwb_index"]).all()
    truth = body[4:114]
    for k, (i, j) in enumerate(((11, 13), (13, 15), (12, 14), (14, 16), (12, 11))):
        d = truth[:, j] - truth[:, i]
        d = d / np.linalg.norm(d, axis=-1, keepdims=True) @ Y_UP
        assert kept["arm_valid"][:, k].all()
        np.testing.assert_allclose(kept["arm_dirs"][:, k].astype(np.float64), d, atol=2e-3)
    np.testing.assert_allclose(kept["pose"], clip.pose / [1280, 720], atol=1e-6)
    assert kept["chest_valid"].all() and kept["forearm_valid"].all()
    assert (kept["label_mask"][:, :2] == 0).all() and kept["label_mask"][:, 2:].all() and kept["labels"][:, 2:].all()
    from signtranslator.data_engineering.video_landmarks import finger_joints
    for s, side in enumerate(("Left", "Right")):
        h = hands[side][4:114]
        for k, joint in enumerate(finger_joints(side)):
            d = h[:, joint.end] - h[:, joint.start]
            d = d / np.linalg.norm(d, axis=-1, keepdims=True) @ Y_UP
            ok = kept["finger_valid"][:, s, k]
            assert ok.mean() > .9
            np.testing.assert_allclose(kept["finger_dirs"][ok, s, k].astype(np.float64), d[ok], atol=3e-3)


def test_labels_follow_our_validity_and_mask_frames_without_our_chest():
    ours_visible, clip, body, hands = _semlex_pair()
    frames = 110
    visible = np.ones((frames, 33), bool)
    visible[:30, 16] = False                       # our right wrist unseen: right forearm invalid
    visible[30:40, 14] = False                     # our right elbow unseen: right upper arm and forearm invalid
    visible[40:50, 23:25] = False                  # no hips: our chest invalid, every label masked
    times = ours_visible.timestamps
    packed = {s: (h[4:114] - h[4:114, :1], np.ones(frames, bool)) for s, h in hands.items()}
    ours = _state(body[4:114], times, packed, visible)
    kept = pairs_job.pair_arrays(ours, clip, FPS, 25, dict(pairs_job.FILTERS, min_bone_frames=0))
    assert kept["status"] == "kept"
    labels, mask = kept["labels"], kept["label_mask"]
    assert (~mask[40:50]).all() and not kept["chest_valid"][40:50].any()
    assert (labels[:30, 3] == 1).all() and (labels[:30, 5] == 0).all() and mask[:30, 5].all()   # wrist label 0
    assert (labels[30:40, 3] == 0).all() and not mask[30:40, 5].any()          # no upper arm: wrist unlabelled
    assert (labels[50:, 2:] == 1).all() and not kept["forearm_valid"][:40, 1].any()


def test_pairs_builder_refuses_held_out_videos_everywhere(tmp_path):
    with pytest.raises(pairs_job.HeldOutRefused):
        pairs_job.refuse_held_out(["vid0", "vid2"], {"vid2"})
    split = dict(train=dict(videos=["vid0"], channels=["ch0"]), lifter_val=dict(videos=[], channels=[]),
                 held_out=dict(videos=["vid2"], channels=["ch2"]))
    with pytest.raises(pairs_job.HeldOutRefused):
        pairs_job.candidate_jobs(split, {}, {"vid2": {}}, {})
    pairs_job._WORKER.clear()
    pairs_job._WORKER.update(held_out=frozenset({"vid2"}), settings={}, store=None, archives={})
    with pytest.raises(pairs_job.HeldOutRefused):            # before anything is read
        pairs_job._build(dict(video_id="vid2", clip_id="c", sample_id="s", part="train", channel="ch2", fps=25.,
                              meta_fps=25, zip="z", state_key="k", state_sha256="0"))
    pairs_job._WORKER.clear()


def test_pairs_main_builds_shards_of_train_and_lifter_val_only(tmp_path):
    common = build_store(tmp_path / "s")
    split_sha = crafted_split(tmp_path / "split.json")
    out = tmp_path / "pairs"
    assert pairs_job.main(common + ["--split-file", str(tmp_path / "split.json"), "--workers", "1",
                                    "--shard-pairs", "3", "--out-dir", str(out)]) == 0
    index = json.loads((out / "pairs-index.json").read_text())
    assert index["split_sha256"] == split_sha and index["videos_read"] == 2
    kept = index["kept_pairs"]
    assert set(kept) == {"train", "lifter_val"} and index["candidate_pairs"] == 8
    assert sum(kept.values()) + sum(index["refused_pairs"].values()) == 8
    assert set(index["refused_pairs"]) <= {"fewer than 400 bone-frames", "lag outside {-1, 0, 1}"}
    assert [s["pairs"] for s in index["shards"]][:-1] == [3] * (len(index["shards"]) - 1)
    assert sum(s["pairs"] for s in index["shards"]) == sum(kept.values())
    videos = set()
    for entry in index["shards"]:
        z = pairs_job.load_shard(out / entry["file"], entry["sha256"])
        videos |= set(z["pair_video"].tolist())
        assert z["pair_offset"][-1] == len(z["pose"]) == entry["frames"]
    assert videos == {"vid0", "vid1"}
    with pytest.raises(SystemExit, match="SHA-256"):
        pairs_job.load_shard(out / index["shards"][0]["file"], "0" * 64)
    with pytest.raises(SystemExit, match="not empty"):
        pairs_job.main(common + ["--split-file", str(tmp_path / "split.json"), "--out-dir", str(out)])


# ----------------------------------------------------------------------------------------------------------------------
# bundles, validity head, head rule

def test_bundles_without_a_head_save_load_and_predict_exactly_as_before():
    bundle = xy2_bundle()
    payload = bundle.save()
    loaded = wr.WorldRegressors.load(payload)
    assert loaded.validity is None and loaded.save() == payload
    assert "validity" not in torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    with pytest.raises(ValueError, match="no validity head"):
        loaded.point_validity(np.zeros((2, 33, 2)), 1.)
    headed = with_head(loaded, {"Left": .4, "Right": .6})
    payload2 = headed.save()
    back = wr.WorldRegressors.load(payload2)
    assert back.validity[3] == {"Left": .4, "Right": .6}
    image = semlex_array((640, 480), frames=6)[0][:, 478:511, :2]
    for a, b in zip(loaded.pose_world(image, 4 / 3), back.pose_world(image, 4 / 3)):
        np.testing.assert_array_equal(a, b)
    hand = semlex_array((640, 480), frames=6)[0][:, 511:532, :2]
    np.testing.assert_array_equal(loaded.hands(hand, 4 / 3, "Left")[0], back.hands(hand, 4 / 3, "Left")[0])
    p, ok = back.point_validity(image, 4 / 3)
    kept, _ = back.validity_visible(image, 4 / 3)
    assert p.shape == (6, 6) and ok.all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_array_equal(kept, p >= np.array([.4, .6, .4, .6, .4, .6]))
    features, valid = wr.validity_features(image, 4 / 3)
    assert features.shape == (6, wr.VALIDITY_INPUTS) and valid.all()
    np.testing.assert_array_equal(features[:, :50], wr.pose_features_xy2(image, 4 / 3)[0])
    np.testing.assert_allclose(features[:, 50:62], image[:, 11:17].reshape(6, -1), atol=1e-6)
    with pytest.raises(ValueError, match="thresholds"):
        with_head(loaded, {"Left": .5}).save()


def test_a_head_attached_bundle_gives_byte_identical_gate_v1_rows(tmp_path):
    """The gate (v1 mode) with and without a validity head on the same bundle: identical but for the bundle SHA."""
    common = build_store(tmp_path / "s", frames=20) + ["--workers", "1"]
    plain = xy2_bundle()
    reports = []
    for name, b in (("plain", plain), ("headed", with_head(plain, {"Left": .5, "Right": .5}))):
        payload = b.save()
        (tmp_path / f"{name}.pt").write_bytes(payload)
        out = tmp_path / f"{name}.json"
        assert gate.main(common + ["--regressors", str(tmp_path / f"{name}.pt"), "--regressors-sha256",
                                   hashlib.sha256(payload).hexdigest(), "--visibility", "inside",
                                   "--candidate-visibility", "learned,inside", "--bootstrap", "20",
                                   "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        for key in ("regressors", "regressors_sha256", "out"):
            report["settings"].pop(key)
        report.pop("regressors_sha256")
        reports.append(report)
    assert reports[0] == reports[1]
    assert "split" not in reports[0] and not {"split_file", "lag_curves", "secondary_visibility"} & set(reports[0]["settings"])
    assert all("curve" not in p["alignment"] for p in reports[0]["pairs"])


class HeadStub(TrueWorldXY):
    validity = ("stub",)

    def __init__(self, body, hands, keep):
        super().__init__(body, hands)
        self.keep = keep

    def validity_visible(self, image, aspect):
        return self.keep[:len(image)], np.ones(len(image), bool)


def test_head_rule_keeps_points_11_to_16_by_the_head_and_the_rest_inside():
    data, body, hands = uwb_clip_json(size=(1280, 720))
    clip = uk.parse_clip(data, "vid.000000-000007")
    keep = np.zeros((len(body), 6), bool)
    keep[::2] = True
    keep[:, 3] = False                               # the right elbow never kept
    stub = HeadStub(body, hands, keep)
    head = uk.clip_landmarks(clip, 30., regressors=stub, visibility="head")[0].pose_visibility
    inside = uk.clip_landmarks(clip, 30., regressors=stub, visibility="inside")[0].pose_visibility
    np.testing.assert_array_equal(head[:, 11:17], keep.astype(float))
    others = [k for k in range(33) if not 11 <= k <= 16]
    np.testing.assert_array_equal(head[:, others], inside[:, others])
    assert uk.visibility_margin("head") == 0.
    with pytest.raises(ValueError, match="validity head"):
        uk.clip_landmarks(clip, 30., regressors=TrueWorldXY(body, hands), visibility="head")
    with pytest.raises(ValueError, match="validity head"):
        uk.clip_landmarks(clip, 30., world="pseudo", visibility="head")
    state, report = uk.convert_clip(clip, 30., source_sha256="a" * 64, fps_source="estimated", regressors=stub,
                                    visibility="head")
    assert "validity head (per-side thresholds)" in state.channels["body"].inference_method
    real = with_head(xy2_bundle(), {"Left": 1.01, "Right": 0.})
    vis = uk.clip_landmarks(clip, 30., regressors=real, visibility="head")[0].pose_visibility
    assert not vis[:, [11, 13, 15]].any() and vis[:, [12, 14, 16]].all()


# ----------------------------------------------------------------------------------------------------------------------
# thresholds and trainer

def test_thresholds_maximise_agreement_subject_to_uwb_only():
    rng = np.random.default_rng(0)
    n = 4000
    ours = rng.random((n, 2)) < .6
    p = np.where(np.repeat(ours, 3, axis=1)[:, [0, 3, 1, 4, 2, 5]], rng.uniform(.35, 1., (n, 6)),
                 rng.uniform(0., .7, (n, 6)))
    chest = rng.random(n) < .9
    chosen = trainer.choose_thresholds(p, ours, chest)
    for k, side in enumerate(("Left", "Right")):
        q = p[chest][:, list(trainer.SIDE_POINTS[side])].min(1)
        o = ours[chest][:, k]

        def score(t):
            kept = q >= t
            return (kept == o).mean(), (kept & ~o).mean()
        t = chosen[side]["threshold"]
        agreement, uwb_only = score(t)
        assert uwb_only <= .07 and chosen[side]["agreement"] == round(agreement, 4)
        assert all(score(u)[0] <= agreement + 1e-12 for u in trainer.THRESHOLDS if score(u)[1] <= .07)
        assert chosen[side]["frames"] == chest.sum()
    never = trainer.choose_thresholds(np.ones((600, 6)), np.zeros((600, 2), bool), np.ones(600, bool))
    assert never["Left"]["threshold"] == 1.01 and never["Left"]["uwb_only"] == 0.


def test_thresholds_are_refused_below_500_chest_valid_frames_per_side():
    """Review finding 6: no chest-valid frame (or too few) used to give a threshold silently."""
    with pytest.raises(ValueError, match="0 chest-valid"):
        trainer.choose_thresholds(np.zeros((10, 6)), np.zeros((10, 2), bool), np.zeros(10, bool))
    chest = np.zeros(1000, bool)
    chest[:499] = True
    with pytest.raises(ValueError, match="499 chest-valid"):
        trainer.choose_thresholds(np.zeros((1000, 6)), np.zeros((1000, 2), bool), chest)
    chest[499] = True
    assert trainer.choose_thresholds(np.zeros((1000, 6)), np.zeros((1000, 2), bool), chest)["Left"]["frames"] == 500


def test_thresholds_respect_the_cap_and_prefer_the_higher_on_ties():
    rng = np.random.default_rng(0)
    n = 5000
    ours = rng.random((n, 2)) < .6
    p = np.clip(ours[:, [0, 1, 0, 1, 0, 1]] * .6 + rng.random((n, 6)) * .4, 0, 1)
    out = trainer.choose_thresholds(p, ours, np.ones(n, bool))
    for s in ("Left", "Right"):
        feasible = [c for c in out[s]["curve"] if c["uwb_only"] <= .07]
        best = max(c["agreement"] for c in feasible)
        assert out[s]["uwb_only"] <= .07 and out[s]["agreement"] == best
        assert out[s]["threshold"] == max(c["threshold"] for c in feasible if c["agreement"] == best)


def _write_h2s_pairs(directory, legacy_dir, name, signer, seed):
    from test_train_world_regressors_script import _write_pairs
    _write_pairs(directory, name, signer, seed, legacy_dir)


def _write_crop(crop_dir, legacy_dir, name, seed, frames=12):
    """A tight crop in validate_semlex_adapter's layout (meta + ref_* Tasks landmarks) and its legacy twin."""
    array, body, hands = semlex_array((640, 480), frames=frames, seed=seed)
    vis = np.where(np.arange(33) < 23, .9, .1)[None].repeat(frames, 0)
    vis[::2, 14] = .2
    fields = dict(times=np.arange(frames) / 30., pose_world=body @ Y_UP, pose_visibility=vis,
                  pose_image=array[:, 478:511, :2], pose_found=np.ones(frames, bool),
                  hands_world=np.zeros((frames, 2, 21, 3)), hands_image=np.zeros((frames, 2, 21, 2)),
                  hands_found=np.zeros((frames, 2), bool), blendshapes=np.zeros((frames, 0)),
                  face_matrix=np.tile(np.eye(4), (frames, 1, 1)), face_found=np.zeros(frames, bool))
    meta = dict(name=name[:-4], video=name.split("__")[0] + ".mp4", shape="4:3-tight", box=[0, 0, 640, 480],
                size=[640, 480], names=[])
    np.savez(Path(crop_dir) / name, meta=np.array(json.dumps(meta)), **{f"ref_{k}": v for k, v in fields.items()})
    np.savez(Path(legacy_dir) / name, holistic=array.astype(np.float16), size=np.array([640, 480]),
             times=np.arange(frames) / 30., pose_visibility=np.zeros((frames, 33), np.float16))


def test_signer_9_and_validation_signers_are_never_read(tmp_path):
    with pytest.raises(SystemExit, match="never read"):
        crops.check_signers(["3", "9"])
    with pytest.raises(SystemExit, match="never read"):
        crops.check_signers(["1"])
    names = [f"a_{k}-{s}-rgb_front.mp4" for k in range(4) for s in (1, 2, 3, 9, 11)]
    picked = crops.sample_members(names, crops.check_signers(["3", "11"]), 2, 0)
    assert len(picked) == 4 and {crops.signer_of(n) for n in picked} == {"3", "11"}
    c, legacy = tmp_path / "c", tmp_path / "l"
    c.mkdir()
    legacy.mkdir()
    (c / "v_1-9-rgb_front__4x3-tight.npz").write_bytes(b"never opened")
    with pytest.raises(SystemExit, match="signer 9"):
        trainer.load_crops(c, legacy)
    (c / "v_1-9-rgb_front__4x3-tight.npz").unlink()
    (c / "v_1-1-rgb_front__4x3-tight.npz").write_bytes(b"never opened")
    with pytest.raises(SystemExit, match="not a training signer"):
        trainer.load_crops(c, legacy)
    p = tmp_path / "pairs"
    p.mkdir()
    (p / "v_1-9-rgb_front__1x1.npz").write_bytes(b"never opened")
    with pytest.raises(SystemExit, match="signer 9"):
        trainer.how2sign_chunks(p, legacy, [])


def test_crops_legacy_models_must_match_the_v2_pairs():
    crops.check_models(json.loads(json.dumps(crops.LEGACY_MODELS)))
    other = json.loads(json.dumps(crops.LEGACY_MODELS))
    other["models"]["pose_landmark/pose_landmark_full.tflite"] = "0" * 64
    with pytest.raises(SystemExit, match="pose_landmark_full"):
        crops.check_models(other)
    with pytest.raises(SystemExit, match="mediapipe"):
        crops.check_models(dict(crops.LEGACY_MODELS, mediapipe="0.10.21"))


def test_trainer_pose_select_and_validity_end_to_end(tmp_path):
    common = build_store(tmp_path / "s")
    split_sha = crafted_split(tmp_path / "split.json")
    yt = tmp_path / "yt"
    assert pairs_job.main(common + ["--split-file", str(tmp_path / "split.json"), "--workers", "1",
                                    "--out-dir", str(yt)]) == 0
    h2s, legacy = tmp_path / "h2s", tmp_path / "h2s-legacy"
    c, cl = tmp_path / "crops", tmp_path / "crops-legacy"
    for d in (h2s, legacy, c, cl):
        d.mkdir()
    for k, signer in enumerate(("h2s-1", "h2s-3", "h2s-5")):
        _write_h2s_pairs(h2s, legacy, f"v{k}_1-{signer[4:]}-rgb_front__1x1.npz", signer, k + 1)
    for k, signer in enumerate(("3", "8")):
        _write_crop(c, cl, f"w{k}_1-{signer}-rgb_front__4x3-tight.npz", k + 7)
    init = xy2_bundle()
    payload = init.save()
    (tmp_path / "xy2.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    base = ["pose", "--split-file", str(tmp_path / "split.json"), "--yt-pairs", str(yt), "--h2s-pairs", str(h2s),
            "--h2s-legacy", str(legacy), "--init-bundle", str(tmp_path / "xy2.pt"), "--init-sha256", sha,
            "--epochs", "1", "--batch", "64"]
    with pytest.raises(SystemExit):                          # amendment 2: the crops never reach the pose regressor
        trainer.main(base + ["--crops", str(c), "--out", str(tmp_path / "x.pt"), "--report", str(tmp_path / "x.json")])
    for name, extra in (("ft", []), ("scratch", ["--from-scratch"]), ("hands", ["--finetune-hands"])):
        assert trainer.main(base + extra + ["--out", str(tmp_path / f"{name}.pt"),
                                            "--report", str(tmp_path / f"{name}.json")]) == 0
    ft = json.loads((tmp_path / "ft.json").read_text())
    assert ft["split_sha256"] == split_sha and ft["arm"] == "fine-tuned from xy2"
    assert {m["signer"] for m in ft["how2sign_pairs_manifest"]} == {"h2s-1", "h2s-3", "h2s-5"}
    assert ft["youtube"]["train_frames"] > 0 and ft["youtube"]["lifter_val_frames"] > 0
    assert ft["lifter_val"]["pose"]["forearms"]["n"] > 0 and ft["lifter_val"]["init_pose"]["forearms"]["n"] > 0
    assert "crops_manifest" not in ft and ft["how2sign_frames"] == 24          # the two training-signer pairs only
    tuned = wr.WorldRegressors.load((tmp_path / "ft.pt").read_bytes(), expected_sha256=ft["bundle_sha256"])
    for side in ("Left", "Right"):                         # hands frozen: the xy2 hands exactly
        for a, b in zip(tuned.hand[side][0].state_dict().values(), init.hand[side][0].state_dict().values()):
            assert torch.equal(a, b)
    hands = json.loads((tmp_path / "hands.json").read_text())
    assert hands["settings"]["hands"] == "fine-tuned" and hands["lifter_val"]["hands"]["Left"]["youtube_frames"] > 0
    assert trainer.main(["select", "--reports", str(tmp_path / "ft.json"), str(tmp_path / "scratch.json"),
                         "--out", str(tmp_path / "selection.json")]) == 0
    selection = json.loads((tmp_path / "selection.json").read_text())
    scores = [trainer.selection_score(json.loads((tmp_path / f"{n}.json").read_text())) for n in ("ft", "scratch")]
    assert selection["chosen"] == (0 if scores[0] <= scores[1] else 1)
    assert [c["score"] for c in selection["candidates"]] == [round(x, 4) for x in scores]
    assert ft["peak_rss_mb"] > 0
    with pytest.raises(SystemExit, match="frozen hands"):
        trainer.select([ft, hands])
    chosen = ("ft", "scratch")[selection["chosen"]]
    chosen_sha = hashlib.sha256((tmp_path / f"{chosen}.pt").read_bytes()).hexdigest()
    args = ["validity", "--split-file", str(tmp_path / "split.json"), "--yt-pairs", str(yt), "--crops", str(c),
            "--crops-legacy", str(cl), "--bundle", str(tmp_path / f"{chosen}.pt"), "--bundle-sha256", chosen_sha,
            "--epochs", "2", "--batch", "64", "--out", str(tmp_path / "final.pt"), "--report", str(tmp_path / "v.json")]
    with pytest.raises(SystemExit, match="at least 500 needed"):            # amendment 2: the frame floor
        trainer.main(args)
    args[-4:-4] = ["--min-threshold-frames", "50"]                          # the tiny synthetic lifter-val
    assert trainer.main(args) == 0
    report = json.loads((tmp_path / "v.json").read_text())
    final = wr.WorldRegressors.load((tmp_path / "final.pt").read_bytes(), expected_sha256=report["bundle_sha256"])
    assert final.validity[3] == report["thresholds"]
    for side in ("Left", "Right"):
        assert report["lifter_val"][side]["uwb_only"] <= .07
    assert report["frames"]["youtube_train"] > 0 and report["frames"]["crops"] == 24
    assert {m["signer"] for m in report["crops_manifest"]} == {"h2s-3", "h2s-8"}
    with pytest.raises(FileExistsError):
        trainer.main(args)
    with pytest.raises(SystemExit, match="already has a validity head"):
        trainer.main(args[:-4] + ["--bundle", str(tmp_path / "final.pt"), "--bundle-sha256", report["bundle_sha256"],
                                  "--out", str(tmp_path / "x.pt"), "--report", str(tmp_path / "x.json")])
    crafted_split(tmp_path / "other.json", train=("vid0", "vid3"))
    with pytest.raises(SystemExit, match="pairs index of split"):
        trainer.load_youtube(yt, *gate.load_split(None, str(tmp_path / "other.json")), "train")
    with pytest.raises(SystemExit, match="only train and lifter_val"):
        trainer.load_youtube(yt, *gate.load_split(None, str(tmp_path / "split.json")), "held_out")


def test_arm_selection_uses_the_mean_of_upper_arm_and_forearm_medians():
    """Amendment 1: forearm-only would pick the arm that trades upper arms away; the mean does not."""
    def arm(name, upper, fore):
        return dict(stage="pose", arm=name, settings=dict(hands="frozen (xy2)"), split_sha256="s", bundle_sha256=name,
                    lifter_val=dict(pose=dict(upper_arms=dict(median=upper), forearms=dict(median=fore))))
    ft, scratch = arm("fine-tuned from xy2", 8., 12.), arm("from-scratch", 14., 11.)
    out = trainer.select([ft, scratch])
    assert out["chosen"] == 0 and [c["score"] for c in out["candidates"]] == [10., 12.5]
    assert trainer.select([scratch, ft])["chosen"] == 1
    tie = arm("from-scratch", 9., 11.)                                     # mean 10: tie goes to the fine-tuned arm
    assert trainer.select([tie, ft])["chosen"] == 1
    assert trainer.select([arm("from-scratch", 8., 11.9), ft])["chosen"] == 0
    with pytest.raises(SystemExit, match="different splits"):
        trainer.select([ft, dict(scratch, split_sha256="t")])


# ----------------------------------------------------------------------------------------------------------------------
# timing

def _timing_pair(video, curve, *, fps=25., meta=25, method="arm-2d"):
    return dict(status="evaluated", video_id=video, fps=fps, meta_fps=meta, hips_visible_share=1.,
                frame_count_difference=0,
                alignment=dict(method=method, lag=0, curve={str(k): [v, 100] for k, v in curve.items()}))


def test_pooled_lag_sums_cost_curves_per_video():
    lags = range(-3, 4)
    pairs = [_timing_pair("A", {k: abs(k - 3) + 5 for k in lags}), _timing_pair("A", {k: abs(k + 3) + 5 for k in lags}),
             _timing_pair("A", {k: abs(k) + 1 for k in lags}),                                   # pooled minimum: 0
             _timing_pair("B", {k: abs(k) for k in lags}), _timing_pair("B", {k: abs(k) for k in lags}),   # 2 pairs
             *[_timing_pair("C", {k: 10 - 3 * (k == 3) for k in lags}) for _ in range(3)],       # lag 3, gains 3 deg
             *[_timing_pair("D", {k: abs(k - 3) for k in lags}, fps=50., meta=25) for _ in range(3)],   # stream
             *[_timing_pair("E", {k: 5 - (k == 2) for k in lags}) for _ in range(3)],            # lag 2, gains 1 deg
             *[_timing_pair("F", {k: abs(k) for k in lags}, method="none") for _ in range(3)],
             *[_timing_pair("G", {k: 4 - (abs(k) == 1) for k in lags}) for _ in range(3)]]       # tie: -1
    pooled = gate.pooled_video_lags(pairs)
    assert set(pooled) == {"A", "C", "E", "G"}
    assert pooled["A"]["lag"] == 0 and pooled["C"]["lag"] == 3 and pooled["E"]["lag"] == 2 and pooled["G"]["lag"] == -1
    assert pooled["C"]["gain_over_lag0"] == 3. and pooled["E"]["gain_over_lag0"] == 1.
    rows = verdict.pooled_lag_rows(pairs)
    assert rows[0]["value"] == .5 and rows[0]["status"] == "fail"
    assert rows[1]["value"] == 1 and rows[1]["status"] == "fail" and list(rows[1]["detail"]["offenders"]) == ["C"]
    good = [p for p in pairs if p["video_id"] in ("A", "G")]
    assert [r["status"] for r in verdict.pooled_lag_rows(good)] == ["pass", "pass"]
    assert [r["status"] for r in verdict.pooled_lag_rows([])] == ["not-evaluable", "not-evaluable"]
    partial = [_timing_pair("H", {k: abs(k) for k in range(-6, 7) if k != 5}),
               _timing_pair("H", {k: abs(k) for k in range(-6, 7)}), _timing_pair("H", {k: abs(k) for k in (-6, 6, 0)})]
    assert set(gate.pooled_video_lags(partial)["H"]["curve"]) == {"-6", "0", "6"}      # lags common to every pair


def test_review_adversarial_cases():
    """The independent review's adversarial cases (findings 1, 4, 6), now refused, plus its passing timing checks."""
    import uwb_outside_frame_accuracy as ofa
    def cell(frames, survival, videos=5):
        return dict(frames=frames, survival=survival, kept=dict(videos=videos, n=int(frames * survival)))
    tight = dict(by_decision={d: {s: {"Forearm": {"outside": cell(1000, surv)}} for s in ("Left", "Right")}
                              for d, surv in (("head", 0.0), ("inside", 0.0))},
                 verdicts={"head": dict(admissible=True, failures=[])})
    tight["by_decision"]["mediapipe"] = {s: {"Forearm": {"outside": cell(1000, .018, videos=3)}} for s in ("Left", "Right")}
    rows = verdict.chest_up_head_rows(dict(regressors_sha256="b" * 64, tight=tight), "b" * 64)
    assert all(r["status"] == "not-evaluable" for r in rows)      # a head keeping nothing out of frame never passes
    cells = {s: {b: {n: dict(admitted=False) for n, *_ in ofa.BINS} for b in ofa.BONES} for s in ofa.SIDES}
    anchors = {s: {b: dict(n=100, median=10., p90=20.) for b in ofa.BONES} for s in ofa.SIDES}
    assert ofa.rule_verdict(cells, anchors, None)["admissible"] is True   # vacuous: hence the admitted-bin requirement
    curve = {"-1": [5., 100], "1": [6., 100]}                    # no lag 0, pooled lag within 1: not an offender
    pairs = [dict(status="evaluated", video_id="v", fps=30., meta_fps=30., alignment=dict(method="arm-2d", curve=curve))
             for _ in range(3)]
    assert [r["status"] for r in verdict.pooled_lag_rows(pairs)] == ["pass", "pass"]
    good = {str(k): [10. + abs(k - 3), 100] for k in range(-6, 7)}
    pairs = [dict(status="evaluated", video_id="v", fps=30., meta_fps=30., alignment=dict(method="arm-2d", curve=good))
             for _ in range(3)]
    pairs += [dict(status="evaluated", video_id="w", fps=30., meta_fps=60., alignment=dict(method="arm-2d", curve=good))
              for _ in range(5)]
    lags = gate.pooled_video_lags(pairs)
    assert set(lags) == {"v"} and lags["v"]["lag"] == 3 and lags["v"]["gain_over_lag0"] == 3.
    assert [r["status"] for r in verdict.pooled_lag_rows(pairs)] == ["fail", "fail"]


def test_downstream_lifts_with_the_conversion_rule_head(tmp_path):
    """Amendment 2: the downstream row reads lifter[head] states (``--visibility head``), recorded beside the gate's."""
    import uwb_downstream_check as ds
    from test_de_world_regressor_xy import xy_bundle
    from test_uwb_downstream_check import make_store as ds_store
    uwb_prefix, yt_prefix, *_ = ds_store(tmp_path)
    payload = with_head(xy_bundle(), {"Left": .5, "Right": .5}).save()
    (tmp_path / "h.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    store = ["--store", f"dir:{tmp_path}"]
    assert gate.main(store + ["--uwb-prefix", uwb_prefix, "--youtube-prefix", yt_prefix, "--workers", "1",
                              "--regressors", str(tmp_path / "h.pt"), "--regressors-sha256", sha, "--visibility",
                              "inside", "--secondary-visibility", "head", "--bootstrap", "20",
                              "--out", str(tmp_path / "g.json")]) == 0
    seen = []

    def translate(states):
        seen.append([s.channels["body"].inference_method for s in states])
        return ["a b c"] * len(states)
    common = store + ["--gate-report", str(tmp_path / "g.json"), "--regressors", str(tmp_path / "h.pt"),
                      "--regressors-sha256", sha, "--face", "none", "--device", "cpu", "--workers", "1",
                      "--bootstrap", "20", "--out", str(tmp_path / "d.json")]
    assert ds.main(common + ["--visibility", "head"], translate_fn=translate) == 0
    report = json.loads((tmp_path / "d.json").read_text())
    assert report["visibility"] == "head" and report["gate_visibility"] == "inside"
    assert all("validity head" in (m or "") for m in seen[0] if m)
    with pytest.raises(SystemExit, match="visibility rule"):
        ds.main(common + ["--visibility", "outside"], translate_fn=translate)


# ----------------------------------------------------------------------------------------------------------------------
# gate v2 runs and verdict

@pytest.fixture(scope="module")
def v2_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2")
    common = build_store(tmp / "s", frames=20) + ["--workers", "1"]
    split_sha = crafted_split(tmp / "split.json", train=("vid7",), val=("vid8",), held=("vid0", "vid1", "vid2"))
    plain = xy2_bundle()
    runs = {}
    for name, b, extra in (("v1", plain, []), ("v2", with_head(xy2_bundle(5), {"Left": .5, "Right": .5}),
                                                ["--secondary-visibility", "head"]),
                           ("v2-nohead", with_head(xy2_bundle(5), {"Left": .5, "Right": .5}), [])):
        payload = b.save()
        (tmp / f"{name}.pt").write_bytes(payload)
        out = tmp / f"{name}.json"
        assert gate.main(common + ["--regressors", str(tmp / f"{name}.pt"), "--regressors-sha256",
                                   hashlib.sha256(payload).hexdigest(), "--visibility", "inside",
                                   "--candidate-visibility", verdict.V2["candidates"], "--split-file",
                                   str(tmp / "split.json"), "--lag-curves", "--pairs-per-video", "12",
                                   "--bootstrap", "20", "--out", str(out)] + extra) == 0
        runs[name] = json.loads(out.read_text())
        runs[name + "_sha"] = hashlib.sha256(payload).hexdigest()
    runs.update(split=json.loads((tmp / "split.json").read_text()), split_sha=split_sha, tmp=tmp)
    return runs


def test_gate_v2_options_restrict_the_population_store_curves_and_keep_common_frames(v2_runs):
    v2 = v2_runs["v2"]
    assert v2["split"]["sha256"] == v2_runs["split_sha"] and v2["settings"]["lag_curves"] is True
    assert {p["video_id"] for p in v2["pairs"]} == {"vid0", "vid1", "vid2"}
    assert all(p["alignment"]["curve"] for p in v2["pairs"] if p["alignment"]["method"] == "arm-2d")
    assert "lifter[head]" in v2["summary"]["methods"] and "lifter[head]" not in v2_runs["v2-nohead"]["summary"]["methods"]
    assert v2["summary"]["methods"]["lifter[inside]"]["common"] == \
        v2_runs["v2-nohead"]["summary"]["methods"]["lifter[inside]"]["common"]        # head is outside the common frames
    assert {p["channel"] for p in v2["pairs"]} == {"ch0", "ch1", "ch2"}
    common = v2["summary"]["methods"]["lifter[inside]"]["common"]
    assert all("median_ci95_by_channel" in s for s in common.values())               # amendment 2: reported beside
    g = next(g for g, x in common.items() if x["median_ci95_by_channel"])      # synthetic arms leave the frame
    assert len(common[g]["median_ci95_by_channel"]) == 2
    assert all("median_ci95_by_channel" not in x for x in v2["summary"]["methods"]["lifter[learned]"]["common"].values())
    rows = {r["id"]: r for r in verdict.channel_ci_rows(v2["summary"]["methods"]["lifter[inside]"])}
    assert rows[f"channel_ci.median.{g}"]["gating"] is False
    assert rows[f"channel_ci.median.{g}"]["value"] == common[g]["median_ci95_by_channel"][1]


def _h2s(sha, medians, n=500, videos=10, primary="lifter[inside]"):
    return dict(regressors_sha256=sha, candidate_visibility=verdict.V2["candidates"].split(","),
                summary=dict(primary=primary, methods={primary: dict(common={
                    g: dict(n=n, videos=videos, median=m) for g, m in medians.items()})}))


def _tight(head, mp, inside, mp_frames=400, mp_videos=6, admissible=True, head_bins=("0-0.1",)):
    def cell(survival, frames=1000, videos=6, bins=()):
        forearm = dict(outside=dict(frames=frames, survival=survival, kept=dict(videos=videos)))
        for b in ("0-0.1", "0.1-0.3", "0.3-inf"):
            forearm[b] = dict(admitted=b in bins, unmeasured=False)
        return {side: dict(Forearm=forearm) for side in ("Left", "Right")}
    return dict(by_decision=dict(head=cell(head, bins=head_bins), inside=cell(inside),
                                 mediapipe=cell(mp, frames=mp_frames / mp if mp else 0, videos=mp_videos)),
                verdicts=dict(head=dict(admissible=admissible, failures=[])))


def test_gate_v2_verdict_rows_and_not_evaluable_handling(v2_runs, monkeypatch, tmp_path):
    monkeypatch.setattr(verdict, "BUNDLE_SHA256", v2_runs["v1_sha"])
    v2, sha = v2_runs["v2"], v2_runs["v2_sha"]
    rows = {r["id"]: r for r in verdict.build_rows_v2(v2, split=v2_runs["split"], split_sha=v2_runs["split_sha"],
                                                     bundle_sha=sha)}
    assert rows["held_out_only"]["status"] == "pass" and rows["split_disjoint"]["status"] == "pass"
    for rid in ("pairs_on_this_split", "bundle_trained_on_this_split", "v1_reference_on_identical_sample",
                "full_frame_forearm_median.Left<=v1", "chest_up_head_forearm_survival.Right",
                "how2sign_val_v2_minus_v1.forearms<=0.5", "relative.forearms<=1.6x", "downstream_back_translator"):
        assert rows[rid]["status"] == "not-evaluable" and rows[rid]["gating"], rid
    assert rows["bundle_is_lifter_v2"]["status"] == "pass" and rows["split_is_the_declared_split"]["status"] == "pass"
    assert rows["head_scored_as_secondary"]["status"] == "pass" and rows["lag_curves_stored"]["status"] == "pass"
    assert rows["pairs_per_video"]["status"] == "pass"
    assert rows["hand_agreement.Left>=0.88"]["note"] == "method lifter[head]" and not rows["hand_agreement.Left>=0.88"]["proposal_key"]
    assert not rows["inside.hand_agreement.Left>=0.88"]["gating"]
    assert "head_kept.median.forearms<=18.0" in rows and not rows["head_kept.median.forearms<=18.0"]["gating"]
    full = verdict.build_rows_v2(v2, split=v2_runs["split"], split_sha=v2_runs["split_sha"], bundle_sha=sha)
    check = verdict.cross_check(full, v2["summary"])
    assert not check["disagreements"] and check["compared"] >= 12
    assert verdict.overall(full, check) in ("NOT-EVALUABLE", "FAIL")
    # the v1 reference: identical sample -> pass; another sample -> not-evaluable
    ref = {r["id"]: r for r in verdict.v1_reference_rows(v2, v2_runs["v1"])}
    assert ref["v1_reference_on_identical_sample"]["status"] == "pass"
    assert ref["v1_reference.median.forearms"]["status"] == "reported"
    fewer = dict(v2_runs["v1"], pairs=v2_runs["v1"]["pairs"][1:])
    assert verdict.v1_reference_rows(v2, fewer)[0]["status"] == "not-evaluable"
    other = dict(v2_runs["v1"], regressors_sha256="f" * 64)
    assert "not xy2" in verdict.v1_reference_rows(v2, other)[0]["detail"]["reason"]
    # amendment 2: every gate setting (head, excluded videos, text matches, prefixes) and the evaluated pair sets
    for key, value in (("head", "affine"), ("exclude_videos", "other.txt"), ("min_text_matches", 4),
                       ("uwb_prefix", "elsewhere"), ("youtube_prefix", "elsewhere")):
        changed = dict(v2_runs["v1"], settings=dict(v2_runs["v1"]["settings"], **{key: value}))
        row = verdict.v1_reference_rows(v2, changed)[0]
        assert row["status"] == "not-evaluable" and key in row["detail"]["reason"], key
    flipped = json.loads(json.dumps(v2_runs["v1"]))
    flipped["pairs"][0]["status"] = "refused"
    row = verdict.v1_reference_rows(v2, flipped)[0]
    assert row["status"] == "not-evaluable" and "evaluated pairs" in row["detail"]["reason"]
    assert verdict.v1_reference_rows(v2, dict(v2_runs["v1"], excluded_videos=["x"]))[0]["status"] == "not-evaluable"
    # How2Sign non-inferiority
    v1m = dict(upper_arms=8., forearms=7., wrists=10., fingers=12.)
    rows = verdict.h2s_noninferiority_rows(_h2s(sha, {g: m + .4 for g, m in v1m.items()}), _h2s(v2_runs["v1_sha"], v1m),
                                          sha, "lifter[inside]", verdict.V2["candidates"])
    assert [r["status"] for r in rows] == ["pass"] * 4
    worse = dict(v1m, forearms=7.6)
    rows = verdict.h2s_noninferiority_rows(_h2s(sha, worse), _h2s(v2_runs["v1_sha"], v1m), sha, "lifter[inside]",
                                          verdict.V2["candidates"])
    assert rows[1]["status"] == "fail" and rows[0]["status"] == "pass"
    thin = verdict.h2s_noninferiority_rows(_h2s(sha, v1m, n=100), _h2s(v2_runs["v1_sha"], v1m), sha, "lifter[inside]",
                                          verdict.V2["candidates"])
    assert {r["status"] for r in thin} == {"not-evaluable"}
    swapped = verdict.h2s_noninferiority_rows(_h2s(v2_runs["v1_sha"], v1m), _h2s(v2_runs["v1_sha"], v1m), sha,
                                             "lifter[inside]", verdict.V2["candidates"])
    assert {r["status"] for r in swapped} == {"not-evaluable"}
    # truth anchor
    v1_anchor = dict(regressors_sha256=v2_runs["v1_sha"], anchor=dict(Left=dict(Forearm=dict(n=460, median=28.24)),
                                                                     Right=dict(Forearm=dict(n=474, median=29.07))))
    anchor = dict(regressors_sha256=sha, anchor=dict(Left=dict(Forearm=dict(n=900, median=28.24)),
                                                     Right=dict(Forearm=dict(n=900, median=29.08))))
    rows = verdict.truth_anchor_rows(anchor, sha, v1_anchor)
    assert [r["status"] for r in rows] == ["pass", "fail"] and [r["bar"] for r in rows] == [28.24, 29.07]
    assert {r["status"] for r in verdict.truth_anchor_rows(anchor, sha)} == {"not-evaluable"}          # no v1 report
    assert {r["status"] for r in verdict.truth_anchor_rows(anchor, "0" * 64, v1_anchor)} == {"not-evaluable"}
    # review finding 4: xy2 passes its own anchor (the rounded 28.2 bar failed it on the left)
    assert {r["status"] for r in verdict.truth_anchor_rows(v1_anchor, v2_runs["v1_sha"], v1_anchor)} == {"pass"}
    # chest-up survival of head: the MediaPipe branch, then the inside-and-admissible branch
    outside = dict(regressors_sha256=sha, tight=_tight(head=.5, mp=.6, inside=0.))
    assert [r["status"] for r in verdict.chest_up_head_rows(outside, sha)] == ["pass", "pass"]   # 0.833 >= 0.8
    outside = dict(regressors_sha256=sha, tight=_tight(head=.4, mp=.6, inside=0.))
    assert [r["status"] for r in verdict.chest_up_head_rows(outside, sha)] == ["fail", "fail"]
    outside = dict(regressors_sha256=sha, tight=_tight(head=.3, mp=.01, inside=.1, mp_frames=18, mp_videos=2))
    rows = verdict.chest_up_head_rows(outside, sha)
    assert [r["status"] for r in rows] == ["pass", "pass"] and "inside" in rows[0]["detail"]["branch"]
    outside = dict(regressors_sha256=sha, tight=_tight(head=.3, mp=.01, inside=.1, mp_frames=18, admissible=False))
    assert [r["status"] for r in verdict.chest_up_head_rows(outside, sha)] == ["fail", "fail"]
    # amendment 2: equal to inside, or no admitted measured outside bin -> not-evaluable, never pass
    for tight in (_tight(head=.1, mp=.01, inside=.1, mp_frames=18), _tight(head=.3, mp=.01, inside=.1, mp_frames=18,
                                                                          head_bins=())):
        rows = verdict.chest_up_head_rows(dict(regressors_sha256=sha, tight=tight), sha)
        assert [r["status"] for r in rows] == ["not-evaluable"] * 2 and rows[0]["detail"]["reason"]
    assert {r["status"] for r in verdict.chest_up_head_rows(dict(regressors_sha256=sha), sha)} == {"not-evaluable"}
    # downstream: the row reads lifter[head] states (amendment 2); an inside read is not-evaluable
    for rule, ok in (("inside", False), ("head", True)):
        ds_report = dict(gate_report_sha256="g", regressors_sha256=sha, visibility=rule, bootstrap=2000)
        row = verdict.downstream_rows(ds_report, "g", v2, expected_visibility="head")[0]
        assert ("visibility rule" in (row["detail"]["reason"] or "")) is not ok
    # population: a gated video of the training lists fails; provenance of pairs and bundle
    bad_split = json.loads(json.dumps(v2_runs["split"]))
    bad_split["held_out"]["videos"].remove("vid2")
    rows = {r["id"]: r for r in verdict.population_rows(v2, bad_split, "s", dict(split_sha256="s"),
                                                        dict(split_sha256="s", bundle_sha256=sha), sha)}
    assert rows["held_out_only"]["status"] == "fail" and rows["pairs_on_this_split"]["status"] == "pass"
    assert rows["bundle_trained_on_this_split"]["status"] == "pass"
    rows = {r["id"]: r for r in verdict.population_rows(v2, v2_runs["split"], "s", dict(split_sha256="t"),
                                                        dict(split_sha256="s", bundle_sha256="x"), sha)}
    assert rows["pairs_on_this_split"]["status"] == "fail" and rows["bundle_trained_on_this_split"]["status"] == "fail"


def test_gate_v2_main_writes_the_v2_verdict_and_v1_mode_refuses_v2_inputs(v2_runs, monkeypatch, tmp_path):
    monkeypatch.setattr(verdict, "BUNDLE_SHA256", v2_runs["v1_sha"])
    t = v2_runs["tmp"]
    out = tmp_path / "v.json"
    assert verdict.main(["--store", "none", "--gate-version", "2", "--gate-report", str(t / "v2.json"),
                         "--v1-reference", str(t / "v1.json"), "--split-file", str(t / "split.json"),
                         "--bundle-sha256", v2_runs["v2_sha"], "--out", str(out),
                         "--markdown", str(tmp_path / "v.md")]) == 0
    report = json.loads(out.read_text())
    assert report["gate_version"] == 2 and report["verdict"] in ("NOT-EVALUABLE", "FAIL")
    assert report["inputs"]["split_file"][1] == v2_runs["split_sha"] and "11:12 UTC" in report["declaration"]
    rows = {r["id"]: r for r in report["rows"]}
    assert rows["v1_reference_on_identical_sample"]["status"] == "pass"
    assert "v1_reference.pooled_lag_within_1_share>=0.9" in rows
    assert "**Overall:" in (tmp_path / "v.md").read_text()
    (tmp_path / "of1.json").write_text("{}")
    with pytest.raises(SystemExit, match="pinned"):
        verdict.main(["--store", "none", "--gate-version", "2", "--gate-report", str(t / "v2.json"),
                      "--outside-frame-v1", str(tmp_path / "of1.json"), "--out", str(out)])
    with pytest.raises(SystemExit, match="gate v2 input"):
        verdict.main(["--store", "none", "--gate-report", str(t / "v2.json"), "--split-file", str(t / "split.json"),
                      "--out", str(out)])


def test_outside_frame_accepts_head_and_adds_a_tight_summary_only_when_asked():
    import uwb_outside_frame_accuracy as of
    rows = []
    for k in range(12):
        for side in ("Left", "Right"):
            for bone in ("UpperArm", "Forearm"):
                for t in range(30):
                    d = 0. if t < 10 else .05
                    rows.append(dict(video=f"v{k % 6}", clip=f"c{k}", kind="tight" if k % 2 else "semlex", input="x",
                                     side=side, bone=bone, distance=d, bin=of.bin_of(d), full_view=False,
                                     mediapipe=t < 20, **{"rule:head": t % 2 == 0, "rule:inside": d == 0},
                                     error=float(t % 7)))
    summary = of.decision_summary([r for r in rows if r["kind"] == "tight"], "rule:head", 0, .05, outside=True)
    cell = summary["Left"]["Forearm"]["outside"]
    assert cell["frames"] == 6 * 20 and cell["survival"] == .5 and cell["admitted"] is False
    assert "outside" not in of.decision_summary(rows, "rule:head", 0, .05)["Left"]["Forearm"]
    assert "head" in of.CONSERVATIVE_ORDER
