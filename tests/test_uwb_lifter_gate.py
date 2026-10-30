"""UWB lifter gate: frame-rate estimation, cue pairing, temporal alignment, bone metrics, mirror and hand checks."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import uwb_lifter_gate as gate  # noqa: E402
from semlex_synthetic import rotation  # noqa: E402
from uwb_synthetic import TrueWorldXY, uwb_clip_json  # noqa: E402

from signtranslator.data_engineering import uwb_keypoints as uk  # noqa: E402
from signtranslator.data_engineering.video_landmarks import ClipLandmarks, canonical_from_landmarks  # noqa: E402


def cues_and_clips(fps, *, n=40, span=600., seed=0, rounding=math.floor, texts=True):
    """Our cues (seconds) and UWB clips (frames at the true ``fps``, ``first = floor(start fps)``)."""
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, span, n))
    cues, clips = [], []
    for k, s in enumerate(starts):
        e = s + rng.uniform(1.5, 5.)
        cues.append(dict(start=round(float(s), 3), end=round(float(e), 3), text=f"Sentence number {k}.",
                         sample_id=f"v-{k:04d}"))
        clips.append(dict(clip_id=f"v.{rounding(s * fps):06d}-{rounding(e * fps):06d}", first=int(rounding(s * fps)),
                          last=int(rounding(e * fps)), text=f"sentence  number {k}" if texts else f"other {k}"))
    return cues, clips


@pytest.mark.parametrize("fps", [23.976, 25., 29.97, 30., 15.])
def test_rates_are_recovered_and_snapped_from_text_matched_cues(fps):
    cues, clips = cues_and_clips(fps)
    estimate = gate.estimate_fps(clips, cues)
    assert estimate["method"] == "text-matched-fit" and estimate["fps"] == fps
    assert estimate["snapped_to"] == f"{fps:g}" and estimate["residual_median"] < .6
    assert estimate["offset_frames"] == pytest.approx(-.5, abs=.2)              # floor: half a frame early


def test_29_97_and_30_are_told_apart_and_outliers_dropped():
    cues, clips = cues_and_clips(29.97, n=60, span=1200.)
    clips[3] = dict(clips[3], first=clips[3]["first"] + 40, last=clips[3]["last"] + 40)   # a misaligned clip
    estimate = gate.estimate_fps(clips, cues)
    assert estimate["fps"] == 29.97 and estimate["outliers"] == 2
    vfr = gate.estimate_fps(*reversed(cues_and_clips(29.88, n=60, span=1200.)))
    assert vfr["snapped_to"] is None and vfr["fps"] == pytest.approx(29.88, abs=5e-3)    # an average rate stays raw


def test_without_text_matches_the_time_grid_decides_and_too_little_evidence_gives_none():
    cues, clips = cues_and_clips(25., texts=False)
    estimate = gate.estimate_fps(clips, cues)
    assert estimate["method"] == "time-grid" and estimate["fps"] == 25. and estimate["text_matches"] == 0
    few_cues, few_clips = cues_and_clips(25., n=2)
    assert gate.estimate_fps(few_clips, few_cues)["fps"] is None
    repeated = [dict(c, text="Thank you.") for c in cues]                         # duplicate texts are not anchors
    assert gate.text_pairs([dict(c, text="thank you") for c in clips], repeated).shape == (0, 2)


def test_a_robust_start_survives_many_mismatched_cues_and_implausible_fits_are_refused():
    cues, clips = cues_and_clips(25., n=60, span=1200.)
    rng = np.random.default_rng(5)
    for k in rng.choice(60, 20, replace=False):                                 # a third of the clips misaligned
        shift = int(rng.integers(30, 3000))
        clips[k] = dict(clips[k], first=clips[k]["first"] + shift, last=clips[k]["last"] + shift)
    estimate = gate.estimate_fps(clips, cues)
    assert estimate["fps"] == 25. and estimate["outliers"] == 40 and estimate["theil_sen_start"] == pytest.approx(25, abs=.05)
    late_heavy = [dict(c, first=c["first"] + 4 * c["first"], last=c["last"] + 4 * c["first"]) if k % 3 == 0 else c
                  for k, c in enumerate(cues_and_clips(25., n=60, span=1200.)[1])]  # a third on a 5x-rate line
    rows = gate.text_pairs(late_heavy, cues_and_clips(25., n=60, span=1200.)[0])
    plain = np.linalg.lstsq(np.c_[rows[:, 0], np.ones(len(rows))], rows[:, 1], rcond=None)[0][0]
    assert abs(plain - 25.) > 5.                                                 # plain least squares is dragged off
    assert gate.estimate_fps(late_heavy, cues_and_clips(25., n=60, span=1200.)[0])["fps"] == 25.
    six = gate.estimate_fps(*reversed(cues_and_clips(6., span=2000.)))
    assert six["fps"] is None and "implausible" in six["refused"]
    late = gate.estimate_fps(*reversed(cues_and_clips(25., rounding=lambda v: math.floor(v) + 2)))
    assert late["fps"] is None and "offset" in late["refused"]


def test_the_time_grid_needs_half_the_clips():
    cues, clips = cues_and_clips(25., n=40, texts=False)
    for k in range(0, 40, 5):
        clips[k] = dict(clips[k], first=clips[k]["first"] + 500, last=clips[k]["last"] + 500)
    assert gate.estimate_fps(clips, cues)["fps"] == 25.                         # 32 of 40 land
    for k in range(40):
        if k % 5 in (1, 2, 3):
            clips[k] = dict(clips[k], first=clips[k]["first"] + 700, last=clips[k]["last"] + 700)
    sparse = gate.estimate_fps(clips, cues)                                       # 8 of 40 land
    assert sparse["fps"] is None and sparse["grid"]["hits"] == 8 and sparse["refused"]


def test_pairing_is_one_to_one_by_overlap_and_a_wrong_rate_pairs_little():
    cues, clips = cues_and_clips(25., n=30, span=900.)
    pairs = gate.pair_clips(clips, cues, 25.)
    assert len(pairs) == 30 and all(c["text"].endswith(q["text"].split()[-1].rstrip(".")) for c, q, _ in pairs)
    assert len({q["sample_id"] for _, q, _ in pairs}) == 30 and min(iou for *_, iou in pairs) > .8
    wrong = gate.pair_clips(clips, cues, 30.)                                     # 25 fps frames read as 30 fps
    assert len(wrong) < 10


def test_frame_map_matches_within_half_a_frame_and_shifts_with_the_lag():
    fps = 25.
    times = (100 + np.arange(20)) / fps + .004                                    # our clock, 4 ms off the frames
    i, j = gate.frame_map(times, 100, 20, fps, 0)
    assert (i == j).all() and len(i) == 20
    i, j = gate.frame_map(times, 100, 20, fps, 3)
    assert (i == j + 3).all() and len(i) == 17
    i, j = gate.frame_map(times[::2], 100, 20, fps, 0)                            # our clock at half the rate
    assert (j == 2 * i).all() and len(i) == 10


def _arm_motion(frames):
    """(T, 33, 3) y-up body with the forearms sweeping in the image plane, and the image (pixels, y down)."""
    t = np.arange(frames)
    body = np.zeros((frames, 33, 3))
    body[:, 11], body[:, 12] = [.2, .45, 0.], [-.2, .45, 0.]
    body[:, 13], body[:, 14] = [.3, .2, .05], [-.3, .2, .05]
    angle = .25 * t
    body[:, 15] = body[:, 13] + .25 * np.stack((np.sin(angle), np.cos(angle), .2 * np.ones(frames)), -1)
    body[:, 16] = body[:, 14] + .25 * np.stack((-np.sin(1.3 * angle), np.cos(1.3 * angle), .2 * np.ones(frames)), -1)
    body[:, 23], body[:, 24] = [.1, -.1, 0.], [-.1, -.1, 0.]
    pixels = np.stack((640 + 800 * body[..., 0], 360 - 800 * body[..., 1]), -1)
    return body, pixels


def _state(body, times, hands=None, visible=None):
    frames = len(times)
    hands = hands or {s: (np.zeros((frames, 21, 3)), np.zeros(frames, bool)) for s in ("Left", "Right")}
    clip = ClipLandmarks(times=times, pose_world=np.zeros((frames, 33, 3)), pose_visibility=np.ones((frames, 33)),
                         pose_image=np.zeros((frames, 33, 2)), pose_found=np.ones(frames, bool),
                         hands_world=np.zeros((frames, 2, 21, 3)), hands_image=np.zeros((frames, 2, 21, 2)),
                         hands_found=np.zeros((frames, 2), bool), blendshapes=np.zeros((frames, 0)),
                         blendshape_names=(), face_matrix=np.tile(np.eye(4), (frames, 1, 1)),
                         face_found=np.zeros(frames, bool))
    state, _ = canonical_from_landmarks(clip, sample_id="s", source_id="s", source_sha256="a" * 64, clock_id="c",
                                        inference_method="m",
                                        points=dict(world=body, visible=np.ones((frames, 33), bool) if visible is None
                                                    else visible, hands=hands))
    return state


def test_the_lag_is_found_from_the_image_plane_arms_and_absent_arms_give_no_lag():
    fps, frames, first = 25., 60, 200
    body, pixels = _arm_motion(frames + 10)
    times = (first + np.arange(frames)) / fps
    ours = gate.state_items(_state(body[5:5 + frames], times))
    uwb2d = gate.image_arm_directions(pixels[2:2 + frames], np.ones(frames, bool))   # UWB frame j shows our j - 3
    chosen = gate.choose_lag(times, first, frames, fps, ours, uwb2d, 6)
    assert chosen["method"] == "arm-2d" and chosen["lag"] == -3 and chosen["score"] < 1e-6
    assert gate.choose_lag(times, first, frames, fps, ours, uwb2d, 6, margin=.5)["lag"] == -3   # a real gain stays
    held = gate.choose_lag(times, first, frames, fps, ours, uwb2d, 6, margin=1e3)
    assert held["lag"] == 0 and held["best_lag"] == -3 and held["score"] == held["lag0_score"] > 1.
    hidden = np.ones((frames, 33), bool)
    hidden[:, 11:17] = False
    none = gate.choose_lag(times, first, frames, fps, gate.state_items(_state(body[:frames], times, visible=hidden)),
                           uwb2d, 6)
    assert none == dict(lag=0, method="none", score=None)


def test_identical_states_have_zero_error_and_a_turned_finger_and_a_mirror_are_seen():
    from semlex_synthetic import body_y_up, hand_y_up
    frames = 10
    body = body_y_up(frames)
    hands = {"Left": hand_y_up(body[:, 15], 2), "Right": hand_y_up(body[:, 16], 3)}
    times = np.arange(frames) / 30.
    packed = {s: (h - h[:, :1], np.ones(frames, bool)) for s, h in hands.items()}
    a = gate.state_items(_state(body, times, packed))
    i = j = np.arange(frames)
    errors = gate.compare(a, a, i, j)
    assert set(errors) == set(gate.GROUPS) and all(max(v) < 1e-4 for v in errors.values())
    assert len(errors["fingers"]) == frames * 30 and len(errors["upper_arms"]) == frames * 2
    bent = {s: (h.copy(), ok) for s, (h, ok) in packed.items()}
    bent["Left"][0][:, 8] = bent["Left"][0][:, 7] + (bent["Left"][0][:, 8] - bent["Left"][0][:, 7]) @ rotation((1, 0, 0), 30.).T
    b = gate.state_items(_state(body, times, bent))
    errors = gate.compare(a, b, i, j)
    assert 29 < max(errors["fingers"]) < 31 and max(errors["upper_arms"]) < 1e-4
    assert np.sum(np.asarray(errors["fingers"]) > 1) == frames                     # only the index tip bone
    mirrored = body * [-1., 1., 1.]
    for l, r in ((11, 12), (13, 14), (15, 16), (23, 24)):
        mirrored[:, [l, r]] = mirrored[:, [r, l]]
    m = gate.state_items(_state(mirrored, times))
    plain = gate.compare(a, m, i, j)
    flipped = gate.compare(a, gate.mirrored_arms(m), i, j)
    assert np.median(plain["forearms"]) > 10 and max(flipped["forearms"] + flipped["upper_arms"]) < 1e-4


def test_hand_agreement_counts_and_lag_drift():
    frames = 8
    times = np.arange(frames) / 30.
    from semlex_synthetic import body_y_up, hand_y_up
    body = body_y_up(frames)
    hand = hand_y_up(body[:, 15], 2)
    ours = _state(body, times, {"Left": (hand - hand[:, :1], np.arange(frames) < 6),
                                "Right": (hand - hand[:, :1], np.zeros(frames, bool))})
    theirs = _state(body, times, {"Left": (hand - hand[:, :1], np.arange(frames) >= 2),
                                  "Right": (hand - hand[:, :1], np.zeros(frames, bool))})
    strata = {"first_half": np.arange(frames) < 4}
    counts = gate.hand_agreement(ours, theirs, np.arange(frames), np.arange(frames), strata)
    assert counts["all"]["Left"] == dict(both=4, ours_only=2, uwb_only=2, neither=0)
    assert counts["all"]["Right"] == dict(both=0, ours_only=0, uwb_only=0, neither=8)
    assert counts["first_half"]["Left"] == dict(both=2, ours_only=2, uwb_only=0, neither=0)
    summary = gate.agreement_summary(counts["all"]["Left"])
    assert summary["agreement"] == .5 and summary["uwb_only_rate"] == .25 and summary["frames"] == 8
    assert summary["kappa"] == pytest.approx((.5 - .625) / .375, abs=1e-4)      # below chance: pe = .75^2 + .25^2
    assert gate.kappa(dict(both=5, ours_only=0, uwb_only=0, neither=5)) == 1.
    assert gate.kappa(dict(both=0, ours_only=0, uwb_only=0, neither=0)) is None
    survival = gate.forearm_survival(theirs, np.arange(frames), strata)
    assert survival["all"]["Left"] == [8, 8] and survival["first_half"]["Right"] == [4, 4]
    visible = np.ones((frames, 33), bool)
    visible[4:, 16] = False                                       # their right wrist unseen on the last 4 frames
    arms = gate.arm_agreement(ours, _state(body, times, visible=visible), np.arange(frames), np.arange(frames))
    assert arms["Right"] == dict(both=4, ours_only=4, uwb_only=0, neither=0) and arms["Left"]["both"] == 8
    no_hips = np.ones((frames, 33), bool)
    no_hips[:, 23:25] = False                                     # our chest invalid: nothing is compared
    assert sum(gate.arm_agreement(_state(body, times, visible=no_hips), ours, np.arange(frames),
                                  np.arange(frames))["Left"].values()) == 0
    assert gate.lag_drift([(0., 0), (60., 2), (120., 4)]) == pytest.approx(2.)
    assert gate.lag_drift([(0., 0), (60., 2)]) is None


def test_evaluate_pair_end_to_end_recovers_the_lag_and_ranks_the_methods():
    """Ours: the true points on our clock; UWB: the same motion's image, 2 frames late, lifted by a true-world stub
    (learned) and by the image plane (pseudo)."""
    fps, frames, first = 30., 24, 300
    data, body, hands = uwb_clip_json(size=(1280, 720), frames=frames + 4, seed=4)
    times = (first + np.arange(frames)) / fps + .003
    packed = {s: (h[2:2 + frames] - h[2:2 + frames, :1], np.ones(frames, bool)) for s, h in hands.items()}
    ours = _state(body[2:2 + frames], times, packed)
    clip = uk.parse_clip(data, f"vid.{first:06d}-{first + frames + 3:06d}")
    common = dict(source_sha256="b" * 64, fps_source="estimated")
    states = {"learned": uk.convert_clip(clip, fps, regressors=TrueWorldXY(body, hands), **common)[0],
              "pseudo": uk.convert_clip(clip, fps, world="pseudo", visibility="inside", **common)[0]}
    result = gate.evaluate_pair(ours, states, clip, fps, 6, cap=50)
    assert result["alignment"]["lag"] == -2 and result["alignment"]["method"] == "arm-2d"
    learned, pseudo = result["methods"]["learned"], result["methods"]["pseudo"]
    for group in ("upper_arms", "forearms", "fingers"):
        assert np.max(learned["own"][group]) < 1e-3, group
        assert len(learned["own"][group]) <= 50 and learned["own"][group].dtype == np.float32
    assert np.median(pseudo["own"]["fingers"]) > 1.
    # the elbows lie below this frame: the inside-image rule drops the pseudo arms, the learned head keeps them
    assert len(pseudo["own"].get("upper_arms", ())) == 0 and len(learned["own"]["upper_arms"]) == 2 * frames
    assert result["aligned_frames"] == frames and len(learned["own"]["fingers"]) == 50          # capped
    assert len(pseudo["common"].get("fingers", ())) == len(learned["common"]["fingers"]) > 0
    assert learned["hands"]["all"]["Left"]["both"] == frames and np.median(learned["mirror_arms"]) > 5.
    # the synthetic hips lie below this frame: every aligned frame is in the hips-not-visible stratum
    assert result["hips_visible_share"] == 0. and learned["hands"]["hips_not_visible"]["Left"]["both"] == frames
    assert learned["forearm_survival"]["all"]["Left"] == [frames + 4, frames + 4]
    assert pseudo["forearm_survival"]["all"]["Left"][0] == 0                     # inside-image drops every forearm
    row = dict(result, status="evaluated", video_id="vid", cue_start=10., fps=fps, meta_fps=30,
               inputs=dict(pose=np.full((4, 54), 5., np.float16), Left=np.zeros((4, 40), np.float16)))
    summary = gate.summarise([row, dict(status="refused", reason="Refusal: x")], primary="learned", resamples=50,
                             input_names=gate.names_for([row]))
    assert summary["pairs"] == 1 and summary["methods"]["learned"]["common"]["fingers"]["n"] > 0
    assert summary["methods"]["learned"]["own"]["forearms"]["median"] < 1e-2
    assert summary["methods"]["learned"]["mirror_check"]["mirror_better"] == 0
    assert summary["alignment"]["lags"] == {"-2": 1} and summary["refusals"] == {"Refusal: x": 1}
    assert summary["alignment"]["lags_by_frame_count_difference"] == {"0": {"-2": 1}}
    assert summary["strata"]["hips"] == {"not_visible": summary["strata"]["hips"]["not_visible"]}
    assert summary["strata"]["fps30"]["yes"]["pairs"] == 1 and summary["strata"]["stream_match"]["yes"]["pairs"] == 1
    assert summary["inputs"]["pose"]["max_p99_abs_z"] == 5. and len(summary["inputs"]["pose"]["inputs_over_4"]) == 54
    assert summary["inputs"]["pose"]["inputs_over_4"][0] == "p0.x" and summary["inputs"]["Left"]["max_p99_abs_z"] == 0.
    check = summary["proposal_check"]
    assert check["inputs_p99_abs_z<=4.0"] is False and check["median.fingers<=16.0"] is True
    assert check["hand_agreement.Left>=0.88"] is True and check["mirror_better_share<=0.03"] is True


def test_main_estimates_rates_pairs_cues_and_evaluates_from_a_store(tmp_path):
    """One video in a directory store: our report and states, UWB translations and a zip; fps-only and full runs."""
    import hashlib
    import json
    import zipfile
    from test_de_world_regressor_xy import xy_bundle
    fps, video = 25., "vidX"
    uwb_prefix, yt_prefix = "acq/uwb", "acq/yt"
    (tmp_path / uwb_prefix).mkdir(parents=True)
    (tmp_path / yt_prefix / "videos").mkdir(parents=True)
    (tmp_path / yt_prefix / "states").mkdir(parents=True)
    segments, order, translations = [], [], {}
    with zipfile.ZipFile(tmp_path / uwb_prefix / "raw_keypoints_1.zip", "w") as z:
        for k, start in enumerate((2., 6., 11., 15.)):
            frames = 20
            first = math.floor(start * fps)
            end = (first + frames - 1) / fps
            clip_id = f"{video}.{first:06d}-{first + frames - 1:06d}"
            data, body, hands = uwb_clip_json(size=(1280, 720), frames=frames, seed=k + 1)
            z.writestr(f"{clip_id}.json", json.dumps(data))
            order.append(clip_id)
            translations[clip_id] = {"translation": f"Cue number {k}"}
            times = (first + np.arange(frames)) / fps + .002
            state = _state(body, times, {s: (h - h[:, :1], np.ones(frames, bool)) for s, h in hands.items()})
            key = f"{yt_prefix}/states/{video}-{k:04d}.npz"
            digest = state.save(tmp_path / key)
            segments.append(dict(cue=k, sample_id=f"{video}-{k:04d}", start=start + .01, end=end,
                                 text=f"Cue number {k}.", state_key=key, state_sha256=digest))
    translations["clip_order"] = order
    (tmp_path / uwb_prefix / "YT.translations.all.json").write_text(json.dumps({video: translations}))
    (tmp_path / yt_prefix / "videos" / f"{video}.json").write_text(json.dumps(
        dict(video_id=video, status="converted", meta=dict(fps=25), segments=segments)))
    payload = xy_bundle().save()
    (tmp_path / "xy.pt").write_bytes(payload)
    common = ["--store", f"dir:{tmp_path}", "--uwb-prefix", uwb_prefix, "--youtube-prefix", yt_prefix,
              "--workers", "1"]
    out, table = tmp_path / "fps.json", tmp_path / "table.json"
    assert gate.main(common + ["--fps-only", "--out", str(out), "--fps-table-out", str(table)]) == 0
    report = json.loads(out.read_text())
    assert report["fps"]["videos"][video]["fps"] == fps and report["candidate_pairs"] == 4
    assert report["fps"]["summary"]["agree_with_meta_within_ntsc"] == 1
    written = json.loads(table.read_text())
    assert written["fps"] == {video: fps} and written["schema"] == "uwb-fps-table-v1"
    overlap = tmp_path / "overlap.json"
    assert gate.main(common + ["--fps-only", "--out", str(out), "--overlap-table-out", str(overlap)]) == 0
    table = json.loads(overlap.read_text())
    assert table["schema"] == "uwb-overlap-table-v1" and table["videos"][video]["dedup"] == "cue"
    assert table["videos"][video]["partition"] in ("train", "monitor") and len(table["clips"]) == 4
    assert {v["our_sample_id"] for v in table["clips"].values()} == {f"{video}-{k:04d}" for k in range(4)}
    with pytest.raises(SystemExit, match="regressors"):
        gate.main(common + ["--out", str(out)])
    bundle = ["--regressors", str(tmp_path / "xy.pt"), "--regressors-sha256", hashlib.sha256(payload).hexdigest()]
    with pytest.raises(SystemExit, match="declared rule"):
        gate.main(common + bundle + ["--out", str(out)])
    with pytest.raises(SystemExit, match="visibility rule"):
        gate.main(common + bundle + ["--visibility", "outside", "--out", str(out)])
    with pytest.raises(SystemExit, match="every video"):
        gate.main(common + bundle + ["--visibility", "inside", "--videos", "1", "--overlap-table-out", str(overlap),
                                     "--out", str(out)])
    assert gate.main(common + bundle + ["--visibility", "margin:0.25", "--bootstrap", "20", "--out", str(out),
                                        "--fps-table-out", str(table_path := tmp_path / "t2.json")]) == 0
    report = json.loads(out.read_text())
    summary = report["summary"]
    assert summary["pairs"] == 4 and summary["videos"] == 1 and summary["refusals"] == {}
    assert summary["primary"] == "lifter[margin:0.25]"
    assert set(summary["methods"]) == {"lifter[margin:0.25]", "lifter[learned]", "lifter[inside]", "pseudo"}
    assert summary["methods"]["pseudo"]["hand_valid"]["all.Left"]["agreement"] == 1.
    assert summary["inputs"]["pose"]["frames"] > 0 and "proposal_check" in summary
    assert all(p["status"] == "evaluated" and p["fps"] == fps for p in report["pairs"])
    assert json.loads(table_path.read_text())["drift_checked"] is True
    excluded = tmp_path / "exclude.txt"
    excluded.write_text(f"# smoke videos\n{video}\n")
    assert gate.main(common + bundle + ["--visibility", "inside", "--exclude-videos", str(excluded),
                                        "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["summary"]["pairs"] == 0 and report["excluded_videos"] == [video]
