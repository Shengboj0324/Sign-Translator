"""UWB gate downstream row: pair sampling and reconstruction, the converter path, windows, same references, blank
durations, BLEU from segment statistics, the video-clustered bootstrap and the row's verdict; How2Sign-val medians."""
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import uwb_downstream_check as ds  # noqa: E402
import uwb_lifter_gate as gate  # noqa: E402
from test_uwb_lifter_gate import _state  # noqa: E402
from uwb_synthetic import TrueWorldXY, uwb_clip_json  # noqa: E402

from signtranslator.generation.motion_to_text import corpus_bleu  # noqa: E402

VIDEOS = ("vidX", "vidY")
FPS = 25.


def make_store(tmp_path):
    """Two videos in a directory store: our reports and states, the UWB zip and translations (all and train)."""
    uwb_prefix, yt_prefix = "acq/uwb", "acq/yt"
    (tmp_path / uwb_prefix).mkdir(parents=True)
    (tmp_path / yt_prefix / "videos").mkdir(parents=True)
    (tmp_path / yt_prefix / "states").mkdir(parents=True)
    translations, segments_of = {}, {}
    with zipfile.ZipFile(tmp_path / uwb_prefix / "raw_keypoints_1.zip", "w") as z:
        for n, video in enumerate(VIDEOS):
            entry, order, segments = {}, [], []
            for k, start in enumerate((2., 6., 11., 15.)):
                frames = 20
                first = math.floor(start * FPS)
                clip_id = f"{video}.{first:06d}-{first + frames - 1:06d}"
                data, body, hands = uwb_clip_json(size=(1280, 720), frames=frames, seed=10 * n + k + 1)
                z.writestr(f"{clip_id}.json", json.dumps(data))
                order.append(clip_id)
                entry[clip_id] = {"translation": f"Cue {video} number {k}"}
                times = (first + np.arange(frames)) / FPS + .002
                state = _state(body, times, {s: (h - h[:, :1], np.ones(frames, bool)) for s, h in hands.items()})
                key = f"{yt_prefix}/states/{video}-{k:04d}.npz"
                digest = state.save(tmp_path / key)
                segments.append(dict(cue=k, sample_id=f"{video}-{k:04d}", start=start + .01,
                                     end=(first + frames - 1) / FPS, text=f"Cue {video} number {k}.", state_key=key,
                                     state_sha256=digest))
            entry["clip_order"] = order
            translations[video] = entry
            segments_of[video] = segments
            (tmp_path / yt_prefix / "videos" / f"{video}.json").write_text(json.dumps(
                dict(video_id=video, status="converted", meta=dict(fps=25), segments=segments)))
    (tmp_path / uwb_prefix / "YT.translations.all.json").write_text(json.dumps(translations))
    (tmp_path / uwb_prefix / "YT.translations.train.json").write_text(json.dumps({"vidX": translations["vidX"]}))
    (tmp_path / uwb_prefix / "YT.translations.dev.json").write_text(json.dumps({"vidY": translations["vidY"]}))
    return uwb_prefix, yt_prefix, translations, segments_of


@pytest.fixture(scope="module")
def gate_run(tmp_path_factory):
    """A gate report over the two videos (the declared rule, the xy test bundle)."""
    from test_de_world_regressor_xy import xy_bundle
    tmp_path = tmp_path_factory.mktemp("downstream")
    uwb_prefix, yt_prefix, translations, segments = make_store(tmp_path)
    payload = xy_bundle().save()
    (tmp_path / "xy.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    out = tmp_path / "gate.json"
    assert gate.main(["--store", f"dir:{tmp_path}", "--uwb-prefix", uwb_prefix, "--youtube-prefix", yt_prefix,
                      "--workers", "1", "--regressors", str(tmp_path / "xy.pt"), "--regressors-sha256", sha,
                      "--visibility", "inside", "--bootstrap", "20", "--out", str(out)]) == 0
    return dict(path=tmp_path, report=json.loads(out.read_text()), gate=out, sha=sha, segments=segments,
                translations=translations, uwb_prefix=uwb_prefix, yt_prefix=yt_prefix)


# ----------------------------------------------------------------------------------------------------------------------

def test_sampling_takes_one_pair_per_video_first_and_is_seeded():
    pairs = [dict(status="evaluated", video_id=f"v{v}", clip_id=f"v{v}.{k:06d}-000001") for v in range(6) for k in range(4)]
    pairs.append(dict(status="refused", video_id="v0", clip_id="bad"))
    assert len(ds.sample_pairs(pairs, 0)) == 24 and len(ds.sample_pairs(pairs, 30)) == 24     # all evaluated pairs
    four = ds.sample_pairs(pairs, 4)
    assert len({p["video_id"] for p in four}) == 4 and four == ds.sample_pairs(pairs, 4, 0)
    ten = ds.sample_pairs(pairs, 10)
    assert len({p["video_id"] for p in ten}) == 6 and len({p["clip_id"] for p in ten}) == 10   # every video, then the rest
    assert ds.sample_pairs(pairs, 10, 1) != ten


def test_pairs_are_reconstructed_exactly_and_any_difference_is_refused(gate_run):
    report, segments = gate_run["report"], gate_run["segments"]
    pairs = [p for p in report["pairs"] if p["status"] == "evaluated"]
    assert len(pairs) == 8
    for pair in pairs:
        video = pair["video_id"]
        ours = dict(status="converted", segments=segments[video])
        job = ds.reconstruct(pair, ours, gate_run["translations"][video], report["fps"]["videos"][video])
        cue = next(s for s in segments[video] if s["sample_id"] == pair["sample_id"])
        assert (job["state_key"], job["state_sha256"], job["text"]) == (cue["state_key"], cue["state_sha256"], cue["text"])
        assert job["clip_id"].endswith(f"{job['first']:06d}-{job['last']:06d}") and job["cue_start"] == cue["start"]
        # the gate's own pairing, rerun, gives the same (clip, cue, IoU)
        again = {c["clip_id"]: (q["sample_id"], iou) for c, q, iou in gate.pair_clips(
            gate.uwb_clips_of(gate_run["translations"][video]), segments[video], pair["fps"])}
        assert again[pair["clip_id"]] == (pair["sample_id"], pair["iou"])
        fps_row = report["fps"]["videos"][video]
        with pytest.raises(ds.Mismatch, match="IoU"):
            ds.reconstruct(dict(pair, iou=pair["iou"] - .01), ours, gate_run["translations"][video], fps_row)
        with pytest.raises(ds.Mismatch, match="start"):
            ds.reconstruct(dict(pair, cue_start=pair["cue_start"] + .5), ours, gate_run["translations"][video], fps_row)
        with pytest.raises(ds.Mismatch, match="fps"):
            ds.reconstruct(pair, ours, gate_run["translations"][video], dict(fps_row, fps=30.))
        with pytest.raises(ds.Mismatch, match="not converted"):
            ds.reconstruct(pair, dict(ours, status="failed"), gate_run["translations"][video], fps_row)


def test_the_lifted_state_is_the_converters_own_and_windows_follow_the_gates_frame_map(gate_run):
    import convert_uwb_clips as cu
    from signtranslator.data_engineering import uwb_keypoints as uk
    from signtranslator.data_engineering.world_regressor import WorldRegressors
    from evaluate_text_to_motion import load_state
    from train_text_to_motion import make_store as store_of
    report, segments, root = gate_run["report"], gate_run["segments"], gate_run["path"]
    store = store_of(f"dir:{root}")
    regressors = WorldRegressors.load((root / "xy.pt").read_bytes(), expected_sha256=gate_run["sha"])
    pair = next(p for p in report["pairs"] if p["status"] == "evaluated")
    job = ds.reconstruct(pair, dict(status="converted", segments=segments[pair["video_id"]]),
                         gate_run["translations"][pair["video_id"]], report["fps"]["videos"][pair["video_id"]])
    ds.configure_converter(ds.converter_args("inside", "none", "none"), {pair["video_id"]: pair["fps"]}, regressors)
    with zipfile.ZipFile(root / gate_run["uwb_prefix"] / "raw_keypoints_1.zip") as z:
        member = z.read(pair["clip_id"] + ".json")
    ours = load_state(store, job["state_key"], job["state_sha256"])
    result = ds.lift_pair(job, member, ours)
    # the same bytes through convert_clip with the converter's settings
    clip = uk.parse_clip(member, pair["clip_id"])
    direct, _ = uk.convert_clip(clip, pair["fps"], source_sha256=hashlib.sha256(member).hexdigest(),
                                fps_source="estimated", regressors=regressors, world="learned", visibility="inside",
                                flags="generator")
    record, data = cu.convert_member(pair["clip_id"] + ".json", member, pair["zip"])
    assert record["state_sha256"] == result["lifted_state_sha256"] and record["fps"] == pair["fps"]
    i, j = gate.frame_map(ours.timestamps, clip.first, clip.frames, pair["fps"], pair["alignment"]["lag"])
    assert len(i) == pair["aligned_frames"] == result["aligned_frames"]
    lifted = result["lifted"]
    assert np.array_equal(lifted.timestamps, direct.timestamps[j.min():j.max() + 1])
    for name, ch in lifted.channels.items():
        assert np.array_equal(ch.values, direct.channels[name].values[j.min():j.max() + 1]), name
        assert np.array_equal(ch.valid, direct.channels[name].valid[j.min():j.max() + 1]), name
    assert np.array_equal(result["ours"].timestamps, ours.timestamps[i.min():i.max() + 1])
    assert abs(result["ours"].timestamps[0] - lifted.timestamps[0] - pair["alignment"]["lag"] / pair["fps"]) <= .5 / pair["fps"] + 1e-9
    with pytest.raises(ds.Mismatch, match="SHA-256"):
        ds.lift_pair(dict(job, member_sha256="0" * 64), member, ours)
    with pytest.raises(ds.Mismatch, match="aligned frames"):
        ds.lift_pair(dict(job, aligned_frames=job["aligned_frames"] + 1), member, ours)


def test_main_reads_three_states_against_the_same_references_with_blank_at_our_duration(gate_run, tmp_path):
    root, segments = gate_run["path"], gate_run["segments"]
    texts = {s["sample_id"]: s["text"] for v in segments.values() for s in v}
    calls = []

    expected = [texts[f"{v}-{k:04d}"] for v in VIDEOS for k in range(4)]     # rows are sorted by (video, clip)

    def translate(states):
        from signtranslator.data_engineering.uwb_keypoints import SOURCE_ID
        calls.append(states)
        out = []
        for p, s in enumerate(states):
            if not any(c.valid.any() for c in s.channels.values()):
                out.append("nothing at all")
            elif s.channels["body"].source_id == SOURCE_ID:      # lifted: one word wrong in every other pair
                assert s.sample_id.startswith(VIDEOS[p // 4])
                out.append(expected[p].replace("Cue", "Clue") if p % 2 else expected[p])
            else:                                                # ours: the reference itself
                out.append(expected[p])
        return out
    out = tmp_path / "downstream.json"
    assert ds.main(["--store", f"dir:{root}", "--gate-report", str(gate_run["gate"]), "--regressors",
                    str(root / "xy.pt"), "--regressors-sha256", gate_run["sha"], "--face", "none", "--device", "cpu",
                    "--workers", "1", "--bootstrap", "200", "--out", str(out)], translate_fn=translate) == 0
    report = json.loads(out.read_text())
    assert report["sample"] == dict(rule=report["sample"]["rule"], sampled=8, dev_sampled=4, union=8, reconstructed=8,
                                    scored=8, refused=0)
    assert len(calls) == 3 and all(len(c) == 8 for c in calls)
    lifted, ours, blank = calls
    for a, b in zip(ours, blank):                            # blank: our window's clock, every channel removed
        assert np.array_equal(a.timestamps, b.timestamps) and not any(c.valid.any() for c in b.channels.values())
    for a, b in zip(ours, lifted):                           # lifted and ours: the same window within a frame
        assert abs((a.timestamps[-1] - a.timestamps[0]) - (b.timestamps[-1] - b.timestamps[0])) <= 2 / 24 + 1e-9
    assert [r["text"] for r in report["per_pair"]] == expected
    for row in report["per_pair"]:                           # one reference per pair, the cue's own text
        assert row["text"] == texts[row["sample_id"]] and row["ours"] == row["text"] and row["blank"] == "nothing at all"
    refs = [r["text"] for r in report["per_pair"]]
    for read in ds.READS:                                    # unrounded
        assert report["reads"][read]["bleu"] == pytest.approx(corpus_bleu([r[read] for r in report["per_pair"]], refs)["bleu"])
    assert report["reads"]["ours"]["bleu"] == 100. and report["reads"]["blank"]["bleu"] == 0.
    pooled = report["verdict_pooled"]
    assert pooled["status"] == ("pass" if pooled["ratio_lifted_over_ours"] >= .9 else "fail")
    # the UWB dev partition (vidY, 4 pairs from 1 video) is read on its own and is too small to judge
    assert report["dev"]["pairs"] == 4 and report["dev"]["videos_scored"] == 1
    assert report["verdict"]["dev_pairs"] == 4 and report["verdict"]["status"] == (
        "fail" if pooled["status"] == "fail" else "not-evaluable")
    assert report["exact_match"]["ours_equals_reference"] == 1. and report["exact_match"]["lifted_equals_ours"] == .5
    assert set(report["by_uwb_partition"]) == {"train", "monitor"} and report["instrument"]["read_fps"] == 24.
    assert report["bootstrap"] == 200 and report["visibility"] == "inside"
    assert report["gate_report_sha256"] == hashlib.sha256(gate_run["gate"].read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="bundle"):
        ds.main(["--store", f"dir:{root}", "--gate-report", str(gate_run["gate"]), "--regressors-sha256", "f" * 64,
                 "--device", "cpu", "--out", str(out)], translate_fn=translate)


# ----------------------------------------------------------------------------------------------------------------------

SENTENCES = ["the cat sat on the mat .", "a dog ran in the park today", "so you want to cut it here and there",
             "then we brush it out a little bit", "make sure the power switch is off", "cover everything ."]


def test_bleu_from_segment_statistics_is_corpus_bleu():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(1, 8))
        refs = [SENTENCES[k] for k in rng.integers(0, len(SENTENCES), n)]
        hyps = [" ".join(rng.choice(r.split(), int(rng.integers(1, 9)))) for r in refs]
        stats = np.sum([ds.segment_stats(h, r) for h, r in zip(hyps, refs)], axis=0)
        assert ds.bleu_from(stats) == pytest.approx(corpus_bleu(hyps, refs)["bleu"], abs=1e-9)
    assert ds.bleu_from(ds.segment_stats("", "a b c d")) == 0.


def test_the_bootstrap_resamples_videos_with_all_their_pairs():
    rng = np.random.default_rng(1)
    refs = [SENTENCES[k % len(SENTENCES)] for k in range(24)]
    videos = [f"v{k // 4}" for k in range(24)]                       # 6 videos of 4 pairs
    hyps = {"lifted": [" ".join(r.split()[:int(rng.integers(2, 7))]) for r in refs], "ours": list(refs),
            "blank": ["the" for _ in refs]}
    stats = {r: np.array([ds.segment_stats(h, t) for h, t in zip(hyps[r], refs)]) for r in hyps}
    boot = ds.clustered_bootstrap(stats, videos, resamples=50, seed=3)
    # the same draws, by hand: resample the 6 videos, concatenate their pairs, corpus BLEU
    keys, draws = sorted(set(videos)), {r: [] for r in hyps}
    g = np.random.default_rng(3)
    for _ in range(50):
        pick = g.integers(0, len(keys), len(keys))
        idx = [k for p in pick for k in range(24) if videos[k] == keys[p]]
        for r in hyps:
            draws[r].append(corpus_bleu([hyps[r][k] for k in idx], [refs[k] for k in idx])["bleu"])
    for r in hyps:
        assert boot["reads"][r]["ci95"] == pytest.approx([float(np.percentile(draws[r], q)) for q in (2.5, 97.5)])
    d = np.asarray(draws["lifted"]) - np.asarray(draws["blank"])
    assert boot["differences"]["lifted_minus_blank"]["ci95"] == pytest.approx([float(np.percentile(d, q))
                                                                                for q in (2.5, 97.5)])
    assert boot["videos"] == 6 and boot["resamples"] == 50
    single = ds.clustered_bootstrap(stats, ["v0"] * 24, resamples=50)  # one video: no interval
    assert single["reads"]["ours"]["ci95"] is None and single["resamples"] == 0


def _boot(lifted, ours, blank, lb_lifted_blank, lb_ours_blank):
    return dict(reads=dict(lifted=dict(bleu=lifted), ours=dict(bleu=ours), blank=dict(bleu=blank)),
                differences=dict(lifted_minus_blank=dict(ci95=[lb_lifted_blank, 9.]),
                                 ours_minus_blank=dict(ci95=[lb_ours_blank, 9.]),
                                 lifted_minus_ours=dict(ci95=[-1., 1.])))


def test_the_downstream_verdict_needs_the_ratio_the_blank_margin_and_a_sane_instrument():
    assert ds.verdict(_boot(3.8, 4.0, 1.9, .5, .8))["status"] == "pass"           # 0.95 of ours, above blank
    assert ds.verdict(_boot(3.6, 4.0, 1.9, .5, .8))["status"] == "pass"           # exactly 0.9
    low = ds.verdict(_boot(3.5, 4.0, 1.9, .5, .8))
    assert low["status"] == "fail" and low["ratio_ok"] is False                   # 0.875
    near = ds.verdict(_boot(3.9, 4.0, 1.9, -.1, .8))
    assert near["status"] == "fail" and near["clearly_above_blank"] is False      # not clearly above blank
    assert ds.verdict(_boot(3.9, 4.0, 1.9, .5, -.2))["status"] == "not-evaluable"  # ours not above blank
    assert ds.verdict(_boot(0., 0., 0., 0., 0.))["status"] == "not-evaluable"
    assert ds.RATIO_BAR == .9
    just_below = ds.verdict(_boot(3.59985, 4.0, 1.9, .5, .8))                     # 0.8999625: compared unrounded
    assert just_below["status"] == "fail" and just_below["ratio_lifted_over_ours"] < .9


def test_the_row_passes_only_if_the_pooled_and_the_uwb_dev_reads_both_pass():
    ok, bad = ds.verdict(_boot(3.8, 4.0, 1.9, .5, .8)), ds.verdict(_boot(3.0, 4.0, 1.9, .5, .8))
    na = ds.verdict(_boot(3.8, 4.0, 1.9, .5, -.1))
    size = dict(dev_pairs=300, dev_videos=60, refused=0, sampled=2300)
    assert ds.combined_verdict(ok, ok, **size)["status"] == "pass"
    assert ds.combined_verdict(ok, bad, **size)["status"] == "fail"               # contaminated pooled pass
    assert ds.combined_verdict(bad, ok, **size)["status"] == "fail"
    assert ds.combined_verdict(ok, na, **size)["status"] == "not-evaluable"
    assert ds.combined_verdict(ok, None, **dict(size, dev_pairs=0, dev_videos=0))["status"] == "not-evaluable"
    assert ds.combined_verdict(ok, ok, **dict(size, dev_pairs=199))["status"] == "not-evaluable"
    assert ds.combined_verdict(ok, ok, **dict(size, dev_videos=49))["status"] == "not-evaluable"
    assert ds.combined_verdict(ok, ok, **dict(size, refused=231))["status"] == "not-evaluable"   # > 10% refused
    assert ds.combined_verdict(bad, ok, **dict(size, refused=231))["status"] == "not-evaluable"
    assert ds.combined_verdict(ok, ok, **dict(size, refused=230))["status"] == "pass"
    assert (ds.MIN_DEV_PAIRS, ds.MIN_DEV_VIDEOS, ds.MAX_REFUSED_SHARE, ds.DEV_PARTITION) == (200, 50, .1, "monitor")


def test_worker_count_reads_affinity_and_memory_not_nproc(monkeypatch):
    import os
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    n = ds.worker_count(per_worker_gb=1e-3, cap=10_000, reserve=0)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert n == cpus
    assert ds.worker_count(per_worker_gb=1e9) == 1


# ----------------------------------------------------------------------------------------------------------------------
# How2Sign-val medians (the "relative" row's denominator)

def _pairs_files(directory: Path, legacy: Path, name: str, signer: str, seed: int = 1, frames: int = 12):
    from semlex_synthetic import semlex_array
    from signtranslator.data_engineering.video_landmarks import Y_UP
    size = (640, 480)
    array, body, hands = semlex_array(size, frames=frames, seed=seed)
    vis = np.ones((frames, 33))
    vis[:, 25:] = 0.
    hand_world = np.stack([(h - h[:, :1]) @ Y_UP for h in (hands["Left"], hands["Right"])], axis=1)
    np.savez_compressed(directory / f"{name}.npz", signer=np.array(signer), clip=np.array(f"raw_videos/{name.split('__')[0]}.mp4"),
                        shape=np.array("4:3"), pose_found=np.ones(frames, bool), hand_found=np.ones((frames, 2), bool),
                        pose_img=array[:, 478:511].astype(np.float32), pose_world=(body @ Y_UP).astype(np.float32),
                        pose_vis=vis.astype(np.float32),
                        hand_img=np.stack((array[:, 511:532], array[:, 532:553]), 1).astype(np.float32),
                        hand_world=hand_world.astype(np.float32), size=np.array(size))
    np.savez_compressed(legacy / f"{name}.npz", holistic=array.astype(np.float16), size=np.array(size),
                        times=np.arange(frames) / 24., pose_visibility=vis.astype(np.float16))
    return body, hands


def test_how2sign_val_medians_use_the_gates_code_and_never_read_signer_9(tmp_path):
    import uwb_how2sign_val_medians as h2s
    pairs, legacy = tmp_path / "pairs", tmp_path / "legacy"
    pairs.mkdir(), legacy.mkdir()
    body, hands = _pairs_files(pairs, legacy, "abcVID_3-1-rgb_front__4x3", "h2s-1")
    _pairs_files(pairs, legacy, "abcVID_4-5-rgb_front__4x3", "h2s-5")                  # a training signer: not read
    files = h2s.val_files(pairs, {"h2s-1", "h2s-2"})
    assert [f.name for f in files] == ["abcVID_3-1-rgb_front__4x3.npz"]
    assert h2s.source_video("raw_videos/abcVID_3-1-rgb_front.mp4") == "abcVID"
    with pytest.raises(SystemExit, match="signer 9"):
        h2s.val_files(pairs, {"h2s-1", "h2s-9"})
    rules = gate.method_rules("inside", ["learned", "inside"])
    row = h2s.evaluate_clip(files[0], legacy, TrueWorldXY(body, hands), rules, cap=0, truth_flags="generator")
    assert row["status"] == "evaluated" and row["video_id"] == "abcVID" and row["alignment"]["lag"] == 0
    common = row["methods"]["lifter[inside]"]["common"]
    assert len(common["fingers"]) == 12 * 30 and np.max(common["fingers"]) < .5      # the true world: no error
    own = row["methods"]["lifter[learned]"]["own"]
    assert len(own["forearms"]) == 2 * 12 and np.max(own["forearms"]) < .5
    summary = h2s.summarise([row], "lifter[inside]", 0)
    assert summary["methods"]["lifter[inside]"]["common"]["fingers"]["videos"] == 1
    _pairs_files(pairs, legacy, "abcVID_5-2-rgb_front__4x3", "h2s-5")                  # field and name disagree
    with pytest.raises(SystemExit, match="differs from the file name"):
        h2s.clip_inputs(pairs / "abcVID_5-2-rgb_front__4x3.npz", legacy)
    _pairs_files(pairs, legacy, "abcVID_6-5-rgb_front__4x3", "h2s-5")
    with pytest.raises(SystemExit, match="not a validation signer"):
        h2s.clip_inputs(pairs / "abcVID_6-5-rgb_front__4x3.npz", legacy)
    with pytest.raises(SystemExit, match="validation signers"):
        h2s.val_files(pairs, {"h2s-1", "h2s-5"})
    _pairs_files(pairs, legacy, "abcVID_9-9-rgb_front__4x3", "h2s-9")
    with pytest.raises(SystemExit, match="signer 9"):
        h2s.val_files(pairs, {"h2s-1"})
