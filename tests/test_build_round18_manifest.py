"""Round 18 mix builder on a synthetic store: the real converter (a bundle with a validity head, visibility ``head``,
estimated-fps videos only, stream check) feeds ``build_round18_manifest.py``; dedup against our selected cues, the
amendment-2 join to our extraction reports (rejected cues stay out), leak drops before and after text cleaning,
train-only partitions, monitor channels, the segment rule with our own texts, the 20k floor, the refusals and the
unchanged base."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_round18_manifest as b18  # noqa: E402
import build_youtube_asl_manifest as yt  # noqa: E402
import convert_uwb_clips as cu  # noqa: E402
from test_de_world_regressor_xy import xy_bundle  # noqa: E402
from test_uwb_lifter_v2 import with_head  # noqa: E402
from uwb_synthetic import uwb_clip_json  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402

PREFIX, OUT, REPORTS = "acq/uwb", "acq/uwb/lifter-v2/converted-v1", "yt/videos"
MONITOR_CHANNEL = "chanM22"                      # build_youtube_asl_manifest.monitor_channel(.., 0.01) is true
# clip -> (text, frames, partition file)
CLIPS = {
    "vidA.000100-000159": ("Hello there my good friend", 60, "train"),            # paired with our selected cue
    "vidA.000200-000259": ("We went to the market today", 60, "train"),           # kept: no cue of ours
    "vidA.000300-000359": ("The quick brown fox jumps over the lazy dog", 60, "train"),   # evaluation leak
    "vidA.000400-000459": ("We went to the market today", 60, "train"),           # repeated text in the video
    "vidA.000500-000509": ("Too short to keep here", 10, "train"),                # 0.3 s: duration
    "vidA.000600-000659": ("Signing with my hidden hands", 60, "train"),          # our cue: hands not visible -> kept
    "vidA.000700-000759": ("Two people are talking now", 60, "train"),            # our cue: second person -> dropped
    "vidB.000010-000069": ("A stream that does not match", 60, "train"),          # stream mismatch: never read
    "vidC.000010-000069": ("A video without an estimated rate", 60, "train"),     # assumed fps: never read
    "vidD.000010-000069": ("Our own words from this video", 60, "train"),         # same text as our selected cue
    "vidE.000010-000069": ("Nobody else has this video here", 60, "train"),       # UWB-only video: kept
    "vidE.000080-000139": ("Thanks", 60, "train"),                                # one word
    "vidE.000150-000209": ("I love you very much (love)", 60, "train"),           # leaks only once cleaned
    "vidF.000010-000069": ("A clip from the dev file", 60, "dev"),                # UWB dev -> monitor
    "vidG.000010-000069": ("A channel held out for monitoring", 60, "train"),     # our video on a monitor channel
}
EVAL = [("test", "The quick brown fox jumps over the lazy dog!"), ("val", "I love you very much")]
H2S_VAL = ["I like apples very much", "She reads every night", "The bus was late again", "We play chess on Sunday",
           "My brother lives in Ohio"]
OUR_FILTERS = dict(allow_auto=False, channel_cap=1., corpus_id="youtube-asl-v1", eval_manifest=[], max_abs_yaw=30.,
                   max_chars=300, max_other_person=.1, max_seconds=20., max_words_per_second=4., min_face_coverage=.9,
                   min_hand_coverage=.5, min_seconds=1., min_words=2, min_words_per_second=.5, monitor_share=.01)


def segment(sample, text, start, end, **q):
    return dict(dict(sample_id=sample, cue=int(sample.rsplit("-", 1)[1]), text=text, start=start, end=end, hand_coverage=1., face_coverage=1.,
                     median_abs_yaw_deg=1., other_person_fraction=0., seconds=end - start,
                     state_key=f"states/{sample}.npz", state_sha256=hashlib.sha256(sample.encode()).hexdigest(),
                     available_channels=["body", "left_hand", "right_hand"]), **q)


def video_report(video, channel, segments, **over):
    return dict(dict(video_id=video, status="converted", caption_kind="manual", caption_seconds=None,
                     meta=dict(channel_id=channel, duration=600, height=720, fps=30), segments=segments), **over)


OUR_REPORTS = [
    video_report("vidA", "chanA", [segment("vidA-0000", "Hello there my good friend", 3.3, 5.3),
                                   segment("vidA-0001", "Signing with my hidden hands", 20., 22., hand_coverage=.2),
                                   segment("vidA-0002", "Two people are talking now", 23.3, 25.3,
                                           other_person_fraction=.5)]),
    video_report("vidD", "chanD", [segment("vidD-0000", "Our own words from this video", 30., 32.)]),
    video_report("vidG", MONITOR_CHANNEL, [segment("vidG-0000", "Something said on a monitor channel", 1., 3.)]),
]


def record(sign, corpus, partition, text, **extra):
    return dict(sign=f"{corpus}:{sign}", corpus=corpus, partition=partition, text=text, performer_sha256=None,
                continuous=dict(status="converted", state_key=f"states/{sign}.npz", seconds=2.,
                                state_sha256=hashlib.sha256(sign.encode()).hexdigest(),
                                available_channels=["body", "left_hand", "right_hand"]), **extra)


def put(root: Path, key: str, manifest: dict) -> str:
    data = canonical_json(manifest)
    (root / key).parent.mkdir(parents=True, exist_ok=True)
    (root / key).write_bytes(data)
    return f"{key}@{hashlib.sha256(data).hexdigest()}"


def put_manifest(root: Path, key: str, corpus_id: str, records: list[dict], **top) -> str:
    m = build_corpus_manifest(corpus_id=corpus_id, records=records, inputs={"x": "a" * 64},
                              split=dict(kind="continuous", counts={}))
    return put(root, key, dict(m, **top))


def our_manifest(root: Path) -> tuple[str, list[dict]]:
    """youtube-asl-v1 as build_youtube_asl_manifest builds it from OUR_REPORTS (filters and report digest recorded)."""
    for r in OUR_REPORTS:
        (root / REPORTS).mkdir(parents=True, exist_ok=True)
        (root / REPORTS / f"{r['video_id']}.json").write_text(json.dumps(r))
    records, why = yt.build(OUR_REPORTS, args=SimpleNamespace(**OUR_FILTERS), corpus_id="youtube-asl-v1")
    keys = sorted(f"{REPORTS}/{r['video_id']}.json" for r in OUR_REPORTS)
    spec = put_manifest(root, "yt/manifest-r3b2.json", "youtube-asl-v1", records,
                        filters=dict(OUR_FILTERS, reasons=why),
                        inputs={"reports": hashlib.sha256("".join(keys).encode()).hexdigest()})
    return spec, [dict(r, corpus="youtube-asl-v1") for r in records]


def base_records(ours, extra_val=()):
    recs = list(ours) + [record("t-0", "how2sign-two-view-text-v1", "test", "A test sentence of ours",
                                signer_component="c01")]
    recs += [record(f"v-{k}", "how2sign-two-view-text-v1", "val", t, signer_component=("c01", "c03", "c05")[k % 3])
             for k, t in enumerate(list(H2S_VAL) + list(extra_val))]
    recs += [record(f"tr-{k}", "how2sign-two-view-text-v1", "train", f"training sentence number {k}") for k in range(4)]
    return recs


@pytest.fixture()
def world(tmp_path):
    root = tmp_path
    (root / PREFIX).mkdir(parents=True)
    with zipfile.ZipFile(root / PREFIX / "raw_keypoints_1.zip", "w") as z:
        for k, (clip, (_, frames, _)) in enumerate(CLIPS.items()):
            z.writestr(f"{clip}.json", json.dumps(uwb_clip_json(frames=frames, seed=k % 4 + 1)[0]))
    for name, part in (("YT.translations.train.json", "train"), ("YT.translations.dev.json", "dev")):
        entries = {}
        for clip, (text, _, p) in CLIPS.items():
            if p == part:
                v = entries.setdefault(clip.split(".")[0], {"clip_order": []})
                v["clip_order"].append(clip)
                v[clip] = {"translation": text}
        (root / PREFIX / name).write_text(json.dumps(entries))
    payload = with_head(xy_bundle(), {"Left": .01, "Right": .01}).save()
    (root / "bundle.pt").write_bytes(payload)
    fps = {v: 30. for v in ("vidA", "vidB", "vidD", "vidE", "vidF", "vidG")}
    (root / "fps.json").write_text(json.dumps({"schema": "uwb-fps-table-v1", "fps": fps}))
    rows = {v: dict(fps=30., meta_fps=60 if v == "vidB" else 30) for v in fps}
    (root / "gate.json").write_text(json.dumps({"fps": {"videos": rows}}))
    pair = lambda sample: dict(our_sample_id=sample, iou=.9, overlaps=[dict(sample_id=sample, cover=.95, iou=.9)])  # noqa: E731
    overlap = dict(schema="uwb-overlap-table-v2",
                   clips={"vidA.000100-000159": pair("vidA-0000"), "vidA.000600-000659": pair("vidA-0001"),
                          "vidA.000700-000759": pair("vidA-0002")},
                   videos={"vidA": dict(channel_id="chanA", dedup="cue", partition="train"),
                           "vidD": dict(channel_id="chanD", dedup="cue", partition="train"),
                           "vidG": dict(channel_id=MONITOR_CHANNEL, dedup="cue", partition="monitor")})
    (root / "overlap.json").write_text(json.dumps(overlap))
    ev = put_manifest(root, "eval/manifest.json", "eval", [record(f"e-{k}", "e", p, t) for k, (p, t) in enumerate(EVAL)])
    ours_spec, ours = our_manifest(root)
    base = put_manifest(root, "mix/round14b.json", "round14b-mixed-v1", base_records(ours))
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    w = dict(root=root, eval=ev, base=base, ours=ours_spec, our_records=ours, bundle_sha=hashlib.sha256(payload).hexdigest(),
             fps_sha=sha("fps.json"), gate_sha=sha("gate.json"), overlap_sha=sha("overlap.json"))
    assert convert(w) == 0
    assert cu.main(["--store", f"dir:{root}", "--out-prefix", OUT, "--merge-index"]) == 0
    return w


def convert(w, *extra):
    r = w["root"]
    return cu.main(["--store", f"dir:{r}", "--prefix", PREFIX, "--out-prefix", OUT, "--regressors", str(r / "bundle.pt"),
                    "--regressors-sha256", w["bundle_sha"], "--visibility", "head", "--face", "none",
                    "--fps-table", str(r / "fps.json"), "--fps-sources", "estimated", "--stream-check",
                    str(r / "gate.json"), "--overlap-table", str(r / "overlap.json"), "--eval-manifest", w["eval"],
                    "--chunk-size", "5", "--workers", "1", *extra])


def build_args(w, *extra, base=None, **expect):
    e = dict(bundle=w["bundle_sha"], fps=w["fps_sha"], stream=w["gate_sha"], overlap=w["overlap_sha"]) | expect
    return ["--store", f"dir:{w['root']}", "--base", base or w["base"], "--converted-prefix", OUT,
            "--eval-manifest", w["eval"], "--lifted-key", "lifted/manifest.json", "--manifest-key", "mix/round18.json",
            "--expect-bundle-sha", e["bundle"], "--expect-fps-table-sha", e["fps"], "--expect-stream-check-sha",
            e["stream"], "--expect-overlap-sha", e["overlap"], "--expect-face", "none", "--our-manifest", w["ours"],
            "--our-reports", REPORTS, "--min-lifted", "1", *extra]


def _nl(path: Path) -> Path:
    """load_manifest reads a file ending in a newline (as the store fetch writes it)."""
    out = path.with_suffix(".nl")
    out.write_bytes(path.read_bytes() + b"\n")
    return out


def summary_of(out: str) -> dict:
    start = out.rindex('\n{\n "base"') + 1
    return json.loads(out[start:out.index("LIFTED_SHA=")])


def test_the_mix_is_the_base_plus_the_selected_lifted_train_clips(world, capsys):
    root = world["root"]
    assert b18.main(build_args(world)) == 0
    out = capsys.readouterr().out
    summary = summary_of(out)
    mix_sha = out.split("MANIFEST_SHA=")[1].split()[0]
    assert summary["manifest_sha256"] == mix_sha == hashlib.sha256((root / "mix/round18.json").read_bytes()).hexdigest()
    # the conversion left out the stream-mismatch and assumed-fps videos unread
    assert summary["conversion_statuses"] == {"converted": 13, "skipped": 2}
    lifted = load_manifest(_nl(root / "lifted/manifest.json"), expected_sha256=summary["lifted_sha256"])
    kept = {r["source"]["clip_id"]: r for r in lifted["records"]}
    assert set(kept) == {"vidA.000200-000259", "vidA.000600-000659", "vidE.000010-000069"}
    assert summary["select_training_records"] == {"duplicate_of_our_cue": 1, "evaluation_leak": 1, "kept": 11,
                                                  "status_skipped": 2}
    assert summary["rejected_cues"] == {"our_rejected:second_person": 1}
    assert summary["kept_by_category_before_segment_rule"] == {"never_extracted": 9,
                                                               "our_rejected:hands_not_visible": 1}
    assert summary["segment"] == {"partition_monitor": 2, "repeated_text": 2, "duration": 1, "few_words": 1,
                                  "evaluation_leak_cleaned": 1, "kept": 3}
    assert summary["kept_by_category"] == {"never_extracted": 2, "our_rejected:hands_not_visible": 1}
    assert summary["our_cues"]["report_list_is_the_snapshots"] is True
    assert summary["our_cues"]["snapshot_not_reproduced"] == 0 and summary["our_cues"]["kept_not_in_snapshot"] == 0
    a = kept["vidA.000200-000259"]
    assert a["sign"] == "youtube-asl-uwb-lifted-v2:vidA.000200-000259" and a["partition"] == "train"
    assert a["source"]["partition_source"] == "our-video" and a["source"]["fps_source"] == "estimated"
    assert a["source"]["our_cue_category"] == "never_extracted"
    assert kept["vidA.000600-000659"]["source"]["our_cue_category"] == "our_rejected:hands_not_visible"
    assert a["lifted"]["visibility"] == "head" and a["lifted"]["regressors_sha256"] == world["bundle_sha"]
    assert a["usable"]["continuous"] and a["continuous"]["state_key"].startswith(OUT + "/states/")
    assert kept["vidE.000010-000069"]["source"]["partition_source"] == "uwb"
    # the mix: every base record byte-identical, the lifted corpus added under its own name, train only
    base = load_manifest(_nl(root / "mix/round14b.json"), expected_sha256=world["base"].split("@")[1])
    mix = load_manifest(_nl(root / "mix/round18.json"), expected_sha256=mix_sha)
    by = {r["sign"]: r for r in mix["records"]}
    assert all(canonical_json(by[r["sign"]]) == canonical_json(r) for r in base["records"])
    assert len(mix["records"]) == len(base["records"]) + 3
    assert summary["by_corpus"]["youtube-asl-uwb-lifted-v2/train"] == 3
    assert mix["inputs"] == {"round14b-mixed-v1": world["base"].split("@")[1],
                             "youtube-asl-uwb-lifted-v2": summary["lifted_sha256"]}
    assert summary["val_prefix_unchanged"] == {"1536": True, "2048": True} and summary["pilot_sample_unchanged"]
    from pilot_content_check import validation_sentences
    assert validation_sentences(mix, 1000) == validation_sentences(base, 1000)
    # write-once
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        b18.main(build_args(world))


def test_fewer_than_the_floor_prints_no_meaningful_gain_exits_5_and_writes_nothing(world, capsys):
    assert b18.main(build_args(world)[:-2] + ["--min-lifted", "4"]) == 5
    assert "ROUND18-NO-MEANINGFUL-GAIN 3 lifted records < 4" in capsys.readouterr().out
    assert not (world["root"] / "mix/round18.json").exists() and not (world["root"] / "lifted/manifest.json").exists()
    assert b18.MIN_LIFTED == 20_000 and b18.NO_GAIN_EXIT == 5


def test_a_lifted_text_that_is_an_evaluation_sentence_of_the_base_is_dropped_and_the_guard_holds(world, capsys):
    leaky = put_manifest(world["root"], "mix/leaky.json", "round14b-mixed-v1",
                         base_records(world["our_records"], extra_val=["Nobody else has this video here"]))
    assert b18.main(build_args(world, base=leaky)) == 0
    summary = summary_of(capsys.readouterr().out)
    assert summary["segment"]["evaluation_leak_cleaned"] == 2 and summary["lifted_records"] == 2
    # the guard is an assertion after the filter: a leak that got past it refuses the build
    import build_round18_manifest
    real = build_round18_manifest.segment_filter
    try:
        build_round18_manifest.segment_filter = lambda records, **kw: (
            [dict(r, text=r["text"]) for r in records if r["partition"] == "train"], {})
        with pytest.raises(SystemExit, match="evaluation text"):
            b18.main(build_args(world, base=leaky)[:-6] + ["--lifted-key", "l2.json", "--manifest-key", "m2.json",
                                                          "--expect-face", "none", "--our-manifest", world["ours"],
                                                          "--our-reports", REPORTS, "--min-lifted", "1"])
    finally:
        build_round18_manifest.segment_filter = real


def test_cleaned_text_leaks_and_our_own_texts_are_filtered():
    from build_youtube_asl_manifest import EvalIndex
    rec = lambda text, clip="c1", vid="v1": dict(clip_id=clip, video_id=vid, first_frame=0, text=text,  # noqa: E731
                                                 partition="train", seconds=2., hand_fraction={"Left": .9},
                                                 our_channel_id="UCx")
    index = EvalIndex(["I love you"])
    assert not index.leaks("I love you (love)")              # the raw caption passes the conversion's check
    kept, why = b18.segment_filter([rec("I love you (love)"), rec("We went home late", "c2")], monitor_share=.01,
                                   leak_index=index)
    assert [r["clip_id"] for r in kept] == ["c2"] and why["evaluation_leak_cleaned"] == 1
    kept, why = b18.segment_filter([rec("We went home late"), rec("We went home late", "c2", "v2")], monitor_share=.01,
                                   our_texts={"v1": ["We went home late."]})
    assert [r["clip_id"] for r in kept] == ["c2"] and why["repeated_text"] == 1
    kept, why = b18.segment_filter([dict(rec("A channel held out for monitoring"), our_channel_id=MONITOR_CHANNEL)],
                                   monitor_share=.01)
    assert kept == [] and why == {"monitor_channel": 1}


def test_our_cue_reasons_mirror_build_youtube_asl_manifest_including_the_channel_cap():
    from build_youtube_asl_manifest import EvalIndex
    segs = [segment(f"v1-{k:04d}", f"Sentence number {k} in the first video", 3. * k, 3. * k + 2.) for k in range(8)]
    segs += [segment("v1-0100", "Sentence number 1 in the first video", 40., 42.),            # repeated
             segment("v1-0101", "I love you very much", 43., 45.),                            # evaluation leak
             segment("v1-0102", "Hidden hands here", 46., 48., hand_coverage=.1),
             segment("v1-0103", "Facing away from us", 49., 51., median_abs_yaw_deg=50.),
             segment("v1-0104", "Face hidden from view", 52., 54., face_coverage=.2),
             segment("v1-0105", "A", 55., 57.)]
    reports = [video_report("v1", "chanA", segs),
               video_report("v2", "chanB", [segment(f"v2-{k:04d}", f"Second video sentence {k}", 3. * k, 3. * k + 2.)
                                            for k in range(3)]),
               video_report("v3", "chanC", [segment("v3-0000", "Low resolution video", 1., 3.)],
                            meta=dict(channel_id="chanC", duration=600, height=240, fps=30)),
               video_report("v4", "chanD", [segment("v4-0000", "Auto captions only", 1., 3.)], caption_kind="auto")]
    args = SimpleNamespace(**dict(OUR_FILTERS, channel_cap=.3))
    index = EvalIndex(["I love you very much"])
    records, why = yt.build(reports, args=args, corpus_id="youtube-asl-v1", eval_index=index)
    reasons = b18.our_cue_reasons(reports, args=args, eval_index=index)
    assert {k for k, v in reasons.items() if v == "kept"} == {r["sign"].split(":", 1)[1] for r in records}
    by_segment = {k: v for k, v in why.items() if not k.startswith("video_") and k != "kept"}
    import collections
    mine = collections.Counter(v for v in reasons.values() if not v.startswith("video_") and v != "kept")
    assert dict(mine) == by_segment and why["channel_cap"] > 0
    assert reasons["v3-0000"] == "video_resolution" and reasons["v4-0000"] == "video_auto_captions"
    assert reasons["v1-0102"] == "hands_not_visible" and reasons["v1-0103"] == "not_frontal"


def test_undeclared_settings_and_incomplete_conversions_are_refused(world, capsys):
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, bundle="0" * 64))
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, "--expect-visibility", "inside"))
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, stream="1" * 64))
    for flag, value in (("--expect-face", "blendshapes"), ("--expect-head", "affine"), ("--expect-flags", "upright-chest"),
                        ("--expect-min-frames", "4"), ("--expect-world", "pseudo")):
        with pytest.raises(SystemExit, match="not the declared ones"):
            b18.main(build_args(world, flag, value))
    other_eval = put_manifest(world["root"], "eval/other.json", "eval", [record("e-1", "e", "val", "Another one here")])
    args = build_args(world)
    args[args.index(world["eval"])] = other_eval
    with pytest.raises(SystemExit, match="not the declared ones"):        # the eval list is part of the settings
        b18.main(args)
    index = json.loads((world["root"] / OUT / "index.json").read_text())
    index["complete"] = False
    (world["root"] / OUT / "index.json").write_text(json.dumps(index))
    with pytest.raises(SystemExit, match="not complete"):
        b18.main(build_args(world))
    assert b18.main(build_args(world, "--allow-incomplete", "--dry-run")) == 0
    assert not (world["root"] / "mix/round18.json").exists()          # a dry run writes nothing


def test_our_manifest_must_be_the_bases_youtube_records(world):
    other, _ = our_manifest(world["root"])
    m = json.loads((world["root"] / "yt/manifest-r3b2.json").read_text())
    m["records"] = m["records"][:-1]
    spec = put(world["root"], "yt/short.json", m)
    args = build_args(world)
    args[args.index(other)] = spec
    with pytest.raises(SystemExit, match="does not hold exactly"):
        b18.main(args)


def test_a_record_converted_at_an_assumed_rate_or_with_another_rule_refuses_the_build(world):
    report_path = sorted((world["root"] / OUT / "reports").rglob("chunk-*.json"))[0]
    report = json.loads(report_path.read_text())
    clip = next(c for c in report["clips"] if c["status"] == "converted")
    clip["fps_source"] = "assumed"
    report_path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="assumed rate"):
        b18.main(build_args(world))
    clip["fps_source"], clip["visibility"] = "estimated", "inside"
    report_path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="visibility rule differs"):
        b18.main(build_args(world))
