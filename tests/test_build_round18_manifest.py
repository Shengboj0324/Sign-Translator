"""Round 18 mix builder on a synthetic store: the real converter (lifter-v2-style bundle, visibility ``head``,
estimated-fps videos only, stream check) feeds ``build_round18_manifest.py``; dedup against our selected cues, leak
drops, train-only partitions, monitor channels, the segment rule, the refusals and the unchanged base."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_round18_manifest as b18  # noqa: E402
import convert_uwb_clips as cu  # noqa: E402
from test_de_world_regressor_xy import xy_bundle  # noqa: E402
from test_uwb_lifter_v2 import with_head  # noqa: E402
from uwb_synthetic import uwb_clip_json  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402

PREFIX, OUT = "acq/uwb", "acq/uwb/lifter-v2/converted-v1"
MONITOR_CHANNEL = "chanM22"                      # build_youtube_asl_manifest.monitor_channel(.., 0.01) is true
# clip -> (text, frames, partition file)
CLIPS = {
    "vidA.000100-000159": ("Hello there my good friend", 60, "train"),            # paired with our selected cue
    "vidA.000200-000259": ("We went to the market today", 60, "train"),           # kept (our video, train)
    "vidA.000300-000359": ("The quick brown fox jumps over the lazy dog", 60, "train"),   # evaluation leak
    "vidA.000400-000459": ("We went to the market today", 60, "train"),           # repeated text in the video
    "vidA.000500-000509": ("Too short to keep here", 10, "train"),                # 0.3 s: duration
    "vidB.000010-000069": ("A stream that does not match", 60, "train"),          # stream mismatch: never read
    "vidC.000010-000069": ("A video without an estimated rate", 60, "train"),     # assumed fps: never read
    "vidD.000010-000069": ("Our monitor video says this", 60, "train"),           # our record is monitor
    "vidE.000010-000069": ("Nobody else has this video here", 60, "train"),       # UWB-only video: kept
    "vidE.000080-000139": ("Thanks", 60, "train"),                                # one word
    "vidF.000010-000069": ("A clip from the dev file", 60, "dev"),                # UWB dev -> monitor
    "vidG.000010-000069": ("A channel held out for monitoring", 60, "train"),     # monitor channel by hash
}
EVAL_TEST = "The quick brown fox jumps over the lazy dog!"
H2S_VAL = ["I like apples very much", "She reads every night", "The bus was late again", "We play chess on Sunday",
           "My brother lives in Ohio"]


def record(sign, corpus, partition, text, **extra):
    return dict(sign=f"{corpus}:{sign}", corpus=corpus, partition=partition, text=text, performer_sha256=None,
                continuous=dict(status="converted", state_key=f"states/{sign}.npz", seconds=2.,
                                state_sha256=hashlib.sha256(sign.encode()).hexdigest(),
                                available_channels=["body", "left_hand", "right_hand"]), **extra)


def put_manifest(root: Path, key: str, corpus_id: str, records: list[dict]) -> str:
    m = build_corpus_manifest(corpus_id=corpus_id, records=records, inputs={"x": "a" * 64},
                              split=dict(kind="continuous", counts={}))
    data = canonical_json(m)
    (root / key).parent.mkdir(parents=True, exist_ok=True)
    (root / key).write_bytes(data)
    return f"{key}@{hashlib.sha256(data).hexdigest()}"


def base_records(extra_val=()):
    yt = "youtube-asl-v1"
    recs = [record("vidA-0000", yt, "train", "Hello there my good friend", source=dict(video_id="vidA", channel_id="chanA")),
            record("vidD-0000", yt, "monitor", "Something ours", source=dict(video_id="vidD", channel_id="chanD")),
            record("vidG-0000", yt, "train", "Something else", source=dict(video_id="vidG", channel_id=MONITOR_CHANNEL)),
            record("t-0", "how2sign-two-view-text-v1", "test", "A test sentence of ours", signer_component="c01")]
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
    overlap = dict(schema="uwb-overlap-table-v2",
                   clips={"vidA.000100-000159": dict(our_sample_id="vidA-0000", iou=.9,
                                                     overlaps=[dict(sample_id="vidA-0000", cover=.95, iou=.9)])},
                   videos={"vidA": dict(channel_id="chanA", dedup="cue", partition="train"),
                           "vidD": dict(channel_id="chanD", dedup="cue", partition="train"),
                           "vidG": dict(channel_id=MONITOR_CHANNEL, dedup="cue", partition="monitor")})
    (root / "overlap.json").write_text(json.dumps(overlap))
    ev = put_manifest(root, "eval/manifest.json", "eval", [record("e-0", "e", "test", EVAL_TEST)])
    base = put_manifest(root, "mix/round14b.json", "round14b-mixed-v1", base_records())
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()  # noqa: E731
    w = dict(root=root, eval=ev, base=base, bundle_sha=hashlib.sha256(payload).hexdigest(), fps_sha=sha("fps.json"),
             gate_sha=sha("gate.json"), overlap_sha=sha("overlap.json"))
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
            e["stream"], "--expect-overlap-sha", e["overlap"], *extra]


def test_the_mix_is_the_base_plus_the_selected_lifted_train_clips(world, capsys):
    root = world["root"]
    assert b18.main(build_args(world)) == 0
    out = capsys.readouterr().out
    summary = json.loads(out[:out.index("LIFTED_SHA=")])
    mix_sha = out.split("MANIFEST_SHA=")[1].split()[0]
    assert summary["manifest_sha256"] == mix_sha == hashlib.sha256((root / "mix/round18.json").read_bytes()).hexdigest()
    # the conversion left out the stream-mismatch and assumed-fps videos unread
    assert summary["conversion_statuses"] == {"converted": 10, "skipped": 2}
    lifted = load_manifest(_nl(root / "lifted/manifest.json"), expected_sha256=summary["lifted_sha256"])
    kept = {r["source"]["clip_id"]: r for r in lifted["records"]}
    assert set(kept) == {"vidA.000200-000259", "vidE.000010-000069"}
    assert summary["select_training_records"] == {"duplicate_of_our_cue": 1, "evaluation_leak": 1, "kept": 8,
                                                  "status_skipped": 2}
    assert summary["segment"] == {"partition_monitor": 2, "monitor_channel": 1, "repeated_text": 1, "duration": 1,
                                  "few_words": 1, "kept": 2}
    a = kept["vidA.000200-000259"]
    assert a["sign"] == "youtube-asl-uwb-lifted-v2:vidA.000200-000259" and a["partition"] == "train"
    assert a["source"]["partition_source"] == "our-video" and a["source"]["fps_source"] == "estimated"
    assert a["lifted"]["visibility"] == "head" and a["lifted"]["regressors_sha256"] == world["bundle_sha"]
    assert a["usable"]["continuous"] and a["continuous"]["state_key"].startswith(OUT + "/states/")
    assert kept["vidE.000010-000069"]["source"]["partition_source"] == "uwb"
    # the mix: every base record byte-identical, the lifted corpus added under its own name, train only
    base = load_manifest(_nl(root / "mix/round14b.json"), expected_sha256=world["base"].split("@")[1])
    mix = load_manifest(_nl(root / "mix/round18.json"), expected_sha256=mix_sha)
    by = {r["sign"]: r for r in mix["records"]}
    assert all(canonical_json(by[r["sign"]]) == canonical_json(r) for r in base["records"])
    assert len(mix["records"]) == len(base["records"]) + 2
    assert summary["by_corpus"]["youtube-asl-uwb-lifted-v2/train"] == 2
    assert mix["inputs"] == {"round14b-mixed-v1": world["base"].split("@")[1],
                             "youtube-asl-uwb-lifted-v2": summary["lifted_sha256"]}
    assert summary["val_prefix_unchanged"] == {"1536": True, "2048": True} and summary["pilot_sample_unchanged"]
    from pilot_content_check import validation_sentences
    assert validation_sentences(mix, 1000) == validation_sentences(base, 1000)
    # write-once
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        b18.main(build_args(world))


def _nl(path: Path) -> Path:
    """load_manifest reads a file ending in a newline (as the store fetch writes it)."""
    out = path.with_suffix(".nl")
    out.write_bytes(path.read_bytes() + b"\n")
    return out


def test_a_lifted_text_that_is_an_evaluation_sentence_of_the_base_refuses_the_build(world):
    leaky = put_manifest(world["root"], "mix/leaky.json", "round14b-mixed-v1",
                         base_records(extra_val=["Nobody else has this video here"]))
    with pytest.raises(SystemExit, match="evaluation text"):
        b18.main(build_args(world, base=leaky))
    assert not (world["root"] / "mix/round18.json").exists()


def test_undeclared_settings_and_incomplete_conversions_are_refused(world, capsys):
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, bundle="0" * 64))
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, "--expect-visibility", "inside"))
    with pytest.raises(SystemExit, match="not the declared ones"):
        b18.main(build_args(world, stream="1" * 64))
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
