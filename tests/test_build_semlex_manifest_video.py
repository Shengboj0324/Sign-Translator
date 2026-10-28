"""Sem-Lex manifest builder, video path: per-shard extraction reports with reruns and a crashed pass, on a fake store."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zlib

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_semlex_manifest as job  # noqa: E402
import convert_semlex_poses  # noqa: E402
from semlex_synthetic import FakeBlendshapes, fake_resources, semlex_array  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import (  # noqa: E402
    build_corpus_manifest, canonical_json, load_manifest)
from signtranslator.data_engineering.semlex_pose import convert_array  # noqa: E402

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
PREFIX = f"{BASE}/states-video-v1"
LANDMARKS = f"{BASE}/landmarks-video-v1"
VAL_ARCHIVE, TRAIN_ARCHIVE = "1" * 64, "2" * 64
MODELS = {"face_landmarker.task": "f" * 64, "hand_landmarker.task": "e" * 64, "pose_landmarker_heavy.task": "d" * 64}
CONVENTION = dict(elbow_in_image=False, upright_chest=True, upright_root=True, visibility="mediapipe")
WORDS = "cat\ndog\nfall\nasleep\nhello\nice\n"


def in_shard(stem: str, shard: int, n: int, shards: int = 2) -> list[str]:
    out, k = [], 0
    while len(out) < n:
        name = f"{stem}{k:04d}"
        if zlib.crc32(name.encode()) % shards == shard:
            out.append(name)
        k += 1
    return out


S0 = in_shard("vidTRNa", 0, 10)         # train shard 0: reruns
S1 = in_shard("vidTRNb", 1, 2)          # train shard 1: a crashed pass, then a redo
CAT5 = in_shard("vidTRNc", 0, 5)        # five train clips of "cat" (coverage at 5)
VAL = ["vidVAL0001", "vidVAL0002"]
BOTH = in_shard("vidBOTH", 0, 1)[0]     # listed in train and val, extracted in both archives
GAP = "vidGAP000000000000001"           # in the metadata, not in the train archive (a pinned gap)


def _rows():
    rows = [(VAL[0], "8", "val", "asllex", "cat_3"), (VAL[1], "9", "val", "asllex", "ice_2"),
            (BOTH, "8", "train", "asllex", "ice_2"), (BOTH, "8", "val", "asllex", "ice_2"),
            (GAP, "7", "train", "asllex", "cat_3"), ("vidTEST000000001", "9", "test", "asllex", "cat_3")]
    rows += [(v, "7", "train", "asllex", "dog_1") for v in S0] + [(v, "8", "train", "asllex", "ice_2") for v in S1]
    rows += [(v, str(7 + k % 2), "train", "asllex", "cat_3") for k, v in enumerate(CAT5)]
    return rows


def _state(tmp_path, video_id, split):
    array, *_ = semlex_array((640, 640), frames=30, seed=5)          # a seed whose arms stay in frame
    state, _ = convert_array(array, 1000., fake_resources(), video_id=video_id, source_sha256="a" * 64,
                             blendshape_model=FakeBlendshapes(), image_size=(640, 640), pose_z_scale=1.,
                             hand_z_scale=1., flags="generator")
    key = f"{PREFIX}/{split}/{video_id}.npz"
    path = tmp_path / key
    path.parent.mkdir(parents=True, exist_ok=True)
    return key, state.save(path)


def _result(video_id, split, status, states):
    r = dict(video_id=video_id, split=split, status=status, source_sha256="c" * 64, index=0)
    if status in ("converted", "resumed", "already_present"):
        key, sha = states[(video_id, split)]
        r.update(state_key=key, state_sha256=sha, landmarks_key=f"{LANDMARKS}/{split}/{video_id}.npz",
                 landmarks_sha256="9" * 64, fps=29.97, duration_ratio=.98, image_size=[640, 480],
                 upright_root_applied=True, tracked=dict(forearm=dict(Left=.1, Right=.9) if video_id == VAL[1]
                                                         else dict(Left=.8, Right=.2)))
    elif status == "exists_different":
        r.update(state_key=f"{PREFIX}/{split}/{video_id}.npz", state_sha256=None, stored_sha256="8" * 64)
    else:
        r["reason"] = status
    return r


def _report(tmp_path, split, shard, shards, started, results, meta_sha, *, full=True, models=MODELS, name=None):
    report = dict(schema_version=1, extractor="semlex-video-v1", convention=CONVENTION, inference_method="mp 1.1.1",
                  models=models, split=split, shard=shard, shards=shards, smoke=False, limit=None, match=None,
                  archive_sha256=VAL_ARCHIVE if split == "val" else TRAIN_ARCHIVE, metadata_sha256=meta_sha,
                  states_prefix=PREFIX, landmarks_prefix=LANDMARKS, started=started,
                  archive_stream=dict(matches_pinned=True if full else None), results=results)
    key = f"{PREFIX}/reports/{name or f'{split}-shard-{shard:03d}-of-{shards:03d}-{started}'}.json"
    (tmp_path / key).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / key).write_text(json.dumps(report))
    return key


def _store(tmp_path, monkeypatch):
    """Metadata, ASL-LEX, states, reports (reruns, a crashed pass), a base mix and a recogniser validation report."""
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["", "video_id", "signer_id", "duration", "split", "label_type", "label"])
    for k, (v, s, sp, t, label) in enumerate(_rows()):
        writer.writerow([k, v, s, 1000, sp, t, label])
    meta = text.getvalue().encode()
    meta_sha = hashlib.sha256(meta).hexdigest()
    (tmp_path / BASE / "data").mkdir(parents=True)
    (tmp_path / BASE / "data/semlex_metadata.csv").write_bytes(meta)
    lex = io.StringIO()
    w = csv.writer(lex)
    w.writerow(["EntryID", "LemmaID", "Code", "SignBankReferenceID"])
    w.writerows([("ice_2", "ice_2", "C_01_001", "100"), ("cat_3", "cat_3", "C_02_002", "963"),
                 ("dog_1", "dog", "C_03_003", "5")])
    lex_bytes = lex.getvalue().encode()
    lex_key = tmp_path / "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
    lex_key.parent.mkdir(parents=True)
    lex_key.write_bytes(lex_bytes)
    Path(str(lex_key) + ".provenance.json").write_text(json.dumps(dict(sha256=hashlib.sha256(lex_bytes).hexdigest(),
                                                                       verified_by_r2_reread=True)))
    pins = tmp_path / "known_missing.json"
    pins.write_text(json.dumps(dict(splits=dict(
        val=dict(archive_sha256=VAL_ARCHIVE, format="webm", missing=[]),
        train=dict(archive_sha256=TRAIN_ARCHIVE, format="webm", missing=[GAP])))))
    monkeypatch.setattr(convert_semlex_poses, "KNOWN_MISSING", pins)
    states = {}
    for v in VAL + [BOTH]:
        states[(v, "val")] = _state(tmp_path, v, "val")
    for v in S0 + S1 + CAT5 + [BOTH]:
        states[(v, "train")] = _state(tmp_path, v, "train")
    # val: one report per shard (one shard)
    _report(tmp_path, "val", 0, 1, "2026-10-28T12:00:00.000000Z",
            [_result(v, "val", "converted", states) for v in VAL + [BOTH]], meta_sha)
    # train shard 0: a first pass, then a rerun; per clip (first, rerun)
    plan = {S0[0]: ("error", "resumed"), S0[1]: ("converted", "already_present"),
            S0[2]: ("converted", "exists_different"), S0[3]: ("failed", None), S0[4]: ("refused", "refused"),
            S0[5]: ("converted", "error"), S0[6]: ("converted", None), S0[7]: ("converted", "converted"),
            S0[8]: ("converted", "converted"), S0[9]: ("converted", "refused"), BOTH: ("converted", None)}
    plan.update({v: ("converted", None) for v in CAT5})
    first = [_result(v, "train", a, states) for v, (a, _) in plan.items()]
    rerun = [_result(v, "train", b, states) for v, (_, b) in plan.items() if b]
    _report(tmp_path, "train", 0, 2, "2026-10-28T13:00:00.000000Z", first, meta_sha)
    _report(tmp_path, "train", 0, 2, "2026-10-28T15:00:00.000000Z", rerun, meta_sha)
    # train shard 1: a crashed pass (its partial report names a wrong state), then the redo
    crashed = [dict(_result(S1[0], "train", "converted", states), state_sha256="7" * 64)]
    _report(tmp_path, "train", 1, 2, "2026-10-28T13:05:00.000000Z", crashed, meta_sha, full=False)
    _report(tmp_path, "train", 1, 2, "2026-10-28T16:00:00.000000Z",
            [_result(S1[0], "train", "resumed", states), _result(S1[1], "train", "already_present", states)], meta_sha)
    (tmp_path / "words.txt").write_text(WORDS)
    # the base mix: How2Sign sentences (the R5 vocabulary) and an existing one-word corpus
    records = [dict(sign=f"h2s-{k}", corpus="how2sign-two-view-text-v1", partition="train", text=t,
                    continuous=dict(status="converted", state_key=f"x/{k}.npz", state_sha256=f"{k:064x}",
                                    available_channels=["body"]))
               for k, t in enumerate(["The cat saw ice.", "A dog and a cat.", "Hello there."])]
    records += [dict(sign=f"rit-{k}", corpus="rit-text-r6", partition="train", text="dog",
                     continuous=dict(status="converted", state_key=f"r/{k}.npz", state_sha256=f"{100 + k:064x}",
                                     available_channels=["body"])) for k in range(5)]
    mix = build_corpus_manifest(corpus_id="base-mix", inputs={"x": "0" * 64}, records=records, split=dict(kind="x"))
    mix_bytes = canonical_json(mix)
    (tmp_path / "mixes").mkdir()
    (tmp_path / "mixes/base.json").write_bytes(mix_bytes)
    validation = json.dumps(dict(per_gloss=dict(reliable_glosses=["CAT", "DOG", "HELLO", "ICE", "ZEBRA", "A1"]))).encode()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs/validation.json").write_bytes(validation)
    return meta_sha, states, hashlib.sha256(mix_bytes).hexdigest(), hashlib.sha256(validation).hexdigest()


def _argv(tmp_path, meta_sha, mix_sha, val_sha, out="manifest.json", *extra, strict=False):
    return ["--store", f"dir:{tmp_path}", "--source", "video", "--metadata-sha256", meta_sha,
            "--wordlist", str(tmp_path / "words.txt")] + ([] if strict else ["--allow-unfinished"]) + [
            "--coverage-mix", f"mixes/base.json@{mix_sha}",
            "--recognizer-validation", f"runs/validation.json@{val_sha}", "--one-word-corpus", "rit-text-r6",
            "--output", str(tmp_path / out), "--summary", str(tmp_path / (out + ".summary.json")), *extra]


def test_video_reports_reruns_and_crashed_pass(tmp_path, monkeypatch):
    meta_sha, states, mix_sha, val_sha = _store(tmp_path, monkeypatch)
    # a clip whose best result is still failed (S0[3]) stops the build: a rerun of its shard would recover it
    with pytest.raises(SystemExit, match="without a final extraction result.*drop:extraction_failed"):
        job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha, strict=True))
    assert not (tmp_path / "manifest.json").exists()
    assert job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha)) == 0
    summary = json.loads((tmp_path / "manifest.json.summary.json").read_text())
    manifest = load_manifest(tmp_path / "manifest.json", expected_sha256=summary["manifest_sha256"])
    by = {r["source"]["video_id"]: r for r in manifest["records"]}
    counts = summary["counts"]
    # the latest final result per clip wins; non-final later results keep the earlier state
    assert by[S0[0]]["continuous"]["state_sha256"] == states[(S0[0], "train")][1]          # error, then resumed
    assert by[S0[0]]["source"]["report"].endswith("T15:00:00.000000Z.json")
    assert by[S0[1]]["source"]["report"].endswith("T15:00:00.000000Z.json")                 # already_present
    assert by[S0[2]]["continuous"]["state_sha256"] == states[(S0[2], "train")][1]           # exists_different later
    assert by[S0[2]]["source"]["report"].endswith("T13:00:00.000000Z.json")
    assert S0[5] in by and S0[6] in by                                                       # error later; absent later
    assert S0[3] not in by and S0[4] not in by
    assert counts["drop:extraction_failed"] == 1 and counts["drop:extraction_refused"] == 1
    assert counts["info:later_exists_different_kept_earlier_final"] == 1
    assert counts["info:later_error_kept_earlier_final"] == 1
    assert counts["info:final_after_a_non_final_result"] == 1
    # a later refusal never replaces a hashed state
    assert by[S0[9]]["continuous"]["state_sha256"] == states[(S0[9], "train")][1]
    assert counts["info:refused_after_a_stored_state_kept_the_state"] == 1
    # the crashed pass is not used: its wrong state hash would have stopped the build
    assert by[S1[0]]["continuous"]["state_sha256"] == states[(S1[0], "train")][1]
    ignored = summary["reports"]["ignored"]
    assert len(ignored) == 1 and "not a full pass" in ignored[0]["reason"] and "T13:05:00" in ignored[0]["report"]
    assert summary["reports"]["shards"]["train"]["reports_per_shard"] == {"0": 2, "1": 1}
    # partitions: val and train kept, train+val listing kept in train only, test left out, the gap counted
    assert by[BOTH]["partition"] == "train" and by[BOTH]["continuous"]["state_key"].endswith(f"train/{BOTH}.npz")
    assert {r["partition"] for r in manifest["records"]} == {"train", "val"}
    assert counts["skip:test_split_not_built_held_out"] == 1 and counts["drop:not_in_archive_pinned_gap"] == 1
    assert counts["info:train_val_listing_kept_in_train"] == 1
    assert manifest["split"]["signer_disjoint"] is False and set(manifest["split"]["signer_overlap"]) == {"val&train"}
    assert manifest["split"]["counts"]["test"] == 0
    # text, provenance and the record's source
    assert by[S0[0]]["text"] == "dog" and by[VAL[1]]["text"] == "ice"
    source = by[S0[0]]["source"]
    assert source["extraction"] == "semlex-video-v1" and source["landmarks_key"] == f"{LANDMARKS}/train/{S0[0]}.npz"
    assert source["landmarks_sha256"] == "9" * 64 and source["video_sha256"] == "c" * 64
    prov = manifest["provenance"]
    assert prov["source"] == "video_extraction" and prov["convention"] == "generator"
    assert prov["mediapipe_tasks_models"] == MODELS and prov["convention_settings"] == CONVENTION
    assert prov["archives"] == {"val": VAL_ARCHIVE, "train": TRAIN_ARCHIVE} and "held out" in prov["test_split"]
    assert prov["landmarks_prefixes"] == [LANDMARKS]
    assert sum(k.startswith("report:") for k in manifest["inputs"]) == 4                     # the used reports, hashed
    for r in manifest["records"]:
        lo, hi = r["continuous"]["window_seconds"]
        assert 0 <= lo < hi and r["usable"]["continuous"] and r["corpus"] == "semlex-text-v1"
    # a record loads as training reads it: hash-checked, masked, cut to its window
    from signtranslator.data.text_motion_dataset import DirectoryStore, TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample
    cache = ContentCache(tmp_path / "cache", 2 ** 26)
    record = by[S0[0]]
    state = load_sample(record, "continuous", cache=cache, fetch=DirectoryStore(tmp_path).get)
    lo, hi = record["continuous"]["window_seconds"]
    assert float(state.timestamps[-1] - state.timestamps[0]) <= hi - lo + 1e-6
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=DirectoryStore(tmp_path).get, fps=24.)
    text, served = train[0]
    assert text == train.records[0]["text"] and len(served.timestamps) > 1
    # coverage: R5 = the reliable alphabetic glosses in How2Sign's training words: CAT, DOG, HELLO, ICE (not ZEBRA, A1);
    # DOG already has 5 one-word clips; Sem-Lex brings CAT (5 train clips) and ICE (3)
    cov = summary["coverage"]
    assert cov["r5_words"] == 4 and cov["covered_before"] == 1 and sorted(cov["clips_per_word"]) == ["CAT", "HELLO", "ICE"]
    assert cov["words"] == ["CAT", "ICE"] and cov["words_with_5"] == ["CAT"] and cov["still_uncovered"] == ["HELLO"]
    assert cov["clips_per_word"]["CAT"] == 5 and cov["covered_before_5"] == 1 and cov["covered_after_5"] == 2
    assert cov["ceiling_before"] == .25 and cov["projected_ceiling"] == .75 and cov["projected_ceiling_5"] == .5
    d = summary["describe"]
    assert d["all"]["records"] == len(manifest["records"]) and d["train"]["signers"] == 2
    assert d["all"]["trimmed_seconds"]["median"] > 0 and d["val"]["texts"] == 2
    assert d["val"]["signers_left_by_forearm_tracking"] == 1 and d["val"]["clips_tracking_left_forearm_more"] == 1
    assert d["train"]["signers_left_by_forearm_tracking"] == 2
    assert by[VAL[1]]["quality"]["forearm_tracked"] == {"Left": .1, "Right": .9}
    with pytest.raises(FileExistsError):                                                     # written once
        job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha))


def test_video_workers_match_one_process(tmp_path, monkeypatch):
    meta_sha, _, mix_sha, val_sha = _store(tmp_path, monkeypatch)
    assert job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha, "one.json")) == 0
    # the spawned workers need the patched pins only in the parent (reports are read there)
    assert job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha, "two.json", "--workers", "2")) == 0
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()


def test_video_refusals(tmp_path, monkeypatch):
    meta_sha, states, mix_sha, val_sha = _store(tmp_path, monkeypatch)
    reports = tmp_path / PREFIX / "reports"
    argv = _argv(tmp_path, meta_sha, mix_sha, val_sha, "m.json")
    # a state whose bytes differ from its report
    key = tmp_path / states[(S0[7], "train")][0]
    good = key.read_bytes()
    key.write_bytes(good + b"x")
    with pytest.raises(SystemExit, match="differ from the report"):
        job.main(argv)
    # the coverage inputs are checked before any state is read (a bad pin fails in seconds, not after the pass)
    bad = [a if not a.startswith("runs/validation.json@") else "runs/validation.json@" + "0" * 64 for a in argv]
    with pytest.raises(SystemExit, match="differ from the pinned SHA-256"):
        job.main(bad)
    key.write_bytes(good)
    # two final results naming different states
    rerun = next(reports.glob("train-shard-000-of-002-2026-10-28T15*"))
    payload = rerun.read_text()
    data = json.loads(payload)
    data["results"][1]["state_sha256"] = "6" * 64
    rerun.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="two reports name different states"):
        job.main(argv)
    rerun.write_text(payload)
    # another model set
    _report(tmp_path, "val", 0, 1, "2026-10-28T17:00:00.000000Z", [], meta_sha, models=dict(MODELS, x="0" * 64))
    with pytest.raises(SystemExit, match="more than one MediaPipe Tasks model set"):
        job.main(argv)
    next(reports.glob("val-shard-000-of-001-2026-10-28T17*")).unlink()
    # a shard without a full-pass report: refused, unless a smoke build
    redo = next(reports.glob("train-shard-001-of-002-2026-10-28T16*"))
    saved = redo.read_bytes()
    redo.unlink()
    with pytest.raises(SystemExit, match=r"shard\(s\) \[1\] of 2 have no full-pass report"):
        job.main(argv)
    assert job.main(argv[:-4] + ["--output", str(tmp_path / "smoke.json"), "--limit", "3"]) == 0
    smoke = json.loads((tmp_path / "smoke.json").read_text())
    assert len(smoke["records"]) <= 3 and smoke["smoke"]["limit"] == 3
    with pytest.raises(SystemExit, match="uploads only under"):
        job.main(argv[:-4] + ["--output", str(tmp_path / "smoke2.json"), "--limit", "3", "--output-key", "x/m.json"])
    redo.write_bytes(saved)
    # a result in a report of another shard
    data = json.loads(saved)
    data["results"].append(_result(S0[7], "train", "converted", states))
    redo.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="belongs to shard"):
        job.main(argv)
    redo.write_bytes(saved)
    # another metadata file
    with pytest.raises(SystemExit, match="pinned acquisition"):
        job.main(argv[:5] + ["0" * 64] + argv[6:])


def test_video_uploads_once(tmp_path, monkeypatch):
    meta_sha, _, mix_sha, val_sha = _store(tmp_path, monkeypatch)
    out_key, sum_key = f"{BASE}/semlex-text-v1/manifest.json", f"{BASE}/semlex-text-v1/summary.json"
    argv = _argv(tmp_path, meta_sha, mix_sha, val_sha, "m.json", "--output-key", out_key, "--summary-key", sum_key)
    assert job.main(argv) == 0
    summary = json.loads((tmp_path / sum_key).read_text())
    stored = (tmp_path / out_key).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == summary["manifest_sha256"] and not stored.endswith(b"\n")
    assert (tmp_path / "m.json").read_bytes() == stored + b"\n" and summary["manifest_key"] == out_key
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        job.main(_argv(tmp_path, meta_sha, mix_sha, val_sha, "m2.json", "--output-key", out_key))
