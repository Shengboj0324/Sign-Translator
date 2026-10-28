"""Sem-Lex un-mirroring job on a fake store: re-solve from flipped raw landmarks, checked, write-once, reported."""
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import convert_semlex_poses  # noqa: E402
import unmirror_semlex_states as job  # noqa: E402
from extract_semlex_videos import solve, tag_landmarks  # noqa: E402
from test_build_semlex_manifest_video import LANDMARKS, PREFIX, TRAIN_ARCHIVE, VAL_ARCHIVE, _report  # noqa: E402
from test_flip_landmarks import _clip  # noqa: E402

from signtranslator.data_engineering.video_landmarks import (  # noqa: E402
    flip_landmarks, load_raw_landmarks, save_raw_landmarks)
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
OUT = f"{BASE}/states-video-v1-unmirrored"
SMOKE = f"{BASE}/adapter-dev/smoke/states-video-v1-unmirrored"
METHOD = "monocular estimate: synthetic; generator convention"
IDS = [f"vidUNM{k:04d}" for k in range(5)]


def _store(tmp_path, monkeypatch):
    pins = tmp_path / "known_missing.json"
    pins.write_text(json.dumps(dict(splits=dict(val=dict(archive_sha256=VAL_ARCHIVE, format="webm", missing=[]),
                                                train=dict(archive_sha256=TRAIN_ARCHIVE, format="webm", missing=[])))))
    monkeypatch.setattr(convert_semlex_poses, "KNOWN_MISSING", pins)
    results = []
    for k, video_id in enumerate(IDS):
        raw = tag_landmarks(save_raw_landmarks(_clip(np.random.default_rng(k), right_motion=.1)), METHOD)
        lkey, skey = f"{LANDMARKS}/val/{video_id}.npz", f"{PREFIX}/val/{video_id}.npz"
        (tmp_path / lkey).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / lkey).write_bytes(raw)
        state, _ = solve(load_raw_landmarks(raw), video_id, "c" * 64, METHOD)      # as the extractor solves
        (tmp_path / skey).parent.mkdir(parents=True, exist_ok=True)
        sha = state.save(tmp_path / skey)
        status = "refused" if k == 4 else "converted"
        results.append(dict(video_id=video_id, split="val", status=status, source_sha256="c" * 64, index=k,
                            **({} if k == 4 else dict(state_key=skey, state_sha256=sha, landmarks_key=lkey,
                                                      landmarks_sha256=hashlib.sha256(raw).hexdigest()))))
    _report(tmp_path, "val", 0, 1, "2026-10-28T12:00:00.000000Z", results, "f" * 64)
    return results


def _state(tmp_path, key, sha):
    return MultichannelMotion.load(tmp_path / key, expected_sha256=sha)


def test_unmirror_writes_checked_states_once(tmp_path, monkeypatch, capsys):
    source = _store(tmp_path, monkeypatch)
    argv = ["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"]
    assert job.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status_counts"] == {"converted": 4} and report["expected"] == 4     # the refused clip is not a job
    assert report["check_max_abs"] < 1e-9
    full = json.loads((tmp_path / report["report_key"]).read_text())
    assert full["states_prefix"] == OUT and not full["smoke"]
    for r, src in zip(full["results"], source):
        assert r["state_key"] == f"{OUT}/val/{src['video_id']}.npz" and r["source_state_sha256"] == src["state_sha256"]
        new = _state(tmp_path, r["state_key"], r["state_sha256"])
        raw = (tmp_path / src["landmarks_key"]).read_bytes()
        want, _ = solve(flip_landmarks(load_raw_landmarks(raw)), src["video_id"], "c" * 64, METHOD + job.UNMIRROR_METHOD)
        for name, channel in want.channels.items():                   # exactly the solve of the flipped landmarks
            np.testing.assert_array_equal(new.channels[name].valid, channel.valid, err_msg=name)
            np.testing.assert_allclose(new.channels[name].values, channel.values, atol=1e-12, err_msg=name)
        assert "un-mirrored" in new.channels["body"].inference_method
        assert r["tracked"]["forearm"]["Left"] == 1.                   # the swinging right arm is now the left
    # a rerun resumes: every state is already there
    assert job.main(argv) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["status_counts"] == {"already_present": 4} and again["report_key"] != report["report_key"]


def test_unmirror_refusals_and_smoke(tmp_path, monkeypatch, capsys):
    source = _store(tmp_path, monkeypatch)
    argv = ["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"]
    # smoke: under the smoke prefix only
    assert job.main(argv + ["--limit", "2"]) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["report_key"].startswith(SMOKE + "/") and smoke["status_counts"] == {"converted": 2}
    assert not (tmp_path / OUT).exists()
    # tampered raw landmarks: failed, nothing written for that clip, exit 3
    lkey = tmp_path / source[1]["landmarks_key"]
    lkey.write_bytes(lkey.read_bytes() + b"x")
    assert job.main(argv) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["status_counts"] == {"converted": 3, "failed": 1}
    assert not (tmp_path / OUT / "val" / f"{source[1]['video_id']}.npz").exists()
    # a written state that differs from this run's: exists_different, exit 3
    other = tmp_path / OUT / "val" / f"{source[0]['video_id']}.npz"
    state = _state(tmp_path, f"{OUT}/val/{source[0]['video_id']}.npz", hashlib.sha256(other.read_bytes()).hexdigest())
    other.unlink()
    from dataclasses import replace
    body = state.channels["body"]
    replace(state, channels=dict(state.channels, body=replace(body, values=body.values + .01 * body.valid[..., None]))) \
        .save(other)
    assert job.main(argv) == 3
    assert json.loads(capsys.readouterr().out)["status_counts"].get("exists_different") == 1


def test_the_check_refuses_a_solve_that_is_not_the_mirror(tmp_path, monkeypatch, capsys):
    _store(tmp_path, monkeypatch)
    from signtranslator.data_engineering import video_landmarks
    real = video_landmarks.flip_landmarks
    monkeypatch.setattr(video_landmarks, "flip_landmarks", lambda clip: real(real(clip)))   # a no-op "flip"
    assert job.main(["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"]) == 3
    out = json.loads(capsys.readouterr().out)
    assert out["status_counts"] == {"failed": 4} and not (tmp_path / OUT / "val").exists()


def _metadata(tmp_path):
    """Metadata, ASL-LEX and a word list for the clips (all val, signer 8, label cat_3)."""
    import csv
    import io
    text = io.StringIO()
    w = csv.writer(text)
    w.writerow(["", "video_id", "signer_id", "duration", "split", "label_type", "label"])
    for k, v in enumerate(IDS):
        w.writerow([k, v, "8", 400, "val", "asllex", "cat_3"])            # 12 frames at 30 fps
    meta = text.getvalue().encode()
    (tmp_path / BASE / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / BASE / "data/semlex_metadata.csv").write_bytes(meta)
    lex = b"EntryID,LemmaID,Code,SignBankReferenceID\ncat_3,cat_3,C_02_002,963\n"
    key = tmp_path / "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(lex)
    Path(str(key) + ".provenance.json").write_text(json.dumps(dict(sha256=hashlib.sha256(lex).hexdigest(),
                                                                   verified_by_r2_reread=True)))
    (tmp_path / "words.txt").write_text("cat\n")
    return hashlib.sha256(meta).hexdigest()


def test_the_manifest_takes_the_unmirrored_states(tmp_path, monkeypatch, capsys):
    import build_semlex_manifest as builder
    source = _store(tmp_path, monkeypatch)
    meta_sha = _metadata(tmp_path)
    report = next((tmp_path / PREFIX / "reports").glob("val-*"))
    data = json.loads(report.read_text())
    data["metadata_sha256"] = meta_sha
    report.write_text(json.dumps(data))
    argv = ["--store", f"dir:{tmp_path}", "--source", "video", "--splits", "val", "--metadata-sha256", meta_sha,
            "--wordlist", str(tmp_path / "words.txt"), "--states-prefix", OUT, "--min-hand", "0", "--min-seconds", ".1"]
    with pytest.raises(SystemExit, match="un-mirror split val"):                         # no un-mirror report yet
        builder.main(argv + ["--output", str(tmp_path / "m0.json")])
    assert job.main(["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"]) == 0
    capsys.readouterr()
    assert builder.main(argv + ["--output", str(tmp_path / "m.json")]) == 0
    manifest = json.loads((tmp_path / "m.json").read_text())
    assert manifest["unmirrored"] is True and manifest["provenance"]["unmirrored"] is True
    assert manifest["provenance"]["states_prefix"] == OUT and manifest["provenance"]["extraction_states_prefix"] == PREFIX
    assert any(k.startswith("report:unmirror:") for k in manifest["inputs"])
    assert manifest["records"]
    for r in manifest["records"]:
        src = next(s for s in source if s["video_id"] == r["source"]["video_id"])
        assert r["continuous"]["state_key"] == f"{OUT}/val/{src['video_id']}.npz"
        assert r["source"]["unmirrored"] is True and r["source"]["extracted_state_sha256"] == src["state_sha256"]
        assert r["continuous"]["state_sha256"] != src["state_sha256"]
    # an un-mirror result that names another extracted state is refused
    um = next((tmp_path / OUT / "reports").glob("val-*"))
    data = json.loads(um.read_text())
    data["results"][0]["source_state_sha256"] = "0" * 64
    um.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="not of the extracted state"):
        builder.main(argv + ["--output", str(tmp_path / "m2.json")])
    # a clip without an un-mirror result stops the build unless --allow-unfinished
    data["results"] = data["results"][1:]
    um.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="drop:unmirror_missing"):
        builder.main(argv + ["--output", str(tmp_path / "m3.json")])
    assert builder.main(argv + ["--output", str(tmp_path / "m4.json"), "--allow-unfinished"]) == 0
    assert len(json.loads((tmp_path / "m4.json").read_text())["records"]) == len(manifest["records"]) - 1


def test_dominance_switches_on_the_unmirrored_states(tmp_path, monkeypatch, capsys):
    import handedness_check
    _store(tmp_path, monkeypatch)
    _metadata(tmp_path)
    monkeypatch.setattr("convert_semlex_poses.METADATA_SHA256", hashlib.sha256(
        (tmp_path / BASE / "data/semlex_metadata.csv").read_bytes()).hexdigest())
    assert job.main(["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"]) == 0
    capsys.readouterr()
    sides = {}
    for name, extra in (("stored", []), ("unmirrored", ["--semlex-states-prefix", OUT])):
        out = tmp_path / f"{name}.json"
        assert handedness_check.main(["--store", f"dir:{tmp_path}", "--semlex", "--semlex-splits", "val",
                                      "--output", str(out), *extra]) == 0
        report = json.loads(out.read_text())
        sides[name] = {c["clip"]: c["side"] for c in report["clips"]}
        assert report["inputs"]["semlex_states_prefix"].startswith(OUT if extra else "states-video-v1")
    assert set(sides["stored"].values()) == {"Right"} and set(sides["unmirrored"].values()) == {"Left"}
    assert sides["stored"].keys() == sides["unmirrored"].keys() and len(sides["stored"]) == 4


def test_a_missing_extraction_shard_stops_the_job(tmp_path, monkeypatch):
    _store(tmp_path, monkeypatch)
    report = next((tmp_path / PREFIX / "reports").glob("val-*"))
    data = json.loads(report.read_text())
    data["shards"] = 2
    data["results"] = [r for r in data["results"] if zlib.crc32(r["video_id"].encode()) % 2 == 0]
    report.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="no full-pass report"):
        job.main(["--split", "val", "--store", f"dir:{tmp_path}", "--workers", "0"])
