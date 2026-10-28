"""Sem-Lex manifest builder: texts, partitions, filters, trimming, provenance and coverage on a fake store."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_semlex_manifest as job  # noqa: E402
from semlex_synthetic import FakeBlendshapes, fake_resources, semlex_array  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.data_engineering.semlex_pose import convert_array  # noqa: E402

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
PREFIX = f"{BASE}/states-test"
WORDS = "cat\ndog\nfall\nasleep\nhello\nRoosevelt\nice\n"


def test_texts():
    assert job.asllex_text("cat_3") == "cat" and job.asllex_text("hard_of_hearing") == "hard of hearing"
    assert job.asllex_text("mcdonalds") == "mcdonalds"
    words = {"fall", "asleep", "cat", "dog's", "dog"}
    assert job.freetext_text("fall_asleep", words) == "fall asleep"
    assert job.freetext_text(" cat ", words) == "cat" and job.freetext_text("cat_2", words) == "cat"
    for bad in ("roosevelt", "5_grader", "@", "tempature", "cat dog cat dog cat", "Cat", ""):
        assert job.freetext_text(bad, words) is None, bad


def _store(tmp_path):
    """Metadata, ASL-LEX, states and one shard report in a directory store."""
    rows = [  # video_id, signer, split, label_type, label, duration (ms)
        ("vidTRAIN0000000000001", "7", "train", "asllex", "ice_2", 30000 / 30),
        ("vidTRAIN0000000000002", "8", "train", "freetext", "fall_asleep", 30000 / 30),
        ("vidTRAIN0000000000003", "8", "train", "freetext", "roosevelt", 30000 / 30),
        ("vidTRAIN0000000000004", "7", "train", "signbank", "963", 30000 / 30),
        ("vidVAL00000000000005", "8", "val", "asllex", "cat_3", 30000 / 30),
        ("vidTEST0000000000006", "9", "test", "asllex", "cat_3", 30000 / 30),
        ("vidTRAIN0000000000007", "7", "train", "asllex", "cat_3", 30000 / 30),
        ("vidSLOW0000000000008", "7", "train", "asllex", "cat_3", 30000 / 15),    # 15 fps
        ("vidREFUSED0000000009", "7", "train", "asllex", "cat_3", 30000 / 30),
    ]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["", "video_id", "signer_id", "duration", "split", "label_type", "label"])
    for k, (v, s, sp, t, l, d) in enumerate(rows):
        writer.writerow([k, v, s, d, sp, t, l])
    meta = text.getvalue().encode()
    (tmp_path / BASE / "data").mkdir(parents=True)
    (tmp_path / BASE / "data/semlex_metadata.csv").write_bytes(meta)
    lex = io.StringIO()
    w = csv.writer(lex)
    w.writerow(["EntryID", "LemmaID", "Code", "SignBankReferenceID"])
    w.writerows([("ice_2", "ice_2", "C_01_001", "100"), ("cat_3", "cat_3", "C_02_002", "963")])
    lex_bytes = lex.getvalue().encode()
    lex_key = tmp_path / "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
    lex_key.parent.mkdir(parents=True)
    lex_key.write_bytes(lex_bytes)
    Path(str(lex_key) + ".provenance.json").write_text(json.dumps(dict(sha256=hashlib.sha256(lex_bytes).hexdigest(),
                                                                       verified_by_r2_reread=True)))
    results = []
    for k, (v, s, sp, t, l, d) in enumerate(rows):
        if v.startswith("vidREFUSED"):
            results.append(dict(video_id=v, split=sp, status="refused", reason="aspect"))
            continue
        array, *_ = semlex_array((640, 640), frames=30, seed=5)          # a seed whose arms stay in frame
        state, report = convert_array(array, d, fake_resources(), video_id=v, source_sha256="a" * 64,
                                      blendshape_model=FakeBlendshapes(), image_size=(640, 640), pose_z_scale=1.,
                                      hand_z_scale=1., flags="generator")
        key = f"{PREFIX}/{sp}/{v}.npz"
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = state.save(path)
        results.append(dict(video_id=v, split=sp, status="converted", state_key=key, state_sha256=digest,
                            adapter="semlex-holistic-learned-world-v2", regressors_sha256="b" * 64))
    (tmp_path / PREFIX / "reports").mkdir(parents=True)
    (tmp_path / PREFIX / "reports/test-shard-000-of-001-x.json").write_text(json.dumps(dict(
        regressors_sha256="b" * 64, results=results)))
    (tmp_path / "words.txt").write_text(WORDS)
    (tmp_path / "r5.json").write_text(json.dumps(dict(r5=["CAT", "ICE", "DOG", "HELLO"],
                                                      round13=dict(covered={"CAT": 2, "HELLO": 1},
                                                                   uncovered={"ICE": 0, "DOG": 0}))))
    return hashlib.sha256(meta).hexdigest()


def test_builds_the_manifest(tmp_path):
    meta_sha = _store(tmp_path)
    out = tmp_path / "manifest.json"
    argv = ["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--metadata-sha256", meta_sha,
            "--wordlist", str(tmp_path / "words.txt"), "--r5", str(tmp_path / "r5.json"), "--output", str(out),
            "--summary", str(tmp_path / "summary.json")]
    assert job.main(argv) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    manifest = load_manifest(out, expected_sha256=summary["manifest_sha256"])
    by = {r["source"]["video_id"]: r for r in manifest["records"]}
    counts = summary["counts"]
    assert counts["skip:freetext_not_plain_english"] == 1 and counts["skip:signbank_no_english_in_metadata"] == 1
    assert counts["info:signbank_mappable_through_asllex"] == 1
    assert counts["drop:conversion_refused"] == 1 and counts["drop:fps_below_min"] == 1
    assert by["vidTRAIN0000000000001"]["text"] == "ice" and by["vidTRAIN0000000000001"]["gloss"]["variant"] == "ice_2"
    assert by["vidTRAIN0000000000001"]["gloss"]["asl_lex_code"] == "C_01_001"
    assert by["vidTRAIN0000000000002"]["text"] == "fall asleep"
    assert {r["partition"] for r in manifest["records"]} <= {"train", "val", "test"}
    assert by["vidTEST0000000000006"]["partition"] == "test" and by["vidVAL00000000000005"]["partition"] == "val"
    assert by["vidTRAIN0000000000007"]["performer_sha256"] == hashlib.sha256(b"semlex-signer:7").hexdigest()
    for r in manifest["records"]:
        lo, hi = r["continuous"]["window_seconds"]
        assert 0 <= lo < hi and r["usable"]["continuous"] and r["corpus"] == "semlex-text-v1"
    assert manifest["split"]["signer_disjoint"] is False and manifest["split"]["signer_overlap"]["train&val"] == 1
    assert "CC BY-NC-SA 4.0" in manifest["license"] and "2310.00196" in manifest["provenance"]["citation"]
    assert manifest["provenance"]["regressors_sha256"] == "b" * 64
    cov = summary["coverage"]
    assert cov["now_with_semlex_train_clips"] == 1 and cov["words"] == ["ICE"] and cov["still_uncovered"] == ["DOG"]
    assert cov["projected_ceiling"] == .75 and cov["ceiling_before"] == .5
    with pytest.raises(FileExistsError):                       # written once
        job.main(argv)


def test_mixed_bundles_and_unverified_inputs_are_refused(tmp_path):
    meta_sha = _store(tmp_path)
    reports = tmp_path / PREFIX / "reports"
    (reports / "test-shard-000-of-001-y.json").write_text(json.dumps(dict(regressors_sha256="c" * 64, results=[])))
    argv = ["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--metadata-sha256", meta_sha,
            "--wordlist", str(tmp_path / "words.txt"), "--output", str(tmp_path / "m.json")]
    with pytest.raises(SystemExit, match="more than one regressor bundle"):
        job.main(argv)
    (reports / "test-shard-000-of-001-y.json").unlink()
    with pytest.raises(SystemExit, match="pinned acquisition"):
        job.main(argv[:5] + ["0" * 64] + argv[6:])
    lex = tmp_path / "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
    lex.write_bytes(lex.read_bytes() + b"tampered,x,y,z\n")
    with pytest.raises(SystemExit, match="ASL-LEX"):
        job.main(argv)
