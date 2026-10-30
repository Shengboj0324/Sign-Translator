"""UWB converter script: provenance, clock source, leak and overlap checks, refusals, resumable and sharded units with
a settings check, restricted runs, the merged index and the manifest selection helper."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import convert_uwb_clips as job  # noqa: E402
from test_de_world_regressor import small_bundle  # noqa: E402
from test_de_world_regressor_xy import xy_bundle  # noqa: E402
from uwb_synthetic import uwb_clip_json  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import canonical_json  # noqa: E402
from signtranslator.data_engineering.uwb_manifest import select_training_records  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

PREFIX = "acq/uwb"
CLIPS = {"vidA.000100-000107": dict(seed=1), "vidA.000107-000114": dict(seed=2, missing_left=(0, 1, 2)),
         "vidB.000050-000057": dict(seed=3), "vidB.000058-000065": None}       # None: a malformed member
TEXTS = {"vidA.000100-000107": "Hello   there.", "vidA.000107-000114": "The quick brown fox jumps over the lazy dog",
         "vidB.000050-000057": "dev text"}


def make_store(root: Path) -> dict:
    base = root / PREFIX
    base.mkdir(parents=True)
    with zipfile.ZipFile(base / "raw_keypoints_1.zip", "w") as z:
        for clip, kw in CLIPS.items():
            payload = b"{broken" if kw is None else json.dumps(uwb_clip_json(**kw)[0]).encode()
            z.writestr(f"{clip}.json", payload)
    (base / "YT.translations.train.json").write_text(json.dumps(
        {"vidA": {"clip_order": ["vidA.000100-000107", "vidA.000107-000114"],
                  **{c: {"translation": TEXTS[c]} for c in ("vidA.000100-000107", "vidA.000107-000114")}}}))
    (base / "YT.translations.dev.json").write_text(json.dumps(
        {"vidB": {"clip_order": ["vidB.000050-000057"], "vidB.000050-000057": {"translation": TEXTS["vidB.000050-000057"]}}}))
    payload = xy_bundle().save()
    (root / "xy.pt").write_bytes(payload)
    manifest = dict(records=[dict(partition="test", english="The quick brown fox jumps over the lazy dog!", usable={}),
                             dict(partition="train", english="Hello there", usable={})])
    data = canonical_json(manifest)
    (root / "eval").mkdir()
    (root / "eval/manifest.json").write_bytes(data)
    overlap = dict(schema="uwb-overlap-table-v1",
                   clips={"vidA.000100-000107": dict(our_sample_id="vidA-0000", iou=.93)},
                   videos={"vidA": dict(partition="monitor")})
    (root / "overlap.json").write_text(json.dumps(overlap))
    return dict(bundle=str(root / "xy.pt"), sha=hashlib.sha256(payload).hexdigest(),
                eval=f"eval/manifest.json@{hashlib.sha256(data).hexdigest()}", overlap=str(root / "overlap.json"))


def run(root, s, *extra, visibility="learned"):
    return job.main(["--store", f"dir:{root}", "--prefix", PREFIX, "--out-prefix", "out/uwb", "--regressors", s["bundle"],
                     "--regressors-sha256", s["sha"], "--face", "none", "--chunk-size", "2", "--workers", "1",
                     "--eval-manifest", s["eval"], "--overlap-table", s["overlap"], "--visibility", visibility, *extra])


def reports(root):
    return sorted((root / "out/uwb/reports/raw_keypoints_1").glob("chunk-?????.json"))


def test_convert_writes_states_with_provenance_leak_overlap_and_refusals(tmp_path):
    s = make_store(tmp_path)
    table = tmp_path / "fps.json"
    table.write_text(json.dumps({"schema": "uwb-fps-table-v1", "fps": {"vidB": 25.}}))
    assert run(tmp_path, s, "--fps-table", str(table)) == 0
    files = reports(tmp_path)
    assert [f.name for f in files] == ["chunk-00000.json", "chunk-00001.json"]
    clips = {c["member"]: c for f in files for c in json.loads(f.read_text())["clips"]}
    assert set(clips) == {f"{c}.json" for c in CLIPS}
    a = clips["vidA.000100-000107.json"]
    assert a["status"] == "converted" and a["text"] == "Hello there." and a["uwb_partition"] == "train"
    assert a["partition"] == a["our_partition"] == "monitor" and a["our_video"]          # our channel split wins
    assert a["duplicate_of_our_cue"] == "vidA-0000" and a["our_cue_iou"] == .93 and a["eval_leak"] is False
    assert a["fps"] == 30. and a["fps_source"] == "assumed" and a["first_frame"] == 100 and a["last_frame"] == 107
    assert a["zip"] == "raw_keypoints_1.zip" and a["video_id"] == "vidA" and a["image_size"] == [1280, 720]
    assert "CC BY 4.0" in a["licence"] and a["visibility"] == "learned" and len(a["eval_index_sha256"]) == 64
    member = zipfile.ZipFile(tmp_path / PREFIX / "raw_keypoints_1.zip").read("vidA.000100-000107.json")
    assert a["member_sha256"] == hashlib.sha256(member).hexdigest() and a["regressors_sha256"] == s["sha"]
    state = MultichannelMotion.load(tmp_path / a["state_key"], expected_sha256=a["state_sha256"])
    np.testing.assert_allclose(state.timestamps, (100 + np.arange(8)) / 30.)
    assert state.channels["body"].source_sha256 == a["member_sha256"]
    leak = clips["vidA.000107-000114.json"]
    assert leak["eval_leak"] is True and leak["duplicate_of_our_cue"] is None and leak["partition"] == "monitor"
    b = clips["vidB.000050-000057.json"]
    assert b["fps"] == 25. and b["fps_source"] == "estimated" and b["partition"] == "monitor" and not b["our_video"]
    broken = clips["vidB.000058-000065.json"]
    assert broken["status"] == "refused" and "unreadable JSON" in broken["reason"] and broken["text"] is None
    first = json.loads(files[0].read_text())
    assert first["checks"]["eval_leak"] == 1 and first["checks"]["duplicate_of_our_cue"] == 1
    # the manifest helper keeps only the clean, non-duplicate clip
    records = [c for f in files for c in json.loads(f.read_text())["clips"]]
    kept, why = select_training_records(records, eval_index_sha256=a["eval_index_sha256"])
    assert [r["clip_id"] for r in kept] == ["vidB.000050-000057"]
    assert why == {"kept": 1, "evaluation_leak": 1, "duplicate_of_our_state": 1, "status_refused": 1}
    assert not (tmp_path / "out/uwb/index.json").exists()                           # merged separately
    assert job.main(["--store", f"dir:{tmp_path}", "--out-prefix", "out/uwb", "--merge-index"]) == 0
    index = json.loads((tmp_path / "out/uwb/index.json").read_text())
    assert index["counts"] == {"converted": 3, "refused": 1} and index["units_finished"] == 2
    assert index["checks"]["eval_leak"] == 1 and len({u["settings_fingerprint"] for u in index["units"]}) == 1
    assert index["fps_sources"] == {"assumed": 2, "estimated": 1} and "CC BY 4.0" in index["licence"]
    assert (tmp_path / "out/uwb/shards/shard-000-of-001.json").exists()


def test_the_selection_helper_refuses_unchecked_records():
    ok = dict(status="converted", clip_id="c", text="t", eval_leak=False, eval_index_sha256="a" * 64,
              duplicate_of_our_cue=None, partition="train")
    assert select_training_records([ok])[0] == [ok]
    for field in ("eval_leak", "duplicate_of_our_cue", "partition"):
        with pytest.raises(ValueError, match="leak and overlap checks"):
            select_training_records([{k: v for k, v in ok.items() if k != field}])
    with pytest.raises(ValueError, match="leak and overlap checks"):
        select_training_records([dict(ok, eval_index_sha256=None)])
    with pytest.raises(ValueError, match="another evaluation index"):
        select_training_records([ok], eval_index_sha256="b" * 64)
    assert select_training_records([dict(status="refused")])[1] == {"status_refused": 1}       # nothing to check


def test_finished_units_are_skipped_only_with_the_same_settings_and_shards_partition_the_units(tmp_path, capsys):
    s = make_store(tmp_path)
    assert run(tmp_path, s, "--shard", "1", "--shards", "2") == 0
    assert [f.name for f in reports(tmp_path)] == ["chunk-00001.json"]
    before = reports(tmp_path)[0].read_bytes()
    assert run(tmp_path, s, "--shard", "0", "--shards", "2") == 0
    assert [f.name for f in reports(tmp_path)] == ["chunk-00000.json", "chunk-00001.json"]
    capsys.readouterr()
    assert run(tmp_path, s) == 0
    out = capsys.readouterr().out
    assert out.count('"status": "present"') == 2 and reports(tmp_path)[1].read_bytes() == before
    # another visibility rule, fps default or bundle on a finished unit: refused, nothing rewritten
    assert run(tmp_path, s, visibility="inside") == 2
    assert '"differ": ["visibility"]' in capsys.readouterr().out
    assert run(tmp_path, s, "--fps", "25") == 2 and reports(tmp_path)[1].read_bytes() == before
    other = xy_bundle(seed=1).save()
    (tmp_path / "xy1.pt").write_bytes(other)
    assert run(tmp_path, dict(s, bundle=str(tmp_path / "xy1.pt"), sha=hashlib.sha256(other).hexdigest())) == 2
    # units written under different settings are never merged into one index
    report = json.loads(reports(tmp_path)[0].read_text())
    report["settings"]["visibility"] = "inside"
    report.pop("settings_fingerprint")
    reports(tmp_path)[0].write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="disagree on settings"):
        job.main(["--store", f"dir:{tmp_path}", "--out-prefix", "out/uwb", "--merge-index"])


def test_restricted_runs_never_mark_a_unit_finished(tmp_path):
    s = make_store(tmp_path)
    assert run(tmp_path, s, "--limit", "1") == 0
    names = sorted(p.name for p in (tmp_path / "out/uwb/reports/raw_keypoints_1").iterdir())
    assert names == ["chunk-00000.partial.json", "chunk-00001.partial.json"]
    assert job.main(["--store", f"dir:{tmp_path}", "--out-prefix", "out/uwb", "--merge-index"]) == 0
    assert json.loads((tmp_path / "out/uwb/index.json").read_text())["units_finished"] == 0
    assert run(tmp_path, s, "--video", "vidB") == 0
    partial = json.loads((tmp_path / "out/uwb/reports/raw_keypoints_1/chunk-00001.partial.json").read_text())
    assert {c["video_id"] for c in partial["clips"]} == {"vidB"}


def test_bad_bundles_and_arguments_are_refused(tmp_path):
    s = make_store(tmp_path)
    v1 = small_bundle().save()
    (tmp_path / "v1.pt").write_bytes(v1)
    with pytest.raises(SystemExit, match="UWB needs"):
        run(tmp_path, dict(s, bundle=str(tmp_path / "v1.pt"), sha=hashlib.sha256(v1).hexdigest()))
    with pytest.raises(ValueError, match="SHA-256"):
        run(tmp_path, dict(s, sha="0" * 64))
    base = ["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--regressors", s["bundle"], "--regressors-sha256",
            s["sha"], "--eval-manifest", s["eval"], "--overlap-table", s["overlap"]]
    with pytest.raises(SystemExit, match="must differ"):
        job.main(base + ["--out-prefix", PREFIX, "--visibility", "inside"])
    with pytest.raises(SystemExit, match="visibility inside"):
        job.main(base + ["--out-prefix", "o", "--world", "pseudo", "--visibility", "learned"])
    for rule in (None, "all", "margin:9"):                    # the rule is declared, never a default or a diagnostic
        with pytest.raises(SystemExit):
            job.main(base + ["--out-prefix", "o"] + ([] if rule is None else ["--visibility", rule]))
    with pytest.raises(SystemExit, match="eval-manifest is required"):
        job.main([a for a in base if a not in ("--eval-manifest", s["eval"])] + ["--out-prefix", "o", "--visibility",
                                                                               "inside"])
    with pytest.raises(SystemExit, match="overlap-table is required"):
        job.main([a for a in base if a not in ("--overlap-table", s["overlap"])] + ["--out-prefix", "o",
                                                                                  "--visibility", "inside"])
    with pytest.raises(ValueError, match="hash mismatch"):
        job.main([a if a != s["eval"] else "eval/manifest.json@" + "0" * 64 for a in base]
                 + ["--out-prefix", "o", "--visibility", "inside"])
    with pytest.raises(SystemExit, match="KEY@SHA256"):
        run(tmp_path, s, "--fps-table", "nothing-here")


def test_margin_visibility_is_recorded_in_the_states(tmp_path):
    s = make_store(tmp_path)
    assert run(tmp_path, s, "--video", "vidA", visibility="margin:0.25") == 0
    partial = json.loads((tmp_path / "out/uwb/reports/raw_keypoints_1/chunk-00000.partial.json").read_text())
    assert partial["settings"]["visibility"] == "margin:0.25"
    clip = partial["clips"][0]
    state = MultichannelMotion.load(tmp_path / clip["state_key"], expected_sha256=clip["state_sha256"])
    assert "extended by 0.25 of its size" in state.channels["body"].inference_method
