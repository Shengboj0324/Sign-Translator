"""UWB converter script: provenance, clock source, refusals, resumable and sharded units, restricted runs."""
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

from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402

PREFIX = "acq/uwb"
CLIPS = {"vidA.000100-000107": dict(seed=1), "vidA.000107-000114": dict(seed=2, missing_left=(0, 1, 2)),
         "vidB.000050-000057": dict(seed=3), "vidB.000058-000065": None}       # None: a malformed member


def make_store(root: Path) -> tuple[str, str]:
    base = root / PREFIX
    base.mkdir(parents=True)
    with zipfile.ZipFile(base / "raw_keypoints_1.zip", "w") as z:
        for clip, kw in CLIPS.items():
            payload = b"{broken" if kw is None else json.dumps(uwb_clip_json(**kw)[0]).encode()
            z.writestr(f"{clip}.json", payload)
    translations = {"vidA": {"clip_order": ["vidA.000100-000107", "vidA.000107-000114"],
                             "vidA.000100-000107": {"translation": "Hello   there."},
                             "vidA.000107-000114": {"translation": "Second  cue"}}}
    (base / "YT.translations.train.json").write_text(json.dumps(translations))
    (base / "YT.translations.dev.json").write_text(json.dumps(
        {"vidB": {"clip_order": ["vidB.000050-000057"], "vidB.000050-000057": {"translation": "dev text"}}}))
    payload = xy_bundle().save()
    (root / "xy.pt").write_bytes(payload)
    return str(root / "xy.pt"), hashlib.sha256(payload).hexdigest()


def run(root, bundle, sha, *extra):
    return job.main(["--store", f"dir:{root}", "--prefix", PREFIX, "--out-prefix", "out/uwb", "--regressors", bundle,
                     "--regressors-sha256", sha, "--face", "none", "--chunk-size", "2", "--workers", "1",
                     *(() if "--visibility" in extra else ("--visibility", "learned")), *extra])


def reports(root):
    return sorted((root / "out/uwb/reports/raw_keypoints_1").glob("chunk-*.json"))


def test_convert_writes_states_with_provenance_and_refusals(tmp_path):
    bundle, sha = make_store(tmp_path)
    table = tmp_path / "fps.json"
    table.write_text(json.dumps({"schema": "uwb-fps-table-v1", "fps": {"vidB": 25.}}))
    assert run(tmp_path, bundle, sha, "--fps-table", str(table)) == 0
    files = reports(tmp_path)
    assert [f.name for f in files] == ["chunk-00000.json", "chunk-00001.json"]
    clips = {c["member"]: c for f in files for c in json.loads(f.read_text())["clips"]}
    assert set(clips) == {f"{c}.json" for c in CLIPS}
    a = clips["vidA.000100-000107.json"]
    assert a["status"] == "converted" and a["text"] == "Hello there." and a["partition"] == "train"
    assert a["fps"] == 30. and a["fps_source"] == "assumed" and a["first_frame"] == 100 and a["last_frame"] == 107
    assert a["zip"] == "raw_keypoints_1.zip" and a["video_id"] == "vidA" and a["image_size"] == [1280, 720]
    member = zipfile.ZipFile(tmp_path / PREFIX / "raw_keypoints_1.zip").read("vidA.000100-000107.json")
    assert a["member_sha256"] == hashlib.sha256(member).hexdigest() and a["regressors_sha256"] == sha
    state = MultichannelMotion.load(tmp_path / a["state_key"], expected_sha256=a["state_sha256"])
    np.testing.assert_allclose(state.timestamps, (100 + np.arange(8)) / 30.)
    assert state.channels["body"].source_sha256 == a["member_sha256"]
    b = clips["vidB.000050-000057.json"]
    assert b["fps"] == 25. and b["fps_source"] == "estimated" and b["partition"] == "monitor"
    second = clips["vidA.000107-000114.json"]
    assert second["hand_fraction"]["Left"] < second["hand_fraction"]["Right"]
    broken = clips["vidB.000058-000065.json"]
    assert broken["status"] == "refused" and "unreadable JSON" in broken["reason"] and broken["text"] is None
    index = json.loads((tmp_path / "out/uwb/index.json").read_text())
    assert index["counts"] == {"converted": 3, "refused": 1} and index["units_finished"] == index["units_total"] == 2
    assert index["fps_sources"] == {"assumed": 2, "estimated": 1} and "CC BY 4.0" in index["licence"]


def test_finished_units_are_skipped_and_shards_partition_the_units(tmp_path, capsys):
    bundle, sha = make_store(tmp_path)
    assert run(tmp_path, bundle, sha, "--shard", "1", "--shards", "2") == 0
    assert [f.name for f in reports(tmp_path)] == ["chunk-00001.json"]
    before = reports(tmp_path)[0].read_bytes()
    assert run(tmp_path, bundle, sha, "--shard", "0", "--shards", "2") == 0
    assert [f.name for f in reports(tmp_path)] == ["chunk-00000.json", "chunk-00001.json"]
    capsys.readouterr()
    assert run(tmp_path, bundle, sha) == 0
    out = capsys.readouterr().out
    assert out.count('"status": "present"') == 2 and reports(tmp_path)[1].read_bytes() == before


def test_restricted_runs_never_mark_a_unit_finished(tmp_path):
    bundle, sha = make_store(tmp_path)
    assert run(tmp_path, bundle, sha, "--limit", "1") == 0
    names = sorted(p.name for p in (tmp_path / "out/uwb/reports/raw_keypoints_1").iterdir())
    assert names == ["chunk-00000.partial.json", "chunk-00001.partial.json"]
    assert json.loads((tmp_path / "out/uwb/index.json").read_text())["units_finished"] == 0
    assert run(tmp_path, bundle, sha, "--video", "vidB") == 0
    partial = json.loads((tmp_path / "out/uwb/reports/raw_keypoints_1/chunk-00001.partial.json").read_text())
    assert {c["video_id"] for c in partial["clips"]} == {"vidB"}


def test_bad_bundles_and_arguments_are_refused(tmp_path):
    bundle, sha = make_store(tmp_path)
    v1 = small_bundle().save()
    (tmp_path / "v1.pt").write_bytes(v1)
    with pytest.raises(SystemExit, match="UWB needs"):
        run(tmp_path, str(tmp_path / "v1.pt"), hashlib.sha256(v1).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        run(tmp_path, bundle, "0" * 64)
    with pytest.raises(SystemExit, match="must differ"):
        job.main(["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--out-prefix", PREFIX, "--world", "pseudo",
                  "--visibility", "inside"])
    with pytest.raises(SystemExit, match="visibility inside"):
        job.main(["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--out-prefix", "o", "--world", "pseudo",
                  "--visibility", "learned"])
    with pytest.raises(SystemExit):                        # the visibility rule is never left to a default
        job.main(["--store", f"dir:{tmp_path}", "--prefix", PREFIX, "--out-prefix", "o", "--regressors", bundle,
                  "--regressors-sha256", sha])


def test_inside_visibility_is_recorded_in_the_states(tmp_path):
    bundle, sha = make_store(tmp_path)
    assert run(tmp_path, bundle, sha, "--visibility", "inside", "--video", "vidA") == 0
    partial = json.loads((tmp_path / "out/uwb/reports/raw_keypoints_1/chunk-00000.partial.json").read_text())
    assert partial["settings"]["visibility"] == "inside"
    clip = partial["clips"][0]
    state = MultichannelMotion.load(tmp_path / clip["state_key"], expected_sha256=clip["state_sha256"])
    assert "visibility = finite and inside the image" in state.channels["body"].inference_method
    with pytest.raises(SystemExit, match="KEY@SHA256"):
        run(tmp_path, bundle, sha, "--fps-table", "nothing-here")
