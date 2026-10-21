"""ASL Citizen manifest: publisher splits joined to converted states, signer-disjointness checked."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_asl_citizen_manifest as builder  # noqa: E402

SHA = "a" * 64
HEADER = "Participant ID,Video file,Gloss,ASL-LEX Code\n"


def setup_store(tmp_path, splits, results, upright=True, extra_reports=()):
    zip_key = "acq/ASL_Citizen.zip"
    (tmp_path / "acq").mkdir()
    with zipfile.ZipFile(tmp_path / zip_key, "w") as z:
        for split, rows in splits.items():
            z.writestr(f"ASL_Citizen/splits/{split}.csv", HEADER + "".join(",".join(r) + "\n" for r in rows))
    (tmp_path / (zip_key + ".provenance.json")).write_text(json.dumps(
        dict(sha256=SHA, verified_by_r2_reread=True, license="research only")))
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep/shard-0.json").write_text(json.dumps(dict(upright_chest=upright, results=results)))
    for i, extra in enumerate(extra_reports, 1):
        (tmp_path / f"rep/shard-{i}.json").write_text(json.dumps(dict(upright_chest=upright, results=extra)))
    return zip_key


def converted(sample, arms=10, hands=8, frames=20):
    return dict(sample_id=sample, status="converted", state_key=f"s/{sample}.npz", state_sha256="b" * 64,
                available_channels=["body", "right_hand"], seconds=1., frames=frames, pose_frames=frames,
                face_frames=frames, left_hand_frames=0, right_hand_frames=hands,
                arm_frames={"LeftForearm": 0, "RightForearm": arms})


def run(tmp_path, zip_key):
    return builder.main(["--store", f"dir:{tmp_path}", "--zip-key", zip_key, "--zip-sha256", SHA,
                         "--reports-prefix", "rep/", "--manifest-key", "out/m.json"])


def test_manifest_joins_publisher_labels_and_partitions(tmp_path):
    splits = {"train": [("P1", "1-LIBRARY.mp4", "LIBRARY", "A_01_001")],
              "val": [("P2", "2-BOOK.mp4", "BOOK", "")],
              "test": [("P3", "3-LIBRARY.mp4", "LIBRARY", "A_01_001")]}
    results = [converted("1-LIBRARY"), converted("2-BOOK", arms=0), dict(sample_id="3-LIBRARY", status="refused",
                                                                          reason="no video stream")]
    assert run(tmp_path, setup_store(tmp_path, splits, results)) == 0
    manifest = json.loads((tmp_path / "out/m.json").read_text())
    records = {r["sign"]: r for r in manifest["records"]}
    assert records["1-LIBRARY"]["partition"] == "train" and records["2-BOOK"]["partition"] == "val"
    assert records["1-LIBRARY"]["gloss"] == dict(status="publisher_label", label="LIBRARY", asl_lex_code="A_01_001",
                                                 source="ASL_Citizen/splits")
    assert records["2-BOOK"]["gloss"]["asl_lex_code"] is None
    assert "P1" not in json.dumps(manifest)                     # participants only as pseudonyms
    assert manifest["extraction"]["missing"] == 1 and manifest["extraction"]["statuses"]["refused"] == 1
    assert manifest["quality"]["train"]["arm_clips"] == 1 and manifest["quality"]["val"]["arm_clips"] == 0


def test_a_participant_in_two_splits_stops_the_build(tmp_path):
    splits = {"train": [("P1", "1-A.mp4", "A", "")], "val": [("P1", "2-B.mp4", "B", "")], "test": []}
    with pytest.raises(ValueError, match="more than one split"):
        run(tmp_path, setup_store(tmp_path, splits, [converted("1-A")]))


def test_states_without_the_upright_prior_are_refused(tmp_path):
    splits = {"train": [("P1", "1-A.mp4", "A", "")], "val": [], "test": []}
    with pytest.raises(SystemExit, match="upright"):
        run(tmp_path, setup_store(tmp_path, splits, [converted("1-A")], upright=False))


def test_a_resumed_shard_keeps_clips_found_already_converted(tmp_path):
    splits = {"train": [("P1", "1-A.mp4", "A", ""), ("P1", "2-B.mp4", "B", "")], "val": [], "test": []}
    present = dict(converted("2-B", arms=4), status="already_present", quality_from_state=True)
    error = dict(sample_id="1-A", status="error", reason="ReadTimeoutError")
    again = dict(converted("1-A"), status="already_present")            # found again by a later shard run
    zip_key = setup_store(tmp_path, splits, [error, present], extra_reports=[[converted("1-A"), again]])
    assert run(tmp_path, zip_key) == 0
    manifest = json.loads((tmp_path / "out/m.json").read_text())
    records = {r["sign"]: r for r in manifest["records"]}
    assert set(records) == {"1-A", "2-B"} and manifest["extraction"]["missing"] == 0
    assert records["2-B"]["quality"]["quality_from_state"] and records["2-B"]["quality"]["arm_frames"]["RightForearm"] == 4
    assert records["1-A"]["quality"]["quality_from_state"] is None     # the conversion's own report wins
