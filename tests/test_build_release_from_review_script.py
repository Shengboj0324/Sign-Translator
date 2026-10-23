"""Release from the reviewed selection: built only after a passed strict review, pinning the reviewed setting."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_release_from_review as build  # noqa: E402
from signtranslator.serving.release import Release  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402

SEL = dict(run="p-sweep-h1.2-f1.0-hs0.2-fc", setting="guidance 1", guidance=1., head_smoothing=.2,
           part_temperature="body=1.2,left_hand=1.2,right_hand=1.2,face=1.0", face_calibrated=True)
CAL = dict(labels=["browDownLeft"], center_generated=[.1], center_real=[.2], gain=[2.])


def setup(tmp_path, passed):
    store = DirectoryStore(tmp_path / "s")
    store.put("ck.pt", b"weights")
    store.put("strict.json", json.dumps(dict(pass_=None, selected=SEL, face_calibration=CAL,
                                             rules=dict(R1=dict(pass_=True))) | {"pass": passed}).encode())
    store.put("gate.json", json.dumps(dict(selected=dict(run=SEL["run"], setting="guidance 1", failing=["x"]))).encode())
    return store


def argv(tmp_path, out):
    return ["--store", f"dir:{tmp_path / 's'}", "--strict-key", "strict.json", "--gate-key", "gate.json",
            "--checkpoint-key", "ck.pt", "--manifest-sha256", "b" * 64, "--base-release",
            str(ROOT / "releases/research-v2b-planner-v2.json"), "--version", "v3", "--output", str(out)]


def test_refuses_without_a_passed_review(tmp_path):
    setup(tmp_path, False)
    assert build.main(argv(tmp_path, tmp_path / "r.json")) == 4 and not (tmp_path / "r.json").exists()


def test_pins_the_reviewed_setting(tmp_path):
    setup(tmp_path, True)
    out = tmp_path / "r.json"
    assert build.main(argv(tmp_path, out)) == 0
    data = json.loads(out.read_text())
    r = Release.from_dict(data)
    assert r.generator.checkpoint.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert r.generator.head_smoothing == .2 and r.generator.face_calibration == CAL
    assert r.generator.part_temperature["face"] == 1. and data["selection"]["known_limits"] == ["x"]
    assert r.planner is not None and r.provenance
