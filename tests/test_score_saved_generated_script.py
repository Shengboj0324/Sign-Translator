"""Saved-generation scoring: a pooled face calibration is estimated across signers and applied to each."""
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_saved_generated as ssg  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402
from tests.test_generation_canonical_skeleton import rest_state  # noqa: E402


def faced(seed, spread):
    s = rest_state(24)
    f = s.channels["face"]
    rng = np.random.default_rng(seed)
    values = np.clip(.3 + spread * rng.normal(size=f.values.shape), 0, 1)
    face = replace(f, values=values, valid=np.ones(f.valid.shape, bool), confidence=np.ones(f.valid.shape),
                   inference_method="estimate")
    return replace(s, channels=dict(s.channels, face=face))


def put_state(store, key, state):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "s.npz"; digest = state.save(p); store.put(key, p.read_bytes())
    return digest


def test_pooled_calibration_comes_from_every_signer(tmp_path):
    store = DirectoryStore(tmp_path)
    records = []
    for g, signs in (("a", ["s1", "s2"]), ("b", ["s3", "s4"])):
        for i, sign in enumerate(signs):
            digest = put_state(store, f"real/{sign}.npz", faced(i + 10 * len(records), .2))
            records.append(dict(sign=sign, text="hello there", partition="val",
                                continuous=dict(status="converted", state_key=f"real/{sign}.npz", state_sha256=digest)))
            put_state(store, f"gen-{g}/guidance-1/{sign}.npz", faced(100 + len(records), .05))   # under-varied face
    m = build_corpus_manifest(corpus_id="t", records=records, inputs={"x": "a" * 64}, split=dict(kind="continuous"))
    body = canonical_json(m); store.put("m.json", body)
    sha = hashlib.sha256(body).hexdigest()
    assert ssg.main(["--store", f"dir:{tmp_path}", "--manifest-key", "m.json", "--manifest-sha256", sha, "--generated",
                     "gen-a", "--calibration-pool", "gen-a", "gen-b", "--output-key", "out.json"]) == 0
    out = json.loads(store.get("out.json"))
    (name, res), = out["results"].items()
    assert name.endswith("face-calibrated-pooled") and res["face_calibration"]["pooled_from"] == ["gen-a", "gen-b"]
    assert res["face_calibration"]["clips"] == dict(generated=4, real=4) and max(res["face_calibration"]["gain"]) > 1


def test_postprocessing_grid_writes_every_variant_per_group(tmp_path):
    import score_postprocessing_grid as grid
    store = DirectoryStore(tmp_path)
    records = []
    for g, signs in (("c01", ["s1", "s2"]), ("c03", ["s3", "s4"])):
        for i, sign in enumerate(signs):
            digest = put_state(store, f"real/{sign}.npz", faced(i + 10 * len(records), .2))
            records.append(dict(sign=sign, text="hello there", partition="val",
                                continuous=dict(status="converted", state_key=f"real/{sign}.npz", state_sha256=digest)))
            for tag in ("guidance-1", "guidance-2"):
                put_state(store, f"gen/R-{g}/{tag}/{sign}.npz", faced(100 + len(records), .05))
    m = build_corpus_manifest(corpus_id="t", records=records, inputs={"x": "a" * 64}, split=dict(kind="continuous"))
    body = canonical_json(m); store.put("m.json", body)
    assert grid.main(["--store", f"dir:{tmp_path}", "--manifest-key", "m.json", "--manifest-sha256",
                      hashlib.sha256(body).hexdigest(), "--run", "R", "--groups", "c01,c03",
                      "--generated-template", "gen/{run}-{group}", "--evaluations", "ev", "--head-smoothing", "0,0.2"]) == 0
    names = sorted(k for k in store.list("ev/"))
    assert names == sorted(f"ev/R-hs{h}{fc}-{g}.json" for h in ("0", "0.2") for fc in ("", "-fc") for g in ("c01", "c03"))
    a = json.loads(store.get("ev/R-hs0.2-fc-c01.json"))["results"]["motion_tokens_v2 guidance 2"]["face_calibration"]
    b = json.loads(store.get("ev/R-hs0.2-fc-c03.json"))["results"]["motion_tokens_v2 guidance 2"]["face_calibration"]
    assert a == b and a["clips"] == dict(generated=4, real=4)       # one pooled calibration for both signers
