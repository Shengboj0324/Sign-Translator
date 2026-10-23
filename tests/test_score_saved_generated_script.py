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
