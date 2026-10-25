"""Tokenizer domain check: new-corpus reconstruction error against the reference corpus's, own or fixed (round 3d)."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_fixed_reference_replaces_the_tokenizers_own_reference_errors(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    from tests.test_train_text_to_motion_script import state
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "store"
    records = []
    for i, (corpus, partition) in enumerate([("ref", "train"), ("ref", "train"), ("ref", "val"), ("new", "val")]):
        s = state(i, 24 + 6 * i)
        path = root / f"states/s{i}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        records.append(dict(sign=f"s{i}", text="x", corpus=corpus, partition=partition, performer_sha256=None,
                            continuous=dict(status="converted", state_key=f"states/s{i}.npz", state_sha256=s.save(path),
                                            available_channels=["body", "head_rotation", "face"],
                                            seconds=len(s.timestamps) / 24.)))
    payload = canonical_json(build_corpus_manifest(corpus_id="t", inputs={"x": "b" * 64}, records=records, split={}))
    (root / "manifest.json").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    train = load("tokens", "scripts/train_motion_tokens.py")
    assert train.main(["--store", f"dir:{root}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                       "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
                       "--trained-channels", "body,head_rotation,face", "--architecture", "rvq", "--stage", "tokenizer",
                       "--run", "tok", "--hidden", "16", "--code-dim", "8", "--codebook", "16", "--stages", "2",
                       "--down-layers", "1", "--steps", "2"]) == 0
    domain = load("domain", "scripts/tokenizer_domain_check.py")
    common = ["--store", f"dir:{root}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
              "--tokenizer-key", "training-runs/tok/checkpoints/step-00000002.pt", "--conform-to-corpus", "ref",
              "--corpus", "new", "--limit", "4"]
    domain.main(common + ["--output-key", "own.json"])
    own = json.loads((root / "own.json").read_text())
    names = sorted(own["reference"])
    # a baseline twice as inaccurate on the first channel (the candidate improved it) and half on the others
    fixed = {k: (2 if k == names[0] else .5) * v for k, v in own["reference"].items()}
    (root / "baseline.json").write_text(json.dumps(dict(reference=fixed)))
    domain.main(common + ["--fixed-reference-key", "baseline.json", "--output-key", "fixed.json"])
    report = json.loads((root / "fixed.json").read_text())
    new, old = report["corpora"]["new"], own["corpora"]["new"]
    assert report["own_reference"] == own["reference"]
    assert report["reference"] == {k: max(v, fixed[k]) for k, v in own["reference"].items()}
    for k, v in old["ratios"].items():
        expect = v / 2 if k == names[0] else v          # the larger denominator: the baseline's, or the candidate's own
        assert abs(new["ratios"][k] - expect) <= 1e-3 and new["ratios_own_reference"][k] == v
