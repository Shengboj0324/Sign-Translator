"""Training entrypoint: runs on a directory store, checkpoints, resumes exactly."""
import hashlib
import importlib.util
import json
from pathlib import Path

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
from tests.test_generation_text_to_motion import state

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_t2m", ROOT / "scripts/train_text_to_motion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def store_with_manifest(root: Path):
    records = []
    for i, (text, partition) in enumerate([("a dog runs", "train"), ("cats sleep", "train"), ("birds fly", "train"),
                                           ("rain falls", "val")]):
        s = state(i, 24 + 6 * i)
        path = root / f"states/s{i}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = s.save(path)
        records.append(dict(sign=f"s{i}", text=text, partition=partition, performer_sha256=None,
                            continuous=dict(status="converted", state_key=f"states/s{i}.npz", state_sha256=digest,
                                            available_channels=["body", "head_rotation", "face"],
                                            seconds=len(s.timestamps) / 24.)))
    manifest = build_corpus_manifest(corpus_id="t", inputs={"x": "b" * 64}, records=records, split={})
    payload = canonical_json(manifest)
    (root / "manifest.json").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_trains_checkpoints_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    common = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
              "--run", "smoke", "--batch-size", "2", "--d-model", "32", "--heads", "4", "--text-layers", "1",
              "--motion-layers", "1", "--feedforward", "64", "--warmup", "2", "--eval-every", "2", "--eval-batches", "1",
              "--checkpoint-every", "2", "--workers", "0", "--trained-channels", "body,head_rotation,face", "--amp", "none"]
    assert script.main(common + ["--steps", "4"]) == 0
    checkpoints = sorted((tmp_path / "store/training-runs/smoke/checkpoints").iterdir())
    assert [c.name for c in checkpoints] == ["step-00000002.pt", "step-00000004.pt"]
    run = json.loads((tmp_path / "store/training-runs/smoke/run.json").read_text())
    assert run["train_records"] == 3 and run["val_records"] == 1
    assert script.main(common + ["--steps", "6"]) == 0  # resumes from step 4
    log = json.loads((tmp_path / "store/training-runs/smoke/log/step-00000006.json").read_text())
    assert [e["step"] for e in log] == [5, 6] and "val/body" in log[-1]
