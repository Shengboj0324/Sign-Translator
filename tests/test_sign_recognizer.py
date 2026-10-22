"""Isolated-sign recogniser: learns a toy gloss mapping, top-k maths, trainer selects by validation top-1."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
from signtranslator.generation.sign_recognizer import SignRecognizer, SignRecognizerConfig, top_k_accuracy
from tests.test_generation_text_to_motion import config as motion_config, state

ROOT = Path(__file__).resolve().parents[1]


def test_top_k_accuracy():
    logits = torch.tensor([[3., 2, 1], [1., 3, 2], [1., 2, 3]])
    out = top_k_accuracy(logits, torch.tensor([0, 2, 0]), ks=(1, 2, 5))
    assert abs(out["top1"] - 1 / 3) < 1e-6 and abs(out["top2"] - 2 / 3) < 1e-6 and out["top5"] == 1.


def test_learns_a_toy_gloss_mapping():
    torch.manual_seed(0)
    states = [state(i, 24) for i in range(4)]
    model = SignRecognizer(SignRecognizerConfig(motion_config(states), 4, d_model=32, heads=4, layers=1,
                                                feedforward=64, dropout=0.))
    batch, labels = collate_multichannel(states), torch.arange(4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss, terms = model.loss(batch, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    assert terms["top1"] == 1. and (model.eval()(batch).argmax(1) == labels).all()


def test_trainer_labels_from_train_and_best_by_top1(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "store"; records = []
    for i, (gloss, partition) in enumerate([("BOOK", "train"), ("CAT", "train"), ("BOOK", "train"), ("CAT", "val"),
                                            ("DOG", "val")]):
        s = state(i, 24); path = root / f"states/s{i}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        digest = s.save(path)
        records.append(dict(sign=f"s{i}", text=None, partition=partition, performer_sha256=None,
                            gloss=dict(status="publisher_label", label=gloss),
                            quality=dict(frames=24, left_hand_frames=0, right_hand_frames=20),
                            continuous=dict(status="converted", state_key=f"states/s{i}.npz", state_sha256=digest,
                                            available_channels=["body", "head_rotation", "face"], seconds=1.)))
    payload = canonical_json(build_corpus_manifest(corpus_id="t", inputs={"x": "b" * 64}, records=records, split={}))
    (root / "manifest.json").write_bytes(payload)
    spec = importlib.util.spec_from_file_location("train_sr", ROOT / "scripts/train_sign_recognizer.py")
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    common = ["--store", f"dir:{root}", "--manifest-key", "manifest.json", "--manifest-sha256",
              hashlib.sha256(payload).hexdigest(), "--run", "sr", "--batch-size", "2", "--d-model", "32", "--heads", "4",
              "--layers", "1", "--feedforward", "64", "--warmup", "2", "--val-every", "2", "--checkpoint-every", "2",
              "--workers", "0", "--trained-channels", "body,head_rotation,face"]
    assert script.main(common + ["--steps", "4"]) == 0
    run = json.loads((root / "training-runs/sr/run.json").read_text())
    assert run["labels"] == ["BOOK", "CAT"] and run["train_records"] == 3 and run["val_records"] == 1   # DOG: unseen gloss
    assert "val_top1" in json.loads((root / "training-runs/sr/best.json").read_text())
    assert script.main(common + ["--steps", "6"]) == 0


def test_trainer_adds_an_extra_corpus_through_the_exact_gloss_map_with_fps_and_augmentation(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    root = tmp_path / "store"

    def manifest(name, rows):
        records = []
        for i, (gloss, partition) in enumerate(rows):
            s = state(i, 24); path = root / f"{name}/s{i}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
            digest = s.save(path)
            records.append(dict(sign=f"{name}{i}", text=None, partition=partition, performer_sha256=None,
                                gloss=dict(status="x", label=gloss),
                                continuous=dict(status="converted", state_key=f"{name}/s{i}.npz", state_sha256=digest,
                                                available_channels=["body", "head_rotation", "face"], seconds=1.)))
        payload = canonical_json(build_corpus_manifest(corpus_id=name, inputs={"x": "b" * 64}, records=records, split={}))
        (root / f"{name}.json").write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    a = manifest("citizen", [("BOOK", "train"), ("CAT", "train"), ("BOOK", "val")])
    b = manifest("asllvd", [("BOOK", "train"), ("CAT", "val"), ("BOOK", "test"), ("DOG", "train")])
    (root / "map.json").write_text(json.dumps({"mapping": {
        "BOOK": {"method": "exact", "candidates": ["BOOK"]}, "CAT": {"method": "exact", "candidates": ["CAT"]},
        "DOG": {"method": "letters", "candidates": ["DOG"]}}}))
    spec = importlib.util.spec_from_file_location("train_sr2", ROOT / "scripts/train_sign_recognizer.py")
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    assert script.main(["--store", f"dir:{root}", "--manifest-key", "citizen.json", "--manifest-sha256", a, "--run", "sr2",
                        "--extra-manifest-key", "asllvd.json", "--extra-manifest-sha256", b, "--gloss-map-key", "map.json",
                        "--fps", "20", "--augment", "--batch-size", "2", "--d-model", "32", "--heads", "4", "--layers", "1",
                        "--feedforward", "64", "--warmup", "2", "--val-every", "2", "--checkpoint-every", "2",
                        "--workers", "0", "--trained-channels", "body,head_rotation,face", "--steps", "4"]) == 0
    run = json.loads((root / "training-runs/sr2/run.json").read_text())
    assert run["train_records"] == 3                    # citizen BOOK, CAT + asllvd BOOK; DOG is not an exact match
    best = json.loads((root / "training-runs/sr2/best.json").read_text())
    assert "cross/top1" in best["validation"]           # asllvd CAT (val signer) reported, test BOOK never read


def test_augmentation_changes_speed_and_keeps_a_valid_state():
    import random
    from signtranslator.generation.sign_recognizer import augment_sign
    from signtranslator.pose.resample import resample_motion
    s = state(0, 48)
    out = resample_motion(augment_sign(s, random.Random(1), speed=(1.5, 1.5), trim=0., face_dropout=0.), 24)
    assert abs((out.timestamps[-1] - out.timestamps[0]) - (s.timestamps[-1] - s.timestamps[0]) / 1.5) < 1 / 24
    out.validate()
