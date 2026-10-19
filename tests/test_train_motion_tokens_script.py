"""Motion-token training: tokenizer then prior, hash-pinned hand-off, resume and refusals."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_train_text_to_motion_script import store_with_manifest

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_tokens", ROOT / "scripts/train_motion_tokens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_stages_train_hand_off_by_hash_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    base = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
            "--trained-channels", "body,head_rotation,face"]
    tokenizer = base + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8", "--codebook", "16"]
    assert script.main(tokenizer + ["--steps", "2"]) == 0
    assert script.main(tokenizer + ["--steps", "4"]) == 0                  # resumes at step 2
    log = json.loads((tmp_path / "store/training-runs/tok/log/step-00000004.json").read_text())
    assert [e["step"] for e in log] == [3, 4]
    key = "training-runs/tok/checkpoints/step-00000004.pt"
    tok_sha = hashlib.sha256((tmp_path / "store" / key).read_bytes()).hexdigest()
    prior = base + ["--stage", "prior", "--run", "pri", "--tokenizer-key", key, "--d-model", "32", "--heads", "4",
                    "--text-layers", "1", "--token-layers", "1", "--feedforward", "64"]
    with pytest.raises(SystemExit, match="differs"):
        script.main(prior + ["--tokenizer-sha256", "0" * 64, "--steps", "2"])
    assert script.main(prior + ["--tokenizer-sha256", tok_sha, "--steps", "2"]) == 0
    run = json.loads((tmp_path / "store/training-runs/pri/run.json").read_text())
    assert run["kind"] == "motion_token_prior" and run["config"]["tokenizer"]["codebook"] == 16
    with pytest.raises(SystemExit, match="MPS"):
        script.main(tokenizer + ["--steps", "6", "--device", "mps"])
