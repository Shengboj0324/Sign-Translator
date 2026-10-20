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


def test_evaluation_reports_grammar_for_a_trained_prior(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    base = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
            "--trained-channels", "body,head_rotation,face"]
    assert script.main(base + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8",
                               "--codebook", "16", "--steps", "2"]) == 0
    key = "training-runs/tok/checkpoints/step-00000002.pt"
    tok_sha = hashlib.sha256((tmp_path / "store" / key).read_bytes()).hexdigest()
    assert script.main(base + ["--stage", "prior", "--run", "pri", "--tokenizer-key", key, "--tokenizer-sha256", tok_sha,
                               "--d-model", "32", "--heads", "4", "--text-layers", "1", "--token-layers", "1",
                               "--feedforward", "64", "--steps", "2"]) == 0
    run = json.loads((tmp_path / "store/training-runs/pri/run.json").read_text())
    assert run["config"]["grammar_features"] is True                       # on by default
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate_text_to_motion.py")
    evaluate = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)
    assert evaluate.main(["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256",
                          digest, "--checkpoint-key", "training-runs/pri/checkpoints/step-00000002.pt",
                          "--partition", "train", "--limit", "3", "--guidance", "1", "--output-key", "eval.json"]) == 0
    report = json.loads((tmp_path / "store/eval.json").read_text())
    result = next(iter(report["results"].values()))
    assert "grammar" in result and {"tests", "distributions"} <= set(result["grammar"])


def test_grammar_tracks_are_written_for_every_manifest_record(tmp_path):
    import sys
    digest = store_with_manifest(tmp_path / "store")
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("annotate", ROOT / "scripts/annotate_grammar_tracks.py")
    annotate = importlib.util.module_from_spec(spec); spec.loader.exec_module(annotate)
    assert annotate.main(["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json",
                          "--manifest-sha256", digest, "--prefix", "grammar", "--workers", "2"]) == 0
    index = json.loads((tmp_path / "store/grammar/index.json").read_text())
    assert index["records"] and all((tmp_path / "store" / r["key"]).exists() for r in index["records"])
    record = json.loads((tmp_path / "store" / index["records"][0]["key"]).read_text())
    assert record["schema_version"] == 1 and record["manifest_sha256"] == digest


def test_prior_trains_with_locus_conditioning(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    base = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
            "--trained-channels", "body,head_rotation,face"]
    assert script.main(base + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8",
                               "--codebook", "16", "--steps", "2"]) == 0
    key = "training-runs/tok/checkpoints/step-00000002.pt"
    tok_sha = hashlib.sha256((tmp_path / "store" / key).read_bytes()).hexdigest()
    assert script.main(base + ["--stage", "prior", "--run", "pri", "--tokenizer-key", key, "--tokenizer-sha256", tok_sha,
                               "--d-model", "32", "--heads", "4", "--text-layers", "1", "--token-layers", "1",
                               "--feedforward", "64", "--steps", "2", "--locus-conditioning"]) == 0
    run = json.loads((tmp_path / "store/training-runs/pri/run.json").read_text())
    assert run["config"]["locus_conditioning"] is True
