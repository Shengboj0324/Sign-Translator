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


def test_evaluation_adds_r_precision_and_saves_generated_states(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    store = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest]
    base = store + ["--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
                    "--trained-channels", "body,head_rotation,face"]
    assert script.main(base + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8",
                               "--codebook", "16", "--steps", "2"]) == 0
    key = "training-runs/tok/checkpoints/step-00000002.pt"
    tok_sha = hashlib.sha256((tmp_path / "store" / key).read_bytes()).hexdigest()
    assert script.main(base + ["--stage", "prior", "--run", "pri", "--tokenizer-key", key, "--tokenizer-sha256", tok_sha,
                               "--d-model", "32", "--heads", "4", "--text-layers", "1", "--token-layers", "1",
                               "--feedforward", "64", "--steps", "2"]) == 0
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("train_retrieval", ROOT / "scripts/train_text_motion_retrieval.py")
    retrieval = importlib.util.module_from_spec(spec); spec.loader.exec_module(retrieval)
    assert retrieval.main(store + ["--run", "ret", "--batch-size", "3", "--d-model", "32", "--heads", "4",
                                   "--motion-layers", "1", "--text-layers", "1", "--feedforward", "64", "--embedding", "16",
                                   "--warmup", "1", "--val-every", "2", "--steps", "2", "--workers", "0",
                                   "--trained-channels", "body,head_rotation,face"]) == 0
    retrieval_key = json.loads((tmp_path / "store/training-runs/ret/best.json").read_text())["checkpoint_key"]
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate_text_to_motion.py")
    evaluate = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)
    assert evaluate.main(store + ["--checkpoint-key", "training-runs/pri/checkpoints/step-00000002.pt", "--partition",
                                  "train", "--limit", "3", "--guidance", "1", "--output-key", "eval.json",
                                  "--retrieval-key", retrieval_key, "--save-generated", "generated/pri"]) == 0
    report = json.loads((tmp_path / "store/eval.json").read_text())
    result = next(iter(report["results"].values()))
    scores = result["retrieval"]
    assert {"r_at_1", "r_at_3", "matching_distance"} <= set(scores["generated"]) and scores["reference"]["clips"] == 3
    saved = sorted(p.name for p in (tmp_path / "store/generated/pri/guidance-1").iterdir())
    assert len(saved) == 3 and all(name.endswith(".npz") for name in saved)
    from signtranslator.pose.multichannel import MultichannelMotion
    path = tmp_path / "store/generated/pri/guidance-1" / saved[0]
    state = MultichannelMotion.load(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(state.timestamps) > 0


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
    from signtranslator.grammar_motion.tracks import SCHEMA_VERSION
    assert record["schema_version"] == SCHEMA_VERSION and record["manifest_sha256"] == digest


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


def test_weight_ema_is_the_exponential_average_and_swaps_cleanly():
    import torch
    script = load_script()
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.)
    ema = script.WeightEMA(model, decay=.9)
    expected = 0.
    for t, value in enumerate((1., 2., 3.)):
        with torch.no_grad():
            model.weight.fill_(value)
        d = min(.9, (1 + t) / (10 + t))                                          # warm-up: .1, .18, .25
        assert abs(ema.current_decay() - d) < 1e-12
        ema.update(model)
        expected = d * expected + (1 - d) * value                                # θ̄_t = d_t θ̄_{t-1} + (1-d_t) θ_t
    assert abs(float(ema.shadow["weight"][0, 0]) - expected) < 1e-6
    live = ema.swap(model)
    assert abs(float(model.weight[0, 0]) - expected) < 1e-6
    ema.restore(model, live)
    assert float(model.weight[0, 0]) == 3.
    late = script.WeightEMA(model, decay=.999); late.updates = 10 ** 6
    assert late.current_decay() == .999                                           # the cap holds late in training


def test_tokens_v2_train_validate_select_best_resume_and_evaluate(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    store = tmp_path / "store"
    base = ["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--val-every", "1", "--val-batches", "2",
            "--workers", "0", "--trained-channels", "body,head_rotation,face", "--architecture", "rvq"]
    tok = base + ["--stage", "tokenizer", "--run", "tok2", "--hidden", "16", "--code-dim", "8", "--codebook", "16",
                  "--stages", "2", "--down-layers", "1"]
    assert script.main(tok + ["--steps", "2"]) == 0
    assert script.main(tok + ["--steps", "4"]) == 0                              # resumes, EMA included
    best = json.loads((store / "training-runs/tok2/best.json").read_text())
    assert best["checkpoint_key"].startswith("training-runs/tok2/checkpoints/") and "val/loss" in best["validation"]
    log = json.loads((store / "training-runs/tok2/log/step-00000004.json").read_text())
    assert all("val/loss" in e for e in log)
    import torch, io as _io
    saved = torch.load(_io.BytesIO((store / "training-runs/tok2/checkpoints/step-00000004.pt").read_bytes()),
                       weights_only=False)
    assert saved["kind"] == "motion_rvq_tokenizer" and set(saved["ema"]) == set(saved["model"])
    key = "training-runs/tok2/checkpoints/step-00000004.pt"
    sha = hashlib.sha256((store / key).read_bytes()).hexdigest()
    prior = base + ["--stage", "prior", "--run", "pri2", "--tokenizer-key", key, "--tokenizer-sha256", sha,
                    "--d-model", "32", "--heads", "4", "--text-layers", "1", "--token-layers", "1", "--depth-layers", "1",
                    "--feedforward", "64", "--steps", "2"]
    assert script.main(prior) == 0
    run = json.loads((store / "training-runs/pri2/run.json").read_text())
    assert run["kind"] == "motion_rvq_prior" and run["config"]["grammar_features"] is True
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate_text_to_motion.py")
    evaluate = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)
    assert evaluate.main(["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                          "--checkpoint-key", "training-runs/pri2/checkpoints/step-00000002.pt", "--partition", "train",
                          "--limit", "2", "--guidance", "1", "--output-key", "eval2.json"]) == 0
    report = json.loads((store / "eval2.json").read_text())
    assert report["generator"] == "motion_tokens_v2"


def test_jitter_is_zero_on_uniform_rotation_and_the_diagnosis_runs(tmp_path, monkeypatch):
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("diagnose", ROOT / "scripts/diagnose_tokenizer.py")
    diagnose = importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnose)
    from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
    angles = torch.linspace(0, 1.2, 9, dtype=torch.float64)[:, None, None] * torch.tensor([[0., 0., 1.]], dtype=torch.float64)
    r6 = matrix_to_rotation_6d(axis_angle_to_matrix(angles + 1e-9))
    assert float(diagnose.jitter_deg(r6).max()) < 1e-4                     # constant angular velocity: no jitter
    bumped = r6.clone(); bumped[4] = matrix_to_rotation_6d(axis_angle_to_matrix(torch.tensor([[0., .3, .6]], dtype=torch.float64)))
    assert float(diagnose.jitter_deg(bumped)[3]) > 5.
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    assert script.main(["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256",
                        digest, "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0",
                        "--trained-channels", "body,head_rotation,face", "--architecture", "rvq", "--stage", "tokenizer",
                        "--run", "t", "--hidden", "16", "--code-dim", "8", "--codebook", "16", "--stages", "2",
                        "--down-layers", "1", "--steps", "2"]) == 0
    assert diagnose.main(["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256",
                          digest, "--tokenizer-key", "training-runs/t/checkpoints/step-00000002.pt", "--partition", "train",
                          "--limit", "2", "--output-key", "diag.json"]) == 0
    report = json.loads((tmp_path / "store/diag.json").read_text())
    assert report["overall"]["full_joint_deg"] is not None and report["overall"]["noise_joint_deg"] is not None
    # unavailable joints (zero payload) are masked, not fatal: the real data has off-screen legs
    import numpy as np
    from signtranslator.data.multichannel import collate_multichannel
    from tests.test_generation_signing_space import signer_state
    from dataclasses import replace
    state = signer_state(0, frames=8)
    body = state.channels["body"]
    valid = body.valid.copy(); valid[:, -3:] = False
    state = replace(state, channels={**state.channels, "body": replace(body, valid=valid,
                    values=np.where(valid[..., None], body.values, 0.), confidence=valid.astype(float))})
    batch = collate_multichannel([state])
    identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float64)
    ref = torch.where(batch.channels["body"].valid[0][..., None], batch.channels["body"].values[0].double(), identity)
    assert torch.isfinite(diagnose.geodesic_deg(ref, ref)).all() and torch.isfinite(diagnose.jitter_deg(ref)).all()


def test_early_stopping_ends_training_and_keeps_the_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    losses = iter([1.0, .5, .6, .7, .8, .9, 1.0, 1.1])
    real = script.rvq_loss
    def fake_validation(model, motion, **kw):                       # a validation curve that turns up after step 2
        loss, terms = real(model, motion, **kw)
        if not model.training:
            value = next(losses)
            return loss * 0 + value, terms
        return loss, terms
    monkeypatch.setattr(script, "rvq_loss", fake_validation)
    store = tmp_path / "store"
    assert script.main(["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                        "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "100", "--val-every", "1",
                        "--val-batches", "1", "--workers", "0", "--trained-channels", "body,head_rotation,face",
                        "--architecture", "rvq", "--stage", "tokenizer", "--run", "es", "--hidden", "16", "--code-dim",
                        "8", "--codebook", "16", "--stages", "2", "--down-layers", "1", "--steps", "50",
                        "--patience", "3"]) == 0
    best = json.loads((store / "training-runs/es/best.json").read_text())
    assert best["step"] == 2 and abs(best["val_loss"] - .5) < 1e-9          # the best checkpoint is kept even off-schedule
    checkpoints = sorted(p.name for p in (store / "training-runs/es/checkpoints").glob("*.pt"))
    assert checkpoints[-1] == "step-00000005.pt"                           # stopped 3 validations after the best


def test_a_non_finite_training_loss_stops_the_run_before_anything_is_saved(tmp_path, monkeypatch):
    """Metrics are read off the device only when logged, so the finiteness check runs there — still before a save."""
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    real, calls = script.rvq_loss, iter(range(10 ** 6))
    def poisoned(model, motion, **kw):
        loss, terms = real(model, motion, **kw)
        return (loss * float("nan") if model.training and next(calls) == 3 else loss), terms
    monkeypatch.setattr(script, "rvq_loss", poisoned)
    store = tmp_path / "store"
    with pytest.raises(FloatingPointError, match="non-finite loss at step 3"):
        script.main(["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                     "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "8", "--val-every", "100",
                     "--workers", "0", "--trained-channels", "body,head_rotation,face", "--architecture", "rvq",
                     "--stage", "tokenizer", "--run", "nan", "--hidden", "16", "--code-dim", "8", "--codebook", "16",
                     "--stages", "2", "--down-layers", "1", "--steps", "10"])
    assert not list((store / "training-runs/nan").glob("checkpoints/*.pt"))


def test_code_statistics_measure_repeat_rate_entropy_and_rate():
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from diagnose_tokenizer import code_statistics
    constant = (np.zeros((10, 2), int), np.ones(10, bool))                  # never changes: no information
    stats = code_statistics([constant], codebook=4, steps_per_second=5.)
    assert stats["repeat_rate"] == [1., 1.] and stats["unigram_entropy"] == [0., 0.] and stats["unigram_rate_nats_per_s"] == 0.
    cycle = (np.stack([np.arange(40) % 4, np.arange(40) % 4], 1), np.ones(40, bool))    # uniform but deterministic
    stats = code_statistics([cycle], codebook=4, steps_per_second=10.)
    np.testing.assert_allclose(stats["unigram_entropy"], [np.log(4)] * 2)
    np.testing.assert_allclose(stats["conditional_entropy"], [0., 0.], atol=1e-12)   # the next code is determined
    assert stats["repeat_rate"] == [0., 0.] and abs(stats["unigram_rate_nats_per_s"] - 20 * np.log(4)) < 1e-9
    masked = (np.array([[0], [1], [0], [1]]), np.array([True, False, False, True]))     # no valid adjacent pair
    assert code_statistics([masked], codebook=2, steps_per_second=1.)["repeat_rate"] == [0.]
