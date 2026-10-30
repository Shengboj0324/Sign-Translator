"""Round 17 CLIs on a tiny store: plan tokenizer, plan prior, plan-conditioned detail prior, two-stage evaluation."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest
import torch

from tests.test_train_text_to_motion_script import store_with_manifest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_tokenizer_plan_prior_and_plan_conditioned_prior_train_hand_off_by_hash_and_generate(tmp_path, monkeypatch,
                                                                                                    capsys):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    store = tmp_path / "store"
    tmt, tpt = load("train_motion_tokens"), load("train_plan_tokenizer")
    data = ["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--batch-size", "2", "--warmup", "1", "--checkpoint-every", "2", "--workers", "0"]
    rvq = data + ["--trained-channels", "body,head_rotation,face", "--architecture", "rvq"]
    # a d2-like tokenizer: one step per frame
    assert tmt.main(rvq + ["--stage", "tokenizer", "--run", "tok", "--hidden", "16", "--code-dim", "8", "--codebook",
                           "16", "--stages", "2", "--down-layers", "0", "--base-blocks", "1", "--steps", "2"]) == 0
    tok_key = "training-runs/tok/checkpoints/step-00000002.pt"
    tok_sha = sha(store / tok_key)
    plan_args = data + ["--run", "plan", "--tokenizer-key", tok_key, "--tokenizer-sha256", tok_sha, "--window", "4",
                        "--hidden", "16", "--code-dim", "8", "--codebook", "16", "--plan-stages", "2", "--blocks", "1",
                        "--val-every", "1", "--val-batches", "1", "--fidelity-clips", "2"]
    with pytest.raises(SystemExit, match="differs"):
        tpt.main(plan_args[:plan_args.index(tok_sha)] + ["0" * 64] + plan_args[plan_args.index(tok_sha) + 1:]
                 + ["--steps", "2"])
    assert tpt.main(plan_args + ["--steps", "2"]) == 0
    assert tpt.main(plan_args + ["--steps", "4"]) == 0                          # resumes at step 2
    log = json.loads((store / "training-runs/plan/log/step-00000004.json").read_text())
    assert [e["step"] for e in log] == [3, 4]
    assert {"val/latent_relative", "val/motion", "fidelity/plan/body_speed_ratio",
            "fidelity/tokenizer/body_speed_ratio"} <= set(log[-1])
    best = json.loads((store / "training-runs/plan/best.json").read_text())
    assert best["checkpoint_key"].startswith("training-runs/plan/checkpoints/")
    with pytest.raises(SystemExit, match="different config"):
        tpt.main([a if a != "4" else "2" for a in plan_args] + ["--steps", "6"])   # another window cannot resume
    plan_key = "training-runs/plan/checkpoints/step-00000004.pt"
    plan_sha = sha(store / plan_key)
    from signtranslator.generation.plan_codes import load_plan_tokenizer
    plan, _, saved = load_plan_tokenizer((store / plan_key).read_bytes(), plan_sha, tokenizer_sha256=tok_sha)
    assert saved["kind"] == "plan_vq_tokenizer" and saved["tokenizer_key"] == tok_key and set(saved["ema"]) == set(saved["model"])

    prior = rvq + ["--tokenizer-key", tok_key, "--tokenizer-sha256", tok_sha, "--d-model", "32", "--heads", "4",
                   "--text-layers", "1", "--token-layers", "1", "--depth-layers", "1", "--feedforward", "64",
                   "--steps", "2", "--val-every", "1", "--val-batches", "1"]
    assert tmt.main(prior + ["--stage", "prior", "--run", "plain"]) == 0             # the plan-less prior (W14b's role)
    plain_key = "training-runs/plain/checkpoints/step-00000002.pt"
    plain_sha = sha(store / plain_key)
    with_plan = ["--plan-tokenizer-key", plan_key, "--plan-tokenizer-sha256", plan_sha]
    # the plan prior, its text side from the plan-less prior
    with pytest.raises(SystemExit, match="needs --architecture rvq and --plan-tokenizer-key"):
        tmt.main(prior + ["--stage", "plan-prior", "--run", "pp-bad"])
    with pytest.raises(SystemExit, match="plan-tokenizer-key"):
        tmt.main(prior + ["--stage", "plan-prior", "--run", "pp-bad", "--plan-tokenizer-key", plan_key,
                          "--plan-tokenizer-sha256", "0" * 64])
    capsys.readouterr()
    assert tmt.main(prior + ["--stage", "plan-prior", "--run", "pp", *with_plan, "--init-text-key", plain_key,
                             "--init-text-sha256", plain_sha]) == 0
    assert f"text encoder from {plain_key}" in capsys.readouterr().out
    pp_key = "training-runs/pp/checkpoints/step-00000002.pt"
    pp = torch.load(io.BytesIO((store / pp_key).read_bytes()), weights_only=False)
    assert pp["kind"] == "motion_plan_prior" and pp["config"]["tokenizer"]["window"] == 4
    assert pp["plan_tokenizer_sha256"] == plan_sha and pp["tokenizer_sha256"] == tok_sha
    # the plan-conditioned detail prior, warm-started from the plan-less prior
    detail = prior + ["--stage", "prior", *with_plan, "--plan-dropout", "0.5", "--init-key", plain_key, "--init-sha256",
                      plain_sha]
    with pytest.raises(SystemExit, match="different kind or config"):
        tmt.main([a if a != "--token-layers" else "--depth-layers" for a in detail] + ["--run", "bad"])
    capsys.readouterr()
    assert tmt.main(detail + ["--run", "detail"]) == 0
    out = capsys.readouterr().out
    assert "plan modules zero-initialised" in out and f"initialised from {plain_key}" in out
    detail_key = "training-runs/detail/checkpoints/step-00000002.pt"
    saved = torch.load(io.BytesIO((store / detail_key).read_bytes()), weights_only=False)
    assert saved["config"]["plan_window"] == 4 and saved["config"]["plan_dropout"] == .5
    assert saved["plan_tokenizer_sha256"] == plan_sha and saved["plan_tokenizer_state"] is not None
    logged = json.loads((store / "training-runs/detail/log/step-00000002.json").read_text())
    assert {"val/tokens_no_plan", "val/plan_gain"} <= set(logged[-1])                  # the informativeness measure
    assert tmt.main(detail + ["--run", "detail", "--steps", "4"]) == 0                    # resumes under the same plan
    with pytest.raises(SystemExit, match="plan-tokenizer-key"):
        tmt.main(prior + ["--stage", "prior", "--run", "x", "--plan-dropout", "0.5"])

    # two-stage evaluation: load_generator with the pinned plan prior and plan tokenizer
    from signtranslator.data.text_motion_dataset import DirectoryStore
    ev = load("evaluate_text_to_motion")
    ds = DirectoryStore(store)
    pins = dict(prior_key=pp_key, prior_sha256=sha(store / pp_key), tokenizer_key=plan_key, tokenizer_sha256=plan_sha)
    with pytest.raises(SystemExit, match="plan-conditioned: name its plan prior"):
        ev.load_generator(ds, detail_key, digest)
    with pytest.raises(SystemExit, match="not a plan-conditioned prior"):
        ev.load_generator(ds, plain_key, digest, plan=pins)
    with pytest.raises(SystemExit, match="pinned SHA-256"):
        ev.load_generator(ds, detail_key, digest, plan=dict(pins, tokenizer_sha256="0" * 64))
    label, channels, sample = ev.load_generator(ds, detail_key, digest, plan=dict(pins, temperature=1.2, guidance=1.))
    assert label == "motion_tokens_v2" and sample.plan_prior is not None
    states = sample.batch(["a dog runs", "rain falls"], 24., [.5, .3], 1.5, 0)
    assert [len(s.timestamps) for s in states] == [12, 7]
    again = sample.batch(["a dog runs", "rain falls"], 24., [.5, .3], 1.5, 0)
    assert all((a.channels["body"].values == b.channels["body"].values).all() for a, b in zip(states, again))
    one = sample("a dog runs", 24., None, 1.5, 50, 3)                                  # predicted duration
    one.validate()
    # the oracle-plan diagnostic: the detail prior given fixed plans, not the plan prior's
    J = sample.plan_prior.config.tokenizer.codes_per_step                            # body and face: 2 parts x 2
    fixed = [torch.zeros(3, J, dtype=torch.long), torch.zeros(2, J, dtype=torch.long)]   # 12 and 7 steps, W = 4
    first = sample.batch_with_plans(["a dog runs", "rain falls"], 24., [.5, .3], 1.5, 0, fixed)
    second = sample.batch_with_plans(["a dog runs", "rain falls"], 24., [.5, .3], 1.5, 0, fixed)
    assert all((a.channels["body"].values == b.channels["body"].values).all() for a, b in zip(first, second))
    # the plan's fidelity instruments accept the plan tokenizer only over its own tokenizer
    _, _, plain_sample = ev.load_generator(ds, plain_key, digest)
    assert getattr(plain_sample, "plan_prior", None) is None
