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
    """End to end on a tiny store, with the review fixes: plan corruption arms and the round-17 diagnostics."""
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
    assert {"val/tokens_no_plan", "val/plan_gain", "val/plan_gain_stage1"} <= set(logged[-1])   # informativeness
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
    _, _, plain_sample = ev.load_generator(ds, plain_key, digest)
    assert getattr(plain_sample, "plan_prior", None) is None

    # review fix 4: plan corruption, from random codes and from the plan prior's teacher-forced samples
    pp_sha = sha(store / pp_key)
    with pytest.raises(SystemExit, match="plan-corruption-prior-key"):
        tmt.main(detail + ["--run", "c-bad", "--plan-corruption", "0.5"])
    with pytest.raises(SystemExit, match="plan-corruption-prior-key"):
        tmt.main(detail + ["--run", "c-bad", "--plan-corruption", "0.5", "--plan-corruption-source", "random",
                           "--plan-corruption-prior-key", pp_key, "--plan-corruption-prior-sha256", pp_sha])
    assert tmt.main(detail + ["--run", "c-random", "--plan-corruption", "1", "--plan-corruption-max", "1",
                              "--plan-corruption-source", "random"]) == 0
    assert tmt.main(detail + ["--run", "c-prior", "--plan-corruption", "1", "--plan-corruption-max", "1",
                              "--plan-corruption-prior-key", pp_key, "--plan-corruption-prior-sha256", pp_sha]) == 0
    for run in ("c-random", "c-prior"):
        saved = torch.load(io.BytesIO((store / f"training-runs/{run}/checkpoints/step-00000002.pt").read_bytes()),
                           weights_only=False)
        assert saved["config"]["plan_corruption"] == 1. and saved["config"]["plan_tokenizer"] == plan.config.sha256()
        logged = json.loads((store / f"training-runs/{run}/log/step-00000002.json").read_text())
        assert any("train/plan_corrupted" in e for e in logged)
    assert saved["plan_corruption_prior_sha256"] == pp_sha
    with pytest.raises(SystemExit, match="not a plan prior over this plan tokenizer"):
        tmt.main(detail + ["--run", "c-bad", "--plan-corruption", "1", "--plan-corruption-prior-key", plain_key,
                           "--plan-corruption-prior-sha256", plain_sha])
    assert tmt.main(prior + ["--stage", "plan-prior", "--run", "pp2", *with_plan, "--seed", "1"]) == 0
    pp2 = "training-runs/pp2/checkpoints/step-00000002.pt"
    with pytest.raises(SystemExit, match="different config or manifest"):       # resume under another plan prior
        tmt.main(detail + ["--run", "c-prior", "--steps", "4", "--plan-corruption", "1", "--plan-corruption-max", "1",
                           "--plan-corruption-prior-key", pp2, "--plan-corruption-prior-sha256", sha(store / pp2)])

    # review fix 5 and 3: the round-17 diagnostics (back-translation replaced by a stand-in)
    diag = load("round17_diagnostics")
    common = ["--store", f"dir:{store}", "--device", "cpu", "--part-temperature", "body=1.1,face=1.0"]   # 2 parts here
    for key in (plain_key, pp_key, detail_key):
        out = f"diag/text-gain-{key.split('/')[1]}.json"
        assert diag.main(["text-gain", *common, "--checkpoint-key", key, "--checkpoint-sha256", sha(store / key),
                          "--manifest-key", "manifest.json", "--manifest-sha256", digest, "--partition", "train",
                          "--conform-to-corpus", "", "--batch-size", "3", "--batches", "2", "--output-key", out]) == 0
        report = json.loads((store / out).read_text())
        assert {"text_gain", "text_gain_stage1", "tokens"} <= set(report) and report["batches"] == 1
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        diag.main(["text-gain", *common, "--checkpoint-key", pp_key, "--checkpoint-sha256", "0" * 64,
                   "--manifest-key", "manifest.json", "--manifest-sha256", digest, "--output-key", "x.json"])
    from evaluate_text_to_motion import load_state
    manifest = json.loads((store / "manifest.json").read_text())
    states = [load_state(ds, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in manifest["records"]]
    texts = [r["text"] for r in manifest["records"]]
    seconds = [float(st.timestamps[-1] - st.timestamps[0]) + 1 / 24. for st in states]
    monkeypatch.setattr(diag, "pilot_clips", lambda store_, args: (texts, states, seconds))
    reads = []
    fake = lambda batch: reads.append(len(batch)) or [texts[i % len(texts)] for i in range(len(batch))]
    pins = ["--plan-tokenizer-key", plan_key, "--plan-tokenizer-sha256", plan_sha, "--mix-key", "manifest.json",
            "--mix-sha256", digest]
    assert diag.main(["plan-route", *common, "--checkpoint-key", pp_key, "--checkpoint-sha256", pp_sha, *pins,
                      "--output-key", "diag/plan-route.json"], translate_fn=fake) == 0
    route = json.loads((store / "diag/plan-route.json").read_text())
    assert {"plan_route", "plan_round_trip", "real", "blank", "plan_route_minus_blank",
            "plan_route_minus_plan_round_trip"} <= set(route) and route["clips"] == 4
    with pytest.raises(SystemExit, match="needs a plan prior"):
        diag.main(["plan-route", *common, "--checkpoint-key", detail_key, "--checkpoint-sha256", sha(store / detail_key),
                   *pins, "--output-key", "x.json"], translate_fn=fake)
    assert diag.main(["plan-following", *common, "--checkpoint-key", detail_key, "--checkpoint-sha256",
                      sha(store / detail_key), *pins, "--output-key", "diag/following.json"], translate_fn=fake) == 0
    following = json.loads((store / "diag/following.json").read_text())
    assert following["pairs"] == [[0, 1], [2, 3], [1, 0], [3, 2]]
    for name in ("empty_text", "other_text", "real"):
        assert following[name]["pairs"] == 4 and 0 <= following[name]["rate"] <= 1
