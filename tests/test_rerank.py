"""Word reranking by the prior's own teacher-forced PMI: the seeds (sample 0 the word's own), the score (equal to the
step-by-step autoregressive log-likelihood; only the prior is consulted), N = 1 identical to plain generation, N > 1
deterministic and the argmax, the composition path and the evaluation script."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.round15_compose_eval as ce
import signtranslator.generation.compositional as comp
import signtranslator.generation.rerank as rr
from tests.test_round15_compose_eval import run_args
from tests.test_round15_stage2 import stage2_store  # noqa: F401
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_sentence_exposure_diagnostic import TEMPERATURE, tiny_prior

IDENTITY = lambda s: s                                                  # noqa: E731


def test_seeds_and_choice():
    assert rr.rerank_seeds(20_000_005, 3) == [20_000_005, 120_000_005, 220_000_005]
    assert rr.rerank_seeds(7, 1) == [7]
    with pytest.raises(ValueError):
        rr.rerank_seeds(7, 0)
    assert rr.choose([.1, .3, .3, -1.]) == 1 and rr.choose([-2.]) == 0


def test_word_pmi_is_the_autoregressive_log_likelihood():
    from signtranslator.generation.motion_rvq import generate_rvq_motion
    from signtranslator.generation.motion_tokens import _empty_like, token_seconds
    from signtranslator.generation.text_to_motion import PAD, empty_loci, encode_texts, generation_loci
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    _, codes = generate_rvq_motion(prior, tokenizer, ["cut"], fps=24., durations=[.3], guidance=1.5,
                                   temperature=TEMPERATURE, seed=4, return_codes=True)
    codes = codes[0]
    score = rr.word_pmi(prior, "cut", codes, .3)
    config = prior.config

    def stepwise(text):                    # one step at a time, as the sampler conditions each step
        if text:
            tokens, mask = encode_texts([text], max_bytes=config.max_text_bytes)
            memory, _, mask = prior.encode(tokens, mask, generation_loci(config, [text], None, "cpu"))
        else:
            empty = _empty_like(encode_texts(["cut"])[0][:1, :2])
            memory, _, mask = prior.encode(empty, empty != PAD, empty_loci(config, "cpu"))
        clock = token_seconds(config, codes.shape[0], 24., "cpu")
        seq = torch.full((1, 1, prior.J), prior.K, dtype=torch.long)
        total = 0.
        with torch.no_grad():
            for k in range(codes.shape[0]):
                h = prior.temporal_states(memory, mask, seq, clock[:, :k + 1], torch.tensor([.3]))[:, -1]
                for j in range(prior.J):
                    lp = prior.depth_logits(h, codes[None, k, :j])[:, -1].log_softmax(-1)
                    total += float(lp[0, codes[k, j]])
                seq = torch.cat((seq, codes[None, None, k]), 1)
        return total
    assert score["conditional"] == pytest.approx(stepwise("cut"), abs=1e-3)
    assert score["unconditional"] == pytest.approx(stepwise(""), abs=1e-3)
    assert score["codes"] == codes.numel()
    assert score["pmi_per_code"] == pytest.approx((score["conditional"] - score["unconditional"]) / codes.numel(),
                                                  abs=1e-5)


def test_only_the_prior_scores():
    assert list(inspect.signature(rr.word_pmi).parameters) == ["prior", "word", "codes", "seconds", "fps"]
    source = Path(rr.__file__).read_text()
    tree = ast.parse(source)
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "torch", "motion_rvq", "motion_tokens", "text_to_motion", "pose.resample"}
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"', "'")))
    body = ast.get_source_segment(source, next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                               and n.name == "rerank_word"))
    for banned in ("recogni", "translat", "spotting", "bleu", "back_translat", "sign_recognizer"):
        assert banned not in body.lower() and banned not in code.split('"""')[-1].lower(), banned


def test_one_sample_is_plain_generation_and_more_pick_the_argmax():
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    plain = comp.generate_word(prior, tokenizer, "cut", .4, 9, guidance=1.5, temperature=TEMPERATURE, post=IDENTITY)
    one, info = rr.rerank_word(prior, tokenizer, "cut", .4, 9, 1, guidance=1.5, temperature=TEMPERATURE,
                               post=IDENTITY)
    assert info["chosen"] == 0 and info["seeds"] == [9]
    for name in plain.channels:
        np.testing.assert_array_equal(one.channels[name].values, plain.channels[name].values)
    best, info = rr.rerank_word(prior, tokenizer, "cut", .4, 9, 4, guidance=1.5, temperature=TEMPERATURE,
                                post=IDENTITY)
    again, info2 = rr.rerank_word(prior, tokenizer, "cut", .4, 9, 4, guidance=1.5, temperature=TEMPERATURE,
                                  post=IDENTITY)
    assert info == info2 and info["chosen"] == int(np.argmax(info["scores"]))
    assert len(set(info["scores"])) > 1                                   # the samples differ
    chosen = comp.generate_word(prior, tokenizer, "cut", .4, info["seeds"][info["chosen"]], guidance=1.5,
                                temperature=TEMPERATURE, post=IDENTITY)
    for name in chosen.channels:
        np.testing.assert_array_equal(best.channels[name].values, chosen.channels[name].values)
        np.testing.assert_array_equal(again.channels[name].values, chosen.channels[name].values)


def test_compose_cores_rerank():
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    kw = dict(guidance=1.5, temperature=TEMPERATURE, post=IDENTITY, device=torch.device("cpu"))
    a = comp.compose_cores(prior, tokenizer, ["cut", "go"], [.4, .3], "core_crossfade", [3, 4], **kw)
    b = comp.compose_cores(prior, tokenizer, ["cut", "go"], [.4, .3], "core_crossfade", [3, 4], rerank=1, **kw)
    for p, q in zip(a["pieces"], b["pieces"]):
        np.testing.assert_array_equal(p["state"].channels["right_hand"].values, q["state"].channels["right_hand"].values)
    assert "rerank" not in a["segments"][0]
    c = comp.compose_cores(prior, tokenizer, ["cut", "go"], [.4, .3], "core_crossfade", [3, 4], rerank=3, **kw)
    for s in c["segments"]:
        assert len(s["rerank"]["scores"]) == 3 and s["rerank"]["chosen"] == int(np.argmax(s["rerank"]["scores"]))
        assert s["rerank"]["seeds"][0] == s["seed"]
    with pytest.raises(ValueError):
        comp.compose_cores(prior, tokenizer, ["cut"], [.4], "core_chained", [3], rerank=2, **kw)


def test_compose_eval_rerank(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    base = run_args(tmp_path, sha, "--clips", "5")
    base[base.index("core_crossfade,oracle_stage2")] = "core_crossfade"
    assert ce.main(base + ["--output-key", "r/plain.json"]) == 0
    assert ce.main(base + ["--rerank", "1", "--output-key", "r/one.json"]) == 0
    plain, one = json.loads(store.get("r/plain.json")), json.loads(store.get("r/one.json"))
    assert [r["hypotheses"] for r in plain["clips"]] == [r["hypotheses"] for r in one["clips"]]
    assert all(r["rerank"] is None for r in one["clips"])
    assert ce.main(base + ["--rerank", "2", "--output-key", "r/two.json"]) == 0
    assert ce.main(base + ["--rerank", "2", "--output-key", "r/two-again.json"]) == 0
    two, again = json.loads(store.get("r/two.json")), json.loads(store.get("r/two-again.json"))
    assert [r["hypotheses"] for r in two["clips"]] == [r["hypotheses"] for r in again["clips"]]
    assert two["settings"]["rerank"] == 2
    for r in two["clips"]:
        for block, words in r["rerank"]["core_crossfade"].items():
            for w in words:
                assert len(w["scores"]) == 2 and w["chosen"] == int(np.argmax(w["scores"]))
    for bad in (["--rerank", "0"], ["--rerank", "2", "--timing", "stage1"],
                ["--rerank", "2", "--variants", "core_chained"]):
        args = [a for a in base if a != "--core-margin" and a != "0.1"] if "stage1" in bad else base
        with pytest.raises(SystemExit):
            ce.main(args + bad + ["--output-file", str(tmp_path / "x.json")])
