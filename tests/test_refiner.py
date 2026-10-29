"""Sentence refiner: the model (warm start from the prior, no text path, loss, sampling), deterministic pair files, the
pair builder's hygiene (no val / test / signer-9 / unknown-component How2Sign record and no ASL Citizen or non-train
word clip can enter a pair; no evaluation sentence becomes a target), its determinism across shardings and its
write-once resumption, the trainer's fixed batches and selection on held-out train signers, and a tiny end-to-end run
pair build -> train 2 steps -> refine inside round15_compose_eval (incl. /rest_only, @recon, the refined rule and
sharded == unsharded)."""
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.build_refiner_pairs as bp
import scripts.round15_compose_eval as ce
import scripts.train_refiner as tr
import round15_stage2_fit as fit  # noqa: E402  (the module the scripts import, on their path)
from signtranslator.data_engineering.corpus_manifest import KINDS, canonical_json, load_manifest, usable_channels
from signtranslator.generation import refiner as rf
from signtranslator.generation.compositional import ROUND15_CORPORA, parts_from_pieces, take_frames
from signtranslator.pose.multichannel import CHANNELS
from tests.test_generation_signing_space import signer_state
from tests.test_round15_compositional import stubbed  # noqa: F401  (the stub instruments fixture)
from tests.test_round15_oracle import TRAIN
from tests.test_round15_stage2 import asllrp_inputs
from tests.test_sentence_exposure_diagnostic import build_store, tiny_prior

TOKKEY = "training-runs/tok-test/checkpoints/step-00000001.pt"
H2S = "how2sign-two-view-text-v1"
CITIZEN = "asl-citizen-text-v1"


# ---- the model -----------------------------------------------------------------------------------------------------

def f12_like(seed=0):
    return tiny_prior(seed, history_stages=1, history_dropout=.5, part_valid_loss=True)


def test_warm_start_takes_the_prior_and_drops_the_text_path():
    tokenizer, prior = f12_like()
    config = rf.RefinerConfig.from_prior(prior.config)
    assert (config.source_layers, config.history_dropout, config.part_valid_loss) == (1, .5, True)
    assert rf.RefinerConfig.from_prior(prior.config, history_dropout=.2).history_dropout == .2
    model = rf.SentenceRefiner(config)
    for name in rf.TEXT_MODULES:
        assert not hasattr(model.prior, name)
    info = model.warm_start(prior.state_dict())
    assert info["source_encoder_copied"] == info["source_encoder_tensors"]
    assert set(info["unused_prefixes"]) == {"embed", "grammar", "duration", "text_encoder"}
    theirs = prior.state_dict()
    for k, v in model.prior.state_dict().items():
        assert torch.equal(v, theirs[k]), k
    for k, v in model.source_encoder.state_dict().items():
        assert torch.equal(v, theirs[f"text_encoder.{k}"]), k
    assert torch.equal(model.source_time.weight, prior.time.weight)
    assert rf.RefinerConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    bad = dict(theirs)
    bad.pop("heads_bias")
    with pytest.raises(ValueError, match="lacks"):
        rf.SentenceRefiner(config).warm_start(bad)


def test_loss_trains_and_sampling_keeps_the_input_length_and_its_seed():
    tokenizer, prior = f12_like()
    torch.manual_seed(0)
    model = rf.SentenceRefiner(rf.RefinerConfig.from_prior(prior.config, source_dropout=.3))
    model.warm_start(prior.state_dict())
    a, b = (rf.tokenize_state(tokenizer, signer_state(s, frames=18, amplitude=.4)) for s in (1, 2))
    source, mask = rf.pad_sources(model, [a["codes"], a["codes"][:10]], "cpu")
    target = torch.stack([b["codes"], b["codes"]])
    valid = torch.stack([b["valid"], torch.arange(18) < 10])
    part = torch.stack([b["part"], b["part"]])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    losses = []
    for _ in range(25):
        loss, terms = rf.refiner_loss(model, source, mask, target, valid, part)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] and set(terms) == {"tokens", "token_accuracy", "stage1_accuracy"}
    model.eval()
    one = rf.refine_codes(model, [a["codes"], a["codes"][:7]], guidance=1.5, temperature={"body": 1.1}, seed=4)
    two = rf.refine_codes(model, [a["codes"], a["codes"][:7]], guidance=1.5, temperature={"body": 1.1}, seed=4)
    assert [x.shape for x in one] == [(18, model.J), (7, model.J)]
    assert all(torch.equal(x, y) for x, y in zip(one, two))
    assert ((one[0] >= 0) & (one[0] < model.K)).all()
    with pytest.raises(ValueError, match="unknown parts"):
        rf.refine_codes(model, [a["codes"]], temperature={"tail": 1.})
    states = rf.refine_states(model, tokenizer, [signer_state(3, frames=12)], seed=0)
    assert len(states[0].timestamps) == 12
    states[0].validate()


def test_pair_files_are_deterministic_and_checked():
    arrays = dict(target_codes=np.arange(12, dtype=np.int16).reshape(4, 3), target_valid=np.array([1, 1, 0, 1], bool),
                  source_codes_0=np.zeros((4, 3), np.int16))
    meta = dict(sign="s", b=[1, 2], a="x")
    payload = rf.pack_pair(arrays, meta)
    assert payload == rf.pack_pair(dict(reversed(list(arrays.items()))), dict(reversed(list(meta.items()))))
    back, m = rf.unpack_pair(payload)
    assert m == meta and all(np.array_equal(back[k], arrays[k]) and back[k].dtype == arrays[k].dtype for k in arrays)
    with pytest.raises(ValueError, match="truncated"):
        rf.unpack_pair(payload[:-1])
    with pytest.raises(ValueError, match="integer or boolean"):
        rf.pack_pair(dict(x=np.zeros(2)), {})
    with pytest.raises(ValueError, match="not a refiner pair"):
        rf.unpack_pair(b"nope" + payload)


# ---- builder logic ---------------------------------------------------------------------------------------------------

def h2s(sign, text, partition="train", component="c02", **kw):
    return dict(corpus=H2S, sign=sign, text=text, partition=partition, signer_component=component,
                continuous=dict(status="converted", state_key=f"states/{sign}.npz", state_sha256="0" * 64,
                                available_channels=list(CHANNELS), seconds=1.), **kw)


def usable(r):
    r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
    return r


def test_sentence_jobs_keep_only_train_sentences_of_readable_signers():
    records = [usable(r) for r in (
        h2s("a", "Cut the onion."), h2s("b", "stir it", component="c04"),
        h2s("v", "Cut the onion in half", "val", "c01"), h2s("t", "the test", "test", "c00"),
        h2s("dup", "cut the onion in HALF!"),                                   # an evaluation sentence: excluded
        h2s("nine", "nine", component="c00"), h2s("unknown", "who", component=None),
        h2s("nine2", "nine two", source=dict(signer_id="9")),
        h2s("moved", "moved", source_partition="test"),
        h2s("held", "held out", "heldout_official"), h2s("empty", "?!"),
        dict(h2s("x", "other corpus"), corpus="asllvd-text-v1"))]
    jobs, counts = bp.sentence_jobs(dict(records=records))
    assert [r["sign"] for r in jobs] == ["a", "b"]
    assert counts["signer9_or_unknown_component"] == 3 and counts["text_in_val_or_test"] == 1
    assert counts["not_train:train/test"] == 1 and counts["no_text"] == 1 and counts["train_sentences"] == 2
    with pytest.raises(SystemExit, match="twice"):
        bp.sentence_jobs(dict(records=records[:1] * 2))


def test_selection_components_hold_out_the_smallest_train_signers():
    jobs = [dict(signer_component=c) for c in ["c02"] * 10 + ["c04"] * 3 + ["c06"] * 1 + ["c08"] * 1]
    assert bp.selection_components(jobs, .05) == ["c06"]               # ties by name; 1 of 15 >= 5%
    assert bp.selection_components(jobs, .2) == ["c04", "c06", "c08"]
    assert bp.selection_components(jobs, .99) == ["c04", "c06", "c08"]  # never every component
    assert bp.selection_components(jobs, .05, ["c02"]) == ["c02"]
    with pytest.raises(SystemExit, match="unknown"):
        bp.selection_components(jobs, .05, ["c09"])
    with pytest.raises(SystemExit, match="every"):
        bp.selection_components(jobs, .05, ["c02", "c04", "c06", "c08"])
    with pytest.raises(SystemExit, match="held out"):
        bp.selection_components(jobs[:10], .05)


def test_draws_are_seeded_per_sentence_and_input_and_refuse_forbidden_clips():
    pools = {"cut": [dict(sign=f"c{k}") for k in range(50)], "salt": [dict(sign=f"s{k}") for k in range(50)]}
    record = h2s("a", "cut salt")
    one = bp.draw_records(pools, record, 0, ["cut", "salt", "cut"])
    assert one == bp.draw_records(pools, record, 0, ["cut", "salt", "cut"])
    assert one != bp.draw_records(pools, record, 1, ["cut", "salt", "cut"])
    assert bp.draw_seed(record, 0) != bp.draw_seed(h2s("b", "cut salt"), 0)
    with pytest.raises(KeyError):
        bp.draw_records(pools, record, 0, ["onion"])
    with pytest.raises(SystemExit, match="Citizen"):
        bp.check_corpora(("asllvd-text-v1", CITIZEN))
    with pytest.raises(SystemExit):
        bp.check_corpora((H2S,))
    ok = dict(corpus="asllvd-text-v1", sign="w", partition="train")
    bp.check_word_record(ok, ("asllvd-text-v1",))
    for bad in (dict(ok, corpus=CITIZEN), dict(ok, partition="val"), dict(ok, partition="test"),
                dict(ok, source_partition="test")):
        with pytest.raises(RuntimeError):
            bp.check_word_record(bad, ("asllvd-text-v1", CITIZEN))


def test_batches_are_fixed_by_the_step():
    a = tr.batch_items(7, 10, 4, 4, 0)
    assert a == tr.batch_items(7, 10, 4, 4, 0) and len(a) == 4
    epoch = [i for s in range(5) for i, _ in tr.batch_items(s, 10, 2, 4, 0)]
    assert sorted(epoch) == list(range(10))                                # one epoch: every pair once
    assert tr.batch_items(0, 3, 5, 4, 0)[:3] != [] and len(tr.batch_items(0, 3, 5, 4, 0)) == 5
    assert all(0 <= k < 4 for _, k in tr.batch_items(3, 10, 8, 4, 1))


def test_rest_material_holds_each_clip_at_its_first_frame():
    clips = [signer_state(s, frames=10, amplitude=.4) for s in (1, 2)]
    pieces = [dict(state=c, lo=2, hi=6) for c in clips]
    material = dict(pieces=pieces, parts=parts_from_pieces(pieces))
    rest = ce.rest_material(material)
    for p, c in zip(rest["pieces"], clips):
        v = p["state"].channels["right_hand"].values
        assert len(v) == 10 and (v == c.channels["right_hand"].values[0]).all()
    for core in rest["parts"]["cores"]:
        assert len(core.timestamps) == 5
    chain = dict(parts=dict(cores=[take_frames(clips[0], np.arange(2, 5)), take_frames(clips[0], np.arange(6, 8))],
                            transitions=[None], lead=(clips[0], 2), trail=(clips[0], 8)))
    r = ce.rest_material(chain)["parts"]
    assert [len(c.timestamps) for c in r["cores"]] == [3, 2]
    assert (r["cores"][1].channels["body"].values == clips[0].channels["body"].values[0]).all()
    assert material["pieces"][0]["state"] is clips[0]                         # the input is untouched


# ---- end to end ------------------------------------------------------------------------------------------------------

SENTENCES = [("s01", "Cut the onion now.", "c02"), ("s02", "Stir the salt.", "c02"), ("s03", "heat the knife", "c02"),
             ("s04", "Thank you for the salt", "c06"), ("s05", "cut and stir the onion", "c06"),
             ("s06", "hello there", "c02"), ("s07", "Salt, then heat.", "c04"), ("s08", "cut it", "c04")]
FORBIDDEN = [("nine", "cut the salt", "train", "c00"), ("unknown", "stir the onion", "train", None),
             ("dupval", "Cut the onion in half", "train", "c02"), ("tst", "heat the salt", "test", "c00")]


def put_state(store, key, state):
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "s.npz"
        sha = state.save(path)
        store.put(key, path.read_bytes())
    return sha


def add(store, records, sign, text, partition, corpus, component, frames, seed):
    state = signer_state(seed, frames=frames, amplitude=.4)
    key = f"states/{sign}.npz"
    sha = put_state(store, key, state)
    r = dict(sign=sign, corpus=corpus, partition=partition, signer_component=component, text=text,
             continuous=dict(status="converted", state_key=key, state_sha256=sha, available_channels=list(CHANNELS),
                             seconds=frames / 24.))
    records.append(usable(r))


@pytest.fixture
def refiner_store(tmp_path, stubbed, monkeypatch):  # noqa: F811
    store, _, _ = build_store(tmp_path / "store", corpus="asllvd-text-v1", train_texts=TRAIN)
    monkeypatch.setattr(fit, "ASLLEX_SHA256", asllrp_inputs(store))
    monkeypatch.setattr(fit, "annotated_seconds", lambda manifest, vocab: {e: .3 for e in vocab})
    manifest = json.loads(store.get("mix.json"))
    records = manifest["records"]
    for k, (sign, text, component) in enumerate(SENTENCES):
        add(store, records, sign, text, "train", H2S, component, 30 + 2 * k, 100 + k)
    for k, (sign, text, partition, component) in enumerate(FORBIDDEN):
        add(store, records, sign, text, partition, H2S, component, 30, 200 + k)
    for k, w in enumerate(("cut", "onion", "salt")):                      # second ASLLVD clips: draws vary with k
        add(store, records, f"asllvd-{w}-2", w, "train", "asllvd-text-v1", None, 16, 300 + k)
        add(store, records, f"asllvd-{w}-val", w, "val", "asllvd-text-v1", None, 16, 310 + k)
        add(store, records, f"citizen-{w}", w, "train", CITIZEN, None, 16, 320 + k)
    payload = canonical_json(dict(records=records))
    store.put("rmix.json", payload)
    sha = hashlib.sha256(payload).hexdigest()
    tokenizer, prior = f12_like()
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_tokenizer", config=tokenizer.config.to_dict(), model=tokenizer.state_dict(),
                    ema=None), buffer)
    store.put(TOKKEY, buffer.getvalue())
    tsha = hashlib.sha256(buffer.getvalue()).hexdigest()
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_prior", config=prior.config.to_dict(), model=prior.state_dict(),
                    ema=prior.state_dict(), tokenizer_state=tokenizer.state_dict(), tokenizer_key=TOKKEY,
                    manifest_sha256=sha, config_sha256=prior.config.sha256(), step=60000), buffer)
    store.put("f12.pt", buffer.getvalue())
    psha = hashlib.sha256(buffer.getvalue()).hexdigest()
    return store, sha, tsha, psha


def build_args(tmp_path, sha, tsha, prefix, *extra):
    return ["--store", f"dir:{tmp_path / 'store'}", "--prefix", prefix, "--mix-key", "rmix.json", "--mix-sha256", sha,
            "--tokenizer-key", TOKKEY, "--tokenizer-sha256", tsha, "--prior-checkpoint-key", "f12.pt",
            "--expect-model-sha256", "", "--k", "2", "--cache-dir", str(tmp_path / "cache"),
            "--selection-components", "c04", *extra]


def test_pair_build_hygiene_determinism_and_resumption(tmp_path, refiner_store):
    store, sha, tsha, _ = refiner_store
    assert bp.main(build_args(tmp_path, sha, tsha, "pa")) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "pa", "--merge")) == 0
    for s in (0, 1):
        assert bp.main(build_args(tmp_path, sha, tsha, "pb", "--shards", "2", "--shard", str(s))) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "pb", "--shards", "2", "--merge")) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "pw", "--workers", "2")) == 0          # worker processes
    assert bp.main(build_args(tmp_path, sha, tsha, "pw", "--merge")) == 0
    a, b = json.loads(store.get("pa/index.json")), json.loads(store.get("pb/index.json"))
    w = json.loads(store.get("pw/index.json"))
    assert a["config"] == b["config"] == w["config"] and a["status_counts"] == b["status_counts"]
    hashes = {e["sign"]: e.get("pair_sha256") for e in a["entries"]}
    assert hashes == {e["sign"]: e.get("pair_sha256") for e in b["entries"]}
    assert hashes == {e["sign"]: e.get("pair_sha256") for e in w["entries"]}
    # hygiene: only the train sentences of readable signers, and their words from train clips of allowed corpora
    assert sorted(e["sign"] for e in a["entries"]) == sorted([s for s, _, _ in SENTENCES] + ["clip0"])
    counts = a["config"]["sentence_counts"]
    assert counts["signer9_or_unknown_component"] == 2 and counts["text_in_val_or_test"] == 1
    status = {e["sign"]: e["status"] for e in a["entries"]}
    assert status["s06"] == status["clip0"] == "no_words" and status["s01"] == "converted"
    assert a["config"]["corpora"] == list(ROUND15_CORPORA) and a["config"]["selection_components"] == ["c04"]
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get("rmix.json") + b"\n")
        manifest = load_manifest(path, expected_sha256=sha)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    done = [e for e in a["entries"] if e["status"] == "converted"]
    assert len(done) == 7 and a["splits"] == {"train": 5, "selection": 2}
    drawn = set()
    for e in done:
        arrays, meta = rf.unpack_pair(store.get(e["pair_key"]))
        assert hashlib.sha256(store.get(e["pair_key"])).hexdigest() == e["pair_sha256"]
        assert meta["split"] == ("selection" if meta["signer_component"] == "c04" else "train") == e["split"]
        assert len(meta["inputs"]) == 2 and set(arrays) == {"target_codes", "target_valid", "target_part",
                                                            "source_codes_0", "source_codes_1"}
        assert arrays["source_codes_0"].shape == arrays["target_codes"].shape
        for row in meta["inputs"]:
            for corpus, sign, state_sha in row["records"]:
                r = by_sign[sign]
                assert corpus == r["corpus"] == "asllvd-text-v1" and r["partition"] == "train"
                assert state_sha == r["continuous"]["state_sha256"]
                drawn.add(sign)
    assert not any("citizen" in s or s.endswith("-val") for s in drawn) and "asllvd-cut-2" in by_sign
    # resumption: a finished shard is skipped; a lost index is rebuilt from the stored pairs, never rewritten
    assert bp.main(build_args(tmp_path, sha, tsha, "pa")) == 0
    (tmp_path / "store" / "pa" / "index" / "shard-000-of-001.json").unlink()
    before = {e["pair_key"]: store.get(e["pair_key"]) for e in done}
    assert bp.main(build_args(tmp_path, sha, tsha, "pa")) == 0
    again = json.loads(store.get("pa/index/shard-000-of-001.json"))
    assert {e["status"] for e in again["entries"] if e["sign"] in {d["sign"] for d in done}} == {"already_present"}
    assert all(store.get(k) == v for k, v in before.items())
    # the prefix is written once: another configuration is refused, as is a tokenizer that is not the prior's
    with pytest.raises(SystemExit, match="another configuration"):
        bp.main(build_args(tmp_path, sha, tsha, "pa", "--k", "3"))
    tokenizer, _ = tiny_prior(5)
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_tokenizer", config=tokenizer.config.to_dict(), model=tokenizer.state_dict()),
               buffer)
    store.put("training-runs/other/checkpoints/step-00000001.pt", buffer.getvalue())
    args = build_args(tmp_path, sha, hashlib.sha256(buffer.getvalue()).hexdigest(), "pc")
    args[args.index(TOKKEY)] = "training-runs/other/checkpoints/step-00000001.pt"
    with pytest.raises(SystemExit, match="tokenizer_state"):
        bp.main(args)
    with pytest.raises(SystemExit, match="Citizen"):
        bp.main(build_args(tmp_path, sha, tsha, "pd", "--corpora", f"asllvd-text-v1,{CITIZEN}"))


def test_end_to_end_build_train_refine(tmp_path, refiner_store):
    store, sha, tsha, psha = refiner_store
    assert bp.main(build_args(tmp_path, sha, tsha, "pairs")) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "pairs", "--merge")) == 0
    train = ["--store", f"dir:{tmp_path / 'store'}", "--pairs-prefix", "pairs", "--init-key", "f12.pt",
             "--init-sha256", psha, "--run", "refiner-test", "--batch-size", "2", "--val-every", "1",
             "--checkpoint-every", "2", "--warmup", "1", "--device", "cpu"]
    with pytest.raises(SystemExit, match="init-sha256"):
        tr.main([a if a != psha else "0" * 64 for a in train] + ["--steps", "2"])
    assert tr.main(train + ["--steps", "2"]) == 0
    run = json.loads(store.get("training-runs/refiner-test/run.json"))
    assert run["train_pairs"] == 5 and run["selection_pairs"] == 2 and run["selection_items"] == 4
    assert run["selection_components"] == ["c04"]
    best = json.loads(store.get("training-runs/refiner-test/best.json"))
    assert best["checkpoint_key"].startswith("training-runs/refiner-test/checkpoints/step-0000000")
    assert set(best["validation"]) >= {"selection/loss", "selection/tokens"}
    ckpt = "training-runs/refiner-test/checkpoints/step-00000002.pt"
    saved = torch.load(io.BytesIO(store.get(ckpt)), map_location="cpu", weights_only=False)
    assert saved["kind"] == rf.KIND and saved["step"] == 2 and saved["tokenizer_key"] == TOKKEY
    assert tr.main(train + ["--steps", "3"]) == 0                          # resumes at step 2
    log = json.loads(store.get("training-runs/refiner-test/log/step-00000003.json"))
    assert [e["step"] for e in log] == [3]
    # refine inside the compose evaluation: oracle_stage2 and its controls, /rest_only, @recon
    common = ["run", "--mix-key", "rmix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
              "--device", "cpu", "--cache-dir", str(tmp_path / "cache"), "--variants", "oracle_stage2", "--block", "2",
              "--core-margin", "0.1", "--clips", "5", "--refiner-checkpoint-key", ckpt, "--refiner-sha256",
              hashlib.sha256(store.get(ckpt)).hexdigest()]
    assert ce.main(common + ["--output-key", "e/whole.json"]) == 0
    for s in (0, 1):
        assert ce.main(common + ["--shards", "2", "--shard", str(s), "--output-key", f"e/p{s}.json"]) == 0
    whole = json.loads(store.get("e/whole.json"))
    v = "oracle_stage2"
    forms = set(whole["clips"][0]["hypotheses"])
    assert {f"{v}@refined", f"{v}@recon", f"{v}/rest_only", f"{v}/rest_only@refined", f"{v}/shuffled@refined",
            f"{v}/mismatched_words@refined"} <= forms
    assert set(whole["clips"][0]["spotting"]) == {v, f"{v}@refined"}
    assert whole["settings"]["refiner"]["checkpoint_key"] == ckpt
    rule = "bleu:unrefined,bleu:mismatched_words,bleu:rest_only,spotting"
    merge = ["merge", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "20", "--refined-rule", rule]
    assert ce.main(merge + ["--shard-keys", "e/whole.json", "--output-key", "e/m1.json"]) == 0
    assert ce.main(merge + ["--shard-keys", "e/p1.json,e/p0.json", "--output-key", "e/m2.json"]) == 0
    m1, m2 = json.loads(store.get("e/m1.json")), json.loads(store.get("e/m2.json"))
    for name in m1["forms"]:
        assert m1["forms"][name]["hypotheses"] == m2["forms"][name]["hypotheses"], name
    row = m1["forms"][f"{v}@refined"]
    assert {"minus_unrefined", "minus_recon", "minus_mismatched_words", "minus_rest_only", "minus_shuffled"} <= set(row)
    d = m1["verdicts"][f"{v}@refined"]
    assert d["verdict"] in ("PROMISING", "NOT PROMISING") and set(d["checks"]) == {
        "bleu_vs_unrefined_lower_above_0", "bleu_vs_mismatched_words_lower_above_0", "bleu_vs_rest_only_lower_above_0",
        "spotting_recall_above_mismatched"}
    assert m1["refiner"]["sha256"] == hashlib.sha256(store.get(ckpt)).hexdigest()
    # refusals: a refined rule on unrefined shards; a refiner that is not the pinned file
    plain = [a for a in common[:-4]] + ["--output-key", "e/plain.json"]
    assert ce.main(plain) == 0
    with pytest.raises(SystemExit, match="not refined"):
        ce.main(merge + ["--shard-keys", "e/plain.json", "--output-key", "e/m3.json"])
    with pytest.raises(SystemExit, match="pinned"):
        ce.main(common[:-1] + ["0" * 64, "--output-key", "e/bad.json"])


def test_a_forbidden_record_that_reaches_the_builder_stops_it(tmp_path, refiner_store, monkeypatch):
    """Adversarial: the filters are bypassed (poisoned word pools, a poisoned sentence list) and the per-record guards
    must still stop the run at once (never retried or skipped), before that sentence's pair is written."""
    import round15_oracle
    store, sha, tsha, _ = refiner_store
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get("rmix.json") + b"\n")
        records = {r["sign"]: r for r in load_manifest(path, expected_sha256=sha)["records"]}
    original = round15_oracle.word_pools
    for bad in ("citizen-cut", "asllvd-cut-val"):
        def poisoned(manifest, corpora, partition="train", bad=bad):
            pools = original(manifest, corpora, partition)
            pools["cut"] = [records[bad]]
            return pools
        monkeypatch.setattr(round15_oracle, "word_pools", poisoned)
        with pytest.raises(bp.HygieneError, match="may not provide words"):
            bp.main(build_args(tmp_path, sha, tsha, f"poison-{bad}"))
        assert not store.exists(bp.pair_key(f"poison-{bad}", records["s01"]))       # s01 signs CUT
    monkeypatch.setattr(round15_oracle, "word_pools", original)
    original_jobs = bp.sentence_jobs
    for bad in ("nine", "unknown", "tst", "clip20"):                    # signer 9, unknown component, test, val
        def jobs(manifest, bad=bad):
            kept, counts = original_jobs(manifest)
            return [records[bad]] + kept, counts
        monkeypatch.setattr(bp, "sentence_jobs", jobs)
        with pytest.raises(bp.HygieneError):
            bp.main(build_args(tmp_path, sha, tsha, f"poison-{bad}"))
        assert not store.exists(bp.pair_key(f"poison-{bad}", records[bad]))
