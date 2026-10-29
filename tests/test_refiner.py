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


def test_history_none_reads_no_target_history_and_refines_every_step_at_once():
    """r2 (round 15 amendment 9): the decoder input is the start code at every step, in the loss and when refining;
    r1's configuration (and its SHA-256) is unchanged by the new field."""
    tokenizer, prior = f12_like()
    r1 = rf.RefinerConfig.from_prior(prior.config)
    assert r1.history == "teacher" and "history" not in r1.to_dict()
    legacy = dict(r1.to_dict())
    assert rf.RefinerConfig.from_dict(legacy) == r1                           # an r1 checkpoint's config loads
    config = rf.RefinerConfig.from_prior(prior.config, history="none", source_dropout=0.)
    assert config.to_dict()["history"] == "none" and config.sha256() != r1.sha256()
    assert rf.RefinerConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="history must be"):
        rf.RefinerConfig.from_prior(prior.config, history="free")
    assert config.history_dropout == 0                                       # nothing to drop, so none recorded
    with pytest.raises(ValueError, match="history_dropout must be 0"):
        rf.RefinerConfig.from_prior(prior.config, history="none", history_dropout=.5)
    torch.manual_seed(0)
    a, b = (rf.tokenize_state(tokenizer, signer_state(s, frames=18, amplitude=.4)) for s in (1, 2))
    source, mask = rf.pad_sources(rf.SentenceRefiner(config), [a["codes"]], "cpu")
    target = b["codes"][None].clone()
    valid = torch.ones((1, 18), dtype=torch.bool)
    valid[0, 5] = False
    changed = target.clone()
    changed[0, 5] = (changed[0, 5] + 1) % prior.K                             # step 5 is only ever read as history
    for history, same in (("none", True), ("teacher", False)):
        model = rf.SentenceRefiner(rf.RefinerConfig.from_prior(prior.config, history=history, source_dropout=0.))
        model.warm_start(prior.state_dict())
        model.eval()
        one, _ = rf.refiner_loss(model, source, mask, target, valid, train=False)
        two, _ = rf.refiner_loss(model, source, mask, changed, valid, train=False)
        assert torch.isclose(one, two).item() is same, history
    # the empty-source loss is the dropped-row loss: the composition replaced, the clock kept
    model = rf.SentenceRefiner(config)
    model.warm_start(prior.state_dict())
    model.eval()
    with_source, _ = rf.refiner_loss(model, source, mask, target, valid, train=False)
    kept = source.clone(), mask.clone()
    empty, _ = rf.refiner_loss(model, source, mask, target, valid, train=False, empty_source=True)
    assert torch.equal(source, kept[0]) and torch.equal(mask, kept[1])      # the caller's tensors are not touched
    with torch.no_grad():                                                     # by hand: one start-code position,
        e_codes, e_mask = model.empty_source(1, "cpu")                        # the clock on the source's 18 steps
        h = model.prior.temporal_states(model.encode_source(e_codes, e_mask), e_mask, model.no_history(1, 18, "cpu"),
                                        model.clock(18, "cpu"), torch.tensor([18 * model.step / rf.FPS]))
        gold = target[valid]
        logits = model.prior.depth_logits(h[valid], gold[:, :-1])
        by_hand = torch.nn.functional.cross_entropy(logits.reshape(-1, model.K), gold.reshape(-1))
    assert not torch.isclose(with_source, empty) and torch.isclose(empty, by_hand)
    # a padded row scores in a batch as it scores alone (token-weighted), and nothing leaks between rows
    pair = [a["codes"], b["codes"][:11]]
    batch_source, batch_mask = rf.pad_sources(model, pair, "cpu")
    batch_target = torch.stack([b["codes"], torch.cat((a["codes"][:11], torch.zeros_like(a["codes"][11:])))])
    batch_valid = torch.stack([torch.ones(18, dtype=torch.bool), torch.arange(18) < 11])
    together_loss, _ = rf.refiner_loss(model, batch_source, batch_mask, batch_target, batch_valid, train=False)
    alone_losses = [rf.refiner_loss(model, *rf.pad_sources(model, [pair[r]], "cpu"),
                                    batch_target[r:r + 1, :len(pair[r])], batch_valid[r:r + 1, :len(pair[r])],
                                    train=False)[0] for r in (0, 1)]
    assert torch.isclose(together_loss, (alone_losses[0] * 18 + alone_losses[1] * 11) / 29, atol=1e-6)
    # refining: the input's length, seeded, in range; greedy draws do not depend on the batch a source shares
    sources = [a["codes"], a["codes"][:7]]
    one = rf.refine_codes(model, sources, guidance=1.5, temperature={"body": 1.1}, seed=4)
    two = rf.refine_codes(model, sources, guidance=1.5, temperature={"body": 1.1}, seed=4)
    assert [x.shape for x in one] == [(18, model.J), (7, model.J)]
    assert all(torch.equal(x, y) for x, y in zip(one, two))
    assert ((one[0] >= 0) & (one[0] < model.K)).all()
    together = rf.refine_codes(model, sources, temperature=1e-6, guidance=1.5)
    alone = [rf.refine_codes(model, [s], temperature=1e-6, guidance=1.5)[0] for s in sources]
    assert all(torch.equal(x, y) for x, y in zip(together, alone))
    # greedy refined codes, scored as the target of a mixed-length batch, are every step's argmax (accuracy 1)
    greedy_pair = rf.refine_codes(model, pair, temperature=1e-6)
    g_target = torch.stack([greedy_pair[0], torch.cat((greedy_pair[1], torch.zeros((7, model.J), dtype=torch.long)))])
    _, terms = rf.refiner_loss(model, batch_source, batch_mask, g_target, batch_valid, train=False)
    assert terms["token_accuracy"] == 1.
    # the r2 path computes the states once per branch (never the step-by-step loop)
    calls, original = [], model.prior.temporal_states
    model.prior.temporal_states = lambda *x, **k: calls.append(1) or original(*x, **k)
    try:
        rf.refine_codes(model, sources, guidance=1.5, seed=1)
        assert len(calls) == 2
        calls.clear()
        rf.refine_codes(model, sources, seed=1)
        assert len(calls) == 1
    finally:
        del model.prior.temporal_states
    # the refined steps are the argmax of the loss path's own logits (training and refining read the same inputs)
    with torch.no_grad():
        memory = model.encode_source(source, mask)
        h = model.prior.temporal_states(memory, mask, model.no_history(1, 18, "cpu"), model.clock(18, "cpu"),
                                        mask.sum(1).float() * model.step / rf.FPS)[0]
        greedy = rf.refine_codes(model, [a["codes"]], temperature=1e-6)[0]
        logits = model.prior.depth_logits(h, greedy[:, :-1])
    assert torch.equal(logits.argmax(-1), greedy)


def _r1_refine_codes(model, sources, *, guidance, temperature, top_k, seed):
    """r1's step-by-step sampler as it stood before r2 (commit 6a3d920), kept verbatim as the golden reference."""
    import math
    prior = model.prior
    codes, mask = rf.pad_sources(model, sources, "cpu")
    B, L = mask.shape
    memory = model.encode_source(codes, mask)
    if guidance != 1.:
        e_codes, e_mask = model.empty_source(1, "cpu")
        empty = model.encode_source(e_codes, e_mask).expand(B, -1, -1)
        empty_mask = e_mask.expand(B, -1)
    lengths = mask.sum(1).float()
    duration = lengths * model.step / rf.FPS
    clock = model.clock(L, "cpu").expand(B, -1)
    names = [part for part, _ in model.config.prior.tokenizer.part_slices()]
    per_code = rf._per_code(temperature, names, model.J)
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.full((B, 1, model.J), model.K, dtype=torch.long)
    for k in range(L):
        h = prior.temporal_states(memory, mask, sequence, clock[:, :k + 1], duration)[:, -1]
        h_free = (prior.temporal_states(empty, empty_mask, sequence, clock[:, :k + 1], duration)[:, -1]
                  if guidance != 1. else None)
        step_codes = torch.zeros((B, 0), dtype=torch.long)
        for j in range(model.J):
            logits = prior.depth_logits(h, step_codes)[:, -1]
            if h_free is not None:
                free = prior.depth_logits(h_free, step_codes)[:, -1]
                logits = free + guidance * (logits - free)
            logits = logits / max(per_code[j], 1e-6)
            if top_k:
                threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -math.inf)
            choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator)
            step_codes = torch.cat((step_codes, choice), 1)
        sequence = torch.cat((sequence, step_codes[:, None]), 1)
    return [sequence[r, 1:int(lengths[r]) + 1] for r in range(B)]


def test_r1_sampling_is_unchanged_by_the_r2_refactor():
    tokenizer, prior = f12_like()
    torch.manual_seed(0)
    model = rf.SentenceRefiner(rf.RefinerConfig.from_prior(prior.config))
    model.warm_start(prior.state_dict())
    model.eval()
    a = rf.tokenize_state(tokenizer, signer_state(1, frames=18, amplitude=.4))
    sources = [a["codes"], a["codes"][:9]]
    for kw in (dict(guidance=1.5, temperature={"body": 1.1}, top_k=5, seed=3),
               dict(guidance=1., temperature=.8, top_k=0, seed=0)):
        with torch.no_grad():
            golden = _r1_refine_codes(model, sources, **kw)
        now = rf.refine_codes(model, sources, **kw)
        assert all(torch.equal(x, y) for x, y in zip(now, golden)), kw


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
        h2s("v2", "Show you how to stir it", "val", "c03"),
        h2s("near", "cut the onion in half now"),                            # Jaccard 5/6 with a val sentence
        h2s("within", "today I will show you how to stir it slowly"),        # contains a 6-token val sentence
        h2s("short", "cut the onion"),                                       # inside a val sentence, but 3 tokens
        dict(h2s("x", "other corpus"), corpus="asllvd-text-v1"))]
    jobs, counts = bp.sentence_jobs(dict(records=records))
    assert [r["sign"] for r in jobs] == ["a", "b", "short"]
    assert counts["near_val_or_test:jaccard"] == 1 and counts["near_val_or_test:containment"] == 1
    assert [r["sign"] for r in bp.sentence_jobs(dict(records=records), larger=.7)[0]] == ["a", "b", "short", "within"]
    loose = bp.sentence_jobs(dict(records=records), near=.9, min_tokens=6)[0]
    assert [r["sign"] for r in loose] == ["a", "b", "near", "short"]
    assert "within" in {r["sign"] for r in bp.sentence_jobs(dict(records=records), min_tokens=7)[0]}
    assert counts["signer9_or_unknown_component"] == 3 and counts["text_in_val_or_test"] == 1
    assert counts["not_train:train/test"] == 1 and counts["no_text"] == 1 and counts["train_sentences"] == 3
    with pytest.raises(SystemExit, match="twice"):
        bp.sentence_jobs(dict(records=records[:1] * 2))


def test_near_duplicates_need_the_overlap_to_cover_the_larger_sentence():
    """Review of 36247fb: containment of a short held-out phrase ("what do you see") in a long train sentence is not
    a near-duplicate; edited copies of held-out sentences still are."""
    held = ["What do you see?", "Today we are going to talk about how to tie a fly.",
            "Make sure you keep the knife sharp at all times.", "This is the best way to cut the onion in half.",
            "Hold the brush lightly between your thumb and finger.", "We'll start by mixing the flour and the sugar."]
    near = bp.NearDuplicates(held)
    for text in ("So now what do you see in this picture here, look at it closely.",   # contains a 4-token phrase
                 "What do you see now?",                                               # 4/5: Jaccard .8 -> caught
                 "I want to show you how to tie a knot."):
        assert (near.reason(text) is None) == (text != "What do you see now?"), text
    edited = ["Today we're going to talk about how to tie a fly",                  # contraction
              "Make sure you keep your knife sharp at all times!",                 # one word changed
              "This is the best way to cut an onion in half",                      # article changed
              "Hold the brush lightly between your thumb and your finger.",        # one word added
              "We will start by mixing the flour and sugar."]                      # one dropped, one expanded
    assert all(near.reason(t) for t in edited), [t for t in edited if not near.reason(t)]
    assert bp.NearDuplicates(held, larger=0.).reason("So now what do you see in this picture here") == "containment"


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
    assert run["selection_components"] == ["c04"] and run["steps"] == 2 and run["epochs"] == .8
    assert run["declared_epochs"] is None and run["target_index_sha256"] == hashlib.sha256(
        store.get("training-runs/refiner-test/target-index.bin")).hexdigest()
    with pytest.raises(SystemExit, match="max-epochs"):
        tr.main(train + ["--steps", "40", "--run", "too-long"])
    with pytest.raises(SystemExit, match="max-epochs"):
        tr.main(train + ["--epochs", "12", "--run", "too-long"])
    best = json.loads(store.get("training-runs/refiner-test/best.json"))
    assert best["checkpoint_key"].startswith("training-runs/refiner-test/checkpoints/step-0000000")
    assert set(best["validation"]) >= {"selection/loss", "selection/tokens", "selection/empty_loss",
                                       "selection/source_gain"}
    v0 = run["initial_validation"]                                        # the warm start, validated first
    assert v0["selection/source_gain"] == pytest.approx(v0["selection/empty_loss"] - v0["selection/loss"])
    ckpt = "training-runs/refiner-test/checkpoints/step-00000002.pt"
    saved = torch.load(io.BytesIO(store.get(ckpt)), map_location="cpu", weights_only=False)
    assert saved["kind"] == rf.KIND and saved["step"] == 2 and saved["tokenizer_key"] == TOKKEY
    with pytest.raises(SystemExit, match="declared"):                     # a resume must keep the declared length
        tr.main(train + ["--steps", "3"])
    with pytest.raises(SystemExit, match="declared"):
        tr.main(train + ["--steps", "2", "--batch-size", "1"])
    final = store.get(ckpt)
    (tmp_path / "store" / ckpt).unlink()                                    # interrupted after step 1
    assert tr.main(train + ["--steps", "2"]) == 0                          # resumes at step 1
    log = json.loads(store.get("training-runs/refiner-test/log/step-00000002.json"))
    assert [e["step"] for e in log] == [0, 1, 2]               # the log (warm start first) is kept across the resume
    resumed = torch.load(io.BytesIO(store.get(ckpt)), map_location="cpu", weights_only=False)
    first = torch.load(io.BytesIO(final), map_location="cpu", weights_only=False)
    assert all(torch.equal(resumed["model"][k], first["model"][k]) for k in first["model"])   # the same step 2
    # refine inside the compose evaluation: oracle_stage2 and its controls, /rest_only, @recon
    common = ["run", "--mix-key", "rmix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
              "--device", "cpu", "--cache-dir", str(tmp_path / "cache"), "--variants", "oracle_stage2", "--block", "2",
              "--core-margin", "0.1", "--clips", "5", "--recognizer-run", "sign-recognizer-v3d",
              "--refiner-checkpoint-key", ckpt, "--refiner-sha256",
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
    rule = ce.REFINED_RULE
    assert rule.split(",")[0] == "bleu:recon"
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
        "bleu_vs_recon_lower_above_0", "bleu_vs_mismatched_words_lower_above_0", "bleu_vs_rest_only_lower_above_0",
        "spotting_recall_above_mismatched"}
    # memorisation (report-only): text reads of the listed forms, code reads with the real clips as chance level
    mem = m1["memorisation"]
    assert {"real", v, f"{v}@recon", f"{v}@refined", f"{v}/rest_only@refined"} <= set(mem["text"])
    assert set(mem["codes"]) == {"real", v, f"{v}@refined"} and mem["train_texts"] == 12
    for row in mem["text"].values():
        assert row["n"] == 5 and 0 <= row["mean_best_bleu"] <= 100 and 0 <= row["exact_match_rate"] <= 1
    assert mem == m2["memorisation"]
    # the refined rule reads spotting with recogniser v3 only
    shard = json.loads(store.get("e/whole.json"))
    shard["settings"]["recognizer_run"] = "sign-recognizer-v2"
    store.put("e/v2.json", json.dumps(shard).encode())
    with pytest.raises(SystemExit, match="sign-recognizer-v3d"):
        ce.main(merge + ["--shard-keys", "e/v2.json", "--output-key", "e/m4.json"])
    assert m1["refiner"]["sha256"] == hashlib.sha256(store.get(ckpt)).hexdigest()
    # refusals: a refined rule on unrefined shards; a refiner that is not the pinned file
    plain = [a for a in common[:-4]] + ["--output-key", "e/plain.json"]
    assert ce.main(plain) == 0
    with pytest.raises(SystemExit, match="not refined"):
        ce.main(merge + ["--shard-keys", "e/plain.json", "--output-key", "e/m3.json"])
    with pytest.raises(SystemExit, match="pinned"):
        ce.main(common[:-1] + ["0" * 64, "--output-key", "e/bad.json"])
    # r2 (--history none): trained, resumable configuration recorded, and refined inside the compose evaluation
    assert tr.main(train + ["--steps", "2", "--run", "refiner-r2-test", "--history", "none"]) == 0
    r2 = "training-runs/refiner-r2-test/checkpoints/step-00000002.pt"
    model, _, meta = rf.load_refiner(store, r2)
    assert model.config.history == "none" and meta["config_sha256"] == model.config.sha256()
    with pytest.raises(SystemExit, match="another configuration"):     # r1's flags cannot resume an r2 run
        tr.main(train + ["--steps", "2", "--run", "refiner-r2-test"])
    r2_args = [a if a != ckpt else r2 for a in common[:-1]] + [hashlib.sha256(store.get(r2)).hexdigest()]
    assert ce.main(r2_args + ["--refiner-temperature", "0.7", "--output-key", "e/r2.json"]) == 0
    r2_shard = json.loads(store.get("e/r2.json"))
    assert {f"{v}@refined", f"{v}/rest_only@refined"} <= set(r2_shard["clips"][0]["hypotheses"])
    assert r2_shard["settings"]["refiner"]["checkpoint_key"] == r2
    assert r2_shard["settings"]["refiner"]["temperature"] == {"body": .7, "left_hand": .7, "right_hand": .7,
                                                              "face": .7}
    assert whole["settings"]["refiner"]["temperature"] == {"body": 1.1, "left_hand": 1.1, "right_hand": 1.1,
                                                           "face": 1.}                # default: --part-temperature


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
        def jobs(manifest, bad=bad, **kw):
            kept, counts = original_jobs(manifest, **kw)
            return [records[bad]] + kept, counts
        monkeypatch.setattr(bp, "sentence_jobs", jobs)
        with pytest.raises(bp.HygieneError):
            bp.main(build_args(tmp_path, sha, tsha, f"poison-{bad}"))
        assert not store.exists(bp.pair_key(f"poison-{bad}", records[bad]))


# ---- memorisation diagnostics, composition parity, knobs -------------------------------------------------------------

def test_memorisation_reads():
    from signtranslator.generation import memorisation as mz
    assert mz.sentence_bleu("cut the onion now".split(), "cut the onion now".split()) == pytest.approx(100.)
    assert mz.sentence_bleu("salt".split(), "cut the onion".split()) == 0.
    assert 0 < mz.sentence_bleu("cut the salt".split(), "cut the onion now".split()) < 100
    index = mz.TextIndex(["Cut the onion now.", "Add some salt", "stir it slowly"], candidates=2)
    assert index.nearest("cut the onion now .") == dict(best_bleu=100., nearest=0, exact=True)
    near = index.nearest("add the salt")
    assert near["nearest"] == 1 and not near["exact"] and 0 < near["best_bleu"] < 100
    assert index.nearest("zebra")["best_bleu"] == 0.
    summary = index.summary(["cut the onion now .", "zebra"])
    assert summary["exact_match_rate"] == .5 and summary["n"] == 2
    rng = np.random.default_rng(0)
    targets = [rng.integers(0, 32, size=(40, 12)) for _ in range(6)]
    rest = np.zeros((30, 12), dtype=np.int64)
    targets += [np.concatenate([rest, t]) for t in targets[:2]]               # a rest pose shared by targets 6, 7
    codes = mz.CodeIndex.build(targets, parts=4, codebook=32, n=6, max_df=.2)
    hit = codes.query(targets[3])
    assert hit["max_containment"] == 1. and hit["nearest"] == 3 and hit["seen_fraction"] == 1.
    miss = codes.query(rng.integers(0, 32, size=(40, 12)))
    assert miss["max_containment"] < .1
    assert codes.query(rest)["ngrams"] == 0                                    # common n-grams are not evidence
    assert codes.query(targets[0][:3])["max_containment"] is None              # shorter than n
    again = mz.CodeIndex.from_arrays(*rf.unpack_pair(rf.pack_pair(*codes.to_arrays())))
    assert again.query(targets[3]) == hit
    assert mz.code_summary([hit, miss, codes.query(rest)])["without_ngrams"] == 1


def eval_context(tmp_path, store, sha, *, space=None, bridge="linear", tighten=False):
    """The pair builder's composition context on the fixture, built as its main builds it."""
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    from round15_oracle import TEMPLATE_CORPUS, word_clip, word_pools
    from train_motion_tokens import layout_template
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get("rmix.json") + b"\n")
        manifest = load_manifest(path, expected_sha256=sha)
    corpora = isolated_corpora(manifest)
    vocab = training_vocabulary(manifest, corpora)
    model, _ = fit.fit_model(fit.agc._store(f"dir:{tmp_path / 'store'}"), "rmix.json", sha, corpora)
    cache = ContentCache(tmp_path / "pcache", 2 ** 28)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
    context = dict(pools=word_pools(manifest, corpora), loaded={}, model=model, vocab=vocab,
                   fractions=fit.sign_fractions(manifest, vocab), margin=.1, tighten=tighten, space=space,
                   bridge=bridge)
    load = lambda r: word_clip(r, cache=cache, fetch=store.get, template=template)  # noqa: E731
    return manifest, context, load


def test_pair_composition_is_the_evaluation_form(tmp_path, refiner_store):
    """build_pair's composition (compose_input) equals round15_compose_eval's oracle_stage2 form (its jobs from
    plan_jobs, oracle_material by the oracle's draws, render_form) for the same records, with and without knobs."""
    import round15_stage2 as r2
    import scripts.round15_space_stats as ss
    from scripts.pilot_content_check import validation_sentences
    from signtranslator.generation.compositional import CORE_RULE
    from round15_oracle import choose_record
    store, sha, _, _ = refiner_store
    assert ss.main(["--mix-key", "mix.json", "--mix-sha256", hashlib.sha256(store.get("mix.json")).hexdigest(),
                    "--store", f"dir:{tmp_path / 'store'}", "--output-key", "stats.json"]) == 0   # no signer 9 there
    stats = json.loads(store.get("stats.json"))["statistics"]
    for knobs in (dict(), dict(space=dict(stats=stats, target="sentence"), bridge="minimum-jerk", tighten=True)):
        manifest, context, load = eval_context(tmp_path, store, sha, **knobs)
        val = validation_sentences(manifest, 5)
        details, words = ce.stage1_selection([r["text"] for r in val], context["vocab"])
        plans = [r2.plan_words(w, context["model"], context["vocab"], rows=d) for w, d in zip(words, details)]
        seconds = [1.2 + .1 * i for i in range(len(val))]
        jobs = r2.plan_jobs(["oracle_stage2"], plans, [1, 2, 3, 4, 0], [list(range(len(p["entries"]))) for p in plans],
                            seconds=seconds, vocab=context["vocab"], timing="sign-fitted", overlap=4)
        compared = 0
        for i, plan in enumerate(plans):
            job = jobs[("oracle_stage2", "", i)]
            if job is None:
                continue
            material = r2.oracle_material(job, i, context["pools"], loaded={}, load=load,
                                          rule=dict(CORE_RULE, margin=.1), timing="sign-fitted",
                                          fractions=context["fractions"], margin=.1, tighten=context["tighten"])
            theirs, plan_theirs = r2.render_form(material, job, seconds[i], context["model"], timing="sign-fitted",
                                                 total_fit="pad", sample_id="x", space=context["space"],
                                                 bridge=context["bridge"])
            records = [choose_record(context["pools"], i, w, "matched") for w in job["words"]]
            ours, plan_ours, _ = bp.compose_input(bp.composition_job(plan), records, seconds[i], context=context,
                                                  load=load, sample_id="x")
            assert plan_ours == plan_theirs and len(ours.timestamps) == len(theirs.timestamps)
            for name, c in theirs.channels.items():
                np.testing.assert_array_equal(ours.channels[name].values, c.values)
                np.testing.assert_array_equal(ours.channels[name].valid, c.valid)
            compared += 1
        assert compared >= 3


def test_knobs_reach_the_pairs_and_the_rest_only_control(tmp_path, refiner_store):
    """The composition knobs are recorded in the pairs' configuration and change the inputs; the refined evaluation
    with every knob on renders /rest_only (static poses under space normalisation) without a warning."""
    import scripts.round15_space_stats as ss
    store, sha, tsha, psha = refiner_store
    assert ss.main(["--mix-key", "mix.json", "--mix-sha256", hashlib.sha256(store.get("mix.json")).hexdigest(),
                    "--store", f"dir:{tmp_path / 'store'}", "--output-key", "stats.json"]) == 0   # no signer 9 there
    ssha = hashlib.sha256(store.get("stats.json")).hexdigest()
    knobs = ["--space-normalise", "sentence", "--space-stats-key", "stats.json", "--space-stats-sha256", ssha,
             "--bridge", "minimum-jerk", "--tighten-cores"]
    with pytest.raises(SystemExit):
        bp.main(build_args(tmp_path, sha, tsha, "pk", "--space-normalise", "sentence"))
    assert bp.main(build_args(tmp_path, sha, tsha, "pk", *knobs)) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "pk", "--merge")) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "p0")) == 0
    assert bp.main(build_args(tmp_path, sha, tsha, "p0", "--merge")) == 0
    k, plain = json.loads(store.get("pk/index.json")), json.loads(store.get("p0/index.json"))
    assert k["config"]["knobs"] == dict(space_normalise="sentence", bridge="minimum-jerk", tighten_cores=True,
                                        space_stats=dict(key="stats.json", sha256=ssha))
    assert plain["config"]["knobs"]["space_normalise"] == "off" and plain["config"]["near_duplicate"] == .8
    a = {e["sign"]: e.get("pair_sha256") for e in k["entries"] if e["status"] == "converted"}
    b = {e["sign"]: e.get("pair_sha256") for e in plain["entries"] if e["status"] == "converted"}
    assert a.keys() == b.keys() and all(a[s] != b[s] for s in a)
    train = ["--store", f"dir:{tmp_path / 'store'}", "--pairs-prefix", "pk", "--init-key", "f12.pt", "--init-sha256",
             psha, "--run", "refiner-knobs", "--batch-size", "2", "--steps", "1", "--device", "cpu"]
    assert tr.main(train) == 0
    ckpt = "training-runs/refiner-knobs/checkpoints/step-00000001.pt"
    run = ["run", "--mix-key", "rmix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}", "--device",
           "cpu", "--cache-dir", str(tmp_path / "cache"), "--variants", "oracle_stage2", "--block", "2",
           "--core-margin", "0.1", "--clips", "5", "--recognizer-run", "sign-recognizer-v3d",
           "--refiner-checkpoint-key", ckpt, *knobs, "--output-key", "e/knobs.json"]
    assert ce.main(run) == 0
    shard = json.loads(store.get("e/knobs.json"))
    assert shard["settings"]["space_normalise"] == "sentence" and shard["settings"]["refiner"]["checkpoint_key"] == ckpt
    assert {"oracle_stage2/rest_only", "oracle_stage2/rest_only@refined"} <= set(shard["clips"][0]["hypotheses"])
