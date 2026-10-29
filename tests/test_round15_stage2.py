"""Round 15 stage 2: the stage-2 selection (contractions, the Sign Bank / ASL-LEX lexicon), the structure fit and its
operations (keep, map, insert IX, drop, reorder), the fit guard (ASLLRP train only, never How2Sign), the timing model
and the frame plan, the sign-core cut and the rendering, the generated and real-clip material, the evaluation
script's seeds, jobs, rule and an end-to-end run on a tiny prior with stub instruments."""
import collections
import random
import csv
import hashlib
import io
import json

import numpy as np
import pytest
import torch

import scripts.asllrp_gloss_corpus as agc
import scripts.round15_stage2 as r2
import round15_stage2_fit as fit  # noqa: E402  (the module the scripts import, on their path)
import signtranslator.generation.compositional as comp
import signtranslator.generation.compositional_grammar as cg
from tests.test_compositional_baseline import full_state
from tests.test_generation_signing_space import signer_state
from tests.test_sentence_exposure_diagnostic import TEMPERATURE, tiny_prior

CPU = torch.device("cpu")
IDENTITY = lambda s: s                                                  # noqa: E731


def utt(english, gold, k=0, partition="train", collection="asllrp-continuous", timed=None, seconds=3.):
    return dict(id=f"u{k}", collection=collection, partition=partition, english=english, gold=list(gold),
                seconds=seconds, timed=timed or [])


# ---- tokens and selection ----------------------------------------------------------------------------------------

def test_tokens_split_contractions_as_the_groundwork():
    text = "I don't know; it's Mom's car. Won't, can't, I'm, we'll"
    assert cg.tokens(text) == agc.english_tokens(text)
    assert cg.tokens("They're here, we've gone, he'd go") == ["they", "are", "here", "we", "have", "gone", "he",
                                                             "would", "go"]


def test_selection_rows_use_the_lexicon_only_where_the_vocabulary_has_no_entry():
    vocab = {"child": 1., "dog": 1., "not": 1., "the": 1., "thank you": 1., "mother": 1., "buy": 1.}
    lexicon = {"children": dict(entry="child", tier="signbank_variant"), "dog": dict(entry="mother", tier="x"),
               "mom": dict(entry="mother", tier="asllex_translation"), "ice cream": dict(entry="buy", tier="t"),
               "a": dict(entry="dog", tier="t"), "x": dict(entry="dog", tier="t"), "puppy": dict(entry="zzz", tier="t")}
    rows = cg.selection_rows("The children don't like Mom's dog, moms, ice cream, a x puppy", vocab, lexicon)
    got = [(r["text"], r["entry"], r["kind"]) for r in rows]
    assert got == [("the", "the", "stop"), ("children", "child", "lexicon"), ("do", None, "stop"),
                   ("not", "not", "exact"), ("like", None, "unmatched"), ("mom", "mother", "lexicon"),
                   ("'s", None, "unmatched"), ("dog", "dog", "exact"),                 # the vocabulary wins
                   ("moms", "mother", "lexicon"),                                      # inflection, then lexicon
                   ("ice cream", "buy", "lexicon"), ("a", None, "stop"),             # stop words are never keys
                   ("x", None, "unmatched"), ("puppy", None, "unmatched")]          # letters; unknown entries
    assert cg.selected(rows) == ["child", "not", "mother", "dog", "mother", "buy"]
    assert [r["tier"] for r in rows if r["kind"] == "lexicon"] == ["signbank_variant", "asllex_translation",
                                                                   "asllex_translation", "t"]
    # without a lexicon, stage 2's selection is stage 1's on the split tokens
    assert cg.selected(cg.selection_rows("Mom's dogs", vocab)) == comp.select_words("Mom 's dogs", vocab) == ["dog"]


# ---- the fit guard -------------------------------------------------------------------------------------------------

def test_fits_refuse_anything_but_asllrp_train():
    vocab = {"cut": 1., "onion": 1.}
    good = [utt("cut the onion", ["cut", "onion"], k) for k in range(3)]
    cg.check_fit_records(good)
    for bad in (dict(collection="how2sign-two-view-text-v1"), dict(partition="val"), dict(partition="test"),
                dict(collection=None)):
        rows = good + [dict(good[0], id="bad", **bad)]
        with pytest.raises(ValueError):
            cg.check_fit_records(rows)
        with pytest.raises(ValueError):
            cg.fit_structure(rows, vocab)
        with pytest.raises(ValueError):
            cg.fit_timing(rows, vocab, {"cut": .5})
        with pytest.raises(ValueError):
            fit.fit_stage2(rows, vocab, {}, {"cut": .5})
    cg.check_fit_records([dict(good[0], collection="rit-sentences")])


# ---- structure -----------------------------------------------------------------------------------------------------

VOCAB = {"cut": 1., "onion": 1., "with": 1., "dog": 1., "child": 1., "children": 1., "ix": 1., "sell": 1.,
         "store": 1., "what": 1., "go": 1., "not happy": 1., "happy": 1., "she": 1., "poss": 1., "red": 1.}


def corpus():
    """Six distinct sentences of each pattern (the trailing digit makes them distinct; digits are never entries)."""
    rows = []
    for k in range(6):
        rows.append(utt(f"Cut the onion with the knife {k}", ["cut", "onion"], len(rows)))    # "with" never signed
        rows.append(utt(f"The dog is red {k}", ["red"], len(rows)))                           # open class, unsigned
        rows.append(utt(f"The children go {k}", ["child", "go"], len(rows)))                  # a learned map
        rows.append(utt(f"He will go to the store {k}", ["ix", "go", "sell"], len(rows)))     # insert IX, a map
        rows.append(utt(f"He will go {k}", ["ix", "go"], len(rows)))
        rows.append(utt(f"What did she cut {k}?", ["ix", "cut", "what"], len(rows)))          # WH-final
        rows.append(utt(f"I am not happy {k}", ["happy", "not"], len(rows)))                  # negation kept
    return rows


def kind_of(name):
    from signtranslator.pose.multichannel import CHANNELS
    return CHANNELS[name][0]


def test_fit_structure_keeps_maps_inserts_and_drops_by_hit_rate():
    model = cg.fit_structure(corpus(), VOCAB, threshold=.5, min_count=5, move_min_count=5)
    t = model["tokens"]
    assert t["with"]["op"] == "drop" and t["with"]["rate"] == 0.                 # closed class, never signed
    assert t["dog"]["op"] == "keep" and t["dog"]["entry"] == "dog"               # open class: never dropped
    assert t["children"]["op"] == "map" and t["children"]["entry"] == "child"    # learned by linking
    assert t["store"]["op"] == "map" and t["store"]["entry"] == "sell"
    assert t["he"]["op"] == "insert" and t["he"]["entry"] == "ix"                 # a pronoun points
    assert t["she"]["op"] == "map" and t["she"]["entry"] == "ix"                  # the pointing sign over the word
    assert t["cut"]["op"] == "keep" and t["onion"]["op"] == "keep"
    assert t["not happy"]["entry"] == "not happy"                                 # never a map that loses "not"
    assert "happy" in t["not happy"]["learned"] and t["not happy"]["op"] == "keep"
    assert model["moves"] == {"what": "end"}                                      # WH-final, fitted
    assert model["move_scores"]["what"]["tau_end"] > model["move_scores"]["what"]["tau_stay"]
    assert model["pointing"]["ix"]["op"] == "insert"
    entries, trace = cg.apply_structure(cg.selection_rows("What will she cut with the dog?", VOCAB), model, VOCAB)
    assert entries == ["ix", "cut", "dog", "what"]                                # "will" unseen: none
    ops = {x["text"]: x["op"] for x in trace}
    assert ops == {"what": "keep", "will": "none", "she": "map", "cut": "keep", "with": "drop", "dog": "keep"}
    assert [x.get("moved") for x in trace if x["text"] == "what"] == ["end"]
    # moves off: English order
    assert cg.apply_structure(cg.selection_rows("What did she cut?", VOCAB), model, VOCAB, moves=False)[0] == [
        "what", "ix", "cut"]


def test_rare_tokens_keep_stage_one_and_rare_pronouns_use_the_pooled_pointing_rate():
    model = cg.fit_structure(corpus(), VOCAB, threshold=.5, min_count=5, move_min_count=5)
    assert "they" not in model["tokens"]
    rows = cg.selection_rows("They cut a red onion", VOCAB)
    entries, trace = cg.apply_structure(rows, model, VOCAB)
    assert entries == ["ix", "cut", "red", "onion"]                              # pooled: insert
    assert trace[0]["op"] == "insert"
    no_pool = dict(model, pointing={})
    assert cg.apply_structure(rows, no_pool, VOCAB)[0] == ["cut", "red", "onion"]
    # an empty model is stage 1's selection on the split tokens
    assert cg.apply_structure(rows, {}, VOCAB)[0] == cg.selected(rows)


def test_learned_links_credit_the_token_that_explains_best_and_count_distinct_sentences():
    vocab = {"teach": 1., "agent": 1.}
    rows = []
    for k in range(5):
        rows.append(utt(f"listen to the teacher {k}", ["teach", "agent"], len(rows)))
        rows.append(utt(f"the teacher {k}", ["teach", "agent"], len(rows)))
    links = cg.learned_links(rows, vocab, None, min_count=5)
    assert set(links["teacher"]) == {"teach", "agent"} and "listen" not in links
    same = [utt("the teacher", ["teach"], k) for k in range(9)]                 # one sentence, nine signers
    assert cg.learned_links(same, {"teach": 1.}, None, min_count=2) == {}


def test_names_learn_nothing():
    rows = [utt(f"Then {w} came to San Diego {k}.", ["come", "move away"], k) for k, w in enumerate(
        ["Bob", "Mary", "John", "Ann", "Sue", "Tom"])]
    assert cg.proper_nouns(rows) >= {"san", "diego"} and "came" not in cg.proper_nouns(rows)
    assert "then" not in cg.proper_nouns(rows)                                 # sentence-initial capitals do not count
    links = cg.learned_links(rows, {"come": 1., "move away": 1.}, None, min_count=5)
    assert "diego" not in links and "san" not in links and links["came"]["come"] == 6


def test_apply_moves_is_stable():
    assert cg.apply_moves(["a", "w", "b", "t", "v"], {"w": "end", "v": "end", "t": "front"}) == ["t", "a", "b", "w",
                                                                                                  "v"]
    assert cg.apply_moves([], {"w": "end"}) == []


def test_droppable_and_negation_guard():
    assert cg.droppable("with") and cg.droppable("he") and not cg.droppable("dog") and not cg.droppable("not")
    assert cg.drops_negation("not happy", "happy") and not cg.drops_negation("not have", "none")
    assert not cg.drops_negation("happy", "sad") and not cg.drops_negation(None, "happy")
    # a token whose every candidate would lose its negation keeps stage 1's behaviour
    assert cg._decide("nothing", 5, {"something": 5}, None, .5) == dict(op="none", entry=None, n=5, rate=0.)
    # an entry (its own included) that loses the token's negation is never signed: "not happy" -> upset is refused
    assert cg._decide("not happy", 5, {"upset": 5, "happy": 5}, "upset", .5) == dict(op="none", entry=None, n=5, rate=0.)


# ---- metrics -------------------------------------------------------------------------------------------------------

def test_tau_and_metrics_reproduce_the_groundwork_rule_check():
    assert cg.tau_of(["a", "b", "c"], ["a", "b", "c"]) == 1.
    assert cg.tau_of(["c", "b", "a"], ["a", "b", "c"]) == -1.
    assert cg.tau_of(["a", "x"], ["a", "y"]) is None
    assert cg.tau_of(["a", "b", "a"], ["a", "a", "b"]) == pytest.approx(1 / 3)  # duplicates left to right
    pred = [["a", "b"], ["c"], []]
    gold = [["a", "c", "b"], ["c", "c"], ["d"]]
    agg, rows = cg.sequence_metrics(pred, gold)
    assert agg["bag_precision"] == 1. and agg["bag_recall"] == pytest.approx(3 / 6, abs=1e-4)
    assert agg["entry_wer"] == pytest.approx((1 + 1 + 1) / 6, abs=1e-4)
    assert agg["tau_mean"] == 1. and agg["tau_utterances"] == 1
    # the stage-1 rule's metrics equal the groundwork's rule_check on the same table
    vocab = {"cut": 1., "onion": 1., "knife": 1., "ix": 1.}
    records = [dict(english="Cut the onion with a knife", partition="val",
                    signs=[dict(gloss="KNIFE"), dict(gloss="CUT"), dict(gloss="ONION"), dict(gloss="IX")]),
               dict(english="The knife", partition="val", signs=[dict(gloss="KNIFE"), dict(gloss="XX")])]
    table = {"KNIFE": dict(tier="name", entries=("knife",)), "CUT": dict(tier="signbank_id", entries=("cut",)),
             "ONION": dict(tier="name", entries=("onion",)), "IX": dict(tier="signbank_id", entries=("ix",)),
             "XX": dict(tier=None, entries=())}
    groundwork = agc.rule_check(records, table, vocab)
    ours, _ = cg.sequence_metrics([comp.select_words(r["english"], vocab) for r in records],
                                  [fit.gold_entries(r, {fit.sign_key(s): table[s["gloss"]] for s in r["signs"]})
                                   for r in records])
    for k in ("bag_precision", "bag_recall", "entry_wer"):
        assert ours[k] == groundwork[k]
    assert ours["tau_mean"] == groundwork["shared_entry_order_kendall_tau"]["mean"]
    assert ours["tau_np_mean"] == pytest.approx(-1 / 3, abs=1e-4)                # gold knife cut onion: 2, 0, 1


def test_paired_bootstrap_is_zero_on_itself_and_signs_differences():
    _, a = cg.sequence_metrics([["a", "b"], ["c", "d"], ["e", "f"]], [["a", "b"], ["c", "d"], ["f", "e"]])
    _, b = cg.sequence_metrics([["a"], ["c", "x"], ["f", "e"]], [["a", "b"], ["c", "d"], ["f", "e"]])
    same = cg.paired_bootstrap(a, a, boot=50)
    assert all(v["difference"] == 0 for v in same.values())
    d = cg.paired_bootstrap(a, b, boot=200)
    assert d["bag_recall"]["difference"] == pytest.approx(4 / 6 - 1., abs=1e-4)
    assert d["tau_paired"]["difference"] == pytest.approx(2.) and d["tau_paired"]["utterances"] == 1
    lo, hi = d["bag_recall"]["ci95"]
    assert lo <= d["bag_recall"]["difference"] <= hi
    with pytest.raises(ValueError):
        cg.paired_bootstrap(a, b[:2])


# ---- timing --------------------------------------------------------------------------------------------------------

def test_fit_timing_medians_ratio_quantiles_and_targets():
    typical = {"cut": 2., "onion": 1.5, "rare": 3., "knife": 1.}
    annotated = {"cut": .5, "onion": .6, "knife": .4}
    rows = []
    for k in range(4):
        rows.append(utt("x", [], k, seconds=2., timed=[("cut", "lexical", .2, .4), ("onion", "lexical", .6, .9),
                                                       (None, "index", 1.0, 1.1), ("knife", "lexical", 1.2, 1.5)]))
    t = cg.fit_timing(rows, typical, annotated, min_signs=3, floor_q=.25, ceiling_q=.9)
    assert t["sentence_seconds"] == {"cut": .2, "knife": .3, "onion": .3}
    assert t["ratio"] == pytest.approx(float(np.median([.2 / .5, .3 / .6, .3 / .4] * 4)), abs=1e-4)
    lexical = [.2, .3, .3] * 4
    assert t["floor"] == pytest.approx(np.quantile(lexical, .25), abs=1e-4)
    assert t["ceiling"] == pytest.approx(np.quantile(lexical, .9), abs=1e-4)
    assert t["gap"] == pytest.approx(.1, abs=1e-4)                               # gaps .2, .1, .1 (x4)
    assert t["lead_share"] == pytest.approx(.2 / (.2 + .5), abs=1e-4)
    assert t["trimmed_over_annotated"] == pytest.approx(float(np.median([4., 2.5, 2.5])), abs=1e-4)
    assert cg.target_seconds("cut", t, typical) == (pytest.approx(max(.2, t["floor"])), "asllrp")
    t2 = dict(t, sentence_seconds={}, floor=.01, ceiling=9.)
    assert cg.target_seconds("cut", t2, typical) == (pytest.approx(t["ratio"] * .5), "annotated")
    assert cg.target_seconds("rare", t2, typical) == (pytest.approx(t["ratio"] * 3. / t["trimmed_over_annotated"]),
                                                      "trimmed")
    clipped = cg.target_seconds("rare", dict(t2, ceiling=.1), typical)
    assert clipped == (.1, "trimmed_clipped")
    with pytest.raises(ValueError):
        cg.fit_timing([utt("x", [], timed=[])], typical, annotated)


def test_allocate_and_plan_frames():
    assert cg.allocate([1, 1, 1], 7, [1, 1, 1]) == [3, 2, 2]
    assert sum(cg.allocate([5, 3, 2, 0], 20, [1, 1, 1, 0])) == 20
    assert cg.allocate([5, 5], 1, [1, 1]) == [1, 1]                             # the minima win
    out = cg.allocate([10., 1., 1.], 6, [1, 1, 0])
    assert sum(out) == 6 and out[1] >= 1
    p = cg.plan_frames([.25, .5], .1667, 2., "pad", lead_share=.4)
    assert p["cores"] == [6, 12] and p["gaps"] == [4] and p["fit"] == "padded"
    assert p["lead"] == round(26 * .4) and p["lead"] + p["trail"] == 48 - 22 and p["frames"] == 48
    c = cg.plan_frames([.5, .5, .5], .25, .5, "pad")
    assert c["fit"] == "compressed" and c["frames"] == 12 and c["lead"] == c["trail"] == 0 and min(c["cores"]) >= 1
    s = cg.plan_frames([.25, .5], .1667, 2., "scale")
    assert s["fit"] == "scaled" and s["frames"] == 48 and s["scale"] > 2 and s["lead"] == 0
    n_ = cg.plan_frames([.25, .5], .1667, 2., "none")
    assert n_["frames"] == 22 and n_["fit"] == "none"
    one = cg.plan_frames([.3], .2, .3, "pad")
    assert one["gaps"] == [] and one["frames"] == 7
    with pytest.raises(ValueError):
        cg.plan_frames([], .1, 1., "pad")
    with pytest.raises(ValueError):
        cg.plan_frames([.2], .1, 1., "stretch")


# ---- the core cut and the rendering --------------------------------------------------------------------------------

def test_core_window_is_the_active_span_of_the_manifest_rule():
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    state = signer_state(4, frames=40, amplitude=.6)
    c = comp.core_window(state)
    joints, pos, _ = skeleton_positions(state)
    tracks = comp.side_tracks(state)
    side = c["side"]
    ok = np.asarray(tracks[side][0], bool)
    h = pos[:, joints.index(f"{side}Hand"), 1] - pos[:, joints.index(f"{side}Shoulder"), 1]
    rest, peak = np.percentile(h[ok], 10), np.percentile(h[ok], 95)
    active = np.flatnonzero(ok & (h > rest + .1 * (peak - rest)))
    assert (c["lo"], c["hi"]) == (active[0], active[-1]) and not c["whole"] and c["reason"] == "core"
    # the manifests' window with margin 0 is the same span (in rounded seconds)
    w, _ = comp.trim_window(state, .1, 0., .3, 2 / 24.)
    assert w == (round(active[0] / 24., 4), round(active[-1] / 24., 4))
    wide = comp.core_window(state, dict(margin=2 / 24.))
    assert wide["lo"] == max(c["lo"] - 2, 0) and wide["hi"] == min(c["hi"] + 2, 39)
    untracked = comp.core_window(state, dict(min_tracked=1.01))
    assert untracked["whole"] and (untracked["lo"], untracked["hi"]) == (0, 39) and untracked["reason"] == "arm_untracked"
    short = comp.core_window(state, dict(min_seconds=10.))
    assert short["whole"] and short["reason"] == "short_window"


def frames_of(state):
    return len(state.timestamps)


def test_bridge_lead_in_and_lead_out():
    a, b = full_state(0, frames=8), full_state(1, frames=8, offset=3.)
    g = comp.bridge(a, b, 3)
    assert frames_of(g) == 3
    name = "root_translation" if "root_translation" in a.channels else next(
        n for n, c in a.channels.items() if c.values.shape[-1] == 3 and c.valid.all())
    va, vb = a.channels[name].values[-1], b.channels[name].values[0]
    for k, w in enumerate((.25, .5, .75)):
        if kind_of(name) == "translation":
            np.testing.assert_allclose(g.channels[name].values[k], (1 - w) * va + w * vb, atol=1e-6)
    assert not any(c.observed.any() for c in g.channels.values())
    # observed real data (no inference method): the bridged frames name theirs
    from dataclasses import replace
    real = replace(a, channels={n: replace(c, observed=c.valid.copy(), inference_method=None)
                                for n, c in a.channels.items()})
    g2 = comp.bridge(real, b, 2)
    assert g2.channels[name].inference_method == "compositional bridge" and not g2.channels[name].observed.any()
    s = full_state(2, frames=10)
    lead = comp.lead_in(s, 3, 5)                                       # 3 approach frames, held at frame 0 before
    np.testing.assert_array_equal(lead.channels[name].values, s.channels[name].values[[0, 0, 0, 1, 2]])
    np.testing.assert_array_equal(comp.lead_in(s, 6, 2).channels[name].values, s.channels[name].values[[4, 5]])
    np.testing.assert_array_equal(comp.lead_in(s, 0, 2).channels[name].values, s.channels[name].values[[0, 0]])
    out = comp.lead_out(s, 7, 4)                                       # frames 8, 9, then held at 9
    np.testing.assert_array_equal(out.channels[name].values, s.channels[name].values[[8, 9, 9, 9]])
    np.testing.assert_array_equal(comp.lead_out(s, 9, 2).channels[name].values, s.channels[name].values[[9, 9]])


def test_render_follows_the_plan_with_transitions_or_bridges():
    clips = [full_state(k, frames=12, offset=float(k)) for k in range(3)]
    pieces = [dict(state=c, lo=2, hi=8) for c in clips]
    parts = comp.parts_from_pieces(pieces)
    assert [frames_of(c) for c in parts["cores"]] == [7, 7, 7] and parts["transitions"] == [None, None]
    plan = cg.plan_frames([.2, .3, .25], .125, 2., "pad", lead_share=.5)
    state = comp.render(parts, plan)
    assert frames_of(state) == plan["frames"] == 48
    # the cores are time-resampled copies: the first core starts at the lead's end with the first clip's frame 2
    name = next(n for n, c in clips[0].channels.items() if c.valid.all() and c.values.shape[-1] == 3
                and kind_of(n) == "translation")
    np.testing.assert_allclose(state.channels[name].values[plan["lead"]], clips[0].channels[name].values[2], atol=1e-6)
    # reordered pieces: the lead comes from the new first clip
    shuffled = comp.parts_from_pieces(pieces, [2, 0, 1])
    s2 = comp.render(shuffled, plan)
    np.testing.assert_allclose(s2.channels[name].values[plan["lead"]], clips[2].channels[name].values[2], atol=1e-6)
    # a generated transition is resampled into the gap instead of a bridge
    t = full_state(9, frames=5, offset=100.)
    with_t = dict(parts, transitions=[t, None])
    s3 = comp.render(with_t, plan)
    k = plan["lead"] + plan["cores"][0]
    np.testing.assert_allclose(s3.channels[name].values[k], t.channels[name].values[0], atol=1e-6)
    with pytest.raises(ValueError):
        comp.render(parts, cg.plan_frames([.2, .3], .125, 2., "pad"))


# ---- generated material ----------------------------------------------------------------------------------------------

def test_core_crossfade_generates_each_word_alone_and_cuts_its_core():
    tokenizer, prior = tiny_prior(locus_conditioning=True)
    out = comp.compose_cores(prior, tokenizer, ["cut", "go"], [.5, .4], "core_crossfade", [3, 4], guidance=1.5,
                             temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    word = comp.generate_word(prior, tokenizer, "cut", .5, 3, guidance=1.5, temperature=TEMPERATURE, post=IDENTITY)
    first = out["pieces"][0]["state"]
    for name in word.channels:
        np.testing.assert_allclose(first.channels[name].values, word.channels[name].values, atol=1e-6)
    for p, s in zip(out["pieces"], out["segments"]):
        assert (p["lo"], p["hi"]) == (s["lo"], s["hi"]) and 0 <= p["lo"] <= p["hi"] < s["frames"]
        c = comp.core_window(p["state"])
        assert (c["lo"], c["hi"]) == (p["lo"], p["hi"])
    assert [frames_of(c) for c in out["parts"]["cores"]] == [s["hi"] - s["lo"] + 1 for s in out["segments"]]
    with pytest.raises(ValueError):
        comp.compose_cores(prior, tokenizer, ["cut"], [.5], "glued", [1], guidance=1., temperature=None, post=IDENTITY)
    with pytest.raises(ValueError):
        comp.compose_cores(prior, tokenizer, ["cut"], [.5, .2], "core_chained", [1], guidance=1., temperature=None,
                           post=IDENTITY)


def test_core_chained_continues_from_the_kept_codes_up_to_each_core_end(monkeypatch):
    tokenizer, prior = tiny_prior(locus_conditioning=True, history_stages=1)
    cores = iter([dict(lo=2, hi=6, reason="core", side="Right", whole=False),
                  dict(lo=0, hi=9, reason="no_sign_window", side=None, whole=True),
                  dict(lo=3, hi=4, reason="core", side="Right", whole=False)])
    seen = []
    monkeypatch.setattr(comp, "core_window", lambda s, rule=None: (seen.append(frames_of(s)), next(cores))[1])
    prefixes = []
    original = comp.continue_from

    def recording(prior_, tokenizer_, decode, text, prefix, count, seed, **kw):
        prefixes.append((text, prefix.shape[1], count))
        return original(prior_, tokenizer_, decode, text, prefix, count, seed, **kw)
    monkeypatch.setattr(comp, "continue_from", recording)
    out = comp.compose_cores(prior, tokenizer, ["cut", "go", "now"], [.4, .42, .3], "core_chained", [1, 2, 3],
                             guidance=1.5, temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    # segments of 10, 10, 7 frames; codes kept up to each core's end; the last word keeps its return
    assert seen == [10, 10, 7]
    assert prefixes == [("cut", 0, 10), ("go", 7, 17), ("now", 17, 24)]
    rows = out["segments"]
    assert [(r["start"], r["lo"], r["hi"], r["kept"]) for r in rows] == [(0, 2, 6, 7), (7, 7, 16, 10),
                                                                          (17, 20, 21, 7)]
    assert out["codes"].shape[0] == 24
    parts = out["parts"]
    assert [frames_of(c) for c in parts["cores"]] == [5, 10, 2]
    assert parts["transitions"][0] is None and frames_of(parts["transitions"][1]) == 3      # frames 17..19
    assert parts["lead"][1] == 2 and parts["trail"][1] == 21
    # the chain only appends: the first segment's codes are the one-word chain's
    monkeypatch.undo()
    one = comp.compose_cores(prior, tokenizer, ["cut"], [.4], "core_chained", [1], guidance=1.5,
                             temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    assert torch.equal(out["codes"][:7], one["codes"][:7])


def test_core_chained_with_the_real_rule_renders():
    tokenizer, prior = tiny_prior()
    out = comp.compose_cores(prior, tokenizer, ["cut", "go"], [.6, .5], "core_chained", [1, 2], guidance=1.5,
                             temperature=TEMPERATURE, post=IDENTITY, device=CPU)
    assert all(r["reason"] in ("core", "no_sign_window", "arm_untracked", "short_window") for r in out["segments"])
    plan = cg.plan_frames([.25, .2], .1667, 1.5, "pad")
    assert frames_of(comp.render(out["parts"], plan)) == plan["frames"] == 36


# ---- the evaluation script: seeds, jobs, rule ------------------------------------------------------------------------

def test_word_seeds_are_unique_and_apart_from_stage_one():
    import scripts.round15_compositional as r15
    seeds = [s for i in range(300) for m in (False, True) for s in r2.word_seeds(i, 40, mismatched=m)]
    assert len(seeds) == len(set(seeds))
    stage1 = {s for i in range(300) for m in (False, True) for s in r15.word_seeds(i, 40, mismatched=m)}
    assert not stage1 & set(seeds)
    with pytest.raises(ValueError):
        r2.word_seeds(0, r2.MAX_WORDS + 1)


def test_plan_jobs_controls_and_seeds():
    plans = [dict(entries=["cut", "onion", "ix"], generate=[1., 1.2, .9], targets=[.2, .3, .17]),
             dict(entries=[], generate=[], targets=[]), dict(entries=["go"], generate=[1.1], targets=[.25])]
    orders = [[2, 0, 1], [], [0]]
    perm = [2, 0, 1]
    jobs = r2.plan_jobs(list(r2.VARIANTS), plans, perm, orders)
    j = jobs[("core_chained", "/shuffled", 0)]
    assert j["words"] == ["ix", "cut", "onion"] and j["targets"] == [.17, .2, .3]
    assert j["seeds"] == [r2.word_seeds(0, 3)[k] for k in (2, 0, 1)]
    assert jobs[("core_crossfade", "/shuffled", 0)] is None and jobs[("oracle_stage2", "/shuffled", 0)] is None
    assert jobs[("core_crossfade", "", 1)] is None                               # no sign: the blank clip
    m = jobs[("core_crossfade", "/mismatched_words", 0)]
    assert m["words"] == ["go"] and m["seeds"] == r2.word_seeds(0, 1, mismatched=True) and m["block"] == \
        "mismatched_words"
    assert jobs[("core_crossfade", "/mismatched_words", 2)] is None             # its source has no sign
    assert jobs[("oracle_stage2", "", 0)]["block"] == "matched"


def test_rule_parsing_and_verdict():
    assert r2.parse_rule(None) == [] and r2.parse_rule("") == []
    checks = r2.parse_rule("bleu:end_to_end, bleu:stage1_chained>0.1,spotting")
    assert checks == [dict(kind="bleu", pair="end_to_end", above=0.), dict(kind="bleu", pair="stage1_chained", above=.1),
                      dict(kind="spotting")]
    with pytest.raises(ValueError):
        r2.parse_rule("bleu:end to end")
    row = {"minus_end_to_end": dict(difference=.5, ci95=[.05, .9]),
           "minus_stage1_chained": dict(difference=.3, ci95=[.08, .5])}
    spot, mism = dict(mean=.3, ci95=[.25, .35]), dict(mean=.1, ci95=[.05, .15])
    v = r2.verdict(checks, row, spot, mism)
    assert v["verdict"] == "NOT PROMISING" and v["checks"]["bleu_vs_end_to_end_lower_above_0"]
    assert not v["checks"]["bleu_vs_stage1_chained_lower_above_0.1"] and v["checks"]["spotting_recall_above_mismatched"]
    assert r2.verdict(checks[:1] + checks[2:], row, spot, mism)["verdict"] == "PROMISING"
    assert r2.verdict([], row, spot, mism)["verdict"] == "UNDECLARED"
    with pytest.raises(ValueError):
        r2.verdict(r2.parse_rule("bleu:oracle_stage1"), row, spot, mism)


def test_form_pairs_compare_variants_with_every_comparator():
    pairs = r2.form_pairs(["core_chained", "oracle_stage2"], ["scale"])
    assert pairs["core_chained"]["stage1_chained"] == "stage1_chained"
    assert pairs["core_chained"]["oracle_stage1"] == "oracle_stage1"
    assert pairs["core_chained"]["variant_oracle_stage2"] == "oracle_stage2"
    assert pairs["oracle_stage2"]["mismatched_words"] == "oracle_stage2/mismatched_words"
    assert pairs["core_chained@scale"]["primary_fit"] == "core_chained"
    assert pairs["core_chained/shuffled"] == {"end_to_end": "end_to_end"}


# ---- end to end ----------------------------------------------------------------------------------------------------

from tests.test_round15_compositional import stubbed  # noqa: E402,F401  (the stub instruments fixture)
from tests.test_round15_oracle import TRAIN, source_report  # noqa: E402
from tests.test_sentence_exposure_diagnostic import build_store  # noqa: E402

ASLLRP_TRAIN = [("Cut the onion {k}", ["CUT", "ONION"]), ("Stir the salt in {k}", ["SALT", "STIR"]),
                ("He cuts the knife {k}", ["IX", "KNIFE", "CUT"]), ("Heat it with salt {k}", ["HEAT", "SALT"])]


def asllrp_inputs(store):
    """A tiny ASLLRP corpus (Sign Bank utterances, English pairs: train, val and a test pair that must never be read),
    an empty citizen map, a Sign Bank lexicon and an ASL-LEX table, at the fit script's keys."""
    items, pairs = [], []

    def add(source, k, english, glosses, partition):
        signs = [dict(gloss=g, variant=g, occurrence=g, sign_type="Lexical Signs", frames=[10 + 12 * j, 18 + 12 * j])
                 for j, g in enumerate(glosses)]
        items.append(dict(source=source, utterance=k, collection="asllrp-continuous", frames=[0, 20 + 12 * len(signs)],
                          signs=signs))
        pairs.append(dict(id=f"{source}-{k}", source=source, utterance=k, partition=partition, english=english,
                          glosses=glosses, surface=glosses, markers=[]))
    for k in range(6):
        for j, (english, glosses) in enumerate(ASLLRP_TRAIN):
            add("Cory_2013-6-25_sc01", 10 * k + j, english.format(k=k), glosses, "train")
    add("Jonathan_2013-6-25_sc02", 1, "Cut the salt", ["CUT", "SALT"], "val")
    pairs.append(dict(id="test-1", source="4-Ben-Deaf-Speech", utterance=1, partition="test"))   # unreadable
    store.put(fit.KEYS["utterances"], json.dumps(dict(items=items)).encode())
    store.put(fit.KEYS["pairs"], json.dumps(dict(pairs=pairs)).encode())
    store.put(fit.KEYS["citizen_map"], json.dumps(dict(mapping={})).encode())
    store.put(fit.KEYS["signbank_lexicon"], json.dumps(dict(lexicon={"KNIFE": dict(variants={"KNIFE": 3,
                                                                                            "BLADE": 1})})).encode())
    buffer = io.StringIO()
    w = csv.DictWriter(buffer, ["EntryID", "LemmaID", "Code", "DominantTranslation", "SignBankEnglishTranslations"])
    w.writeheader()
    w.writerow(dict(EntryID="stir", LemmaID="stir", Code="A_1", DominantTranslation="stir",
                    SignBankEnglishTranslations="mix, stir"))
    payload = buffer.getvalue().encode()
    store.put(fit.KEYS["asllex"], payload)
    return hashlib.sha256(payload).hexdigest()


def stage1_and_oracle_reports(store, sha):
    def tamper(r):
        n = r["clips"]
        r["forms"]["chained"] = dict(bleu=1., hypotheses=[f"chained cut {i}" for i in range(n)])
        for s in ("", "_mismatched"):
            r["spotting"]["scores"][f"chained{s}"] = dict(recall=dict(mean=.1, ci95=[.05, .15], n=n))
        r["motion_ratio_to_real"]["chained"] = dict(wrist_speed=3.)
    s1 = source_report(store, sha, tamper=tamper)
    oracle = dict(references=s1["references"],
                  forms=dict(oracle=dict(bleu=2., hypotheses=[f"oracle cut the {i}" for i in range(s1["clips"])])),
                  spotting=dict(scores={"oracle": dict(recall=dict(mean=.5, ci95=[.4, .6], n=5)),
                                        "oracle_mismatched": dict(recall=dict(mean=.1, ci95=[.05, .15], n=5))}),
                  motion_ratio_to_real=dict(oracle=dict(wrist_speed=2.)))
    store.put("oracle.json", json.dumps(oracle).encode())
    return s1, oracle


@pytest.fixture
def stage2_store(tmp_path, stubbed, monkeypatch):  # noqa: F811
    store, sha, _ = build_store(tmp_path / "store", corpus="asllvd-text-v1", train_texts=TRAIN)
    monkeypatch.setattr(fit, "ASLLEX_SHA256", asllrp_inputs(store))
    monkeypatch.setattr(fit, "annotated_seconds", lambda manifest, vocab: {e: .3 for e in vocab})
    s1, oracle = stage1_and_oracle_reports(store, sha)
    argv = ["--checkpoint-key", "prior.pt", "--stage1-report-key", "source.json", "--oracle-report-key",
            "oracle.json", "--store", f"dir:{tmp_path / 'store'}", "--device", "cpu", "--bootstrap", "30",
            "--cache-dir", str(tmp_path / "cache"), "--fit-min-count", "2", "--fit-move-min-count", "2"]
    return store, sha, s1, oracle, argv


def fitted(tmp_path, sha, key="mix.json"):
    return fit.fit_model(fit.agc._store(f"dir:{tmp_path / 'store'}"), key, sha,
                         parameters=dict(fit.DEFAULTS, min_count=2, move_min_count=2))


def test_stage2_end_to_end_on_a_tiny_prior(tmp_path, stage2_store, capsys):
    store, sha, s1, oracle, argv = stage2_store
    run = argv + ["--total-fit", "pad,scale", "--rule", "bleu:end_to_end,bleu:mismatched_words,spotting",
                  "--output-key", "reports/s2.json"]
    assert r2.main(run) == 0
    report = json.loads(store.get("reports/s2.json"))
    variants = list(r2.VARIANTS)
    expected = ["real", "blank"] + [f for v in variants for f in [f"{v}{c}" for c in r2.CONTROLS] + [f"{v}@scale"]]
    assert [f for f in report["forms"] if f not in r2.STORED] == expected
    for name, (which, source) in r2.STORED.items():                      # the stored hypotheses, scored here
        stored = (s1 if which == "stage1" else oracle)["forms"][source]["hypotheses"]
        assert report["forms"][name]["hypotheses"] == stored
    for v in variants:
        row = report["forms"][v]
        assert {"minus_end_to_end", "minus_stage1_chained", "minus_oracle_stage1", "minus_shuffled",
                "minus_mismatched_words"} <= set(row)
        d = report["verdicts"][v]
        assert set(d["checks"]) == {"bleu_vs_end_to_end_lower_above_0", "bleu_vs_mismatched_words_lower_above_0",
                                    "spotting_recall_above_mismatched"}
        assert d["verdict"] == ("PROMISING" if all(d["checks"].values()) else "NOT PROMISING")
        assert {v, f"{v}_mismatched"} <= set(report["spotting"]["scores"])
        assert report["motion"][v]["seconds_over_reference"] is not None
    # the plans: the stage-2 model fitted on the ASLLRP train utterances only
    model, _ = fitted(tmp_path, sha)
    assert report["model_sha256"] == model["sha256"] and model["train_utterances"] == 24
    vocab = comp.training_vocabulary(json.loads(store.get("mix.json")))
    for i, row in enumerate(report["per_clip"]):
        plan = r2.plan_clip(row["text"], model, vocab)
        assert row["entries"] == plan["entries"] and row["targets"] == [round(t, 4) for t in plan["targets"]]
        assert row["stage1_words"] == s1["per_clip"][i]["words"]
        for v in variants:
            if not row["entries"]:
                assert v not in row
                continue
            p = row[v]["plan"]
            assert p["frames"] == row[v]["frames"] == p["lead"] + sum(p["cores"]) + sum(p["gaps"]) + p["trail"]
            if p["fit"] == "padded":
                assert p["frames"] == row["frames_reference"]
            assert [s["word"] for s in row[v]["segments"]] == row["entries"]
    assert report["timing"]["core_chained"]["core_compression_median"] > 0
    assert report["selection"]["operations"]
    printed = capsys.readouterr().out
    assert "VERDICT core_chained:" in printed and "wrist speed / real" in printed
    with pytest.raises(SystemExit, match="refusing to overwrite"):           # write-once
        r2.main(run)
    # a variant subset reproduces the same compositions; the model can be pinned by its hash
    sub = argv + ["--variants", "core_chained", "--expect-model-sha256", model["sha256"], "--output-key",
                  "reports/sub.json"]
    assert r2.main(sub) == 0
    again = json.loads(store.get("reports/sub.json"))
    for name in ("core_chained", "core_chained/shuffled", "core_chained/mismatched_words"):
        assert again["forms"][name]["hypotheses"] == report["forms"][name]["hypotheses"]
    assert again["verdicts"]["core_chained"]["verdict"] == "UNDECLARED"
    with pytest.raises(SystemExit, match="not the declared"):
        r2.main(argv + ["--expect-model-sha256", "0" * 64, "--output-file", str(tmp_path / "x.json")])


def test_stage2_reads_a_model_file_and_refuses_a_tampered_one(tmp_path, stage2_store):
    store, sha, s1, oracle, argv = stage2_store
    model, _ = fitted(tmp_path, sha)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    assert argv[-4:] == ["--fit-min-count", "2", "--fit-move-min-count", "2"]
    with pytest.raises(SystemExit):                                     # --fit-* flags with a model file: refused
        r2.main(argv + ["--variants", "oracle_stage2", "--model-file", str(path), "--output-file",
                        str(tmp_path / "f.json")])
    oracle_only = [a for a in argv[:-4] if a not in ("--checkpoint-key", "prior.pt")]
    other = dict(model, corpora=["asllvd-text-v1"])                     # a model of other corpora: refused
    other["sha256"] = cg.model_sha256({k: v for k, v in other.items() if k != "sha256"})
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="corpora"):
        r2.main(oracle_only + ["--variants", "oracle_stage2", "--model-file", str(tmp_path / "other.json"),
                               "--output-file", str(tmp_path / "c.json")])
    assert r2.main(oracle_only + ["--variants", "oracle_stage2", "--model-file", str(path), "--output-key",
                                  "reports/o.json"]) == 0
    report = json.loads(store.get("reports/o.json"))
    assert report["model_sha256"] == model["sha256"] and report["checkpoint_key"] is None
    clips = [s for row in report["per_clip"] if "oracle_stage2" in row for s in row["oracle_stage2"]["segments"]]
    assert clips and all(s["corpus"] == "asllvd-text-v1" for s in clips)
    model["timing"]["gap"] = 9.
    path.write_text(json.dumps(model))
    with pytest.raises(SystemExit, match="does not match its sha256"):
        r2.main(oracle_only + ["--variants", "oracle_stage2", "--model-file", str(path), "--output-file",
                               str(tmp_path / "y.json")])
    with pytest.raises(SystemExit):
        r2.main(oracle_only + ["--output-file", str(tmp_path / "z.json")])     # generated variants need a prior


def test_the_fit_never_reads_how2sign_or_asllrp_test(tmp_path, stage2_store):
    store, sha, s1, oracle, argv = stage2_store
    model, prep = fitted(tmp_path, sha)
    assert {r["partition"] for r in prep["records"]} == {"train", "val"}      # the test pair was never read
    assert all(r["collection"] == "asllrp-continuous" for r in prep["train"])
    # changing every How2Sign sentence of the mix changes neither the vocabulary nor the model
    manifest = json.loads(store.get("mix.json"))
    for r in manifest["records"]:
        if r["corpus"] == "how2sign-two-view-text-v1":
            r["text"] = "cut cut salt heat " + r["text"]
    from signtranslator.data_engineering.corpus_manifest import canonical_json
    payload = canonical_json(manifest)
    store.put("mix2.json", payload)
    model2, _ = fitted(tmp_path, hashlib.sha256(payload).hexdigest(), "mix2.json")
    strip = lambda m: {k: v for k, v in m.items() if k not in ("inputs", "mix_key", "sha256")}   # noqa: E731
    assert strip(model2) == strip(model)
    # and the fit functions refuse a How2Sign row or an ASLLRP val row outright
    h2s = dict(prep["train"][0], collection="how2sign-two-view-text-v1")
    with pytest.raises(ValueError):
        fit.fit_stage2(prep["train"] + [h2s], prep["vocab"], prep["lexicon"], prep["annotated"])
    with pytest.raises(ValueError):
        fit.fit_stage2(prep["train"] + prep["val"], prep["vocab"], prep["lexicon"], prep["annotated"])


def test_fit_script_end_to_end(tmp_path, stage2_store):
    store, sha, s1, oracle, argv = stage2_store
    out = ["--store", f"dir:{tmp_path / 'store'}", "--mix-key", "mix.json", "--mix-sha256", sha, "--min-count", "2",
           "--move-min-count", "2", "--bootstrap", "20", "--model-file", str(tmp_path / "m.json"), "--output-file",
           str(tmp_path / "r.json")]
    assert fit.main(out) == 0
    model, report = json.loads((tmp_path / "m.json").read_text()), json.loads((tmp_path / "r.json").read_text())
    fit.check_model(model)
    assert report["model_sha256"] == model["sha256"] and report["utterances"] == dict(train=24, val=1, val_heldout=1)
    assert set(report["val_heldout_sentences"]) >= {"stage1", "stage2_full"}
    assert model["lexicon"] == {"blade": dict(entry="knife", tier="signbank_variant", votes=1),
                                "mix": dict(entry="stir", tier="asllex_translation", votes=1)}
    assert set(report["val"]) >= {"stage1", "stage2_selection", "stage2_tokens", "stage2_full", "operations"}
    assert report["reproduction_groundwork_rule_check"]["val"]["bag_precision"] == report["val"]["stage1"][
        "bag_precision"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        fit.main(out)
    bad = out[:5] + ["0" * 64] + out[6:-4] + ["--model-file", str(tmp_path / "m2.json"), "--output-file",
                                              str(tmp_path / "r2.json")]
    with pytest.raises(SystemExit, match="mix-sha256"):
        fit.main(bad)


def test_timing_check_on_asllrp_val_utterances(tmp_path, stage2_store, capsys):
    import scripts.round15_stage2_timing_check as tc
    from signtranslator.data_engineering.corpus_manifest import canonical_json
    store, sha, s1, oracle, argv = stage2_store
    manifest = json.loads(store.get("mix.json"))
    h2s = [r for r in manifest["records"] if r["corpus"] == "how2sign-two-view-text-v1" and r["partition"] == "val"]
    for k, (r, english) in enumerate(zip(h2s, ["Cut the onion", "Stir the salt", "Heat the knife"])):
        manifest["records"].append(dict(r, corpus=tc.ASLLRP_CORPUS, sign=f"asllrp-{k}", english=english,
                                        text=english))
    payload = canonical_json(manifest)
    store.put("mix3.json", payload)
    sha3 = hashlib.sha256(payload).hexdigest()
    model, _ = fitted(tmp_path, sha3, "mix3.json")
    (tmp_path / "m3.json").write_text(json.dumps(model))
    out = tmp_path / "timing.json"
    assert tc.main(["--model-file", str(tmp_path / "m3.json"), "--mix-key", "mix3.json", "--mix-sha256", sha3,
                    "--store", f"dir:{tmp_path / 'store'}", "--cache-dir", str(tmp_path / "cache"),
                    "--output-file", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["utterances"] == 3 and set(report["motion"]) == {"real", "stage1_oracle", "stage2_oracle@pad",
                                                                   "stage2_oracle@scale", "stage2_oracle@none",
                                                                   "stage2_words@stage1_timing",
                                                                   "stage2_oracle@sign_fitted"}
    assert report["motion"]["stage2_oracle@sign_fitted"]["seconds_over_reference"] == 1.
    assert report["sign_fitted"]["margin"] == .1 and report["sign_fitted"]["reasons"]
    assert report["motion_ratio_to_real"]["real"]["wrist_speed"] == 1.
    assert report["motion"]["stage2_oracle@pad"]["seconds_over_reference"] == 1.
    assert report["motion"]["stage2_oracle@none"]["seconds_over_reference"] < 1.
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        tc.main(["--model-file", str(tmp_path / "m3.json"), "--mix-key", "mix3.json", "--mix-sha256", sha3,
                 "--output-file", str(out)])
    with pytest.raises(SystemExit, match="another vocabulary"):
        tc.main(["--model-file", str(tmp_path / "m3.json"), "--mix-key", "mix.json", "--mix-sha256", sha,
                 "--store", f"dir:{tmp_path / 'store'}", "--output-file", str(tmp_path / "t2.json")])


def test_build_lexicon_tiers_conflicts_and_own_signs():
    vocab = {"knife": 1., "stir": 1., "child": 1., "school": 1., "principal": 1., "thumbtack": 1.}
    signbank = {"CHILD": collections.Counter({"child": 3})}
    lex = {"lexicon": {"CHILD": dict(variants={"CHILDREN": 2, "CHILD": 5}), "KNIFE": dict(variants={"BLADE": 1}),
                       "SCHOOL": dict(variants={"SCHOOL": 1}), "THUMBTACK/BUTTON": dict(variants={})}}
    rows = [dict(EntryID="principal", LemmaID="principal", Code="A", DominantTranslation="",
                 SignBankEnglishTranslations="principal, school, headmaster, children"),
            dict(EntryID="kid", LemmaID="kid", Code="B", DominantTranslation="kid",
                 SignBankEnglishTranslations="kid"),
            dict(EntryID="mix_2", LemmaID="mix", Code="C", DominantTranslation="Stir", SignBankEnglishTranslations="")]
    out = fit.build_lexicon(vocab, lex, signbank, rows, {}, collections.Counter({"child": 3, "principal": 1}))
    assert out["children"] == dict(entry="child", tier="signbank_variant", votes=1)   # the variant tier first
    assert out["blade"]["entry"] == "knife"
    assert out["button"] == dict(entry="thumbtack", tier="signbank_alternative", votes=1)
    assert out["headmaster"]["entry"] == "principal"
    assert "school" not in out                         # in the vocabulary (and a sign of its own besides)
    assert "kid" not in out                            # its ASL-LEX sign is not in the vocabulary
    assert "stir" not in out and "mix" not in out      # capitals are glosses, not translations
    # a key with a negation never maps to an entry without one; keys are written split; contraction entries'
    # split forms are the vocabulary's, never lexicon keys
    rows2 = [dict(EntryID="like", LemmaID="like", Code="D", DominantTranslation="",
                  SignBankEnglishTranslations="not like, enjoy"),
             dict(EntryID="refuse", LemmaID="refuse", Code="E", DominantTranslation="",
                  SignBankEnglishTranslations="won't, decline"),
             dict(EntryID="can't", LemmaID="can't", Code="F", DominantTranslation="", SignBankEnglishTranslations="")]
    out2 = fit.build_lexicon({"like": 1., "refuse": 1., "can't": 1., "cannot": 1.}, {"lexicon": {}}, {}, rows2, {},
                             collections.Counter())
    assert "not like" not in out2 and out2["enjoy"]["entry"] == "like"
    assert "won't" not in out2 and "will not" not in out2                  # "will not" -> refuse loses the negation
    assert "can not" not in out2


# ---- the independent review's adversarial cases (2026-10-28, commit 45cd2d9) -------------------------------------------

def test_val_occurrence_does_not_change_train_gold():
    vocab = {"me": 1., "he": 1.}

    def rec(part, occ):
        return dict(id=part, collection="asllrp-continuous", partition=part, english="x", seconds=1.,
                    signs=[dict(gloss="IX", occurrence=occ, sign_type="", category="index", start=0., end=.2)])
    train = rec("train", "IX-3p")
    alone = fit.fit_rows([train], fit.gloss_table([train], vocab, {}, {}, {}))[0]["gold"]
    with_val = fit.fit_rows([train], fit.gloss_table([rec("val", "IX-1p"), train], vocab, {}, {}, {}))[0]["gold"]
    assert alone == with_val == ["he"]
    # the groundwork's per-type table is the one that depended on the first occurrence (kept for its reproduction)
    assert fit.groundwork_table([rec("val", "IX-1p"), train], vocab, {}, {}, {})["IX"]["entries"] == ("me",)


def test_lexicon_cannot_drop_negation():
    vocab = {"not": 1., "like": 1.}
    lex = {"not like": dict(entry="like", tier="asllex_translation")}
    got = cg.selected(cg.selection_rows("I do not like it", vocab, lex))
    assert got == ["not", "like"]
    assert cg._decide("not like", 10, {"like": 10}, "like", .5)["entry"] != "like"
    assert not cg.lexicon_allowed("nobody came", "come") and cg.lexicon_allowed("not like", "dislike not")


def test_contraction_entries_stay_reachable():
    vocab = {"can't": 1., "go": 1., "don't know": 1., "can": 1., "not": 1.}
    s1 = comp.select_words("I can't go, I don't know", vocab)
    s2 = cg.selected(cg.selection_rows("I can't go, I don't know", vocab))
    assert s1 == ["can't", "go", "don't know"] and s2 == s1
    assert cg.vocabulary_keys(vocab)["can not"] == "can't" and cg.vocabulary_keys(vocab)["do not know"] == "don't know"
    # the entry stays negated and can move: a negation entry reached through its split form
    model = dict(moves={"can't": "end"})
    assert cg.apply_structure(cg.selection_rows("I can't go", vocab), model, vocab)[0] == ["go", "can't"]


@pytest.mark.parametrize("text,literal,contracted,entry", [
    ("You are happy", ["happy"], "You're happy", "you're"),         # "you" stays a pronoun (it points), no you're
    ("I am happy", ["happy"], "I'm happy", "i'm"),
    ("I will not go", ["not", "go"], "I won't go", "won't"),
    ("Do not go", ["not", "go"], "Don't go", "don't"),
])
def test_split_forms_match_only_contracted_text(text, literal, contracted, entry):
    vocab = {"you're": 1., "i'm": 1., "won't": 1., "don't": 1., "happy": 1., "go": 1., "not": 1.}
    assert cg.selected(cg.selection_rows(text, vocab)) == literal          # as stage 1 on the split tokens
    assert comp.select_words(text, vocab) == literal
    assert entry in cg.selected(cg.selection_rows(contracted, vocab))
    assert cg.selected(cg.selection_rows(contracted, vocab)) == comp.select_words(contracted, vocab)
    assert cg.marked_tokens(contracted)[:2] != [(t, False) for t in cg.tokens(contracted)[:2]]


def test_vocabulary_index_follows_in_place_changes():
    vocab = {"can't": 1., "go": 1.}
    assert cg.selected(cg.selection_rows("I can't go", vocab)) == ["can't", "go"]
    del vocab["can't"]
    vocab["don't"] = 1.
    assert cg.selected(cg.selection_rows("I can't go", vocab)) == ["go"]
    assert cg.selected(cg.selection_rows("I don't go", vocab)) == ["don't", "go"]
    assert cg.sentence_key("I don't know") == cg.sentence_key("I do not know") == "i do not know"


@pytest.mark.parametrize("fit_", ["pad", "scale"])
def test_plan_equals_reference_even_with_many_signs(fit_):
    p = cg.plan_frames([.3] * 30, .1, .5, fit_)
    assert p["frames"] == 12 and p["overlong"] and p["truncated_signs"] == 18 and p["fit"] == "truncated"
    assert p["cores"] == [1] * 12 + [0] * 18 and set(p["gaps"]) == {0}
    clips = [full_state(k, frames=6) for k in range(30)]
    state = comp.render(comp.parts_from_pieces([dict(state=c, lo=1, hi=4) for c in clips]), p)
    assert len(state.timestamps) == 12
    assert not cg.plan_frames([.3] * 3, .1, 2., fit_)["overlong"]


def test_plan_equals_reference_random():
    rng = random.Random(0)
    for _ in range(3000):
        n = rng.randint(1, 12)
        t = [rng.uniform(.01, 2.) for _ in range(n)]
        R = rng.uniform(n / 24., 15.)
        for f in ("pad", "scale"):
            p = cg.plan_frames(t, rng.uniform(0, .5), R, f, lead_share=rng.random())
            assert p["frames"] == max(1, round(R * 24)) and min(p["cores"]) >= 1 and min(p["gaps"] or [0]) >= 0


def test_f1_zero_not_none():
    agg, _ = cg.sequence_metrics([["x"]], [["a"]])
    assert agg["bag_f1"] == 0.


def test_target_seconds_without_trimmed_ratio():
    t = dict(sentence_seconds={}, annotated={}, ratio=.5, trimmed_over_annotated=None, floor=.1, ceiling=1.)
    assert cg.target_seconds("rare", t, {"rare": 1.}) == (.1, "default")
    assert cg.target_seconds("rare", dict(t, default_seconds=.25), {"rare": 1.}) == (.25, "default")


def test_fit_timing_without_seconds():
    import warnings
    rows = [utt("x", [], k, seconds=None, timed=[("cut", "lexical", .2, .4), ("go", "lexical", .5, .8)])
            for k in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = cg.fit_timing(rows, {"cut": 1., "go": 1.}, {"cut": .5, "go": .5})
    assert t["lead_share"] == .5 and t["lead_seconds"] is None and np.isfinite(t["gap"])
    single = [utt("x", [], k, seconds=1., timed=[("cut", "lexical", .2, .4)]) for k in range(3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cg.fit_timing(single, {"cut": 1.}, {"cut": .5})["gap"] == 0.


def test_moves_are_a_permutation():
    rng = random.Random(1)
    words = ["a", "b", "what", "not", "ix", "finish"]
    for _ in range(2000):
        s = [rng.choice(words) for _ in range(rng.randint(0, 8))]
        m = {w: rng.choice(cg.MOVES) for w in words}
        assert sorted(cg.apply_moves(s, m)) == sorted(s)


def test_tau_repeated_ix():
    for seed in range(500):
        rng = random.Random(seed)
        g = [rng.choice(["ix", "go", "cut", "ix"]) for _ in range(rng.randint(2, 7))]
        p = g[:]
        rng.shuffle(p)
        t = cg.tau_of(p, g)
        assert t is None or -1 <= t <= 1
    assert cg.tau_of(["ix", "ix"], ["ix", "ix"]) == 1.


def test_allocate_property():
    rng = random.Random(2)
    for _ in range(3000):
        n = rng.randint(1, 10)
        L = [rng.uniform(0, 5) for _ in range(n)]
        mins = [rng.randint(0, 2) for _ in range(n)]
        total = rng.randint(sum(mins), sum(mins) + 60)
        out = cg.allocate(L, total, mins)
        assert sum(out) == total and all(o >= m for o, m in zip(out, mins))


def test_render_edge_lengths():
    clips = [full_state(k, frames=1 + k) for k in range(3)]
    pieces = [dict(state=c, lo=0, hi=len(c.timestamps) - 1) for c in clips]
    for R in (.05, .2, 1., 3.):
        for f in ("pad", "scale", "none"):
            plan = cg.plan_frames([.01, .5, .04], .04, R, f)
            parts = comp.parts_from_pieces(pieces)
            parts["transitions"] = [full_state(9, frames=1), None]
            assert len(comp.render(parts, plan).timestamps) == plan["frames"]


def test_wh_time_words_and_quantifiers_are_not_droppable():
    for w in ("when", "before", "after", "then", "will", "all", "every", "each", "some", "any", "what", "now"):
        assert not cg.droppable(w), w
    assert not set(cg.DROPPABLE) & {w for ws in cg.MOVE_CLASSES.values() for w in ws}
    assert {"nobody", "nowhere", "neither", "nor", "without"} <= cg.NEGATIONS


def test_first_person_without_me():
    vocab = {"ix": 1., "go": 1.}
    rows = [utt(f"I go {k}", ["ix", "go"], k) for k in range(10)]
    m = cg.fit_structure(rows, vocab, min_count=5)
    assert cg.apply_structure(cg.selection_rows("I go", vocab), m, vocab)[0] == ["go"]


def test_empty_inputs():
    vocab = {"go": 1.}
    assert cg.selection_rows("", vocab) == []
    assert cg.apply_structure(cg.selection_rows("the a of", vocab), {}, vocab)[0] == []


def test_token_decisions_count_distinct_sentences():
    vocab = {"with": 1., "cut": 1.}
    same = [utt("cut with care", ["cut"], k) for k in range(9)]                 # one sentence, nine signers
    assert "with" not in cg.fit_structure(same, vocab, min_count=5)["tokens"]
    distinct = [utt(f"cut with care {k}", ["cut"], k) for k in range(5)]
    t = cg.fit_structure(distinct, vocab, min_count=5)["tokens"]
    assert t["with"]["op"] == "drop" and t["with"]["sentences"] == 5


def test_rule_comparisons_are_checked_before_any_work(tmp_path, stage2_store):
    r2.check_rule(r2.parse_rule("bleu:end_to_end,bleu:mismatched,bleu:blank,spotting"), list(r2.VARIANTS))
    with pytest.raises(ValueError, match="not compared with"):
        r2.check_rule(r2.parse_rule("bleu:primary_fit"), ["core_chained"])        # only the extra fits have it
    r2.check_rule(r2.parse_rule("bleu:stage1_chained"), ["core_chained"], ["scale"])
    store, sha, s1, oracle, argv = stage2_store
    calls = []
    import evaluate_text_to_motion as etm
    original = etm.load_generator
    etm.load_generator = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        with pytest.raises(SystemExit):
            r2.main(argv + ["--rule", "bleu:oracle", "--output-file", str(tmp_path / "x.json")])
    finally:
        etm.load_generator = original
    assert not calls and not (tmp_path / "x.json").exists()


# ---- timing modes (2026-10-29: stage1 and sign-fitted beside the declared asllrp) ------------------------------------

TIMING = dict(annotated={"cut": .5}, trimmed_over_annotated=2.5, default_seconds=.25, floor=.17, gap=.2,
              sentence_seconds={}, ratio=.4, ceiling=.47, lead_share=.4)


def test_isolated_sign_seconds_and_gap_ratio():
    typical = {"cut": 2., "go": 1.5}
    assert cg.isolated_sign_seconds("cut", TIMING, typical) == (.5, "annotated")
    assert cg.isolated_sign_seconds("go", TIMING, typical) == (pytest.approx(.6), "trimmed")
    assert cg.isolated_sign_seconds("go", dict(TIMING, trimmed_over_annotated=None), typical) == (.25, "default")
    assert cg.gap_ratio(TIMING) == pytest.approx(.8)


def test_fitted_plan_fills_the_reference_in_proportion():
    p = cg.fitted_plan([.5, .3, .2], .8, 2.)                                  # R = 48 frames
    # weights .5 .3 .2 and transitions .8 (.5+.3)/2 = .32, .8 (.3+.2)/2 = .2: W = 1.52
    W = 1. + .32 + .2
    assert p["frames"] == 48 and p["lead"] == p["trail"] == 0 and p["fit"] == "fitted" and not p["overlong"]
    assert sum(p["cores"]) + sum(p["gaps"]) == 48
    for got, w in zip(p["cores"] + p["gaps"], [.5, .3, .2, .32, .2]):
        assert abs(got - 48 * w / W) < 1
    assert cg.fitted_plan([.5], .8, 1.)["cores"] == [24] and cg.fitted_plan([.5], .8, 1.)["gaps"] == []
    t = cg.fitted_plan([.3] * 10, .7, .25)                                     # 6 frames for 10 signs
    assert t["overlong"] and t["frames"] == 6 and t["cores"] == [1] * 6 + [0] * 4 and t["truncated_signs"] == 4
    rng = random.Random(3)
    for _ in range(2000):
        n = rng.randint(1, 10)
        R = rng.uniform(.05, 12.)
        q = cg.fitted_plan([rng.uniform(.05, 1.) for _ in range(n)], rng.uniform(0, 1.5), R)
        assert q["frames"] == max(1, round(R * 24))
    with pytest.raises(ValueError):
        cg.fitted_plan([], .7, 1.)


def test_fraction_window_frames():
    s = full_state(0, frames=48)                                              # D = 2 s
    w = comp.fraction_window(s, (.25, .75), 0.)
    assert (w["lo"], w["hi"]) == (12, 36)
    w = comp.fraction_window(s, (.25, .75), .1)                               # 0.1 s = 2.4 frames either side
    assert (w["lo"], w["hi"]) == (10, 38)
    w = comp.fraction_window(s, (0., 1.), .5)
    assert (w["lo"], w["hi"]) == (0, 47) and w["whole"]
    w = comp.fraction_window(s, (.99, .995), 0.)
    assert w["lo"] <= w["hi"] <= 47


def test_record_sign_fraction_and_sign_fractions():
    def rec(sign, text, frames, window, corpus="asllvd-text-v1", partition="train"):
        return dict(sign=sign, text=text, corpus=corpus, partition=partition, source=dict(frames=frames),
                    continuous=dict(window_seconds=window, seconds=window[1] - window[0]))
    # a 0.5 s sign (30 frames at 60 fps) starting 0.833 s into the clip; window [0.5, 1.75]
    r = rec("a", "cut", [100, 129], [.5, 1.75])
    a, b = fit.record_sign_fraction(r)
    assert a == pytest.approx((50 / 60 - .5) / 1.25) and b == pytest.approx((50 / 60 - .5 + .5) / 1.25)
    assert fit.record_sign_fraction(dict(r, corpus="rit-text-r6")) is None
    assert fit.record_sign_fraction(dict(r, source={})) is None
    assert fit.record_sign_fraction(rec("x", "cut", [100, 129], [1.4, 1.6])) is None     # sign before the window
    manifest = dict(records=[r, rec("b", "cut", [100, 159], [.5, 2.]), rec("c", "go", [0, 29], [.6, 1.6]),
                             rec("d", "cut", [0, 29], [.5, 1.75], partition="val")])
    f = fit.sign_fractions(manifest, {"cut": 1., "go": 1.})
    assert set(f["records"]) == {"a", "b", "c"} and f["clips"] == 3             # train only
    assert f["entries"]["cut"][0] == pytest.approx(np.median([(50 / 60 - .5) / 1.25, (50 / 60 - .5) / 1.5]), abs=1e-4)
    assert set(f["entries"]) == {"cut", "go"} and len(f["all"]) == 2


def test_stage1_timing_jobs_and_the_stage1_oracle_composition():
    import scripts.round15_compositional as r15
    import scripts.round15_oracle as ro
    vocab = {"cut": 1.2, "onion": .9, "go": 1.5}
    plans = [dict(entries=["cut", "onion", "go"], generate=[1.2, .9, 1.5], targets=[.2, .3, .25], weights=[.5, .4, .6]),
             dict(entries=["go"], generate=[1.5], targets=[.25], weights=[.6])]
    jobs = r2.plan_jobs(list(r2.VARIANTS), plans, [1, 0], [[2, 0, 1], [0]], seconds=[3., 2.], vocab=vocab,
                        timing="stage1", overlap=4)
    for v in r2.VARIANTS:
        mode = "crossfade" if v != "core_chained" else "chained"
        assert jobs[(v, "", 0)]["durations"] == r15.plan_durations(mode, ["cut", "onion", "go"], vocab, 3., 4)
        assert jobs[(v, "/mismatched_words", 0)]["durations"] == r15.plan_durations(mode, ["go"], vocab, 3., 4)
    d = jobs[("core_chained", "", 0)]["durations"]
    assert jobs[("core_chained", "/shuffled", 0)]["durations"] == [d[2], d[0], d[1]]
    # the oracle in stage1 timing is the stage-1 oracle's composition of the whole clips, in any order
    clips = [full_state(k, frames=14 + k) for k in range(3)]
    material = dict(pieces=[dict(state=c, lo=0, hi=len(c.timestamps) - 1) for c in clips], segments=[])
    job = jobs[("oracle_stage2", "", 0)]
    frames = [comp.word_frames(x) for x in job["durations"]]
    for order in (None, [2, 0, 1]):
        state, plan = r2.render_form(material, job, 3., {"timing": TIMING}, timing="stage1", order=order)
        idx = order or [0, 1, 2]
        expected = ro.compose_clips([clips[k] for k in idx], [frames[k] for k in idx], 4, fitted=True)
        np.testing.assert_allclose(state.channels["root_translation"].values,
                                   expected.channels["root_translation"].values)
        assert plan["fit"] == "stage1" and plan["frames"] == len(state.timestamps)
    # generated crossfade material: the clips concatenated with the crossfade
    gen = dict(clips=clips, state=None, segments=[])
    state, _ = r2.render_form(gen, job, 3., {"timing": TIMING}, timing="stage1", order=[1, 2, 0])
    expected = comp.concatenate([clips[1], clips[2], clips[0]], 4)
    np.testing.assert_allclose(state.channels["root_translation"].values, expected.channels["root_translation"].values)


def test_sign_fitted_render_fills_the_reference():
    clips = [full_state(k, frames=24) for k in range(3)]
    fractions = dict(entries={"cut": (.25, .75)}, all=(.3, .6))
    pieces = [dict(state=c, **r2.sign_core(w, c, fractions=fractions, margin=.1)) for w, c in
              zip(["cut", "go", "cut"], clips)]
    assert [p["reason"] for p in pieces] == ["entry", "all", "entry"]
    assert (pieces[0]["lo"], pieces[0]["hi"]) == (4, 20) and (pieces[1]["lo"], pieces[1]["hi"]) == (5, 16)
    material = dict(pieces=pieces, parts=comp.parts_from_pieces(pieces), segments=[])
    job = dict(words=["cut", "go", "cut"], weights=[.5, .3, .5], targets=[.2, .2, .2], durations=[None] * 3)
    for order in (None, [2, 1, 0]):
        state, plan = r2.render_form(material, job, 2.5, {"timing": TIMING}, timing="sign-fitted", order=order)
        assert len(state.timestamps) == plan["frames"] == 60 and plan["lead"] == 0


def test_generated_material_in_each_timing():
    tokenizer, prior = tiny_prior(locus_conditioning=True, history_stages=1)

    class Sample:
        pass
    sample = Sample()
    sample.prior, sample.tokenizer, sample.decoder_gains = prior, tokenizer, None
    import exposure_diagnostic
    real_post = exposure_diagnostic.make_post
    exposure_diagnostic.make_post = lambda gains: IDENTITY
    try:
        job = dict(mode="core_crossfade", words=["cut", "go"], generate=[.5, .4], seeds=[3, 4], durations=[.3, .35],
                   overlap=4)
        m1 = r2.generate_job(sample, job, guidance=1.5, temperature=TEMPERATURE, rule=None, timing="stage1")
        state, info = comp.compose(prior, tokenizer, ["cut", "go"], [.3, .35], "crossfade", [3, 4], guidance=1.5,
                                   temperature=TEMPERATURE, post=IDENTITY, crossfade=4)
        np.testing.assert_allclose(m1["state"].channels["root_translation"].values,
                                   state.channels["root_translation"].values)              # stage 1's V1 exactly
        fractions = dict(entries={"cut": (.25, .75)}, all=(.3, .6))
        for mode in ("core_crossfade", "core_chained"):
            m2 = r2.generate_job(sample, dict(job, mode=mode), guidance=1.5, temperature=TEMPERATURE, rule=None,
                                 timing="sign-fitted", fractions=fractions, margin=.1)
            assert all(s["reason"] in ("entry", "all") for s in m2["segments"])
            state2, plan = r2.render_form(m2, dict(job, weights=[.5, .3], targets=[.2, .2]), 1.5,
                                          {"timing": TIMING}, timing="sign-fitted")
            assert len(state2.timestamps) == plan["frames"] == 36
        m3 = r2.generate_job(sample, dict(job, mode="core_chained"), guidance=1.5, temperature=TEMPERATURE,
                             rule=None, timing="stage1")
        assert m3["clips"] is None and len(m3["state"].timestamps) == sum(comp.word_frames(d) for d in [.3, .35])
    finally:
        exposure_diagnostic.make_post = real_post


@pytest.mark.parametrize("timing", ["stage1", "sign-fitted"])
def test_oracle_runs_in_each_new_timing(tmp_path, stage2_store, timing):
    store, sha, s1, oracle, argv = stage2_store
    oracle_only = [a for a in argv if a not in ("--checkpoint-key", "prior.pt")]
    assert r2.main(oracle_only + ["--variants", "oracle_stage2", "--timing", timing, "--output-key",
                                  f"reports/{timing}.json"]) == 0
    report = json.loads(store.get(f"reports/{timing}.json"))
    assert report["timing_mode"] == timing and list(report["forms"])[2:5] == [
        "oracle_stage2", "oracle_stage2/shuffled", "oracle_stage2/mismatched_words"]
    for row in report["per_clip"]:
        if "oracle_stage2" not in row:
            continue
        plan = row["oracle_stage2"]["plan"]
        if timing == "sign-fitted":
            assert plan["frames"] == row["frames_reference"] and plan["lead"] == plan["trail"] == 0
            assert report["core_margin"] == .1
        else:
            assert plan["fit"] == "stage1" and len(plan["durations"]) == len(row["entries"])
    assert report["motion"]["oracle_stage2"]["wrist_speed"] is not None


def test_timing_flags_are_checked(tmp_path, stage2_store):
    store, sha, s1, oracle, argv = stage2_store
    base = [a for a in argv if a not in ("--checkpoint-key", "prior.pt")] + ["--variants", "oracle_stage2",
                                                                               "--output-file", str(tmp_path / "o.json")]
    for extra in (["--timing", "stage2"], ["--timing", "stage1", "--total-fit", "pad,scale"],
                  ["--timing", "sign-fitted", "--total-fit", "scale"], ["--timing", "stage1", "--core-margin", ".1"]):
        with pytest.raises(SystemExit):
            r2.main(base + extra)
    assert not (tmp_path / "o.json").exists()
