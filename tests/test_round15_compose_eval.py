"""Round 15 larger paired evaluation: shards (deterministic, translation by global block), the on-the-fly stage-1
selection and its check against a stored report, the merge (complete, disjoint, one run), sharded == unsharded, the
end-to-end comparator from a pilot report, --paired-with, and the pilot's 300 / 1000 samples."""
import json
import random

import pytest

import scripts.round15_compose_eval as ce
from scripts.pilot_content_check import validation_sentences
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures and helpers)
from tests.test_round15_compositional import stubbed  # noqa: F401


def test_shards_are_blocks_dealt_round_robin():
    assert ce.shard_clips(10, 0, 2, 2) == [0, 1, 4, 5, 8, 9] and ce.shard_clips(10, 1, 2, 2) == [2, 3, 6, 7]
    assert sorted(ce.shard_clips(1000, 0, 4) + ce.shard_clips(1000, 1, 4) + ce.shard_clips(1000, 2, 4)
                  + ce.shard_clips(1000, 3, 4)) == list(range(1000))
    assert [len(ce.shard_clips(1000, s, 4)) for s in range(4)] == [256, 256, 256, 232]
    assert ce.blocks([0, 1, 4, 5, 8], 2) == [[0, 1], [4, 5], [8]]
    with pytest.raises(ValueError):
        ce.shard_clips(10, 2, 2)


def test_the_pilot_samples_300_and_1000_by_different_methods():
    """random.sample draws k = 300 from 3,096 with its set method and k = 1000 with its pool method: the 300 are not
    a prefix of the 1000; 289 of them are among the 1000 (the indices depend only on n, k and the seed)."""
    pool = [dict(corpus="how2sign-two-view-text-v1", partition="val", signer_component="c01", text=f"t{k}",
                 sign=f"s{k}") for k in range(3096)]
    a = [r["sign"] for r in validation_sentences(dict(records=pool), 300)]
    b = [r["sign"] for r in validation_sentences(dict(records=pool), 1000)]
    assert b[:300] != a and len(set(a) & set(b)) == 289
    assert a == [r["sign"] for r in random.Random(0).sample(pool, 300)]


def test_merge_requires_one_complete_run():
    def part(shard, clips, **kw):
        return dict(shard=shard, settings=dict(dict.fromkeys(ce.SETTINGS), clips=4, shards=2, **kw), texts=[],
                    signs=[], clips=[dict(clip=i) for i in clips])
    assert [r["clip"] for r in ce.merge_partials([part(1, [2, 3]), part(0, [0, 1])])["rows"]] == [0, 1, 2, 3]
    with pytest.raises(SystemExit, match="not 0..1"):
        ce.merge_partials([part(0, [0, 1])])
    with pytest.raises(SystemExit, match="two shards"):
        ce.merge_partials([part(0, [0, 1, 2]), part(1, [2, 3])])
    with pytest.raises(SystemExit, match="cover"):
        ce.merge_partials([part(0, [0]), part(1, [2, 3])])
    with pytest.raises(SystemExit, match="differs"):
        ce.merge_partials([part(0, [0, 1]), part(1, [2, 3], seed=5)])
    other = part(1, [2, 3])                                                   # same settings, other clips' texts
    other["texts"] = ["x"]
    with pytest.raises(SystemExit, match="texts"):
        ce.merge_partials([part(0, [0, 1]), other])
    other = dict(part(1, [2, 3]), signs=["s9"])
    with pytest.raises(SystemExit, match="signs"):
        ce.merge_partials([part(0, [0, 1]), other])


def test_the_default_corpora_reproduce_the_declared_model(tmp_path, stage2_store):  # noqa: F811
    """The vocabulary's corpora default to isolated_corpora(eval mix): round 15's four for a mix without the later
    corpora (round10-mixed-v1), so the default run fits the declared model; ISOLATED_CORPORA (five since round 14b)
    would record another corpus list and another hash."""
    from signtranslator.generation.compositional import ISOLATED_CORPORA, ROUND15_CORPORA
    store, sha, s1, oracle, argv = stage2_store
    import round15_stage2_fit as fit
    declared, _ = fit.fit_model(fit.agc._store(f"dir:{tmp_path / 'store'}"), "mix.json", sha, ROUND15_CORPORA)
    five, _ = fit.fit_model(fit.agc._store(f"dir:{tmp_path / 'store'}"), "mix.json", sha, ISOLATED_CORPORA)
    assert five["sha256"] != declared["sha256"]
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "oracle_stage2"
    assert ce.main(args + ["--clips", "5", "--expect-model-sha256", declared["sha256"], "--output-key", "e/d.json"]) == 0
    shard = json.loads(store.get("e/d.json"))
    assert shard["settings"]["corpora"] == list(ROUND15_CORPORA)
    assert shard["settings"]["model_sha256"] == declared["sha256"]
    assert "position" in shard["indexing"]
    assert ce.main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--shard-keys", "e/d.json", "--bootstrap", "20",
                    "--output-key", "e/dm.json"]) == 0
    assert json.loads(store.get("e/dm.json"))["indexing"] == shard["indexing"]


def test_check_stage1_report():
    val = [dict(sign="a"), dict(sign="b"), dict(sign="c")]
    stored = dict(per_clip=[dict(sign="a", words=["cut"]), dict(sign="z", words=["go"])])
    assert ce.check_stage1_report(val, [["cut"], [], ["go"]], stored) == dict(stored_clips=2, shared_clips=1,
                                                                              identical=1)
    with pytest.raises(SystemExit, match="differs"):
        ce.check_stage1_report(val, [["onion"], [], []], stored)


def pilot_report(store, s1, *, references=None):
    n = s1["clips"]
    meaning = {name: dict(bleu=1., hypotheses=[f"{name} {i}" for i in range(n)]) for name in ("generated", "real",
                                                                                              "blank")}
    meaning["references"] = references or s1["references"]
    store.put("pilot.json", json.dumps(dict(checkpoint_key="prior.pt", meaning=meaning)).encode())


def run_args(tmp_path, sha, *extra):
    return ["run", "--mix-key", "mix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
            "--device", "cpu", "--cache-dir", str(tmp_path / "cache"), "--checkpoint-key", "prior.pt",
            "--variants", "core_crossfade,oracle_stage2", "--block", "2", "--core-margin", "0.1", *extra]


def test_sharded_equals_unsharded_and_merge_scores(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    pilot_report(store, s1)
    clips = ["--clips", "5", "--stage1-report-key", "source.json"]
    assert ce.main(run_args(tmp_path, sha, *clips, "--shards", "1", "--output-key", "e/whole.json")) == 0
    for s in (0, 1):
        assert ce.main(run_args(tmp_path, sha, *clips, "--shards", "2", "--shard", str(s),
                                "--output-key", f"e/part{s}.json")) == 0
    whole = json.loads(store.get("e/whole.json"))
    assert whole["stage1_check"] == dict(stored_clips=5, shared_clips=5, identical=5)
    assert whole["settings"]["model_sha256"] and whole["settings"]["selection"] == "stage1"
    assert [r["stage1_words"] for r in whole["clips"]] == [r["words"] for r in s1["per_clip"]]
    rule = "bleu:end_to_end,bleu:mismatched_words,spotting"
    common = ["merge", "--store", f"dir:{tmp_path / 'store'}", "--end-to-end-report", "pilot.json", "--rule", rule,
              "--bootstrap", "40"]
    assert ce.main(common + ["--shard-keys", "e/whole.json", "--output-key", "e/m1.json"]) == 0
    assert ce.main(common + ["--shard-keys", "e/part1.json,e/part0.json", "--output-key", "e/m2.json"]) == 0
    m1, m2 = json.loads(store.get("e/m1.json")), json.loads(store.get("e/m2.json"))
    assert m1["forms"].keys() == m2["forms"].keys()
    for name in m1["forms"]:
        assert m1["forms"][name]["hypotheses"] == m2["forms"][name]["hypotheses"], name
        assert m1["forms"][name]["bleu"] == m2["forms"][name]["bleu"]
    assert m1["spotting"] == m2["spotting"] and m1["motion"] == m2["motion"] and m1["verdicts"] == m2["verdicts"]
    assert m1["forms"]["end_to_end"]["hypotheses"] == [f"generated {i}" for i in range(5)]
    for v in ("core_crossfade", "oracle_stage2"):
        assert {"minus_end_to_end", "minus_shuffled", "minus_mismatched_words"} <= set(m1["forms"][v])
        assert m1["verdicts"][v]["verdict"] in ("PROMISING", "NOT PROMISING")
    assert m1["motion_ratio_to_real"]["core_crossfade"]["wrist_speed"] is not None
    # --paired-with another merged report on the same clips
    assert ce.main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--shard-keys", "e/whole.json", "--paired-with",
                    "e/m1.json", "--bootstrap", "40", "--output-key", "e/m3.json"]) == 0
    m3 = json.loads(store.get("e/m3.json"))
    assert m3["forms"]["core_crossfade"]["minus_paired_core_crossfade"]["difference"] == 0.
    assert m3["comparators"]["paired_with"]["variants"] == ["core_crossfade", "oracle_stage2"]
    # refusals: a pilot of other clips, a rule naming a missing comparator, write-once
    pilot_report(store, s1, references=list(reversed(s1["references"])))
    with pytest.raises(SystemExit, match="references"):
        ce.main(common + ["--shard-keys", "e/whole.json", "--output-key", "e/x.json"])
    with pytest.raises(SystemExit, match="end-to-end-report"):
        ce.main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--shard-keys", "e/whole.json", "--rule", rule,
                 "--output-key", "e/y.json"])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ce.main(run_args(tmp_path, sha, *clips, "--shards", "1", "--output-key", "e/whole.json"))


def test_run_pins_the_model_and_takes_a_generator_mix(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    with pytest.raises(SystemExit, match="not the declared"):
        ce.main(run_args(tmp_path, sha, "--clips", "5", "--expect-model-sha256", "0" * 64, "--output-file",
                         str(tmp_path / "x.json")))
    gsha = generator_store(store)
    args = run_args(tmp_path, sha, "--clips", "5", "--variants", "core_crossfade")
    args[args.index("prior.pt")] = "prior_g.pt"
    with pytest.raises(SystemExit, match="different manifest"):
        ce.main(args + ["--output-file", str(tmp_path / "y.json")])
    assert ce.main(args + ["--generator-mix-key", "gmix.json", "--generator-mix-sha256", gsha, "--output-key",
                           "e/g.json"]) == 0
    assert json.loads(store.get("e/g.json"))["settings"]["generator_mix"]["key"] == "gmix.json"
    # tampered stage-1 report: refused
    bad = dict(s1)
    bad["per_clip"] = [dict(r, words=r["words"] + ["salt"]) for r in s1["per_clip"]]
    store.put("bad.json", json.dumps(bad).encode())
    with pytest.raises(SystemExit, match="differs"):
        ce.main(run_args(tmp_path, sha, "--clips", "5", "--stage1-report-key", "bad.json", "--output-file",
                         str(tmp_path / "z.json")))


# ---- composed-state export -----------------------------------------------------------------------------------------

def export_setup(tmp_path, store, sha, monkeypatch, *, signs_of=(1, 3)):
    """A D20-like set, a word cache on it, and a named sentence list. {args of a composition from the cache}."""
    import hashlib
    import torch
    import numpy as np
    import validate_sign_recognizer as vsr
    from scripts.round20_word_cache import main as cache_main
    from tests.test_round20_eval import put_set, manifest_of
    import scripts.round20_lib as lib
    set_sha, _ = put_set(store, sha, order=(4, 2, 0, 3))
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):
        out = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            out.append(torch.tensor([np.sin(level * (k + 3) * 50.) for k in range(len(labels))]))
        return torch.stack(out)
    monkeypatch.setattr(vsr, "logits_of", logits)
    assert cache_main(["run", "--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/w", "--mix-key", "mix.json",
                       "--mix-sha256", sha, "--sentences-key", "sets/d20.json", "--sentences-sha256", set_sha,
                       "--clips", "4", "--checkpoint-key", "prior.pt", "--samples", "3", "--choose", "1,3",
                       "--device", "cpu"]) == 0
    assert cache_main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/w"]) == 0
    cache_sha = hashlib.sha256(store.get("cache/w/index.json")).hexdigest()
    pool = lib.validation_pool(manifest_of(store, sha))
    order = [pool[k]["sign"] for k in (4, 2, 0, 3)]
    listed = [order[k] for k in signs_of]
    listing = tmp_path / "list.json"
    listing.write_text(json.dumps(dict(name="rater-test", signs=listed)))
    list_sha = hashlib.sha256(listing.read_bytes()).hexdigest()
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "core_crossfade"
    args += ["--clips", "4", "--sentences-key", "sets/d20.json", "--sentences-sha256", set_sha,
             "--word-cache-key", "cache/w/index.json", "--word-cache-sha256", cache_sha, "--word-cache-n", "3"]
    export = ["--export-arm", "arm-a", "--export-sentences-file", str(listing), "--export-sentences-sha256", list_sha]
    return args, export, dict(cache_sha=cache_sha, set_sha=set_sha, list_sha=list_sha, listed=listed, order=order,
                              listing=listing)


def without_seconds(report):
    return {k: v for k, v in report.items() if k != "seconds"}


def test_the_export_records_arm_cache_and_seeds_and_changes_nothing_else(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    import hashlib
    from evaluate_text_to_motion import load_state
    store, sha, s1, oracle, argv = stage2_store
    args, export, info = export_setup(tmp_path, store, sha, monkeypatch)
    assert ce.main(args + ["--output-key", "e/plain.json"]) == 0
    assert ce.main(args + ["--export-prefix", "x/full", *export, "--output-key", "e/full.json"]) == 0
    plain, full = json.loads(store.get("e/plain.json")), json.loads(store.get("e/full.json"))
    assert without_seconds(plain) == without_seconds(full)                 # the report does not know about the export
    index = json.loads(store.get("x/full/index.json"))
    assert index["kind"] == "round15-composed-states" and index["arm"] == "arm-a"
    assert index["sentences"] == dict(name="rater-test", sha256=info["list_sha"], listed=2)
    assert index["sentence_set"]["sha256"] == info["set_sha"] and index["sentence_set"]["set"] == "D20"
    assert index["word_cache"]["sha256"] == info["cache_sha"] and index["word_cache"]["n"] == 3
    assert index["settings"]["guidance"] == 1.5 and index["settings"]["timing"] == "sign-fitted"
    assert index["model_sha256"] == plain["settings"]["model_sha256"]
    assert sorted(r["sign"] for r in index["states"]) == sorted(info["listed"])
    picks = {r["entry"]: r["chosen"]["3"] for r in json.loads(store.get("cache/w/index.json"))["entries"]}
    by_clip = {r["clip"]: r for r in plain["clips"]}
    assert len(index["states"]) == 2
    for row in index["states"]:
        assert row["sign"] == info["order"][row["clip"]] and row["form"] == "core_crossfade" and not row["empty"]
        payload = store.get(row["state_key"])
        assert hashlib.sha256(payload).hexdigest() == row["state_sha256"]
        state = load_state(store, row["state_key"], row["state_sha256"])
        assert len(state.timestamps) == row["frames"] == row["reference_frames"]
        assert [w["word"] for w in row["words"]] == by_clip[row["clip"]]["entries"]
        assert row["words"]
        for w in row["words"]:
            pick = picks[w["word"]]
            assert w["seed"] == pick["seed"] is not None and w["sample"] == pick["index"]
            assert w["clip_key"] == pick["state_key"] and w["clip_sha256"] == pick["state_sha256"]
    # export only: the same states, bit for bit, without an instrument
    monkeypatch.setattr(ce, "SPOT", None)                                   # spotting would fail if reached
    assert ce.main(args + ["--export-prefix", "x/only", *export, "--export-only", "--export-forms", "core_crossfade",
                           "--output-file", str(tmp_path / "only.json")]) == 0
    only = json.loads((tmp_path / "only.json").read_text())
    assert only["kind"] == "round15-composed-states" and json.loads(store.get("x/only/index.json")) == only
    assert ([(r["sign"], r["state_sha256"], r["words"]) for r in only["states"]]
            == [(r["sign"], r["state_sha256"], r["words"]) for r in index["states"]])
    # write once
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ce.main(args + ["--export-prefix", "x/only", *export, "--output-file", str(tmp_path / "again.json")])


def test_the_export_is_refused_when_it_cannot_be_trusted(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, export, info = export_setup(tmp_path, store, sha, monkeypatch)
    out = ["--output-file", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit):                                        # options without the prefix
        ce.main(args + export + out)
    with pytest.raises(SystemExit):                                        # the prefix without its options
        ce.main(args + ["--export-prefix", "x/a"] + out)
    with pytest.raises(SystemExit):                                        # one shard only
        ce.main(args + ["--export-prefix", "x/a", *export, "--shards", "2"] + out)
    no_set = [a for a in args]
    k = no_set.index("--sentences-key")
    del no_set[k:k + 4]
    with pytest.raises(SystemExit):                                        # only from a stored set
        ce.main(no_set + ["--export-prefix", "x/a", *export] + out)
    with pytest.raises(SystemExit):                                        # a form the run does not make
        ce.main(args + ["--export-prefix", "x/a", *export, "--export-forms", "core_crossfade@predicted"] + out)
    bad = list(export)
    bad[bad.index(info["list_sha"])] = "0" * 64
    with pytest.raises(SystemExit, match="does not match"):
        ce.main(args + ["--export-prefix", "x/a", *bad] + out)
    import hashlib
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(dict(name="x", signs=["clip-not-in-the-set"])))
    other = list(export)
    other[other.index(str(info["listing"]))] = str(outside)
    other[other.index(info["list_sha"])] = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="not in this run's set"):
        ce.main(args + ["--export-prefix", "x/a", *other] + out)
    assert not store.exists("x/a/index.json")


def test_a_sentence_with_no_word_is_exported_as_empty(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, export, info = export_setup(tmp_path, store, sha, monkeypatch)
    real = ce.stage1_selection

    def no_words_for_the_first_listed(texts, vocab):
        details, words = real(texts, vocab)
        k = info["order"].index(info["listed"][0])
        details[k], words[k] = [], []
        return details, words
    monkeypatch.setattr(ce, "stage1_selection", no_words_for_the_first_listed)
    assert ce.main(args + ["--export-prefix", "x/e", *export, "--export-only", "--output-file",
                           str(tmp_path / "e.json")]) == 0
    rows = {r["sign"]: r for r in json.loads((tmp_path / "e.json").read_text())["states"]}
    assert rows[info["listed"][0]]["empty"] is True and rows[info["listed"][0]]["words"] == []
    assert rows[info["listed"][1]]["empty"] is False
