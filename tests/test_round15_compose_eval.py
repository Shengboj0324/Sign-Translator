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
