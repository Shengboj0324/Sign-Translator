"""Round 20 tooling end to end on the round-15 synthetic store (tiny prior, stub translator and recogniser): a stored
sentence set, the Sem-Lex activity core, continuous retiming with tempo, the low-pass, the likelihood metric, the
report-only predicted-duration form, the continuous-form oracle, the word cache (and its circularity note), the
paired comparison and the pod script's syntax. The default run keeps the round-15 report (no round-20 rows)."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest

import scripts.round15_compose_eval as ce
import scripts.round20_lib as lib
from tests.test_round15_compose_eval import run_args
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures)
from tests.test_round15_compositional import stubbed  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def likelihood_stub(monkeypatch):
    """A motion-dependent stand-in for the back-translator's reference likelihood."""
    import score_back_translation
    import scripts.score_back_translation as sbt

    def ll(model, kind, states, texts, batch=32):
        out = []
        for s, t in zip(states, texts):
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            v = -2. - (len(t) % 7) / 10. + level
            out.append(dict(log_likelihood=v * 5, tokens=5, per_token=round(v, 5)))
        return out
    for m in (sbt, score_back_translation):
        monkeypatch.setattr(m, "reference_log_likelihood", ll)


def manifest_of(store, sha):
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get("mix.json") + b"\n")
        return load_manifest(path, expected_sha256=sha)


def put_set(store, sha, order=(4, 2, 0, 3)):
    pool = lib.validation_pool(manifest_of(store, sha))
    payload = json.dumps(lib.set_payload("D20", [pool[k] for k in order], manifest_key="mix.json", manifest_sha256=sha,
                                         rule="test", counts={})).encode()
    store.put("sets/d20.json", payload)
    return hashlib.sha256(payload).hexdigest(), [pool[k]["text"] for k in order]


def merge(tmp_path, keys, out, *extra):
    assert ce.main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--shard-keys", keys, "--bootstrap", "20",
                    "--output-key", out, *extra]) == 0


def test_the_default_run_has_no_round20_rows(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    assert ce.main(run_args(tmp_path, sha, "--clips", "5", "--output-key", "e/d.json")) == 0
    shard = json.loads(store.get("e/d.json"))
    assert "round20" not in shard and "round20" not in shard["settings"]
    assert all("likelihood" not in r and "round20" not in json.dumps(r["plans"]) for r in shard["clips"])
    assert not any(k.endswith(ce.PREDICTED) for k in shard["clips"][0]["hypotheses"])
    merge(tmp_path, "e/d.json", "e/dm.json")
    merged = json.loads(store.get("e/dm.json"))
    assert "round20" not in merged and "likelihood" not in merged


def test_sentence_set_retiming_lowpass_likelihood_and_predicted(tmp_path, stage2_store, likelihood_stub,  # noqa: F811
                                                                 monkeypatch):
    from signtranslator.generation import composition_v2 as cv2
    store, sha, s1, oracle, argv = stage2_store
    # the tiny store has one How2Sign train clip: the duration model is given (its fit is unit-tested)
    monkeypatch.setattr(cv2, "fit_duration_model", lambda manifest: dict(intercept=.4, slope=.05, records=1))
    ssha, texts = put_set(store, sha)
    r20 = ["--sentences-key", "sets/d20.json", "--sentences-sha256", ssha, "--semlex-core", "activity",
           "--retime", "continuous", "--tempo", "1.25", "--lowpass-hz", "6", "--likelihood", "--predicted-duration"]
    for s in (0, 1):
        assert ce.main(run_args(tmp_path, sha, *r20, "--shards", "2", "--shard", str(s),
                                "--output-key", f"e/p{s}.json")) == 0
    shard = json.loads(store.get("e/p0.json"))
    assert shard["texts"] == texts and shard["settings"]["clips"] == 4
    st = shard["settings"]["round20"]
    assert st["sentences"]["set"] == "D20" and st["retime"] == "continuous" and st["tempo"] == 1.25
    assert st["lowpass_hz"] == 6. and st["likelihood"] and st["predicted_duration"]
    assert shard["round20"]["coverage"]["sentences"] == 4 and "duration_model" in shard["round20"]
    row = shard["clips"][0]
    assert set(row["likelihood"]) == set(row["hypotheses"])
    for v in ("core_crossfade", "oracle_stage2"):
        if v in row["plans"]:
            rec = row["plans"][v]["round20"]
            assert rec["fit"] in ("padded", "compressed", "truncated") and rec["frames"] == row["frames_reference"]
            assert len(rec["duration_sources"]) == len(row["entries"])
            assert row["plans"][v]["predicted"]["frames"] > 0
            assert f"{v}{ce.PREDICTED}" in row["hypotheses"]
    merge(tmp_path, "e/p1.json,e/p0.json", "e/m.json")
    m = json.loads(store.get("e/m.json"))
    assert m["references"] == texts
    ll = m["likelihood"]["forms"]
    assert {"minus_blank", "minus_mismatched", "minus_shuffled", "minus_mismatched_words"} <= set(ll["core_crossfade"])
    assert len(m["likelihood"]["per_form"]["core_crossfade"]) == 4
    assert "minus_reference_duration" in m["forms"][f"core_crossfade{ce.PREDICTED}"]
    pv = m["round20"]["per_variant"]["oracle_stage2"]
    assert pv["words"] > 0 and pv["continuous_share"] == 0. and set(pv["sources"]) == {"isolated"}
    assert set(m["round20"]["per_variant"]["core_crossfade"]["sources"]) == {"generated"}
    assert m["round20"]["per_variant"]["core_crossfade"]["velocity_raised_share"] is not None
    # the paired D20 comparison of two forms (BLEU and the likelihood metric)
    from scripts.round20_compare import compare_forms
    c = compare_forms(m, "core_crossfade", m, "oracle_stage2", boot=50)
    assert c["sentences"] == 4 and "difference" in c["likelihood_minus_blank"]["difference"]
    assert c["both_improve"] == (c["bleu"]["difference"] > 0 and c["likelihood_minus_blank"]["difference"]["difference"] > 0)
    for field, value in (("signs", ["x"] * 4), ("references", ["y"] * 4)):
        with pytest.raises(SystemExit, match="not paired"):
            compare_forms(m, "core_crossfade", dict(m, **{field: value}), "oracle_stage2", boot=20)
    for field, value in (("seed", 7), ("translator_key", "other.pt"),
                         ("round20", dict(m["settings"]["round20"], sentences=dict(sha256="0" * 64)))):
        with pytest.raises(SystemExit, match="not paired"):
            compare_forms(m, "core_crossfade", dict(m, settings=dict(m["settings"], **{field: value})),
                          "oracle_stage2", boot=20)
    assert m["round20"]["per_variant"]["oracle_stage2"]["right_arm_tracked"]["words"] > 0
    same = compare_forms(m, "core_crossfade", m, "core_crossfade", boot=20)
    assert same["bleu"]["difference"] == 0 and same["likelihood_minus_blank"]["difference"]["difference"] == 0
    assert not same["both_improve"]
    # --paired-with pairs the likelihoods too
    merge(tmp_path, "e/p0.json,e/p1.json", "e/m2.json", "--paired-with", "e/m.json")
    m2 = json.loads(store.get("e/m2.json"))
    assert m2["likelihood"]["forms"]["core_crossfade"]["minus_paired_core_crossfade"]["difference"] == 0.
    # refusals
    with pytest.raises(SystemExit, match="sentences-sha256"):
        ce.main(run_args(tmp_path, sha, "--sentences-key", "sets/d20.json", "--sentences-sha256", "0" * 64,
                         "--output-file", str(tmp_path / "x.json")))
    for bad in (["--tempo", "1.5"], ["--retime", "continuous", "--timing", "stage1"], ["--lowpass-hz", "13"],
                ["--continuous-ratio", ".4"], ["--bridge-min-speed", ".5"], ["--retime", "continuous",
                                                                           "--no-bridge-velocity", "--bridge-max-seconds",
                                                                           "1"],
                ["--activity-margin", ".1"], ["--word-cache-n", "8"], ["--gloss-manifest-key", "g.json"],
                ["--lowpass-hz", "6", "--lowpass-report-hz", "6"],
                ["--semlex-oracle-tracked", "--variants", "core_crossfade"],
                ["--semlex-core", "activity", "--timing", "asllrp"], ["--oracle-source", "continuous",
                                                                       "--variants", "core_crossfade"]):
        with pytest.raises(SystemExit):
            ce.main(run_args(tmp_path, sha, *bad, "--output-file", str(tmp_path / "y.json")))


def put_gloss_inputs(store):
    """Utterance videos for the fixture's ASLLRP utterances, and a gloss manifest of their states (train)."""
    import round15_stage2_fit as fit
    from signtranslator.data_engineering.corpus_manifest import KINDS, canonical_json, usable_channels
    from tests.test_generation_signing_space import signer_state
    items = json.loads(store.get(fit.KEYS["utterances"]))["items"]
    records = []
    for k, u in enumerate(items):
        u["utterance_video"] = f"u{k}.mov"
        frames = u["frames"][1] - u["frames"][0] + 1
        state = signer_state(50 + k, frames=frames, amplitude=.4)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            ssha = state.save(path)
            store.put(f"utt/{k}.npz", path.read_bytes())
        r = dict(sign=f"{u['collection']}:u{k}", partition="train" if u["source"].startswith("Cory") else "val",
                 text="X", continuous=dict(status="converted", state_key=f"utt/{k}.npz", state_sha256=ssha,
                                           available_channels=list(state.channels), seconds=frames / 24.))
        r["usable"] = {kind: list(usable_channels(r, kind)) for kind in KINDS}
        records.append(r)
    store.put(fit.KEYS["utterances"], json.dumps(dict(items=items)).encode())
    payload = canonical_json(dict(records=records))
    store.put("gloss.json", payload)
    return hashlib.sha256(payload).hexdigest()


def test_the_continuous_form_oracle(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    gsha = put_gloss_inputs(store)
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "oracle_stage2"
    assert ce.main(args + ["--clips", "5", "--oracle-source", "continuous", "--gloss-manifest-key", "gloss.json",
                           "--gloss-manifest-sha256", gsha, "--output-key", "e/c.json"]) == 0
    assert ce.main(args + ["--clips", "5", "--semlex-oracle-tracked", "--lowpass-report-hz", "6",
                           "--output-key", "e/i.json"]) == 0
    shard = json.loads(store.get("e/c.json"))
    assert shard["settings"]["round20"]["gloss_manifest"] == dict(key="gloss.json", sha256=gsha)
    assert shard["round20"]["segments"]["segments"] > 0
    merge(tmp_path, "e/c.json", "e/cm.json")
    pv = json.loads(store.get("e/cm.json"))["round20"]["per_variant"]["oracle_stage2"]
    assert pv["sources"].get("continuous", 0) > 0 and 0 < pv["continuous_share"] <= 1
    assert pv["reasons"].get("continuous") == pv["sources"]["continuous"]
    iso = json.loads(store.get("e/i.json"))
    assert iso["clips"][0]["entries"] == shard["clips"][0]["entries"]          # same words, other material
    assert iso["round20"]["semlex_oracle_tracked"]["floor"] == .3              # no Sem-Lex-only word here
    merge(tmp_path, "e/i.json", "e/im.json")
    im = json.loads(store.get("e/im.json"))
    assert "minus_unfiltered" in im["forms"]["oracle_stage2@lowpass"]
    assert im["forms"]["oracle_stage2@lowpass"]["hypotheses"] != [] and "oracle_stage2@lowpass" in im["motion"]


def test_the_word_cache_and_composition_from_it(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    import torch
    import validate_sign_recognizer as vsr
    from scripts.round20_word_cache import main as cache_main
    store, sha, s1, oracle, argv = stage2_store
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW"]           # no STIR: the prior's PMI decides it
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):
        out = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            out.append(torch.tensor([np.sin(level * (k + 3) * 50.) for k in range(len(labels))]))
        return torch.stack(out)
    monkeypatch.setattr(vsr, "logits_of", logits)
    common = ["--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/w"]
    run = ["run", *common, "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "5", "--checkpoint-key", "prior.pt",
           "--samples", "3", "--choose", "1,3", "--device", "cpu", "--shards", "2"]
    for s in (0, 1):
        assert cache_main(run + ["--shard", str(s)]) == 0
    with pytest.raises(SystemExit, match="refusing"):
        cache_main(run + ["--shard", "0"])
    assert cache_main(["merge", *common, "--shards", "2"]) == 0
    payload = store.get("cache/w/index.json")
    index = json.loads(payload)
    rows = {r["entry"]: r for r in index["entries"]}
    assert rows["stir"]["scorer"] == "prior_pmi" and rows["cut"]["scorer"] == "recogniser"
    assert len(rows["cut"]["probabilities"]) == 3 and all(1 <= k <= len(labels) for k in rows["cut"]["ranks"])
    for r in rows.values():
        assert r["chosen"]["1"]["index"] == 0
        assert r["chosen"]["3"]["index"] == int(np.argmax(r["scores"]))
    assert index["summary"]["labelled"] == len(rows) - 1 and "best_of_3" in index["summary"]
    # best-of-1 is the composition's own generate_word at the entry's seed
    from evaluate_text_to_motion import load_generator, load_state
    from exposure_diagnostic import make_post
    from signtranslator.generation.compositional import generate_word
    temperature = {"body": 1.1, "left_hand": 1.1, "right_hand": 1.1, "face": 1.}
    _, _, sample = load_generator(store, "prior.pt", sha, "cpu", temperature=temperature)
    w = generate_word(sample.prior, sample.tokenizer, "cut", rows["cut"]["seconds"], lib.entry_seed("cut"),
                      guidance=1.5, temperature=temperature, post=make_post(sample.decoder_gains))
    got = load_state(store, rows["cut"]["chosen"]["1"]["state_key"], rows["cut"]["chosen"]["1"]["state_sha256"])
    np.testing.assert_array_equal(w.channels["body"].values, got.channels["body"].values)
    # composition from the cache: no generation, the cache's spotting note
    csha = hashlib.sha256(payload).hexdigest()
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "core_crossfade"
    cache = ["--word-cache-key", "cache/w/index.json", "--word-cache-sha256", csha, "--word-cache-n", "3"]
    assert ce.main(args + ["--clips", "5", *cache, "--output-key", "e/w.json"]) == 0
    shard = json.loads(store.get("e/w.json"))
    assert shard["settings"]["round20"]["word_cache"]["n"] == 3 and shard["settings"]["checkpoint_key"] is None
    merge(tmp_path, "e/w.json", "e/wm.json")
    wm = json.loads(store.get("e/wm.json"))["round20"]
    assert wm["spotting_note"].startswith("independent") and set(wm["per_variant"]["core_crossfade"]["sources"]) == {"cache"}
    assert ce.main(args + ["--clips", "5", *cache, "--recognizer-run", "sign-recognizer-v3d",
                           "--output-key", "e/w3.json"]) == 0
    merge(tmp_path, "e/w3.json", "e/w3m.json")
    assert json.loads(store.get("e/w3m.json"))["round20"]["spotting_note"].startswith("CIRCULAR")
    with pytest.raises(SystemExit, match="best-of-5"):
        ce.main(args + ["--clips", "5", *cache[:-1], "5", "--output-file", str(tmp_path / "z.json")])
    with pytest.raises(SystemExit):                                       # the cache replaces the checkpoint
        ce.main(run_args(tmp_path, sha, "--clips", "5", *cache, "--output-file", str(tmp_path / "q.json")))
    # the choice compares unrounded log-probabilities: all-tiny probabilities are still ordered
    def tiny_logits(model, states, device, batch=64):
        out = []
        for s in states:
            rh = s.channels["right_hand"]
            level = float(np.abs(rh.values[rh.valid]).mean()) if rh.valid.any() else 0.
            row = torch.zeros(len(labels), dtype=torch.float64)
            row[:] = 60.
            row[labels.index("CUT")] = level * 1e-3        # p(CUT) ~ e^-60: rounds to 0 at 6 places
            row[labels.index("ONION")] = level * 2e-3
            out.append(row)
        return torch.stack(out)
    monkeypatch.setattr(vsr, "logits_of", tiny_logits)
    tiny = ["run", "--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/tiny", "--mix-key", "mix.json",
            "--mix-sha256", sha, "--clips", "5", "--checkpoint-key", "prior.pt", "--samples", "3", "--choose", "3",
            "--device", "cpu"]
    assert cache_main(tiny) == 0
    rows_t = {r["entry"]: r for r in json.loads(store.get("cache/tiny/shard-0-of-1.json"))["entries"]}
    for e in ("cut", "onion"):
        r = rows_t[e]
        assert r["probabilities"] == [0., 0., 0.] and len(set(r["log_probabilities"])) == 3
        assert r["chosen"]["3"]["index"] == int(np.argmax(r["log_probabilities"])) and r["scores"] == r["log_probabilities"]
    # A5: two indexes compared entry by entry
    from scripts.round20_compare import compare_words
    out = compare_words(index, index, boot=20)
    assert out["entries"] == len(rows) - 1 and out["sample0"]["p_label"]["difference"]["difference"] == 0.
    assert "best_of_3" in out


def test_the_pod_script_parses():
    script = ROOT / "runpod/fleet/round20_phase_a.sh"
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    for needle in ("--sentences-key", "--vocabulary-mix-key", "--oracle-source continuous", "--retime continuous",
                   "--lowpass-report-hz", "round20_word_cache.py", "--likelihood", "SMOKE", "r14b-fraction",
                   "--semlex-oracle-tracked", "kill $q"):
        assert needle in text, needle
