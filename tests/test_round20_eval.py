"""Round 20 tooling end to end on the round-15 synthetic store (tiny prior, stub translator and recogniser): a stored
sentence set, the Sem-Lex activity core, continuous retiming with tempo, the low-pass, the likelihood metric, the
report-only predicted-duration form, the continuous-form oracle, the word cache (and its circularity note), the
paired comparison and the pod script's syntax. The default run keeps the round-15 report (no round-20 rows)."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.round15_compose_eval as ce
import scripts.round20_lib as lib
from tests.test_round15_compose_eval import run_args
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures)
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_sentence_exposure_diagnostic import TEMPERATURE

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
                   "--semlex-oracle-tracked", "kill $q", "an early failure stops the others at once"):
        assert needle in text, needle


# ---- A7: composition without the accepted decoder gains ----------------------------------------------------------------

GAINS = {"body": 1.4, "head_rotation": 1.25}
FRACTIONS = dict(entries={"cut": (.25, .75)}, all=(.3, .6))
JOB_SETTINGS = dict(guidance=1.5, temperature=TEMPERATURE, rule=None, timing="sign-fitted", fractions=FRACTIONS, margin=.1)


def put_gains(store):
    """The tiny prior now names a tokenizer whose accepted gains are stored, as W14b's does."""
    import torch
    from fit_tokenizer_gains import decoder_gains_key
    tokenizer_key = "training-runs/tok/checkpoints/step-00000001.pt"
    checkpoint = torch.load(io.BytesIO(store.get("prior.pt")), map_location="cpu", weights_only=False)
    buffer = io.BytesIO()
    torch.save(dict(checkpoint, tokenizer_key=tokenizer_key), buffer)
    store.put("prior.pt", buffer.getvalue())
    store.put(decoder_gains_key(tokenizer_key), json.dumps(dict(tokenizer_key=tokenizer_key, gains=GAINS)).encode())


def loaded(store, sha, **kw):
    from evaluate_text_to_motion import load_generator
    return load_generator(store, "prior.pt", sha, "cpu", temperature=TEMPERATURE, **kw)[2]


def assert_same_state(a, b):
    assert set(a.channels) == set(b.channels)
    for name in a.channels:
        np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values)
        np.testing.assert_array_equal(a.channels[name].valid, b.channels[name].valid)


def differs(a, b, channel):
    return not np.array_equal(a.channels[channel].values, b.channels[channel].values)


WORDS = dict(mode="core_crossfade", words=["cut", "onion"], generate=[.5, .4], seeds=[3, 4])


def test_a_word_without_gains_is_the_plain_decode_plus_head_smoothing(tmp_path, stage2_store):  # noqa: F811
    import torch
    import scripts.round15_stage2 as r2
    from exposure_diagnostic import make_post
    from signtranslator.generation.compositional import generate_word
    from signtranslator.generation.motion_rvq import generate_rvq_motion
    store, sha, *_ = stage2_store
    put_gains(store)
    on, off = loaded(store, sha), loaded(store, sha, decoder_gains=False)
    assert on.decoder_gains == GAINS and off.decoder_gains is None
    a, b = (r2.generate_job(s, WORDS, **JOB_SETTINGS) for s in (on, off))
    assert a["segments"] == b["segments"]                                        # same seeds and same (fractional) cuts
    for k, (w, d, s) in enumerate(zip(WORDS["words"], WORDS["generate"], WORDS["seeds"])):
        codes = [generate_rvq_motion(x.prior, x.tokenizer, [w], fps=24., durations=[d], guidance=1.5,
                                     temperature=TEMPERATURE, seed=s, return_codes=True)[1][0] for x in (on, off)]
        assert torch.equal(*codes)                                               # the same seed samples the same codes
        plain = generate_word(off.prior, off.tokenizer, w, d, s, guidance=1.5, temperature=TEMPERATURE, post=lambda x: x)
        assert_same_state(b["pieces"][k]["state"], make_post(None)(plain))
        assert_same_state(a["pieces"][k]["state"], make_post(GAINS)(plain))
        assert differs(b["pieces"][k]["state"], plain, "head_rotation")          # head smoothing stays on
        assert differs(b["pieces"][k]["state"], a["pieces"][k]["state"], "body")  # the gain is off
        assert not differs(b["pieces"][k]["state"], a["pieces"][k]["state"], "right_hand")   # no gain there: same codes
    # the sample wrapper has no gain either, and the default load is the gained one
    args = ("cut", 24., .5, 1.5, None, 3)
    from fit_tokenizer_gains import apply_gains
    plain = generate_rvq_motion(off.prior, off.tokenizer, ["cut"], fps=24., durations=[.5], guidance=1.5,
                                temperature=TEMPERATURE, seed=3)[0]
    assert_same_state(off(*args), plain)
    assert_same_state(on(*args), apply_gains(plain, GAINS))
    # the chained sampler and the stage-1 timing go through the same post
    chained = dict(WORDS, mode="core_chained", durations=[.3, .35], overlap=4)
    a, b = (r2.generate_job(s, chained, **JOB_SETTINGS) for s in (on, off))
    assert a["segments"] == b["segments"]
    assert differs(b["parts"]["cores"][0], a["parts"]["cores"][0], "body")
    assert not differs(b["parts"]["cores"][0], a["parts"]["cores"][0], "right_hand")
    timing = dict(JOB_SETTINGS, timing="stage1")
    a, b = (r2.generate_job(s, chained, **timing) for s in (on, off))
    assert differs(b["state"], a["state"], "body") and not differs(b["state"], a["state"], "right_hand")


def test_no_gains_reach_a_spawned_workers_job(tmp_path, stage2_store):  # noqa: F811
    import scripts.round15_stage2 as r2
    store, sha, *_ = stage2_store
    put_gains(store)
    on, off = loaded(store, sha), loaded(store, sha, decoder_gains=False)
    jobs = [WORDS, dict(WORDS, words=["go", "cut"], seeds=[5, 6])]
    worker = (f"dir:{tmp_path / 'store'}", "prior.pt", sha, "cpu", TEMPERATURE, 1)
    here = r2.run_jobs(off, jobs, JOB_SETTINGS, decoder_gains=False)
    there = r2.run_jobs(None, jobs, JOB_SETTINGS, workers=2, worker_args=worker, decoder_gains=False)
    gained = r2.run_jobs(None, jobs, JOB_SETTINGS, workers=2, worker_args=worker)       # default: the gains stay
    for h, t, g, plain in zip(here, there, gained, r2.run_jobs(on, jobs, JOB_SETTINGS)):
        for p, q, r, s in zip(h["pieces"], t["pieces"], g["pieces"], plain["pieces"]):
            assert_same_state(p["state"], q["state"])
            assert_same_state(r["state"], s["state"])
            assert differs(q["state"], r["state"], "body")
    with pytest.raises(ValueError, match="without the gains"):                   # a gained parent cannot claim otherwise
        r2.run_jobs(on, jobs, JOB_SETTINGS, decoder_gains=False)


def test_a7_shares_a0s_jobs_and_codes_in_spawned_workers_and_pairs_with_it(tmp_path, stage2_store, likelihood_stub,  # noqa: F811
                                                                            monkeypatch):
    import round15_stage2
    from scripts.round20_compare import main as compare_main
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, texts = put_set(store, sha)
    real, runs = round15_stage2.run_jobs, []

    def recording(sample, jobs, settings, **kw):
        out = real(sample, jobs, settings, **kw)
        runs.append(dict(jobs=jobs, settings=settings, material=out))
        return out
    monkeypatch.setattr(round15_stage2, "run_jobs", recording)
    a0 = ["--sentences-key", "sets/d20.json", "--sentences-sha256", ssha, "--variants", "core_crossfade", "--likelihood",
          "--lowpass-report-hz", "6"]
    assert ce.main(run_args(tmp_path, sha, *a0, "--output-key", "e/a0.json")) == 0
    assert ce.main(run_args(tmp_path, sha, *a0, "--no-decoder-gains", "--output-key", "e/a7.json")) == 0
    # the CLI's own worker path: a fix in the parent process alone leaves the gains on here, and this fails
    assert ce.main(run_args(tmp_path, sha, *a0, "--no-decoder-gains", "--word-workers", "2",
                            "--output-key", "e/a7w.json")) == 0
    gained, here, there = runs
    assert gained["jobs"] == here["jobs"] == there["jobs"] and gained["settings"] == here["settings"]
    assert any(len(j["seeds"]) for j in there["jobs"])
    body = 0
    for g, h, t in zip(gained["material"], here["material"], there["material"]):
        assert g["segments"] == h["segments"] == t["segments"]
        for p, q, r in zip(g["pieces"], h["pieces"], t["pieces"]):
            assert_same_state(q["state"], r["state"])                            # one worker or two: the same words
            assert not differs(p["state"], r["state"], "right_hand")             # same codes as a0
            body += differs(p["state"], r["state"], "body")
    assert body > 0
    x, y = (json.loads(store.get(f"e/{k}.json")) for k in ("a0", "a7"))
    assert "no_decoder_gains" not in x["settings"]["round20"] and y["settings"]["round20"]["no_decoder_gains"] is True
    y_settings = dict(y["settings"], round20={k: v for k, v in y["settings"]["round20"].items() if k != "no_decoder_gains"})
    assert x["settings"] == y_settings                                           # every other setting is a0's
    merge(tmp_path, "e/a0.json", "e/a0-w14b-r10.json")
    merge(tmp_path, "e/a7w.json", "e/a7-nogains-r10.json")
    m0, m7 = (json.loads(store.get(k)) for k in ("e/a0-w14b-r10.json", "e/a7-nogains-r10.json"))
    assert m7["round20"]["settings"]["no_decoder_gains"] is True and m7["references"] == m0["references"] == texts
    assert "no_decoder_gains" not in m0["round20"]["settings"]
    assert m7["forms"]["real"]["hypotheses"] == m0["forms"]["real"]["hypotheses"]
    out = tmp_path / "pair.json"
    assert compare_main(["forms", "--store", f"dir:{tmp_path / 'store'}", "--a", "e/a7-nogains-r10.json:core_crossfade",
                         "--b", "e/a0-w14b-r10.json:core_crossfade", "--bootstrap", "20", "--output-file", str(out)]) == 0
    pair = json.loads(out.read_text())
    assert pair["sentences"] == len(texts) and "likelihood_minus_blank" in pair


def test_the_option_off_changes_nothing(tmp_path, stage2_store):  # noqa: F811
    store, sha, *_ = stage2_store
    put_gains(store)
    assert ce.ROUND20_DEFAULTS["no_decoder_gains"] is False
    assert ce.main(run_args(tmp_path, sha, "--clips", "5", "--output-key", "e/d.json")) == 0
    assert ce.main(run_args(tmp_path, sha, "--clips", "5", "--lowpass-report-hz", "6", "--output-key", "e/r.json")) == 0
    plain, r20 = (json.loads(store.get(k)) for k in ("e/d.json", "e/r.json"))
    assert "round20" not in plain and "round20" not in plain["settings"]         # the option alone adds no row
    assert "no_decoder_gains" not in json.dumps(plain) and "no_decoder_gains" not in json.dumps(r20)
    assert set(r20["settings"]["round20"]) == {           # the settings already in the stored round-20 reports
        "sentences", "vocabulary_mix", "semlex_core", "activity_margin", "oracle_source", "gloss_manifest", "retime",
        "tempo", "bridge_velocity", "lowpass_hz", "word_cache", "likelihood", "predicted_duration",
        "semlex_oracle_tracked", "lowpass_report_hz"}
    assert loaded(store, sha).decoder_gains == GAINS == loaded(store, sha, decoder_gains=True).decoder_gains


def test_no_gains_refusals(tmp_path, stage2_store, capsys, monkeypatch):  # noqa: F811
    import round15_stage2
    import torch
    from evaluate_text_to_motion import load_generator
    from scripts.round20_word_cache import main as cache_main
    store, sha, *_ = stage2_store
    off = ["--no-decoder-gains", "--output-file", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit, match="no accepted decoder gains"):           # the tiny prior names no tokenizer
        ce.main(run_args(tmp_path, sha, "--clips", "5", *off))
    with pytest.raises(SystemExit, match="no accepted decoder gains"):
        cache_main(["run", "--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/none", "--mix-key", "mix.json",
                    "--mix-sha256", sha, "--clips", "5", "--checkpoint-key", "prior.pt", "--samples", "1", "--choose",
                    "1", "--device", "cpu", "--no-decoder-gains"])
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_token_prior", manifest_sha256=sha), buffer)
    store.put("old.pt", buffer.getvalue())
    with pytest.raises(SystemExit, match="not an rvq prior"):
        load_generator(store, "old.pt", sha, decoder_gains=False)
    put_gains(store)
    args = [a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
    args[args.index("core_crossfade,oracle_stage2")] = "oracle_stage2"
    with pytest.raises(SystemExit):                                              # oracle-only: no generated word
        ce.main(args + ["--clips", "5", *off])
    assert "needs a generated variant" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                              # a cache's gains are already fixed
        ce.main([a for a in run_args(tmp_path, sha) if a not in ("--checkpoint-key", "prior.pt")]
                + ["--variants", "core_crossfade", "--clips", "5", "--word-cache-key", "c.json", "--word-cache-sha256",
                   "0" * 64, *off])
    assert "a cache's gains are fixed" in capsys.readouterr().err
    plan_jobs = round15_stage2.plan_jobs                                         # no sentence selects a word
    monkeypatch.setattr(round15_stage2, "plan_jobs", lambda *a, **k: {
        key: None if key[0] in round15_stage2.GENERATED else job for key, job in plan_jobs(*a, **k).items()})
    with pytest.raises(SystemExit, match="generates no word"):
        ce.main(run_args(tmp_path, sha, "--clips", "5", *off))
    assert not (tmp_path / "o.json").exists()


def test_a_word_cache_without_gains(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    import torch
    import validate_sign_recognizer as vsr
    from evaluate_text_to_motion import load_state
    from exposure_diagnostic import make_post
    from scripts.round20_word_cache import main as cache_main
    from signtranslator.generation.compositional import generate_word
    store, sha, *_ = stage2_store
    put_gains(store)
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))
    monkeypatch.setattr(vsr, "logits_of", lambda model, states, device, batch=64: torch.zeros(len(states), len(labels)))

    def build(prefix, *extra):
        common = ["--store", f"dir:{tmp_path / 'store'}", "--prefix", prefix]
        assert cache_main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "5", "--limit", "2",
                           "--checkpoint-key", "prior.pt", "--samples", "2", "--choose", "1,2", "--device", "cpu",
                           *extra]) == 0
        assert cache_main(["merge", *common]) == 0
        payload = store.get(f"{prefix}/index.json")
        return json.loads(payload), hashlib.sha256(payload).hexdigest()
    on, _ = build("cache/on")
    off, off_sha = build("cache/off", "--no-decoder-gains")
    assert "no_decoder_gains" not in on["settings"] and off["settings"]["no_decoder_gains"] is True
    assert json.loads(store.get("cache/off/shard-0-of-1.json"))["settings"]["no_decoder_gains"] is True
    sample = loaded(store, sha, decoder_gains=False)
    for r, s in zip(on["entries"], off["entries"]):
        assert r["entry"] == s["entry"] and r["seeds"] == s["seeds"]             # the same seeds
        plain = generate_word(sample.prior, sample.tokenizer, r["entry"], r["seconds"], r["seeds"][0], guidance=1.5,
                              temperature=TEMPERATURE, post=lambda x: x)
        for row, gains in ((r, GAINS), (s, None)):
            pick = row["chosen"]["1"]
            assert_same_state(load_state(store, pick["state_key"], pick["state_sha256"]), make_post(gains)(plain))
    # a composition from the no-gains cache states it (load_word_cache: the sentences' words must be in the cache)
    args = SimpleNamespace(word_cache_key="cache/off/index.json", word_cache_sha256=off_sha, word_cache_n=1,
                           vocabulary_mix_key=None, vocabulary_mix_sha256=None, mix_key="mix.json", mix_sha256=sha)
    words = [r["entry"] for r in off["entries"]]
    info, states = ce.load_word_cache(store, args, dict.fromkeys(words), off["settings"]["corpora"], words)
    assert info["no_decoder_gains"] is True and set(states) == set(words)
    on_payload = store.get("cache/on/index.json")
    args.word_cache_key, args.word_cache_sha256 = "cache/on/index.json", hashlib.sha256(on_payload).hexdigest()
    assert "no_decoder_gains" not in ce.load_word_cache(store, args, dict.fromkeys(words), on["settings"]["corpora"],
                                                        words)[0]


def test_the_pod_script_runs_a7_as_a0_plus_the_option():
    text = (ROOT / "runpod/fleet/round20_phase_a.sh").read_text()
    a0 = text.split("      compose a0-w14b-$VOCAB ")[1].split(" ;;")[0]
    a7 = text.split("      compose a7-nogains-$VOCAB ")[1].split(" ;;")[0]
    assert a7 == a0 + " --no-decoder-gains" and "$GEN --lowpass-report-hz 6" in a0
    assert "#   a7    (GPU, VOCAB=r10 only" in text and "a1 a0 a7 a2" in text


def test_the_pod_script_runs_a8_as_a0_with_r16s_words():
    text = (ROOT / "runpod/fleet/round20_phase_a.sh").read_text()
    a8 = text.split("      compose a8-r16-$VOCAB ")[1].split(" ;;")[0].replace("\\\n       ", "")
    assert " ".join(a8.split()) == ("--variants core_crossfade $VMIX --checkpoint-key $R16 --generator-mix-key $R14B "
                                    "--generator-mix-sha256 $R14BSHA --word-workers ${WORD_WORKERS:-4} "
                                    "--lowpass-report-hz 6")
    gen = text.split('GEN="')[1].split('"')[0].replace("\\\n ", "")
    assert " ".join(gen.split()).replace("$W14B", "$R16") == " ".join(a8.split()).replace(" --lowpass-report-hz 6", "")
    assert "a8) pin $R16 $R16SHA" in text


# ---- A7: proving the same codes ---------------------------------------------------------------------------------------

def test_a0s_smoke_jobs_are_the_full_runs_first_jobs_and_only_the_controls_move():
    import scripts.round15_stage2 as r2
    from sentence_exposure_diagnostic import derangement
    plans = [dict(entries=[f"w{k}", "x"], generate=[.3, .4], targets=[.2, .2]) for k in range(500)]
    orders = [[1, 0]] * 500

    def jobs(n):
        return r2.plan_jobs(["core_crossfade"], plans[:n], derangement(n, 1), orders[:n], overlap=4)
    full, smoke = jobs(500), jobs(5)
    for i in range(5):
        assert smoke[("core_crossfade", "", i)] == full[("core_crossfade", "", i)]     # words and position-derived seeds
    assert any(smoke[("core_crossfade", "/mismatched_words", i)] != full[("core_crossfade", "/mismatched_words", i)]
               for i in range(5))                                                      # the derangement depends on n


def test_same_codes_check(tmp_path, stage2_store, likelihood_stub):  # noqa: F811
    from scripts.round20_compare import main as compare_main, same_codes_check
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, _ = put_set(store, sha)
    a0 = ["--sentences-key", "sets/d20.json", "--sentences-sha256", ssha, "--variants", "core_crossfade", "--likelihood",
          "--lowpass-report-hz", "6"]
    assert ce.main(run_args(tmp_path, sha, *a0, "--output-key", "e/full.json")) == 0
    assert ce.main(run_args(tmp_path, sha, *a0, "--clips", "2", "--output-key", "e/smoke.json")) == 0
    assert ce.main(run_args(tmp_path, sha, *a0, "--no-decoder-gains", "--clips", "2", "--output-key", "e/off.json")) == 0
    full, smoke, off = (json.loads(store.get(f"e/{k}.json")) for k in ("full", "smoke", "off"))
    assert smoke["settings"]["clips"] == 2 and full["settings"]["clips"] == 4
    assert same_codes_check(smoke, full, first=2) == dict(same=True, form="core_crossfade", clips=2, words=sum(
        len(r["entries"]) for r in smoke["clips"]))                                  # the smoke is the run's first clips

    def changed(edit):
        copy = json.loads(json.dumps(smoke))
        edit(copy)
        return copy

    def bump(c):                                                                 # one value, by the smallest amount
        c["clips"][1]["motion"]["core_crossfade"]["wrist_speed"] = np.nextafter(
            c["clips"][1]["motion"]["core_crossfade"]["wrist_speed"], np.inf).item()
    with pytest.raises(SystemExit, match=r"clip 1 motion.core_crossfade.wrist_speed"):
        same_codes_check(changed(bump), full, first=2)
    with pytest.raises(SystemExit, match="clip 0 entries"):
        same_codes_check(changed(lambda c: c["clips"][0].update(entries=["other"])), full, first=2)
    with pytest.raises(SystemExit, match="clip 0: no motion"):
        same_codes_check(changed(lambda c: c["clips"][0]["motion"].pop("core_crossfade")), full, first=2)
    with pytest.raises(SystemExit, match="settings.checkpoint_key"):
        same_codes_check(changed(lambda c: c["settings"].update(checkpoint_key="other.pt")), full, first=2)
    with pytest.raises(SystemExit, match="another sentence set"):
        same_codes_check(changed(lambda c: c["settings"]["round20"]["sentences"].update(sha256="0" * 64)), full, first=2)
    with pytest.raises(SystemExit, match="--first 3"):                           # more clips than the smoke has
        same_codes_check(smoke, full, first=3)
    with pytest.raises(SystemExit, match="clip 1: not in the stored run"):
        same_codes_check(smoke, dict(full, clips=full["clips"][:1]), first=2)
    with pytest.raises(SystemExit, match="no word"):
        same_codes_check(changed(lambda c: [r.update(entries=[]) for r in c["clips"]]),
                         dict(full, clips=[dict(r, entries=[]) for r in full["clips"]]), first=2)
    nan = changed(lambda c: c["clips"][0]["motion"]["core_crossfade"].update(wrist_spread=float("nan")))
    assert same_codes_check(nan, json.loads(json.dumps(nan)), first=2)["same"]   # NaN read twice is the same value
    with pytest.raises(SystemExit, match="no_decoder_gains"):                    # an ungained run never passes as a0
        same_codes_check(off, full, first=2)
    unmarked = json.loads(json.dumps(off))
    del unmarked["settings"]["round20"]["no_decoder_gains"]
    with pytest.raises(SystemExit, match="motion.core_crossfade"):               # and the motion itself tells
        same_codes_check(unmarked, full, first=2)
    root = ["--store", f"dir:{tmp_path / 'store'}", "--first", "2"]
    assert compare_main(["same-codes", *root, "--a", "e/smoke.json", "--b", "e/full.json"]) == 0
    with pytest.raises(SystemExit, match="not the same codes") as e:
        compare_main(["same-codes", *root, "--a", "e/off.json", "--b", "e/full.json"])
    assert e.value.code not in (0, None)                                         # the pod script stops on this


def phase(tmp_path, *, vocab="r10", gpu=True, same=True, stored=True, smoke=False, done=False):
    """The real phase script on stubs: every python run logged, the store and the pod's paths replaced."""
    import os
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    log = tmp_path / "commands.log"
    log.write_text("")
    (stubs / "python").write_text('#!/bin/bash\ncase "$1" in scripts/*) echo "$*" >> "$STUB_LOG" ;; esac\n'
                                  'case "$*" in *same-codes*) [ "$STUB_SAME" = 1 ] || exit 1 ;; esac\nexit 0\n')
    (stubs / "uv").write_text("#!/bin/bash\nexit 0\n")
    smi = stubs / "nvidia-smi"
    smi.write_text('#!/bin/bash\necho "GPU 0: Test RTX 4090 (UUID: GPU-x)"\n')
    for f in stubs.iterdir():
        f.chmod(0o755)
    if not gpu:
        smi.unlink()
    text = (ROOT / "runpod/fleet/round20_phase_a.sh").read_text()
    text = text.replace("source /root/r2env.sh\n", "").replace("cd /root/work/Sign-Translator\n", "")
    text = text.replace("/root/venv/bin/python", "python")
    text = re.sub(r"^sha\(\) .*$", "sha() { echo stubsha; }", text, flags=re.M)
    text = re.sub(r"^pin\(\) .*$", "pin() { :; }", text, flags=re.M)
    # the stored a0 shard exists (or not), and the stored a7 report when done
    text = re.sub(r"^exists\(\) .*$", 'exists() { case "$1" in *phase-a/a0-w14b-r10-shard0.json) [ "$STUB_STORED" = 1 ] ;; '
                  '*phase-a/a7-nogains-r10.json) [ "$STUB_DONE" = 1 ] ;; *) return 1 ;; esac; }', text, flags=re.M)
    script = tmp_path / "phase.sh"
    script.write_text(text)
    env = dict(os.environ, PATH=f"{stubs}:/usr/bin:/bin", THREADS="1", R20_PHASE="a7", VOCAB=vocab, STUB_LOG=str(log),
               STUB_SAME=str(int(same)), STUB_STORED=str(int(stored)), STUB_DONE=str(int(done)))
    if smoke:
        env["SMOKE"] = "1"
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    return run.returncode, run.stdout, log.read_text().splitlines()


def test_the_a7_phase_proves_the_codes_before_the_full_run(tmp_path):
    code, out, commands = phase(tmp_path)
    assert code == 0 and "R20 a7 GPU: GPU 0: Test RTX 4090" in out and "R20-PHASE-A-DONE 0" in out
    run = [c for c in commands if "round15_compose_eval.py run" in c]
    assert len(run) == 2
    a0, a7 = run
    stamp = re.search(r"phase-a/smoke/(\d{8}T\d{6}Z)/a0-w14b-r10-shard0\.json", a0)
    assert stamp and a0.index("--clips 500") < a0.index("--clips 5 ")                 # the last --clips wins: the smoke
    assert "--no-decoder-gains" not in a0 and "--lowpass-report-hz 6" in a0
    assert a7.endswith("--no-decoder-gains --output-key "
                       "acquisition/2026-10-17-v1/youtube-asl-v1/round20/phase-a/a7-nogains-r10-shard0.json")
    assert a7.replace(" --no-decoder-gains", "").split(" --output-key")[0] == \
        a0.replace(" --clips 5 ", " ").split(" --output-key")[0]                        # a0's arguments, plus the option
    check = next(k for k, c in enumerate(commands) if "same-codes" in c)
    assert commands.index(a0) < check < commands.index(a7)
    assert (f"--a acquisition/2026-10-17-v1/youtube-asl-v1/round20/phase-a/smoke/{stamp[1]}/a0-w14b-r10-shard0.json "
            "--b acquisition/2026-10-17-v1/youtube-asl-v1/round20/phase-a/a0-w14b-r10-shard0.json "
            "--form core_crossfade --first 5") in commands[check]
    # a mismatch: an ALERT naming the GPU, and nothing further runs
    code, out, commands = phase(tmp_path, same=False)
    assert code == 1 and "ALERT" in out and "Test RTX 4090" in out and "R20-PHASE-A-DONE 1" in out
    assert not any("a7-nogains" in c for c in commands)
    # no stored a0 shard: no smoke, no run
    code, out, commands = phase(tmp_path, stored=False)
    assert code == 1 and "ALERT" in out and "Test RTX 4090" in out and "round15_compose_eval.py" not in "".join(commands)
    # a CPU fallback dies before anything runs, a smoke may use it and needs no stored a0
    code, out, commands = phase(tmp_path, gpu=False)
    assert code == 1 and "needs a GPU" in out and "round15_compose_eval.py" not in "".join(commands)
    code, out, commands = phase(tmp_path, gpu=False, smoke=True, stored=False)
    assert code == 0 and not any("same-codes" in c for c in commands)
    assert any("/smoke/" in c and "a7-nogains-r10-shard0" in c and "--no-decoder-gains" in c for c in commands)
    # only r10, smoke or not
    for smoke in (False, True):
        for vocab in ("r14b-fraction", "r14b-activity"):
            code, out, commands = phase(tmp_path, vocab=vocab, smoke=smoke)
            assert code == 1 and "VOCAB must be r10" in out and not commands
    # a7 already stored: no second smoke
    code, out, commands = phase(tmp_path, done=True)
    assert code == 0 and "stored:" in out and not commands


# ---- the spotting guard, and the review's small refusals --------------------------------------------------------------

def test_the_spotting_guard_reads_the_shards_and_leaves_the_rule_alone(tmp_path, stage2_store, likelihood_stub):  # noqa: F811
    from scripts.round20_compare import compare_forms, main as compare_main
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, texts = put_set(store, sha)
    a0 = ["--sentences-key", "sets/d20.json", "--sentences-sha256", ssha, "--likelihood"]
    assert ce.main(run_args(tmp_path, sha, *a0, "--output-key", "e/a0-shard0.json")) == 0
    assert ce.main(run_args(tmp_path, sha, *a0, "--no-decoder-gains", "--output-key", "e/a7-shard0.json")) == 0
    merge(tmp_path, "e/a0-shard0.json", "e/a0.json")
    merge(tmp_path, "e/a7-shard0.json", "e/a7.json")
    s0, s7, m0, m7 = (json.loads(store.get(f"e/{k}.json")) for k in ("a0-shard0", "a7-shard0", "a0", "a7"))
    recall = lambda s, f: [r["spotting"][f]["own"]["recall"] for r in s["clips"]]  # noqa: E731
    out = compare_forms(m7, "core_crossfade", m0, "core_crossfade", boot=50, shards_a=[s7], shards_b=[s0])
    g = out["spotting_guard"]
    assert g["available"] and g["recogniser"] == "sign-recognizer-v2" and g["word_source"] == ["generated"]
    assert g["difference"] == lib.paired_mean(recall(s7, "core_crossfade"), recall(s0, "core_crossfade"), boot=50)
    assert g["a"] == lib.paired_mean(recall(s7, "core_crossfade"), boot=50)
    assert out["spotting_guard_lower_bound"] == g["difference"]["ci95"][0] and "same word source" in g["note"]
    # the adoption rule does not read it, and no shards simply means no guard
    plain = compare_forms(m7, "core_crossfade", m0, "core_crossfade", boot=50)
    assert plain["both_improve"] == out["both_improve"] == (
        out["bleu"]["difference"] > 0 and out["likelihood_minus_blank"]["difference"]["difference"] > 0)
    assert not plain["spotting_guard"]["available"] and plain["spotting_guard_lower_bound"] is None
    # known recalls: the own words' recall, paired over the sentences both have (a None drops that sentence)
    def with_recalls(shard, own, other):
        rows = json.loads(json.dumps(shard["clips"]))
        for r, x, y in zip(rows, own, other):
            r["spotting"]["core_crossfade"] = dict(r["spotting"]["core_crossfade"], own=dict(recall=x),
                                                   mismatched=dict(recall=y))
        return dict(shard, clips=rows)
    known = compare_forms(m7, "core_crossfade", m0, "core_crossfade", boot=50,
                          shards_a=[with_recalls(s7, [1., .5, None, 0.], [0.] * 4)],
                          shards_b=[with_recalls(s0, [.5, .5, 1., 1.], [1.] * 4)])["spotting_guard"]
    assert known["difference"]["difference"] == pytest.approx((.5 + 0. - 1.) / 3, abs=1e-4) and known["difference"]["n"] == 3
    # a form better on BLEU and likelihood stays adopted whatever the guard reads, or if there is none
    better = json.loads(json.dumps(m7))
    better["forms"]["core_crossfade"]["hypotheses"] = list(better["references"])
    better["likelihood"]["per_form"]["core_crossfade"] = [
        [x + 1. if x is not None else None for x in row] if isinstance(row, list) else row + 1.
        for row in m0["likelihood"]["per_form"]["core_crossfade"]]
    bad = [with_recalls(s7, [0.] * 4, [0.] * 4)], [with_recalls(s0, [1.] * 4, [0.] * 4)]
    worse = compare_forms(better, "core_crossfade", m0, "core_crossfade", boot=50, shards_a=bad[0], shards_b=bad[1])
    assert worse["both_improve"] and worse["spotting_guard_lower_bound"] == -1.
    assert compare_forms(better, "core_crossfade", m0, "core_crossfade", boot=50)["both_improve"]
    # the checkpoint and mix keys are reported, not compared
    assert out["a"]["checkpoint_key"] == out["b"]["checkpoint_key"] == "prior.pt"
    assert out["a"]["mix_key"] == "mix.json" and out["a"]["generator_mix_key"] == "mix.json"
    other = dict(m0, settings=dict(m0["settings"], checkpoint_key="other.pt"))
    assert compare_forms(m7, "core_crossfade", other, "core_crossfade", boot=20)["b"]["checkpoint_key"] == "other.pt"
    # withheld across word sources, for other recognisers and for forms without per-clip spotting
    cross = compare_forms(m7, "core_crossfade", m0, "oracle_stage2", boot=20, shards_a=[s7], shards_b=[s0])
    assert not cross["spotting_guard"]["available"] and "word sources" in cross["spotting_guard"]["reason"]
    assert cross["spotting_guard_lower_bound"] is None and "difference" not in cross["spotting_guard"]
    v3 = dict(m0, settings=dict(m0["settings"], recognizer_run="sign-recognizer-v3d"))
    assert "not sign-recognizer-v2" in compare_forms(m7, "core_crossfade", v3, "core_crossfade", boot=20, shards_a=[s7],
                                                     shards_b=[s0])["spotting_guard"]["reason"]
    assert not compare_forms(m7, "core_crossfade/shuffled", m0, "core_crossfade/shuffled", boot=20, shards_a=[s7],
                             shards_b=[s0])["spotting_guard"]["available"]
    with pytest.raises(SystemExit, match="not this report's"):                   # shards of other clips
        compare_forms(m7, "core_crossfade", m0, "core_crossfade", boot=20, shards_a=[s7],
                      shards_b=[dict(s0, clips=list(reversed(s0["clips"]))[1:])])
    with pytest.raises(SystemExit, match="not this report's"):                   # or of another run
        compare_forms(m7, "core_crossfade", m0, "core_crossfade", boot=20, shards_a=[s0], shards_b=[s0])
    # the command line: shards by the phase script's names, by given keys, or none
    root = ["forms", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "20"]
    pair = ["--a", "e/a7.json:core_crossfade", "--b", "e/a0.json:core_crossfade"]
    assert compare_main([*root, *pair, "--output-file", str(tmp_path / "p1.json")]) == 0
    assert json.loads((tmp_path / "p1.json").read_text())["spotting_guard"]["available"]
    store.put("e/copy.json", store.get("e/a0-shard0.json"))
    store.put("e/b.json", store.get("e/a0.json"))
    swapped = ["--a", "e/a7.json:core_crossfade", "--b", "e/b.json:core_crossfade"]
    assert compare_main([*root, *swapped, "--output-file", str(tmp_path / "p2.json")]) == 0
    p2 = json.loads((tmp_path / "p2.json").read_text())
    assert not p2["spotting_guard"]["available"] and "not found" in p2["spotting_guard"]["reason"]
    assert compare_main([*root, *swapped, "--b-shard-keys", "e/copy.json", "--output-file", str(tmp_path / "p3.json")]) == 0
    assert json.loads((tmp_path / "p3.json").read_text())["spotting_guard"]["available"]


def test_the_refiner_and_mixed_caches_are_refused(tmp_path, stage2_store, capsys):  # noqa: F811
    from scripts.round20_compare import compare_words
    store, sha, *_ = stage2_store
    put_gains(store)
    with pytest.raises(SystemExit):
        ce.main(run_args(tmp_path, sha, "--clips", "5", "--no-decoder-gains", "--refiner-checkpoint-key", "refiner.pt",
                         "--output-file", str(tmp_path / "o.json")))
    assert "the refiner's post reapplies" in capsys.readouterr().err
    row = dict(entry="cut", label="CUT", probabilities=[.5], ranks=[1], chosen={"1": dict(index=0)})
    index = lambda **s: dict(settings=dict(checkpoint_key="p.pt", choose=[1], recognizer_run="v3d", **s),  # noqa: E731
                             entries=[row])
    with pytest.raises(SystemExit, match="without the decoder gains"):
        compare_words(index(no_decoder_gains=True), index(), boot=10)
    with pytest.raises(SystemExit, match="without the decoder gains"):
        compare_words(index(), index(no_decoder_gains=True), boot=10)
    assert compare_words(index(no_decoder_gains=True), index(no_decoder_gains=True), boot=10)["entries"] == 1
    assert compare_words(index(), index(), boot=10)["entries"] == 1
