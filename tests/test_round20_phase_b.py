"""Round 20 Phase B: the pod job's argument lists on stubs (the comparator's settings apart from the adopted options), its
refusals, and the verdict script: each rule item, each missing field, on tiny reports and on real merges of the round-15
synthetic store. Nothing is read from R2."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import pytest

import scripts.round15_compose_eval as ce
import scripts.round20_phase_b_verdict as pv
import scripts.round20_word_cache as cw
from signtranslator.generation.motion_to_text import corpus_bleu
from tests.test_round15_compose_eval import pilot_report, run_args
from tests.test_round15_stage2 import stage2_store, generator_store  # noqa: F401  (fixtures)
from tests.test_round15_compositional import stubbed  # noqa: F401
from tests.test_round20_eval import likelihood_stub, put_gains  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runpod/fleet/round20_phase_b.sh"
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
OUT = f"{O}/round20/phase-b"
CMP = f"{O}/round15/eval1000/w14b-step30000.json"
W14B = "training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt"
R16 = "training-runs/tokens-rvq-d0b-prior-r16-rollout/checkpoints/step-00010000.pt"
FORM = "core_crossfade"
# read from the stored comparator's settings (2026-10-31)
COMPARATOR = dict(
    clips=1000, shards=4, block=32, seed=0, variants=[FORM],
    mix_key=f"{O}/round10-mixed-v1/manifest.json",
    mix_sha256="ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a",
    corpora=["asllvd-text-v1", "rit-text-r6", "wlasl-text-r6", "semlex-text-v1"], selection="stage1", timing="sign-fitted",
    core_margin=.1, total_fit="pad", guidance=1.5, part_temperature=dict(body=1.1, left_hand=1.1, right_hand=1.1, face=1.),
    checkpoint_key=W14B,
    generator_mix=dict(key=f"{O}/round14b-mixed-v1/manifest.json",
                       sha256="710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d",
                       check=dict(evaluation_records=142124, generator_records=177946, generator_only=35822)),
    model_sha256="42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21",
    translator_key="training-runs/back-translator-kp2d-uwb-ft-v1/checkpoints/step-00009000.pt",
    recognizer_run="sign-recognizer-v2", recognizer_checkpoint="training-runs/sign-recognizer-v2/checkpoints/step-00027000.pt")


# ---- the job on stubs --------------------------------------------------------------------------------------------------

def job(tmp_path, phase, *, prior="w14b", no_gains="1", best_of="8", sentence_set="val1000", smoke=False, gpu=True, gate=True,
        stored=(), fail_shard="", hang=False, **env):
    """The real job on stubs: every scripts/ run and pin logged, the store and the pod's paths replaced. Returns
    (exit code, stdout, [logged lines])."""
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    log = tmp_path / "commands.log"
    log.write_text("")
    (stubs / "python").write_text(
        '#!/bin/bash\n'
        'case "$1" in scripts/*round20_word_cache*) echo "$* #OMP=$OMP_NUM_THREADS" >> "$STUB_LOG" ;;\n'
        '  scripts/*) echo "$*" >> "$STUB_LOG" ;; esac\n'
        'case "$*" in *c20-gate*) [ "$STUB_GATE" = 1 ] || exit 1 ;; esac\n'
        'case "$*" in *"round20_phase_b_verdict.py verdict "*) exit ${STUB_VERDICT:-0} ;; esac\n'
        'if [ -n "$STUB_FAIL_SHARD" ]; then case "$*" in *"--shard $STUB_FAIL_SHARD "*) exit 1 ;; esac; fi\n'
        'if [ -n "$STUB_HANG" ]; then case "$*" in *round20_word_cache*" run "*) exec /bin/sleep 30 ;; esac; fi\n'
        'exit 0\n')
    (stubs / "uv").write_text("#!/bin/bash\nexit 0\n")
    (stubs / "sleep").write_text("#!/bin/bash\nexit 0\n")
    smi = stubs / "nvidia-smi"
    smi.write_text('#!/bin/bash\necho "GPU 0: Test L40S (UUID: GPU-x)"\n')
    for f in stubs.iterdir():
        f.chmod(0o755)
    if not gpu:
        smi.unlink()
    text = SCRIPT.read_text()
    text = text.replace("source /root/r2env.sh\n", "").replace("cd /root/work/Sign-Translator\n", "")
    text = text.replace("/root/venv/bin/python", "python").replace("/root/r20b-", f"{tmp_path}/r20b-")
    text = re.sub(r"^sha\(\) .*$", "sha() { echo stubsha; }", text, flags=re.M)
    text = re.sub(r"^pin\(\) .*$", 'pin() { echo "pin $1" >> "$STUB_LOG"; }', text, flags=re.M)
    text = re.sub(r"^exists\(\) .*$", 'exists() { case " $STUB_EXISTS " in *" $1 "*) return 0 ;; esac; return 1; }', text,
                  flags=re.M)
    script = tmp_path / "job.sh"
    script.write_text(text)
    e = dict(os.environ, PATH=f"{stubs}:/usr/bin:/bin", THREADS="8", R20B_PHASE=phase, PRIOR=prior, NO_GAINS=no_gains,
             BEST_OF=best_of, SET=sentence_set, STUB_LOG=str(log), STUB_GATE=str(int(gate)),
             STUB_EXISTS=" ".join(stored), STUB_FAIL_SHARD=fail_shard, STUB_HANG="1" if hang else "", **env)
    e.pop("SMOKE", None)
    if smoke:
        e["SMOKE"] = "1"
    run = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=60)
    return run.returncode, run.stdout, log.read_text().splitlines()


def scripts_run(commands):
    return [c for c in commands if c.startswith("scripts/")]


def parsed_run(tmp_path, monkeypatch, line, name):
    """A logged ``round15_compose_eval.py run`` line through the script's own parser and its refusals."""
    seen = {}
    monkeypatch.setattr(ce, "run", lambda args: seen.update(vars(args)) or {})
    argv = line.split()[1:] + ["--store", f"dir:{tmp_path / ('s-' + name)}"]
    assert ce.main(argv) == 0
    return seen


def comparator_command() -> str:
    """The declared run's command, from the job that reran it (without that job's second translator)."""
    text = (ROOT / "runpod/fleet/state_back_translator.sh").read_text()
    comp = text.split('COMP="')[1].split('"')[0].replace("\\\n", " ")
    comp = comp.replace(" --also-translator-run $RUN", "").replace("$O", O)
    return comp.replace("$MIX", COMPARATOR["generator_mix"]["key"]).replace("$MSHA", COMPARATOR["generator_mix"]["sha256"])


def test_the_job_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_the_declared_command_has_the_stored_reports_settings(tmp_path, monkeypatch):
    c = parsed_run(tmp_path, monkeypatch, f"scripts/round15_compose_eval.py run {comparator_command()} --shard 0 --output-key x.json", "c")
    for k in ("clips", "shards", "block", "seed", "mix_key", "mix_sha256", "selection", "timing", "core_margin", "total_fit",
              "guidance", "checkpoint_key"):
        assert c[k] == COMPARATOR[k], k
    assert c["variants"] == FORM and c["expect_model_sha256"] == COMPARATOR["model_sha256"]
    assert c["generator_mix_key"] == COMPARATOR["generator_mix"]["key"]
    assert c["recognizer_run"] == COMPARATOR["recognizer_run"]
    assert c["translator_run"] in COMPARATOR["translator_key"]
    assert {k: float(v) for k, v in (x.split("=") for x in c["part_temperature"].split(","))} == COMPARATOR["part_temperature"]


@pytest.mark.parametrize("prior,no_gains,best_of,changed", [
    ("w14b", "1", "0", {"no_decoder_gains": True}),
    ("r16", "0", "0", {"checkpoint_key": R16}),
    ("r16", "1", "0", {"checkpoint_key": R16, "no_decoder_gains": True}),
    ("w14b", "0", "4", {"checkpoint_key": None, "word_workers": 1, "word_cache_n": 4}),
    ("w14b", "1", "8", {"checkpoint_key": None, "word_workers": 1, "word_cache_n": 8}),
    ("r16", "0", "16", {"checkpoint_key": None, "word_workers": 1, "word_cache_n": 16}),
    ("r16", "1", "4", {"checkpoint_key": None, "word_workers": 1, "word_cache_n": 4}),
])
def test_the_run_is_the_comparators_command_apart_from_the_adopted_options(tmp_path, monkeypatch, prior, no_gains, best_of,
                                                                          changed):
    gains = "nogains" if no_gains == "1" else "gains"
    index = f"{OUT}/cache-{prior}-{gains}-val1000-n{best_of}/index.json"
    code, out, commands = job(tmp_path, "run", prior=prior, no_gains=no_gains, best_of=best_of, stored=[index])
    assert code == 0 and "R20-PHASE-B-DONE 0" in out and "Test L40S" in out
    runs = [c for c in commands if "round15_compose_eval.py run" in c]
    report = f"{OUT}/{prior}-{gains}-n{best_of}-val1000"
    assert [c.split("--output-key ")[1] for c in runs] == [f"{report}-shard{s}.json" for s in range(4)]
    base = parsed_run(tmp_path, monkeypatch, f"scripts/round15_compose_eval.py run {comparator_command()} --shard 0 --output-key x.json", "base")
    for s, line in enumerate(runs):
        arm = parsed_run(tmp_path, monkeypatch, line, f"a{s}")
        expected = dict(base, shard=s, output_key=f"{report}-shard{s}.json", likelihood=True, store=arm["store"], **changed)
        if best_of != "0":
            expected.update(word_cache_key=index, word_cache_sha256="stubsha")
        assert {k: (arm[k], expected[k]) for k in arm if arm[k] != expected[k]} == {}
    merge = next(c for c in commands if "round15_compose_eval.py merge" in c)
    pilot = f"{O}/round14b/pilot-w14b-step30000.json" if prior == "w14b" else f"{O}/round16/pilot-r16.json"
    assert merge == ("scripts/round15_compose_eval.py merge --shard-keys " + ",".join(f"{report}-shard{s}.json" for s in range(4))
                     + f" --end-to-end-report {pilot} --paired-with {CMP} --rule bleu:end_to_end,bleu:mismatched_words,"
                     f"spotting --output-key {report}.json")
    pins = [c for c in commands if c.startswith("pin ")]
    assert f"pin {CMP}" in pins and f"pin {pilot}" in pins
    assert (f"pin {R16 if prior == 'r16' else W14B}" in pins) == (best_of == "0")       # only where words are generated
    assert all(f"pin {CMP[:-5]}-shard{s}.json" in pins for s in range(4))
    assert not any("round20/c20" in c for c in commands)      # the 1,000's run reads nothing of C20


@pytest.mark.parametrize("prior,no_gains,best_of", [("w14b", "0", "4"), ("w14b", "1", "8"), ("r16", "0", "16"), ("r16", "1", "4")])
def test_the_cache_phase(tmp_path, monkeypatch, prior, no_gains, best_of):
    gains = "nogains" if no_gains == "1" else "gains"
    code, out, commands = job(tmp_path, "cache", prior=prior, no_gains=no_gains, best_of=best_of)
    assert code == 0 and "R20-PHASE-B-DONE 0" in out and f"pin {R16 if prior == 'r16' else W14B}" in commands
    prefix = f"{OUT}/cache-{prior}-{gains}-val1000-n{best_of}"
    runs = [c for c in commands if "round20_word_cache.py run" in c]
    assert sorted(re.search(r"--shard (\d)", c)[1] for c in runs) == ["0", "1", "2", "3"]
    assert all(c.endswith("#OMP=2") for c in runs)                         # 8 threads over 4 processes, as Phase A
    for k, line in enumerate(runs):
        seen = {}
        monkeypatch.setattr(cw, "run", lambda args, seen=seen: seen.update(vars(args)) or {})
        argv = line.split(" #OMP")[0].split()[1:]
        assert cw.main(argv + ["--store", f"dir:{tmp_path / f's{k}'}"]) == 0
        assert seen["prefix"] == prefix and seen["samples"] == int(best_of) and seen["choose"] == best_of
        assert seen["checkpoint_key"] == (R16 if prior == "r16" else W14B) and seen["clips"] == 1000
        assert seen["mix_key"] == COMPARATOR["mix_key"] and seen["vocabulary_mix_key"] is None and seen["sentences_key"] is None
        assert seen["no_decoder_gains"] == (no_gains == "1") and seen["recognizer_run"] == "sign-recognizer-v3d"
        assert seen["generator_mix_sha256"] == COMPARATOR["generator_mix"]["sha256"] and seen["shards"] == 4
    merge = [c for c in commands if "round20_word_cache.py merge" in c]
    assert len(merge) == 1 and f"--prefix {prefix} --shards 4" in merge[0]
    # stored shards are not rerun; a stored index stops at once
    code, out, commands = job(tmp_path, "cache", prior=prior, no_gains=no_gains, best_of=best_of,
                              stored=[f"{prefix}/shard-1-of-4.json"])
    assert code == 0 and sorted(re.search(r"--shard (\d)", c)[1] for c in commands if " run " in c) == ["0", "2", "3"]
    code, out, commands = job(tmp_path, "cache", prior=prior, no_gains=no_gains, best_of=best_of, stored=[f"{prefix}/index.json"])
    assert code == 0 and "stored:" in out and not scripts_run(commands)
    # CACHE_PROCS sets the processes and the threads' share
    code, out, commands = job(tmp_path, "cache", prior=prior, no_gains=no_gains, best_of=best_of, CACHE_PROCS="2")
    assert code == 0 and all(c.endswith("#OMP=4") for c in commands if " run " in c)
    assert sum(" run " in c for c in commands) == 2 and f"--prefix {prefix} --shards 2" in commands[-1]


def test_a_failed_cache_shard_stops_the_others(tmp_path):
    code, out, commands = job(tmp_path, "cache", fail_shard="2")
    assert code == 1 and "a cache shard failed" in out and "R20-PHASE-B-DONE 1" in out
    assert not any("round20_word_cache.py merge" in c for c in commands)
    start = time.time()
    code, out, commands = job(tmp_path, "cache", fail_shard="0", hang=True)         # the others would sleep 30 s
    assert code == 1 and "a cache shard failed" in out and time.time() - start < 20


@pytest.mark.parametrize("kwargs,message", [
    (dict(best_of="0"), "BEST_OF=0 uses no cache"),
    (dict(prior="f12"), "PRIOR must be"),
    (dict(prior=""), "PRIOR must be"),
    (dict(no_gains="2"), "NO_GAINS must be"),
    (dict(no_gains=""), "NO_GAINS must be"),
    (dict(best_of="2"), "BEST_OF must be"),
    (dict(sentence_set="d20"), "SET must be"),
])
def test_invalid_combinations_are_refused_before_anything_runs(tmp_path, kwargs, message):
    code, out, commands = job(tmp_path, "cache", **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "R20-PHASE-B-DONE 1" in out and not commands


def test_other_refusals(tmp_path):
    code, out, commands = job(tmp_path, "nothing")
    assert code == 1 and "R20B_PHASE must be" in out and not commands
    for phase in ("run", "verdict"):       # the comparator itself is no arm
        code, out, commands = job(tmp_path, phase, prior="w14b", no_gains="0", best_of="0")
        assert code == 1 and "no lever adopted" in out and not commands
    code, out, commands = job(tmp_path, "verdict", smoke=True)
    assert code == 1 and "no smoke" in out and not commands
    # a full run needs the comparator's kind of device, a smoke does not
    for phase in ("cache", "run"):
        code, out, commands = job(tmp_path, phase, gpu=False, stored=[f"{OUT}/cache-w14b-nogains-val1000-n8/index.json"])
        assert code == 1 and "needs a GPU" in out and not scripts_run(commands)
    # no cache stored: the run says which, and runs nothing
    code, out, commands = job(tmp_path, "run")
    assert code == 1 and "run the cache phase first" in out and not scripts_run(commands)
    code, out, commands = job(tmp_path, "verdict")
    assert code == 1 and "run the run phase first" in out and not scripts_run(commands)


def test_c20_stays_closed_until_a_stored_verdict_opens_it(tmp_path):
    verdict = f"{OUT}/verdict-w14b-nogains-n8-val1000.json"
    for phase in ("cache", "run", "verdict"):
        code, out, commands = job(tmp_path, phase, sentence_set="c20")               # no stored verdict
        assert code == 1 and "no stored val1000 verdict" in out and not commands
        code, out, commands = job(tmp_path, phase, sentence_set="c20", stored=[verdict], gate=False)  # below 2.55 or failed
        assert code == 1 and "does not open it" in out
        assert commands == [f"scripts/round20_phase_b_verdict.py c20-gate --verdict-key {verdict}"]   # nothing of C20
    code, out, commands = job(tmp_path, "cache", sentence_set="c20", stored=[verdict], smoke=True)    # nor by a smoke
    code2, out2, _ = job(tmp_path, "cache", sentence_set="c20", smoke=True)
    assert code == 0 and code2 == 1 and "no stored val1000 verdict" in out2
    # another arm's verdict does not open it
    code, out, commands = job(tmp_path, "cache", prior="r16", sentence_set="c20", stored=[verdict])
    assert code == 1 and "no stored val1000 verdict" in out and not commands
    # open: the gate first, then the pin, then the set; the comparator and the pilots cannot pair with it
    c20 = f"--sentences-key {O}/round20/c20.json --sentences-sha256 24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956"
    code, out, commands = job(tmp_path, "cache", sentence_set="c20", stored=[verdict])
    assert code == 0 and commands[0].endswith(f"c20-gate --verdict-key {verdict}") and commands[1] == f"pin {O}/round20/c20.json"
    assert all(c20 in c and f"--prefix {OUT}/cache-w14b-nogains-c20-n8 " in c for c in commands if " run " in c)
    index = f"{OUT}/cache-w14b-nogains-c20-n8/index.json"
    code, out, commands = job(tmp_path, "run", sentence_set="c20", stored=[verdict, index])
    runs = [c for c in commands if "round15_compose_eval.py run" in c]
    assert code == 0 and len(runs) == 4 and all(c20 in c and "--stage1-report-key" not in c for c in runs)
    assert all("--clips 1000" in c and f"--output-key {OUT}/w14b-nogains-n8-c20-shard" in c for c in runs)
    merge = next(c for c in commands if "round15_compose_eval.py merge" in c)
    assert "--paired-with" not in merge and "--end-to-end-report" not in merge
    code, out, commands = job(tmp_path, "verdict", sentence_set="c20", stored=[verdict, f"{OUT}/w14b-nogains-n8-c20.json"])
    line = next(c for c in commands if " verdict " in c)
    assert code == 0 and f"--val1000-verdict-key {verdict}" in line and "--end-to-end-key" not in line
    assert f"--set c20 --output-key {OUT}/verdict-w14b-nogains-n8-c20.json" in line


def test_a_smoke_reads_d20_only_with_write_once_keys_of_its_own(tmp_path):
    d20 = f"--sentences-key {O}/round20/d20.json"
    code, out, commands = job(tmp_path, "cache", smoke=True, gpu=False, best_of="16")
    runs = [c for c in commands if " run " in c]
    assert code == 0 and len(runs) == 4 and all(d20 in c and "--clips 5" in c and "--samples 4 --choose 4" in c for c in runs)
    assert all(re.search(rf"--prefix {OUT}/smoke/\d{{8}}T\d{{6}}Z/cache-w14b-nogains-val1000-n16 ", c) for c in runs)
    code, out, commands = job(tmp_path, "run", smoke=True, gpu=False)
    assert code == 1 and "give CACHE_KEY" in out
    code, out, commands = job(tmp_path, "run", smoke=True, gpu=False, CACHE_KEY="k/index.json", stored=["k/index.json"])
    run = [c for c in commands if "round15_compose_eval.py run" in c]
    merge = next(c for c in commands if "round15_compose_eval.py merge" in c)
    assert code == 0 and len(run) == 1 and d20 in run[0] and "--clips 5 " in run[0] and "--stage1-report-key" not in run[0]
    assert "--word-cache-key k/index.json" in run[0] and "--word-cache-n 4" in run[0] and "--shards 1 --shard 0" in run[0]
    assert "/smoke/" in run[0] and "--paired-with" not in merge and "--end-to-end-report" not in merge
    assert not any("round20/c20" in c for c in commands)
    # fresh words: the option goes to the run; the option is not given a cached run (its gains are the cache's)
    code, out, commands = job(tmp_path, "run", smoke=True, gpu=False, best_of="0")
    assert code == 0 and "--no-decoder-gains" in next(c for c in commands if " run " in c)


def test_written_once(tmp_path):
    report = f"{OUT}/w14b-nogains-n8-val1000"
    index = f"{OUT}/cache-w14b-nogains-val1000-n8/index.json"
    code, out, commands = job(tmp_path, "run", stored=[index, f"{report}.json"])
    assert code == 0 and "stored:" in out and not scripts_run(commands)
    code, out, commands = job(tmp_path, "run", stored=[index, f"{report}-shard1.json"])
    runs = [c for c in commands if "round15_compose_eval.py run" in c]
    assert len(runs) == 3 and not any("--shard 1 " in c for c in runs)
    assert f"{report}-shard1.json" in next(c for c in commands if " merge " in c)
    code, out, commands = job(tmp_path, "verdict", stored=[f"{OUT}/verdict-w14b-nogains-n8-val1000.json"])
    assert code == 0 and "stored:" in out and not scripts_run(commands)


@pytest.mark.parametrize("prior,pilot", [("w14b", "round14b/pilot-w14b-step30000.json"), ("r16", "round16/pilot-r16.json")])
def test_the_verdict_phase_picks_the_priors_own_end_to_end(tmp_path, prior, pilot):
    arm = f"{OUT}/{prior}-gains-n4-val1000.json"
    code, out, commands = job(tmp_path, "verdict", prior=prior, no_gains="0", best_of="4", stored=[arm])
    line = next(c for c in commands if " verdict " in c)
    assert code == 0 and f"--end-to-end-key {O}/{pilot} " in line and f"--arm-key {arm} " in line
    assert f"--comparator-key {CMP} " in line and f"--output-key {OUT}/verdict-{prior}-gains-n4-val1000.json" in line
    assert f"--prior {prior} --prior-checkpoint-key {R16 if prior == 'r16' else W14B} --no-gains 0 --best-of 4 --set val1000" in line
    # the script's own parser takes the list: it reaches the comparator's hash
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'p'}")
    for key in (arm, CMP):
        store.put(key, b"{}")
    with pytest.raises(SystemExit, match="does not match --comparator-sha256"):
        pv.main(line.split()[1:] + ["--store", f"dir:{tmp_path / 'p'}"])


@pytest.mark.parametrize("exit_code", [1, 2])
def test_a_verdict_that_stores_nothing_is_an_alert(tmp_path, exit_code):
    code, out, commands = job(tmp_path, "verdict", stored=[f"{OUT}/w14b-nogains-n8-val1000.json"],
                              STUB_VERDICT=str(exit_code))
    assert code == 1 and "ALERT" in out and "nothing stored" in out and "R20-PHASE-B-DONE 1" in out
    code, out, commands = job(tmp_path, "verdict", stored=[f"{OUT}/w14b-nogains-n8-val1000.json"])
    assert code == 0 and f"R20B verdict {OUT}/verdict-w14b-nogains-n8-val1000.json stubsha" in out


# ---- the verdict on tiny reports ----------------------------------------------------------------------------------------

REFS = ["the cat sat on the mat today", "a dog ran in the big park", "she reads a good book at night",
        "we walk to the old town", "he made a warm soup for all", "they sing a song at the end"]
SIGNS = [f"sign{k}" for k in range(len(REFS))]
HYP_ARM = ["the cat sat on a rug now", "a dog ran to the big lake", "she reads a long book at noon",
           "we walk to the new town", "he made a cold soup for us", "they sing a tune at the end"]
HYP_CMP = ["the cat ran on a rug", "a dog sat in a lake", "she has a long book", "we go to the town", "he ate a cold soup",
           "they sing at the start"]
GOOD = dict(difference=.4, ci95=[.1, .7])


def hyp_form(hyps, **extra):
    return dict(bleu=round(corpus_bleu(hyps, REFS)["bleu"], 3), ci95=[0., 9.], hypotheses=list(hyps), **extra)


def shard(settings, delta=0., base=.3):
    return dict(settings=settings, clips=[dict(clip=k, sign=s, spotting={FORM: dict(own=dict(recall=base + delta))})
                                          for k, s in enumerate(SIGNS)])


def cache_settings(*, no_gains=True, n=8, checkpoint=W14B):
    return dict(key="c/index.json", sha256="0" * 64, n=n, checkpoint_key=checkpoint, scorer="sign-recognizer-v3d", samples=n,
                **({"no_decoder_gains": True} if no_gains else {}))


def round20(**over):
    return {**dict(sentences=None, vocabulary_mix=None, semlex_core="fraction", activity_margin=None, oracle_source="isolated",
                   gloss_manifest=None, retime="reference", tempo=1., bridge_velocity=None, lowpass_hz=None,
                   word_cache=cache_settings(), likelihood=True, predicted_duration=False, semlex_oracle_tracked=False,
                   lowpass_report_hz=None), **over}


def reports(hyp_arm=HYP_ARM, delta=0.):
    comparator = dict(kind="round15-compose-eval", settings=dict(COMPARATOR), references=REFS, signs=SIGNS, clips=len(REFS),
                      shards=[0], forms={FORM: hyp_form(HYP_CMP), "real": hyp_form(REFS), "blank": hyp_form(["the"] * 6)},
                      motion_ratio_to_real={FORM: dict(wrist_speed=2.6448)})
    settings = dict(COMPARATOR, rerank=1, checkpoint_key=None, round20=round20())
    pilot = dict(checkpoint_key=W14B, meaning=dict(references=REFS, generated=dict(bleu=1.786, hypotheses=["end"] * 6)))
    arm = dict(kind="round15-compose-eval", settings=settings, references=REFS, signs=SIGNS, clips=len(REFS), shards=[0],
               forms={FORM: hyp_form(hyp_arm, minus_paired_core_crossfade=dict(GOOD), minus_mismatched_words=dict(GOOD),
                                     minus_end_to_end=dict(GOOD)),
                      f"paired:{FORM}": hyp_form(HYP_CMP), "end_to_end": hyp_form(["end"] * 6), "real": hyp_form(REFS),
                      "blank": hyp_form(["the"] * 6)},
               comparators=dict(end_to_end=dict(key="p.json", checkpoint_key=W14B, real_identical=6, blank_identical=6),
                                paired_with=dict(key=CMP, variants=[FORM])),
               likelihood=dict(forms={FORM: dict(minus_blank=dict(difference=.02, ci95=[-.01, .05], n=6)),
                                      "real": dict(minus_blank=dict(difference=.3, ci95=[.2, .4], n=6))}),
               motion_ratio_to_real={FORM: dict(wrist_speed=2.1), "real": dict(wrist_speed=1.)},
               round20=dict(per_variant={FORM: dict(sources=dict(cache=40))}))
    return dict(arm=arm, comparator=comparator, pilot=pilot, shards_arm=[shard(settings, delta)],
                shards_comparator=[shard(COMPARATOR)])


def judge(r, **over):
    args = dict(prior_checkpoint=W14B, no_gains=True, best_of=8, boot=50)
    return pv.judge_val1000(r["arm"], r["comparator"], r["pilot"], shards_arm=r["shards_arm"],
                            shards_comparator=r["shards_comparator"], **{**args, **over})


def test_a_passing_verdict_judges_each_item_and_reports_the_rest():
    v = judge(reports())
    assert v["rule_passed"] and v["complete"] and not v["missing"]
    assert set(v["items"]) == {"bleu_vs_w14b_composition", "bleu_vs_other_sentence_words", "bleu_vs_own_end_to_end",
                               "v2_spotting_vs_w14b_composition", "wrist_speed_reported"}
    assert all(i["available"] and i["passed"] for i in v["items"].values())
    assert v["items"]["bleu_vs_w14b_composition"]["lower_bound"] == .1
    assert v["items"]["wrist_speed_reported"] == dict(available=True, passed=True, ratio_to_real=2.1, against=2.64,
                                                     comparator_ratio_to_real=2.6448)
    assert v["items"]["v2_spotting_vs_w14b_composition"]["difference"] == 0.
    reads = v["next_to_bleu"]
    assert reads["length_ratio"][FORM] == 1. and reads["likelihood_minus_blank"]["difference"] == .02
    assert reads["real_likelihood_minus_blank"]["difference"] == .3
    assert "no A3 arm" in v["predicted"] and "@predicted" not in json.dumps(v["next_to_bleu"])
    assert v["comparator"]["paired_likelihood"]["available"] is False
    assert "without --likelihood" in v["comparator"]["paired_likelihood"]["reason"]
    assert v["comparator"]["real_hypotheses_identical"] == 6 and "note" in v["end_to_end"]
    assert not any("artefact" in k or "tempo" in k for k in v)        # no tempo-artefact logic
    gained = reports()
    gained["arm"]["settings"]["round20"]["word_cache"] = cache_settings(no_gains=False)
    assert "note" not in judge(gained, no_gains=False)["end_to_end"]   # a gained arm against the gained pilot: no caveat


@pytest.mark.parametrize("item,field", [("bleu_vs_w14b_composition", "minus_paired_core_crossfade"),
                                        ("bleu_vs_other_sentence_words", "minus_mismatched_words"),
                                        ("bleu_vs_own_end_to_end", "minus_end_to_end")])
@pytest.mark.parametrize("low,passes", [(.001, True), (0., False), (-.2, False)])
def test_each_bleu_item_fails_alone(item, field, low, passes):
    r = reports()
    r["arm"]["forms"][FORM][field] = dict(difference=.2, ci95=[low, .5])
    v = judge(r)
    assert v["complete"] and v["rule_passed"] is passes and v["items"][item]["passed"] is passes
    assert all(i["passed"] for name, i in v["items"].items() if name != item)


@pytest.mark.parametrize("delta,passes", [(-.019, True), (-.02, False), (-.05, False), (.1, True)])
def test_the_spotting_guard_fails_alone(delta, passes):
    v = judge(reports(delta=delta))
    item = v["items"]["v2_spotting_vs_w14b_composition"]
    assert item["passed"] is passes and v["rule_passed"] is passes and item["threshold"] == -.02 and item["lower_bound"] == pytest.approx(delta)
    assert item["recogniser"] == "sign-recognizer-v2" and item["word_source"] == ["generated"]


def missing(r, *path):
    x = r
    for k in path[:-1]:
        x = x[k]
    del x[path[-1]]
    return r


@pytest.mark.parametrize("edit,says", [
    (lambda r: missing(r, "arm", "forms", FORM, "minus_paired_core_crossfade"), "bleu_vs_w14b_composition"),
    (lambda r: missing(r, "arm", "forms", FORM, "minus_mismatched_words"), "bleu_vs_other_sentence_words"),
    (lambda r: missing(r, "arm", "forms", FORM, "minus_end_to_end"), "bleu_vs_own_end_to_end"),
    (lambda r: missing(r, "arm", "forms", FORM, "minus_end_to_end", "ci95"), "bleu_vs_own_end_to_end"),
    (lambda r: r["arm"]["forms"][FORM]["minus_end_to_end"].update(ci95=None) or r, "bleu_vs_own_end_to_end"),
    (lambda r: missing(r, "arm", "motion_ratio_to_real"), "wrist_speed_reported"),
    (lambda r: missing(r, "arm", "motion_ratio_to_real", FORM), "wrist_speed_reported"),
    (lambda r: missing(r, "arm", "likelihood"), "likelihood minus blank"),
    (lambda r: missing(r, "arm", "likelihood", "forms", FORM, "minus_blank"), "likelihood minus blank"),
    (lambda r: dict(r, shards_arm=None), "the shards of a report were not found"),
    (lambda r: dict(r, shards_comparator=None), "the shards of a report were not found"),
    (lambda r: missing(r, "shards_arm", 0, "clips", 0, "spotting", FORM), "no per-clip spotting"),
])
def test_a_missing_field_is_incomplete_and_never_a_pass(edit, says):
    v = judge(edit(reports()))
    assert v["complete"] is False and v["rule_passed"] is False and says in " ".join(v["missing"])


def set_setting(key, value):
    return lambda r: r["arm"]["settings"].update({key: value}) or r


def set_round20(**over):
    return lambda r: r["arm"]["settings"]["round20"].update(over) or r


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["arm"].update(references=list(reversed(REFS))) or r, "not paired"),
    (lambda r: r["arm"].update(signs=list(reversed(SIGNS))) or r, "not paired"),
    (set_setting("seed", 1), "not paired"),
    (set_setting("translator_key", "other.pt"), "not paired"),
    (set_setting("guidance", 2.), "settings.guidance"),
    (set_setting("clips", 300), "settings.clips"),
    (set_setting("shards", 1), "settings.shards"),
    (set_setting("block", 8), "settings.block"),
    (set_setting("core_margin", 0.), "settings.core_margin"),
    (set_setting("timing", "asllrp"), "settings.timing"),
    (set_setting("generator_mix", dict(key="m", sha256="s", check=None)), "settings.generator_mix"),
    (set_setting("model_sha256", "0"), "settings.model_sha256"),
    (set_setting("recognizer_run", "sign-recognizer-v3d"), "settings.recognizer_run"),
    (set_setting("rerank", 4), "settings.rerank"),
    (set_setting("space_normalise", "sentence"), "settings.space_normalise"),
    (set_setting("checkpoint_key", W14B), "generated words as well"),
    (set_round20(likelihood=False), "likelihood is not on"),
    (set_round20(tempo=1.25), "round20.tempo"),
    (set_round20(lowpass_report_hz=6.), "round20.lowpass_report_hz"),
    (set_round20(lowpass_hz=6.), "round20.lowpass_hz"),
    (set_round20(retime="continuous"), "round20.retime"),
    (set_round20(predicted_duration=True), "round20.predicted_duration"),
    (set_round20(vocabulary_mix=dict(key="m", sha256="s")), "round20.vocabulary_mix"),
    (set_round20(sentences=dict(key="d20", sha256="s")), "not paired"),
    (set_round20(word_cache=None), "no word cache"),
    (set_round20(word_cache=cache_settings(n=4)), "word_cache.n"),
    (set_round20(word_cache=cache_settings(no_gains=False)), "word_cache.no_decoder_gains"),
    (set_round20(word_cache=cache_settings(checkpoint=R16)), "word_cache.checkpoint_key"),
    (set_round20(word_cache=dict(cache_settings(), scorer="sign-recognizer-v2")), "word_cache.scorer"),
    (set_round20(no_decoder_gains=True), "gains are its cache's"),
    (lambda r: r["arm"]["forms"].pop(f"paired:{FORM}") and r, "--paired-with"),
    (lambda r: r["arm"]["forms"][f"paired:{FORM}"].update(hypotheses=["x"] * 6) or r, "--paired-with"),
    (lambda r: r["arm"]["comparators"].pop("end_to_end") and r, "--end-to-end-report"),
    (lambda r: r["arm"]["forms"].pop("end_to_end") and r, "--end-to-end-report"),
    (lambda r: r["pilot"].update(checkpoint_key=R16) or r, "not the arm's prior"),
    (lambda r: r["arm"]["comparators"]["end_to_end"].update(checkpoint_key=R16) or r, "not the arm's prior"),
    (lambda r: r["pilot"]["meaning"].update(references=["x"] * 6) or r, "pilot's references"),
    (lambda r: r["arm"]["forms"][FORM].update(bleu=2.55) or r, "not its hypotheses'"),
    (lambda r: r["arm"]["forms"].pop(FORM) and r, "no core_crossfade"),
    (lambda r: r["comparator"]["forms"].pop(FORM) and r, "comparator has no"),
    (lambda r: r["arm"].update(clips=5) or r, "one complete run"),
    (lambda r: r["arm"].update(stage1_check=dict(stored_clips=300, shared_clips=289, identical=289)) or r, "selected words"),
    (lambda r: r["arm"].update(per_clip=[dict(sign="sign0", stage1_words=["other"], entries=["other"])]) or r, "selected words"),
])
def test_what_is_not_the_arm_or_not_paired_is_refused(edit, match):
    with pytest.raises(SystemExit, match=match):
        judge(edit(reports()))


def test_fresh_arms_and_arms_of_another_prior():
    r = reports()
    r["arm"]["settings"].update(checkpoint_key=W14B)
    r["arm"]["settings"]["round20"].update(word_cache=None, no_decoder_gains=True)
    assert judge(r, best_of=0, no_gains=True)["rule_passed"]
    for over, match in ((dict(best_of=0, no_gains=False), "no_decoder_gains"), (dict(best_of=8, no_gains=True), "no word cache"),
                        (dict(best_of=0, no_gains=True, prior_checkpoint=R16), "settings.checkpoint_key")):
        with pytest.raises(SystemExit, match=match):
            judge(copy.deepcopy(r), **over)
    r = reports()
    r["arm"]["settings"]["round20"]["word_cache"] = cache_settings(checkpoint=R16)
    r["arm"]["comparators"]["end_to_end"]["checkpoint_key"] = R16
    r["pilot"]["checkpoint_key"] = R16
    v = judge(r, prior_checkpoint=R16)
    assert v["rule_passed"] and v["end_to_end"]["checkpoint_key"] == R16


def test_m1_is_reported_on_this_set_and_not_part_of_the_rule():
    assert pv.m1_read(2.55)["met"] and pv.m1_read(2.5501)["met"] and not pv.m1_read(2.5499)["met"]
    high = judge(reports(hyp_arm=REFS))       # the arm reads every reference exactly: BLEU 100
    assert high["m1"]["met"] and high["m1"]["point_estimate"] == 100. and high["rule_passed"]
    low = judge(reports(hyp_arm=["word"] * 6))
    assert low["m1"]["met"] is False and low["rule_passed"] and "C20" in low["m1"]["note"]


def test_m1_uses_the_unrounded_bleu(monkeypatch):
    import sentence_exposure_diagnostic as sed
    r = reports()
    monkeypatch.setattr(sed, "bleu_from_stats", lambda stats: 2.5496)     # the 3 decimals would read 2.55
    r["arm"]["forms"][FORM]["bleu"] = 2.55
    v = judge(r)
    assert v["m1"]["point_estimate"] == 2.5496 and v["m1"]["met"] is False


def store_all(tmp_path, r, arm_key="r/arm.json", comparator_key="r/w14b.json", pilot_key="r/pilot.json"):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'tiny'}")
    payloads = dict(arm=json.dumps(r["arm"]).encode(), comparator=json.dumps(r["comparator"]).encode(),
                    pilot=json.dumps(r["pilot"]).encode())
    store.put(arm_key, payloads["arm"])
    store.put(comparator_key, payloads["comparator"])
    store.put(pilot_key, payloads["pilot"])
    for stem, shards in ((arm_key, r["shards_arm"]), (comparator_key, r["shards_comparator"])):
        for s, part in enumerate(shards or ()):
            store.put(f"{stem[:-5]}-shard{s}.json", json.dumps(part).encode())
    argv = ["verdict", "--store", f"dir:{tmp_path / 'tiny'}", "--arm-key", arm_key, "--comparator-key", comparator_key,
            "--comparator-sha256", hashlib.sha256(payloads["comparator"]).hexdigest(), "--end-to-end-key", pilot_key,
            "--end-to-end-sha256", hashlib.sha256(payloads["pilot"]).hexdigest(), "--prior", "w14b",
            "--prior-checkpoint-key", W14B, "--no-gains", "1", "--best-of", "8", "--set", "val1000", "--bootstrap", "50",
            "--output-key", "r/verdict.json"]
    return store, argv


def test_the_cli_stores_a_verdict_once(tmp_path, capsys):
    store, argv = store_all(tmp_path, reports())
    assert pv.main(argv) == 0
    stored = json.loads(store.get("r/verdict.json"))
    assert stored["rule_passed"] and stored["arm"]["sha256"] == hashlib.sha256(store.get("r/arm.json")).hexdigest()
    assert stored["comparator"]["key"] == "r/w14b.json" and stored["end_to_end"]["key"] == "r/pilot.json"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pv.main(argv)
    assert store.get("r/verdict.json") == json.dumps(stored, indent=1).encode()


def test_the_cli_stores_a_failing_verdict_and_refuses_an_incomplete_one(tmp_path, capsys):
    r = reports()
    r["arm"]["forms"][FORM]["minus_end_to_end"]["ci95"] = [-.1, .9]
    store, argv = store_all(tmp_path, r)
    assert pv.main(argv) == 0
    assert json.loads(store.get("r/verdict.json"))["rule_passed"] is False
    r = missing(reports(), "arm", "forms", FORM, "minus_end_to_end")
    store, argv = store_all(tmp_path, r, arm_key="r2/arm.json", comparator_key="r2/w14b.json", pilot_key="r2/pilot.json")
    argv[argv.index("--output-key") + 1] = "r2/verdict.json"
    assert pv.main(argv) == 2
    assert not store.exists("r2/verdict.json") and "INCOMPLETE" in capsys.readouterr().err


def test_the_cli_refuses_another_comparator_or_pilot(tmp_path):
    store, argv = store_all(tmp_path, reports())
    for flag, message in (("--comparator-sha256", "does not match --comparator-sha256"),
                          ("--end-to-end-sha256", "does not match --end-to-end-sha256")):
        bad = list(argv)
        bad[bad.index(flag) + 1] = "0" * 64
        with pytest.raises(SystemExit, match=message):
            pv.main(bad)
    assert not store.exists("r/verdict.json")
    with pytest.raises(SystemExit):        # val1000 needs its pilot
        pv.main([a for k, a in enumerate(argv) if a != "--end-to-end-key" and argv[k - 1] != "--end-to-end-key"])


# ---- C20's gate, and its verdict ----------------------------------------------------------------------------------------

def val_verdict(*, passed=True, point=2.6, complete=True):
    return dict(kind="round20-phase-b-verdict", set="val1000", complete=complete, rule_passed=passed,
                m1=dict(threshold=2.55, point_estimate=point, met=point >= 2.55))


@pytest.mark.parametrize("verdict,match", [
    (val_verdict(point=2.54), "below 2.55"),
    (val_verdict(passed=False), "rule did not pass"),
    (val_verdict(complete=False), "not a complete"),
    (dict(val_verdict(), set="c20"), "not a complete"),
    (dict(val_verdict(), kind="other"), "not a complete"),
    (dict(val_verdict(), m1=None), "below 2.55"),
    (dict(val_verdict(), m1=dict(met=True, point_estimate=2.5)), "below 2.55"),
])
def test_the_c20_gate_stays_closed(tmp_path, verdict, match):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'g'}")
    store.put("v.json", json.dumps(verdict).encode())
    with pytest.raises(SystemExit, match=match):
        pv.main(["c20-gate", "--store", f"dir:{tmp_path / 'g'}", "--verdict-key", "v.json"])


def test_the_c20_gate_opens_and_needs_a_stored_verdict(tmp_path):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'g'}")
    with pytest.raises(SystemExit, match="no stored verdict"):
        pv.main(["c20-gate", "--store", f"dir:{tmp_path / 'g'}", "--verdict-key", "v.json"])
    store.put("v.json", json.dumps(val_verdict(point=2.55)).encode())
    assert pv.main(["c20-gate", "--store", f"dir:{tmp_path / 'g'}", "--verdict-key", "v.json"]) == 0


def c20_arm(hyp_arm=REFS):
    r = reports(hyp_arm=hyp_arm)
    r["arm"]["settings"]["round20"]["sentences"] = dict(key="c20.json", sha256="c" * 64, set="C20", stored=1000, used=1000)
    return r


def judge_c20(r, verdict):
    return pv.judge_c20(r["arm"], verdict, prior_checkpoint=W14B, no_gains=True, best_of=8, comparator=r["comparator"],
                        sentences=dict(key="c20.json", sha256="c" * 64))


def test_the_c20_verdict():
    v = judge_c20(c20_arm(), val_verdict())
    assert v["m1"]["met"] and v["m1_claim"] and v["complete"] and v["set"] == "c20" and v["val1000_point_estimate"] == 2.6
    assert "bleu_vs_w14b_composition" not in v["items"]               # nothing else read this set
    below = judge_c20(c20_arm(hyp_arm=["word"] * 6), val_verdict())
    assert below["m1"]["met"] is False and below["m1_claim"] is False
    with pytest.raises(SystemExit, match="below 2.55"):
        judge_c20(c20_arm(), val_verdict(point=2.5))
    with pytest.raises(SystemExit, match="round20.sentences"):
        judge_c20(reports(hyp_arm=REFS), val_verdict())
    gone = missing(c20_arm(), "arm", "likelihood")
    assert judge_c20(gone, val_verdict())["complete"] is False
    assert judge_c20(gone, val_verdict())["m1_claim"] is False


# ---- real merges of the synthetic store ---------------------------------------------------------------------------------

def merged_pair(tmp_path, store, sha, words, *, arm_stem="e/arm"):
    """A comparator (no likelihood, as the stored one) and an arm on the same clips, both merged as the job merges them."""
    def shards(stem, *extra):
        for s in (0, 1):
            assert ce.main(run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key", "source.json",
                                    "--shards", "2", "--shard", str(s), *extra,
                                    "--output-key", f"{stem}-shard{s}.json")) == 0
        return f"{stem}-shard0.json,{stem}-shard1.json"
    root = ["merge", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "50", "--end-to-end-report", "pilot.json",
            "--rule", "bleu:end_to_end,bleu:mismatched_words,spotting"]
    if not store.exists("e/w14b.json"):
        assert ce.main([*root, "--shard-keys", shards("e/w14b"), "--output-key", "e/w14b.json"]) == 0
    args = shards(arm_stem, "--likelihood", *words)
    assert ce.main([*root, "--shard-keys", args, "--paired-with", "e/w14b.json", "--output-key", f"{arm_stem}.json"]) == 0


def verdict_argv(tmp_path, store, *, no_gains, best_of, arm="e/arm", output="e/verdict.json"):
    return ["verdict", "--store", f"dir:{tmp_path / 'store'}", "--arm-key", f"{arm}.json", "--comparator-key", "e/w14b.json",
            "--comparator-sha256", hashlib.sha256(store.get("e/w14b.json")).hexdigest(), "--end-to-end-key", "pilot.json",
            "--end-to-end-sha256", hashlib.sha256(store.get("pilot.json")).hexdigest(), "--prior", "w14b",
            "--prior-checkpoint-key", "prior.pt", "--no-gains", str(no_gains), "--best-of", str(best_of), "--set", "val1000",
            "--bootstrap", "50", "--output-key", output]


def test_the_verdict_reads_real_merges_of_a_fresh_arm_without_gains(tmp_path, stage2_store, likelihood_stub):  # noqa: F811
    store, sha, s1, *_ = stage2_store
    put_gains(store)
    pilot_report(store, s1)
    merged_pair(tmp_path, store, sha, ["--no-decoder-gains"])
    arm = json.loads(store.get("e/arm.json"))
    assert "likelihood" not in json.loads(store.get("e/w14b.json")) and "minus_paired_core_crossfade" in arm["forms"][FORM]
    assert pv.main(verdict_argv(tmp_path, store, no_gains=1, best_of=0)) == 0
    v = json.loads(store.get("e/verdict.json"))
    assert v["complete"] and v["items"]["bleu_vs_w14b_composition"]["available"]
    assert set(v["items"]) == {"bleu_vs_w14b_composition", "bleu_vs_other_sentence_words", "bleu_vs_own_end_to_end",
                               "v2_spotting_vs_w14b_composition", "wrist_speed_reported"}
    assert all(k in v["items"]["bleu_vs_w14b_composition"] for k in ("difference", "ci95", "lower_bound"))
    assert v["items"]["bleu_vs_w14b_composition"]["difference"] == arm["forms"][FORM]["minus_paired_core_crossfade"]["difference"]
    assert v["items"]["bleu_vs_own_end_to_end"]["difference"] == arm["forms"][FORM]["minus_end_to_end"]["difference"]
    assert v["items"]["v2_spotting_vs_w14b_composition"]["word_source"] == ["generated"]
    assert v["items"]["wrist_speed_reported"]["ratio_to_real"] == arm["motion_ratio_to_real"][FORM]["wrist_speed"]
    assert v["next_to_bleu"]["likelihood_minus_blank"] == arm["likelihood"]["forms"][FORM]["minus_blank"]
    assert v["comparator"]["paired_likelihood"]["available"] is False and "note" in v["end_to_end"]
    assert v["rule_passed"] == all(i["passed"] for i in v["items"].values())
    # the same reports under another name of the arm are refused: gains on, or a cache
    for over, match in ((dict(no_gains=0, best_of=0), "no_decoder_gains"), (dict(no_gains=1, best_of=8), "no word cache")):
        with pytest.raises(SystemExit, match=match):
            pv.main(verdict_argv(tmp_path, store, output="e/other.json", **over))


def test_the_verdict_reads_real_merges_of_a_cached_arm(tmp_path, stage2_store, likelihood_stub, monkeypatch):  # noqa: F811
    import torch
    import validate_sign_recognizer as vsr
    store, sha, s1, *_ = stage2_store
    pilot_report(store, s1)
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))
    monkeypatch.setattr(vsr, "logits_of", lambda model, states, device, batch=64: torch.zeros(len(states), len(labels)))
    common = ["--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/n4"]
    for s in (0, 1):                                      # the job's flags: samples and choice are BEST_OF
        assert cw.main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "5",
                        "--checkpoint-key", "prior.pt", "--samples", "4", "--choose", "4", "--device", "cpu",
                        "--shards", "2", "--shard", str(s)]) == 0
    assert cw.main(["merge", *common, "--shards", "2"]) == 0
    index = hashlib.sha256(store.get("cache/n4/index.json")).hexdigest()
    words = ["--word-cache-key", "cache/n4/index.json", "--word-cache-sha256", index, "--word-cache-n", "4"]

    def without_checkpoint(args):
        k = args.index("--checkpoint-key")
        return args[:k] + args[k + 2:]
    # the comparator generates; the arm serves the cache instead of the checkpoint (the option refuses both)
    for s in (0, 1):
        assert ce.main(run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key", "source.json",
                                "--shards", "2", "--shard", str(s), "--output-key", f"e/w14b-shard{s}.json")) == 0
        assert ce.main(without_checkpoint(run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key",
                                                   "source.json", "--shards", "2", "--shard", str(s), "--likelihood", *words,
                                                   "--output-key", f"e/arm-shard{s}.json"))) == 0
    root = ["merge", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "50", "--end-to-end-report", "pilot.json"]
    assert ce.main([*root, "--shard-keys", "e/w14b-shard0.json,e/w14b-shard1.json", "--output-key", "e/w14b.json"]) == 0
    assert ce.main([*root, "--shard-keys", "e/arm-shard0.json,e/arm-shard1.json", "--paired-with", "e/w14b.json",
                    "--output-key", "e/arm.json"]) == 0
    assert pv.main(verdict_argv(tmp_path, store, no_gains=0, best_of=4)) == 0
    v = json.loads(store.get("e/verdict.json"))
    assert v["complete"] and v["items"]["v2_spotting_vs_w14b_composition"]["available"]
    assert v["items"]["v2_spotting_vs_w14b_composition"]["word_source"] == ["generated"]   # a cached word is a generated word
    with pytest.raises(SystemExit, match="word_cache.n"):
        pv.main(verdict_argv(tmp_path, store, no_gains=0, best_of=8, output="e/other.json"))
