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
from tests.test_pilot_content_check import pilot, pcc  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runpod/fleet/round20_phase_b.sh"
PHASE_A = ROOT / "runpod/fleet/round20_phase_a.sh"
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
OUT = f"{O}/round20/phase-b"
OUT_A = f"{O}/round20/phase-a"
CMP = f"{O}/round15/eval1000/w14b-step30000.json"
W14B = "training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt"
R16 = "training-runs/tokens-rvq-d0b-prior-r16-rollout/checkpoints/step-00010000.pt"
FORM = "core_crossfade"
PILOTS = [f"{OUT}/pilot-{p}-nogains-val1000.json" for p in ("w14b", "r16")]
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
        stored=(), fail_shard="", hang=False, pilots=True, **env):
    """The real job on stubs. Returns (exit code, stdout, [logged lines])."""
    e = dict(R20B_PHASE=phase, PRIOR=prior, NO_GAINS=no_gains, BEST_OF=best_of, SET=sentence_set, STUB_GATE=str(int(gate)),
             STUB_EXISTS=" ".join([*stored, *(PILOTS if pilots else ())]),
             STUB_FAIL_SHARD=fail_shard, STUB_HANG="1" if hang else "", **env)
    return on_stubs(tmp_path, SCRIPT.read_text(), e, smoke=smoke, gpu=gpu)


def phase_a(tmp_path, phase, *, vocab="r10", arm=None, smoke=False, gpu=True, stored=(), text=None, **env):
    """Phase A's job on the same stubs."""
    e = dict(R20_PHASE=phase, VOCAB=vocab, STUB_EXISTS=" ".join(stored), STUB_GATE="1", STUB_FAIL_SHARD="", STUB_HANG="", **env)
    if arm:
        e["ARM"] = arm
    return on_stubs(tmp_path, text or PHASE_A.read_text(), e, smoke=smoke, gpu=gpu)


def on_stubs(tmp_path, text, env, *, smoke, gpu):
    """A job's text on stubs: every scripts/ run and pin logged, the store and the pod's paths replaced. Returns
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
        'case "$*" in *" stack-check "*) exit ${STUB_STACK:-0} ;; *" one-arm "*) exit ${STUB_ONE_ARM:-0} ;; esac\n'
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
    text = text.replace("source /root/r2env.sh\n", "").replace("cd /root/work/Sign-Translator\n", "")
    text = text.replace("/root/venv/bin/python", "python").replace("/root/r20b-", f"{tmp_path}/r20b-")
    text = text.replace("/root/r20-", f"{tmp_path}/r20-")
    text = re.sub(r"^sha\(\) .*$", "sha() { echo stubsha; }", text, flags=re.M)
    text = re.sub(r"^pin\(\) .*$", 'pin() { echo "pin $1" >> "$STUB_LOG"; }', text, flags=re.M)
    text = re.sub(r"^exists\(\) .*$", 'exists() { case " $STUB_EXISTS " in *" $1 "*) return 0 ;; esac; return 1; }', text,
                  flags=re.M)
    script = tmp_path / "job.sh"
    script.write_text(text)
    e = dict(os.environ, PATH=f"{stubs}:/usr/bin:/bin", THREADS="8", STUB_LOG=str(log))
    for k in ("SMOKE", "LOWPASS", "NO_GAINS", "ARM", "CACHE_KEY"):
        e.pop(k, None)
    e.update(env)
    if smoke:
        e["SMOKE"] = "1"
    run = subprocess.run(["bash", str(script)], env=e, capture_output=True, text=True, timeout=60)
    return run.returncode, run.stdout, log.read_text().splitlines()


def scripts_run(commands):
    """What ran, apart from the D20 reads' check that precedes every full phase."""
    return [c for c in commands if c.startswith("scripts/") and " stack-check " not in c]


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
    stored_pilot = f"{O}/round14b/pilot-w14b-step30000.json" if prior == "w14b" else f"{O}/round16/pilot-r16.json"
    pilot = f"{OUT}/pilot-{prior}-nogains-val1000.json" if no_gains == "1" else stored_pilot
    assert merge == ("scripts/round15_compose_eval.py merge --shard-keys " + ",".join(f"{report}-shard{s}.json" for s in range(4))
                     + f" --end-to-end-report {pilot} --paired-with {CMP} --rule bleu:end_to_end,bleu:mismatched_words,"
                     f"spotting --output-key {report}.json")
    pins = [c for c in commands if c.startswith("pin ")]
    assert f"pin {CMP}" in pins and f"pin {stored_pilot}" in pins
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
        assert len(commands) == 1 and f"round20_phase_b_verdict.py c20-gate --verdict-key {verdict} " in commands[0]   # nothing of C20
        assert (f"--out {OUT} --report w14b-nogains-n8-val1000 --prior w14b --prior-checkpoint-key {W14B} --no-gains 1 "
                "--best-of 8 --lowpass 0") in commands[0]
    code, out, commands = job(tmp_path, "cache", sentence_set="c20", stored=[verdict], smoke=True)    # nor by a smoke
    code2, out2, _ = job(tmp_path, "cache", sentence_set="c20", smoke=True)
    assert code == 0 and code2 == 1 and "no stored val1000 verdict" in out2
    # another arm's verdict does not open it
    code, out, commands = job(tmp_path, "cache", prior="r16", sentence_set="c20", stored=[verdict])
    assert code == 1 and "no stored val1000 verdict" in out and not commands
    # open: the gate first, then the pin, then the set; the comparator and the pilots cannot pair with it
    c20 = f"--sentences-key {O}/round20/c20.json --sentences-sha256 24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956"
    code, out, commands = job(tmp_path, "cache", sentence_set="c20", stored=[verdict])
    assert code == 0 and f"c20-gate --verdict-key {verdict} " in commands[0] and commands[1] == f"pin {O}/round20/c20.json"
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
    stored = dict(checkpoint_key=W14B, clips=6, guidance=1.5, part_temperature=dict(COMPARATOR["part_temperature"]),
                  meaning=dict(references=REFS, generated=dict(bleu=1.786, hypotheses=["stored"] * 6),
                               real=dict(hypotheses=list(REFS)), blank=dict(hypotheses=["the"] * 6)))
    pilot = copy.deepcopy(stored)                     # the arm has no gains (below): its pilot is the rerun without them
    pilot["no_decoder_gains"] = True
    pilot["meaning"]["generated"] = dict(bleu=1.9, hypotheses=["end"] * 6)
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
    return dict(arm=arm, comparator=comparator, pilot=pilot, stored_pilot=stored, shards_arm=[shard(settings, delta)],
                shards_comparator=[shard(COMPARATOR)])


def judge(r, **over):
    args = dict(prior_checkpoint=W14B, no_gains=True, best_of=8, boot=50)
    return pv.judge_val1000(r["arm"], r["comparator"], r["pilot"], shards_arm=r["shards_arm"],
                            shards_comparator=r["shards_comparator"], stored_pilot=r["stored_pilot"], **{**args, **over})


def gained_arm(r):
    """The same reports as an arm with gains: its cache's, and the stored pilot as its own end-to-end."""
    r["arm"]["settings"]["round20"]["word_cache"] = cache_settings(no_gains=False)
    r["pilot"] = r["stored_pilot"]
    r["arm"]["forms"]["end_to_end"]["hypotheses"] = r["pilot"]["meaning"]["generated"]["hypotheses"]
    return r


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
    assert v["comparator"]["real_hypotheses_identical"] == 6 and v["end_to_end"]["no_decoder_gains"] is True
    assert "WITHOUT gains" in v["items"]["bleu_vs_own_end_to_end"]["note"]
    assert v["end_to_end"]["identical_to_stored_pilot"] == dict(real=6, blank=6)
    assert not any("artefact" in k or "tempo" in k for k in v)        # no tempo-artefact logic
    gained = judge(gained_arm(reports()), no_gains=False)
    assert "with gains" in gained["items"]["bleu_vs_own_end_to_end"]["note"] and gained["end_to_end"]["no_decoder_gains"] is False
    assert "identical_to_stored_pilot" not in gained["end_to_end"]


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
    r["stored_pilot"]["checkpoint_key"] = R16
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
    stored_key = pilot_key.replace("pilot", "stored-pilot")
    stored_payload = json.dumps(r["stored_pilot"]).encode()
    store.put(stored_key, stored_payload)
    for stem, shards in ((arm_key, r["shards_arm"]), (comparator_key, r["shards_comparator"])):
        for s, part in enumerate(shards or ()):
            store.put(f"{stem[:-5]}-shard{s}.json", json.dumps(part).encode())
    argv = ["verdict", "--store", f"dir:{tmp_path / 'tiny'}", "--arm-key", arm_key, "--comparator-key", comparator_key,
            "--comparator-sha256", hashlib.sha256(payloads["comparator"]).hexdigest(), "--end-to-end-key", pilot_key,
            "--end-to-end-sha256", hashlib.sha256(payloads["pilot"]).hexdigest(), "--stored-pilot-key", stored_key,
            "--stored-pilot-sha256", hashlib.sha256(stored_payload).hexdigest(), "--out", arm_key.split("/")[0],
            "--report", "w14b-nogains-n8-val1000", "--prior", "w14b",
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
                          ("--end-to-end-sha256", "does not match --end-to-end-sha256"),
                          ("--stored-pilot-sha256", "does not match --stored-pilot-sha256")):
        bad = list(argv)
        bad[bad.index(flag) + 1] = "0" * 64
        with pytest.raises(SystemExit, match=message):
            pv.main(bad)
    assert not store.exists("r/verdict.json")
    with pytest.raises(SystemExit):        # val1000 needs its pilot
        pv.main([a for k, a in enumerate(argv) if a != "--end-to-end-key" and argv[k - 1] != "--end-to-end-key"])


# ---- C20's gate, and its verdict ----------------------------------------------------------------------------------------

def val_verdict(*, passed=True, point=2.6, complete=True, **arm):
    return dict(kind="round20-phase-b-verdict", set="val1000", complete=complete, rule_passed=passed,
                m1=dict(threshold=2.55, point_estimate=point, met=point >= 2.55),
                arm=dict(dict(prior="w14b", prior_checkpoint_key=W14B, no_decoder_gains=True, best_of=8, lowpass=False), **arm))


def gate_argv(tmp_path, *extra):
    return ["c20-gate", "--store", f"dir:{tmp_path / 'g'}", "--verdict-key", "o/v.json", "--out", "o", "--report",
            "w14b-nogains-n8-val1000", "--prior", "w14b", "--prior-checkpoint-key", W14B, "--no-gains", "1", "--best-of", "8",
            "--lowpass", "0", *extra]


@pytest.mark.parametrize("verdict,match", [
    (val_verdict(point=2.54), "below 2.55"),
    (val_verdict(passed=False), "rule did not pass"),
    (val_verdict(complete=False), "not a complete"),
    (dict(val_verdict(), set="c20"), "not a complete"),
    (dict(val_verdict(), kind="other"), "not a complete"),
    (dict(val_verdict(), m1=None), "below 2.55"),
    (dict(val_verdict(), m1=dict(met=True, point_estimate=2.5)), "below 2.55"),
    (val_verdict(best_of=4), "another arm's"),
    (val_verdict(no_decoder_gains=False), "another arm's"),
    (val_verdict(lowpass=True), "another arm's"),
    (val_verdict(prior="r16"), "another arm's"),
    (val_verdict(prior_checkpoint_key=R16), "another arm's"),
    (dict(val_verdict(), arm=None), "another arm's"),
])
def test_the_c20_gate_stays_closed(tmp_path, verdict, match):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'g'}")
    store.put("o/v.json", json.dumps(verdict).encode())
    with pytest.raises(SystemExit, match=match):
        pv.main(gate_argv(tmp_path))


def test_the_c20_gate_opens_and_needs_a_stored_verdict(tmp_path):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'g'}")
    with pytest.raises(SystemExit, match="no stored verdict"):
        pv.main(gate_argv(tmp_path))
    store.put("o/v.json", json.dumps(val_verdict(point=2.55)).encode())
    assert pv.main(gate_argv(tmp_path)) == 0
    # another arm on the 1,000 keeps it shut, even for a declared second arm: that decision is declared separately
    store.put("o/w14b-gains-n8-val1000-shard2.json", b"{}")
    with pytest.raises(SystemExit, match="also hold .'w14b-gains-n8-val1000'."):
        pv.main(gate_argv(tmp_path))
    verdict = dict(val_verdict(point=2.55), declared_replacement="cf4b1559a3")
    (tmp_path / "g/o/v.json").unlink()
    store.put("o/v.json", json.dumps(verdict).encode())
    with pytest.raises(SystemExit, match="declared separately"):
        pv.main(gate_argv(tmp_path))
    (tmp_path / "g/o/w14b-gains-n8-val1000-shard2.json").unlink()
    with pytest.raises(SystemExit, match="declared separately"):             # the flag alone keeps it shut
        pv.main(gate_argv(tmp_path))


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

def merged_pair(tmp_path, store, sha, words, *, arm_stem="e/arm", pilot="pilot.json", merge_extra=()):
    """A comparator (no likelihood, as the stored one) and an arm on the same clips, both merged as the job merges them."""
    def shards(stem, *extra):
        for s in (0, 1):
            assert ce.main(run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key", "source.json",
                                    "--shards", "2", "--shard", str(s), *extra,
                                    "--output-key", f"{stem}-shard{s}.json")) == 0
        return f"{stem}-shard0.json,{stem}-shard1.json"
    root = ["merge", "--store", f"dir:{tmp_path / 'store'}", "--bootstrap", "50", "--end-to-end-report", pilot,
            "--rule", "bleu:end_to_end,bleu:mismatched_words,spotting"]
    if not store.exists("e/w14b.json"):
        assert ce.main([*root, "--shard-keys", shards("e/w14b"), "--output-key", "e/w14b.json"]) == 0
    args = shards(arm_stem, "--likelihood", *words)
    assert ce.main([*root, "--shard-keys", args, "--paired-with", "e/w14b.json", *merge_extra,
                    "--output-key", f"{arm_stem}.json"]) == 0


def verdict_argv(tmp_path, store, *, no_gains, best_of, arm="e/arm", output="e/verdict.json", pilot="pilot.json"):
    stored = ["--stored-pilot-key", "pilot.json", "--stored-pilot-sha256", hashlib.sha256(store.get("pilot.json")).hexdigest()]
    return ["verdict", "--store", f"dir:{tmp_path / 'store'}", "--arm-key", f"{arm}.json", "--comparator-key", "e/w14b.json",
            "--comparator-sha256", hashlib.sha256(store.get("e/w14b.json")).hexdigest(), "--end-to-end-key", pilot,
            "--end-to-end-sha256", hashlib.sha256(store.get(pilot)).hexdigest(), *(stored if pilot != "pilot.json" else ()),
            "--out", "e", "--report", "w14b-nogains-n0-val1000", "--prior", "w14b",
            "--prior-checkpoint-key", "prior.pt", "--no-gains", str(no_gains), "--best-of", str(best_of), "--set", "val1000",
            "--bootstrap", "50", "--output-key", output]


def test_the_verdict_reads_real_merges_of_a_fresh_arm_without_gains(tmp_path, stage2_store, likelihood_stub):  # noqa: F811
    store, sha, s1, *_ = stage2_store
    put_gains(store)
    pilot_report(store, s1)
    rerun = dict(json.loads(store.get("pilot.json")), no_decoder_gains=True)
    store.put("pilot-nogains.json", json.dumps(rerun).encode())
    merged_pair(tmp_path, store, sha, ["--no-decoder-gains"], pilot="pilot-nogains.json")
    arm = json.loads(store.get("e/arm.json"))
    assert "declared_replacement" not in arm and "set_read" not in arm
    assert "likelihood" not in json.loads(store.get("e/w14b.json")) and "minus_paired_core_crossfade" in arm["forms"][FORM]
    assert pv.main(verdict_argv(tmp_path, store, no_gains=1, best_of=0, pilot="pilot-nogains.json")) == 0
    v = json.loads(store.get("e/verdict.json"))
    assert v["declared_replacement"] is None and v["other_arms"] == []
    assert not {"c20_arm_choice", "stack_check", "set_read"} & set(v)
    assert v["complete"] and v["items"]["bleu_vs_w14b_composition"]["available"]
    assert set(v["items"]) == {"bleu_vs_w14b_composition", "bleu_vs_other_sentence_words", "bleu_vs_own_end_to_end",
                               "v2_spotting_vs_w14b_composition", "wrist_speed_reported"}
    assert all(k in v["items"]["bleu_vs_w14b_composition"] for k in ("difference", "ci95", "lower_bound"))
    assert v["items"]["bleu_vs_w14b_composition"]["difference"] == arm["forms"][FORM]["minus_paired_core_crossfade"]["difference"]
    assert v["items"]["bleu_vs_own_end_to_end"]["difference"] == arm["forms"][FORM]["minus_end_to_end"]["difference"]
    assert v["items"]["v2_spotting_vs_w14b_composition"]["word_source"] == ["generated"]
    assert v["items"]["wrist_speed_reported"]["ratio_to_real"] == arm["motion_ratio_to_real"][FORM]["wrist_speed"]
    assert v["next_to_bleu"]["likelihood_minus_blank"] == arm["likelihood"]["forms"][FORM]["minus_blank"]
    assert v["comparator"]["paired_likelihood"]["available"] is False
    assert "WITHOUT gains" in v["items"]["bleu_vs_own_end_to_end"]["note"]
    assert v["rule_passed"] == all(i["passed"] for i in v["items"].values())
    # the same reports under another name of the arm are refused: gains on, or a cache
    for over, match in ((dict(no_gains=0, best_of=0), "no_decoder_gains"), (dict(no_gains=1, best_of=8), "no word cache")):
        with pytest.raises(SystemExit, match=match):
            pv.main(verdict_argv(tmp_path, store, output="e/other.json", pilot="pilot-nogains.json", **over))
    with pytest.raises(SystemExit, match="end-to-end-report|lenient"):          # the merge read another pilot than this
        pv.main(verdict_argv(tmp_path, store, no_gains=1, best_of=0, output="e/other.json"))


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
    assert "with gains" in v["items"]["bleu_vs_own_end_to_end"]["note"]


# ---- amendment 4: the D20 stack read (Phase A's no-gains cache and a9) ----------------------------------------------------

def test_phase_a_default_cache_is_unchanged(tmp_path):
    code, out, commands = phase_a(tmp_path, "cache", arm="w14b")
    runs = [c.split(" #OMP")[0] for c in commands if "round20_word_cache.py run" in c]
    assert code == 0 and "R20-PHASE-A-DONE 0" in out and len(runs) == 4
    assert sorted(runs)[0] == (
        f"scripts/round20_word_cache.py run --prefix {OUT_A}/cache-w14b-r10 --mix-key {COMPARATOR['mix_key']} "
        f"--mix-sha256 {COMPARATOR['mix_sha256']} --sentences-key {O}/round20/d20.json "
        "--sentences-sha256 d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe "
        f"--checkpoint-key {W14B} --generator-mix-sha256 {COMPARATOR['generator_mix']['sha256']} --samples 16 "
        "--choose 4,8,16 --recognizer-run sign-recognizer-v3d --device cuda --shards 4 --shard 0 --clips 500")
    assert f"R20 cache-w14b-r10 {OUT_A}/cache-w14b-r10/index.json stubsha" in out
    code, out, commands = phase_a(tmp_path, "cache", arm="r16", NO_GAINS="0")
    assert code == 0 and all("--no-decoder-gains" not in c and "/cache-r16-r10 " in c for c in commands if " run " in c)


def test_phase_a_no_gains_cache(tmp_path, monkeypatch):
    code, out, commands = phase_a(tmp_path, "cache", arm="w14b", NO_GAINS="1")
    runs = [c for c in commands if "round20_word_cache.py run" in c]
    assert code == 0 and len(runs) == 4 and f"R20 cache-w14b-nogains-r10 {OUT_A}/cache-w14b-nogains-r10/index.json" in out
    base = phase_a(tmp_path, "cache", arm="w14b")[2]
    plain = sorted(c for c in base if " run " in c)
    assert sorted(c.replace(" --no-decoder-gains", "").replace("cache-w14b-nogains-r10", "cache-w14b-r10") for c in runs) == plain
    assert all(c.split(" #OMP")[0].endswith("--clips 500 --no-decoder-gains") for c in runs)
    assert f"--prefix {OUT_A}/cache-w14b-nogains-r10 --shards 4" in next(c for c in commands if " merge " in c)
    seen = {}
    monkeypatch.setattr(cw, "run", lambda args: seen.update(vars(args)) or {})
    assert cw.main(runs[0].split(" #OMP")[0].split()[1:] + ["--store", f"dir:{tmp_path / 'c'}"]) == 0
    assert seen["no_decoder_gains"] is True
    stored = phase_a(tmp_path, "cache", arm="w14b", NO_GAINS="1", stored=[f"{OUT_A}/cache-w14b-nogains-r10/index.json"])
    assert stored[0] == 0 and "stored:" in stored[1] and not scripts_run(stored[2])


@pytest.mark.parametrize("phase,kwargs,message", [
    ("a4", dict(arm="n8", PRIOR="f12"), "PRIOR must be"),
    ("a9", dict(arm="n8", PRIOR="f12"), "PRIOR must be"),
    ("cache", dict(arm="r16", PRIOR="r16"), "PRIOR belongs"),
    ("a0", dict(PRIOR="r16"), "PRIOR belongs"),
    ("a8", dict(PRIOR="r16"), "PRIOR belongs"),
    ("a9", dict(arm="n16", PRIOR="r16"), "is not stored"),
    ("cache", dict(arm="w14b", vocab="r14b-fraction", NO_GAINS="1"), "VOCAB=r10 only"),
    ("cache", dict(arm="r16", vocab="r14b-fraction", NO_GAINS="1"), "VOCAB=r10 only"),
    ("cache", dict(arm="w14b", NO_GAINS="2"), "NO_GAINS must be"),
    ("a4", dict(arm="n8", NO_GAINS="1"), "belongs to the cache phase"),
    ("a0", dict(NO_GAINS="1"), "belongs to the cache phase"),
    ("a9", dict(arm="n8", NO_GAINS="1"), "belongs to the cache phase"),
    ("a9", dict(arm="n5"), "ARM n4, n8 or n16"),
    ("a9", dict(), "ARM n4, n8 or n16"),
    ("a9", dict(arm="n8", vocab="r14b-activity"), "VOCAB must be r10"),
    ("a9", dict(arm="n8"), "is not stored"),
    ("a9", dict(arm="n8", smoke=True), "give CACHE_KEY"),
])
def test_phase_a_refusals(tmp_path, phase, kwargs, message):
    code, out, commands = phase_a(tmp_path, phase, **kwargs)
    assert code == 1 and message in out and "ALERT" in out and "R20-PHASE-A-DONE 1" in out and not scripts_run(commands)


@pytest.mark.parametrize("arm,n", [("n4", 4), ("n8", 8), ("n16", 16)])
def test_a9_composes_from_the_no_gains_cache(tmp_path, monkeypatch, arm, n):
    index = f"{OUT_A}/cache-w14b-nogains-r10/index.json"
    code, out, commands = phase_a(tmp_path, "a9", arm=arm, stored=[index])
    assert code == 0 and "R20-PHASE-A-DONE 0" in out
    run = next(c for c in commands if "round15_compose_eval.py run" in c)
    assert f"--word-cache-key {index} --word-cache-sha256 stubsha --word-cache-n {n} " in run
    assert "--recognizer-run sign-recognizer-v2 --lowpass-report-hz 6 --output-key" in run
    assert run.endswith(f"--output-key {OUT_A}/a9-nogains-{arm}-r10-shard0.json")
    assert "--no-decoder-gains" not in run and "--checkpoint-key" not in run and "--clips 500" in run and "--likelihood" in run
    seen = parsed_run(tmp_path, monkeypatch, run, "a9")               # the script's own refusals pass it
    assert seen["word_cache_n"] == n and seen["lowpass_report_hz"] == 6.
    merge = next(c for c in commands if " merge " in c)
    assert merge.endswith(f"--output-key {OUT_A}/a9-nogains-{arm}-r10.json")
    # against a4's own list only the cache and the report option differ
    a4 = phase_a(tmp_path, "a4", arm=arm, stored=[f"{OUT_A}/cache-w14b-r10/index.json"])[2]
    a4 = next(c for c in a4 if "round15_compose_eval.py run" in c)
    assert run.replace("cache-w14b-nogains-r10", "cache-w14b-r10").replace(" --lowpass-report-hz 6", "").replace(
        f"a9-nogains-{arm}-r10", f"a4-w14b-{arm}-r10") == a4
    # written once
    stored = phase_a(tmp_path, "a9", arm=arm, stored=[index, f"{OUT_A}/a9-nogains-{arm}-r10.json"])
    assert stored[0] == 0 and "stored:" in stored[1] and not scripts_run(stored[2])


def test_a9_smoke_reads_the_smoke_cache_at_four(tmp_path):
    code, out, commands = phase_a(tmp_path, "a9", arm="n16", smoke=True, gpu=False, CACHE_KEY="s/index.json", stored=["s/index.json"])
    run = next(c for c in commands if "round15_compose_eval.py run" in c)
    assert code == 0 and "--word-cache-key s/index.json" in run and "--word-cache-n 4 " in run and "--clips 5 " in run
    assert f"{OUT_A}/smoke/" in run


def test_the_phase_list_keeps_its_order():
    text = PHASE_A.read_text()
    assert "a1 a0 a7 a2 a3 cache a4 a5 a8 a9" in text and "#   a9    (CPU or GPU, VOCAB=r10 only)" in text


def d20_report(tmp_path, store, sha, ssha, stem, *words):
    args = [a for a in run_args(tmp_path, sha) if a != "core_crossfade,oracle_stage2"]
    args.insert(args.index("--variants") + 1, FORM)
    if "--word-cache-key" in words:
        k = args.index("--checkpoint-key")
        args = args[:k] + args[k + 2:]
    assert ce.main([*args, "--sentences-key", "sets/d20.json", "--sentences-sha256", ssha, "--likelihood", *words,
                    "--output-key", f"{stem}-shard0.json"]) == 0
    assert ce.main(["merge", "--store", f"dir:{tmp_path / 'store'}", "--shard-keys", f"{stem}-shard0.json", "--bootstrap", "30",
                    "--output-key", f"{stem}.json"]) == 0
    return json.loads(store.get(f"{stem}.json")), [json.loads(store.get(f"{stem}-shard0.json"))]


def d20_cache(tmp_path, store, sha, ssha, prefix, *extra):
    common = ["--store", f"dir:{tmp_path / 'store'}", "--prefix", prefix]
    assert cw.main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key", "sets/d20.json",
                    "--sentences-sha256", ssha, "--checkpoint-key", "prior.pt", "--samples", "4", "--choose", "4",
                    "--device", "cpu", *extra]) == 0
    assert cw.main(["merge", *common]) == 0
    return ["--word-cache-key", f"{prefix}/index.json", "--word-cache-n", "4", "--word-cache-sha256",
            hashlib.sha256(store.get(f"{prefix}/index.json")).hexdigest()]


def test_a9_states_its_origin_and_pairs_with_a7_and_a4_with_the_guard(tmp_path, stage2_store, likelihood_stub,  # noqa: F811
                                                                      monkeypatch):
    import torch
    import validate_sign_recognizer as vsr
    from scripts.round20_compare import compare_forms
    from tests.test_round20_eval import put_set
    store, sha, *_ = stage2_store
    put_gains(store)
    ssha, _ = put_set(store, sha)
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))
    monkeypatch.setattr(vsr, "logits_of", lambda model, states, device, batch=64: torch.zeros(len(states), len(labels)))
    gained = d20_cache(tmp_path, store, sha, ssha, "cache/w14b")
    plain = d20_cache(tmp_path, store, sha, ssha, "cache/w14b-nogains", "--no-decoder-gains")
    a9, s9 = d20_report(tmp_path, store, sha, ssha, "e/a9", *plain, "--lowpass-report-hz", "6")
    a4, s4 = d20_report(tmp_path, store, sha, ssha, "e/a4", *gained)
    a7, s7 = d20_report(tmp_path, store, sha, ssha, "e/a7", "--no-decoder-gains")
    # the report states the origin; a4's has no such key; a9 passes no option of its own
    cache = a9["settings"]["round20"]["word_cache"]
    assert cache["no_decoder_gains"] is True and cache["key"] == "cache/w14b-nogains/index.json"
    assert "no_decoder_gains" not in a9["settings"]["round20"]
    assert "no_decoder_gains" not in a4["settings"]["round20"]["word_cache"]
    assert a7["settings"]["round20"]["no_decoder_gains"] is True and a7["settings"]["round20"]["word_cache"] is None
    assert a9["round20"]["spotting_note"].startswith("independent")
    # forms pairs it with both, and the guard is not withheld: all three are generated words
    for other, shards, sources in ((a7, s7, {"generated"}), (a4, s4, {"cache"})):
        assert set(other["round20"]["per_variant"][FORM]["sources"]) == sources
        c = compare_forms(a9, FORM, other, FORM, boot=20, shards_a=s9, shards_b=shards)
        assert c["sentences"] == 4 and "difference" in c["likelihood_minus_blank"]["difference"]
        assert c["spotting_guard"]["available"] and c["spotting_guard"]["word_source"] == ["generated"]
        assert c["spotting_guard_lower_bound"] is not None
    # the stack's @lowpass against the stack: both estimates read; spotting exists for the main forms only, so it is withheld
    c = compare_forms(a9, f"{FORM}@lowpass", a9, FORM, boot=20, shards_a=s9, shards_b=s9)
    assert "difference" in c["bleu"] and "difference" in c["likelihood_minus_blank"]["difference"]
    assert c["spotting_guard"]["available"] is False and "no per-clip spotting" in c["spotting_guard"]["reason"]
    assert c["spotting_guard_lower_bound"] is None


# ---- amendment 4: LOWPASS in Phase B --------------------------------------------------------------------------------------

STACK = dict(prior="w14b", no_gains="1", best_of="8")
STACK_INDEX = f"{OUT}/cache-w14b-nogains-val1000-n8/index.json"


def test_lowpass_is_one_more_option_on_the_stack(tmp_path, monkeypatch):
    code, out, commands = job(tmp_path, "run", **STACK, stored=[STACK_INDEX], LOWPASS="1")
    runs = [c for c in commands if "round15_compose_eval.py run" in c]
    assert code == 0 and len(runs) == 4 and all(c.count("--lowpass-hz 6") == 1 and "--lowpass-report-hz" not in c for c in runs)
    report = f"{OUT}/w14b-nogains-n8-lp6-val1000"
    assert [c.split("--output-key ")[1] for c in runs] == [f"{report}-shard{s}.json" for s in range(4)]
    plain = [c for c in job(tmp_path, "run", **STACK, stored=[STACK_INDEX])[2] if "round15_compose_eval.py run" in c]
    assert [c.replace(" --lowpass-hz 6", "").replace("-lp6", "") for c in runs] == plain
    assert not any("--lowpass" in c for c in plain)              # unset: as before
    assert [c for c in job(tmp_path, "run", **STACK, stored=[STACK_INDEX], LOWPASS="0")[2]
            if "round15_compose_eval.py run" in c] == plain
    base = parsed_run(tmp_path, monkeypatch, f"scripts/round15_compose_eval.py run {comparator_command()} --shard 0 "
                      "--output-key x.json", "base")
    arm = parsed_run(tmp_path, monkeypatch, runs[0], "lp")
    expected = dict(base, shard=0, output_key=f"{report}-shard0.json", likelihood=True, store=arm["store"], checkpoint_key=None,
                    word_workers=1, word_cache_n=8, word_cache_key=STACK_INDEX, word_cache_sha256="stubsha", lowpass_hz=6.)
    assert {k: (arm[k], expected[k]) for k in arm if arm[k] != expected[k]} == {}
    assert f"--output-key {report}.json" in next(c for c in commands if " merge " in c)
    # the filter comes after rendering: the same cache
    code, out, commands = job(tmp_path, "cache", **STACK, LOWPASS="1")
    assert code == 0 and all(f"--prefix {OUT}/cache-w14b-nogains-val1000-n8 " in c for c in commands if " run " in c)
    # the verdict and C20's gate name the low-passed arm
    code, out, commands = job(tmp_path, "verdict", **STACK, stored=[f"{report}.json"], LOWPASS="1")
    line = next(c for c in commands if " verdict " in c)
    assert code == 0 and "--lowpass 1 " in line and f"--output-key {OUT}/verdict-w14b-nogains-n8-lp6-val1000.json" in line
    code, out, commands = job(tmp_path, "cache", **STACK, sentence_set="c20", LOWPASS="1",
                              stored=[f"{OUT}/verdict-w14b-nogains-n8-val1000.json"])
    assert code == 1 and "no stored val1000 verdict" in out      # the unfiltered arm's verdict does not open it
    verdict = f"{OUT}/verdict-w14b-nogains-n8-lp6-val1000.json"
    code, out, commands = job(tmp_path, "cache", **STACK, sentence_set="c20", LOWPASS="1", stored=[verdict])
    gate = next(c for c in commands if " c20-gate " in c)
    assert code == 0 and f"--verdict-key {verdict} " in gate and gate.endswith("--best-of 8 --lowpass 1")
    assert "--report w14b-nogains-n8-lp6-val1000 " in gate


@pytest.mark.parametrize("kwargs,message", [
    (dict(prior="w14b", no_gains="0", best_of="8"), "LOWPASS=1: only on a stack"),   # gained words
    (dict(prior="w14b", no_gains="1", best_of="0"), "LOWPASS=1: only on a stack"),   # fresh words: other seeds
    (dict(prior="r16", no_gains="0", best_of="8"), "LOWPASS=1: only on a stack"),
    (dict(prior="r16", no_gains="1", best_of="0"), "LOWPASS=1: only on a stack"),
    (dict(prior="r16", no_gains="0", best_of="0"), "LOWPASS=1: only on a stack"),
])
def test_lowpass_is_refused_off_the_stack(tmp_path, kwargs, message):
    for phase in ("run", "verdict"):
        code, out, commands = job(tmp_path, phase, **kwargs, LOWPASS="1")
        assert code == 1 and message in out and "R20-PHASE-B-DONE 1" in out and not commands


def test_lowpass_must_be_0_or_1(tmp_path):
    code, out, commands = job(tmp_path, "run", **STACK, LOWPASS="2")
    assert code == 1 and "LOWPASS must be 0 or 1" in out and not commands


def test_phase_b_cache_prefix_differs_by_gains(tmp_path):
    prefixes = {}
    for no_gains in ("0", "1"):
        commands = job(tmp_path, "cache", prior="w14b", no_gains=no_gains, best_of="8")[2]
        prefixes[no_gains] = {re.search(r"--prefix (\S+)", c)[1] for c in commands if " run " in c}
        assert all(("--no-decoder-gains" in c) == (no_gains == "1") for c in commands if " run " in c)
    assert prefixes == {"0": {f"{OUT}/cache-w14b-gains-val1000-n8"}, "1": {f"{OUT}/cache-w14b-nogains-val1000-n8"}}


def test_the_verdict_names_the_low_passed_arm():
    r = reports()
    with pytest.raises(SystemExit, match="round20.lowpass_hz: None, not 6.0"):
        judge(copy.deepcopy(r), lowpass=True)
    r["arm"]["settings"]["round20"]["lowpass_hz"] = 6.
    v = judge(copy.deepcopy(r), lowpass=True)
    assert v["rule_passed"] and "not low-passed" in v["end_to_end"]["note"]
    with pytest.raises(SystemExit, match="round20.lowpass_hz: 6.0, not None"):
        judge(r)
    r["arm"]["settings"]["round20"]["lowpass_hz"] = 5.
    with pytest.raises(SystemExit, match="round20.lowpass_hz: 5.0, not 6.0"):
        judge(r, lowpass=True)


def test_the_main_form_filter_comes_after_rendering(tmp_path, stage2_store, likelihood_stub):  # noqa: F811
    """--lowpass-hz is @lowpass as the main form: the same words, seeds and plans; composed forms and controls filtered, real
    and blank not."""
    store, sha, *_ = stage2_store
    common = ["--variants", FORM, "--clips", "5", "--likelihood"]
    assert ce.main(run_args(tmp_path, sha, *common, "--lowpass-report-hz", "6", "--output-key", "e/report.json")) == 0
    assert ce.main(run_args(tmp_path, sha, *common, "--lowpass-hz", "6", "--output-key", "e/main.json")) == 0
    report, main = (json.loads(store.get(f"e/{k}.json")) for k in ("report", "main"))
    assert main["settings"]["round20"]["lowpass_hz"] == 6. and report["settings"]["round20"]["lowpass_report_hz"] == 6.
    assert main["words_derangement"] == report["words_derangement"]
    for a, b in zip(main["clips"], report["clips"]):
        assert a["entries"] == b["entries"] and a["shuffled_order"] == b["shuffled_order"]
        assert {k: v for k, v in a["plans"][FORM].items() if k != "round20"} == \
            {k: v for k, v in b["plans"][FORM].items() if k != "round20"}                    # the plan is the unfiltered one
        assert a["motion"][FORM] == b["motion"][f"{FORM}@lowpass"]                            # the same filter on the same render
        assert a["hypotheses"][FORM] == b["hypotheses"][f"{FORM}@lowpass"]
        assert a["likelihood"][FORM] == b["likelihood"][f"{FORM}@lowpass"]
        assert a["motion"]["real"] == b["motion"]["real"] and a["hypotheses"]["real"] == b["hypotheses"]["real"]
        assert a["hypotheses"]["blank"] == b["hypotheses"]["blank"]
        for control in ("/shuffled", "/mismatched_words"):                                    # the controls are filtered too
            assert a["motion"][f"{FORM}{control}"] != b["motion"][f"{FORM}{control}"]


# ---- review fix 1: its own end-to-end without gains ---------------------------------------------------------------------------

def nogains_stub(monkeypatch, seen):
    import evaluate_text_to_motion as etm
    from tests.test_pilot_content_check import stub_sample

    def load(store, key, sha, device, temperature=None, **kw):
        seen.append(kw)
        return "stub", (), stub_sample
    monkeypatch.setattr(etm, "load_generator", load)


def test_the_pilot_without_gains_changes_only_its_decoder_and_its_record(pilot, monkeypatch):  # noqa: F811
    store, run = pilot
    seen = []
    nogains_stub(monkeypatch, seen)
    default = json.loads(run())
    plain = json.loads(run("--no-decoder-gains", output="reports/nogains.json"))
    assert seen == [{}, {"decoder_gains": False}]                    # the default call passes nothing new
    assert plain.pop("no_decoder_gains") is True and "no_decoder_gains" not in default
    assert plain == default                                          # same seeds, clips and words (the stub does not decode)
    with pytest.raises(SystemExit, match="refusing to overwrite"):   # written once, before any generation
        run("--no-decoder-gains", output="reports/nogains.json")
    assert len(seen) == 2
    with pytest.raises(SystemExit, match="reuse a gained one"):
        run("--no-decoder-gains", "--meaning-from-key", "reports/pilot.json", output="reports/x.json")
    for bad in (("--first-clips", "2"), ("--no-decoder-gains", "--first-clips", "1")):
        with pytest.raises(SystemExit, match="--first-clips"):
            run(*bad, output="reports/y.json")
    first = json.loads(run("--no-decoder-gains", "--first-clips", "2", output="reports/smoke.json"))
    assert first["first_clips"] == 2 and first["clips"] == 2
    assert first["meaning"]["references"] == default["meaning"]["references"][:2]


def test_every_gain_path_of_the_pilot_goes_through_load_generator(tmp_path, stage2_store):  # noqa: F811
    from evaluate_text_to_motion import load_generator
    from fit_tokenizer_gains import apply_gains
    from tests.test_round20_eval import GAINS
    source = (ROOT / "scripts/pilot_content_check.py").read_text()
    for path in ("apply_gains", "make_post", "multiprocessing", "run_jobs", "decoder_gains="):
        assert path not in source, path                               # no other route for a gain, no workers
    assert source.count("load_generator(") == 1
    assert "smooth" in source.split("def postprocess")[1][:300]
    store, sha, *_ = stage2_store
    put_gains(store)
    temperature = dict(COMPARATOR["part_temperature"])
    on = load_generator(store, "prior.pt", sha, "cpu", temperature=temperature)[2]
    off = load_generator(store, "prior.pt", sha, "cpu", temperature=temperature, decoder_gains=False)[2]
    assert on.decoder_gains and off.decoder_gains is None
    for a, b in ((on("cut the onion", 24., 1., 1.5, 50, 7), off("cut the onion", 24., 1., 1.5, 50, 7)),
                 (on.batch(["cut the onion"], 24., [1.], 1.5, 1000)[0], off.batch(["cut the onion"], 24., [1.], 1.5, 1000)[0])):
        regained = apply_gains(b, GAINS)
        assert (a.channels["body"].values == regained.channels["body"].values).all()   # the stub's gains are body and head
        assert (a.channels["body"].values != b.channels["body"].values).any()


@pytest.mark.parametrize("prior,words,checkpoint", [
    ("w14b", f"--word-split-key {O}/round14/r5-word-split.json --word-split-sha256 "
             "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a --recognizer-run sign-recognizer-v2", W14B),
    ("r16", f"--word-split-key {O}/round14/r5-word-split.json --word-split-sha256 "
            "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a "
            f"--word-list-key {O}/round14/r5-word-split.json --word-list-sha256 "
            "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a --recognizer-run sign-recognizer-v3d", R16),
])
def test_the_e2e_phase_is_the_stored_pilots_command_without_gains(tmp_path, pilot, prior, words, checkpoint):  # noqa: F811
    code, out, commands = job(tmp_path, "e2e", prior=prior, no_gains="1", best_of="0", pilots=False)
    line = next(c for c in commands if "pilot_content_check.py" in c)
    key = f"{OUT}/pilot-{prior}-nogains-val1000.json"
    assert code == 0 and f"R20B e2e {key} stubsha" in out and "R20-PHASE-B-DONE 0" in out and f"pin {checkpoint}" in commands
    assert line == (f"scripts/pilot_content_check.py --checkpoint-key {checkpoint} --mix-key {COMPARATOR['generator_mix']['key']} "
                    f"--mix-sha256 {COMPARATOR['generator_mix']['sha256']} --vocab-mix-key {COMPARATOR['mix_key']} "
                    f"--vocab-mix-sha256 {COMPARATOR['mix_sha256']} --clips 1000 "
                    "--part-temperature body=1.1,left_hand=1.1,right_hand=1.1,face=1.0 --guidance 1.5 "
                    f"--translator-run back-translator-kp2d-uwb-ft-v1 {words} --device cuda --no-decoder-gains --output-key {key}")
    store, _ = pilot
    store.put(key, b"{}")                                           # the script's own parser takes the list
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pcc.main(line.split()[1:])
    # one pilot per prior, whatever the arm's cache
    other = next(c for c in job(tmp_path, "e2e", prior=prior, no_gains="1", best_of="8", pilots=False)[2] if "pilot_content" in c)
    assert other == line
    # stored: not rerun
    code, out, commands = job(tmp_path, "e2e", prior=prior, no_gains="1", best_of="0")
    assert code == 0 and "stored:" in out and not scripts_run(commands)


def test_e2e_refusals_and_smoke(tmp_path):
    code, out, commands = job(tmp_path, "e2e", no_gains="0", best_of="8")
    assert code == 1 and "only for NO_GAINS=1" in out and not commands
    code, out, commands = job(tmp_path, "e2e", sentence_set="c20")
    assert code == 1 and "SET=val1000 only" in out and not commands
    code, out, commands = job(tmp_path, "e2e", gpu=False, pilots=False)
    assert code == 1 and "needs a GPU" in out and not scripts_run(commands)
    code, out, commands = job(tmp_path, "e2e", smoke=True, gpu=False, pilots=False)
    line = next(c for c in commands if "pilot_content_check.py" in c)
    assert code == 0 and "--first-clips 5" in line and re.search(rf"--output-key {OUT}/smoke/\d{{8}}T\d{{6}}Z/pilot-w14b-nogains-val1000", line)
    assert not any("round20/c20" in c for c in commands) and not any(" stack-check " in c for c in commands)


def test_a_no_gains_arm_needs_its_own_end_to_end(tmp_path):
    index = f"{OUT}/cache-w14b-nogains-val1000-n8/index.json"
    code, out, commands = job(tmp_path, "run", stored=[index], pilots=False)
    assert code == 1 and "run the e2e phase first" in out and not scripts_run(commands)
    code, out, commands = job(tmp_path, "verdict", stored=[f"{OUT}/w14b-nogains-n8-val1000.json"], pilots=False)
    assert code == 1 and "without gains" in out and not scripts_run(commands)
    code, out, commands = job(tmp_path, "verdict", stored=[f"{OUT}/w14b-nogains-n8-val1000.json"])
    line = next(c for c in commands if " verdict " in c)
    assert code == 0 and f"--end-to-end-key {OUT}/pilot-w14b-nogains-val1000.json --end-to-end-sha256 stubsha " in line
    assert f"--stored-pilot-key {O}/round14b/pilot-w14b-step30000.json --stored-pilot-sha256 090f68b1" in line
    # a gained arm needs no e2e and names the stored pilot
    index = f"{OUT}/cache-w14b-gains-val1000-n8/index.json"
    code, out, commands = job(tmp_path, "run", no_gains="0", stored=[index], pilots=False)
    assert code == 0 and f"--end-to-end-report {O}/round14b/pilot-w14b-step30000.json " in next(c for c in commands if " merge " in c)
    code, out, commands = job(tmp_path, "verdict", no_gains="0", stored=[f"{OUT}/w14b-gains-n8-val1000.json"], pilots=False)
    assert code == 0 and "--stored-pilot-key" not in next(c for c in commands if " verdict " in c)
    # C20 has no pilot at all
    verdict = f"{OUT}/verdict-w14b-nogains-n8-val1000.json"
    code, out, commands = job(tmp_path, "run", sentence_set="c20", stored=[verdict, f"{OUT}/cache-w14b-nogains-c20-n8/index.json"],
                              pilots=False)
    assert code == 0 and "--end-to-end-report" not in next(c for c in commands if " merge " in c)


@pytest.mark.parametrize("edit,match", [
    (lambda r: r.update(pilot=r["stored_pilot"]) or r["arm"]["forms"]["end_to_end"].update(
        hypotheses=r["stored_pilot"]["meaning"]["generated"]["hypotheses"]) or r, "lenient"),   # the gained pilot for a no-gains arm
    (lambda r: r["pilot"].update(first_clips=5) or r, "a smoke"),
    (lambda r: r.update(stored_pilot=None) or r, "--stored-pilot-key"),
    (lambda r: r["pilot"].update(guidance=2.) or r, "differs from the stored pilot in .'guidance'"),
    (lambda r: r["pilot"].update(clips=300) or r, "differs from the stored pilot in .'clips'"),
    (lambda r: r["pilot"].update(part_temperature=dict(body=1.)) or r, "differs from the stored pilot in .'part_temperature'"),
    (lambda r: r["stored_pilot"].update(checkpoint_key=R16) or r, "differs from the stored pilot in .'checkpoint_key'"),
    (lambda r: r["stored_pilot"]["meaning"].update(references=["x"] * 6) or r, "differs from the stored pilot in .'references'"),
])
def test_a_no_gains_arm_refuses_any_other_end_to_end(edit, match):
    with pytest.raises(SystemExit, match=match):
        judge(edit(reports()))


def test_a_gained_arm_refuses_a_pilot_without_gains():
    r = gained_arm(reports())
    r["pilot"] = dict(r["pilot"], no_decoder_gains=True)
    with pytest.raises(SystemExit, match="the end-to-end pilot does not"):
        judge(r, no_gains=False)


# ---- review fix 2: one arm, and the stored D20 reads ----------------------------------------------------------------------------

def test_other_arms_reads_names_on_this_set_only():
    keys = [f"o/{k}" for k in ("w14b-nogains-n8-val1000.json", "w14b-nogains-n8-val1000-shard0.json",     # this arm
                               "verdict-r16-gains-n0-val1000.json", "pilot-w14b-nogains-val1000.json",   # no arms
                               "cache-w14b-gains-val1000-n4/index.json", "r16-gains-n4-c20.json")]       # other set
    keys += ["o/smoke/20261031T000000Z/r16-gains-n0-val1000.json", "other/r16-gains-n0-val1000.json"]     # elsewhere
    assert pv.other_arms(keys, "o", "val1000", "w14b-nogains-n8-val1000") == []
    assert pv.other_arms(keys + ["o/r16-gains-n0-val1000-shard3.json"], "o", "val1000", "w14b-nogains-n8-val1000") == ["r16-gains-n0-val1000"]
    assert pv.other_arms(keys + ["o/w14b-nogains-n8-lp6-val1000.json"], "o/", "val1000", "w14b-nogains-n8-val1000") == \
        ["w14b-nogains-n8-lp6-val1000"]                                                  # the low-passed arm is another arm
    assert pv.other_arms(keys, "o", "c20", "w14b-nogains-n8-c20") == ["r16-gains-n4-c20"]


FIRST_ARM, SECOND_ARM, DECLARED = "w14b-gains-n16-val1000", "w14b-nogains-n16-val1000", "cf4b1559a3"


def test_one_arm_cli(tmp_path, capsys):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'a'}")
    argv = ["one-arm", "--store", f"dir:{tmp_path / 'a'}", "--out", "o", "--set", "val1000", "--report", SECOND_ARM]
    assert pv.main(argv) == 0
    with pytest.raises(SystemExit, match="beside exactly w14b-gains-n16-val1000, but this set holds \\[\\]"):   # no first arm
        pv.main([*argv, "--declared-replacement", DECLARED])
    store.put(f"o/{FIRST_ARM}.json", b"{}")
    with pytest.raises(SystemExit, match="no amendment declares"):
        pv.main(argv)
    for bad in ("v2", "cf4b155", "cf4b1559a3f", "ABC1234", "abc1234; rm"):         # the commit's other spellings open nothing
        with pytest.raises(SystemExit, match="declares no such arm"):
            pv.main([*argv, "--declared-replacement", bad])
    third = ["one-arm", "--store", f"dir:{tmp_path / 'a'}", "--out", "o", "--set", "val1000", "--report", "w14b-nogains-n8-val1000"]
    with pytest.raises(SystemExit, match="declares no such arm"):                  # the commit does not open another arm
        pv.main([*third, "--declared-replacement", DECLARED])
    capsys.readouterr()
    assert pv.main([*argv, "--declared-replacement", DECLARED]) == 0              # the second arm, beside exactly the first
    assert f"DECLARED REPLACEMENT {DECLARED}" in capsys.readouterr().out
    store.put(f"o/{SECOND_ARM}.json", b"{}")
    assert pv.main([*argv, "--declared-replacement", DECLARED]) == 0              # its own keys are not another arm
    with pytest.raises(SystemExit, match="no amendment declares"):                 # a third arm, undeclared
        pv.main(third)
    store.put("o/r16-nogains-n16-val1000-shard1.json", b"{}")
    with pytest.raises(SystemExit, match="but this set holds"):                    # a third arm's shard shuts the second's rerun too
        pv.main([*argv, "--declared-replacement", DECLARED])


def test_the_verdict_refuses_a_second_arm(tmp_path, capsys):
    store, argv = store_all(tmp_path, reports())
    store.put("r/w14b-gains-n8-val1000-shard0.json", b"{}")
    with pytest.raises(SystemExit, match="no amendment declares"):
        pv.main(argv)
    assert not store.exists("r/verdict.json")
    r = reports()
    r["arm"]["declared_replacement"] = DECLARED                                    # the report's claim alone opens nothing
    store, argv = store_all(tmp_path, r, arm_key="r/arm2.json", comparator_key="r/w14b2.json", pilot_key="r/pilot2.json")
    with pytest.raises(SystemExit, match="the job names None"):
        pv.main(argv)
    with pytest.raises(SystemExit, match="the report records declared replacement None"):
        store3, argv3 = store_all(tmp_path, reports(), arm_key="r/arm3.json", comparator_key="r/w14b3.json", pilot_key="r/pilot3.json")
        pv.main([*argv3, "--declared-replacement", DECLARED])


def test_run_refuses_a_second_arm_and_logs_the_declared_one(tmp_path):
    index = f"{OUT}/cache-w14b-nogains-val1000-n16/index.json"
    second = dict(prior="w14b", no_gains="1", best_of="16", stored=[index])
    code, out, commands = job(tmp_path, "run", **second, STUB_ONE_ARM="1")
    assert code == 1 and "already holds another arm" in out and "R20-PHASE-B-DONE 1" in out
    assert not any("round15_compose_eval" in c for c in commands)                # before any shard
    code, out, commands = job(tmp_path, "run", **second)
    check = next(k for k, c in enumerate(commands) if " one-arm " in c)
    assert commands[check] == f"scripts/round20_phase_b_verdict.py one-arm --out {OUT} --set val1000 --report {SECOND_ARM}"
    assert check < next(k for k, c in enumerate(commands) if " run " in c)
    assert "--declared-replacement" not in " ".join(commands) and "DECLARED" not in out
    code, out, commands = job(tmp_path, "run", **second, R20B_DECLARED_REPLACEMENT=DECLARED)
    assert code == 0 and f"R20B DECLARED REPLACEMENT {DECLARED}: {SECOND_ARM}" in out
    assert next(c for c in commands if " one-arm " in c).endswith(f"--declared-replacement {DECLARED}")
    assert f"--declared-replacement {DECLARED} --output-key {OUT}/{SECOND_ARM}.json" in next(c for c in commands if " merge " in c)
    assert not any("--declared-replacement" in c for c in commands if "compose_eval.py run" in c)
    # the first arm's keys are never written: the run's and the verdict's outputs are the second arm's
    written = [c.split("--output-key ")[1].split()[0] for c in commands if "--output-key" in c]
    assert len(written) == 5 and all(SECOND_ARM in k for k in written) and not any(FIRST_ARM in c for c in written)
    assert not any(FIRST_ARM in c for c in commands)                             # the job does not even name it; the verdict reads it
    # a smoke has its own prefix and no check; a stored arm is not checked again
    code, out, commands = job(tmp_path, "run", smoke=True, gpu=False, CACHE_KEY="k/i.json", stored=["k/i.json"])
    assert code == 0 and not any(" one-arm " in c for c in commands)
    code, out, commands = job(tmp_path, "run", **dict(second, stored=[index, f"{OUT}/{SECOND_ARM}.json"]))
    assert code == 0 and not any(" one-arm " in c for c in commands)


@pytest.mark.parametrize("kwargs", [
    dict(prior="w14b", no_gains="0", best_of="16"),      # the first arm
    dict(prior="w14b", no_gains="1", best_of="8"),       # another N
    dict(prior="w14b", no_gains="1", best_of="0"),       # fresh words
    dict(prior="r16", no_gains="1", best_of="16"),       # another prior
    dict(prior="w14b", no_gains="1", best_of="16", LOWPASS="1"),
    dict(prior="w14b", no_gains="1", best_of="16", sentence_set="c20"),
])
@pytest.mark.parametrize("phase", ["run", "verdict", "cache", "e2e"])
def test_the_variable_opens_no_other_arm(tmp_path, kwargs, phase):
    code, out, commands = job(tmp_path, phase, R20B_DECLARED_REPLACEMENT=DECLARED, **kwargs)
    assert code == 1 and "ALERT" in out and "R20-PHASE-B-DONE 1" in out and not commands
    assert "declares no such arm" in out or phase in ("cache", "e2e")          # those have their own earlier refusals too


@pytest.mark.parametrize("commit", ["abc1234", "cf4b155", "cf4b1559a3f", "CF4B1559A3", "v2"])
def test_another_commit_opens_nothing(tmp_path, commit):
    code, out, commands = job(tmp_path, "run", prior="w14b", no_gains="1", best_of="16", R20B_DECLARED_REPLACEMENT=commit)
    assert code == 1 and "declares no such arm" in out and not commands


def test_the_variable_belongs_to_a_full_run_or_verdict(tmp_path):
    second = dict(prior="w14b", no_gains="1", best_of="16", R20B_DECLARED_REPLACEMENT=DECLARED)
    for phase, kwargs in (("cache", {}), ("e2e", {}), ("run", dict(smoke=True))):
        code, out, commands = job(tmp_path, phase, **second, **kwargs)
        assert code == 1 and "belongs to a full run or verdict" in out and not commands


def test_the_stack_check_runs_under_the_declaration(tmp_path):
    index = f"{OUT}/cache-w14b-nogains-val1000-n16/index.json"
    second = dict(prior="w14b", no_gains="1", best_of="16")
    for phase, stored in (("run", [index]), ("verdict", [f"{OUT}/{SECOND_ARM}.json"])):
        code, out, commands = job(tmp_path, phase, **second, stored=stored, R20B_DECLARED_REPLACEMENT=DECLARED)
        line = next(c for c in commands if " stack-check " in c)
        assert code == 0 and line.endswith(f"--declared-replacement {DECLARED}")
        assert f"R20B STACK-CHECK under declared replacement {DECLARED} on {SECOND_ARM}" in out
        # the script itself decides: a crash or a refused name still stops the job
        code, out, commands = job(tmp_path, phase, **second, stored=stored, R20B_DECLARED_REPLACEMENT=DECLARED, STUB_STACK="1")
        assert code == 1 and "do not allow this arm" in out
        # unset: as before, a refusal stops it and the flag is absent
        code, out, commands = job(tmp_path, phase, **second, stored=stored, STUB_STACK="1")
        assert code == 1 and "do not allow this arm" in out
        assert "--declared-replacement" not in next(c for c in commands if " stack-check " in c)
    code, out, commands = job(tmp_path, "verdict", **second, stored=stored, R20B_DECLARED_REPLACEMENT=DECLARED)
    line = next(c for c in commands if " verdict " in c)
    assert f"--declared-replacement {DECLARED} --phase-a-out {OUT_A} " in line
    assert f"--end-to-end-key {OUT}/pilot-w14b-nogains-val1000.json " in line and f"--output-key {OUT}/verdict-{SECOND_ARM}.json" in line
    # the script's parser takes it, and reaches the comparator's hash
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 'p'}")
    store.put(f"{OUT}/{SECOND_ARM}.json", b"{}")
    store.put(CMP, b"{}")
    with pytest.raises(SystemExit, match="does not match --comparator-sha256"):
        pv.main(line.split()[1:] + ["--store", f"dir:{tmp_path / 'p'}"])


def test_stack_check_cli_overrides_only_the_declared_arm(tmp_path, capsys):
    argv = ["stack-check", "--store", f"dir:{tmp_path / 's'}", "--phase-a-out", "a", "--prior", "w14b", "--w14b-checkpoint-key", W14B,
            "--r16-checkpoint-key", R16, "--no-gains", "1", "--best-of", "16", "--lowpass", "0", "--sentences-sha256", D20_SHA]
    with pytest.raises(SystemExit, match="unread: a0-w14b-r10"):
        pv.main(argv)
    capsys.readouterr()
    assert pv.main([*argv, "--declared-replacement", DECLARED]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == dict(overridden=True, by=DECLARED, refusal="unread: a0-w14b-r10 is not stored")
    other = list(argv)
    other[other.index("--best-of") + 1] = "8"
    with pytest.raises(SystemExit, match="declares no such arm"):
        pv.main([*other, "--declared-replacement", DECLARED])
    with pytest.raises(SystemExit, match="declares no such arm"):
        pv.main([*argv, "--declared-replacement", "abc1234"])
    assert pv.checked_under_declaration(DECLARED, lambda: dict(decision="x")) == dict(overridden=False, by=DECLARED, result=dict(decision="x"))


def test_the_job_and_the_verdict_declare_the_same_arm():
    text = SCRIPT.read_text()
    spec = pv.DECLARED_ARMS
    assert list(spec) == [DECLARED] and spec[DECLARED]["report"] == SECOND_ARM and spec[DECLARED]["replaces"] == FIRST_ARM
    assert f"    {DECLARED}:w14b:1:16:0:val1000) ;;" in text and len(re.findall(r"^    [0-9a-f]{7,}:", text, flags=re.M)) == 1


# ---- amendment 6: c20_arm_choice -------------------------------------------------------------------------------------------------

def first_and_second(*, hyps=REFS, ll=-2.0, first_ll=-2.1, wrist=2.1, first_wrist=2.6, delta=0.):
    """Two arms of one 1,000 as merged reports hold them: per-clip likelihood, and spotting in the shards only."""
    r = reports(hyp_arm=hyps, delta=delta)
    arm = r["arm"]
    arm["likelihood"]["per_form"] = {FORM: [ll] * 6, "blank": [-3.] * 6}
    arm["motion_ratio_to_real"][FORM]["wrist_speed"] = wrist
    first = copy.deepcopy(arm)
    first["settings"]["round20"]["word_cache"] = cache_settings(no_gains=False)
    first["forms"][FORM] = hyp_form(HYP_ARM)
    first["likelihood"]["per_form"] = {FORM: [first_ll] * 6, "blank": [-3.] * 6}
    first["motion_ratio_to_real"][FORM]["wrist_speed"] = first_wrist
    return dict(arm=arm, first=first, shards_arm=[shard(arm["settings"], delta)], shards_first=[shard(first["settings"])])


def choose(**over):
    c = first_and_second(**over)
    return pv.c20_arm_choice(c["arm"], c["first"], shards_arm=c["shards_arm"], shards_first=c["shards_first"],
                             spec=pv.DECLARED_ARMS[DECLARED], boot=50)


def test_the_choice_picks_the_stack_when_all_hold():
    v = choose()
    assert v["all_hold"] and v["arm"] == SECOND_ARM and not v["missing"] and "whether C20 runs" in v["note"]
    assert set(v["items"]) == {"bleu_at_least_2_55", "paired_likelihood_above_first_arm", "v2_spotting_against_first_arm",
                               "wrist_closer_to_1_than_first_arm"}
    assert all(i["available"] and i["passed"] for i in v["items"].values())
    assert v["items"]["paired_likelihood_above_first_arm"]["lower_bound"] == pytest.approx(.1)
    assert v["items"]["wrist_closer_to_1_than_first_arm"] == dict(available=True, passed=True, ratio_to_real=2.1, first_arm_ratio_to_real=2.6)
    assert "difference" in v["paired_bleu_against_first_arm"]


@pytest.mark.parametrize("over,item", [
    (dict(hyps=["word"] * 6), "bleu_at_least_2_55"),
    (dict(ll=-2.1), "paired_likelihood_above_first_arm"),                       # equal: the lower bound is not above 0
    (dict(ll=-2.2), "paired_likelihood_above_first_arm"),
    (dict(delta=-.02), "v2_spotting_against_first_arm"),                        # at the floor
    (dict(delta=-.05), "v2_spotting_against_first_arm"),
    (dict(wrist=2.6), "wrist_closer_to_1_than_first_arm"),                      # equal is not closer
    (dict(wrist=3.0), "wrist_closer_to_1_than_first_arm"),
    (dict(wrist=.1, first_wrist=1.2), "wrist_closer_to_1_than_first_arm"),      # closer means to 1, from either side
])
def test_each_failure_keeps_the_first_arm(over, item):
    v = choose(**over)
    assert v["all_hold"] is False and v["arm"] == FIRST_ARM and not v["missing"] and v["items"][item]["passed"] is False
    assert all(i["passed"] for k, i in v["items"].items() if k != item)


@pytest.mark.parametrize("point,met", [(2.5496, False), (2.55, True), (2.5504, True)])
def test_the_choice_reads_bleu_unrounded(monkeypatch, point, met):
    import sentence_exposure_diagnostic as sed
    c = first_and_second()
    c["arm"]["forms"][FORM]["bleu"] = round(point, 3)                          # 3 decimals round all of these to 2.55
    monkeypatch.setattr(sed, "bleu_from_stats", lambda stats: point)
    v = pv.c20_arm_choice(c["arm"], c["first"], shards_arm=c["shards_arm"], shards_first=c["shards_first"],
                          spec=pv.DECLARED_ARMS[DECLARED], boot=20)
    assert v["items"]["bleu_at_least_2_55"]["passed"] is met and v["arm"] == (SECOND_ARM if met else FIRST_ARM)


def test_the_choice_at_its_edges():
    assert choose(delta=-.019)["arm"] == SECOND_ARM
    assert choose(wrist=.9, first_wrist=1.2)["arm"] == SECOND_ARM                # below 1 and nearer
    # BLEU is not compared directly: below the first arm's, above 2.55, the stack can still be chosen
    assert choose(hyps=HYP_ARM)["items"]["bleu_at_least_2_55"]["passed"] == (corpus_bleu(HYP_ARM, REFS)["bleu"] >= 2.55)
    c = first_and_second()
    c["first"]["forms"][FORM] = hyp_form(REFS)
    c["arm"]["forms"][FORM] = hyp_form(HYP_ARM)
    assert corpus_bleu(HYP_ARM, REFS)["bleu"] >= 2.55
    v = pv.c20_arm_choice(c["arm"], c["first"], shards_arm=c["shards_arm"], shards_first=c["shards_first"],
                          spec=pv.DECLARED_ARMS[DECLARED], boot=50)
    assert v["paired_bleu_against_first_arm"]["difference"] < 0 and v["arm"] == SECOND_ARM


@pytest.mark.parametrize("edit,says", [
    (lambda c: c["arm"].pop("likelihood"), "paired_likelihood_above_first_arm"),
    (lambda c: c["first"].pop("likelihood"), "paired_likelihood_above_first_arm"),
    (lambda c: c.update(shards_arm=None), "v2_spotting_against_first_arm"),
    (lambda c: c.update(shards_first=None), "v2_spotting_against_first_arm"),
    (lambda c: c["arm"].pop("motion_ratio_to_real"), "wrist_closer_to_1_than_first_arm"),
    (lambda c: c["first"]["motion_ratio_to_real"].pop(FORM), "wrist_closer_to_1_than_first_arm"),
])
def test_a_missing_field_chooses_no_arm(edit, says):
    c = first_and_second()
    edit(c)
    v = pv.c20_arm_choice(c["arm"], c["first"], shards_arm=c["shards_arm"], shards_first=c["shards_first"],
                          spec=pv.DECLARED_ARMS[DECLARED], boot=50)
    assert v["arm"] is None and v["all_hold"] is False and says in " ".join(v["missing"])


def test_the_choice_refuses_arms_that_do_not_pair():
    c = first_and_second()
    c["first"]["references"] = list(reversed(REFS))
    with pytest.raises(SystemExit, match="not paired"):
        pv.c20_arm_choice(c["arm"], c["first"], shards_arm=c["shards_arm"], shards_first=c["shards_first"],
                          spec=pv.DECLARED_ARMS[DECLARED], boot=50)


def test_the_declared_verdict_on_real_merges(tmp_path, stage2_store, likelihood_stub, monkeypatch):  # noqa: F811
    """Both arms really merged, with the job's flags and keys, as the first arm is stored: 4-shard-style names, per-clip likelihood."""
    import torch
    import validate_sign_recognizer as vsr
    store, sha, s1, *_ = stage2_store
    put_gains(store)
    pilot_report(store, s1)
    gained_pilot = json.loads(store.get("pilot.json"))
    store.put("pilot-nogains.json", json.dumps(dict(gained_pilot, no_decoder_gains=True)).encode())
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))
    monkeypatch.setattr(vsr, "logits_of", lambda model, states, device, batch=64: torch.zeros(len(states), len(labels)))
    root = f"dir:{tmp_path / 'store'}"

    def cache(prefix, *extra):
        common = ["--store", root, "--prefix", prefix]
        assert cw.main(["run", *common, "--mix-key", "mix.json", "--mix-sha256", sha, "--clips", "5", "--checkpoint-key", "prior.pt",
                        "--samples", "4", "--choose", "4", "--device", "cpu", *extra]) == 0
        assert cw.main(["merge", *common]) == 0
        return ["--word-cache-key", f"{prefix}/index.json", "--word-cache-n", "4", "--word-cache-sha256",
                hashlib.sha256(store.get(f"{prefix}/index.json")).hexdigest()]

    def arm(name, words, pilot, *merge):
        args = run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key", "source.json", "--shards", "2")
        k = args.index("--checkpoint-key")
        args = args[:k] + args[k + 2:]
        for s in (0, 1):
            assert ce.main([*args, "--shard", str(s), "--likelihood", *words, "--output-key", f"o/{name}-shard{s}.json"]) == 0
        assert ce.main(["merge", "--store", root, "--bootstrap", "50", "--end-to-end-report", pilot, "--paired-with", "o/w14b.json",
                        "--shard-keys", f"o/{name}-shard0.json,o/{name}-shard1.json", *merge, "--output-key", f"o/{name}.json"]) == 0
    for s in (0, 1):
        assert ce.main(run_args(tmp_path, sha, "--variants", FORM, "--clips", "5", "--stage1-report-key", "source.json", "--shards", "2",
                                "--shard", str(s), "--output-key", f"o/w14b-shard{s}.json")) == 0
    assert ce.main(["merge", "--store", root, "--bootstrap", "50", "--end-to-end-report", "pilot.json",
                    "--shard-keys", "o/w14b-shard0.json,o/w14b-shard1.json", "--output-key", "o/w14b.json"]) == 0
    first, second = "w14b-gains-n4-val1000", "w14b-nogains-n4-val1000"
    arm(first, cache("c/gains"), "pilot.json")
    arm(second, cache("c/nogains", "--no-decoder-gains"), "pilot-nogains.json", "--declared-replacement", "abc1234")
    report = json.loads(store.get(f"o/{second}.json"))
    assert report["declared_replacement"] == "abc1234" and "second read" in report["set_read"]
    assert "declared_replacement" not in json.loads(store.get(f"o/{first}.json"))
    monkeypatch.setitem(pv.DECLARED_ARMS, "abc1234", dict(
        report=second, replaces=first, replaces_sha256=hashlib.sha256(store.get(f"o/{first}.json")).hexdigest(), prior="w14b",
        no_gains=True, best_of=4, lowpass=False))
    argv = ["verdict", "--store", root, "--arm-key", f"o/{second}.json", "--comparator-key", "o/w14b.json",
            "--comparator-sha256", hashlib.sha256(store.get("o/w14b.json")).hexdigest(), "--end-to-end-key", "pilot-nogains.json",
            "--end-to-end-sha256", hashlib.sha256(store.get("pilot-nogains.json")).hexdigest(), "--stored-pilot-key", "pilot.json",
            "--stored-pilot-sha256", hashlib.sha256(store.get("pilot.json")).hexdigest(), "--out", "o", "--report", second, "--prior", "w14b",
            "--prior-checkpoint-key", "prior.pt", "--no-gains", "1", "--best-of", "4", "--set", "val1000", "--bootstrap", "50",
            "--output-key", f"o/verdict-{second}.json", "--declared-replacement", "abc1234", "--phase-a-out", "nowhere",
            "--w14b-checkpoint-key", "prior.pt", "--r16-checkpoint-key", "r16.pt", "--d20-sha256", D20_SHA]
    with pytest.raises(SystemExit):                                              # the job's four declaration flags are needed
        pv.main(argv[:argv.index("--phase-a-out")])
    before = {k: store.get(f"o/{first}{k}.json") for k in ("", "-shard0", "-shard1")}
    assert pv.main(argv) == 0
    v = json.loads(store.get(f"o/verdict-{second}.json"))
    assert v["declared_replacement"] == "abc1234" and v["other_arms"] == [first] and "second read" in v["set_read"] and first in v["set_read"]
    assert v["stack_check"] == dict(overridden=True, by="abc1234", refusal="unread: a0-w14b-r10 is not stored")
    assert set(v["items"]) == {"bleu_vs_w14b_composition", "bleu_vs_other_sentence_words", "bleu_vs_own_end_to_end",
                               "v2_spotting_vs_w14b_composition", "wrist_speed_reported"}
    assert "WITHOUT gains" in v["items"]["bleu_vs_own_end_to_end"]["note"]
    choice = v["c20_arm_choice"]
    assert choice["arm"] in (first, second) and not choice["missing"] and choice["all_hold"] == (choice["arm"] == second)
    assert choice["items"]["paired_likelihood_above_first_arm"]["available"] and choice["items"]["v2_spotting_against_first_arm"]["available"]
    assert {k: store.get(f"o/{first}{k}.json") for k in before} == before        # the first arm is untouched
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pv.main(argv)
    again = [*argv[:argv.index("--output-key") + 1], "o/other.json", *argv[argv.index("--output-key") + 2:]]
    # the job's arm must be the table's, whatever the report says
    for flag, value in (("--best-of", "8"), ("--no-gains", "0"), ("--prior", "r16")):
        other = list(again)
        other[other.index(flag) + 1] = value
        with pytest.raises(SystemExit, match="abc1234 declares"):
            pv.main(other)
    with pytest.raises(SystemExit, match="abc1234 declares"):
        pv.main([*again, "--lowpass", "1"])
    # a tampered first arm: refused before anything is stored
    good = pv.DECLARED_ARMS["abc1234"]
    monkeypatch.setitem(pv.DECLARED_ARMS, "abc1234", dict(good, replaces_sha256="0" * 64))
    with pytest.raises(SystemExit, match="first arm's declared hash"):
        pv.main(again)
    monkeypatch.setitem(pv.DECLARED_ARMS, "abc1234", good)
    # the first arm's shard gone: the choice cannot be read, so nothing is stored
    shard0 = tmp_path / f"store/o/{first}-shard0.json"
    kept = shard0.read_bytes()
    shard0.unlink()
    assert pv.main(again) == 2 and not store.exists("o/other.json")
    shard0.write_bytes(kept)
    # a third arm's shard stops the second arm's verdict
    store.put("o/r16-gains-n0-val1000-shard0.json", b"{}")
    with pytest.raises(SystemExit, match="but this set holds"):
        pv.main(again)
    assert not store.exists("o/other.json")


@pytest.mark.parametrize("prior,no_gains,best_of", [("w14b", "1", "8"), ("w14b", "1", "0"), ("w14b", "0", "4"), ("r16", "0", "0"),
                                                    ("r16", "1", "16")])
def test_only_run_and_verdict_consult_the_stored_reads(tmp_path, prior, no_gains, best_of):
    gains = "nogains" if no_gains == "1" else "gains"
    report = f"{OUT}/{prior}-{gains}-n{best_of}-val1000"
    stored = [report + ".json", f"{OUT}/cache-{prior}-{gains}-val1000-n{best_of}/index.json"]
    for phase in ("run", "verdict"):
        code, out, commands = job(tmp_path, phase, prior=prior, no_gains=no_gains, best_of=best_of, stored=stored)
        line = next(c for c in commands if " stack-check " in c)
        assert code == 0 and line == (
            f"scripts/round20_phase_b_verdict.py stack-check --phase-a-out {OUT_A} --prior {prior} --w14b-checkpoint-key {W14B} "
            f"--r16-checkpoint-key {R16} --no-gains {no_gains} --best-of {best_of} --lowpass 0 --sentences-sha256 "
            "d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe")
        code, out, commands = job(tmp_path, phase, prior=prior, no_gains=no_gains, best_of=best_of, stored=stored, STUB_STACK="1")
        assert code == 1 and "do not allow this arm" in out and "R20-PHASE-B-DONE 1" in out and not scripts_run(commands)
    # a cache and an e2e are built ahead: no D20 read is opened, or even looked for, and a failing check cannot reach them
    phases = (["cache"] if best_of != "0" else []) + (["e2e"] if no_gains == "1" else [])
    for phase in phases:
        code, out, commands = job(tmp_path, phase, prior=prior, no_gains=no_gains, best_of=best_of, pilots=False, STUB_STACK="1")
        assert code == 0 and "R20-PHASE-B-DONE 0" in out and not any(" stack-check " in c or "phase-a" in c for c in commands)


def test_the_jobs_exists_never_asks_for_a_d20_read():
    text = SCRIPT.read_text()
    assert "exists $A/" not in text and "$A/" not in text and " --phase-a-out $A " in text
    assert not re.search(r"(a0|a4|a7|a8|a9)-[a-z0-9-]*r10", text.split("# ---- the sentences")[1])


def test_lowpass_reaches_the_check_and_a_smoke_never_does(tmp_path):
    code, out, commands = job(tmp_path, "run", **STACK, stored=[STACK_INDEX], LOWPASS="1", STUB_STACK="1")
    assert code == 1 and "do not allow" in out and not any("compose_eval" in c for c in commands)
    code, out, commands = job(tmp_path, "run", **STACK, stored=[STACK_INDEX], LOWPASS="1")
    assert "--lowpass 1 " in next(c for c in commands if " stack-check " in c)
    code, out, commands = job(tmp_path, "run", smoke=True, gpu=False, CACHE_KEY="k/i.json", stored=["k/i.json"], STUB_STACK="1")
    assert code == 0 and not any(" stack-check " in c for c in commands)


D20_SHA = "d" * 64
A0 = ["the cat ran on a rug", "a dog sat in a lake", "she has a long book", "we go to the town", "he ate a cold soup",
      "they sing at the start"]
LOW, MID, TOP = A0, HYP_ARM, REFS          # BLEU rises along these (LOW reads 0)
LOWER = ["the cat sat on a mat", "a dog ran on the hill", "she reads a tale at home", "we walk a long road", "he made a pie",
         "they sing a tune now"]               # above LOW and below MID
HYP_LOW2 = ["the cat sat on a bed", "a dog ran off", "she reads", "we walk", "he made", "they sing"]    # above LOW and below LOWER


def d20(checkpoint=W14B, *, n=0, gains_off=False, hyps=MID, ll=-2.5, lowpass=None, lowpass_ll=None, sentences=D20_SHA):
    """A stored D20 read: per-sentence likelihood ``ll``, blank at -3."""
    cache = dict(n=n, checkpoint_key=checkpoint, **({"no_decoder_gains": True} if gains_off else {})) if n else None
    r20 = dict(sentences=dict(sha256=sentences), vocabulary_mix=None, lowpass_hz=None, retime="reference", word_cache=cache,
               **({"no_decoder_gains": True} if gains_off and not n else {}))
    forms = {FORM: hyp_form(hyps), "blank": hyp_form(["the"] * 6)}
    per = {FORM: [ll] * 6, "blank": [-3.] * 6}
    if lowpass is None and n and gains_off:      # a stack read always has its @lowpass; by default it improves nothing
        lowpass, lowpass_ll = hyps, ll
    if lowpass is not None:
        forms[f"{FORM}@lowpass"] = hyp_form(lowpass)
        per[f"{FORM}@lowpass"] = [lowpass_ll] * 6
    return dict(settings=dict(seed=0, translator_key="t.pt", recognizer_run="sign-recognizer-v2",
                              checkpoint_key=None if n else checkpoint, round20=r20),
                references=REFS, signs=SIGNS, shards=[0], forms=forms, likelihood=dict(per_form=per))


def world(**over):
    """The stored reads: n16 is rule 1's N (+0.3 against a0), a7 is +0.05, the stack is above both, a8 misses amendment 3."""
    w = {"a0-w14b-r10": d20(hyps=LOW, ll=-2.5),
         "a8-r16-r10": d20(R16, hyps=LOW, ll=-2.6),
         "a7-nogains-r10": d20(gains_off=True, ll=-2.45),
         "a4-w14b-n4-r10": d20(n=4, ll=-2.4), "a4-w14b-n8-r10": d20(n=8, ll=-2.3), "a4-w14b-n16-r10": d20(n=16, ll=-2.2),
         "a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=TOP, ll=-2.1)}
    w.update(over)
    return {k: v for k, v in w.items() if v is not None}


def check(w, *, prior="w14b", no_gains=True, best_of=16, lowpass=False, a8_delta=0., r16_delta=0., shards=True):
    def load_shards(name, report):
        if not shards:
            return None
        return [shard(report["settings"], a8_delta if name.startswith("a8") else r16_delta if "-r16-" in name else 0.)]
    return pv.stack_check(w.get, load_shards, prior=prior, w14b_checkpoint=W14B, r16_checkpoint=R16, no_gains=no_gains,
                          best_of=best_of, lowpass=lowpass, sentences_sha256=D20_SHA)


# what today's reads looked like when A7 alone and the wrong N passed
A7_ALONE = dict(no_gains=True, best_of=0)
A4_AT = lambda n: dict(no_gains=False, best_of=n)   # noqa: E731


def test_the_stack_goes_at_rule_1s_n_and_the_check_says_what_held():
    out = check(world())
    assert out["decision"] == "stack, best of 16" and out["rule1"]["n"] == 16
    assert out["a8"]["met"] is False and out["a8"]["spotting_lower_bound"] == 0. and out["a8"]["spotting_floor"] == -.02
    assert out["rule3"]["passed"] and out["rule3"]["against_a7"]["both_improve"] and out["rule3"]["against_a4"]["both_improve"]
    assert out["rule1"]["arms"]["16"]["likelihood"] == .3 and out["a7"]["likelihood"] == .05


@pytest.mark.parametrize("n", [4, 8])
def test_a_stack_or_a4_at_another_n_is_refused(n):
    w = world(**{f"a9-nogains-n{n}-r10": d20(n=n, gains_off=True, hyps=TOP, ll=-2.1)})
    with pytest.raises(SystemExit, match="rule 1 chooses best of 16, not " + str(n)):
        check(w, best_of=n)
    with pytest.raises(SystemExit, match="rule 1 chooses best of 16, not " + str(n)):
        check(w, **A4_AT(n))


def test_rule_1_ties_and_arms_that_miss_the_rule():
    assert pv.rule1_n({4: dict(likelihood=.3001, both_improve=True), 8: dict(likelihood=.3004, both_improve=True),
                       16: dict(likelihood=.2, both_improve=True)}) == 4
    assert pv.rule1_n({4: dict(likelihood=.3, both_improve=True), 8: dict(likelihood=.301, both_improve=True)}) == 8
    assert pv.rule1_n({4: dict(likelihood=.9, both_improve=False), 8: dict(likelihood=.1, both_improve=True)}) == 8
    assert pv.rule1_n({4: dict(likelihood=.9, both_improve=False)}) is None
    # n16 highest on likelihood but BLEU not above a0's: rule 1 falls to n8, and the n8 stack is the one that must exist
    w = world(**{"a4-w14b-n16-r10": d20(n=16, hyps=LOW, ll=-2.2)})
    with pytest.raises(SystemExit, match="unread: a9-nogains-n8-r10"):
        check(w, best_of=8)
    w["a9-nogains-n8-r10"] = d20(n=8, gains_off=True, hyps=TOP, ll=-2.1)
    assert check(w, best_of=8)["rule1"]["n"] == 8


@pytest.mark.parametrize("gone", ["a0-w14b-r10", "a4-w14b-n4-r10", "a4-w14b-n8-r10", "a4-w14b-n16-r10", "a7-nogains-r10",
                                  "a9-nogains-n16-r10"])
def test_every_read_of_the_question_must_be_stored(gone):
    for arm in (dict(), A7_ALONE, A4_AT(16)):
        with pytest.raises(SystemExit, match=f"unread: {gone}"):
            check(world(**{gone: None}), **arm)


@pytest.mark.parametrize("edit", [dict(ll=-2.5), dict(ll=-2.2), dict(hyps=MID, ll=-2.1)])
def test_the_stack_must_be_above_both_single_levers(edit):
    """Likelihood below a7's or equal to a4-n16's, or BLEU equal to both: rule 3 is not met."""
    a9 = d20(n=16, gains_off=True, **{"hyps": TOP, **edit})
    with pytest.raises(SystemExit, match="both a7 and a4-n16"):
        check(world(**{"a9-nogains-n16-r10": a9}))


def test_a_single_lever_needs_the_stacks_stored_failure_and_must_be_the_better():
    failed = d20(n=16, gains_off=True, hyps=TOP, ll=-2.3)                         # below a4-n16
    # today's case: the stack is unread, so nothing single goes
    for arm in (A7_ALONE, A4_AT(16)):
        with pytest.raises(SystemExit, match="unread: a9-nogains-n16-r10"):
            check(world(**{"a9-nogains-n16-r10": None}), **arm)
    # the stack passes: no single lever
    for arm in (A7_ALONE, A4_AT(16)):
        with pytest.raises(SystemExit, match="the stack goes, not a single lever"):
            check(world(), **arm)
    # it failed: the better one by likelihood against a0 goes, the other does not
    w = world(**{"a9-nogains-n16-r10": failed})
    out = check(w, **A4_AT(16))
    assert out["decision"].startswith("a4 alone") and out["singles"] == dict(a4=.3, a7=.05) and out["rule3"]["passed"] is False
    with pytest.raises(SystemExit, match="better single lever is a4"):            # A7 at +0.05 though A4-n16 is +0.3
        check(w, **A7_ALONE)
    with pytest.raises(SystemExit, match="does not have both point estimates above 0 against both"):
        check(w)                                                                  # and the stack itself is refused
    w = world(**{"a7-nogains-r10": d20(gains_off=True, ll=-2.15), "a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=TOP, ll=-2.18)})
    assert check(w, **A7_ALONE)["decision"].startswith("a7 alone")
    with pytest.raises(SystemExit, match="better single lever is a7"):
        check(w, **A4_AT(16))
    w = world(**{"a7-nogains-r10": d20(gains_off=True, ll=-2.2), "a9-nogains-n16-r10": failed})
    with pytest.raises(SystemExit, match="tie on likelihood"):
        check(w, **A7_ALONE)
    # a7 missing its own rule cannot be the better single
    w = world(**{"a7-nogains-r10": d20(gains_off=True, hyps=LOW, ll=-2.0), "a9-nogains-n16-r10": failed})
    with pytest.raises(SystemExit, match="better single lever is a4"):
        check(w, **A7_ALONE)


def test_no_a4_arm_meets_the_rule():
    w = world(**{f"a4-w14b-n{n}-r10": d20(n=n, hyps=LOW, ll=-2.2) for n in (4, 8, 16)})
    assert check(w, **A7_ALONE)["decision"].startswith("a7 alone: no a4 arm")
    with pytest.raises(SystemExit, match="no best-of-N arm goes"):
        check(w, best_of=8)
    w["a7-nogains-r10"] = d20(gains_off=True, hyps=LOW, ll=-2.45)
    with pytest.raises(SystemExit, match="neither a4 nor a7"):
        check(w, **A7_ALONE)


def met_world(**over):
    """a8 meets amendment 3. W14b: a0 LOW, a7 and a4 LOWER, the stack MID; R16's stack and single arm above W14b's."""
    w = world(**{"a8-r16-r10": d20(R16, hyps=LOWER, ll=-2.4),
                 "a7-nogains-r10": d20(gains_off=True, hyps=LOWER, ll=-2.45),
                 **{f"a4-w14b-n{n}-r10": d20(n=n, hyps=LOWER, ll=-2.5 + .1 * k) for k, n in enumerate((4, 8, 16), 1)},
                 "a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=MID, ll=-2.1),
                 "a9-r16-nogains-n16-r10": d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.0),
                 "a4-r16-n16-r10": d20(R16, n=16, hyps=MID, ll=-2.1)})
    w.update(over)
    return {k: v for k, v in w.items() if v is not None}


W14B_STACK_FAILED = {"a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=MID, ll=-2.3)}       # likelihood below a4-n16's -2.2
R16_STACK = dict(prior="r16", no_gains=True, best_of=16)
R16_SINGLE = dict(prior="r16", no_gains=False, best_of=16)


def test_a8_stored_and_judged_first():
    for arm in (dict(), A7_ALONE, A4_AT(16), R16_STACK, R16_SINGLE):
        with pytest.raises(SystemExit, match="until a8 is judged"):               # not stored
            check(world(**{"a8-r16-r10": None}), **arm)
        with pytest.raises(SystemExit, match="cannot be judged"):                 # no spotting: not silently a failure
            check(met_world(), shards=False, **arm)


def test_a8_failing_needs_no_r16_read_and_w14b_goes():
    assert not any("-r16-n" in k or "a9-r16" in k for k in world())
    out = check(world())
    assert out["a8"]["met"] is False and "amendment5" not in out and out["decision"] == "stack, best of 16"
    for a8 in (d20(R16, hyps=LOW, ll=-2.4), d20(R16, ll=-2.5), d20(R16, ll=-2.6)):                # BLEU, or likelihood, not above
        assert check(world(**{"a8-r16-r10": a8}))["a8"]["met"] is False
    for delta in (-.02, -.05):                                                    # both estimates up, spotting at or below the floor
        out = check(world(**{"a8-r16-r10": d20(R16, ll=-2.4)}), a8_delta=delta)
        assert out["a8"]["met"] is False and out["a8"]["both_improve"] is True and "amendment5" not in out
        assert out["a8"]["spotting_lower_bound"] == pytest.approx(delta)
    for arm in (R16_STACK, R16_SINGLE, dict(prior="r16", no_gains=False, best_of=0)):
        with pytest.raises(SystemExit, match="R16's words are not adopted"):
            check(world(), **arm)


def test_a8_met_w14b_needs_the_r16_read():
    with pytest.raises(SystemExit, match="unread: a9-r16-nogains-n16-r10"):
        check(met_world(**{"a9-r16-nogains-n16-r10": None}))
    with pytest.raises(SystemExit, match="unread: a4-r16-n16-r10"):               # W14b's stack failed: the single arm's read
        check(met_world(**W14B_STACK_FAILED, **{"a4-r16-n16-r10": None}), **A4_AT(16))
    # each route needs only its own read
    assert check(met_world(**{"a4-r16-n16-r10": None}), **R16_STACK)["decision"].startswith("r16 stack")
    assert check(met_world(**W14B_STACK_FAILED, **{"a9-r16-nogains-n16-r10": None}), **R16_SINGLE)["decision"].startswith("r16 a4")


def test_the_r16_stack_wins_inside_the_stack():
    out = check(met_world(), **R16_STACK)
    assert out["decision"] == "r16 stack, best of 16" and out["a8"]["met"] and out["rule3"]["passed"]
    route = out["amendment5"]
    assert route["met"] and route["words"] == "r16" and route["r16"] == "a9-r16-nogains-n16-r10"
    assert route["w14b"] == "a9-nogains-n16-r10" and route["likelihood"] == pytest.approx(.1) and route["spotting_lower_bound"] == 0.
    for arm in (dict(), A7_ALONE, A4_AT(16)):                                     # W14b's arms are refused
        with pytest.raises(SystemExit, match="amendment 5 gives r16's words, not w14b's"):
            check(met_world(), **arm)
    for arm in (R16_SINGLE, dict(prior="r16", no_gains=False, best_of=0), dict(prior="r16", no_gains=True, best_of=0)):
        with pytest.raises(SystemExit, match="R16's words go as the read that won, NO_GAINS=1 BEST_OF=16"):
            check(met_world(), **arm)
    with pytest.raises(SystemExit, match="rule 1 chooses best of 16, not 8"):
        check(met_world(), prior="r16", no_gains=True, best_of=8)


@pytest.mark.parametrize("a9,delta,why", [
    (d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.0), 0., "BLEU equal"),
    (d20(R16, n=16, gains_off=True, hyps=LOWER, ll=-2.0), 0., "BLEU below"),
    (d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.1), 0., "likelihood equal"),
    (d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.2), 0., "likelihood below"),
    (d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.0), -.02, "spotting at the floor"),
    (d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.0), -.05, "spotting below it"),
])
def test_the_r16_stack_loses_and_w14b_goes_by_amendment_4(a9, delta, why):
    w = met_world(**{"a9-r16-nogains-n16-r10": a9})
    out = check(w, r16_delta=delta)
    assert out["decision"] == "stack, best of 16" and out["amendment5"]["met"] is False and out["amendment5"]["words"] == "w14b"
    for arm in (R16_STACK, R16_SINGLE):
        with pytest.raises(SystemExit, match="amendment 5 gives w14b's words, not r16's"):
            check(w, r16_delta=delta, **arm)
    for arm, match in ((A7_ALONE, "the stack goes, not a single"), (A4_AT(16), "the stack goes, not a single"),
                       (A4_AT(8), "rule 1 chooses best of 16")):                   # and rules 1 and 3 still decide
        with pytest.raises(SystemExit, match=match):
            check(w, r16_delta=delta, **arm)


def test_the_r16_stack_just_above_the_spotting_floor_wins():
    assert check(met_world(), r16_delta=-.019, **R16_STACK)["amendment5"]["met"] is True


def test_w14b_stack_failing_moves_the_comparison_to_the_single_arms():
    w = met_world(**W14B_STACK_FAILED)
    out = check(w, **R16_SINGLE)
    route = out["amendment5"]
    assert out["decision"].startswith("r16 a4 alone") and out["rule3"]["passed"] is False
    assert route["r16"] == "a4-r16-n16-r10" and route["w14b"] == "a4-w14b-n16-r10" and route["met"]
    for arm in (dict(), A7_ALONE, A4_AT(16)):
        with pytest.raises(SystemExit, match="amendment 5 gives r16's words, not w14b's"):
            check(w, **arm)
    for arm in (R16_STACK, dict(prior="r16", no_gains=True, best_of=0)):
        with pytest.raises(SystemExit, match="R16's words go as the read that won, NO_GAINS=0 BEST_OF=16"):
            check(w, **arm)
    with pytest.raises(SystemExit, match="only inside the stack"):
        check(w, lowpass=True, **R16_SINGLE)
    # R16's single arm loses, on each test: W14b goes, and rule 3 names the better single
    for a4, delta in ((d20(R16, n=16, hyps=LOWER, ll=-2.1), 0.), (d20(R16, n=16, hyps=MID, ll=-2.2), 0.),
                      (d20(R16, n=16, hyps=MID, ll=-2.1), -.02)):
        lost = met_world(**W14B_STACK_FAILED, **{"a4-r16-n16-r10": a4})
        out = check(lost, r16_delta=delta, **A4_AT(16))
        assert out["decision"].startswith("a4 alone") and out["amendment5"]["words"] == "w14b"
        with pytest.raises(SystemExit, match="better single lever is a4"):
            check(lost, r16_delta=delta, **A7_ALONE)
        with pytest.raises(SystemExit, match="amendment 5 gives w14b's words, not r16's"):
            check(lost, r16_delta=delta, **R16_SINGLE)


def test_rule_4_reads_the_stack_actually_chosen():
    up = dict(lowpass=TOP, lowpass_ll=-1.9)
    # R16 chosen: R16's @lowpass decides, W14b's is not consulted
    r16_up = d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.0, **up)
    w = met_world(**{"a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=LOWER, ll=-2.1),
                     **{f"a4-w14b-n{n}-r10": d20(n=n, hyps=HYP_LOW2, ll=-2.5 + .1 * k) for k, n in enumerate((4, 8, 16), 1)},
                     "a7-nogains-r10": d20(gains_off=True, hyps=HYP_LOW2, ll=-2.45), "a9-r16-nogains-n16-r10": r16_up})
    w["a8-r16-r10"] = d20(R16, hyps=HYP_LOW2, ll=-2.4)
    out = check(w, lowpass=True, **R16_STACK)
    assert out["decision"] == "r16 stack, best of 16 with its @lowpass" and out["rule4"]["both_improve"]
    with pytest.raises(SystemExit, match="must run with LOWPASS=1"):
        check(w, **R16_STACK)
    # R16's @lowpass not above it, though W14b's is: LOWPASS=1 refused, LOWPASS=0 goes
    w2 = dict(w, **{"a9-r16-nogains-n16-r10": d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.0),
                    "a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=LOWER, ll=-2.1, lowpass=MID, lowpass_ll=-2.)})
    assert check(w2, **R16_STACK)["rule4"]["both_improve"] is False
    with pytest.raises(SystemExit, match="the stack's @lowpass does not have both"):
        check(w2, lowpass=True, **R16_STACK)
    # W14b chosen: W14b's own @lowpass decides, R16's is not consulted
    w3 = met_world(**{"a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=MID, ll=-2.1, lowpass=TOP, lowpass_ll=-2.),
                      "a9-r16-nogains-n16-r10": d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.0, lowpass=TOP, lowpass_ll=-1.)})
    out = check(w3, lowpass=True)
    assert out["amendment5"]["words"] == "w14b" and out["rule4"]["both_improve"]
    with pytest.raises(SystemExit, match="must run with LOWPASS=1"):
        check(w3)
    w4 = met_world(**{"a9-r16-nogains-n16-r10": d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.0, lowpass=TOP, lowpass_ll=-1.)})
    assert check(w4)["rule4"]["both_improve"] is False                            # R16 lost: its @lowpass adopts nothing
    # a read without the form
    unread = d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.0)
    del unread["forms"][f"{FORM}@lowpass"], unread["likelihood"]["per_form"][f"{FORM}@lowpass"]
    with pytest.raises(SystemExit, match="not in the report"):
        check(met_world(**{"a9-r16-nogains-n16-r10": unread}), **R16_STACK)


@pytest.mark.parametrize("name,report,match", [
    ("a9-r16-nogains-n16-r10", d20(n=16, gains_off=True, hyps=TOP, ll=-2.), "word_cache.checkpoint_key"),      # W14b's words
    ("a9-r16-nogains-n16-r10", d20(R16, n=16, hyps=TOP, ll=-2.), "word_cache.no_decoder_gains"),               # gained
    ("a9-r16-nogains-n16-r10", d20(R16, n=8, gains_off=True, hyps=TOP, ll=-2.), "word_cache.n"),
])
def test_the_r16_reads_must_be_what_their_names_say(name, report, match):
    with pytest.raises(SystemExit, match=match):
        check(met_world(**{name: report}))
    for wrong, why in ((d20(n=16, hyps=MID, ll=-2.1), "word_cache.checkpoint_key"), (d20(R16, n=16, gains_off=True, hyps=MID, ll=-2.1),
                                                                                    "word_cache.no_decoder_gains")):
        with pytest.raises(SystemExit, match=why):
            check(met_world(**W14B_STACK_FAILED, **{"a4-r16-n16-r10": wrong}))


def test_lowpass_only_on_a_passing_stack_above_it():
    assert check(world(), lowpass=False)["decision"] == "stack, best of 16"
    a9 = d20(n=16, gains_off=True, hyps=TOP, ll=-2.1, lowpass=TOP, lowpass_ll=-2.0)
    with pytest.raises(SystemExit, match="the stack's @lowpass does not have both"):
        check(world(**{"a9-nogains-n16-r10": a9}), lowpass=True)                  # BLEU not above
    a9 = d20(n=16, gains_off=True, hyps=TOP, ll=-2.1, lowpass=MID, lowpass_ll=-2.0)
    with pytest.raises(SystemExit, match="the stack's @lowpass does not have both"):
        check(world(**{"a9-nogains-n16-r10": a9}), lowpass=True)
    unread = d20(n=16, gains_off=True, hyps=TOP, ll=-2.1)
    del unread["forms"][f"{FORM}@lowpass"], unread["likelihood"]["per_form"][f"{FORM}@lowpass"]
    for on in (True, False):                                                      # never read: rule 4 cannot be skipped
        with pytest.raises(SystemExit, match="not in the report"):
            check(world(**{"a9-nogains-n16-r10": unread}), lowpass=on)
    # a stack whose BLEU leaves room: above both levers on BLEU, its @lowpass above it
    w = world(**{"a7-nogains-r10": d20(gains_off=True, hyps=LOW, ll=-2.45),
                 **{f"a4-w14b-n{n}-r10": d20(n=n, hyps=LOWER, ll=-2.5 + .1 * k) for k, n in enumerate((4, 8, 16), 1)},
                 "a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=MID, ll=-2.1, lowpass=TOP, lowpass_ll=-2.0)})
    out = check(w, lowpass=True)
    assert out["rule4"]["both_improve"] and out["decision"] == "stack, best of 16 with its @lowpass"
    with pytest.raises(SystemExit, match="must run with LOWPASS=1"):              # rule 4 admits A6: it cannot be left out
        check(w, lowpass=False)
    w["a9-nogains-n16-r10"] = d20(n=16, gains_off=True, hyps=MID, ll=-2.1, lowpass=TOP, lowpass_ll=-2.1)
    with pytest.raises(SystemExit, match="the stack's @lowpass does not have both"):
        check(w, lowpass=True)                                                    # likelihood equal
    assert check(w, lowpass=False)["rule4"]["both_improve"] is False              # and then the stack goes unfiltered
    # not on a stack that failed, nor on a single lever (a7's @lowpass read adopts nothing)
    failed = {"a9-nogains-n16-r10": d20(n=16, gains_off=True, hyps=TOP, ll=-2.3, lowpass=TOP, lowpass_ll=-1.)}
    with pytest.raises(SystemExit, match="against both a7"):
        check(world(**failed), lowpass=True)
    with pytest.raises(SystemExit, match="only inside the stack"):
        check(world(**failed), lowpass=True, **A7_ALONE)
    with pytest.raises(SystemExit, match="only inside the stack"):
        check(world(**failed), lowpass=True, **A4_AT(16))


@pytest.mark.parametrize("name,report,match", [
    ("a9-nogains-n16-r10", d20(n=16, hyps=TOP, ll=-2.1), "word_cache.no_decoder_gains"),           # a gained stack
    ("a9-nogains-n16-r10", d20(n=8, gains_off=True, hyps=TOP, ll=-2.1), "word_cache.n"),
    ("a9-nogains-n16-r10", d20(R16, n=16, gains_off=True, hyps=TOP, ll=-2.1), "word_cache.checkpoint_key"),
    ("a9-nogains-n16-r10", d20(hyps=TOP, gains_off=True, ll=-2.1), "has no word cache"),
    ("a7-nogains-r10", d20(ll=-2.45), "no_decoder_gains"),                                          # a7 with gains
    ("a7-nogains-r10", d20(n=16, gains_off=True, ll=-2.45), "used a word cache"),
    ("a4-w14b-n8-r10", d20(n=8, gains_off=True, ll=-2.3), "word_cache.no_decoder_gains"),
    ("a4-w14b-n8-r10", d20(n=4, ll=-2.3), "word_cache.n"),
    ("a0-w14b-r10", d20(R16, hyps=LOW, ll=-2.5), "settings.checkpoint_key"),
    ("a8-r16-r10", d20(hyps=LOW, ll=-2.6), "settings.checkpoint_key"),
    ("a8-r16-r10", d20(R16, hyps=LOW, ll=-2.6, sentences="0" * 64), "did not read D20"),
])
def test_every_read_must_be_what_its_name_says(name, report, match):
    with pytest.raises(SystemExit, match=match):
        check(world(**{name: report}))


def test_stack_check_cli_reads_the_prefix_and_the_shards(tmp_path, capsys):
    from train_text_to_motion import make_store
    store = make_store(f"dir:{tmp_path / 's'}")
    argv = ["stack-check", "--store", f"dir:{tmp_path / 's'}", "--phase-a-out", "a", "--prior", "w14b", "--w14b-checkpoint-key", W14B,
            "--r16-checkpoint-key", R16, "--no-gains", "1", "--best-of", "16", "--lowpass", "0", "--sentences-sha256", D20_SHA]
    with pytest.raises(SystemExit, match="unread: a0-w14b-r10"):
        pv.main(argv)
    for name, report in world().items():
        store.put(f"a/{name}.json", json.dumps(report).encode())
    with pytest.raises(SystemExit, match="a8-r16-r10 against a0-w14b-r10 cannot be judged"):   # no shard stored: never silently a pass
        pv.main(argv)
    for name in ("a0-w14b-r10", "a8-r16-r10"):
        store.put(f"a/{name}-shard0.json", json.dumps(shard(world()[name]["settings"])).encode())
    capsys.readouterr()
    assert pv.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "stack, best of 16" and out["a8"]["met"] is False and out["rule1"]["n"] == 16


# ---- review fix 3 ---------------------------------------------------------------------------------------------------------------

def test_a_full_phase_a_cache_needs_a_gpu(tmp_path):
    for kwargs in (dict(arm="w14b"), dict(arm="r16"), dict(arm="w14b", NO_GAINS="1")):
        code, out, commands = phase_a(tmp_path, "cache", gpu=False, **kwargs)
        assert code == 1 and "a full cache needs a GPU" in out and not scripts_run(commands)
    code, out, commands = phase_a(tmp_path, "cache", arm="w14b", gpu=False, smoke=True)
    assert code == 0 and any("round20_word_cache.py run" in c for c in commands)


def cache_fixture(tmp_path, store, sha, monkeypatch):
    import torch
    import numpy as np
    import validate_sign_recognizer as vsr
    labels = ["CUT", "ONION", "SALT", "HEAT", "KNIFE", "HALF", "NOW", "STIR"]
    monkeypatch.setattr(vsr, "load_model", lambda store, run, device: (None, labels, {"checkpoint_key": "r.pt"}))

    def logits(model, states, device, batch=64):                               # a score that depends on the very state read
        return torch.stack([torch.tensor([np.sin(float(np.abs(s.channels["right_hand"].values[s.channels["right_hand"].valid]).mean())
                                                 * (k + 3) * 50.) for k in range(len(labels))], dtype=torch.float64) for s in states])
    monkeypatch.setattr(vsr, "logits_of", logits)
    return logits, labels, ["run", "--store", f"dir:{tmp_path / 'store'}", "--prefix", "cache/s", "--mix-key", "mix.json",
                            "--mix-sha256", sha, "--clips", "5", "--checkpoint-key", "prior.pt", "--samples", "3", "--choose",
                            "3", "--device", "cpu", "--limit", "3"]


@pytest.mark.parametrize("gains_off", [False, True])
def test_the_cache_scores_the_state_it_stores(tmp_path, stage2_store, monkeypatch, gains_off):  # noqa: F811
    import torch
    from evaluate_text_to_motion import at_rate, load_state
    store, sha, *_ = stage2_store
    put_gains(store)
    logits, labels, run = cache_fixture(tmp_path, store, sha, monkeypatch)
    assert cw.main(run + (["--no-decoder-gains"] if gains_off else [])) == 0
    rows = json.loads(store.get("cache/s/shard-0-of-1.json"))["entries"]
    labelled = [r for r in rows if r["label"]]
    assert labelled
    for r in labelled:
        pick = r["chosen"]["3"]
        state = load_state(store, pick["state_key"], pick["state_sha256"])
        row = logits(None, [at_rate(state, 24.)], "cpu")[0].double()
        lp = torch.log_softmax(row, 0)[labels.index(r["label"])]
        assert float(lp) == pytest.approx(pick["score"], abs=1e-3)              # the stored file reads as its recorded score
        assert pick["score"] == max(r["scores"][:3])
        assert len(set(r["scores"])) == 3                                       # each sample scored as itself, not one for all


def test_a_repeated_shard_on_other_bits_says_what_to_do(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, *_ = stage2_store
    _, _, run = cache_fixture(tmp_path, store, sha, monkeypatch)
    assert cw.main(run) == 0
    rows = json.loads(store.get("cache/s/shard-0-of-1.json"))["entries"]
    (tmp_path / "store/cache/s/shard-0-of-1.json").unlink()                      # the pod died before the shard's report
    (tmp_path / "store" / rows[0]["chosen"]["3"]["state_key"]).write_bytes(b"another device's bits")
    with pytest.raises(SystemExit) as e:
        cw.main(run)
    text = str(e.value)
    assert "written once" in text and "lost pod" in text and "GPU type" in text and "new --prefix" in text
    assert not store.exists("cache/s/shard-0-of-1.json")


@pytest.mark.parametrize("edit,match", [
    (lambda r: r["arm"]["settings"].update(guidance=2.), "settings.guidance"),
    (lambda r: r["arm"]["settings"].update(timing="asllrp"), "settings.timing"),
    (lambda r: r["arm"]["settings"].update(shards=1), "settings.shards"),
    (lambda r: r["arm"]["settings"].update(checkpoint_key=W14B), "generated words as well"),
    (lambda r: r["arm"]["settings"]["round20"].update(tempo=1.25), "round20.tempo"),
    (lambda r: r["arm"]["settings"]["round20"].update(likelihood=False), "likelihood is not on"),
    (lambda r: r["arm"]["settings"]["round20"].update(lowpass_hz=6.), "lowpass_hz"),
    (lambda r: r["arm"]["settings"]["round20"].update(word_cache=cache_settings(n=4)), "word_cache.n"),
])
def test_the_c20_verdict_checks_the_arms_settings(edit, match):
    r = c20_arm()
    edit(r)
    with pytest.raises(SystemExit, match=match):
        judge_c20(r, val_verdict())


def test_the_c20_verdict_is_this_arms_only():
    for arm in (dict(best_of=4), dict(no_decoder_gains=False), dict(lowpass=True), dict(prior_checkpoint_key=R16)):
        with pytest.raises(SystemExit, match="another arm's"):
            judge_c20(c20_arm(), val_verdict(**arm))


# ---- amendment 5: R16's reads in Phase A, and its route in Phase B ------------------------------------------------------------

def test_phase_a_r16_no_gains_cache(tmp_path):
    code, out, commands = phase_a(tmp_path, "cache", arm="r16", NO_GAINS="1")
    runs = [c for c in commands if "round20_word_cache.py run" in c]
    assert code == 0 and len(runs) == 4 and f"R20 cache-r16-nogains-r10 {OUT_A}/cache-r16-nogains-r10/index.json" in out
    plain = sorted(c for c in phase_a(tmp_path, "cache", arm="r16")[2] if " run " in c)
    assert sorted(c.replace(" --no-decoder-gains", "").replace("cache-r16-nogains-r10", "cache-r16-r10") for c in runs) == plain
    assert all(f"--checkpoint-key {R16} " in c and "--no-decoder-gains" in c for c in runs)


@pytest.mark.parametrize("arm,n", [("n4", 4), ("n8", 8), ("n16", 16)])
def test_phase_a_r16_reads_are_the_w14b_arms_with_r16s_caches(tmp_path, monkeypatch, arm, n):
    def run_line(phase, **kw):
        code, out, commands = phase_a(tmp_path, phase, arm=arm, **kw)
        assert code == 0
        return next(c for c in commands if "round15_compose_eval.py run" in c)
    gained = f"{OUT_A}/cache-r16-r10/index.json"
    plain = f"{OUT_A}/cache-r16-nogains-r10/index.json"
    a4 = run_line("a4", PRIOR="r16", stored=[gained])
    assert f"--word-cache-key {gained} " in a4 and a4.endswith(f"--output-key {OUT_A}/a4-r16-{arm}-r10-shard0.json")
    assert a4.replace("cache-r16-r10", "cache-w14b-r10").replace(f"a4-r16-{arm}", f"a4-w14b-{arm}") == run_line("a4")
    a9 = run_line("a9", PRIOR="r16", stored=[plain])
    assert f"--word-cache-key {plain} " in a9 and a9.endswith(f"--output-key {OUT_A}/a9-r16-nogains-{arm}-r10-shard0.json")
    w14b = run_line("a9", stored=[f"{OUT_A}/cache-w14b-nogains-r10/index.json"])
    assert w14b.endswith(f"--output-key {OUT_A}/a9-nogains-{arm}-r10-shard0.json")     # W14b's keys are unchanged
    assert a9.replace("cache-r16-nogains-r10", "cache-w14b-nogains-r10").replace(f"a9-r16-nogains-{arm}", f"a9-nogains-{arm}") == w14b
    assert run_line("a9", PRIOR="w14b", stored=[f"{OUT_A}/cache-w14b-nogains-r10/index.json"]) == w14b
    seen = parsed_run(tmp_path, monkeypatch, a9, f"r{n}")
    assert seen["word_cache_n"] == n and seen["lowpass_report_hz"] == 6. and seen["no_decoder_gains"] is False
    # the a9 smoke, and the a4 smoke, take their cache
    smoke = phase_a(tmp_path, "a9", arm=arm, PRIOR="r16", smoke=True, gpu=False, CACHE_KEY="s/i.json", stored=["s/i.json"])
    assert smoke[0] == 0 and any(f"a9-r16-nogains-{arm}-r10-shard0" in c for c in smoke[2])


def test_phase_b_r16_route(tmp_path, monkeypatch):
    """R16's stack through the job: its own cache, its pilot without gains, the W14b comparator, its @lowpass by LOWPASS."""
    r16 = dict(prior="r16", no_gains="1", best_of="16")
    code, out, commands = job(tmp_path, "cache", **r16, pilots=False)
    assert code == 0 and all(f"--prefix {OUT}/cache-r16-nogains-val1000-n16 " in c and f"--checkpoint-key {R16} " in c
                             and "--no-decoder-gains" in c for c in commands if " run " in c)
    code, out, commands = job(tmp_path, "e2e", **r16, pilots=False)
    line = next(c for c in commands if "pilot_content_check.py" in c)
    assert code == 0 and f"--checkpoint-key {R16} " in line and f"--output-key {OUT}/pilot-r16-nogains-val1000.json" in line
    assert "--recognizer-run sign-recognizer-v3d" in line and "--word-list-key" in line and "--no-decoder-gains" in line
    index = f"{OUT}/cache-r16-nogains-val1000-n16/index.json"
    code, out, commands = job(tmp_path, "run", **r16, stored=[index], pilots=False)
    assert code == 1 and "run the e2e phase first" in out
    code, out, commands = job(tmp_path, "run", **r16, stored=[index], LOWPASS="1")
    assert code == 0 and "--prior r16 " in next(c for c in commands if " stack-check " in c)
    assert "--lowpass 1 " in next(c for c in commands if " stack-check " in c)
    runs = [c for c in commands if "compose_eval.py run" in c]
    assert len(runs) == 4 and all(f"--word-cache-key {index} " in c and "--lowpass-hz 6" in c and "--no-decoder-gains" not in c for c in runs)
    merge = next(c for c in commands if " merge " in c)
    assert f"--end-to-end-report {OUT}/pilot-r16-nogains-val1000.json --paired-with {CMP} " in merge
    code, out, commands = job(tmp_path, "verdict", **r16, LOWPASS="1", stored=[f"{OUT}/r16-nogains-n16-lp6-val1000.json"])
    line = next(c for c in commands if " verdict " in c)
    assert code == 0 and f"--end-to-end-key {OUT}/pilot-r16-nogains-val1000.json " in line
    assert f"--stored-pilot-key {O}/round16/pilot-r16.json " in line and f"--comparator-key {CMP} " in line
    assert f"--prior r16 --prior-checkpoint-key {R16} --no-gains 1 --best-of 16" in line
    # R16's single arm: the gained cache and the stored pilot, no e2e
    index = f"{OUT}/cache-r16-gains-val1000-n16/index.json"
    code, out, commands = job(tmp_path, "run", prior="r16", no_gains="0", best_of="16", stored=[index], pilots=False)
    assert code == 0 and f"--end-to-end-report {O}/round16/pilot-r16.json " in next(c for c in commands if " merge " in c)
    # the W14b comparator stays the comparator, whichever words
    assert "--paired-with " + CMP in next(c for c in commands if " merge " in c)


def test_the_verdict_takes_an_r16_stack_with_its_own_pilot():
    r = reports()
    r["arm"]["settings"]["round20"]["word_cache"] = cache_settings(checkpoint=R16, n=16)
    r["arm"]["comparators"]["end_to_end"]["checkpoint_key"] = R16
    r["pilot"]["checkpoint_key"] = r["stored_pilot"]["checkpoint_key"] = R16
    v = judge(r, prior_checkpoint=R16, best_of=16)
    assert v["rule_passed"] and "WITHOUT gains" in v["items"]["bleu_vs_own_end_to_end"]["note"]
    gained_pilot = copy.deepcopy(r)
    gained_pilot["pilot"] = gained_pilot["stored_pilot"]
    gained_pilot["arm"]["forms"]["end_to_end"]["hypotheses"] = gained_pilot["pilot"]["meaning"]["generated"]["hypotheses"]
    with pytest.raises(SystemExit, match="lenient"):
        judge(gained_pilot, prior_checkpoint=R16, best_of=16)
    w14b_pilot = copy.deepcopy(r)
    w14b_pilot["pilot"]["checkpoint_key"] = W14B
    with pytest.raises(SystemExit, match="not the arm's prior"):
        judge(w14b_pilot, prior_checkpoint=R16, best_of=16)
