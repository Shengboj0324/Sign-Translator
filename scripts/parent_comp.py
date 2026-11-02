"""Parent composition read: how much of real signing's clip-specific margin does composed motion carry, on the instrument not fine-tuned on How2Sign? Report only: nothing trains, and this code gates and selects nothing. It reads the eight A5 word draws' added-words composition (top10) on D20, through ``back-translator-kp2d-uwb-v2`` step 76,000, sha ``f1ff9446...``, v1's parent.

**How this will be read (stated before any read; the commit holding this text precedes the job).** No label and no bar: this job sets none. The figures are composition's margin and its share of real two-view's margin, per draw and over the eight, and, beside v1's stored top10 on the same measures, the parent's BLEU gap share (``comp_a_read.form_read``'s formula: v1's stored mean is 32.5%) and likelihood gap share, each with a paired per-draw difference and a t interval. **Like is compared with like:** a margin share and a BLEU gap share measure different things and can differ about twofold, so a margin share is never read against v1's BLEU share. v1's fit to two-view may understate its shares.
- A gap share near v1's on its own metric says the parent reads composition about as v1 does.
- A gap share well above it says v1's fit to two-view may be costing composition part of its reading. It does not say the motion carries that content for anyone: the parent is v1's starting weights, trained on MediaPipe image landmarks, the source of the monocular path's hands and so of word clips, and it may read that path generously. Whether its margin moves with hands at all is ``sensitivity_ladder``'s question: read this beside that.
- A gap share well below it says composition's reading on v1 did not come from what the parent reads.
None of these says people understand the motion, and none is a claim about M1 or release.

**Margin**, the parts job's: a clip's mean NLL of 4 mismatched references minus NLL of its own (``retrieval_candidates``, seed 0, the same draw for every form), which is the mismatched-sentence control. **Each form's margin is over its own still pose at its own length,** clip by clip, so a form's duration and flags cancel: ``still_two_view`` for real and nearest, and for each composition its nearest-filled middle frame held on its own clock. **A clip with no selected word is counted,** as composition's reads count it: its state is its blank clip, its still is that same blank, so its margin is exactly zero and the count is reported per draw.

**Forms.** ``real_two_view``, ``blank``, ``nearest_two_view``, ``still_two_view``: the parts job's. ``d<K>_comp``, ``d<K>_still``: draw K's ``core_crossfade@predicted``, composed again by ``round15_compose_eval --export-only`` from that draw's two caches (composed states were never stored). ``d<K>_comp_true``, ``d<K>_still_true``: the same words at the reference's length, ``core_crossfade``: not in the brief's forms, read because they are the direct test of the length question, and they set nothing. ``d<K>_blank``: the blank clip at that draw's composition lengths, the denominator of its gap shares.

**Per draw and over the eight** (``merge``): margin, share of real's and of nearest's, BLEU, each with a bootstrap over clips (n 2,000, seed 0); over the eight a t interval on the draw values alone (``comp_a5_read.t_interval``) and that bootstrap with the draws fixed: the sentences are fixed, the draws are what vary. **Gap shares:** BLEU, (form minus blank) over (real minus blank), and likelihood by the same formula on own NLL, both against this draw's blank at its composition's lengths. **Length:** true length minus predicted, per draw and over the eight, each over its own still; the margin over the composition's own still against over ``still_two_view``, a difference that also holds pose, since the two stills are different frames of different motion, so it does not isolate length; and, report only, per-clip margin on log length ratio. **Per signer, and by how much of a sentence has words,** report only: ``word_counts`` per clip, rows on non-empty clips and by coverage. Few clips make a signer's share noisy, and c01 and c05 are in v1's training list while c03 is its selection pool, which matters for v1's stored shares and not the parent's.
**Beside it:** v1's stored top10 gap share and BLEU per draw (``a5-summary.json``), and that share on clips outside v1's training set, since v1 trained on 234 of D20 (``parent_lib``). Those numbers are read here by the job, not before it.

**Controls.** The four anchors against the parts job's stored forms, pinned by hash (at least 0.98 of hypotheses identical, BLEU within 0.05, mean NLL within 0.001, all five columns). A miss voids the read and writes a void record; ``read`` refuses a composition without a passed control of this code and environment. The parent's monitor control is not repeated: reproducing those forms carries it. **Fidelity, before each draw's read:** every exported clip's frames against the plan in that draw's stored report (``per_clip``, predicted and true length), and v1 re-reads the first ``FIDELITY_CLIPS`` non-empty clips of the predicted form, keeping only the count of hypotheses identical to the stored report: no BLEU, NLL or text. On D20, under ``FIDELITY_SHARE`` of those clips, or any frame count off, stops that draw and the job with status 1, before its read.

**Refused.** A stored anchor on other bytes stops with status 1, no control run and nothing voided. Only the pinned draws, each by its top10 report's hash; a report or export whose caches are not that draw's; a sentence list that is not D20 (``s2_lib``; C20 cannot be named, and the export code refuses it too); c00 and any other partition; an export missing a clip; forms of other code. A smoke (``--smoke-train 5``): 5 D20 sentences composed for one draw, read against c02 train references, so the parent sees no D20 text or real clip: numbers are plumbing.

**Across pods.** Each pod reads its draws and its own anchors under ``pods/<POD>/``; a draw's shares use its own pod's anchors. ``ready`` says when all eight stand, and ``merge`` joins them.

    python scripts/parent_comp.py inputs
    python scripts/parent_comp.py export --draw K --work W --prefix P     (one per draw, CPU)
    python scripts/parent_comp.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/parent_comp.py read --forms control --out-dir W/forms --instrument-cache W/i
    python scripts/parent_comp.py control --out-dir W/forms --void-prefix V
    python scripts/parent_comp.py fidelity --draw K --composed P --out-dir W/forms --instrument-cache W/i
    python scripts/parent_comp.py read --draws 1,2,3,4 --composed P --resume --out-dir W/forms --instrument-cache W/i
    python scripts/parent_comp.py upload --out-dir W/forms --prefix D/pods/POD/forms
    python scripts/parent_comp.py ready --pods a,b --prefix D/pods
    python scripts/parent_comp.py merge --pods a,b --prefix D/pods --output-file W/report.json
    python scripts/parent_comp.py check --report W/report.json

The job is ``runpod/fleet/parent_comp.sh``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a5_read as A5  # noqa: E402
import comp_a_read as A  # noqa: E402
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "parent-comp"
JOB_FILE = "runpod/fleet/parent_comp.sh"
PREFIX = f"{L.O}/round20/parent-comp"
A5_PREFIX = f"{L.O}/round20/comp-a5"
A5_SUMMARY, A5_SUMMARY_SHA = f"{A5_PREFIX}/a5-summary.json", "4f3d1f38bfb6d43ae6a4e60ca056fb3e75b7fec064011bc7645a3b1d814e0b11"
DRAWS = tuple(range(1, 9))
DRAW_INPUTS = {
    1: dict(top10="536deaf8c9e817764fd764b4b685706a55cd26251bce7ff909a4ad4b32dff9dd", cache="b28be9113ccfef0bfe1c315b381a2d2cb2b1f404114f25e08f5c7d2671203120", extra="dd53718bf57d9a425c022407dd546e8e40f5e71742414eecdbfdcb62f777133a"),
    2: dict(top10="305353ac8f3a07d9745a151ba228bc8694b5cb1020e7bf45bd406df93ecbcf0a", cache="2465289c5a44a97890900f7f2d73e33c1414da9f6d221df902410b0717d6426b", extra="39b00b680c5523ef31ae45d863763bc15b35bfdffbae2b2d38a957f3c4deddc7"),
    3: dict(top10="a307cdad1f3ee4e1656a20c5f9bd8a4019dd58321aa6c1a7852f23e3b42f4feb", cache="7519118cba5d5c2343433cf3d5c72af1b34ae75dd10ba91042ad13faa90a47de", extra="28c0dafb9db479377bddaa8ffc606d6773b40732f90f4cc73eab2f16f931447e"),
    4: dict(top10="e7e5bf303d28092201926ae2c47cfee07ff6f98960bcd30343b8a97baeb8e1b2", cache="275ccf26c1767fb9c25e9558e38aac615996e8150b364e897d0caee755adb6a1", extra="09ee77e9c4f88dc841683c16066d6c1e15b7365e906c9d0711d96e3236bfda66"),
    5: dict(top10="066133085edce318cdf9ef75026f1011ffdf671b909c1db1cce54e6ff5218dc9", cache="f1a6b9c8ef3ba6ecd071308168ad094c6abb96cea47aec37a19c410d0dd380a9", extra="f266f552c31833ee002d9efe9f2f9184fc0cada6f83e1d71500587d0ea94653b"),
    6: dict(top10="098fee8990d36602f5b35ec9e4d576c2c359347a8acb731c814843b00f42b0ed", cache="4340d317f345a675fdf5b1915903fb19a4d358ab260fdb8f523033613939b03c", extra="809cfb775988f59a7877ebad160424c92f8b65f5f0dbae3fa8b6aa43d78097c0"),
    7: dict(top10="2caa4467a1da3b8e221692ce971dc35c6f52f4f3b69abb30aece5835746e358f", cache="5bc94c6995da5fb87473fb1bc1adccae1ffc5ff0dcab64d4c8f0b48a8bac1569", extra="9c1cc4eb62dab3b80cab77683ad24985ed8f3150c5335564d7204120e565608b"),
    8: dict(top10="ff0787757fc562a9fcfc7702575a39bbe3b1802e810b850d8801f8b5abbed1e7", cache="33d4cac4a9dc12b8ee3f917e16c17e810b7eb45d32eda817658548beb54beb21", extra="21883a264a467df542ab42644737e7634dda910466bf9b3f2a34c155c3d026bc")}
MODEL_SHA = "42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21"      # runpod/fleet/comp_a5.sh: the declared stage-2 model
DURATION = "-0.083,0.3691,18289"
PREDICTED, TRUE_LENGTH = "core_crossfade@predicted", "core_crossfade"
EXPORT_FORMS = (PREDICTED, TRUE_LENGTH)
ANCHORS = ("real_two_view", "blank", "nearest_two_view", "still_two_view")
KINDS = dict(comp=(PREDICTED, "as_composed"), still=(PREDICTED, "still"), comp_true=(TRUE_LENGTH, "as_composed"), still_true=(TRUE_LENGTH, "still"), blank=(PREDICTED, "blank"))
FORM = re.compile(r"d([1-8])_(comp_true|still_true|comp|still|blank)")
SAME_LENGTH = (("comp", "still"), ("comp_true", "still_true"), ("comp", "blank"))
COVERAGE_BINS = ((0., .25), (.25, .5), (.5, float("inf")))
CANDIDATES = Q.CANDIDATES
BOOT = Q.BOOT
SMOKE_CLIPS = 5
FIDELITY_CLIPS, FIDELITY_SHARE = 32, .75
PLAN = dict(nearest_margin=.2372, nearest_bleu=2.207, real_bleu=2.130, blank_bleu=1.058, still_bleu=.895, v1_share=.325, v1_bleu=2.402)      # docs/NEXT_PHASE_PLAN.md: Parts read, Question 2; A5 result
PLAN_TOLERANCE = dict(margin=.002, bleu=.05, share=.005, v1_bleu=.001)
NOT_DECIDED = "report only: no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    parent="The parent is v1's starting weights, trained on MediaPipe Holistic image landmarks: the source of the monocular path's hands and so of the word clips. It may read that path generously, and how far its margin moves with hands is sensitivity_ladder's question.",
    training="v1 trained on 234 of D20's clips and was selected on a pool holding 265 more, through two-view, so its real read, and its stored share's denominator, are partly a read of its own training clips. The parent never trained on How2Sign.",
    still="A composition's still holds its nearest-filled middle frame on its own clock: one pose, which can carry more than flags and length. A content-blind instrument reads about zero on this measure.",
    empty="A clip with no selected word is its blank clip in every read, with a margin of exactly zero, and it stays in every mean: coverage counts.",
    true_length="d<K>_comp_true and d<K>_still_true are the same words at the reference's length: beyond the brief's forms, the direct test of the length question.",
    blank="Gap shares take d<K>_blank, at the composition's lengths, and real against it, as v1's stored shares do. Flags alone can read an offset against a blank: the margin does not carry it, and the two scales differ, about twofold, so like is compared with like.",
    per_signer="Per-signer rows are report only: few clips make them noisy. c01 and c05 are in v1's training list and c03 is its selection pool, which bears on v1's stored shares; the parent trained on neither.",
    length="Own still minus reference still also differs in pose, since the two stills are different frames of different motion: it does not isolate length. True length minus predicted is the direct test.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- the draws and what they rest on -------------------------------------------------------------------------------------------------

def check_draw(k: int) -> None:
    if k not in DRAW_INPUTS:
        raise SystemExit(f"draw {k}: the pinned draws are {DRAWS}")


def top10_key(k: int) -> str:
    return f"{A5_PREFIX}/draw{k}-attempt1-top10.json"


def cache_key(k: int) -> str:
    return f"{A5_PREFIX}/draw{k}-cache/attempt1/index.json"


def extra_key(k: int) -> str:
    return f"{A5_PREFIX}/draw{k}-extra/attempt1/index.json"


def check_top10(report: dict, k: int) -> None:
    """A draw's stored report holds the settings this read rests on: its own two caches, the top-10 arm, predicted timing, all of D20."""
    settings = report.get("settings") or {}
    r20 = settings.get("round20") or {}
    cache, extra, vocabulary = r20.get("word_cache") or {}, r20.get("word_cache_extra") or {}, r20.get("vocabulary_extra") or {}
    pins = DRAW_INPUTS[k]
    differs = [what for what, ok in (("cache key", cache.get("key") == cache_key(k)), ("cache hash", cache.get("sha256") == pins["cache"]), ("cache draw", cache.get("draw") == k), ("extra key", extra.get("key") == extra_key(k)), ("extra hash", extra.get("sha256") == pins["extra"]),
                                     ("arm", vocabulary.get("top") == 10 and vocabulary.get("sha256") == T.TOKENS_SHA), ("margin", settings.get("core_margin") == 0), ("predicted", r20.get("predicted_duration") is True), ("sentences", (r20.get("sentences") or {}).get("sha256") == L.D20_SHA), ("clips", settings.get("clips") == L.D20_CLIPS)) if not ok]
    if differs:
        raise SystemExit(f"draw {k}: its stored report does not rest on this draw's caches and the top-10 arm: {differs}")


def load_top10(store, k: int) -> dict:
    check_draw(k)
    raw = store.get(top10_key(k))
    P.check_sha(raw, DRAW_INPUTS[k]["top10"], top10_key(k))
    report = json.loads(raw)
    check_top10(report, k)
    return report


def run_inputs(args) -> int:
    """The job's first step: every draw's stored report by its hash, holding its pinned caches, and the summary's hash. Prints hashes, no score."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    for k in DRAWS:
        load_top10(store, k)
    P.check_sha(store.get(A5_SUMMARY), A5_SUMMARY_SHA, A5_SUMMARY)
    for name in ANCHORS:
        Q.stored_form(store, Q.STORED_PARENT, name)
    print(json.dumps(dict(inputs=dict(draws=list(DRAWS), summary=A5_SUMMARY_SHA, caches={k: DRAW_INPUTS[k]["cache"][:8] for k in DRAWS}))), flush=True)
    return 0


# ---- composed states -------------------------------------------------------------------------------------------------------------------

def export_argv(k: int, clips: int, prefix: str, listing: str, sha: str, output: str) -> list[str]:
    """``round15_compose_eval run``'s arguments for draw ``k``: comp_a5.sh's top10 phase, without its likelihood, with the export. No instrument is loaded."""
    check_draw(k)
    pins = DRAW_INPUTS[k]
    return ["run", "--mix-key", L.EVAL_MIX, "--mix-sha256", L.EVAL_MIX_SHA, "--sentences-key", L.D20_KEY, "--sentences-sha256", L.D20_SHA, "--clips", str(clips), "--selection", "stage1", "--timing", "sign-fitted", "--expect-model-sha256", MODEL_SHA, "--device", "cpu",
            "--variants", "core_crossfade", "--word-cache-key", cache_key(k), "--word-cache-sha256", pins["cache"], "--word-cache-n", "16", "--recognizer-run", "sign-recognizer-v2", "--core-margin", "0", "--predicted-duration", f"--expect-duration-model={DURATION}",
            "--vocabulary-extra-file", T.TOKENS_FILE, "--vocabulary-extra-sha256", T.TOKENS_SHA, "--vocabulary-extra-top", "10", "--word-cache-extra-key", extra_key(k), "--word-cache-extra-sha256", pins["extra"],
            "--export-prefix", prefix, "--export-arm", f"a5-draw{k}-top10", "--export-sentences-file", listing, "--export-sentences-sha256", sha, "--export-only", "--export-forms", ",".join(EXPORT_FORMS), "--output-file", output]


def export_differences(index: dict, k: int, clips: int) -> dict:
    """Pinned settings a stored export did not use: {setting: [pinned, stored]}. A stale or mismatched export shows here."""
    pins = DRAW_INPUTS[k]
    want = dict(mix_sha256=L.EVAL_MIX_SHA, sentences_sha256=L.D20_SHA, clips=clips, selection="stage1", timing="sign-fitted", expect_model_sha256=MODEL_SHA, variants="core_crossfade", word_cache_key=cache_key(k), word_cache_sha256=pins["cache"], word_cache_n=16, core_margin=0., predicted_duration=True, expect_duration_model=DURATION,
                vocabulary_extra_top=10, vocabulary_extra_sha256=T.TOKENS_SHA, word_cache_extra_key=extra_key(k), word_cache_extra_sha256=pins["extra"], export_only=True, export_forms=",".join(EXPORT_FORMS))
    settings = index.get("settings") or {}
    return {key: [value, settings.get(key)] for key, value in want.items() if settings.get(key) != value}


def check_export(index: dict, k: int, clips: int) -> None:
    if index.get("kind") != "round15-composed-states":
        raise SystemExit(f"draw {k}: not a composed-state export")
    differs = export_differences(index, k, clips)
    if differs:
        raise SystemExit(f"draw {k}: the export's settings differ from this draw's pins: {differs}")
    if (index.get("word_cache") or {}).get("draw") != k:
        raise SystemExit(f"draw {k}: the export's word cache is draw {(index.get('word_cache') or {}).get('draw')}")


def export_index_key(prefix: str, k: int) -> str:
    return f"{prefix.rstrip('/')}/draw{k}/index.json"


def run_export(args) -> int:
    """One draw's composed states, write-once, nothing read by an instrument: a stored export of these settings is left; another refuses."""
    from train_text_to_motion import make_store
    P.check_pins()
    check_draw(args.draw)
    store = make_store(args.store)
    key = export_index_key(args.prefix, args.draw)
    if store.exists(key):
        check_export(json.loads(store.get(key)), args.draw, args.clips)
        print(json.dumps(dict(export=key, stored=True)), flush=True)
        return 0
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    signs = [r["sign"] for r in records][:args.clips]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    listing = work / f"export-d{args.draw}-{args.clips}-list.json"
    payload = json.dumps(dict(name=f"parent-comp-d20-first-{args.clips}", signs=signs), indent=1).encode()
    listing.write_bytes(payload)
    output = work / f"export-d{args.draw}-{args.clips}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-out.json"
    started = time.time()
    done = subprocess.run([sys.executable, str(ROOT / "scripts/round15_compose_eval.py"), *export_argv(args.draw, args.clips, key.rsplit("/index.json", 1)[0], str(listing), hashlib.sha256(payload).hexdigest(), str(output))], cwd=ROOT)
    if done.returncode:
        raise SystemExit(f"draw {args.draw}: the export stopped with {done.returncode}")
    check_export(json.loads(store.get(key)), args.draw, args.clips)
    print(json.dumps(dict(export=key, stored=False, seconds=round(time.time() - started, 1))), flush=True)
    return 0


def load_export(store, prefix: str, k: int, records, declared: bool) -> dict:
    """{form: its rows, in list order}, checked against this draw's pins. A clip missing from an export refuses. On D20 each row is the record's: its sign, text and reference hash. A smoke's rows are the export's first clips, read against other references."""
    key = export_index_key(prefix, k)
    if not store.exists(key):
        raise SystemExit(f"{key}: no export of draw {k}")
    index = json.loads(store.get(key))
    check_export(index, k, len(records))
    out = {}
    for form in EXPORT_FORMS:
        rows = [r for r in index["states"] if r["form"] == form]
        by_sign = {r["sign"]: r for r in rows}
        if declared:
            missing = [r["sign"] for r in records if r["sign"] not in by_sign]
            if missing:
                raise SystemExit(f"draw {k} {form}: {len(missing)} clips are missing from the export, for example {missing[0]}")
            for record in records:
                row = by_sign[record["sign"]]
                if row["text"] != record["text"] or row["reference_sha256"] != record["continuous"]["state_sha256"]:
                    raise SystemExit(f"{record['sign']}: draw {k}'s row is another clip's")
            out[form] = [by_sign[r["sign"]] for r in records]
        else:
            if len(rows) != len(records):
                raise SystemExit(f"draw {k} {form}: {len(rows)} clips are missing from the export, of {len(records)}")
            out[form] = sorted(rows, key=lambda r: r["clip"])
    return out


def composed_states(store, rows) -> list:
    from evaluate_text_to_motion import load_state
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(lambda r: load_state(store, r["state_key"], r["state_sha256"]), rows))


def still_of(state, row: dict, tokenizer):
    """A composition's still at its own length: its nearest-filled middle frame held, flags on. A clip with no signing stays its blank clip."""
    if row["empty"]:
        return state
    return PP.held_still(T.cap_form("v0_nearest_fill", state, tokenizer))


def blank_at_length(ref, frames: int):
    """The reference's blank clip held for ``frames`` frames: ``round15_compose_eval``'s blank@predicted, at the composition's own length."""
    from evaluate_text_to_motion import blanked
    from round15_compose_eval import held
    return held(blanked([ref])[0], frames)


def kind_states(how: str, states, rows, refs, tokenizer) -> list:
    if how == "as_composed":
        return states
    if how == "still":
        return [still_of(s, r, tokenizer) for s, r in zip(states, rows)]
    return [blank_at_length(ref, len(s.timestamps)) for s, ref in zip(states, refs)]


def build_anchor(form: str, refs, tokenizer) -> list:
    from evaluate_text_to_motion import blanked
    if form == "real_two_view":
        return list(refs)
    if form == "blank":
        return blanked(refs)
    bases = [T.cap_form("v0_nearest_fill", s, tokenizer) for s in refs]
    return bases if form == "nearest_two_view" else [PP.held_still(b) for b in bases]


def draw_form(name: str) -> tuple[int, str]:
    match = FORM.fullmatch(name)
    if match is None:
        raise SystemExit(f"unknown form {name}")
    return int(match[1]), match[2]


def forms_of_draw(k: int) -> list[str]:
    check_draw(k)
    return [f"d{k}_{kind}" for kind in KINDS]


# ---- commands --------------------------------------------------------------------------------------------------------------------------

def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def open_list(store, args) -> tuple[list[dict], bool, dict]:
    if args.smoke_train > SMOKE_CLIPS:
        raise SystemExit(f"a smoke reads at most {SMOKE_CLIPS} clips, not {args.smoke_train}")
    records, declared, described = L.open_read_list(store, args.smoke_train)
    R.check_lookup_records(records, declared)
    return records, declared, described


def tokens_of(text: str) -> int:
    from signtranslator.generation.compositional import normalise
    return len(normalise(text))


def read_one(name: str, states, masks, rows, *, reader, texts, signs, groups, candidates, refs, args, declared, described, digest, place, out_dir, started, probe: str = PROBE) -> None:
    built = time.time()
    raw = reader.read(states, texts, candidates, group=args.nll_group)
    validity = [R.part_validity(s) for s in states]
    report = dict(kind=f"{probe}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, signs=signs, groups=groups, references=texts, hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                  empty=[bool(r["empty"]) for r in rows] if rows else None, word_counts=[len(r["words"]) for r in rows] if rows else None, sentence_tokens=[tokens_of(x) for x in texts], frames=[len(s.timestamps) for s in states], reference_frames=[len(r.timestamps) for r in refs], motion=L.motion_statistics(states, masks), hands=P.hand_statistics(states, masks), tracked=P.tracked_statistics(states), seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5) for s, r in zip(states, refs)], valid_frame_share=[R.valid_frame_share(s) for s in states], part_validity={part: [v[part] for v in validity] for part in R.PARTS},
                  build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
    P.write_report_once(out_dir / f"form-{name}.json", report)
    print(json.dumps({name: dict(clips=len(signs), build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))}), flush=True)


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    torch.set_num_threads(args.threads)
    if bool(args.forms) == bool(args.draws):
        raise SystemExit("give exactly one of --forms control and --draws")
    if args.forms and args.forms != "control":
        raise SystemExit(f"--forms {args.forms}: only control")
    draws = [int(x) for x in args.draws.split(",") if x] if args.draws else []
    for k in draws:
        check_draw(k)
    wanted = list(ANCHORS) if args.forms else [n for k in draws for n in forms_of_draw(k)]
    out_dir = Path(args.out_dir)
    stored = [n for n in wanted if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    wanted = [n for n in wanted if n not in stored]
    if not wanted:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = open_list(store, args)
    digest, place = code_digest(), P.environment()
    if draws and declared:
        R.passed_control(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    candidates = M.retrieval_candidates(len(texts), CANDIDATES, seed=0)
    reader = PP.load_unfitted(store, args)
    shared = dict(reader=reader, texts=texts, signs=signs, groups=groups, candidates=candidates, refs=refs, args=args, declared=declared, described=described, digest=digest, place=place, out_dir=out_dir)
    for name in [n for n in wanted if n in ANCHORS]:
        started = time.time()
        read_one(name, build_anchor(name, refs, tokenizer), list(refs), None, started=started, **shared)
    for k in draws:
        names = [n for n in forms_of_draw(k) if n in wanted]
        if not names:
            continue
        exported = load_export(store, args.composed, k, records, declared)
        for form in EXPORT_FORMS:
            started = time.time()
            rows = exported[form]
            states = composed_states(store, rows)
            for name in names:
                kind = draw_form(name)[1]
                if KINDS[kind][0] != form:
                    continue
                built = kind_states(KINDS[kind][1], states, rows, refs, tokenizer)
                read_one(name, built, list(refs), rows, started=started, **shared)
                started = time.time()
            del states
    return 0


def run_control(args) -> int:
    """Exit 3, voiding the read, unless the anchors reproduce the parts job's stored forms. A smoke reads train clips, so it checks the pins and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in ANCHORS}
    if not read["real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the control's forms")
    stored = {name: Q.stored_form(store, Q.STORED_PARENT, name) for name in ANCHORS}      # a pin failure stops here with status 1: no control runs, nothing is voided
    rows, misses = {}, []
    for name in ANCHORS:
        try:
            rows[name] = Q.same_read(read[name], stored[name])
        except SystemExit as error:
            rows[name] = dict(misses=[str(error)])
        misses += [f"{name} {m}" for m in rows[name]["misses"]]
    result = dict(stored=rows, misses=misses, reproduced=not misses, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the anchors do not reproduce their stored reads: {misses}. The read is void: no composition follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def fidelity_floor(clips: int) -> int:
    return int(np.ceil(FIDELITY_SHARE * clips))


def frames_off(index: dict, stored: dict) -> dict:
    """Every exported clip's frames against the stored report's plan, both forms. A clip with no signing has no plan, and its row must say empty. {compared, off}."""
    plans = {row["sign"]: (row.get("plans") or {}).get("core_crossfade") for row in stored["per_clip"]}
    compared, off = 0, []
    for row in index["states"]:
        plan = plans.get(row["sign"], "absent")
        if plan == "absent":
            off.append(row["sign"])
            continue
        if row["empty"] or plan is None:
            if not (row["empty"] and plan is None):
                off.append(row["sign"])
            continue
        want = plan["predicted"]["frames"] if row["form"] == PREDICTED else plan["frames"]
        compared += 1
        if row["frames"] != want:
            off.append(row["sign"])
    return dict(compared=compared, off=len(off), first_off=off[0] if off else None)


def fidelity_passed(record: dict, declared: bool) -> bool:
    return not declared or (record["frames"]["off"] == 0 and record["identical"] >= fidelity_floor(record["clips"]))


def run_fidelity(args) -> int:
    """Before a draw's read: its exported frames against the stored plan, and v1 on a few exported clips, hypotheses only, against the stored report's, of which only the count is kept. No score is computed. On D20 a miss exits 1 and that draw is not read. A record that stands is checked again, not redone."""
    from score_back_translation import translate
    from train_text_to_motion import make_store
    P.check_pins()
    check_draw(args.draw)
    path = Path(args.out_dir) / f"fidelity-d{args.draw}.json"
    declared = args.clips == L.D20_CLIPS
    if path.exists():
        result = json.loads(path.read_bytes())
    else:
        store = make_store(args.store)
        stored = load_top10(store, args.draw)
        index = json.loads(store.get(export_index_key(args.composed, args.draw)))
        check_export(index, args.draw, args.clips)
        rows = sorted((r for r in index["states"] if r["form"] == PREDICTED and not r["empty"]), key=lambda r: r["clip"])[:FIDELITY_CLIPS]
        where = {s: i for i, s in enumerate(stored["signs"])}
        if any(r["sign"] not in where for r in rows):
            raise SystemExit(f"draw {args.draw}: an exported clip is not in its stored report")
        frames = frames_off(index, stored)
        reader = P.load_reader(store, args)
        said = translate(reader.model, reader.kind, composed_states(store, rows))
        kept = stored["forms"][PREDICTED]["hypotheses"]
        identical = sum(a == kept[where[r["sign"]]] for a, r in zip(said, rows))
        result = dict(kind=f"{PROBE}-fidelity", report_only=True, draw=args.draw, clips=len(rows), identical=identical, frames=frames, code_digest=code_digest(), environment=P.environment(), instrument=reader.info["sha256"], note="hypotheses compared and dropped: no BLEU, NLL or text kept")
        P.write_report_once(path, result)
    print(json.dumps({k: result[k] for k in ("draw", "clips", "identical", "frames")}), flush=True)
    if not fidelity_passed(result, declared):
        raise SystemExit(f"draw {args.draw}: {result['identical']} of {result['clips']} exported clips reproduce their stored hypotheses (floor {fidelity_floor(result['clips'])}), {result['frames']['off']} frame counts off: this draw is not read")
    return 0


# ---- reading ---------------------------------------------------------------------------------------------------------------------------

def over_draws(per_draw, draws_boot) -> dict:
    """Values per draw, their mean, the t interval on the draws alone, and the bootstrap over clips with the draws fixed (``draws_boot``: draws x resamples)."""
    v = np.asarray(per_draw, np.float64)
    return dict(per_draw=[round(float(x), 5) for x in v], mean=round(float(v.mean()), 5), between_draw_sd=round(float(v.std(ddof=1)), 5) if len(v) > 1 else None, t_ci95=A5.t_interval(v) if len(v) > 1 else None, clips_ci95=P._quantiles(np.asarray(draws_boot, np.float64).mean(0)), draws=len(v))


def ratio_draws(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return top / bottom


def bleu_of(read: dict) -> np.ndarray:
    return M.bleu_statistics(read["hypotheses"], read["references"])


def draw_rows(k: int, reads: dict, anchors: dict, idx: np.ndarray) -> dict:
    """One draw's arrays and row, against its own pod's anchors."""
    comp, still, true, still_true, blank = (reads[f"d{k}_{kind}"] for kind in ("comp", "still", "comp_true", "still_true", "blank"))
    two_still = Q.margins(anchors["still_two_view"])
    real = Q.margins(anchors["real_two_view"]) - two_still
    nearest = Q.margins(anchors["nearest_two_view"]) - two_still
    margin = Q.margins(comp) - Q.margins(still)
    margin_true = Q.margins(true) - Q.margins(still_true)
    over_reference_still = Q.margins(comp) - two_still
    stats = bleu_of(comp)
    true_stats = bleu_of(true)
    frames, reference = np.asarray(comp["frames"], np.float64), np.asarray(comp["reference_frames"], np.float64)
    real_stats, blank_stats = bleu_of(anchors["real_two_view"]), bleu_of(blank)
    own, real_own, blank_own = (-np.asarray(r["nll"])[:, 0] for r in (comp, anchors["real_two_view"], blank))
    likelihood_top, likelihood_bottom = own - blank_own, real_own - blank_own
    words = np.asarray(comp["word_counts"], np.float64)
    tokens = np.asarray(comp["sentence_tokens"], np.float64)
    return dict(margin=margin, real_stats=real_stats, blank_stats=blank_stats, likelihood_top=likelihood_top, likelihood_bottom=likelihood_bottom, coverage=np.divide(words, tokens, out=np.zeros_like(words), where=tokens > 0), non_empty=~np.asarray(comp["empty"], bool), margin_true=margin_true, real=real, nearest=nearest, over_reference_still=over_reference_still, stats=stats, true_stats=true_stats, log_ratio=np.log(frames / reference), still_raw=Q.margins(still),
                row=dict(bleu_gap_share=A.share_row(stats, real_stats, blank_stats, boot=BOOT), likelihood_gap_share=dict(point=round(float(likelihood_top.mean() / likelihood_bottom.mean()), 4) if likelihood_bottom.mean() else None, ci95=P._quantiles(ratio_draws(likelihood_top[idx].mean(1), likelihood_bottom[idx].mean(1)))), margin=dict(point=round(float(margin.mean()), 5), ci95=P._quantiles(margin[idx].mean(1))), share_of_real=dict(point=round(float(margin.mean() / real.mean()), 4) if real.mean() else None, ci95=P._quantiles(ratio_draws(margin[idx].mean(1), real[idx].mean(1)))), share_of_nearest=dict(point=round(float(margin.mean() / nearest.mean()), 4) if nearest.mean() else None, ci95=P._quantiles(ratio_draws(margin[idx].mean(1), nearest[idx].mean(1)))),
                         bleu=dict(point=round(float(M.bleu_from_statistics(stats.sum(0))), 4), ci95=P._quantiles(M.bleu_from_statistics(stats[idx].sum(1)))), true_length=dict(margin=round(float(margin_true.mean()), 5), share_of_real=round(float(margin_true.mean() / real.mean()), 4) if real.mean() else None, bleu=round(float(M.bleu_from_statistics(true_stats.sum(0))), 4)),
                         empty=int(sum(comp["empty"])), empty_true_length=int(sum(true["empty"])), seconds_over_reference=round(float(np.mean(frames / reference)), 4), still_bleu=round(float(M.bleu_from_statistics(bleu_of(still).sum(0))), 4), wrist_speed_over_real=round(float(np.mean([x for x in comp["motion"]["wrist_speed"] if x is not None]) / np.mean([x for x in anchors["real_two_view"]["motion"]["wrist_speed"] if x is not None])), 4)))


def composition_block(arrays: dict, idx: np.ndarray) -> dict:
    """The eight draws together: over_draws on margin, its shares, BLEU, and the length rows."""
    order = sorted(arrays)
    a = [arrays[k] for k in order]
    margin_boot = np.stack([x["margin"][idx].mean(1) for x in a])
    real_boot = np.stack([x["real"][idx].mean(1) for x in a])
    nearest_boot = np.stack([x["nearest"][idx].mean(1) for x in a])
    true_boot = np.stack([x["margin_true"][idx].mean(1) for x in a])
    reference_boot = np.stack([x["over_reference_still"][idx].mean(1) for x in a])
    bleu_boot = np.stack([M.bleu_from_statistics(x["stats"][idx].sum(1)) for x in a])
    true_bleu_boot = np.stack([M.bleu_from_statistics(x["true_stats"][idx].sum(1)) for x in a])
    means = dict(margin=[float(x["margin"].mean()) for x in a], true=[float(x["margin_true"].mean()) for x in a], real=[float(x["real"].mean()) for x in a], nearest=[float(x["nearest"].mean()) for x in a], reference=[float(x["over_reference_still"].mean()) for x in a], bleu=[float(M.bleu_from_statistics(x["stats"].sum(0))) for x in a], true_bleu=[float(M.bleu_from_statistics(x["true_stats"].sum(0))) for x in a])
    share = [m / r if r else float("nan") for m, r in zip(means["margin"], means["real"])]
    share_nearest = [m / r if r else float("nan") for m, r in zip(means["margin"], means["nearest"])]
    share_true = [m / r if r else float("nan") for m, r in zip(means["true"], means["real"])]
    bleu_of_stats = lambda s: float(M.bleu_from_statistics(s.sum(0)))      # noqa: E731
    gap = [(bleu_of_stats(x["stats"]) - bleu_of_stats(x["blank_stats"])) / (bleu_of_stats(x["real_stats"]) - bleu_of_stats(x["blank_stats"])) if bleu_of_stats(x["real_stats"]) != bleu_of_stats(x["blank_stats"]) else float("nan") for x in a]
    gap_boot = np.stack([ratio_draws(M.bleu_from_statistics(x["stats"][idx].sum(1)) - M.bleu_from_statistics(x["blank_stats"][idx].sum(1)), M.bleu_from_statistics(x["real_stats"][idx].sum(1)) - M.bleu_from_statistics(x["blank_stats"][idx].sum(1))) for x in a])
    likelihood_gap = [float(x["likelihood_top"].mean() / x["likelihood_bottom"].mean()) if x["likelihood_bottom"].mean() else float("nan") for x in a]
    likelihood_boot = np.stack([ratio_draws(x["likelihood_top"][idx].mean(1), x["likelihood_bottom"][idx].mean(1)) for x in a])
    mean_x = np.mean([x["log_ratio"] for x in a], 0)
    mean_y = np.mean([x["margin"] for x in a], 0)
    return dict(draws=order, margin=over_draws(means["margin"], margin_boot), share_of_real=over_draws(share, ratio_draws(margin_boot, real_boot)), share_of_nearest=over_draws(share_nearest, ratio_draws(margin_boot, nearest_boot)), bleu=over_draws(means["bleu"], bleu_boot), bleu_gap_share=over_draws(gap, gap_boot), likelihood_gap_share=over_draws(likelihood_gap, likelihood_boot),
                length=dict(true_length_margin=over_draws(means["true"], true_boot), true_length_share_of_real=over_draws(share_true, ratio_draws(true_boot, real_boot)), true_length_bleu=over_draws(means["true_bleu"], true_bleu_boot), true_minus_predicted=over_draws([t - m for t, m in zip(means["true"], means["margin"])], true_boot - margin_boot), bleu_true_minus_predicted=over_draws([t - m for t, m in zip(means["true_bleu"], means["bleu"])], true_bleu_boot - bleu_boot),
                            over_reference_still=over_draws(means["reference"], reference_boot), own_still_minus_reference_still=over_draws([m - r for m, r in zip(means["margin"], means["reference"])], margin_boot - reference_boot), per_clip_margin_on_log_length_ratio=R.regression(mean_x.tolist(), mean_y.tolist()),
                            note="each form over its own still: true minus predicted is the same words' margin at another length; own still minus reference still also differs in pose, so it does not isolate length; the regression is across sentences, whose content differs with length, and sets nothing"))


def word_rows(arrays: dict) -> dict:
    """Report only: margin and share of real on non-empty clips, and by the share of a sentence's tokens with a word, pooled over the draws."""
    def row(pick) -> dict:
        kept = [(a["margin"][pick(a)], a["real"][pick(a)]) for a in arrays.values()]
        counts = [len(m) for m, _ in kept]
        if not sum(counts):
            return dict(clips_per_draw=0.)
        margin, real = np.concatenate([m for m, _ in kept]), np.concatenate([r for _, r in kept])
        return dict(clips_per_draw=round(float(np.mean(counts)), 1), margin=round(float(margin.mean()), 5), real_margin=round(float(real.mean()), 5), share_of_real=round(float(margin.mean() / real.mean()), 4) if real.mean() else None)
    return dict(report_only=True, non_empty=row(lambda a: a["non_empty"]), by_coverage={f"{lo}-{hi}": row(lambda a, lo=lo, hi=hi: a["non_empty"] & (a["coverage"] > lo) & (a["coverage"] <= hi)) for lo, hi in COVERAGE_BINS})


def restrict_form(read: dict, signs) -> dict:
    """``read`` on ``signs``: ``path_read.restrict``, with this job's per-clip fields kept in step."""
    where = {s: i for i, s in enumerate(read["signs"])}
    pick = [where[s] for s in signs]
    out = R.restrict(read, signs)
    for key in ("frames", "reference_frames", "empty", "word_counts", "sentence_tokens"):
        if read.get(key) is not None:
            out[key] = [read[key][i] for i in pick]
    return out


def signer_rows(pods: dict, draw_pods: dict, signs, groups, idx_seed: int = 0) -> dict:
    """Report only, no label: each signer's margin and shares over the draws, every form restricted to that signer's clips."""
    out = {}
    for signer in sorted(set(groups)):
        kept = [s for s, g in zip(signs, groups) if g == signer]
        idx = np.random.default_rng(idx_seed).integers(0, len(kept), (BOOT, len(kept)))
        arrays = {}
        for k, pod in sorted(draw_pods.items()):
            here = {n: restrict_form(r, kept) for n, r in pods[pod]["forms"].items() if n in ANCHORS or draw_form(n)[0] == k}
            built = draw_rows(k, here, {a: here[a] for a in ANCHORS}, idx)
            built.pop("row")
            arrays[k] = built
        block = composition_block(arrays, idx) if len(kept) >= 3 else None
        out[signer] = dict(clips=len(kept), **({k: block[k] for k in ("margin", "share_of_real", "share_of_nearest", "bleu_gap_share", "likelihood_gap_share")} if block else {}))
    return out


def check_still_lengths(k: int, reads: dict) -> None:
    """Each form's baseline holds its frames, clip by clip: otherwise duration has not cancelled."""
    for form, base in SAME_LENGTH:
        if reads[f"d{k}_{form}"]["frames"] != reads[f"d{k}_{base}"]["frames"]:
            raise SystemExit(f"draw {k}: d{k}_{base} is not as long as d{k}_{form} on every clip")


def anchor_rows(anchors: dict, idx: np.ndarray) -> dict:
    two_still = Q.margins(anchors["still_two_view"])
    out = {}
    for name, read in anchors.items():
        margin = Q.margins(read) - two_still
        out[name] = dict(margin_over_still=dict(point=round(float(margin.mean()), 5), ci95=P._quantiles(margin[idx].mean(1))), raw_margin=round(float(Q.margins(read).mean()), 5), bleu=M.bleu_ci_fast(read["hypotheses"], read["references"])[0])
    return out


def subset_rows(arrays: dict, masks: dict) -> dict:
    """Report only: margin and share of real on clips in and outside v1's training list, over the draws."""
    out = {}
    for name, mask in masks.items():
        if mask.sum() < 2:
            out[name] = None
            continue
        means = [float(a["margin"][mask].mean()) for a in arrays.values()]
        reals = [float(a["real"][mask].mean()) for a in arrays.values()]
        out[name] = dict(clips=int(mask.sum()), margin=round(float(np.mean(means)), 5), real_margin=round(float(np.mean(reals)), 5), share_of_real=round(float(np.mean(means) / np.mean(reals)), 4) if np.mean(reals) else None)
    return out


def stored_likelihood_share(report: dict) -> float | None:
    """v1's likelihood gap share of a stored top10, ``comp_a_read.form_read``'s formula on ``likelihood.per_form``: (form − blank) / (real − blank), None values left out."""
    per_form = report["likelihood"]["per_form"]
    mean_of = lambda x: float(np.mean([v for v in x if v is not None]))      # noqa: E731
    a, b, r = mean_of(per_form[PREDICTED]), mean_of(per_form["blank@predicted"]), mean_of(per_form["real"])
    return round((a - b) / (r - b), 4) if r != b else None


def v1_beside(summary: dict, top10: dict, masks: dict, signs) -> dict:
    """v1's stored top10, per draw: the summary's gap share and BLEU, its likelihood gap share from each report, and the BLEU share on clips outside v1's training list. A report of other clips or another order refuses."""
    for k, report in top10.items():
        if report["signs"] != list(signs):
            raise SystemExit(f"draw {k}: its stored report holds other clips or another order")
    predicted = summary["forms"]["predicted"]
    likelihood = [stored_likelihood_share(top10[k]) for k in sorted(top10)]
    out = dict(likelihood_gap_share=likelihood, gap_share=predicted["gap_share"]["top10"], bleu=predicted["top10_bleu"], mean_gap_share=round(float(np.mean(predicted["gap_share"]["top10"])), 4), mean_bleu=round(float(np.mean(predicted["top10_bleu"])), 4), t_ci95_gap_share=A5.t_interval(predicted["gap_share"]["top10"]))
    outside = []
    mask = masks["outside_v1_training"]
    for k in sorted(top10):
        report = top10[k]
        refs = report["references"]
        stat = lambda form: M.bleu_from_statistics(M.bleu_statistics(report["forms"][form]["hypotheses"], refs)[mask].sum(0))      # noqa: E731
        real, blank = stat("real"), stat("blank@predicted")
        outside.append(round(float((stat(PREDICTED) - blank) / (real - blank)), 4) if real != blank else None)
    kept = [x for x in outside if x is not None]
    out["gap_share_outside_v1_training"] = outside
    out["mean_gap_share_outside_v1_training"] = round(float(np.mean(kept)), 4) if kept else None
    return out


def paired_with_v1(block: dict, beside: dict) -> dict:
    """Parent minus v1, draw by draw, like with like, with a t interval over the draws."""
    def row(mine: list, theirs: list) -> dict:
        d = [a - b for a, b in zip(mine, theirs) if a is not None and b is not None]
        return dict(per_draw=[round(float(x), 4) for x in d], mean=round(float(np.mean(d)), 4), t_ci95=A5.t_interval(d) if len(d) > 1 else None)
    return dict(bleu_gap_share=row(block["bleu_gap_share"]["per_draw"], beside["gap_share"]), likelihood_gap_share=row(block["likelihood_gap_share"]["per_draw"], beside["likelihood_gap_share"]), note="parent minus v1 on the same metric. A margin share is not compared with v1's BLEU share: the scales differ about twofold.")


def plan_misses(anchors: dict, beside: dict) -> list[str]:
    """This read's anchors and v1's stored means against the plan's figures."""
    two_still = Q.margins(anchors["still_two_view"])
    near = float((Q.margins(anchors["nearest_two_view"]) - two_still).mean())
    out = []
    if abs(near - PLAN["nearest_margin"]) > PLAN_TOLERANCE["margin"]:
        out.append(f"nearest_two_view's margin is {near:.4f}, not the plan's {PLAN['nearest_margin']}")
    for form, figure in (("nearest_two_view", "nearest_bleu"), ("real_two_view", "real_bleu"), ("blank", "blank_bleu"), ("still_two_view", "still_bleu")):
        here = M.bleu_ci_fast(anchors[form]["hypotheses"], anchors[form]["references"])[0]
        if abs(here - PLAN[figure]) > PLAN_TOLERANCE["bleu"]:
            out.append(f"{form} reads BLEU {here}, not the plan's {PLAN[figure]}")
    if abs(beside["mean_gap_share"] - PLAN["v1_share"]) > PLAN_TOLERANCE["share"]:
        out.append(f"v1's stored mean share is {beside['mean_gap_share']}, not the plan's {PLAN['v1_share']}")
    if abs(beside["mean_bleu"] - PLAN["v1_bleu"]) > PLAN_TOLERANCE["v1_bleu"]:
        out.append(f"v1's stored mean BLEU is {beside['mean_bleu']}, not the plan's {PLAN['v1_bleu']}")
    return out


def load_pods(store, prefix: str, pods: list[str]) -> dict:
    """{pod: {name: form}, its control and fidelity records}. Every key is named exactly; other files are left."""
    out = {}
    for pod in pods:
        base = f"{prefix.rstrip('/')}/{pod}/forms/"
        here = dict(forms={}, control=None, fidelity={})
        for key in store.list(base):
            name = key[len(base):]
            if name == "control.json":
                here["control"] = json.loads(store.get(key))
            elif re.fullmatch(r"fidelity-d[1-8]\.json", name):
                record = json.loads(store.get(key))
                here["fidelity"][record["draw"]] = record
            elif name.startswith("form-") and name.endswith(".json"):
                form = name[5:-5]
                if form not in ANCHORS:
                    draw_form(form)
                read = json.loads(store.get(key))
                if read.get("form") != form:
                    raise SystemExit(f"{key} holds form {read.get('form')!r}")
                here["forms"][form] = read
        out[pod] = here
    return out


def draws_of_pods(pods: dict) -> dict:
    """{draw: its pod}. A draw with some of its forms, or read on two pods, refuses."""
    out = {}
    for pod, here in pods.items():
        seen = {draw_form(n)[0] for n in here["forms"] if n not in ANCHORS}
        for k in sorted(seen):
            if any(n not in here["forms"] for n in forms_of_draw(k)):
                raise SystemExit(f"draw {k} on pod {pod}: some of its forms are missing")
            if k in out:
                raise SystemExit(f"draw {k} was read on pods {out[k]} and {pod}")
            out[k] = pod
    return out


def run_ready(args) -> int:
    """Exit 0 when all eight draws stand across the pods, 5 otherwise, with what is missing. A smoke's single draw does not count."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    pods = [p for p in args.pods.split(",") if p]
    missing = []
    for k in DRAWS:
        names = forms_of_draw(k)
        if not any(all(store.exists(f"{args.prefix.rstrip('/')}/{pod}/forms/form-{n}.json") for n in names) for pod in pods):
            missing.append(k)
    print(json.dumps(dict(ready=not missing, missing=missing)), flush=True)
    return 5 if missing else 0


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    pods = load_pods(store, args.prefix, [p for p in args.pods.split(",") if p])
    everything = {f"{pod}/{n}": r for pod, here in pods.items() for n, r in here["forms"].items()}
    if not everything:
        raise SystemExit(f"{args.prefix}: no form")
    digests = {r.get("code_digest") for r in everything.values()}
    if digests != {code_digest()}:
        raise SystemExit(f"the forms are of code {sorted(str(d) for d in digests)}, not this code {code_digest()}: a stale read")
    signatures = {(r["declared_read"], r["list"]["list"], tuple(r["signs"]), tuple(r["references"])) for r in everything.values()}
    if len(signatures) != 1:
        raise SystemExit("the forms read other clips, another list, or a smoke mixed with a declared read")
    if {r["instrument"]["sha256"] for r in everything.values()} != {PP.UNFITTED_SHA}:
        raise SystemExit("a form was not read by the parent")
    first = next(iter(everything.values()))
    declared = first["declared_read"]
    draw_pods = draws_of_pods(pods)
    if declared and tuple(sorted(draw_pods)) != DRAWS:
        raise SystemExit(f"incomplete: draws {sorted(draw_pods)}; the read names {list(DRAWS)}")
    if not draw_pods:
        raise SystemExit("no draw was read")
    alerts, environments, fidelity = [], {}, {}
    for pod, here in pods.items():
        if not here["forms"]:
            continue
        if any(a not in here["forms"] for a in ANCHORS):
            raise SystemExit(f"pod {pod}: its anchors are missing")
        _, environments[pod] = P.check_same_run(here["forms"], f"pod {pod}'s forms")
        if declared and (not here["control"] or here["control"].get("reproduced") is not True or here["control"].get("code_digest") != code_digest() or here["control"].get("environment") != environments[pod]):
            raise SystemExit(f"pod {pod}: no passed control of this code and environment; its read is void")
        fidelity.update(here["fidelity"])
    n = len(first["signs"])
    idx = np.random.default_rng(0).integers(0, n, (BOOT, n))
    arrays, rows = {}, {}
    for k, pod in sorted(draw_pods.items()):
        here = pods[pod]["forms"]
        check_still_lengths(k, here)
        built = draw_rows(k, here, {a: here[a] for a in ANCHORS}, idx)
        rows[k] = dict(built.pop("row"), pod=pod)
        arrays[k] = built
        if rows[k]["empty"] != rows[k]["empty_true_length"]:
            alerts.append(f"draw {k}: {rows[k]['empty']} empty clips at predicted timing, {rows[k]['empty_true_length']} at true length")
        record = fidelity.get(k)
        if record is None or record.get("code_digest") != code_digest():
            alerts.append(f"draw {k}: no fidelity record of this code")
        elif not fidelity_passed(record, declared):
            alerts.append(f"draw {k}: its fidelity record is under its floor: {record['identical']} of {record['clips']}, {record['frames']['off']} frame counts off")
    first_pod = pods[draw_pods[min(draw_pods)]]["forms"]
    anchors = {a: first_pod[a] for a in ANCHORS}
    block = composition_block(arrays, idx)
    masks = Q.subset_masks(first["signs"], Q.v1_training_signs(store))
    beside = None
    if declared:
        raw = store.get(A5_SUMMARY)
        P.check_sha(raw, A5_SUMMARY_SHA, A5_SUMMARY)
        beside = v1_beside(json.loads(raw), {k: load_top10(store, k) for k in DRAWS}, masks, first["signs"])
        beside["parent_minus_v1"] = paired_with_v1(block, beside)
        alerts += plan_misses(anchors, beside)
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE: composed D20 sentences against c02 train references; these numbers are plumbing, not a read", list=first["list"], code_digest=code_digest(), environments=environments, disclosures=DISCLOSURES, alerts=alerts, instrument=first["instrument"], clips=dict(list=n, in_v1_training=int(masks["in_v1_training"].sum()), outside_v1_training=int(masks["outside_v1_training"].sum())),
                  anchors=anchor_rows(anchors, idx), per_draw={str(k): rows[k] for k in sorted(rows)}, composition=block, by_v1_training=subset_rows(arrays, masks), per_signer=signer_rows(pods, draw_pods, first["signs"], first["groups"]), words=word_rows(arrays), v1_stored=beside, fidelity={str(k): {x: fidelity[k][x] for x in ("clips", "identical", "frames")} for k in sorted(fidelity)}, checks=dict(controls={pod: here["control"] for pod, here in pods.items()}, blank_identical_to_real=round(float(np.mean([a == b for a, b in zip(anchors["blank"]["hypotheses"], anchors["real_two_view"]["hypotheses"])])), 4)),
                  resources={k: {x: r[x] for x in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for k, r in everything.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, anchors=report["anchors"], composition={k: block[k] for k in ("margin", "share_of_real", "share_of_nearest", "bleu", "bleu_gap_share", "likelihood_gap_share")}, length={k: block["length"][k] for k in ("true_minus_predicted", "own_still_minus_reference_still")}, v1_stored=None if beside is None else {k: beside[k] for k in ("mean_gap_share", "likelihood_gap_share", "mean_gap_share_outside_v1_training", "mean_bleu", "parent_minus_v1")}, empty={k: rows[k]["empty"] for k in rows}, alerts=alerts)), flush=True)
    return 0


def run_put_report(args) -> int:
    """The merged report to its final key, write-once."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    L.refuse_existing(store, args.key, None)
    store.put(args.key, Path(args.file).read_bytes())
    print(json.dumps(dict(report=args.key)), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, a declared read with a failed control, or any alert. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    problems = list(report["alerts"])
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if report["checks"]["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read") and not all((c or {}).get("reproduced") for c in report["checks"]["controls"].values()):
        problems.append("a declared read without its passed control")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs", help="the job's first step: every draw's pins; prints no score")
    inputs.add_argument("--store", default="r2")
    export = sub.add_parser("export", help="one draw's composed states; nothing is read by an instrument")
    export.add_argument("--store", default="r2")
    export.add_argument("--draw", type=int, required=True)
    export.add_argument("--clips", type=int, default=L.D20_CLIPS)
    export.add_argument("--work", required=True)
    export.add_argument("--prefix", required=True)
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", help="control: the anchors")
    read.add_argument("--draws", help="comma list: each draw's four forms")
    read.add_argument("--composed", help="the prefix of the composed exports")
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=4)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"c02 train references, at most {SMOKE_CLIPS}, not D20")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="the anchors against their stored reads; exit 3, voiding the read, otherwise")
    control.add_argument("--store", default="r2")
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    fidelity = sub.add_parser("fidelity", help="v1 on a few exported clips: the count of stored hypotheses reproduced, nothing else kept")
    fidelity.add_argument("--store", default="r2")
    fidelity.add_argument("--draw", type=int, required=True)
    fidelity.add_argument("--clips", type=int, default=L.D20_CLIPS, help="the export's clips")
    fidelity.add_argument("--composed", required=True)
    fidelity.add_argument("--out-dir", required=True)
    fidelity.add_argument("--instrument-cache")
    fidelity.add_argument("--device", default="cpu")
    ready = sub.add_parser("ready", help="exit 0 when all eight draws stand across the pods, 5 otherwise")
    ready.add_argument("--store", default="r2")
    ready.add_argument("--pods", required=True)
    ready.add_argument("--prefix", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--pods", required=True)
    merge.add_argument("--prefix", required=True)
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    put = sub.add_parser("put-report")
    put.add_argument("--store", default="r2")
    put.add_argument("--file", required=True)
    put.add_argument("--key", required=True)
    log = sub.add_parser("upload-log")
    log.add_argument("--store", default="r2")
    log.add_argument("--file", required=True)
    log.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(inputs=run_inputs, export=run_export, preflight=run_preflight, read=run_read, control=run_control, fidelity=run_fidelity, ready=run_ready, merge=run_merge, check=run_check, upload=PP.run_upload, **{"upload-log": R.run_upload_log, "put-report": run_put_report})[args.command](args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
