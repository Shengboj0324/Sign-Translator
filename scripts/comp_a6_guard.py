"""Post-S2 A6's checks at run time. ``env``: this pod's python, torch, transformers and tokenizers are A5's pinned ones. ``same-pod KEY``: a stored base report came from this code and this pod's environment, GPU included, so a draw's arms
never mix. ``started KEY``: write-once marker that an arm of an attempt began; an arm that began and did not finish is never rerun under that attempt. ``a5-done``: A5's summary is stored, reads confirmed or direction only and is the one this code's slot names: A6's draws run on nothing less, and reveal no per-draw number of A5's read before it. ``a5-match BASE TOP10``:
a draw's base against A5's top10 on the same words, or the screen's against A3's: exit 0 if it matches, 4 if the same GPU's rate is below 98%, and anything else is a crash, never read as a difference. Counts and rates only, never a score. ``a5-caches TOP10``: the word
caches' keys and hashes from a top10 report's own settings. ``hold-gate SCREEN COMMIT``: exit 0 if the declared screen lets the hold arms run, 3 if not, anything else if it cannot tell."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_lib as lib  # noqa: E402
from comp_a5_guard import check_environment  # noqa: E402
from comp_a6_read import A5_GOES_ON, A5_SUMMARY_AT_DECLARATION, BLANK, FORM, HOLD_NOT_RUN, JOB_FILE, MATCH_RATE, PINNED_THREADS, hold_gate, slot_from_summary  # noqa: E402

A5_SUMMARY = "acquisition/2026-10-17-v1/youtube-asl-v1/round20/comp-a5/a5-summary.json"


def check_same_pod(report: dict) -> None:
    stored = ((report.get("settings") or {}).get("round20") or {}).get("provenance")
    here = lib.provenance_row(JOB_FILE)
    if not stored or stored != here:
        raise SystemExit(f"base came from other code or another environment: {stored} against this pod's {here}")


def check_a5_summary(raw: bytes, slot) -> None:
    outcome = (json.loads(raw).get("rule") or {}).get("outcome")
    if outcome not in A5_GOES_ON:
        raise SystemExit(f"A5's summary reads {outcome}: A6's draws run only on {list(A5_GOES_ON)}")
    digest = hashlib.sha256(raw).hexdigest()
    if not slot or slot != slot_from_summary(json.loads(raw), digest):
        raise SystemExit(f"A5's summary is {digest}, reading {outcome}; this code's slot holds {slot}: the bars were not set against it")


def match_counts(base: dict, other: dict) -> dict:
    """Identical hypotheses of base and A5's top10, form by form, and whether their GPUs agree."""
    count = lambda form: int(sum(p == q for p, q in zip(base["forms"][form]["hypotheses"], other["forms"][form]["hypotheses"])))    # noqa: E731
    gpu = lambda r: r["settings"]["round20"]["provenance"]["environment"].get("gpu")      # noqa: E731
    clips = len(base["references"])
    form, real, blank = count(FORM), count("real"), count(BLANK)
    return dict(form=form, real=real, blank_predicted=blank, clips=clips, rate=round(min(form, real, blank) / clips, 4), same_gpu=gpu(base) == gpu(other), threads=PINNED_THREADS)


def match_stops(counts: dict) -> bool:
    """On the same GPU and pinned threads, below the rate the draw stops. Another GPU is reported."""
    return counts["same_gpu"] and counts["rate"] < MATCH_RATE


def caches_of(report: dict) -> str:
    """KEY SHA EXTRA_KEY EXTRA_SHA of a top10 report's word caches: a draw's words come from A5's own report, not a copied hash."""
    r20 = report["settings"]["round20"]
    return " ".join((r20["word_cache"]["key"], r20["word_cache"]["sha256"], r20["word_cache_extra"]["key"], r20["word_cache_extra"]["sha256"]))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from train_text_to_motion import make_store
    if argv == ["env"]:
        check_environment()
    elif argv[:1] == ["same-pod"] and len(argv) == 2:
        check_same_pod(json.loads(make_store("r2").get(argv[1])))
    elif argv[:1] == ["started"] and len(argv) == 2:
        store = make_store("r2")
        if store.exists(argv[1]):
            raise SystemExit(f"{argv[1]} exists")
        store.put(argv[1], b"started")
    elif argv == ["a5-done"]:
        store = make_store("r2")
        if not store.exists(A5_SUMMARY):
            raise SystemExit(f"{A5_SUMMARY} is not stored: A6's draws wait for A5's summary")
        check_a5_summary(store.get(A5_SUMMARY), A5_SUMMARY_AT_DECLARATION)
    elif argv[:1] == ["a5-caches"] and len(argv) == 2:
        print(caches_of(json.loads(make_store("r2").get(argv[1]))), flush=True)
    elif argv[:1] == ["hold-gate"] and len(argv) == 3:
        gate = hold_gate(json.loads(make_store("r2").get(argv[1])), argv[2])
        print(f"COMP-A6-HOLD-GATE {'open' if gate['passed'] else HOLD_NOT_RUN}", flush=True)
        return 0 if gate["passed"] else 3
    elif argv[:1] == ["a5-match"] and len(argv) == 3:
        store = make_store("r2")
        counts = match_counts(json.loads(store.get(argv[1])), json.loads(store.get(argv[2])))
        print("COMP-A6-MATCH " + json.dumps(counts), flush=True)
        return 4 if match_stops(counts) else 0
    else:
        raise SystemExit("usage: comp_a6_guard.py env | same-pod KEY | started KEY | a5-done | a5-match BASE TOP10 | a5-caches TOP10 | hold-gate SCREEN COMMIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
