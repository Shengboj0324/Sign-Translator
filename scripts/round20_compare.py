"""Round 20's paired comparisons on D20 (docs/NEXT_PHASE_PLAN.md, "Round 20": "a lever is adopted only if both point
estimates improve on D20"). Reads stored reports only.

``forms``: two forms of merged ``round15_compose_eval.py`` reports (``--a KEY:FORM``, ``--b KEY:FORM``; the same
references in the same order) compared sentence by sentence: the paired corpus-BLEU difference a - b
(``sentence_exposure_diagnostic.paired_bleu_difference``: the pilot's bootstrap) and, when both reports carry
``--likelihood``, the paired difference of the selection metric, the reference log-likelihood per token minus its
blank's, per sentence (``round20_lib.paired_mean``). ``both_improve``: both point estimates above 0 (the adoption rule;
intervals reported, not required by the rule).

``words``: two ``round20_word_cache.py`` indexes (``--a KEY``, ``--b KEY``; A5: two priors on the same entries)
compared entry by entry over the entries labelled in both: the probability of the label, top-1 and top-5, on sample 0,
on all samples (each entry's mean) and on each shared best-of-N choice.

Writes JSON to ``--output-key`` / ``--output-file`` (write-once) or prints it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def _split(spec: str) -> tuple[str, str]:
    key, _, form = spec.rpartition(":")
    if not key or not form:
        raise SystemExit(f"{spec!r}: give KEY:FORM")
    return key, form


def compare_forms(a: dict, form_a: str, b: dict, form_b: str, *, boot: int = 1000) -> dict:
    from sentence_exposure_diagnostic import paired_bleu_difference
    from round20_lib import paired_mean
    if a["references"] != b["references"]:
        raise SystemExit("the two reports read other references")
    refs = a["references"]
    for r, f in ((a, form_a), (b, form_b)):
        if f not in r["forms"]:
            raise SystemExit(f"form {f!r} is not in the report")
    bleu = paired_bleu_difference(a["forms"][form_a]["hypotheses"], refs, b["forms"][form_b]["hypotheses"], refs,
                                  n=boot)
    out = dict(sentences=len(refs), a=dict(form=form_a, bleu=a["forms"][form_a]["bleu"]),
               b=dict(form=form_b, bleu=b["forms"][form_b]["bleu"]), bleu=bleu)
    la, lb = (r.get("likelihood", {}).get("per_form", {}) for r in (a, b))
    if form_a in la and form_b in lb and "blank" in la and "blank" in lb:
        da = [None if x is None or y is None else x - y for x, y in zip(la[form_a], la["blank"])]
        db = [None if x is None or y is None else x - y for x, y in zip(lb[form_b], lb["blank"])]
        out["likelihood_minus_blank"] = dict(a=paired_mean(da, boot=boot), b=paired_mean(db, boot=boot),
                                             difference=paired_mean(da, db, boot=boot))
    ll = out.get("likelihood_minus_blank", {}).get("difference")
    out["both_improve"] = bool(bleu["difference"] > 0 and ll is not None and ll["difference"] > 0)
    out["rule"] = "adopt only if both point estimates (paired BLEU, likelihood minus blank) are above 0"
    if ll is None:
        out["note"] = "no likelihood in one of the reports (run with --likelihood): the rule cannot be applied"
    return out


def compare_words(a: dict, b: dict, *, boot: int = 1000) -> dict:
    from round20_lib import paired_mean
    ra = {r["entry"]: r for r in a["entries"] if r["label"]}
    rb = {r["entry"]: r for r in b["entries"] if r["label"]}
    shared = sorted(set(ra) & set(rb))
    if not shared:
        raise SystemExit("no labelled entry in both indexes")

    def reads(pick):
        out = {}
        for name, f in (("p_label", lambda p, k: p), ("top1", lambda p, k: float(k == 1)),
                        ("top5", lambda p, k: float(k <= 5))):
            xa = [pick(ra[e], f) for e in shared]
            xb = [pick(rb[e], f) for e in shared]
            out[name] = dict(a=round(float(np.mean(xa)), 4), b=round(float(np.mean(xb)), 4),
                             difference=paired_mean(xa, xb, boot=boot))
        return out
    out = dict(entries=len(shared), a=dict(checkpoint=a["settings"]["checkpoint_key"]),
               b=dict(checkpoint=b["settings"]["checkpoint_key"]),
               sample0=reads(lambda r, f: f(r["probabilities"][0], r["ranks"][0])),
               all_samples=reads(lambda r, f: float(np.mean([f(p, k) for p, k in zip(r["probabilities"], r["ranks"])]))))
    for n in sorted(set(a["settings"]["choose"]) & set(b["settings"]["choose"])):
        out[f"best_of_{n}"] = reads(lambda r, f, n=n: f(r["probabilities"][r["chosen"][str(n)]["index"]],
                                                       r["ranks"][r["chosen"][str(n)]["index"]]))
    if a["settings"]["recognizer_run"] != b["settings"]["recognizer_run"]:
        out["warning"] = "the two indexes were read by different recognisers"
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("forms", "words"))
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    for k in [args.output_key] if args.output_key else []:
        if store.exists(k):
            raise SystemExit(f"{k} exists: refusing to overwrite")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    if args.mode == "forms":
        (ka, fa), (kb, fb) = _split(args.a), _split(args.b)
        out = compare_forms(json.loads(store.get(ka)), fa, json.loads(store.get(kb)), fb, boot=args.bootstrap)
        out.update(a_key=ka, b_key=kb)
    else:
        out = compare_words(json.loads(store.get(args.a)), json.loads(store.get(args.b)), boot=args.bootstrap)
        out.update(a_key=args.a, b_key=args.b)
    data = json.dumps(out, indent=1).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:
            stream.write(data)
    if args.output_key:
        store.put(args.output_key, data)
    print(data.decode(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
