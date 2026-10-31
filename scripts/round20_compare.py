"""Round 20's paired comparisons on D20 (docs/NEXT_PHASE_PLAN.md, "Round 20": "a lever is adopted only if both point
estimates improve on D20"). Reads stored reports only.

``forms``: two forms of merged ``round15_compose_eval.py`` reports (``--a KEY:FORM``, ``--b KEY:FORM``; refused unless
both read the same clip ids and references in order, with the same derangement seed, back-translator checkpoint and
sentence-set hash) compared sentence by sentence: the paired corpus-BLEU difference a - b
(``sentence_exposure_diagnostic.paired_bleu_difference``: the pilot's bootstrap) and, when both reports carry
``--likelihood``, the paired difference of the selection metric, the reference log-likelihood per token minus its
blank's, per sentence (``round20_lib.paired_mean``). ``both_improve``: both point estimates above 0 (the adoption rule;
intervals reported, not required by the rule). Each report's checkpoint and mix keys are in the output (``check_pairable``
does not compare them). The **spotting guard** (``spotting_guard``, its 95% lower bound in ``spotting_guard_lower_bound``;
``both_improve`` ignores it) is the paired per-sentence difference of recogniser v2's recall, read from each report's shards
(merged reports keep no per-clip spotting; ``--a-shard-keys``/``--b-shard-keys``, default ``<report>-shard<s>.json``). It is
withheld across word sources (real words against generated ones: the recogniser reads them with different biases), where a
cached generated word counts as generated.

``same-codes``: ``--a SHARD --b SHARD``: a short run's clips (a0's smoke) against the same clips of a stored run, before a
paired re-decode (A7) relies on sampling the same codes. Exits non-zero unless ``--form``'s per-clip motion is exactly equal
on the first ``--first`` clips and the two runs share their generation settings. Reports store no codes, so equality is
necessary, not sufficient: only these clips are read, and a draw that changes neither the kept frames nor the keypoint
statistics goes unseen.

``words``: two ``round20_word_cache.py`` indexes (``--a KEY``, ``--b KEY``; A5: two priors on the same entries)
compared entry by entry over the entries labelled in both: the probability of the label, top-1 and top-5, on sample 0,
on all samples (each entry's mean) and on each shared best-of-N choice. Refused between a gained and an ungained cache
(``--no-decoder-gains``): the difference would mix the decoder into the prior's.

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


def check_pairable(a: dict, b: dict) -> None:
    """Stop unless the two merged reports read the same clips (ids and references, in order) with the same reference
    derangement seed, the same back-translator checkpoint and the same sentence set (its hash)."""
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    differ = [name for name, x, y in (
        ("references", a.get("references"), b.get("references")), ("clip ids", a.get("signs"), b.get("signs")),
        ("seed", sa.get("seed"), sb.get("seed")), ("translator", sa.get("translator_key"), sb.get("translator_key")),
        ("sentence set", ((sa.get("round20") or {}).get("sentences") or {}).get("sha256"),
         ((sb.get("round20") or {}).get("sentences") or {}).get("sha256"))) if x != y]
    if differ:
        raise SystemExit(f"the two reports are not paired: other {', '.join(differ)}")


SPOTTING_RECOGNISER = "sign-recognizer-v2"
SOURCE_NOTE = ("meaningful only between forms of the same word source: real words and generated words are read by the "
               "recogniser with different biases")
# what decides the sampled codes and the kept frames (same-codes); other settings (clips, reads) may differ
GENERATION_SETTINGS = ("checkpoint_key", "generator_mix", "mix_key", "mix_sha256", "corpora", "selection", "timing",
                       "core_margin", "rerank", "total_fit", "guidance", "part_temperature", "model_sha256")
GENERATION_ROUND20 = ("vocabulary_mix", "semlex_core", "activity_margin", "retime", "tempo", "lowpass_hz", "word_cache",
                      "no_decoder_gains")


def word_source(report: dict, form: str) -> list[str]:
    """The kinds of word in a form: ``generated`` (fresh or cached) or the real words' sources."""
    sources = ((report.get("round20") or {}).get("per_variant") or {}).get(form, {}).get("sources")
    if sources is None:
        sources = ["generated"] if form.startswith("core_") else [form]
    return sorted({"generated" if s in ("generated", "cache") else s for s in sources})


def shard_recalls(report: dict, shards: list[dict], form: str) -> list | None:
    """Per clip, in the report's order, the form's own-word recall in the report's shards; None if they do not read it."""
    rows = sorted((r for s in shards for r in s["clips"]), key=lambda r: r["clip"])
    if [r["sign"] for r in rows] != report["signs"] or any(s["settings"] != report["settings"] for s in shards):
        raise SystemExit("the given shards are not this report's (other clips or settings)")
    if any(form not in r["spotting"] for r in rows):
        return None
    return [r["spotting"][form]["own"]["recall"] for r in rows]


def spotting_guard(a: dict, form_a: str, b: dict, form_b: str, shards_a, shards_b, *, boot: int = 1000) -> dict:
    """The paired recogniser-v2 recall difference a - b, or ``{"available": False, "reason"}``. Only after
    ``check_pairable``: the same clips and references."""
    def absent(reason):
        return dict(available=False, reason=reason, note=SOURCE_NOTE)
    from round20_lib import paired_mean
    if shards_a is None or shards_b is None:
        return absent("the shards of a report were not found")
    runs = {r.get("settings", {}).get("recognizer_run") for r in (a, b)}
    if runs != {SPOTTING_RECOGNISER}:
        return absent(f"spotting was read by {sorted(map(str, runs))}, not {SPOTTING_RECOGNISER} on both")
    sources = word_source(a, form_a), word_source(b, form_b)
    if sources[0] != sources[1]:
        return absent(f"other word sources: {sources[0]} against {sources[1]}")
    ra, rb = shard_recalls(a, shards_a, form_a), shard_recalls(b, shards_b, form_b)
    if ra is None or rb is None:
        return absent("a form has no per-clip spotting (only the composed variants do)")
    d = paired_mean(ra, rb, boot=boot)
    if d is None:
        return absent("no sentence has a recall in both")
    return dict(available=True, recogniser=SPOTTING_RECOGNISER, word_source=sources[0],
                a=paired_mean(ra, boot=boot), b=paired_mean(rb, boot=boot), difference=d, note=SOURCE_NOTE)


def same_codes_check(short: dict, stored: dict, form: str = "core_crossfade", *, first: int = 5) -> dict:
    """Stop unless the short run's first clips are the stored run's and read exactly the same (module docstring)."""
    wrong = []
    sa, sb = short.get("settings") or {}, stored.get("settings") or {}
    wrong += [f"settings.{k}: {sa.get(k)!r} against {sb.get(k)!r}" for k in GENERATION_SETTINGS if sa.get(k) != sb.get(k)]
    ra, rb = sa.get("round20") or {}, sb.get("round20") or {}
    wrong += [f"settings.round20.{k}: {ra.get(k)!r} against {rb.get(k)!r}" for k in GENERATION_ROUND20
              if ra.get(k) != rb.get(k)]
    if (ra.get("sentences") or {}).get("sha256") != (rb.get("sentences") or {}).get("sha256"):
        wrong.append("settings.round20.sentences: another sentence set")
    if not 0 < first <= min(sa.get("clips", 0), sb.get("clips", 0)):
        wrong.append(f"--first {first}: the runs have {sa.get('clips')} and {sb.get('clips')} clips")
    clips_a, clips_b = ({r["clip"]: r for r in x.get("clips", [])} for x in (short, stored))
    words = 0
    for i in range(first):
        x, y = clips_a.get(i), clips_b.get(i)
        if x is None or y is None:
            wrong.append(f"clip {i}: not in {'the short run' if x is None else 'the stored run'}")
            continue
        for k in ("sign", "entries", "stage1_words", "frames_reference"):
            if x.get(k) != y.get(k):
                wrong.append(f"clip {i} {k}: {x.get(k)!r} against {y.get(k)!r}")
        mx, my = (r.get("motion", {}).get(form) for r in (x, y))
        if mx is None or my is None:
            wrong.append(f"clip {i}: no motion of {form}")
        else:
            for k in sorted(set(mx) | set(my)):      # by repr: exact, and NaN equals NaN
                if repr(mx.get(k)) != repr(my.get(k)):
                    wrong.append(f"clip {i} motion.{form}.{k}: {mx.get(k)!r} against {my.get(k)!r}")
        words += len(x.get("entries") or ())
    if not words and not wrong:
        wrong.append(f"no word in the first {first} clips: nothing was compared")
    if wrong:
        raise SystemExit("not the same codes: " + "; ".join(wrong[:20]))
    return dict(same=True, form=form, clips=first, words=words)


def report_shards(store, key: str, report: dict, given: str | None):
    """A merged report's shards: the given keys (they must exist), else ``<report>-shard<s>.json``; None if not all there."""
    if given:
        keys = [k for k in given.split(",") if k]
    else:
        stem = key[:-5] if key.endswith(".json") else key
        keys = [f"{stem}-shard{s}.json" for s in report.get("shards") or ()]
        if not keys or not all(store.exists(k) for k in keys):
            return None
    return [json.loads(store.get(k)) for k in keys]


def provenance(report: dict) -> dict:
    s = report.get("settings") or {}
    return dict(checkpoint_key=s.get("checkpoint_key"), mix_key=s.get("mix_key"),
                generator_mix_key=(s.get("generator_mix") or {}).get("key"),
                vocabulary_mix_key=((s.get("round20") or {}).get("vocabulary_mix") or {}).get("key"))


def compare_forms(a: dict, form_a: str, b: dict, form_b: str, *, boot: int = 1000, shards_a=None, shards_b=None) -> dict:
    from sentence_exposure_diagnostic import paired_bleu_difference
    from round20_lib import paired_mean
    check_pairable(a, b)
    refs = a["references"]
    for r, f in ((a, form_a), (b, form_b)):
        if f not in r["forms"]:
            raise SystemExit(f"form {f!r} is not in the report")
    bleu = paired_bleu_difference(a["forms"][form_a]["hypotheses"], refs, b["forms"][form_b]["hypotheses"], refs,
                                  n=boot)
    out = dict(sentences=len(refs), a=dict(form=form_a, bleu=a["forms"][form_a]["bleu"], **provenance(a)),
               b=dict(form=form_b, bleu=b["forms"][form_b]["bleu"], **provenance(b)), bleu=bleu)
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
    guard = out["spotting_guard"] = spotting_guard(a, form_a, b, form_b, shards_a, shards_b, boot=boot)
    out["spotting_guard_lower_bound"] = guard["difference"]["ci95"][0] if guard["available"] else None
    return out


def compare_words(a: dict, b: dict, *, boot: int = 1000) -> dict:
    from round20_lib import paired_mean
    if bool(a["settings"].get("no_decoder_gains")) != bool(b["settings"].get("no_decoder_gains")):
        raise SystemExit("one index is without the decoder gains, the other with them: not a comparison of priors")
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
    p.add_argument("mode", choices=("forms", "words", "same-codes"))
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--a-shard-keys", help="forms: report a's shards, comma-separated (default <report>-shard<s>.json)")
    p.add_argument("--b-shard-keys", help="forms: report b's shards")
    p.add_argument("--form", default="core_crossfade", help="same-codes: the form whose motion is compared")
    p.add_argument("--first", type=int, default=5, help="same-codes: how many first clips")
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
        ra, rb = json.loads(store.get(ka)), json.loads(store.get(kb))
        out = compare_forms(ra, fa, rb, fb, boot=args.bootstrap, shards_a=report_shards(store, ka, ra, args.a_shard_keys),
                            shards_b=report_shards(store, kb, rb, args.b_shard_keys))
        out.update(a_key=ka, b_key=kb)
    elif args.mode == "same-codes":
        out = same_codes_check(json.loads(store.get(args.a)), json.loads(store.get(args.b)), args.form, first=args.first)
        out.update(a_key=args.a, b_key=args.b)
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
