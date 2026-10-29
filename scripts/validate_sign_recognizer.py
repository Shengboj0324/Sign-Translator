"""Validate the isolated-sign recogniser as an instrument before it scores generated motion.

Reads the run's best checkpoint (EMA weights) and writes one report with:

1. **In-domain**: top-1/top-5 on the ASL Citizen validation signers (never the test partition).
2. **Controls on the same clips**: every channel removed (only duration left), hands only, everything but the hands.
   An instrument that reads signs must collapse without the hands and stay near chance on the blank clips; a high
   blank or no-hands score means it uses shortcuts (duration, face) and its scores on generated motion are not
   meaningful.
3. **Cross-domain**: top-1/top-5 on ASLLVD (a different studio, cameras and signers; DAI 2), training and validation
   signers only (its test signer stays unread), for the Sign Bank glosses that map *exactly* to one ASL Citizen
   label (``asl-citizen-gloss-map.json``, method ``exact``). The recogniser never saw ASLLVD.
4. **Per-gloss reliability** on the ASL Citizen validation clips: the glosses recognised with top-1 >= ``--reliable``
   over at least ``--min-clips`` clips; sign-level evaluation of generated motion is restricted to that list.

5. **In-context signs** (diagnostic, with ``--continuous-manifest-key``): top-1/top-5 on signs cut from ASLLRP
   *continuous* utterances (DAI 2; ``asllrp_sentence_signs`` CSV frame spans, ``--context-seconds`` either side) of
   its training and validation signers - a studio, camera and signers the recogniser never saw, and coarticulated
   signs of ~0.3 s instead of ~1 s citation forms. It shows how far the instrument is from reading signs inside
   sentences; the verdict below does not use it (generated motion is scored on one-sign plans).

The verdict is a fixed rule (``--min-cross-domain-top1``, ``--max-blank-top1``, ``--max-no-hands-top1``), not a judgement
made after seeing the numbers.

**Recogniser v3 (``--gates``;** docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan"). The run's
``run.json`` defines the data (``train_sign_recognizer.build_v3_data``, checked against its ``data_sha256``); every
``val:<corpus>`` and ``holdout:<corpus>`` set is read by v3 and by ``--baseline-run`` (v2) as **real** clips, their
**decoded** copies (the run's ``build_recon_states.py`` index) and **mirrored** (``pose.mirror``, availability checked),
with **controls** (blank: only the duration; no hands; random code: uniformly random tokenizer codes decoded in the
generated form, one clip per real clip and of its length). **Per-gloss reliability** (v2's rule, top-1 >= ``--reliable``
over >= ``--min-clips`` clips) is pooled over the sets, on real and on decoded clips; the **ASLLRP in-context** read is
v2's diagnostic. The report gives the gate quantities G1-G7 exactly as the plan declares them (``GATE_TEXT``), each with
its threshold as a parameter (``GATE_DEFAULTS``), and a PASS/FAIL table; v3 is usable only if every gate passes.
Where the plan gives no number, the parameter's default says how the gate is read: G1 "at chance" is v2's declared
blank rule (top-1 <= 2%) for both controls; G2 compares top-1 per set; G3 and G4 read top-5 (the lexicon's read);
G5 "orientation consistency" requires the stored read above the mirrored read (top-5) in every set, since every corpus
is stored in the generator's orientation; G6 counts a word reliable on both real and decoded clips.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample, mask_channels  # noqa: E402
from signtranslator.generation.sign_recognizer import SignRecognizer, SignRecognizerConfig, top_k_accuracy  # noqa: E402
from evaluate_text_to_motion import blanked  # noqa: E402
from signtranslator.pose.resample import resample_motion  # noqa: E402
from train_text_to_motion import make_store, to_device  # noqa: E402

HANDS = ("left_hand", "right_hand")


def load_model(store, run: str, device):
    best = json.loads(store.get(f"training-runs/{run}/best.json"))
    saved = torch.load(io.BytesIO(store.get(best["checkpoint_key"])), map_location="cpu", weights_only=False)
    model = SignRecognizer(SignRecognizerConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.to(device).eval(), saved["labels"], best


@torch.no_grad()
def logits_of(model, states, device, batch: int = 64) -> torch.Tensor:
    device = torch.device(device)                          # callers pass a name ("cpu", "cuda") or a device
    out = []
    for i in range(0, len(states), batch):
        motion = to_device((torch.zeros(0), torch.zeros(0), collate_multichannel(states[i:i + batch]), None), device)[2]
        out.append(model(motion).cpu())
    return torch.cat(out) if out else torch.zeros(0, 1)


def manifest_of(store, key, sha):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(key) + b"\n")
        return load_manifest(p, expected_sha256=sha)


def cut(state, start: int, stop: int):
    """Frames ``[start, stop)`` of ``state`` as their own state (clock kept)."""
    from dataclasses import replace
    return replace(state, timestamps=state.timestamps[start:stop].copy(),
                   channels={n: replace(c, values=c.values[start:stop], valid=c.valid[start:stop],
                                        observed=c.observed[start:stop], confidence=c.confidence[start:stop])
                             for n, c in state.channels.items()})


def continuous_segments(store, args, to_citizen, index, cache):
    """Signs cut from ASLLRP continuous utterances of the gloss manifest's train/val partitions, labelled through the
    exact Sign Bank -> ASL Citizen map; (states, labels, info)."""
    import csv
    from concurrent.futures import ThreadPoolExecutor
    manifest = manifest_of(store, args.continuous_manifest_key, args.continuous_manifest_sha256)
    utterances = {Path(r["source"]["utterance_video"]).stem: r for r in manifest["records"]
                  if r.get("partition") in ("train", "val") and (r.get("source") or {}).get("utterance_video")}
    rows = list(csv.DictReader(io.StringIO(store.get(args.continuous_csv_key).decode("utf-8-sig"))))
    wanted = collections.defaultdict(list)
    for row in rows:
        stem = Path(row["Utterance video filename"]).stem
        citizen = to_citizen.get(row["Main entry gloss label"], ())
        if stem in utterances and len(citizen) == 1 and row["Hidden"] != "T":
            wanted[stem].append((int(row["Start frame of the sign video"]), int(row["End frame of the sign video"]),
                                 int(row["Start frame of the containing utterance"]),
                                 int(row["End frame of the containing utterance"]), index[next(iter(citizen))]))

    def one(stem):
        state = load_sample(utterances[stem], "continuous", cache=cache, fetch=store.get)
        out, mismatch = [], 0
        for a, b, u0, u1, y in wanted[stem]:
            if u1 - u0 + 1 != len(state.timestamps) or not (u0 <= a <= b <= u1):
                mismatch += 1; continue
            fps = 1. / float(np.median(np.diff(state.timestamps)))
            pad = int(round(args.context_seconds * fps))
            segment = cut(state, max(a - u0 - pad, 0), min(b - u0 + 1 + pad, len(state.timestamps)))
            out.append((resample_motion(segment, args.fps) if args.fps else segment, y, (b - a + 1) / fps))
        return out, mismatch
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(one, sorted(wanted)))
    items = [x for r, _ in results for x in r]
    info = dict(utterances=len(wanted), span_mismatches=sum(m for _, m in results),
                partitions=["train", "val"], context_seconds=args.context_seconds,
                median_sign_seconds=round(float(np.median([x[2] for x in items])), 3) if items else None)
    return [x[0] for x in items], torch.tensor([x[1] for x in items], dtype=torch.long), info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--run", required=True)
    parser.add_argument("--citizen-manifest-key", help="required without --gates")
    parser.add_argument("--citizen-manifest-sha256", help="required without --gates")
    parser.add_argument("--asllvd-manifest-key", help="required without --gates")
    parser.add_argument("--asllvd-manifest-sha256", help="required without --gates")
    parser.add_argument("--gloss-map-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json")
    parser.add_argument("--max-seconds", type=float, default=8.)
    parser.add_argument("--reliable", type=float, default=.8)
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--min-cross-domain-top1", type=float, default=.25)
    parser.add_argument("--max-blank-top1", type=float, default=.02)
    parser.add_argument("--max-no-hands-top1", type=float, default=.15)
    parser.add_argument("--output-key", help="the report in the store (required without --gates)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=0., help="resample clips to this rate (must match training)")
    parser.add_argument("--continuous-manifest-key", help="ASLLRP gloss manifest (utterance states and partitions)")
    parser.add_argument("--continuous-manifest-sha256")
    parser.add_argument("--continuous-csv-key",
                        default="acquisition/2026-10-17-v1/asllrp-dai2-v1/asllrp-continuous/asllrp_sentence_signs_2025_06_28.csv")
    parser.add_argument("--context-seconds", type=float, default=.1)
    parser.add_argument("--cross-partitions", default="train,val",
                        help="ASLLVD partitions for the cross-domain check; only those the recogniser never trained on")
    add_gate_arguments(parser)
    args = parser.parse_args(argv)
    if args.gates:
        return gates_main(args)
    missing = [f"--{k.replace('_', '-')}" for k in ("citizen_manifest_key", "citizen_manifest_sha256",
                                                     "asllvd_manifest_key", "asllvd_manifest_sha256", "output_key")
               if not getattr(args, k)]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    model, glosses, best = load_model(store, args.run, device)
    index = {g: i for i, g in enumerate(glosses)}

    def load(records):
        from concurrent.futures import ThreadPoolExecutor
        records = [r for r in records if r["continuous"].get("seconds", 0.) <= args.max_seconds]

        def one(r):
            s = load_sample(r, "continuous", cache=cache, fetch=store.get)
            return resample_motion(s, args.fps) if args.fps else s
        with ThreadPoolExecutor(16) as pool:               # R2 fetches dominate: load in parallel, keep the order
            states = list(pool.map(one, records))
        return states, torch.tensor([r["_label"] for r in records], dtype=torch.long)

    def score(states, labels):
        return {k: round(v, 4) for k, v in top_k_accuracy(logits_of(model, states, device), labels).items()} if len(labels) else None

    citizen = manifest_of(store, args.citizen_manifest_key, args.citizen_manifest_sha256)
    val = [dict(r, _label=index[r["gloss"]["label"]]) for r in citizen["records"]
           if r.get("partition") == "val" and r["gloss"]["label"] in index]
    states, labels = load(val)
    report = dict(run=args.run, checkpoint=best["checkpoint_key"], step=best["step"], classes=len(glosses), fps=args.fps or None)
    logits = logits_of(model, states, device)
    report["in_domain"] = dict(clips=len(labels), **{k: round(v, 4) for k, v in top_k_accuracy(logits, labels).items()})
    report["controls"] = dict(blank=score(blanked(states), labels),
                              hands_only=score([mask_channels(s, HANDS) for s in states], labels),
                              no_hands=score([mask_channels(s, [n for n in s.channels if n not in HANDS]) for s in states], labels))
    hits, counts = collections.Counter(), collections.Counter()
    for pred, y in zip(logits.argmax(1).tolist(), labels.tolist()):
        counts[y] += 1; hits[y] += pred == y
    reliable = sorted(glosses[y] for y, n in counts.items() if n >= args.min_clips and hits[y] / n >= args.reliable)
    report["per_gloss"] = dict(glosses_with_clips=len(counts), reliable=len(reliable), reliable_glosses=reliable,
                               rule=f"top-1 >= {args.reliable} over >= {args.min_clips} validation clips")

    mapping = json.loads(store.get(args.gloss_map_key))["mapping"]
    to_citizen = collections.defaultdict(set)
    for g, m in mapping.items():
        if m["method"] == "exact" and g in index:
            for c in m["candidates"]:
                to_citizen[c].add(g)
    cross_parts = tuple(args.cross_partitions.split(","))
    if "test" in cross_parts:
        raise SystemExit("the ASLLVD test signer stays unread here")
    asllvd = manifest_of(store, args.asllvd_manifest_key, args.asllvd_manifest_sha256)
    cross = [dict(r, _label=index[next(iter(to_citizen[r["gloss"]["label"]]))]) for r in asllvd["records"]
             if r.get("partition") in cross_parts and len(to_citizen.get(r["gloss"]["label"], ())) == 1]
    cs, cl = load(cross)
    report["cross_domain_asllvd"] = dict(clips=len(cl), glosses=len(set(cl.tolist())), partitions=list(cross_parts),
                                         **(score(cs, cl) or {}))
    if args.continuous_manifest_key:
        segments, labels_c, info = continuous_segments(store, args, to_citizen, index, cache)
        report["in_context_asllrp"] = dict(clips=len(labels_c), glosses=len(set(labels_c.tolist())), **info,
                                           **(score(segments, labels_c) or {}),
                                           blank=score(blanked(segments), labels_c), diagnostic=True)
    c = report["cross_domain_asllvd"].get("top1", 0.)
    verdict = dict(cross_domain_ok=c >= args.min_cross_domain_top1,
                   blank_ok=(report["controls"]["blank"] or {}).get("top1", 1.) <= args.max_blank_top1,
                   no_hands_ok=(report["controls"]["no_hands"] or {}).get("top1", 1.) <= args.max_no_hands_top1)
    verdict["usable_as_instrument"] = all(verdict.values())
    report["verdict"] = verdict
    report["thresholds"] = dict(min_cross_domain_top1=args.min_cross_domain_top1, max_blank_top1=args.max_blank_top1,
                                max_no_hands_top1=args.max_no_hands_top1)
    store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True).encode())
    print(json.dumps({k: v for k, v in report.items() if k != "per_gloss"} | {"reliable_glosses": len(reliable)}))
    return 0



# --------------------------------------------------------------------------------------- recogniser v3: gates G1-G7

# The gates as the plan declares them (docs/NEXT_PHASE_PLAN.md, "Instrument finding and the recogniser v3 plan",
# 2026-10-29 06:37 UTC), verbatim; each one's measured quantity and threshold follow below and are parameters.
GATE_TEXT = {
    "G1": "blank and random-code reads at chance",
    "G2": "per-corpus real reads at least v2's",
    "G3": "Sem-Lex held-out signers ≥ 0.60",
    "G4": "decoded-real reads ≥ 0.70 on the covered words and ≥ 0.60 on the uncovered",
    "G5": "orientation consistency",
    "G6": "per-gloss reliability: at least 100 of 150 R5 words, and at least 25 of the 42 uncovered",
    "G7": "ASLLRP in-context ≥ 0.07",
}
GATE_DEFAULTS = dict(g1_max_top1=.02, g2_metric="top1", g2_margin=0., g3_corpus="semlex-text-v2", g3_metric="top5",
                     g3_min=.60, g4_metric="top5", g4_covered_min=.70, g4_uncovered_min=.60, g5_metric="top5",
                     g5_min_difference=0., g6_min_r5=100, g6_min_uncovered=25, g6_forms="both", g7_min_top1=.07)
WORD_SPLIT_KEY = "acquisition/2026-10-17-v1/youtube-asl-v1/round14/r5-word-split.json"
WORD_SPLIT_SHA256 = "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a"
CONTINUOUS_MANIFEST = ("acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json",
                       "f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944")


def add_gate_arguments(parser) -> None:
    g = parser.add_argument_group("recogniser v3 gates (--gates; docs/NEXT_PHASE_PLAN.md, recogniser v3 plan)")
    g.add_argument("--gates", action="store_true", help="validate a recogniser v3 run (its run.json defines the "
                   "corpora) and report gates G1-G7 with a PASS/FAIL table")
    g.add_argument("--baseline-run", default="sign-recognizer-v2", help="G2's reference instrument")
    g.add_argument("--word-split-key", default=WORD_SPLIT_KEY, help="the R5 words split into covered / uncovered")
    g.add_argument("--word-split-sha256", default=WORD_SPLIT_SHA256)
    g.add_argument("--max-clips", type=int, default=0, help="read at most this many clips per set (0: all; seed 1234)")
    g.add_argument("--seed", type=int, default=0, help="the random-code control's codes")
    g.add_argument("--output", help="also write the report to this local file")
    g.add_argument("--cache-dir", help="content cache directory (default: the temporary directory)")
    g.add_argument("--g1-max-top1", type=float, default=GATE_DEFAULTS["g1_max_top1"],
                   help="G1 'at chance': blank and random-code top-1 at most this (v2's declared blank rule, 2%%)")
    g.add_argument("--g2-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g2_metric"])
    g.add_argument("--g2-margin", type=float, default=GATE_DEFAULTS["g2_margin"],
                   help="G2: every set's v3 real read at least v2's minus this")
    g.add_argument("--g3-corpus", default=GATE_DEFAULTS["g3_corpus"])
    g.add_argument("--g3-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g3_metric"])
    g.add_argument("--g3-min", type=float, default=GATE_DEFAULTS["g3_min"])
    g.add_argument("--g4-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g4_metric"])
    g.add_argument("--g4-covered-min", type=float, default=GATE_DEFAULTS["g4_covered_min"])
    g.add_argument("--g4-uncovered-min", type=float, default=GATE_DEFAULTS["g4_uncovered_min"])
    g.add_argument("--g5-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g5_metric"])
    g.add_argument("--g5-min-difference", type=float, default=GATE_DEFAULTS["g5_min_difference"],
                   help="G5: in every set, the stored read minus the mirrored read must exceed this")
    g.add_argument("--g6-min-r5", type=int, default=GATE_DEFAULTS["g6_min_r5"])
    g.add_argument("--g6-min-uncovered", type=int, default=GATE_DEFAULTS["g6_min_uncovered"])
    g.add_argument("--g6-forms", choices=("both", "real", "decoded"), default=GATE_DEFAULTS["g6_forms"],
                   help="G6: a word counts when reliable on real and decoded clips (both), or on one form")
    g.add_argument("--g7-min-top1", type=float, default=GATE_DEFAULTS["g7_min_top1"])


def ranks_from_logits(logits, labels) -> "np.ndarray":
    """Rank of each clip's true class (0 = top-1; ties count in the clip's favour)."""
    y = torch.as_tensor(labels, dtype=torch.long)
    if not len(y):
        return np.zeros(0, dtype=np.int64)
    return (logits > logits.gather(1, y[:, None])).sum(1).numpy()


def read(ranks, k: int) -> float | None:
    ranks = np.asarray(ranks)
    return round(float((ranks < k).mean()), 4) if len(ranks) else None


def reliable_classes(labels, ranks, *, threshold: float, min_clips: int) -> set:
    """Classes recognised with top-1 >= ``threshold`` over at least ``min_clips`` clips (v2's per-gloss rule)."""
    hits, counts = collections.Counter(), collections.Counter()
    for y, r in zip(np.asarray(labels).tolist(), np.asarray(ranks).tolist()):
        counts[y] += 1
        hits[y] += r == 0
    return {y for y, n in counts.items() if n >= min_clips and hits[y] / n >= threshold}


def evaluate_gates(q: dict, t: dict) -> list[dict]:
    """The seven gates from the measured quantities ``q`` and thresholds ``t`` (``GATE_DEFAULTS``' keys)."""
    out = []

    def gate(gid, measured, rule, ok):
        out.append(dict(gate=gid, plan=GATE_TEXT[gid], measured=measured, rule=rule, passed=bool(ok)))

    c = q["controls"]
    gate("G1", f"blank top-1 {c['blank_top1']}, random-code top-1 {c['random_code_top1']} (chance {c['chance_top1']})",
         f"both <= {t['g1_max_top1']}",
         c["blank_top1"] is not None and c["random_code_top1"] is not None
         and c["blank_top1"] <= t["g1_max_top1"] and c["random_code_top1"] <= t["g1_max_top1"])
    m = t["g2_metric"]
    rows = {n: (s[f"real_{m}"], s[f"v2_real_{m}"]) for n, s in q["per_set"].items()}
    short = {n: (a, b) for n, (a, b) in rows.items() if a is None or b is None or a < b - t["g2_margin"]}
    worst = min((a - b for a, b in rows.values() if a is not None and b is not None), default=None)
    gate("G2", f"{m} v3 - v2, worst set {None if worst is None else round(worst, 4)}"
               + (f"; short: {sorted(short)}" if short else ""),
         f"every set's v3 {m} >= v2's - {t['g2_margin']}", bool(rows) and not short)
    name = f"holdout:{t['g3_corpus']}"
    value = (q["per_set"].get(name) or {}).get(f"real_{t['g3_metric']}")
    gate("G3", f"{name} real {t['g3_metric']} {value}", f">= {t['g3_min']}", value is not None and value >= t["g3_min"])
    cov, unc = q["decoded_words"]["covered"], q["decoded_words"]["uncovered"]
    gate("G4", f"decoded {t['g4_metric']}: covered {cov[t['g4_metric']]} ({cov['clips']} clips), uncovered "
               f"{unc[t['g4_metric']]} ({unc['clips']} clips)",
         f"covered >= {t['g4_covered_min']}, uncovered >= {t['g4_uncovered_min']}",
         cov[t["g4_metric"]] is not None and unc[t["g4_metric"]] is not None
         and cov[t["g4_metric"]] >= t["g4_covered_min"] and unc[t["g4_metric"]] >= t["g4_uncovered_min"])
    m5 = t["g5_metric"]
    diffs = {n: (None if s[f"real_{m5}"] is None or s[f"mirrored_{m5}"] is None
                 else round(s[f"real_{m5}"] - s[f"mirrored_{m5}"], 4)) for n, s in q["per_set"].items()}
    bad = sorted(n for n, d in diffs.items() if d is None or not d > t["g5_min_difference"])
    gate("G5", f"stored - mirrored {m5}, smallest {min((d for d in diffs.values() if d is not None), default=None)}"
               + (f"; not above: {bad}" if bad else ""),
         f"every set's stored {m5} - mirrored {m5} > {t['g5_min_difference']}", bool(diffs) and not bad)
    r = q["reliability"]
    gate("G6", f"reliable ({t['g6_forms']}): {r['r5']} of {r['r5_words']} R5 words, {r['uncovered']} of "
               f"{r['uncovered_words']} uncovered",
         f">= {t['g6_min_r5']} and >= {t['g6_min_uncovered']}",
         r["r5"] >= t["g6_min_r5"] and r["uncovered"] >= t["g6_min_uncovered"])
    v = q["in_context_top1"]
    gate("G7", f"in-context ASLLRP top-1 {v}", f">= {t['g7_min_top1']}", v is not None and v >= t["g7_min_top1"])
    return out


def gate_table(gates: list[dict]) -> str:
    lines = ["| Gate | Plan | Measured | Rule | Result |", "|---|---|---|---|---|"]
    lines += [f"| {g['gate']} | {g['plan']} | {g['measured']} | {g['rule']} | {'PASS' if g['passed'] else 'FAIL'} |"
              for g in gates]
    return "\n".join(lines)


def gates_main(args) -> int:
    """Recogniser v3 validation: every ``val:`` and ``holdout:`` set of the run's own data definition read real,
    decoded and mirrored, with blank / no-hands / random-code controls, v2 beside it, per-gloss reliability on real and
    decoded clips, the ASLLRP in-context read, and gates G1-G7 (a PASS/FAIL table; the instrument is usable only if
    every gate passes, otherwise v2 stays)."""
    import argparse
    import hashlib
    import random
    from concurrent.futures import ThreadPoolExecutor
    from signtranslator.data_engineering.corpus_manifest import ContentCache, mask_channels
    from signtranslator.generation.sign_recognizer import GlossMotionDataset
    from signtranslator.generation.text_to_motion import diffusion_targets
    from build_recon_states import load_tokenizer
    from evaluate_text_to_motion import blanked
    from exposure_diagnostic import make_post
    from pilot_content_check import mirrored_read
    from semlex_recognition_shift import bootstrap
    from signtranslator.generation.compositional import make_decode
    import train_sign_recognizer as trainer
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: the gate report is written once")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    cache = ContentCache(Path(args.cache_dir or tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    prefix = f"training-runs/{args.run}"
    run = json.loads(store.get(f"{prefix}/run.json"))
    if not run.get("recognizer_v3"):
        raise SystemExit(f"{args.run} is not a recogniser v3 run (use the v2 checks without --gates)")
    targs = argparse.Namespace(**{**trainer.V3_DEFAULTS, **run["args"]})
    records, labels, summary, template, digest = trainer.build_v3_data(store, targs, cache, prefix)
    if digest != run["data_sha256"]:
        raise SystemExit("the run's data definition no longer gives the data it trained on")
    model, glosses, best = load_model(store, args.run, device)
    baseline, base_glosses, base_best = load_model(store, args.baseline_run, device)
    if glosses != labels or base_glosses != glosses:
        raise SystemExit("the v3 run, its data and the baseline must share one label list")
    index = {g: i for i, g in enumerate(glosses)}
    split_payload = store.get(args.word_split_key)
    if hashlib.sha256(split_payload).hexdigest() != args.word_split_sha256:
        raise SystemExit(f"{args.word_split_key}: SHA-256 differs from --word-split-sha256")
    split = json.loads(split_payload)
    covered, uncovered = set(split["covered"]), set(split["uncovered"])
    if covered & uncovered or not (covered | uncovered) <= set(index):
        raise SystemExit("the word split must partition words of the label space")
    recon = summary.get("recon") or {}
    config = recon.get("index_config") or {}
    if not config.get("tokenizer_key"):
        raise SystemExit("the run has no decoded-state index: the decoded reads and the random-code control need one")
    tok_payload = store.get(config["tokenizer_key"])
    if hashlib.sha256(tok_payload).hexdigest() != config["tokenizer_sha256"]:
        raise SystemExit("the tokenizer differs from the decoded states' configuration")
    tokenizer, _ = load_tokenizer(tok_payload, device)
    del tok_payload
    decode = make_decode(tokenizer, tokenizer.config.motion, make_post(config.get("decoder_gains")))
    fps = targs.fps or None
    kw = dict(labels=index, cache=cache, fetch=store.get, min_hand_fraction=targs.min_hand_fraction, fps=fps,
              template=template, hand_fraction_fallback=True)
    manifest = dict(records=records)
    names = sorted({r["partition"] for r in records if r["partition"].startswith(("val:", "holdout:"))})

    def ranks(net, states, y):
        return ranks_from_logits(logits_of(net, states, device), y)

    def random_codes(states, seed):
        generator = torch.Generator().manual_seed(seed)
        out = []
        with torch.no_grad():
            for s in states:
                motion = collate_multichannel([s])
                unit = diffusion_targets(tokenizer.config.motion, motion).to(device)
                _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
                drawn = torch.randint(0, tokenizer.config.codebook, tuple(codes.shape), generator=generator)
                out.append(decode(drawn.to(codes.device), int(motion.frame_valid.shape[1]), "random"))
        return out

    def load_all(dataset, ids):
        with ThreadPoolExecutor(16) as pool:
            return [item[1] for item in pool.map(dataset.__getitem__, ids)]

    def scores(r, groups=None):
        out = {}
        for k in (1, 5):
            h = (np.asarray(r) < k).astype(np.float64)
            out[f"top{k}"] = read(r, k)
            out[f"top{k}_ci"] = bootstrap(h) if len(h) else None
            if groups is not None and len(set(groups)) >= 3:
                out[f"top{k}_ci_signers"] = bootstrap(h, groups)
        return out

    from signtranslator.generation.recognizer_corpora import signer_of
    per_set, pooled = {}, collections.defaultdict(list)
    for n, name in enumerate(names):
        real_ds = GlossMotionDataset(manifest, partition=name, **kw)
        ids = [i for i, r in enumerate(real_ds.records) if r["continuous"].get("seconds", 0.) <= targs.max_seconds]
        if args.max_clips:
            ids = sorted(random.Random(1234).sample(ids, min(args.max_clips, len(ids))))
        if not ids:
            continue
        dec_ds = GlossMotionDataset(manifest, partition=name, decoded=True, **kw)
        position = {r["sign"]: i for i, r in enumerate(dec_ds.records)}
        paired = [i for i in ids if real_ds.records[i]["sign"] in position]
        y = [index[real_ds.records[i]["gloss"]["label"]] for i in ids]
        y_dec = [index[real_ds.records[i]["gloss"]["label"]] for i in paired]
        groups = [signer_of(real_ds.records[i]) for i in ids]
        states = load_all(real_ds, ids)
        forms = dict(real=ranks(model, states, y), v2_real=ranks(baseline, states, y),
                     mirrored=ranks(model, [mirrored_read(s) for s in states], y),
                     blank=ranks(model, blanked(states), y),
                     no_hands=ranks(model, [mask_channels(s, [c for c in s.channels if c not in HANDS]) for s in states],
                                    y),
                     random_code=ranks(model, random_codes(states, args.seed + 1000 * n), y))
        del states
        decoded = load_all(dec_ds, [position[real_ds.records[i]["sign"]] for i in paired])
        forms["decoded"] = ranks(model, decoded, y_dec)
        forms["v2_decoded"] = ranks(baseline, decoded, y_dec)
        del decoded
        row = dict(clips=len(ids), decoded_clips=len(paired), signers=len(set(groups)),
                   glosses=len(set(y)), forms={f: scores(r, groups if len(r) == len(ids) else None)
                                              for f, r in forms.items()})
        for f, r in forms.items():
            for k in (1, 5):
                row[f"{f}_top{k}"] = read(r, k)
        per_set[name] = row
        pooled["y"] += y
        pooled["real"] += forms["real"].tolist()
        pooled["y_dec"] += y_dec
        pooled["decoded"] += forms["decoded"].tolist()
        for f in ("blank", "random_code", "no_hands"):
            pooled[f] += forms[f].tolist()
        print(json.dumps({name: {k: v for k, v in row.items() if k != "forms"}}), flush=True)
    y_all, y_dec = np.asarray(pooled["y"]), np.asarray(pooled["y_dec"])
    controls = dict(clips=len(y_all), chance_top1=round(1 / len(glosses), 6), chance_top5=round(5 / len(glosses), 6),
                    blank_top1=read(pooled["blank"], 1), blank_top5=read(pooled["blank"], 5),
                    random_code_top1=read(pooled["random_code"], 1), random_code_top5=read(pooled["random_code"], 5),
                    no_hands_top1=read(pooled["no_hands"], 1), no_hands_top5=read(pooled["no_hands"], 5))
    decoded_words = {}
    for subset, words in (("covered", covered), ("uncovered", uncovered)):
        mask = np.isin(y_dec, [index[w] for w in words])
        r = np.asarray(pooled["decoded"])[mask]
        decoded_words[subset] = dict(clips=int(mask.sum()), words_read=int(len(set(y_dec[mask].tolist()))),
                                     words=len(words), top1=read(r, 1), top5=read(r, 5))
    rule = dict(threshold=args.reliable, min_clips=args.min_clips)
    real_ok = reliable_classes(y_all, pooled["real"], **rule)
    dec_ok = reliable_classes(y_dec, pooled["decoded"], **rule)
    counted = {"both": real_ok & dec_ok, "real": real_ok, "decoded": dec_ok}[args.g6_forms]
    r5 = {index[w] for w in covered | uncovered}
    reliability = dict(rule=f"top-1 >= {args.reliable} over >= {args.min_clips} clips, pooled over every val and "
                            f"holdout set", forms=args.g6_forms, r5_words=len(r5), uncovered_words=len(uncovered),
                       r5=len(counted & r5), uncovered=len(counted & {index[w] for w in uncovered}),
                       r5_real=len(real_ok & r5), r5_decoded=len(dec_ok & r5), reliable_real=len(real_ok),
                       reliable_decoded=len(dec_ok), reliable_both=len(real_ok & dec_ok),
                       reliable_glosses_real=sorted(glosses[i] for i in real_ok),
                       reliable_glosses_decoded=sorted(glosses[i] for i in dec_ok))
    # in-context ASLLRP signs (the v2 diagnostic, now gate G7)
    if not args.continuous_manifest_key:
        args.continuous_manifest_key, args.continuous_manifest_sha256 = CONTINUOUS_MANIFEST
    args.fps = targs.fps
    mapping = json.loads(store.get(targs.gloss_map_key))["mapping"]
    to_citizen = collections.defaultdict(set)
    for g, m in mapping.items():
        if m["method"] == "exact" and g in index:
            for c in m["candidates"]:
                to_citizen[c].add(g)
    segments, labels_c, info = continuous_segments(store, args, to_citizen, index, cache)
    seg_ranks = ranks(model, segments, labels_c)
    in_context = dict(clips=len(labels_c), **info, top1=read(seg_ranks, 1), top5=read(seg_ranks, 5),
                      v2_top1=read(ranks(baseline, segments, labels_c), 1),
                      blank_top1=read(ranks(model, blanked(segments), labels_c), 1))
    del segments
    thresholds = {k: getattr(args, k) for k in GATE_DEFAULTS}
    quantities = dict(controls=controls, per_set=per_set, decoded_words=decoded_words, reliability=reliability,
                      in_context_top1=in_context["top1"])
    gates = evaluate_gates(quantities, thresholds)
    report = dict(run=args.run, checkpoint=best["checkpoint_key"], step=best["step"], data_sha256=digest,
                  baseline=dict(run=args.baseline_run, checkpoint=base_best["checkpoint_key"]),
                  word_split=dict(key=args.word_split_key, sha256=args.word_split_sha256),
                  decoded_states=dict(prefix=recon.get("prefix"), index_sha256=recon.get("index_sha256")),
                  declared="docs/NEXT_PHASE_PLAN.md, Instrument finding and the recogniser v3 plan (2026-10-29 06:37 UTC)",
                  thresholds=thresholds, **quantities, in_context_asllrp=in_context, gates=gates,
                  all_gates_pass=all(g["passed"] for g in gates),
                  verdict="v3 usable as the instrument" if all(g["passed"] for g in gates) else "v2 stays")
    payload = json.dumps(report, indent=1, sort_keys=True, default=float).encode()
    if args.output:
        Path(args.output).write_bytes(payload)
    if args.output_key:
        store.put(args.output_key, payload)
    print(gate_table(gates))
    print(json.dumps(dict(verdict=report["verdict"], sha256=hashlib.sha256(payload).hexdigest())), flush=True)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
