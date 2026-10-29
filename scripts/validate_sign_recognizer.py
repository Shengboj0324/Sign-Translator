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
``gate:<corpus>`` set (the validation half never used for checkpoint selection) and ``holdout:<corpus>`` set is read by v3 and by ``--baseline-run`` (v2) as **real** clips, their
**decoded** copies (the run's ``build_recon_states.py`` index) and **mirrored** (``pose.mirror``, availability checked),
with **controls** (blank: only the duration; no hands; random code: uniformly random tokenizer codes decoded in the
generated form, one clip per real clip and of its length). **Per-gloss reliability** (v2's rule, top-1 >= ``--reliable``
over >= ``--min-clips`` clips) is pooled over the sets, on real and on decoded clips; the **ASLLRP in-context** read is
v2's diagnostic. The report gives the gate quantities G1-G7 exactly as the plan declares them (``GATE_TEXT``), each with
its threshold as a parameter (``GATE_DEFAULTS``), and a PASS/FAIL table; v3 is usable only if every gate passes.
Where the plan gives no number, the parameter's default says how the gate is read (independent review, 2026-10-29):
G1 "at chance" is per set, for both controls: top-1 <= max(the set's majority-class share, 3/classes) and top-5 <=
max(the share of its five most frequent classes, 15/classes); G2 compares top-1 per set; G3 and G4 read top-5 (the
lexicon's read), G4 pooled over covered and uncovered words and per corpus (``--g4-corpus-min``, default Sem-Lex
decoded top-5 >= 0.50 over its gate half and held-out signers), with every set's decoded read reported; G5
"orientation consistency" requires, in every set, the bootstrap 95% lower bound of the paired stored-minus-mirrored
top-5 above 0, since every corpus is stored in the generator's orientation; G6 counts a word reliable on both real
and decoded clips.

**G6 calibrated on v2 (``--g6-calibration``;** recogniser v3 amendment 2, 2026-10-29 13:00 UTC). ``--baseline-run``
(v2) is read on exactly the clips ``--run``'s gate report read (its gate halves and held-out sets, recorded split,
label space and conform template; every set's clip count is checked against the report), real clips only, by G6's
rule; G6' passes when v3's both-forms R5 count is at least v2's real-clip R5 count, overall and among the uncovered
words. The report (v2's counts and reliable glosses, G6') is written once to ``training-runs/<run>/g6-calibration.json``.
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
    if args.gates and args.g6_calibration:
        parser.error("--gates and --g6-calibration are separate runs")
    if args.gates:
        return gates_main(args)
    if args.g6_calibration:
        return g6_calibration_main(args)
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
GATE_DEFAULTS = dict(g1_top1_floor_classes=3, g1_top5_floor_classes=15, g2_metric="top1", g2_margin=0.,
                     g3_corpus="semlex-text-v2", g3_metric="top5", g3_min=.60, g4_metric="top5", g4_covered_min=.70,
                     g4_uncovered_min=.60, g4_corpus_min={"semlex-text-v2": .50}, g5_metric="top5",
                     g5_min_lower_bound=0., g6_min_r5=100, g6_min_uncovered=25, g6_forms="both", g7_min_top1=.07)
GATE_SETS = ("gate:", "holdout:")     # the gates read the gate halves and the held-out signers, never the select halves
WORD_SPLIT_KEY = "acquisition/2026-10-17-v1/youtube-asl-v1/round14/r5-word-split.json"
WORD_SPLIT_SHA256 = "676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a"
CONTINUOUS_MANIFEST = ("acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json",
                       "f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944")


def add_gate_arguments(parser) -> None:
    g = parser.add_argument_group("recogniser v3 gates (--gates; docs/NEXT_PHASE_PLAN.md, recogniser v3 plan)")
    g.add_argument("--gates", action="store_true", help="validate a recogniser v3 run (its run.json defines the "
                   "corpora) and report gates G1-G7 with a PASS/FAIL table")
    g.add_argument("--g6-calibration", action="store_true", help="amendment 2: read --baseline-run on --run's gate "
                   "halves and held-out sets (real clips) by G6's reliability rule and decide G6' against --run's "
                   "stored gate report (--gates-key)")
    g.add_argument("--gates-key", help="--g6-calibration: the v3 gate report (default training-runs/<run>/gates.json)")
    g.add_argument("--baseline-run", default="sign-recognizer-v2", help="G2's reference instrument")
    g.add_argument("--word-split-key", default=WORD_SPLIT_KEY, help="the R5 words split into covered / uncovered")
    g.add_argument("--word-split-sha256", default=WORD_SPLIT_SHA256)
    g.add_argument("--max-clips", type=int, default=0, help="read at most this many clips per set (0: all; seed 1234)")
    g.add_argument("--seed", type=int, default=0, help="the random-code control's codes")
    g.add_argument("--output", help="also write the report to this local file")
    g.add_argument("--cache-dir", help="content cache directory (default: the temporary directory)")
    g.add_argument("--g1-top1-floor-classes", type=int, default=GATE_DEFAULTS["g1_top1_floor_classes"],
                   help="G1 'at chance', per set: blank and random-code top-1 <= max(the set's majority-class share, "
                        "this many classes / class count)")
    g.add_argument("--g1-top5-floor-classes", type=int, default=GATE_DEFAULTS["g1_top5_floor_classes"],
                   help="G1, per set: top-5 <= max(the share of the set's five most frequent classes, this many "
                        "classes / class count)")
    g.add_argument("--g2-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g2_metric"])
    g.add_argument("--g2-margin", type=float, default=GATE_DEFAULTS["g2_margin"],
                   help="G2: every set's v3 real read at least v2's minus this")
    g.add_argument("--g3-corpus", default=GATE_DEFAULTS["g3_corpus"])
    g.add_argument("--g3-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g3_metric"])
    g.add_argument("--g3-min", type=float, default=GATE_DEFAULTS["g3_min"])
    g.add_argument("--g4-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g4_metric"])
    g.add_argument("--g4-covered-min", type=float, default=GATE_DEFAULTS["g4_covered_min"])
    g.add_argument("--g4-uncovered-min", type=float, default=GATE_DEFAULTS["g4_uncovered_min"])
    g.add_argument("--g4-corpus-min", action="append", metavar="CORPUS=MIN",
                   help="G4 per corpus: that corpus's decoded read (gate half and held-out signers) at least MIN; "
                        "repeated; default semlex-text-v2=0.5")
    g.add_argument("--g5-metric", choices=("top1", "top5"), default=GATE_DEFAULTS["g5_metric"])
    g.add_argument("--g5-min-lower-bound", type=float, default=GATE_DEFAULTS["g5_min_lower_bound"],
                   help="G5: in every set, the bootstrap 95%% lower bound of the paired per-clip stored-minus-mirrored "
                        "read must exceed this")
    g.add_argument("--g6-min-r5", type=int, default=GATE_DEFAULTS["g6_min_r5"])
    g.add_argument("--g6-min-uncovered", type=int, default=GATE_DEFAULTS["g6_min_uncovered"])
    g.add_argument("--g6-forms", choices=("both", "real", "decoded"), default=GATE_DEFAULTS["g6_forms"],
                   help="G6: a word counts when reliable on real and decoded clips (both), or on one form")
    g.add_argument("--g7-min-top1", type=float, default=GATE_DEFAULTS["g7_min_top1"])


def parse_corpus_minima(items) -> dict:
    """``--g4-corpus-min CORPUS=MIN`` (repeated) -> {corpus: min}; None gives the default (Sem-Lex 0.5)."""
    if not items:
        return dict(GATE_DEFAULTS["g4_corpus_min"])
    out = {}
    for item in items:
        corpus, _, value = item.partition("=")
        if not corpus or not value:
            raise SystemExit(f"--g4-corpus-min {item!r}: expected CORPUS=MIN")
        out[corpus] = float(value)
    return out


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

    classes = q["classes"]
    above = []
    for n, s in sorted(q["per_set"].items()):
        bound1 = max(s["majority_top1"], t["g1_top1_floor_classes"] / classes)
        bound5 = max(s["majority_top5"], t["g1_top5_floor_classes"] / classes)
        for form in ("blank", "random_code"):
            v1, v5 = s[f"{form}_top1"], s[f"{form}_top5"]
            if v1 is None or v5 is None or v1 > bound1 or v5 > bound5:
                above.append(f"{n} {form} {v1}/{v5} > {round(bound1, 4)}/{round(bound5, 4)}")
    worst = max(((s[f"{f}_top1"] or 0.) for s in q["per_set"].values() for f in ("blank", "random_code")), default=None)
    gate("G1", f"highest blank/random-code top-1 {worst}" + (f"; above chance: {above}" if above else ""),
         f"per set, top-1 <= max(majority share, {t['g1_top1_floor_classes']}/{classes}) and top-5 <= "
         f"max(top-5-class share, {t['g1_top5_floor_classes']}/{classes})", bool(q["per_set"]) and not above)
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
    m4 = t["g4_metric"]
    cov, unc = q["decoded_words"]["covered"], q["decoded_words"]["uncovered"]
    corpus_reads = {c: (q["decoded_by_corpus"].get(c) or {}).get(m4) for c in t["g4_corpus_min"]}
    corpus_short = sorted(c for c, v in corpus_reads.items() if v is None or v < t["g4_corpus_min"][c])
    gate("G4", f"decoded {m4}: covered {cov[m4]} ({cov['clips']} clips), uncovered {unc[m4]} ({unc['clips']} clips); "
               f"per corpus {corpus_reads}",
         f"covered >= {t['g4_covered_min']}, uncovered >= {t['g4_uncovered_min']}, per corpus >= {t['g4_corpus_min']}",
         cov[m4] is not None and unc[m4] is not None and cov[m4] >= t["g4_covered_min"]
         and unc[m4] >= t["g4_uncovered_min"] and not corpus_short)
    m5 = t["g5_metric"]
    lows = {n: (s.get(f"real_minus_mirrored_{m5}_ci") or [None])[0] for n, s in q["per_set"].items()}
    bad = sorted(n for n, lo in lows.items() if lo is None or not lo > t["g5_min_lower_bound"])
    gate("G5", f"stored - mirrored {m5}, smallest 95% lower bound "
               f"{min((v for v in lows.values() if v is not None), default=None)}" + (f"; not above: {bad}" if bad else ""),
         f"every set's paired stored - mirrored {m5}: 95% lower bound > {t['g5_min_lower_bound']}",
         bool(lows) and not bad)
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
    args.g4_corpus_min = parse_corpus_minima(args.g4_corpus_min)
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
    names = sorted({r["partition"] for r in records if r["partition"].startswith(GATE_SETS)})
    if not any(n.startswith("gate:") for n in names):
        raise SystemExit("the run's data have no gate halves (select/gate split): not a recogniser v3 data definition")

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
        frequency = sorted(collections.Counter(y).values(), reverse=True)
        row["majority_top1"] = round(frequency[0] / len(y), 6)          # G1: a constant guess's top-1 / top-5
        row["majority_top5"] = round(sum(frequency[:5]) / len(y), 6)
        for k in (1, 5):                                                # G5: paired per clip, bootstrap over clips
            diff = (forms["real"] < k).astype(np.float64) - (forms["mirrored"] < k)
            row[f"real_minus_mirrored_top{k}"] = round(float(diff.mean()), 4)
            row[f"real_minus_mirrored_top{k}_ci"] = bootstrap(diff)
        corpus = name.split(":", 1)[1]
        row["corpus"] = corpus
        per_set[name] = row
        pooled["y"] += y
        pooled["real"] += forms["real"].tolist()
        pooled["y_dec"] += y_dec
        pooled["decoded"] += forms["decoded"].tolist()
        pooled[f"y_dec:{corpus}"] += y_dec
        pooled[f"decoded:{corpus}"] += forms["decoded"].tolist()
        for f in ("blank", "random_code", "no_hands"):
            pooled[f] += forms[f].tolist()
        print(json.dumps({name: {k: v for k, v in row.items() if k != "forms"}}), flush=True)
    y_all, y_dec = np.asarray(pooled["y"]), np.asarray(pooled["y_dec"])
    controls = dict(clips=len(y_all), chance_top1=round(1 / len(glosses), 6), chance_top5=round(5 / len(glosses), 6),
                    blank_top1=read(pooled["blank"], 1), blank_top5=read(pooled["blank"], 5),
                    random_code_top1=read(pooled["random_code"], 1), random_code_top5=read(pooled["random_code"], 5),
                    no_hands_top1=read(pooled["no_hands"], 1), no_hands_top5=read(pooled["no_hands"], 5))
    def word_reads(yd, rd):
        yd, rd, out = np.asarray(yd), np.asarray(rd), {}
        for subset, words in (("covered", covered), ("uncovered", uncovered)):
            mask = np.isin(yd, [index[w] for w in words]) if len(yd) else np.zeros(0, bool)
            r = rd[mask] if len(rd) else rd
            out[subset] = dict(clips=int(mask.sum()), words_read=int(len(set(yd[mask].tolist()))) if len(yd) else 0,
                               words=len(words), top1=read(r, 1), top5=read(r, 5))
        return out
    decoded_words = word_reads(y_dec, pooled["decoded"])
    corpora_read = sorted({s["corpus"] for s in per_set.values()})
    decoded_by_corpus = {c: dict(clips=len(pooled[f"decoded:{c}"]), top1=read(pooled[f"decoded:{c}"], 1),
                                 top5=read(pooled[f"decoded:{c}"], 5),
                                 words=word_reads(pooled[f"y_dec:{c}"], pooled[f"decoded:{c}"]))
                         for c in corpora_read}
    rule = dict(threshold=args.reliable, min_clips=args.min_clips)
    real_ok = reliable_classes(y_all, pooled["real"], **rule)
    dec_ok = reliable_classes(y_dec, pooled["decoded"], **rule)
    counted = {"both": real_ok & dec_ok, "real": real_ok, "decoded": dec_ok}[args.g6_forms]
    r5 = {index[w] for w in covered | uncovered}
    reliability = dict(rule=f"top-1 >= {args.reliable} over >= {args.min_clips} clips, pooled over every gate-half "
                            f"and holdout set", forms=args.g6_forms, r5_words=len(r5), uncovered_words=len(uncovered),
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
    quantities = dict(classes=len(glosses), controls=controls, per_set=per_set, decoded_words=decoded_words,
                      decoded_by_corpus=decoded_by_corpus, reliability=reliability, in_context_top1=in_context["top1"],
                      val_split=summary.get("val_split"))
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


# ------------------------------------------------------------------ recogniser v3 amendment 2: G6 calibrated on v2

G6_PRIME = ("G6': v3's R5 words reliable on both real and decoded clips >= v2's R5 words reliable on real clips, "
            "overall and among the uncovered words (docs/NEXT_PHASE_PLAN.md, recogniser v3 amendment 2, "
            "2026-10-29 13:00 UTC)")


def gate_rule(text: str) -> tuple[float, int]:
    """(threshold, min clips) of a gate report's G6 rule text (``top-1 >= 0.8 over >= 3 clips, ...``)."""
    import re
    match = re.match(r"top-1 >= ([0-9.]+) over >= (\d+) clips", text or "")
    if not match:
        raise SystemExit(f"cannot read the gate report's G6 rule {text!r}")
    return float(match[1]), int(match[2])


def same_read(hits: int, recorded_top1: float, clips: int, tolerance: int = 1) -> bool:
    """Whether ``hits`` top-1 hits over ``clips`` is the recorded read (rounded to 4 decimals) within ``tolerance``
    clips: a GPU and a CPU run may break a near-tie differently."""
    return abs(hits - recorded_top1 * clips) <= tolerance + 5e-5 * clips


def g6_prime(v3: dict, v2: dict) -> dict:
    """The amendment-2 verdict from v3's gate-report reliability (both forms) and v2's real-clip counts."""
    overall = v3["r5"] >= v2["r5_real"]
    uncovered = v3["uncovered"] >= v2["uncovered_real"]
    return dict(rule=G6_PRIME, v3_r5_both=v3["r5"], v2_r5_real=v2["r5_real"], v3_uncovered_both=v3["uncovered"],
                v2_uncovered_real=v2["uncovered_real"], overall_holds=overall, uncovered_holds=uncovered,
                passed=overall and uncovered)


def g6_calibration_main(args) -> int:
    """Amendment 2: ``--baseline-run`` (v2) read on exactly the clips v3's gate report read for G6 (the run's gate
    halves and held-out sets, its recorded validation split, label space and conform template), real clips only, by
    G6's rule (top-1 >= ``--reliable`` over >= ``--min-clips`` clips, pooled); then G6' against the stored gate report.
    Every set's clip count must equal the gate report's (the same records), and the run's data definition must give
    its recorded data SHA-256 and validation split. Written once."""
    import argparse
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.sign_recognizer import GlossMotionDataset
    import train_sign_recognizer as trainer
    store = make_store(args.store)
    prefix = f"training-runs/{args.run}"
    output_key = args.output_key or f"{prefix}/g6-calibration.json"
    if store.exists(output_key):
        raise SystemExit(f"{output_key} exists: the calibration is written once")
    gates_key = args.gates_key or f"{prefix}/gates.json"
    gates_payload = store.get(gates_key)
    gates = json.loads(gates_payload)
    if gates.get("run") != args.run or gates["reliability"].get("forms") != "both":
        raise SystemExit(f"{gates_key} is not {args.run}'s gate report with G6 read on both forms")
    if gates["baseline"]["run"] != args.baseline_run:
        raise SystemExit(f"the gate report's baseline is {gates['baseline']['run']}, not {args.baseline_run}")
    rule = gate_rule(gates["reliability"]["rule"])
    if rule != (args.reliable, args.min_clips):
        raise SystemExit(f"--reliable {args.reliable} / --min-clips {args.min_clips} differ from the gate report's G6 "
                         f"rule {rule}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    cache = ContentCache(Path(args.cache_dir or tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    run = json.loads(store.get(f"{prefix}/run.json"))
    if not run.get("recognizer_v3"):
        raise SystemExit(f"{args.run} is not a recogniser v3 run")
    targs = argparse.Namespace(**{**trainer.V3_DEFAULTS, **run["args"]})
    records, labels, summary, template, digest = trainer.build_v3_data(store, targs, cache, prefix)
    if digest != run["data_sha256"] or digest != gates["data_sha256"]:
        raise SystemExit("the run's data definition no longer gives the data its gates read")
    if summary["val_split"] != run["data"]["val_split"] or summary["val_split"] != gates.get("val_split"):
        raise SystemExit("the validation split differs from the run's recorded split")
    baseline, base_glosses, base_best = load_model(store, args.baseline_run, device)
    if base_glosses != labels:
        raise SystemExit("the baseline's labels differ from the run's label space")
    index = {g: i for i, g in enumerate(labels)}
    split_payload = store.get(gates["word_split"]["key"])
    if hashlib.sha256(split_payload).hexdigest() != gates["word_split"]["sha256"]:
        raise SystemExit("the word split differs from the one the gates read")
    split = json.loads(split_payload)
    r5 = {index[w] for w in set(split["covered"]) | set(split["uncovered"])}
    uncovered = {index[w] for w in split["uncovered"]}
    kw = dict(labels=index, cache=cache, fetch=store.get, min_hand_fraction=targs.min_hand_fraction,
              fps=targs.fps or None, template=template, hand_fraction_fallback=True)
    manifest = dict(records=records)
    names = sorted({r["partition"] for r in records if r["partition"].startswith(GATE_SETS)})
    if set(names) != set(gates["per_set"]):
        raise SystemExit(f"the sets {names} are not the gate report's {sorted(gates['per_set'])}")
    per_set, pooled_y, pooled_ranks = {}, [], []
    for name in names:
        dataset = GlossMotionDataset(manifest, partition=name, **kw)
        ids = [i for i, r in enumerate(dataset.records) if r["continuous"].get("seconds", 0.) <= targs.max_seconds]
        if len(ids) != gates["per_set"][name]["clips"]:
            raise SystemExit(f"{name}: {len(ids)} clips, the gate report read {gates['per_set'][name]['clips']} "
                             "(a gate run with --max-clips cannot be calibrated)")
        y = [index[dataset.records[i]["gloss"]["label"]] for i in ids]
        with ThreadPoolExecutor(16) as pool:
            states = [item[1] for item in pool.map(dataset.__getitem__, ids)]
        ranks = ranks_from_logits(logits_of(baseline, states, device), y)
        del states
        reported = gates["per_set"][name].get("v2_real_top1")
        hits = int((np.asarray(ranks) < 1).sum())
        if reported is None or not same_read(hits, reported, len(ids)):
            raise SystemExit(f"{name}: v2 reads top-1 {hits}/{len(ids)} here, the gate report {reported} (more than "
                             "one clip apart: not the same read)")
        per_set[name] = dict(clips=len(ids), v2_real_top1=read(ranks, 1), v2_real_top5=read(ranks, 5),
                             gate_report_v2_real_top1=reported,
                             v2_top1_hits=hits, gate_report_v2_top1_hits=int(round(reported * len(ids))))
        pooled_y += y
        pooled_ranks += ranks.tolist()
        print(json.dumps({name: per_set[name]}), flush=True)
    reliable = reliable_classes(pooled_y, pooled_ranks, threshold=args.reliable, min_clips=args.min_clips)
    v2 = dict(run=args.baseline_run, checkpoint=base_best["checkpoint_key"], forms="real",
              rule=f"top-1 >= {args.reliable} over >= {args.min_clips} clips, pooled over every gate-half and holdout "
                   "set", clips=len(pooled_y), r5_words=len(r5), uncovered_words=len(uncovered),
              r5_real=len(reliable & r5), uncovered_real=len(reliable & uncovered), reliable_real=len(reliable),
              reliable_r5_words=sorted(labels[i] for i in reliable & r5),
              reliable_uncovered_words=sorted(labels[i] for i in reliable & uncovered),
              reliable_glosses_real=sorted(labels[i] for i in reliable))
    verdict = g6_prime(gates["reliability"], v2)
    others = [g for g in gates["gates"] if g["gate"] != "G6"]
    report = dict(run=args.run, declared=G6_PRIME, gates=dict(key=gates_key,
                                                              sha256=hashlib.sha256(gates_payload).hexdigest()),
                  data_sha256=digest, val_split=summary["val_split"], word_split=gates["word_split"],
                  v2=v2, v3=dict(r5_both=gates["reliability"]["r5"], uncovered_both=gates["reliability"]["uncovered"],
                                 r5_real=gates["reliability"].get("r5_real"),
                                 r5_decoded=gates["reliability"].get("r5_decoded")),
                  per_set=per_set, g6_prime=verdict, other_gates_pass=all(g["passed"] for g in others),
                  verdict=("v3 usable as the instrument (amendment 2)" if verdict["passed"]
                           and all(g["passed"] for g in others) else "v2 stays"))
    payload = json.dumps(report, indent=1, sort_keys=True, default=float).encode()
    if args.output:
        Path(args.output).write_bytes(payload)
    store.put(output_key, payload)
    print(json.dumps(dict(g6_prime=verdict, verdict=report["verdict"], key=output_key,
                          sha256=hashlib.sha256(payload).hexdigest())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
