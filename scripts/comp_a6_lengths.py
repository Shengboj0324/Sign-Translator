"""Post-S2 A6's length fit: a better text-only sentence length than the one predicted timing uses (1.198x true on D20, its fit mostly one signer). Metadata only: nothing trains, no state and no reader.

Candidates, every one fitted on How2Sign train and never on D20's lengths (c00 refused, not skipped):

- ``tokens_all``: today's fit, all train records, seconds = a + b x tokens.
- ``tokens_balanced``: the same line on c02 + c04, each signer with equal total weight.
- ``words_balanced``: c02 + c04 balanced, seconds = a + b x (the typical seconds of the top-10 arm's plan entries) + c x (tokens the plan leaves unsigned).

``fit`` chooses on train alone, by a rule fixed here before any candidate's D20 ratio was computed for the words model: five folds of the balanced records split by source video, lowest
mean over signers of the median absolute relative per-clip error, ties (within 0.005) to fewer parameters. A leave-one-signer-out ratio is reported beside it and does not choose.
``d20`` then reads each candidate's pooled and per-clip ratio on D20's metadata, for reporting only: it cannot change the choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
H2S = "how2sign-two-view-text-v1"
BALANCED = ("c02", "c04")
TOKENS_FILE = ROOT / "reports/comp_a3_tokens.json"
TOKENS_SHA = "c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776"
FOLDS = 5
TIE = .005
PARAMETERS = dict(tokens_all=2, tokens_balanced=2, words_balanced=3)
FLOOR = .5


def train_records(manifest) -> list[dict]:
    """How2Sign train records with a text and a length. c00 stops it: that signer is never read."""
    out = []
    for r in manifest["records"]:
        if r.get("corpus") != H2S or r.get("partition") != "train":
            continue
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")
        if r.get("text") and (r.get("continuous") or {}).get("seconds"):
            out.append(r)
    return out


def source_of(record: dict) -> str:
    """A record's source video (How2Sign's ``VIDEO_n-m-rgb_front``): a sentence of one video never sits on both sides of a fold."""
    return record["sign"].removesuffix("-rgb_front").rsplit("_", 1)[0]


def plan_features(text_rows, details, words, vocab) -> np.ndarray:
    """(n, 2): the typical seconds of each sentence's plan entries, and the tokens its plan leaves unsigned."""
    from signtranslator.generation.compositional import KEPT
    out = np.zeros((len(words), 2))
    for i, (rows, ws) in enumerate(zip(details, words)):
        out[i, 0] = sum(float(vocab[w]) for w in ws)
        out[i, 1] = sum(len(r["tokens"]) for r in rows if r["kind"] not in KEPT)
    return out


def balance(group) -> np.ndarray:
    """Weights on the residual giving every signer equal total weight."""
    names, inverse, counts = np.unique(np.asarray(group, dtype=str), return_inverse=True, return_counts=True)
    if len(names) < 2:
        raise ValueError("a signer-balanced fit needs at least two signer components")
    return np.sqrt(1. / counts[inverse])


def fit_line(x: np.ndarray, y: np.ndarray, weights=None) -> np.ndarray:
    """Least squares coefficients of y on an intercept and the columns of x."""
    design = np.column_stack([np.ones(len(x)), x])
    w = np.ones(len(y)) if weights is None else weights
    return np.linalg.lstsq(design * w[:, None], y * w, rcond=None)[0]


def apply_line(coefficients, x: np.ndarray) -> np.ndarray:
    return np.maximum(np.column_stack([np.ones(len(x)), x]) @ np.asarray(coefficients), FLOOR)


def errors(predicted: np.ndarray, true: np.ndarray) -> dict:
    ratio = predicted / true
    return dict(pooled_ratio=round(float(predicted.sum() / true.sum()), 4), median_ratio=round(float(np.median(ratio)), 4),
                mean_ratio=round(float(ratio.mean()), 4), median_abs_relative_error=round(float(np.median(np.abs(ratio - 1))), 4),
                median_abs_seconds=round(float(np.median(np.abs(predicted - true))), 3), clips=len(true))


def folds_of(sources, k: int = FOLDS) -> np.ndarray:
    """A fold per record, fixed by the hash of its source video."""
    return np.array([int(hashlib.sha256(f"comp-a6-lengths|{s}".encode()).hexdigest(), 16) % k for s in sources])


def cross_validated(x, y, group, sources, *, balanced: bool) -> dict:
    """Out-of-fold errors per signer on the balanced records, and the mean of the signers' median errors (the choosing number)."""
    fold = folds_of(sources)
    predicted = np.zeros(len(y))
    for k in range(FOLDS):
        out = fold == k
        predicted[out] = apply_line(fit_line(x[~out], y[~out], balance(group[~out]) if balanced else None), x[out])
    per = {str(g): errors(predicted[group == g], y[group == g]) for g in np.unique(group)}
    return dict(per_signer=per, choosing_error=round(float(np.mean([p["median_abs_relative_error"] for p in per.values()])), 4),
                mean_signer_pooled_ratio=round(float(np.mean([p["pooled_ratio"] for p in per.values()])), 4))


def leave_one_signer_out(x, y, group) -> dict:
    """Fit on the other signers, read on this one: how an unseen tempo lands. Reported; it does not choose."""
    return {str(g): errors(apply_line(fit_line(x[group != g], y[group != g]), x[group == g]), y[group == g]) for g in np.unique(group)}


def choose(rows: dict) -> str:
    """The lowest choosing error, and within ``TIE`` of it the candidate with the fewest parameters, then the first by name."""
    best = min(r["cross_validated"]["choosing_error"] for r in rows.values())
    near = [n for n, r in rows.items() if r["cross_validated"]["choosing_error"] <= best + TIE]
    return sorted(near, key=lambda n: (PARAMETERS[n], n))[0]


def model_row(name: str, coefficients, records: int, components=None) -> dict:
    c = [round(float(x), 4) for x in coefficients]
    if name.startswith("tokens"):
        return dict(kind="tokens", corpus=H2S, partition="train", records=records, intercept=c[0], slope=c[1], components=components, signer_balanced=name == "tokens_balanced", floor=FLOOR)
    return dict(kind="words", corpus=H2S, partition="train", records=records, intercept=c[0], entry_seconds=c[1], unsigned_token=c[2], components=components, signer_balanced=True, floor=FLOOR)


def fit(manifest) -> dict:
    """Every candidate on train, its cross-validation, and the choice."""
    from comp_a_lib import apply_token_map, arm_of, extra_vocabulary, load_token_list
    from round15_compose_eval import stage1_selection
    from signtranslator.generation.compositional import isolated_corpora, normalise, training_vocabulary
    records = train_records(manifest)
    vocab = training_vocabulary(manifest, isolated_corpora(manifest))
    arm = arm_of(load_token_list(TOKENS_FILE, TOKENS_SHA), 10)
    tokens = np.array([len(normalise(r["text"])) for r in records], float)[:, None]
    y = np.array([float(r["continuous"]["seconds"]) for r in records])
    group = np.array([str(r.get("signer_component")) for r in records])
    sources = np.array([source_of(r) for r in records])
    chosen = np.isin(group, BALANCED)
    texts = [r["text"] for r, keep in zip(records, chosen) if keep]
    details, _ = stage1_selection(texts, vocab)
    details, words = apply_token_map(details, arm["token_map"])
    plan = plan_features(texts, details, words, dict(vocab, **extra_vocabulary(arm)))
    rows = {}
    xs = dict(tokens_all=tokens[chosen], tokens_balanced=tokens[chosen], words_balanced=plan)
    for name, x in xs.items():
        balanced = name != "tokens_all"
        whole = fit_line(tokens if name == "tokens_all" else x, y if name == "tokens_all" else y[chosen], balance(group[chosen]) if balanced else None)
        components = None if name == "tokens_all" else list(BALANCED)
        rows[name] = dict(model=model_row(name, whole, len(y) if name == "tokens_all" else int(chosen.sum()), components),
                          cross_validated=cross_validated(x, y[chosen], group[chosen], sources[chosen], balanced=balanced),
                          leave_one_signer_out=leave_one_signer_out(x, y[chosen], group[chosen]))
    return dict(kind="comp-a6-lengths", report_only=True, inputs=dict(key=EVAL_MIX, sha256=EVAL_MIX_SHA, tokens_sha256=TOKENS_SHA), signers={str(g): int((group == g).sum()) for g in np.unique(group)},
                folds=FOLDS, tie=TIE, candidates=rows, chosen=choose(rows))


def read_d20(fitted: dict, manifest, val) -> dict:
    """Each candidate's ratios on D20's metadata. Reporting only: the choice is already in ``fitted``."""
    from comp_a_lib import apply_token_map, arm_of, extra_vocabulary, load_token_list
    from round15_compose_eval import stage1_selection
    from signtranslator.generation.compositional import isolated_corpora, normalise, training_vocabulary
    vocab = training_vocabulary(manifest, isolated_corpora(manifest))
    arm = arm_of(load_token_list(TOKENS_FILE, TOKENS_SHA), 10)
    texts = [r["text"] for r in val]
    details, _ = stage1_selection(texts, vocab)
    details, words = apply_token_map(details, arm["token_map"])
    plan = plan_features(texts, details, words, dict(vocab, **extra_vocabulary(arm)))
    tokens = np.array([len(normalise(t)) for t in texts], float)[:, None]
    true = np.array([float(r["continuous"]["seconds"]) for r in val])
    signers = np.array([str(r.get("signer_component")) for r in val])
    out = {}
    for name, row in fitted["candidates"].items():
        m = row["model"]
        predicted = apply_line([m["intercept"], m["slope"]], tokens) if m["kind"] == "tokens" else apply_line([m["intercept"], m["entry_seconds"], m["unsigned_token"]], plan)
        out[name] = dict(errors(predicted, true), per_signer={str(g): errors(predicted[signers == g], true[signers == g]) for g in np.unique(signers)})
    return dict(clips=len(val), signers={str(g): int((signers == g).sum()) for g in np.unique(signers)}, note="D20's lengths are read here and fit nothing", candidates=out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("fit", "d20"))
    p.add_argument("--store", default="r2")
    p.add_argument("--fit-file", required=True)
    p.add_argument("--output-file", help="d20: the file with both")
    args = p.parse_args(argv)
    from s2_lib import load_manifest_at
    from train_text_to_motion import make_store
    store = make_store(args.store)
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    if args.command == "fit":
        if Path(args.fit_file).exists():
            raise SystemExit(f"{args.fit_file} exists: refusing to overwrite")
        out = fit(manifest)
        Path(args.fit_file).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(json.dumps({n: (r["cross_validated"], r["model"]) for n, r in out["candidates"].items()}), "chosen:", out["chosen"], flush=True)
        return 0
    from comp_a_lib import D20_KEY, D20_SHA, check_d20_payload
    from round20_lib import load_sentence_set
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    fitted = json.loads(Path(args.fit_file).read_text())
    raw = store.get(D20_KEY)
    if hashlib.sha256(raw).hexdigest() != D20_SHA:
        raise SystemExit(f"{D20_KEY} differs from its pin")
    payload = json.loads(raw)
    check_d20_payload(payload)
    val = load_sentence_set(payload, manifest, EVAL_MIX, EVAL_MIX_SHA)
    fitted["d20"] = read_d20(fitted, manifest, val)
    Path(args.output_file).write_text(json.dumps(fitted, indent=1, sort_keys=True) + "\n")
    print(json.dumps(fitted["d20"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
