"""Strict pass test of the full-corpus prior, on top of the exit gate (rules declared before results:
docs/NEXT_PHASE_PLAN.md, "Strict pass test of the full-corpus prior").

The gate (``exit_gate.py``) selects a setting from validation evaluations on realism, articulation and marker rates.
This review then checks the selected setting against six rules. Every rule must pass:

- **R1 meaning (retrieval):** per-clip R@3 hits through the frozen retrieval evaluator, in the evaluator's own fixed
  groups of 32. The bootstrap 95% interval of generated must lie entirely above the blank (duration-only) interval.
  Real is reported as the ceiling.
- **R2 meaning (back-translation):** corpus BLEU-4 of the frozen back-translator's output. Same rule: the generated
  bootstrap interval must lie entirely above blank, by resampling sentences.
- **R2b meaning, second instrument (round 3):** with ``--second-translator-key``, R2's rule through the accepted
  2D keypoint back-translator, which reads states through the FK projection.
- **R3 no copying:** for every generated clip, in the Fréchet feature space (``motion_metrics.clip_features``,
  standardised by the training clips, constant columns dropped), the ratio of its nearest training-clip distance to
  its nearest distance among an equal number of unseen real validation clips, its own reference excluded. A copy is
  far closer to training (ratio < 0.5); an averaged clip is about equally close to both. At most 5% of generated clips
  may be flagged. The same ratio for the real clips is reported.
- **R4 robustness:** the clips are split into sentence-length thirds. In every third, the body, hand and head speed
  and variation ratios must lie in the gate's articulation band [0.7, 1.3].
- **R5 sign level:** the prior is prompted with single English words, the ASL Citizen labels of the validated
  recogniser's reliable glosses that occur as words in the How2Sign training sentences. The recogniser's top-5 on the
  generated signs needs a bootstrap interval entirely above blank. Real clips of those glosses are reported as the
  ceiling.
- **Post-processing:** the selected candidate's head smoothing and face calibration (estimated on validation, stored
  with the gate's candidate) are applied in every rule, including the sign-level and held-out checks.
- **R6 held-out confirmation:** the selected setting is evaluated once on the test partition (signer-disjoint, 92
  clips), and every gate check that passed on validation must pass there too (round 3b, ``--envelope-key``: within
  the test signer's real cross-signer envelope). The ``heldout_official`` clips (up to
  600; signer overlap unverified) are evaluated and reported.

The report (``--output-key``) gives every number, each rule's pass and the overall verdict. A failure is a failure of
task No.1, whatever the gate said.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_metrics import clip_features, compare  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from evaluate_text_to_motion import blanked, load_generator, load_retrieval, load_state  # noqa: E402
from exit_gate import ARTICULATION, ARTICULATED, score_setting  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402

BOOT = 1000


def bootstrap_mean(hits: np.ndarray, *, seed: int = 0, n: int = BOOT) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def separated(gen_ci, blank_ci) -> bool:
    return gen_ci[0] > blank_ci[1]


def per_clip_hits(zm: torch.Tensor, zt: torch.Tensor, *, group: int = 32, k: int = 3, seed: int = 0) -> np.ndarray:
    """Per-row hit@k in ``text_motion_retrieval.r_precision``'s fixed random groups (same permutation and seed)."""
    n = len(zm)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    hits = []
    for start in range(0, n - group + 1, group):
        idx = order[start:start + group]
        sim = zm[idx] @ zt[idx].t()
        rank = (sim > sim.diagonal()[:, None]).sum(1)
        hits += (rank < k).tolist()
    return np.asarray(hits, dtype=float)


@torch.no_grad()
def embeddings(model, states, texts, batch: int = 32):
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import encode_texts
    zm, zt = [], []
    for s in range(0, len(states), batch):
        zm.append(model.encode_motion(collate_multichannel(states[s:s + batch])))
        tokens, mask = encode_texts(texts[s:s + batch])
        zt.append(model.encode_text(tokens, mask))
    return torch.cat(zm), torch.cat(zt)


def bleu_ci(hyps, refs, *, seed: int = 0, n: int = BOOT):
    from signtranslator.generation.motion_to_text import corpus_bleu
    rng = np.random.default_rng(seed)
    point = corpus_bleu(hyps, refs)["bleu"]
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(hyps), len(hyps))
        vals.append(corpus_bleu([hyps[i] for i in idx], [refs[i] for i in idx])["bleu"])
    return round(float(point), 3), [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def load_saved(store, prefix: str):
    out = {}
    for key in sorted(k for k in store.list(prefix.rstrip("/") + "/") if k.endswith(".npz")):
        payload = store.get(key)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.npz"; p.write_bytes(payload)
            out[key.rsplit("/", 1)[-1][:-4]] = MultichannelMotion.load(p, expected_sha256=hashlib.sha256(payload).hexdigest())
    return out


def selected_setting(gate: dict) -> dict:
    """Run, guidance tag, head smoothing and saved prefix of the gate's selected setting."""
    run, setting = gate["selected"]["run"], gate["selected"]["setting"]
    m = re.search(r"guidance ([0-9.]+)", setting)
    guidance = float(m.group(1)) if m else 1.
    fc = run.endswith("-fc")
    core = run[:-3] if fc else run
    hd = re.search(r"-hd([0-9.]+)$", core)                 # round 3: head damping after smoothing
    core = core[:hd.start()] if hd else core
    hs = re.search(r"-hs([0-9.]+)$", core)
    base = core[:hs.start()] if hs else core
    t = re.search(r"-sweep-h([0-9.]+)-f([0-9.]+)", base)
    temperature = (f"body={t.group(1)},left_hand={t.group(1)},right_hand={t.group(1)},face={t.group(2)}" if t
                   else "body=1.2,left_hand=1.2,right_hand=1.2,face=1.0")          # the core chain's evaluation
    return dict(run=run, setting=setting, guidance=guidance, head_smoothing=float(hs.group(1)) if hs else 0.,
                head_damping=float(hd.group(1)) if hd else 1., generated_run=base, tag=f"guidance-{guidance:g}", part_temperature=temperature, face_calibrated=fc)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--derived", required=True)
    parser.add_argument("--gate-key", help="exit_gate.py report in the store")
    parser.add_argument("--gate-file", type=Path, help="... or a local copy")
    parser.add_argument("--retrieval-manifest-sha256", help="when the evaluator was trained on another manifest")
    parser.add_argument("--translator-manifest-sha256", help="when the back-translator was trained on another manifest")
    parser.add_argument("--quick", action="store_true", help="pipeline dry run: few clips per rule; not a verdict")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", help="when the generator was trained on another manifest")
    parser.add_argument("--retrieval-key", required=True)
    parser.add_argument("--translator-key", required=True)
    parser.add_argument("--envelope-key", help="round 3b: cross_signer_baseline.py output with --test-limit; R6 judges "
                                               "the test signer within its real cross-signer envelope")
    parser.add_argument("--second-translator-key", help="R2b: the accepted 2D keypoint back-translator (round 3)")
    parser.add_argument("--recognizer-run", default="sign-recognizer-v2")
    parser.add_argument("--recognizer-validation-key", default="training-runs/sign-recognizer-v2/validation.json")
    parser.add_argument("--citizen-manifest-key", default="acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json")
    parser.add_argument("--citizen-manifest-sha256", default="708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6")
    parser.add_argument("--train-limit", type=int, default=4000, help="training clips sampled for the copy check")
    parser.add_argument("--words", type=int, default=150)
    parser.add_argument("--output-key", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout-corpus", help="mixed manifests: R6 evaluates this corpus's test / heldout_official "
                                                 "records only (round 3: how2sign-two-view-text-v1)")
    parser.add_argument("--groups", help="round 3: comma-separated validation signer components; generated states are "
                                         "pooled from <generated_run>-<group>, the gate is multi_signer_gate.py's")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    by_sign = {r["sign"]: r for r in manifest["records"]}
    gate = json.loads(args.gate_file.read_text() if args.gate_file else store.get(args.gate_key))
    if args.quick:
        args.train_limit, args.words = 200, 10
    sel = selected_setting(gate)
    report = dict(selected=sel, rules={})
    # validation states of the selected setting, as the core chain saved them
    groups = args.groups.split(",") if args.groups else None
    if groups:                                       # every validation signer's clips, pooled
        generated = {}
        for g in groups:
            generated.update(load_saved(store, f"{args.derived}/generated/{sel['generated_run']}-{g}/{sel['tag']}"))
    else:
        generated = load_saved(store, f"{args.derived}/generated/{sel['generated_run']}/{sel['tag']}")
    if sel["head_smoothing"] > 0:
        from signtranslator.generation.postprocess import smooth_rotation_channel
        generated = {k: smooth_rotation_channel(v, "head_rotation", sel["head_smoothing"]) for k, v in generated.items()}
    if sel.get("head_damping", 1.) != 1:
        from signtranslator.generation.postprocess import damp_rotation_channel
        generated = {k: damp_rotation_channel(v, "head_rotation", sel["head_damping"]) for k, v in generated.items()}
    calibration = None
    if sel["face_calibrated"]:                       # the calibration the gate's candidate used, from its stored result
        if groups:                                   # the pooled calibration, identical in every group's report
            stored = json.loads(store.get(f"{args.derived}/evaluations/{sel['run']}-{groups[0]}.json"))["results"]
            calibration = stored[sel["setting"]]["face_calibration"]
        else:
            stored = json.loads(store.get(f"{args.derived}/evaluations/{sel['run']}.json"))["results"]
            calibration = next(v["face_calibration"] for k, v in stored.items() if k.startswith(sel["tag"]) and "face_calibration" in v)
        from signtranslator.generation.postprocess import calibrate_face
        generated = {k: calibrate_face(v, calibration) for k, v in generated.items()}
    report["face_calibration"] = calibration
    signs = sorted(generated)[:64] if args.quick else sorted(generated)
    refs = [load_state(store, by_sign[s]["continuous"]["state_key"], by_sign[s]["continuous"]["state_sha256"]) for s in signs]
    gens = [generated[s] for s in signs]
    texts = [by_sign[s]["text"] for s in signs]
    # R1
    retrieval = load_retrieval(store, args.retrieval_key, args.retrieval_manifest_sha256 or args.manifest_sha256)
    r1 = {}
    for name, states in (("generated", gens), ("real", refs), ("blank", blanked(refs))):
        zm, zt = embeddings(retrieval, states, texts)
        hits = per_clip_hits(zm, zt)
        r1[name] = dict(r_at_3=round(float(hits.mean()), 4), ci95=bootstrap_mean(hits), clips=len(hits))
    r1["pass"] = separated(r1["generated"]["ci95"], r1["blank"]["ci95"])
    report["rules"]["R1_retrieval_above_blank"] = r1
    # R2
    from score_back_translation import load_translator, translate
    model, vocab = load_translator(store, args.translator_key, args.translator_manifest_sha256 or args.manifest_sha256)
    r2 = {}
    for name, states in (("generated", gens), ("real", refs), ("blank", blanked(refs))):
        point, ci = bleu_ci(translate(model, vocab, states), texts)
        r2[name] = dict(bleu=point, ci95=ci)
    r2["pass"] = separated(r2["generated"]["ci95"], r2["blank"]["ci95"])
    report["rules"]["R2_back_translation_above_blank"] = r2
    # R2b (round 3, declared 2026-10-23 17:35 PDT before any round-3 result): the same rule through the accepted,
    # more sensitive 2D back-translator (UWB-pretrained, How2Sign fine-tuned; FK projection), which put the full
    # prior's generations below blank (1.43 / 1.74 vs 1.95) where the 3D one had passed it
    if args.second_translator_key:
        model2, kind2 = load_translator(store, args.second_translator_key, args.manifest_sha256)
        if kind2 != "keypoints":
            raise SystemExit("--second-translator-key must be the 2D keypoint back-translator")
        model2 = model2.to(torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"))
        r2b = dict(translator_key=args.second_translator_key)
        for name, states in (("generated", gens), ("real", refs), ("blank", blanked(refs))):
            point, ci = bleu_ci(translate(model2, kind2, states), texts)
            r2b[name] = dict(bleu=point, ci95=ci)
        r2b["pass"] = separated(r2b["generated"]["ci95"], r2b["blank"]["ci95"])
        report["rules"]["R2b_2d_back_translation_above_blank"] = r2b
    # R3: copying shows as being much closer to a training clip than to unseen real clips; averaging does not
    rng = np.random.default_rng(0)
    train = [r for r in manifest["records"] if r["partition"] == "train" and r.get("text")]
    n_ref = min(len(refs), args.train_limit)
    train = [train[i] for i in rng.choice(len(train), min(n_ref, len(train)), replace=False)]   # same size as held-out set
    feats = lambda states: np.stack([np.nan_to_num(clip_features(s)) for s in states])  # noqa: E731
    tf = feats([load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]) for r in train])
    mu, sd = tf.mean(0), tf.std(0)
    keep = sd > 1e-3 * max(float(np.median(sd)), 1e-12)   # constant columns (unsupported channels) would explode
    z = lambda f: torch.tensor((f[:, keep] - mu[keep]) / sd[keep])  # noqa: E731
    T, R, G = z(tf), z(feats(refs)), z(feats(gens))

    def ratios(X, exclude_own: bool):
        to_train = torch.cdist(X, T).min(1).values
        D = torch.cdist(X, R)
        if exclude_own:                                    # a clip is never compared with its own reference
            D[torch.arange(len(X)), torch.arange(len(X))] = float("inf")
        return (to_train / D.min(1).values).numpy()
    g_ratio, r_ratio = ratios(G, True), ratios(R, True)
    copy_rate = float((g_ratio < .5).mean())
    report["rules"]["R3_no_copying"] = dict(copy_rate=round(copy_rate, 4), threshold=.05, flag_ratio=.5,
                                            median_ratio_generated=round(float(np.median(g_ratio)), 3),
                                            median_ratio_real=round(float(np.median(r_ratio)), 3),
                                            real_flag_rate=round(float((r_ratio < .5).mean()), 4),
                                            train_clips=len(train), heldout_clips=len(refs),
                                            feature_columns=int(keep.sum()), **{"pass": copy_rate <= .05})
    # R4
    lengths = np.array([len(t.split()) for t in texts])
    cuts = np.percentile(lengths, [100 / 3, 200 / 3])
    slices, ok = {}, True
    for name, mask in (("short", lengths <= cuts[0]), ("medium", (lengths > cuts[0]) & (lengths <= cuts[1])),
                       ("long", lengths > cuts[1])):
        idx = np.nonzero(mask)[0]
        res = compare([gens[i] for i in idx], [refs[i] for i in idx], ARTICULATED)
        ratios = {f"{c}_{r}": round(float(res[c][r]), 3) for c in ARTICULATED if c in res
                  for r in ("speed_ratio", "variation_ratio")}
        within = all(ARTICULATION[0] <= v <= ARTICULATION[1] for v in ratios.values())
        ok &= within
        slices[name] = dict(clips=len(idx), ratios=ratios, **{"pass": within})
    report["rules"]["R4_length_slices"] = dict(slices=slices, band=list(ARTICULATION), **{"pass": ok})
    # R5
    from signtranslator.generation.sign_recognizer import top_k_accuracy
    from signtranslator.pose.resample import resample_motion
    from validate_sign_recognizer import load_model, logits_of
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_sample
    validation = json.loads(store.get(args.recognizer_validation_key))
    if not validation["verdict"]["usable_as_instrument"]:
        raise SystemExit("the recogniser failed its validation")
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}
    vocab_words = {w for r in train for w in re.findall(r"[a-z]+", r["text"].lower())}
    words = sorted(g for g in validation["per_gloss"]["reliable_glosses"] if g.isalpha() and g.lower() in vocab_words)
    words = words[:args.words]
    temperature = {k: float(v) for k, v in (x.split("=") for x in sel["part_temperature"].split(","))}
    _, _, sample = load_generator(store, args.checkpoint_key, args.checkpoint_manifest_sha256 or args.manifest_sha256,
                                  args.device, temperature=temperature)
    sign_states, y = [], []
    for k, g in enumerate(words):
        for s in range(2):
            st = resample_motion(sample(g.lower(), 24., 1.2, sel["guidance"], 50, 7000 + 10 * k + s), 24.)
            if sel["head_smoothing"] > 0:
                from signtranslator.generation.postprocess import smooth_rotation_channel
                st = smooth_rotation_channel(st, "head_rotation", sel["head_smoothing"])
            if sel.get("head_damping", 1.) != 1:
                from signtranslator.generation.postprocess import damp_rotation_channel
                st = damp_rotation_channel(st, "head_rotation", sel["head_damping"])
            sign_states.append(calibrate_face(st, calibration) if calibration else st)
            y.append(index[g])
    y = torch.tensor(y)

    def top5_hits(states, targets):
        logits = logits_of(recogniser, states, args.device)
        return (logits.topk(5, dim=1).indices == targets[:, None]).any(1).float().numpy()
    gen_hits, blank_hits = top5_hits(sign_states, y), top5_hits(blanked(sign_states), y)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "c.json"; p.write_bytes(store.get(args.citizen_manifest_key) + b"\n")
        citizen = load_manifest(p, expected_sha256=args.citizen_manifest_sha256)
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    real = [r for r in citizen["records"] if r.get("partition") == "val" and r["gloss"]["label"] in set(words)]
    real_states = [resample_motion(load_sample(r, "continuous", cache=cache, fetch=store.get), 24.) for r in real]
    real_hits = top5_hits(real_states, torch.tensor([index[r["gloss"]["label"]] for r in real])) if real else np.zeros(1)
    r5 = dict(words=len(words), samples=len(y), generated=dict(top5=round(float(gen_hits.mean()), 4), ci95=bootstrap_mean(gen_hits)),
              blank=dict(top5=round(float(blank_hits.mean()), 4), ci95=bootstrap_mean(blank_hits)),
              real=dict(clips=len(real), top5=round(float(real_hits.mean()), 4)))
    r5["pass"] = separated(r5["generated"]["ci95"], r5["blank"]["ci95"])
    report["rules"]["R5_sign_level_words"] = r5
    # R6
    import evaluate_text_to_motion as evaluate
    held = {}
    for partition, limit in (("test", 92), ("heldout_official", 600)):
        limit = 32 if args.quick else limit
        key = f"{args.derived}/evaluations/strict-{sel['run']}-{partition}{'-dryrun' if args.quick else ''}.json"
        extra = []
        if calibration:                                  # the validation calibration, applied unchanged on held-out data
            cal_key = f"{args.derived}/evaluations/strict-{sel['run']}-face-calibration.json"
            store.put(cal_key, json.dumps(calibration).encode())
            extra = ["--face-calibration-key", cal_key]
        try:
            evaluate.main(["--store", args.store,     "--manifest-key", args.manifest_key, "--manifest-sha256", args.manifest_sha256,
                           "--checkpoint-key", args.checkpoint_key, "--partition", partition, "--limit", str(limit),
                           *(["--checkpoint-manifest-sha256", args.checkpoint_manifest_sha256]
                             if args.checkpoint_manifest_sha256 else []),
                           "--guidance", f"{sel['guidance']:g}", "--part-temperature", sel["part_temperature"],
                           "--head-smoothing", str(sel["head_smoothing"]), "--head-damping", str(sel.get("head_damping", 1.)),
                           "--device", args.device,
                           "--retrieval-key", args.retrieval_key, "--output-key", key, "--fps", "24"]
                          + (["--corpus", args.heldout_corpus] if args.heldout_corpus else [])
                          + (["--retrieval-manifest-sha256", args.retrieval_manifest_sha256] if args.retrieval_manifest_sha256 else [])
                          + extra)
        except SystemExit as stop:                          # a partition the manifest does not have
            held[partition] = dict(absent=str(stop))
            continue
        result = next(iter(json.loads(store.get(key))["results"].values()))
        held[partition] = score_setting(sel["run"], partition, result, ARTICULATED)
    row = next(r for r in gate["rows"] if r["run"] == sel["run"] and r["setting"] == sel["setting"])
    if "per_group" in row:                           # multi-signer gate: a check counts if it passed on every signer
        val_passed = set.intersection(*({k for k, (_, ok) in g["checks"].items() if ok} for g in row["per_group"].values()))
    else:
        val_passed = {k for k, (_, ok) in row["checks"].items() if ok}
    if args.envelope_key and "checks" in held["test"]:   # round 3b: the test signer's real cross-signer envelope
        from signtranslator.generation.envelope import apply, bands
        test_band = bands(json.loads(store.get(args.envelope_key))).get("test")
        if test_band is None:
            raise SystemExit("the envelope has no test band (cross_signer_baseline.py --test-limit)")
        held["test"]["checks"] = apply(held["test"]["checks"], test_band)
        held["test"]["envelope"] = test_band
    test_checks = held["test"].get("checks", {})
    regressions = sorted(k for k in val_passed if not test_checks.get(k, (None, False))[1])   # an absent test fails
    report["rules"]["R6_test_confirmation"] = dict(test=held["test"], heldout_official=held["heldout_official"],
                                                    regressions=regressions, **{"pass": not regressions})
    report["pass"] = all(r["pass"] for r in report["rules"].values()) and not args.quick
    report["quick_dry_run"] = args.quick
    store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True, default=float).encode())
    print(json.dumps({k: v["pass"] for k, v in report["rules"].items()} | {"pass": report["pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
