"""Round 15 stage 2: a timing check on ASLLRP val (development data), before any How2Sign run. Real isolated clips of
the stage-2 plan of each ASLLRP val utterance's English are composed with stage-2 timing (sign cores at the fitted
targets, gaps, the sentence fitted to the utterance's own duration) and compared with the real ASLLRP utterance:
wrist speed and the other back-translator-view motion statistics (``compositional_baseline.motion_rows``). Beside
them, the stage-1 oracle's timing on the same utterances (stage 1's selection; whole trimmed clips time-resampled to
the fitted durations, 4-frame crossfades) and the stage-2 words with each --total-fit.

Nothing is fitted here and no How2Sign sentence is read (the layout template is a How2Sign *train* record's layout,
as in training and in the oracle). Writes a JSON report (``--output-file``, write-once) and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
ASLLRP_CORPUS = "asllrp-dai2-plan-v2"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-file", required=True, help="the fitted stage-2 model (round15_stage2_fit.py)")
    p.add_argument("--mix-key", required=True)
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--utterances", type=int, default=200, help="ASLLRP val utterances (a seeded sample)")
    p.add_argument("--total-fit", default="pad,scale,none")
    p.add_argument("--core-threshold", type=float, default=.1)
    p.add_argument("--core-margin", type=float, default=0.)
    p.add_argument("--sign-margin", type=float, default=.1, help="the sign-fitted cores' margin (seconds)")
    p.add_argument("--store", default="r2")
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    import warnings
    warnings.simplefilter("ignore")
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.compositional import (CORE_RULE, concatenate, core_window, parts_from_pieces,
                                                         render, select_words, time_scale, training_vocabulary,
                                                         word_frames)
    from compositional_baseline import motion_rows
    from round15_compositional import plan_durations
    from round15_oracle import TEMPLATE_CORPUS, choose_record, word_clip, word_pools
    from round15_stage2 import frame_plan, plan_clip
    import round15_stage2_fit as fit
    from sentence_exposure_diagnostic import mean_statistics, ratio_to
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
    store = make_store(args.store)
    model = json.loads(Path(args.model_file).read_text())
    fit.check_model(model)
    if model["mix_key"] != args.mix_key or model["inputs"].get(args.mix_key) != args.mix_sha256:
        raise SystemExit("the model was fitted on another vocabulary")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    corpora = tuple(model["corpora"])
    vocab = training_vocabulary(manifest, corpora)
    pools = word_pools(manifest, corpora)
    cache = ContentCache(Path(args.cache_dir), 8 * 2 ** 30)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
    val = sorted((r for r in manifest["records"] if r.get("corpus") == ASLLRP_CORPUS and r.get("partition") == "val"
                  and r.get("english")), key=lambda r: r["sign"])
    val = random.Random(0).sample(val, min(args.utterances, len(val)))
    fits = [f for f in args.total_fit.split(",") if f]
    loaded = {}

    def clip(r):
        if r["sign"] not in loaded:
            loaded[r["sign"]] = word_clip(r, cache=cache, fetch=store.get, template=template)
        return loaded[r["sign"]]

    from round15_stage2 import oracle_material, render_form, stage1_durations
    fractions = fit.sign_fractions(manifest, vocab)
    forms = {name: [] for name in ["real", "stage1_oracle"] + [f"stage2_oracle@{f}" for f in fits]
             + ["stage2_words@stage1_timing", "stage2_oracle@sign_fitted"]}
    sign_rows = []
    refs, rows = [], []
    for i, r in enumerate(val):
        real = word_clip(r, cache=cache, fetch=store.get, template=template)
        seconds = len(real.timestamps) / FPS
        plan = plan_clip(r["english"], model, vocab)
        words1 = select_words(r["english"], vocab)
        if not plan["entries"] or not words1:
            continue
        refs.append(real)
        forms["real"].append(real)
        d1 = plan_durations("crossfade", words1, vocab, seconds, 4)
        clips1 = [clip(choose_record(pools, i, w, "matched")) for w in words1]
        forms["stage1_oracle"].append(concatenate([time_scale(c, word_frames(d)) for c, d in zip(clips1, d1)], 4))
        clips2 = [clip(choose_record(pools, i, w, "matched")) for w in plan["entries"]]
        rule = dict(CORE_RULE, threshold=args.core_threshold, margin=args.core_margin)
        pieces = [dict(state=c, **core_window(c, rule)) for c in clips2]
        for f in fits:
            fp = frame_plan(plan["targets"], seconds, model, f)
            forms[f"stage2_oracle@{f}"].append(render(parts_from_pieces(pieces), fp))
        job = dict(words=plan["entries"], targets=plan["targets"], weights=plan["weights"], block="matched",
                   overlap=4, durations=stage1_durations("oracle_stage2", plan["entries"], vocab, seconds, 4))
        for timing, name in (("stage1", "stage2_words@stage1_timing"), ("sign-fitted", "stage2_oracle@sign_fitted")):
            m = oracle_material(job, i, pools, loaded={}, load=clip, rule=None, timing=timing, fractions=fractions,
                                margin=args.sign_margin)
            state, fp = render_form(m, job, seconds, model, timing=timing)
            forms[name].append(state)
            if timing == "sign-fitted":
                sign_rows += [dict(core=s["hi"] - s["lo"] + 1, planned=f, reason=s["reason"])
                              for s, f in zip(m["segments"], fp["cores"]) if f]
        rows.append(dict(sign=r["sign"], words1=len(words1), entries=len(plan["entries"]),
                         core_frames=[p["hi"] - p["lo"] + 1 for p in pieces],
                         planned=frame_plan(plan["targets"], seconds, model, fits[0])["cores"]))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(val)} composed", flush=True)
    motion = {name: mean_statistics(motion_rows(states, refs, FPS)) for name, states in forms.items()}
    ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    comp_ratio = [c / q for row in rows for c, q in zip(row["core_frames"], row["planned"])]
    report = dict(purpose="stage-2 timing check on ASLLRP val (real isolated clips, development data)",
                  model_sha256=model["sha256"], mix_key=args.mix_key, utterances=len(rows), total_fit=fits,
                  motion=motion, motion_ratio_to_real=ratio,
                  core_compression=dict(median=round(float(np.median(comp_ratio)), 4),
                                        p10_p90=[round(float(np.percentile(comp_ratio, q)), 4) for q in (10, 90)]),
                  sign_fitted=dict(margin=args.sign_margin, fractions_all=fractions["all"],
                                   core_compression_median=round(float(np.median(
                                       [x["core"] / x["planned"] for x in sign_rows])), 4) if sign_rows else None,
                                   core_frames_median=float(np.median([x["core"] for x in sign_rows]))
                                   if sign_rows else None,
                                   reasons=dict(collections.Counter(x["reason"] for x in sign_rows))),
                  signs_per_utterance=dict(stage1=float(np.mean([x["words1"] for x in rows])),
                                           stage2=float(np.mean([x["entries"] for x in rows]))),
                  clips_loaded=len(loaded), per_utterance=rows)
    with open(args.output_file, "x") as stream:
        json.dump(report, stream, indent=1, default=float)
    keys = ("seconds_over_reference", "wrist_speed", "hands_tracked", "wrist_spread")
    for name in forms:
        print(f"{name:28s} " + "  ".join(f"{k} {motion[name].get(k):.3f} ({ratio[name].get(k):.2f}x)"
                                        for k in keys if motion[name].get(k) is not None), flush=True)
    print(json.dumps(dict(core_compression=report["core_compression"], sign_fitted=report["sign_fitted"],
                          signs=report["signs_per_utterance"],
                          utterances=len(rows))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
