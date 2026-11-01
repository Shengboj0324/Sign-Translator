"""Post-S2 A6's motion figures, before any read: A3 generated top10's words from their stored caches, rebuilt at predicted timing and margin 0, under each candidate arm. Motion only: no reader, no hypothesis, nothing trains.
The base form must reproduce A3's stored per-clip wrist speeds, or the run stops."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from comp_a2_left_arm import CACHE_KEY, CACHE_SHA, FPS, MODEL_SHA, keypoints, operating_report, read_operating  # noqa: E402
from comp_a_lib import D20_KEY, D20_SHA, annotated_label, apply_token_map, arm_of, check_d20_payload, entry_annotations, extra_vocabulary, hold_flags, load_rest_pose, load_token_list, require_d20, side_read  # noqa: E402

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
EXTRA_KEY = f"{O}/round20/comp-a/cache-a3-top30-nogains/index.json"
EXTRA_SHA = "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
STORED_KEY = f"{O}/round20/comp-a/a3-gen-top10-shard0.json"
STORED_SHA = "6722fc84648338f06f919e8ddd4c3e24b3919a9de83b65bbbbe5baa22f58a59a"
TOKENS = ROOT / "reports/comp_a3_tokens.json"
TOKENS_SHA = "c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776"
EXCLUDE = .25
TOLERANCE = 1e-6
FORMS = ("base", "hold_a1", "hold_x", "hold_xr", "length_tokens", "length_words", "hold_x_length", "hold_xr_length")
HOLD_TOTALS = ("stretches", "frames", "minimum_jerk", "constant_speed", "held", "ready_pose", "cap_bound", "rest_ramps", "rest_whole", "rest_too_short", "rest_frames")


def held_uses(words, flags, entry_two_share) -> dict:
    """How many uses a form holds, and how many of those are entries with at least ``EXCLUDE`` of their annotated records two-handed."""
    held = [w for w in words if flags.get(w) is True]
    return dict(uses=len(words), held=len(held), held_two_handed_entries=sum(1 for w in held if entry_two_share.get(w, 0.) >= EXCLUDE))


def two_handed_shares(pools) -> dict:
    out = {}
    for entry, records in pools.items():
        labels = [x for x in map(annotated_label, records) if x is not None]
        if labels:
            out[entry] = sum(not x for x in labels) / len(labels)
    return out


def length_row(frames, true_frames) -> dict:
    f, t = np.asarray(frames, float), np.asarray(true_frames, float)
    return dict(clips=len(f), pooled_ratio=round(float(f.sum() / t.sum()), 4), median_ratio=round(float(np.median(f / t)), 4), mean_ratio=round(float((f / t).mean()), 4),
                median_abs_relative_error=round(float(np.median(np.abs(f / t - 1))), 4), median_abs_seconds=round(float(np.median(np.abs(f - t)) / FPS), 3))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--clips", type=int, default=500)
    p.add_argument("--hold-stats-file", default=str(ROOT / "reports/comp_a_hold_stats.json"))
    p.add_argument("--rest-file", default=str(ROOT / "reports/comp_a6_rest_pose.json"))
    p.add_argument("--rest-sha256", required=True)
    p.add_argument("--lengths-file", default=str(ROOT / "reports/comp_a6_lengths.json"))
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from compositional_baseline import motion_rows
    from evaluate_text_to_motion import at_rate, load_state
    from round15_compose_eval import stage1_selection
    from round15_oracle import word_pools
    import round15_stage2 as r2
    import round15_stage2_fit as fit
    from round20_lib import cached_material, load_sentence_set
    from s2_lib import check_read_records, load_manifest_at
    from train_text_to_motion import make_store
    from signtranslator.generation import compositional_grammar as cg
    from signtranslator.generation.composition_v2 import fit_duration_model, hold_left_arm, predicted_seconds
    from signtranslator.generation.compositional import CORE_RULE, KEPT, isolated_corpora, render, training_vocabulary
    import comp_a6_lengths as lengths
    store = make_store(args.store)
    require_d20(D20_KEY, D20_SHA)
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    raw = store.get(D20_KEY)
    if hashlib.sha256(raw).hexdigest() != D20_SHA:
        raise SystemExit(f"{D20_KEY} differs from its pin")
    payload = json.loads(raw)
    check_d20_payload(payload)
    val = load_sentence_set(payload, manifest, EVAL_MIX, EVAL_MIX_SHA)[:args.clips]
    check_read_records(val)
    cache_rows = {}
    for key, sha in ((CACHE_KEY, CACHE_SHA), (EXTRA_KEY, EXTRA_SHA)):
        data = store.get(key)
        if hashlib.sha256(data).hexdigest() != sha:
            raise SystemExit(f"{key} differs from its pin")
        cache_rows.update({r["entry"]: r for r in json.loads(data)["entries"]})
    data = store.get(STORED_KEY)
    if hashlib.sha256(data).hexdigest() != STORED_SHA:
        raise SystemExit(f"{STORED_KEY} differs from its pin")
    stored = {r["sign"]: (r["motion"].get("core_crossfade@predicted") or {}).get("wrist_speed") for r in json.loads(data)["clips"]}     # the motion alone
    del data
    token_list = load_token_list(TOKENS, TOKENS_SHA)
    arm = arm_of(token_list, 10)
    corpora = isolated_corpora(manifest)
    vocab_r10 = training_vocabulary(manifest, corpora)
    vocab = dict(vocab_r10, **extra_vocabulary(arm))
    model, _ = fit.fit_model(fit.agc._store(args.store), EVAL_MIX, EVAL_MIX_SHA, corpora)
    if model["sha256"] != MODEL_SHA:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the declared {MODEL_SHA}")
    plan_model = dict(model, timing=dict(model["timing"], annotated={k: v for k, v in model["timing"]["annotated"].items() if k not in arm["entries"]}))
    fractions = fit.sign_fractions(manifest, vocab_r10)
    fractions = dict(fractions, entries={k: v for k, v in fractions["entries"].items() if k not in arm["entries"]}, min_seconds={e: token_list["min_segment_seconds"] for e in arm["entries"]})
    cores = partial(r2.sign_core, fractions=fractions, margin=0., tighten=False)
    rule = dict(CORE_RULE, margin=0.)
    pools = {e: v for e, v in word_pools(manifest, corpora).items() if e not in arm["entries"]}
    flags_a1 = entry_annotations(pools)
    flags_x = hold_flags(pools, two_handed_share=EXCLUDE, new_entries=arm["entries"])
    shares = two_handed_shares(pools)
    stats = json.loads(Path(args.hold_stats_file).read_bytes())["how2sign_train"]
    rest = load_rest_pose(args.rest_file, args.rest_sha256)
    old = fit_duration_model(manifest)
    if (old["intercept"], old["slope"], old["records"]) != (-.083, .3691, 18289):
        raise SystemExit(f"the duration model is {old}, not a10's")
    balanced = fit_duration_model(manifest, components=("c02", "c04"), signer_balanced=True)
    fitted = json.loads(Path(args.lengths_file).read_text())
    words_model = fitted["candidates"]["words_balanced"]["model"]
    if (balanced["intercept"], balanced["slope"]) != (fitted["candidates"]["tokens_balanced"]["model"]["intercept"], fitted["candidates"]["tokens_balanced"]["model"]["slope"]):
        raise SystemExit(f"the balanced fit {balanced} is not the committed one")
    texts = [r["text"] for r in val]
    details, _ = stage1_selection(texts, vocab_r10)
    details, words1 = apply_token_map(details, arm["token_map"])
    features = lengths.plan_features(texts, details, words1, vocab)
    rows = {k: [] for k in FORMS}
    pooled = {k: [] for k in FORMS}
    holds, uses = {k: [] for k in FORMS}, {k: dict(uses=0, held=0, held_two_handed_entries=0) for k in ("hold_a1", "hold_x")}
    frames, true_frames, frames_with_words, true_with_words = {k: [] for k in ("old", "tokens", "words")}, [], {k: [] for k in ("old", "tokens", "words")}, []
    real_rows, built, cached, replaced = [], [], {}, {k: [0, 0] for k in FORMS}
    for i, record in enumerate(val):
        ref = at_rate(load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"]), FPS)
        points, valid = keypoints(ref)
        real_rows.append(side_read(points, valid))
        real_row = motion_rows([ref], [ref], FPS)[0]
        seconds = dict(old=predicted_seconds(old, record["text"]), tokens=predicted_seconds(balanced, record["text"]),
                       words=float(lengths.apply_line([words_model["intercept"], words_model["entry_seconds"], words_model["unsigned_token"]], features[i:i + 1])[0]))
        true_f = int(round(float(record["continuous"]["seconds"]) * FPS))
        true_frames.append(true_f)
        for k, v in seconds.items():
            frames[k].append(int(round(v * FPS)))
        words = words1[i]
        if not words:
            for k in FORMS:
                rows[k].append(None)
                pooled[k].append((None, real_row["wrist_speed"], None, real_row["wrist_spread"]))
            continue
        true_with_words.append(true_f)
        for k in seconds:
            frames_with_words[k].append(frames[k][-1])
        plan_words = r2.plan_words(words, plan_model, vocab, rows=details[i])
        job = dict(mode="core_crossfade", words=list(words), block="matched", weights=plan_words["weights"])
        for w in words:
            if w not in cached:
                pick = cache_rows[w]["chosen"]["16"]
                cached[w] = load_state(store, pick["state_key"], pick["state_sha256"])
        material = cached_material(job, cached, cores=cores, rule=rule)
        plans = {k: cg.fitted_plan(job["weights"], cg.gap_ratio(model["timing"]), v) for k, v in seconds.items()}
        flags = [flags_a1.get(w) for w in words]
        states = dict(old=render(material["parts"], plans["old"], sample_id=f"a6-old:{i}"), tokens=render(material["parts"], plans["tokens"], sample_id=f"a6-t:{i}"),
                      words=render(material["parts"], plans["words"], sample_id=f"a6-w:{i}"))
        built.append((stored.get(record["sign"]), motion_rows([states["old"]], [ref], FPS)[0]["wrist_speed"]))

        def at_point(name, state, plan, info=None):
            row = motion_rows([state], [ref], FPS)[0]
            rows[name].append(read_operating(state, plan, flags, info))
            pooled[name].append((row["wrist_speed"], real_row["wrist_speed"], row["wrist_spread"], real_row["wrist_spread"]))
            total = len(state.timestamps)
            replaced[name][0] += (info or {}).get("frames", 0)
            replaced[name][1] += total
            if info:
                holds[name].append(info)
        at_point("base", states["old"], plans["old"])
        at_point("length_tokens", states["tokens"], plans["tokens"])
        at_point("length_words", states["words"], plans["words"])
        one = [flags_x.get(w) for w in words]
        cap = dict(ready=stats["ready_pose"], cap=stats["left_speed_p99"])
        for name, length, marks, extra in (("hold_a1", "old", flags, {}), ("hold_x", "old", one, {}), ("hold_xr", "old", one, dict(rest=rest)), ("hold_x_length", "tokens", one, {}), ("hold_xr_length", "tokens", one, dict(rest=rest))):
            state, info = hold_left_arm(states[length], plans[length], marks, **cap, **extra)
            at_point(name, state, plans[length], info)
        for name, table in (("hold_a1", flags_a1), ("hold_x", flags_x)):
            one_use = held_uses(words, table, shares)
            for k, v in one_use.items():
                uses[name][k] += v
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(val)} clips", flush=True)
    worst = max((abs(a - b) for a, b in built if a is not None and b is not None), default=None)
    if worst is None or worst > TOLERANCE:
        raise SystemExit(f"the rebuild does not reproduce A3's stored wrist speeds: largest difference {worst}")
    out = dict(kind="comp-a6-motion", report_only=True, clips=len(val), set=dict(key=D20_KEY, sha256=D20_SHA), note="no hypothesis was read; motion of stored generated words only",
               word_caches=dict(main=dict(key=CACHE_KEY, sha256=CACHE_SHA), extra=dict(key=EXTRA_KEY, sha256=EXTRA_SHA)), rebuild_check=dict(clips=len(built), max_abs_difference=round(float(worst), 9), against=STORED_KEY),
               exclude=EXCLUDE, held_uses=uses, lengths=dict(all_clips={k: length_row(v, true_frames) for k, v in frames.items()}, clips_with_words={k: length_row(v, true_with_words) for k, v in frames_with_words.items()}),
               forms={})
    for name in FORMS:
        report = operating_report(rows[name], real_rows, pooled[name])
        report.pop("regions", None)
        report.pop("hold", None)
        report["replaced_share"] = round(replaced[name][0] / replaced[name][1], 4) if replaced[name][1] else None
        if holds[name]:
            report["hold"] = {k: int(sum(h.get(k, 0) for h in holds[name])) for k in HOLD_TOTALS}
            report["hold"].update(cap=holds[name][0]["cap"], peak=max(h["peak"] for h in holds[name]))
        out["forms"][name] = report
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
