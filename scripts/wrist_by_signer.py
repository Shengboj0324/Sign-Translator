"""Wrist speed and spread of real signing, per signer, on CPU: how fast is real, and could How2Sign's small motion come from how it was lifted? Local and read-only on stored states and metadata. Nothing trains and no reader runs.

**Groups** (train partitions; c00 and Ben are refused, not skipped): How2Sign two-view c02 and c04, YouTube train by channel (the corpus has no signer field, so a channel stands in for one and may hold several), ASLLRP train by signer. D20's clips, their metadata
and motion are the reference only. Every number is the report's own keypoint view at 24 fps, raw and low-passed at 6 and 3 Hz (``composition_v2.lowpass_state``), each ratio against D20 real at the same rate.

**Lifting check.** How2Sign's front clips also have monocular estimates (``derived/how2sign-video-estimate-v1``, extracted as YouTube's states are), so the same signer on the same clip is read through both paths. Each path's stored clock is reported beside it.
A clock check: seconds per English token per group, which a wrong frame rate would move, though signers differ too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
H2S = "how2sign-two-view-text-v1"
YOUTUBE = "youtube-asl-v1"
MONOCULAR = "acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1"
SEED = "wrist-by-signer"
FPS = 24.
RATES = {"raw": None, "6hz": 6., "3hz": 3.}
KEYS = ("wrist_speed", "wrist_spread", "left_speed", "right_speed", "left_spread", "right_spread")
H2S_TRAIN = ("c02", "c04")
PER_SIGNER, YOUTUBE_CHANNELS, PER_CHANNEL, ASLLRP_PER_SIGNER, MONOCULAR_REFERENCE = 300, 60, 5, 150, 300
BOOT = 400
REVIEW = dict(raw=dict(c02=1.08, c04=1.08, youtube=1.51, asllrp=1.56, top10=1.77, hold_x=1.60), low_3hz=dict(c02=1.07, c04=.80, youtube=1.84, asllrp=2.14, top10=2.19, hold_x=1.95))


def refuse_c00(records) -> None:
    for r in records:
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")


def how2sign_train(manifest, signer: str, n: int = PER_SIGNER) -> list[dict]:
    """A fixed draw of one How2Sign train signer's records with a state."""
    if signer == "c00":
        raise SystemExit("How2Sign signer 9 (c00) is never read")
    pool = sorted((r for r in manifest["records"] if r.get("corpus") == H2S and r.get("partition") == "train" and r.get("signer_component") == signer and (r.get("continuous") or {}).get("state_key")), key=lambda r: r["sign"])
    return random.Random(f"{SEED}|{signer}").sample(pool, min(n, len(pool)))


def youtube_by_channel(manifest, channels: int = YOUTUBE_CHANNELS, per: int = PER_CHANNEL) -> dict:
    """{channel: records}: a fixed draw of train channels with enough clips, and of clips in each."""
    pool = {}
    for r in manifest["records"]:
        if r.get("corpus") == YOUTUBE and r.get("partition") == "train" and (r.get("continuous") or {}).get("state_key"):
            pool.setdefault((r.get("source") or {}).get("channel_id") or "unknown", []).append(r)
    names = sorted(k for k, v in pool.items() if len(v) >= per)
    picked = random.Random(f"{SEED}|channels").sample(names, min(channels, len(names)))
    return {k: random.Random(f"{SEED}|{k}").sample(sorted(pool[k], key=lambda r: r["sign"]), per) for k in sorted(picked)}


def asllrp_by_signer(gloss, utterances, n: int = ASLLRP_PER_SIGNER) -> dict:
    """{signer: records} of ASLLRP train, by the gloss manifest's own split. A Ben record marked train stops the run."""
    from build_asllrp_gloss_manifest import PARTITION, signer
    records = {r["sign"]: r for r in gloss["records"]}
    pool = {}
    for u in utterances:
        video = u.get("utterance_video")
        if not video:
            continue
        name, record = signer(u.get("source") or ""), records.get(f"{u['collection']}:{Path(video).stem}")
        if name == "Ben" and (record or {}).get("partition") == "train":
            raise SystemExit(f"{record['sign']}: Ben is the test partition, never read")
        if PARTITION.get(name) == "train" and record is not None and record.get("partition") == "train" and (record.get("continuous") or {}).get("state_key"):
            pool.setdefault(name, []).append(record)
    return {k: random.Random(f"{SEED}|{k}").sample(sorted(v, key=lambda r: r["sign"]), min(n, len(v))) for k, v in sorted(pool.items())}


def clip_row(state) -> dict:
    """One clip at every rate, and its stored clock."""
    from comp_a_lib import keypoint_view, side_read
    from evaluate_text_to_motion import at_rate
    from sentence_exposure_diagnostic import keypoint_statistics
    from signtranslator.generation.composition_v2 import lowpass_state
    t = np.asarray(state.timestamps, dtype=float)
    native = float(1. / np.median(np.diff(t))) if len(t) > 1 else None
    resampled = at_rate(state, FPS)
    out = dict(native_fps=native, frames=len(t), seconds=float(t[-1] - t[0]) if len(t) > 1 else 0.)
    for name, hz in RATES.items():
        s = resampled if hz is None else lowpass_state(resampled, hz)
        points, valid = keypoint_view(s)
        k = keypoint_statistics(points, valid, FPS)
        out[name] = dict(side_read(points, valid), wrist_speed=k["wrist_speed"], wrist_spread=k["wrist_spread"])
    return out


def mean_key(rows, rate: str, key: str) -> float | None:
    values = [r[rate][key] for r in rows if r[rate].get(key) is not None]
    return float(np.mean(values)) if values else None


def group_row(rows: list, reference: list) -> dict:
    """Means per rate and key, and each mean's ratio to the reference at the same rate, with a clip bootstrap on the pooled wrist."""
    out = dict(clips=len(rows), native_fps_median=round(float(np.median([r["native_fps"] for r in rows if r["native_fps"]])), 3) if rows else None)
    for rate in RATES:
        row = {}
        for key in KEYS:
            mine, other = mean_key(rows, rate, key), mean_key(reference, rate, key)
            row[key] = None if mine is None else round(mine, 5)
            row[f"{key}_over_d20"] = None if mine is None or not other else round(mine / other, 4)
        a = np.array([r[rate]["wrist_speed"] for r in rows if r[rate].get("wrist_speed") is not None])
        b = np.array([r[rate]["wrist_speed"] for r in reference if r[rate].get("wrist_speed") is not None])
        if len(a) and len(b):
            rng = np.random.default_rng(0)
            draws = [a[rng.integers(0, len(a), len(a))].mean() / b[rng.integers(0, len(b), len(b))].mean() for _ in range(BOOT)]
            row["wrist_speed_over_d20_ci95"] = [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)]
        out[rate] = row
    return out


def equal_weight(groups: dict, reference: list) -> dict:
    """A corpus's row with every signer (or channel) weighing the same: the mean of their ratios at each rate, with their spread."""
    rows = {k: group_row(v, reference) for k, v in groups.items() if v}
    out = dict(groups=len(rows), clips=sum(r["clips"] for r in rows.values()))
    for rate in RATES:
        out[rate] = {}
        for key in KEYS:
            values = [r[rate][f"{key}_over_d20"] for r in rows.values() if r[rate].get(f"{key}_over_d20") is not None]
            if values:
                out[rate][f"{key}_over_d20"] = round(float(np.mean(values)), 4)
                out[rate][f"{key}_over_d20_range"] = [round(float(np.percentile(values, 10)), 3), round(float(np.percentile(values, 90)), 3)]
    return out


def paired_row(two_view: list, monocular: list) -> dict:
    """The same clips through both paths: monocular over two-view, at each rate, pooled and per clip. Only pairs with both numbers."""
    out = dict(clips=len(two_view), two_view_fps=round(float(np.median([r["native_fps"] for r in two_view if r["native_fps"]])), 3), monocular_fps=round(float(np.median([r["native_fps"] for r in monocular if r["native_fps"]])), 3),
               seconds_ratio_median=round(float(np.median([m["seconds"] / a["seconds"] for a, m in zip(two_view, monocular) if a["seconds"] and m["seconds"]])), 4))
    for rate in RATES:
        out[rate] = {}
        for key in KEYS:
            pairs = [(a[rate][key], m[rate][key]) for a, m in zip(two_view, monocular) if a[rate].get(key) is not None and m[rate].get(key) is not None and a[rate][key] > 0]
            if pairs:
                x, y = np.array(pairs).T
                rng = np.random.default_rng(0)
                pick = [rng.integers(0, len(x), len(x)) for _ in range(BOOT)]
                ratios = [y[p].mean() / x[p].mean() for p in pick]
                out[rate][key] = dict(pairs=len(pairs), monocular_over_two_view=round(float(y.mean() / x.mean()), 4), ci95=[round(float(np.percentile(ratios, 2.5)), 3), round(float(np.percentile(ratios, 97.5)), 3)], median_per_clip=round(float(np.median(y / x)), 4))
    return out


def tokens_rate(rows) -> float | None:
    """Pooled seconds per English token: a wrong clock moves this, and so do signers."""
    from signtranslator.generation.compositional import normalise
    seconds, tokens = 0., 0
    for r in rows:
        s, n = (r.get("continuous") or {}).get("seconds"), len(normalise(r.get("text") or ""))
        if s and n:
            seconds += float(s)
            tokens += n
    return round(seconds / tokens, 4) if tokens else None


def carry_arithmetic(motion: dict) -> dict:
    """What a right-arm carry can do to the pooled wrist. ``carry`` is defined nowhere in the repository, so this reads it as the right arm's path on the bridges going to zero or to real's speed, the steps kept, on A6's stored regions.
    Pooled wrist weighs each side by real's speed: pooled = (left ratio x real left + right ratio x real right) / (real left + real right)."""
    base = motion["forms"]["base"]
    weight = motion["raw_real_weights"]
    left_w = weight["left"] / (weight["left"] + weight["right"])
    pooled = lambda l, r: round(left_w * l + (1 - left_w) * r, 4)         # noqa: E731
    base_cut = motion["forms"]["hold_x"]["ratio_to_real"]["left"] / base["ratio_to_real"]["left"]
    out = dict(left_weight=round(left_w, 4), hold_x_left_cut=round(base_cut, 4), check_top10=pooled(base["ratio_to_real"]["left"], base["ratio_to_real"]["right"]), stored_top10=base["pooled_wrist_ratio"], rows={})
    for name in ("base", "hold_x", "hold_x_length", "length_tokens"):
        form = motion["forms"][name]
        left, right = form["ratio_to_real"]["left"], form["ratio_to_real"]["right"]
        regions = form["regions"]["right"]
        bridges = {k: v for k, v in regions.items() if k != "speed" and "bridge" in k}
        share = sum(v["path_share"] or 0. for v in bridges.values())
        steps = sum(v["step_share"] or 0. for v in bridges.values())
        no_bridge = right * (1 - share)
        real_on_bridges = right * (1 - share) + steps          # in ratio units, a step at real's speed counts 1
        out["rows"][name] = dict(left=left, right=right, pooled=pooled(left, right), pooled_if_right_cut_as_hold_x_cut_left=pooled(left, right * base_cut), right_bridge_path_share=round(share, 4), right_bridge_step_share=round(steps, 4), right_if_bridges_still=round(no_bridge, 4), pooled_if_bridges_still=pooled(left, no_bridge),
                                 right_if_bridges_at_real_speed=round(real_on_bridges, 4), pooled_if_bridges_at_real_speed=pooled(left, real_on_bridges), right_needed_for_1_44=round((1.44 - left_w * left) / (1 - left_w), 4))
    return out


def path_matched(pairs: dict, youtube: dict) -> dict:
    """YouTube over D20 with both on the monocular path: D20's monocular ratio is its signers' monocular rows, by clips."""
    out = {}
    for rate in RATES:
        rows = [(pairs[c]["clips"], pairs[c]["monocular_only"][rate]["wrist_speed_over_d20"]) for c in ("c01", "c03", "c05") if pairs[c].get("clips")]
        d20 = sum(n * x for n, x in rows) / sum(n for n, _ in rows)
        out[rate] = dict(d20_monocular_over_d20_two_view=round(d20, 4), youtube_over_d20_monocular=round(youtube[rate]["wrist_speed_over_d20"] / d20, 4))
    return out


NOTE = ("Real signing's wrist differs by signer and by corpus far more than the review's first numbers show one by one. Same clips through both paths: the monocular path reads 18-39% more wrist motion at 3 Hz and 23-37% more spread than two-view on every How2Sign signer, while its raw speed "
        "is 0.83-1.11x, so How2Sign's small motion could partly come from how it was lifted. Both stored clocks are 24 fps. After the path is matched, YouTube still reads about 1.5x D20 at both raw and 3 Hz. Composition's excess over real grows when jitter is filtered: top10 1.77x raw, 2.20x at 3 Hz.")
UNDECIDED = ("Which path is right: two-view is the reference elsewhere in this project, yet no ground truth for arm amplitude exists here, so this cannot say whether two-view shrinks motion or monocular inflates it. Whether YouTube's remainder is signers, clip cutting (0.57 s per token, against D20's 0.30) or framing. "
             "Whether a stored 24 fps is the sources' true rate: the clock check moves with signers too. A channel is not a signer. ASLLRP's path was not compared on the same clips. Carry: the word is defined nowhere in the repository, so 1.44x is neither confirmed nor refuted: on hold-x it needs the right wrist at the row's right_needed_for_1_44.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--motion-file", default=str(ROOT / "reports/comp_a6_motion.json"))
    p.add_argument("--limit", type=int, help="the first K clips of every group (smoke)")
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from comp_a_lib import D20_KEY, D20_SHA, check_d20_payload
    from evaluate_text_to_motion import load_state
    import round15_stage2_fit as fit
    from round20_lib import GLOSS_MANIFEST, load_sentence_set
    from s2_lib import load_manifest_at
    from train_text_to_motion import make_store
    store = make_store(args.store)
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    raw = store.get(D20_KEY)
    if hashlib.sha256(raw).hexdigest() != D20_SHA:
        raise SystemExit(f"{D20_KEY} differs from its pin")
    payload = json.loads(raw)
    check_d20_payload(payload)
    d20 = load_sentence_set(payload, manifest, EVAL_MIX, EVAL_MIX_SHA)
    refuse_c00(d20)
    cut = (lambda v: v[:args.limit]) if args.limit else (lambda v: v)

    def read(records, label):
        rows = []
        for k, r in enumerate(cut(records)):
            rows.append(clip_row(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"])))
            if (k + 1) % 100 == 0:
                print(f"{label}: {k + 1}", flush=True)
        return rows

    by_signer = {s: how2sign_train(manifest, s) for s in H2S_TRAIN}
    channels = youtube_by_channel(manifest)
    gloss = load_manifest_at(store, *GLOSS_MANIFEST)
    asllrp = asllrp_by_signer(gloss, json.loads(store.get(fit.KEYS["utterances"]))["items"])
    rate_groups = dict(d20=d20, **by_signer, youtube=[r for v in channels.values() for r in v], **{f"asllrp_{k}": v for k, v in asllrp.items()})
    clock = {k: tokens_rate(v) for k, v in rate_groups.items()}
    clock.update({f"d20_{c}": tokens_rate([r for r in d20 if r.get("signer_component") == c]) for c in ("c01", "c03", "c05")})
    monocular_index = {}
    for key in store.list(f"{MONOCULAR}/reports/"):
        if "/test_" in key:                                       # the official test split holds c00: its report stays unread
            continue
        for row in json.loads(store.get(key))["results"]:
            if row.get("state_key") and row.get("state_sha256"):
                monocular_index[row["sample_id"]] = (row["state_key"], row["state_sha256"])
    del gloss
    reference = read(d20, "d20")
    print("reference", json.dumps({r: mean_key(reference, r, "wrist_speed") for r in RATES}), flush=True)
    d20_by_signer = {c: [row for row, r in zip(reference, cut(d20)) if r.get("signer_component") == c] for c in ("c01", "c03", "c05")}
    rows = {s: read(v, s) for s, v in by_signer.items()}
    youtube = {k: read(v, f"youtube {k}") for k, v in channels.items()}
    asllrp_rows = {k: read(v, f"asllrp {k}") for k, v in asllrp.items()}
    pairs = {}
    for signer, records, two_view in [(s, cut(by_signer[s]), rows[s]) for s in H2S_TRAIN] + [(c, [r for r in cut(d20) if r.get("signer_component") == c][:MONOCULAR_REFERENCE], d20_by_signer[c][:MONOCULAR_REFERENCE]) for c in ("c01", "c03", "c05")]:
        a, m = [], []
        for record, row in zip(records, two_view):
            hit = monocular_index.get(record["sign"])
            if hit is None:
                continue
            a.append(row)
            m.append(clip_row(load_state(store, hit[0], hit[1])))
        pairs[signer] = dict(paired_row(a, m), monocular_only=group_row(m, reference), two_view_only=group_row(a, reference), missing=len(records) - len(a)) if a else dict(clips=0, missing=len(records))
        print(f"paired {signer}: {len(a)}", flush=True)
    motion = json.loads(Path(args.motion_file).read_text())
    composition = {}
    for name in ("base", "hold_x", "length_tokens", "hold_x_length"):
        form = motion["forms"][name]
        composition[name] = dict(raw=dict(wrist_speed_over_d20=form["pooled_wrist_ratio"], wrist_spread_over_d20=form["pooled_spread_ratio"], left_speed_over_d20=form["ratio_to_real"]["left"], right_speed_over_d20=form["ratio_to_real"]["right"],
                                          left_spread_over_d20=form["spread_to_real"]["left"], right_spread_over_d20=form["spread_to_real"]["right"]))
        for hz, label in (("6", "6hz"), ("3", "3hz")):
            low, real = motion["low_passed"][hz][name], motion["low_passed"][hz]["real"]
            composition[name][label] = {f"{k}_over_d20": round(low[k] / real[k], 4) for k in KEYS if low.get(k) and real.get(k)}
    motion["raw_real_weights"] = dict(left=mean_key(reference, "raw", "left_speed"), right=mean_key(reference, "raw", "right_speed"))
    youtube_row = equal_weight(youtube, reference)
    out = dict(kind="wrist-by-signer", report_only=True, seed=SEED, path_matched=path_matched(pairs, youtube_row), reference=dict(key=D20_KEY, sha256=D20_SHA, note="D20 metadata and motion are the reference only; nothing is fitted on them", row=group_row(reference, reference), by_signer={c: group_row(v, reference) for c, v in d20_by_signer.items()}),
               how2sign_two_view={s: group_row(v, reference) for s, v in rows.items()}, youtube=dict(youtube_row, note="no signer field: one row per channel, equal weight; a channel may hold several signers", pooled=group_row([r for v in youtube.values() for r in v], reference)),
               asllrp=dict(equal_weight(asllrp_rows, reference), signers={k: group_row(v, reference) for k, v in asllrp_rows.items()}), composition=composition, carry=carry_arithmetic(motion), monocular_against_two_view=pairs, seconds_per_token=clock, review_first_numbers=REVIEW,
               monocular_indexed=len(monocular_index), note=NOTE, cannot_decide=UNDECIDED)
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
