"""Avatar fixes, 2026-11-02: pick clips, measure shake and penetration, sweep the filter, render, and store every render.

Run with the project Python from the worktree, one BLAS thread, under nice::

    avatar_fixes.py pick --work W                 # D20 clips by a fixed draw, real and draw 1's composition; states cached by hash
    avatar_fixes.py sweep --work W --group arm    # one filter group's candidates, report only (--output-json)
    avatar_fixes.py measure --work W --output-json reports/avatar_fixes_measures.json
    avatar_fixes.py speedup --work W --output-json OUT
    avatar_fixes.py render --work W --setting v1-768 --clip 14 --arm real [--baseline-repo ~/src/Sign-Translator]
    avatar_fixes.py sheet --work W --out sheet.png DIR1 DIR2 ...

**Clips.** How2Sign val D20 only (signers c01, c03, c05): a clip that is not in D20, or is in C20, or has another signer, is refused
before anything is fetched. Real states are the D20 record's; generated states are ``round20/parent-controls/composed/draw1``'s stored forms.
**Nothing here sets a render by arm.** ``render`` gives ``render_avatar.py`` the same arguments for a real and a generated state; a
setting is a name for a resolution and a code version, and the arm only chooses which state is read.
**Storage.** ``render`` saves everything under ``renders/avatar-fixes-2026-11-02/<setting>/<clip>/`` with ``INDEX.json`` (bytes, SHA-256),
re-reads each object, and deletes nothing. Baseline settings run the renderer of an untouched checkout of main.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ACQUISITION = "acquisition/2026-10-17-v1/youtube-asl-v1"
D20_KEY = f"{ACQUISITION}/round20/d20.json"
C20_KEY = f"{ACQUISITION}/round20/c20.json"
COMPOSED_KEY = f"{ACQUISITION}/round20/parent-controls/composed/draw1/index.json"
R2_PREFIX = "renders/avatar-fixes-2026-11-02"
SIGNERS = ("c01", "c03", "c05")
SEED = 20261102
PER_SIGNER = 12
FPS = 24.
ARMS = ("real", "gen")


def store():
    from train_text_to_motion import make_store
    return make_store("r2")


def load_clips(work: Path) -> list[dict]:
    return json.loads((work / "clips.json").read_text())


def load_state(work: Path, sha: str):
    from signtranslator.pose.multichannel import MultichannelMotion
    return MultichannelMotion.load(work / "states" / f"{sha}.npz", expected_sha256=sha)


def cmd_pick(args) -> int:
    objects = store()
    d20 = json.loads(objects.get(D20_KEY))
    records = {r["sign"]: r for r in d20["records"]}
    c20 = {r["sign"] for r in json.loads(objects.get(C20_KEY))["records"]}
    rows = [r for r in json.loads(objects.get(COMPOSED_KEY))["rows"]["comp"] if not r["empty"]]
    by_signer: dict[str, list] = {s: [] for s in SIGNERS}
    for row in rows:
        record = records.get(row["sign"])
        if record is None or row["sign"] in c20 or record["signer_component"] not in SIGNERS:
            raise SystemExit(f"{row['sign']}: refused, not a D20 clip of c01, c03 or c05")
        if record["state_sha256"] != row["reference_sha256"]:
            raise SystemExit(f"{row['sign']}: the composed row names another recording")
        by_signer[record["signer_component"]].append(row)
    rng = random.Random(SEED)
    chosen = []
    for signer in SIGNERS:
        pool = sorted((r for r in by_signer[signer] if 60 <= r["frames"] <= 240 and 60 <= r["reference_frames"] <= 300 and len(r["words"]) >= 4),
                      key=lambda r: r["sign"])
        for row in rng.sample(pool, PER_SIGNER):
            record = records[row["sign"]]
            chosen.append(dict(signer=signer, sign=row["sign"], frames=row["frames"], reference_frames=row["reference_frames"],
                               gen_key=row["state_key"], gen_sha=row["state_sha256"], real_key=record["state_key"], real_sha=record["state_sha256"]))
    work = args.work
    (work / "states").mkdir(parents=True, exist_ok=True)

    def fetch(key_sha):
        key, sha = key_sha
        path = work / "states" / f"{sha}.npz"
        if not path.exists():
            payload = objects.get(key)
            if hashlib.sha256(payload).hexdigest() != sha:
                raise SystemExit(f"{key}: bytes do not match")
            path.write_bytes(payload)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(fetch, [(c[f"{a}_key"], c[f"{a}_sha"]) for c in chosen for a in ARMS]))
    (work / "clips.json").write_text(json.dumps(chosen, indent=1))
    print(json.dumps(dict(clips=len(chosen), per_signer=PER_SIGNER, seed=SEED)))
    return 0


def packet_state(state, *, held: bool):
    """The packet's own path to a video's state: 24 fps on a rebuilt clock, and for the old renderer, holds."""
    from build_human_eval_packet import held_for_display, uniform_clock
    from evaluate_text_to_motion import at_rate
    shown = uniform_clock(at_rate(state, FPS), FPS)
    return held_for_display(shown) if held else shown


def _rig():
    from signtranslator.avatar_render.makehuman import load_makehuman_rig
    from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
    from signtranslator.avatar_render.retarget import build_retarget_binding
    intake = ROOT / "Sign Translator Stage Documentation" / "evidence" / "w1-w2-2026-09-26" / "source-intake"
    rig = load_makehuman_rig(intake / "makehuman")
    return rig, build_retarget_binding(rig), load_makehuman_face_units(intake / "makehuman-face", rig)


def _median(values) -> float | None:
    kept = [v for v in values if v is not None and np.isfinite(v)]
    return float(statistics.median(kept)) if kept else None


def _quantile(values, q: float) -> float | None:
    kept = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(kept, q)) if kept else None


def _flatten_shake(record: dict) -> dict:
    out = {}
    for name, row in record["points"].items():
        for key in ("jitter_mm", "hf_share", "below_3hz", "from_3_to_6hz", "above_6hz", "jitter_px"):
            if key in row:
                out[f"{name}.{key}"] = row[key]
    for name, row in record["rotations"].items():
        out[f"{name}.jitter_deg"] = row["jitter_deg"]
        out[f"{name}.rot_hf_share"] = row["hf_share"]
    for side, row in record["fingertips"].items():
        out[f"tips.{side}.jitter_mm"] = row["jitter_mm"]
        out[f"tips.{side}.hf_share"] = row["hf_share"]
    return out


def _lowpassed(state):
    """The state with every rotation channel at the 6 Hz band: what a filter's bite below 6 Hz is read on."""
    from signtranslator.avatar_render.motion_filter import _from_6d, _to_6d
    from signtranslator.avatar_render.motion_quality import lowpass
    channels = dict(state.channels)
    for name in ("body", "left_hand", "right_hand", "head_rotation"):
        channel = channels[name]
        keep = [k for k in range(len(channel.labels)) if channel.valid[:, k].all()]
        if not keep:
            continue
        values = channel.values.astype(np.float64).copy()
        values[:, keep] = _to_6d(_from_6d(lowpass(values[:, keep], FPS, 6.)))
        channels[name] = replace(channel, values=values.astype(channel.values.dtype))
    return replace(state, channels=channels)


def _face_jitter(state) -> dict:
    from signtranslator.avatar_render.motion_quality import lowpass
    face = state.channels["face"]
    keep = [k for k, name in enumerate(face.labels) if name != "_neutral" and face.valid[:, k].all()]
    values = face.values[:, keep, 0].astype(np.float64)
    residual = values - lowpass(values, FPS, 6.)
    blink = state.channels["blink"]
    peak = float(blink.values[:, :, 0].max()) if blink.valid.any() else None
    return dict(face_jitter_rms=float(np.sqrt((residual ** 2).mean(0)).mean()), blink_peak=peak)


def cmd_measure(args) -> int:
    from signtranslator.avatar_render.collision import apply_correction, build_field, penetration
    from signtranslator.avatar_render.motion_filter import FILTER_V1, fill_gaps, filter_state
    from signtranslator.avatar_render.motion_quality import articulation, preservation, shake
    from signtranslator.avatar_render.retarget import retarget_state
    rig, binding, _ = _rig()
    field = build_field(rig)
    pixels_per_unit = 768. / 9.
    rows = []
    started = time.time()
    for clip in load_clips(args.work):
        for arm in ARMS:
            stored = load_state(args.work, clip[f"{arm}_sha"])
            held = packet_state(stored, held=True)
            raw = packet_state(stored, held=False)
            filled, gaps = fill_gaps(raw)
            filtered = filter_state(filled, FILTER_V1)
            r_held = retarget_state(rig, binding, held, allow_inferred=True)
            r_filled = retarget_state(rig, binding, filled, allow_inferred=True)
            r_filtered = retarget_state(rig, binding, filtered, allow_inferred=True)
            r_final, collision = apply_correction(rig, field, r_filtered)
            low = _lowpassed(filled)
            r_low = retarget_state(rig, binding, low, allow_inferred=True)
            r_low_filtered = retarget_state(rig, binding, filter_state(low, FILTER_V1), allow_inferred=True)
            art = {name: articulation(rig, r.local_rotations, FPS) for name, r in
                   (("held", r_held), ("filled", r_filled), ("filtered", r_filtered), ("final", r_final), ("low", r_low), ("low_filtered", r_low_filtered))}
            row = dict(sign=clip["sign"], signer=clip["signer"], arm=arm, frames=len(raw.timestamps),
                       arm_invalid=float(1. - raw.channels["body"].valid[:, raw.channels["body"].labels.index("RightForearm")].mean()),
                       before=_flatten_shake(shake(rig, r_held.local_rotations, FPS, screen_pixels_per_unit=pixels_per_unit)),
                       filled=_flatten_shake(shake(rig, r_filled.local_rotations, FPS, screen_pixels_per_unit=pixels_per_unit)),
                       after_filter=_flatten_shake(shake(rig, r_filtered.local_rotations, FPS, screen_pixels_per_unit=pixels_per_unit)),
                       after=_flatten_shake(shake(rig, r_final.local_rotations, FPS, screen_pixels_per_unit=pixels_per_unit)),
                       dominant=max("LR", key=lambda side: art["filled"][side]["wrist_speed_m_s"]["mean_6hz"]),
                       kept_filter=preservation(art["filled"], art["filtered"], FPS),
                       kept_vs_held=preservation(art["held"], art["filtered"], FPS),
                       kept_correction=preservation(art["filtered"], art["final"], FPS),
                       kept_on_lowpassed=preservation(art["low"], art["low_filtered"], FPS),
                       penetration_before=penetration(rig, field, r_held.local_rotations),
                       penetration_filtered=collision["before"], penetration_after=collision["after"], moved=collision["moved"],
                       face_before=_face_jitter(filled), face_after=_face_jitter(filtered),
                       head_speed_6hz_kept=_head_kept(filled, filtered), gaps_filled=len(gaps))
            rows.append(row)
        print(clip["sign"], round(time.time() - started), flush=True)
    summary = summarise(rows)
    out = dict(kind="avatar-fixes-measures", clips=len(rows), seed=SEED, draw="composed draw1 comp form, D20 real two-view", summary=summary, rows=rows)
    Path(args.output_json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(summary["table"], indent=1))
    return 0


def _head_kept(before, after) -> float | None:
    from signtranslator.avatar_render.motion_filter import _from_6d
    from signtranslator.avatar_render.motion_quality import angular_speed, lowpass
    a = before.channels["head_rotation"]
    b = after.channels["head_rotation"]
    if not a.valid.all():
        return None
    speed = lambda channel: angular_speed(_from_6d(lowpass(channel.values[:, 0].astype(np.float64), FPS, 6.)), FPS)
    x, y = speed(a), speed(b)
    mask = x >= np.percentile(x, 70)
    return float(y[mask].mean() / x[mask].mean()) if x[mask].mean() > 1e-9 else None


def summarise(rows: list[dict]) -> dict:
    points = ["head.jitter_mm", "chest.jitter_mm", "elbow.L.jitter_mm", "elbow.R.jitter_mm", "wrist.L.jitter_mm", "wrist.R.jitter_mm",
              "tips.L.jitter_mm", "tips.R.jitter_mm", "head.jitter_deg", "chest.jitter_deg", "hand.L.jitter_deg", "hand.R.jitter_deg",
              "head.hf_share", "chest.hf_share", "wrist.L.hf_share", "wrist.R.hf_share", "tips.L.hf_share", "tips.R.hf_share",
              "head.jitter_px", "wrist.L.jitter_px", "wrist.R.jitter_px"]
    table: dict = {}
    for arm in ARMS:
        here = [r for r in rows if r["arm"] == arm]
        table[arm] = {p: dict(before=_median(r["before"][p] for r in here), filled=_median(r["filled"][p] for r in here),
                              after_filter=_median(r["after_filter"][p] for r in here), after=_median(r["after"][p] for r in here)) for p in points}
        for source in ("kept_filter", "kept_vs_held", "kept_correction", "kept_on_lowpassed"):
            table[arm][source] = {}
            for key in ("finger_speed_raw", "finger_speed_6hz", "finger_speed_3hz", "finger_active_6hz", "finger_peak_6hz", "finger_range",
                        "wrist_active_6hz", "wrist_p95_6hz", "wrist_max_6hz", "wrist_max_raw", "hold_iou", "hold_position_shift_p95_mm",
                        "hold_jitter_before_mm", "hold_jitter_after_mm", "hold_share_before", "hold_share_after", "hold_count_before",
                        "hold_count_after", "finger_quiet_raw_before", "finger_quiet_raw_after", "finger_bend_lag_frames"):
                values = [r[source][s][key] for r in here for s in ("L", "R")]
                table[arm][source][key] = dict(median=_median(values), p10=_quantile(values, 10))
        for target in ("hand.torso", "hand.head", "forearm.torso", "forearm.head"):
            row = {}
            for when in ("penetration_before", "penetration_filtered", "penetration_after"):
                values = [r[when][target] for r in here]
                frames = sum(r["frames"] for r in here)
                weight = lambda key: sum(v[key] * r["frames"] for v, r in zip(values, here)) / frames
                hits = [v["depth_max_mm"] for v in values if v["depth_max_mm"] > 0.]
                row[when] = dict(frames_any=weight("frames_over_0mm"), frames_over_5mm=weight("frames_over_5mm"), frames_over_10mm=weight("frames_over_10mm"),
                                 frames_over_20mm=weight("frames_over_20mm"), clips_with_any=len(hits) / len(values),
                                 clip_max_depth_median_mm=_median(hits) if hits else 0., clip_max_depth_max_mm=max(hits) if hits else 0.)
            table[arm][target] = row
        table[arm]["kept_filter_dominant_arm"] = {}
        for key in ("wrist_active_6hz", "wrist_p95_6hz", "wrist_max_6hz", "finger_speed_6hz", "finger_peak_6hz", "finger_range", "hold_iou"):
            values = [r["kept_filter"][r["dominant"]][key] for r in here]
            lowpassed = [r["kept_on_lowpassed"][r["dominant"]][key] for r in here]
            table[arm]["kept_filter_dominant_arm"][key] = dict(median=_median(values), p10=_quantile(values, 10), on_lowpassed_median=_median(lowpassed),
                                                              on_lowpassed_p10=_quantile(lowpassed, 10))
        moved = [r["moved"][s] for r in here for s in ("L", "R")]
        table[arm]["moved"] = {key: dict(median=_median(m[key] for m in moved), p95=_quantile((m[key] for m in moved), 95), max=max(m[key] for m in moved))
                               for key in ("wrist_depth_move_cm_max", "wrist_image_shift_cm_max", "swivel_deg_max")}
        table[arm]["face"] = dict(jitter_before=_median(r["face_before"]["face_jitter_rms"] for r in here),
                                  jitter_after=_median(r["face_after"]["face_jitter_rms"] for r in here),
                                  blink_peak_kept=_median(r["face_after"]["blink_peak"] / r["face_before"]["blink_peak"] for r in here if r["face_before"]["blink_peak"]),
                                  head_speed_active_kept=_median(r["head_speed_6hz_kept"] for r in here))
    signers = {}
    for signer in SIGNERS:
        for arm in ARMS:
            here = [r for r in rows if r["arm"] == arm and r["signer"] == signer]
            signers[f"{signer}.{arm}"] = dict(arm_invalid=_median(r["arm_invalid"] for r in here),
                                              **{p: dict(before=_median(r["before"][p] for r in here), after=_median(r["after"][p] for r in here))
                                                 for p in ("head.jitter_mm", "wrist.R.jitter_mm", "tips.R.jitter_mm")})
    return dict(table=table, signers=signers)


SWEEP_GRID = {
    "torso": ["SG7,2", "SG9,2", "SG13,2", "SG15,2", "SG21,2", "OE.5,.5", "OE1,1", "OE1.5,2"],
    "arm": ["SG5,2", "SG7,3", "OE1,3", "OE2,2", "OE2,3", "OE3,2", "OE3,3", "OE1.5,5"],
    "hand": ["SG5,2", "SG7,3", "OE1,3", "OE2,2", "OE2,3", "OE3,2", "OE3,3", "OE1.5,5"],
    "finger": ["SG5,2", "OE1.5,3", "OE2,2", "OE2,3", "OE3,3", "OE3,2", "OE4,2", "OE4,1", "OE2,5"],
    "head": ["SG5,2", "SG7,2", "SG9,2", "OE1,2", "OE2,2", "OE2,5"],
}


def _spec(name: str):
    from signtranslator.avatar_render.motion_filter import OneEuro, SavitzkyGolay
    a, b = name[2:].split(",")
    return SavitzkyGolay(int(a), int(b)) if name.startswith("SG") else OneEuro(float(a), float(b))


def cmd_sweep(args) -> int:
    """One group's candidates, every other group unfiltered, report only: the record behind the settings chosen."""
    from signtranslator.avatar_render.motion_filter import FilterSettings, fill_gaps, filter_state
    from signtranslator.avatar_render.motion_quality import articulation, preservation, shake
    from signtranslator.avatar_render.retarget import retarget_state
    rig, binding, _ = _rig()
    states = [(arm, fill_gaps(packet_state(load_state(args.work, clip[f"{arm}_sha"]), held=False))[0]) for clip in load_clips(args.work) for arm in ARMS]

    def run(settings):
        return [(arm, *(lambda r: (_flatten_shake(shake(rig, r.local_rotations, FPS)), articulation(rig, r.local_rotations, FPS)))(
            retarget_state(rig, binding, filter_state(state, settings), allow_inferred=True))) for arm, state in states]

    base = run(FilterSettings())
    out = dict(group=args.group, candidates={})
    for name in SWEEP_GRID[args.group]:
        result = run(FilterSettings(**{args.group: _spec(name)}))
        row = {}
        for arm in ARMS:
            pairs = [(b, r) for b, r in zip(base, result) if r[0] == arm]
            row[arm] = {p: _median(r[1][p] for _, r in pairs) for p in ("head.jitter_mm", "chest.jitter_mm", "wrist.L.jitter_mm", "wrist.R.jitter_mm",
                                                                       "tips.L.jitter_mm", "tips.R.jitter_mm", "hand.R.jitter_deg")}
            kept = [preservation(b[2], r[2], FPS) for b, r in pairs]
            for key in ("finger_speed_6hz", "finger_speed_3hz", "finger_peak_6hz", "finger_range", "wrist_p95_6hz", "wrist_max_6hz", "hold_iou"):
                row[arm][f"kept.{key}"] = _median(k[s][key] for k in kept for s in ("L", "R"))
        out["candidates"][name] = row
        print(name, json.dumps(row)[:160], flush=True)
    Path(args.output_json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


def cmd_speedup(args) -> int:
    """Does squeezing real signing in time raise hand shake? Real clips played faster, read beside the real and generated clips themselves."""
    from evaluate_text_to_motion import at_rate
    from build_human_eval_packet import uniform_clock
    from signtranslator.avatar_render.motion_filter import fill_gaps
    from signtranslator.avatar_render.motion_quality import shake
    from signtranslator.avatar_render.retarget import retarget_state
    rig, binding, _ = _rig()
    keys = ("wrist.L.jitter_mm", "wrist.R.jitter_mm", "tips.L.jitter_mm", "tips.R.jitter_mm", "hand.L.jitter_deg", "hand.R.jitter_deg", "head.jitter_mm", "chest.jitter_mm")
    table = {}
    for name, arm, factor in (("real", "real", 1.), ("real played 1.5x", "real", 1.5), ("real played 2x", "real", 2.), ("generated", "gen", 1.)):
        rows = []
        for clip in load_clips(args.work):
            state = load_state(args.work, clip[f"{arm}_sha"])
            state = replace(state, timestamps=state.timestamps / factor)
            filled, _ = fill_gaps(uniform_clock(at_rate(state, FPS), FPS))
            rows.append(_flatten_shake(shake(rig, retarget_state(rig, binding, filled, allow_inferred=True).local_rotations, FPS)))
        table[name] = {key: _median(row[key] for row in rows) for key in keys}
        print(name, json.dumps(table[name]), flush=True)
    Path(args.output_json).write_text(json.dumps(table, indent=1) + "\n")
    return 0


def _upload(directory: Path, setting: str, clip_name: str, source: dict, extra: dict) -> dict:
    objects = store()
    prefix = f"{R2_PREFIX}/{setting}/{clip_name}"
    index = dict(setting=setting, clip=clip_name, source=source, **extra, files={})
    paths = sorted(p for p in directory.rglob("*") if p.is_file())

    def put(path):
        payload = path.read_bytes()
        key = f"{prefix}/{path.relative_to(directory)}"
        digest = objects.put(key, payload)              # re-reads and checks
        return str(path.relative_to(directory)), dict(bytes=len(payload), sha256=digest)

    with ThreadPoolExecutor(8) as pool:
        index["files"] = dict(pool.map(put, paths))
    payload = json.dumps(index, indent=1, sort_keys=True).encode()
    objects.put(f"{prefix}/INDEX.json", payload)
    return dict(prefix=prefix, files=len(paths), bytes=sum(f["bytes"] for f in index["files"].values()))


def cmd_render(args) -> int:
    clip = load_clips(args.work)[args.clip]
    sha, key = clip[f"{args.arm}_sha"], clip[f"{args.arm}_key"]
    baseline = args.baseline_repo is not None
    state = packet_state(load_state(args.work, sha), held=baseline)
    # The file holds no arm: every state is written under the same name.
    prepared = args.work / "prepared" / f"{hashlib.sha256(f'{sha}{baseline}'.encode()).hexdigest()[:16]}"
    prepared.mkdir(parents=True, exist_ok=True)
    path = prepared / "state.npz"
    if not path.exists():
        state.save(path)
    clip_name = f"{clip['signer']}-{clip['sign']}-{args.arm}"
    out = args.work / "renders" / args.setting / clip_name
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    repo = Path(args.baseline_repo).expanduser() if baseline else ROOT
    command = [sys.executable, str(repo / "scripts" / "render_avatar.py"), "--state", str(path), "--out-dir", str(out), "--allow-inferred",
               "--resolution", str(args.resolution)]
    started = time.perf_counter()
    subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True)
    seconds = time.perf_counter() - started
    video = out / "avatar.mp4"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "scripts/render_avatar.py", "signtranslator/avatar_render"], cwd=repo,
                                capture_output=True, text=True).stdout.strip())
    extra = dict(render_seconds=seconds, code_commit=commit, code_dirty=dirty, resolution=args.resolution, video_bytes=video.stat().st_size, frames=len(state.timestamps),
                 code="main-4cc0db6 (baseline)" if baseline else "avatar-fixes", d20_clip=True, signer=clip["signer"], arm=args.arm)
    (out / "render_seconds.json").write_text(json.dumps(dict(seconds=seconds)))
    stored = _upload(out, args.setting, clip_name, dict(key=key, sha256=sha), extra)
    print(json.dumps(dict(setting=args.setting, clip=clip_name, frames=extra["frames"], seconds=round(seconds, 1), video_bytes=extra["video_bytes"], **stored)))
    return 0


def cmd_sheet(args) -> int:
    """Rows of evenly spaced frames, one row per render directory, labelled with nothing but the row number."""
    from PIL import Image, ImageDraw
    rows = []
    for directory in args.dirs:
        frames = sorted((Path(directory) / "frames").glob("frame-*.png"))
        picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, args.columns)]
        rows.append([Image.open(p).convert("RGB") for p in picks])
    tile = args.tile
    sheet = Image.new("RGB", (args.columns * tile, len(rows) * tile), "white")
    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            sheet.paste(image.resize((tile, tile), Image.LANCZOS), (c * tile, r * tile))
        ImageDraw.Draw(sheet).text((4, r * tile + 4), str(r + 1), fill="white")
    sheet.save(args.out)
    print(args.out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--work", type=Path, required=True)
    p.set_defaults(run=cmd_pick)
    p = sub.add_parser("measure")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output-json", required=True)
    p.set_defaults(run=cmd_measure)
    p = sub.add_parser("sweep")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--group", choices=sorted(SWEEP_GRID), required=True)
    p.add_argument("--output-json", required=True)
    p.set_defaults(run=cmd_sweep)
    p = sub.add_parser("speedup")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output-json", required=True)
    p.set_defaults(run=cmd_speedup)
    p = sub.add_parser("render")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--clip", type=int, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--resolution", type=int, default=768)
    p.add_argument("--baseline-repo", help="an untouched checkout of main: its renderer, on the packet's held state")
    p.set_defaults(run=cmd_render)
    p = sub.add_parser("sheet")
    p.add_argument("--out", required=True)
    p.add_argument("--columns", type=int, default=6)
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--work", type=Path)
    p.set_defaults(run=cmd_sheet)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
