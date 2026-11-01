"""Post-S2 A2 (reports/post_s2_plan.md section 3): why does the left wrist run 2.70x real and the right 1.77x? Local, read-only on
stored caches and clips; no generation, no translator and no recogniser, so no hypothesis is read. D20 only; never c00 or test.

Each clip is rebuilt as the stored read built it: the real-word oracle (a1's draws, sign-fitted cores, margin 0.1, true length) and
a9's words (the stored no-gains best-of-16 cache). Wrist speed is the report's own metric (``keypoints2d.normalise``, both wrists
apart), and a rebuilt clip must match the stored shard's ``wrist_speed`` (``rebuild_check``).

**One-handed, two ways.** By motion, as the diagnosis: the left wrist's core path (chest frame) under 0.35 of the right's. And by
annotation, where a record has handshapes (ASLLVD and RIT): no non-dominant handshape means one hand. The annotation is the one to
trust: the motion label cannot see a resting arm that moves, which is the very thing being asked, and it also fires on an arm that
is not tracked. A generated word takes its entry's majority annotation. ``annotated`` repeats the tables on this label.

**What is read, per side.**
- The sentence's path split by region: cores of one-handed words, cores of two-handed words, bridges and lead-in/out.
- Per one-handed entry, by corpus: left speed at native time and at planned time, against the right's and real's.
- Each join, by the kinds of word it joins and by whether both clips come from one corpus: the left wrist's travel and speed.
- The left arm's rest pose per corpus, within- and cross-corpus distance between one-handed words' rests.
- ``left_active``: word clips whose left wrist travels further than the right, by corpus (a signer left unmirrored would show here).

**The defect the scale could hide** is a resting arm travelling between rest poses. ``what_if`` gives the arithmetic bound for each
candidate: the left speed ratio if that path were zero with the same steps. These are bounds on a metric, not new motion.

**One set of joins** (``comp_a_lib.bridge_kind``) serves the medians and the region shares alike, so a median and a share always describe the
same joins. ``one_to_two`` is an annotated one-handed word against an annotated two-handed one.

**Operating points.** ``a9_predicted_margin01`` is a10's a9@predicted, ``a9_predicted_margin0`` is A1's margin0 arm, and ``hold`` is that arm with the
left arm really held (``composition_v2.hold_left_arm``): real rotations, real spread, per side, contact, and how the speed cap fell. Motion only: no
reader runs. The right wrist is no reference for the left, because real signing has the same asymmetry: read rest depth (``left_height``), frames and
How2Sign train's own distribution (``reports/comp_a_hold_stats.json``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from comp_a_lib import (D20_KEY, D20_SHA, annotated_label, annotated_two_handed, bridge_kind, check_d20_payload, contact_counts, entry_annotations,  # noqa: E402
                        require_d20, side_read)

FPS = 24.
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
A = f"{O}/round20/phase-a"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
MODEL_SHA = "42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21"
CACHE_KEY = f"{A}/cache-w14b-nogains-r10/index.json"
CACHE_SHA = "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
STORED = dict(oracle=f"{A}/a1-oracle-r10-shard0.json", a9=f"{A}/a9-nogains-n16-r10-shard0.json")
STORED_FORM = dict(oracle="oracle_stage2", a9="core_crossfade")
ONE_HANDED = .35
SIDES = ("left", "right")
WRISTS = (4, 5)                      # the left and right wrists in the keypoint view (keypoints2d)
REGIONS = ("core_one_handed", "core_two_handed", "bridge", "lead_trail")
ANNOTATED_REGIONS = ("core_annotated_one", "core_annotated_two", "core_unannotated", "bridge_one_to_one", "bridge_one_to_two",
                     "bridge_one_to_unannotated", "bridge_other", "lead_trail")


# ---- pure pieces ---------------------------------------------------------------------------------------------------------

def side_steps(points: np.ndarray, valid: np.ndarray, fps: float = FPS) -> dict:
    """{side: (per-step speed, ok mask)} for (T, K, 2) normalised points: the metric's own step on valid pairs."""
    out = {}
    for name, w in zip(SIDES, WRISTS):
        ok = valid[1:, w] & valid[:-1, w]
        out[name] = (np.linalg.norm(points[1:, w] - points[:-1, w], axis=-1) * fps, ok)
    return out


def frame_labels(plan: dict) -> np.ndarray:
    from signtranslator.generation.composition_v2 import frame_labels as labels_of
    return labels_of(plan)


def step_regions(labels: np.ndarray, one_handed) -> list[str]:
    """Each frame-to-frame step's region: a core only when both frames are in the same core; anything else that touches a bridge
    or joins two cores is a bridge; the lead and trail apart."""
    out = []
    for a, b in zip(labels[:-1], labels[1:]):
        if a == b and a >= 0:
            out.append("core_one_handed" if one_handed[int(a)] else "core_two_handed")
        elif a == -2 or b == -2:
            out.append("lead_trail")
        else:
            out.append("bridge")
    return out


def annotated_step_regions(labels: np.ndarray, one_handed) -> list[str]:
    """``step_regions`` by the annotation: ``one_handed[k]`` True, False or None (no annotation). A bridge step takes the core before it and the
    core after it, by ``comp_a_lib.bridge_kind``: the same set the join tables use."""
    labels = np.asarray(labels)
    last, before = None, []
    for x in labels:
        if x >= 0:
            last = int(x)
        before.append(last)
    following, after = None, [None] * len(labels)
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] >= 0:
            following = int(labels[i])
        after[i] = following
    out = []
    for i, (a, b) in enumerate(zip(labels[:-1], labels[1:])):
        if a == b and a >= 0:
            flag = one_handed[int(a)]
            out.append("core_unannotated" if flag is None else "core_annotated_one" if flag else "core_annotated_two")
        elif a == -2 or b == -2:
            out.append("lead_trail")
        else:
            x, y = one_handed[before[i]] if before[i] is not None else None, one_handed[after[i + 1]] if after[i + 1] is not None else None
            out.append(f"bridge_{bridge_kind(x, y)}")
    return out


def is_one_handed(left_path: float, right_path: float, ratio: float = ONE_HANDED) -> bool:
    return bool(left_path < ratio * right_path)


def path_by_region(steps: dict, regions, names=REGIONS) -> dict:
    """{side: {region: [summed speed over valid steps, valid steps]}}."""
    regions = np.asarray(regions)
    out = {}
    for side, (speed, ok) in steps.items():
        out[side] = {r: [float(speed[(regions == r) & ok].sum()), int(((regions == r) & ok).sum())] for r in names}
    return out


def add_paths(total: dict, one: dict) -> None:
    for side, rows in one.items():
        for region, (a, b) in rows.items():
            total[side][region][0] += a
            total[side][region][1] += b


def empty_paths(names=REGIONS) -> dict:
    return {s: {r: [0., 0] for r in names} for s in SIDES}


def region_table(paths: dict) -> dict:
    """Per side and region: the share of the path, the share of the steps and the mean speed (per second, as the metric)."""
    out = {}
    for side, rows in paths.items():
        path, steps = sum(a for a, _ in rows.values()), sum(b for _, b in rows.values())
        out[side] = {r: dict(path_share=round(a / path, 4) if path else None, step_share=round(b / steps, 4) if steps else None,
                             speed=round(a / b, 4) if b else None) for r, (a, b) in rows.items()}
        out[side]["speed"] = round(path / steps, 4) if steps else None
    return out


def what_if(paths: dict, side: str, regions, real_speed: float) -> float | None:
    """The pooled speed ratio to real if the path in ``regions`` were zero, the steps kept."""
    rows = paths[side]
    steps = sum(b for _, b in rows.values())
    if not steps or not real_speed:
        return None
    return round(sum(a for r, (a, _) in rows.items() if r not in regions) / steps / real_speed, 4)


def join_kind(left_is_one: bool, right_is_one: bool) -> str:
    return f"{'one' if left_is_one else 'two'}_to_{'one' if right_is_one else 'two'}"


def summary(values) -> dict | None:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None
    return dict(n=len(v), mean=round(float(np.mean(v)), 4), median=round(float(np.median(v)), 4),
                p90=round(float(np.percentile(v, 90)), 4))


# ---- clips ----------------------------------------------------------------------------------------------------------------

def keypoints(state):
    from signtranslator.data.keypoints2d import from_state, normalise
    from evaluate_text_to_motion import at_rate
    points, valid = from_state(at_rate(state, FPS))
    return normalise(points, valid, y_down=False), valid


def core_paths(piece) -> tuple[float, float]:
    """Each wrist's path over a piece's core in the chest frame (metres), on valid steps: the diagnosis's one-handed measure."""
    from signtranslator.generation.composition_space import wrist_from_shoulder
    idx = np.arange(piece["lo"], piece["hi"] + 1)
    out = []
    for side in ("Left", "Right"):
        p, ok = wrist_from_shoulder(piece["state"], side)
        good = ok[idx[1:]] & ok[idx[:-1]]
        out.append(float(np.linalg.norm(p[idx[1:]] - p[idx[:-1]], axis=-1)[good].sum()))
    return out[0], out[1]


def native_core_read(piece) -> dict:
    """A piece's core at native time: per side speed, tracked share and mean wrist position in the metric's view."""
    from signtranslator.generation.compositional import take_frames
    core = take_frames(piece["state"], np.arange(piece["lo"], piece["hi"] + 1))
    points, valid = keypoints(core)
    steps = side_steps(points, valid)
    out = {}
    for name, w in zip(SIDES, WRISTS):
        speed, ok = steps[name]
        out[name] = dict(speed=float(speed[ok].mean()) if ok.any() else None, tracked=float(valid[:, w].mean()),
                         position=[float(x) for x in points[valid[:, w], w].mean(0)] if valid[:, w].any() else None)
    return out


def read_clip(material, plan, state, *, records=None, annotations=None) -> dict:
    """One rebuilt sentence: per side the paths by region, each word's read and each join's. ``annotations``: the entry-level
    label for words without a record."""
    pieces = material["pieces"]
    words = []
    for k, piece in enumerate(pieces):
        left, right = core_paths(piece)
        native = native_core_read(piece)
        record = records[k] if records else None
        words.append(dict(word=material["segments"][k]["word"], corpus=record["corpus"] if record else "generated",
                          left_path=left, right_path=right, one_handed=is_one_handed(left, right),
                          annotated_two_handed=annotated_two_handed(record), native=native,
                          annotated_one=annotated_label(record) if record else (annotations or {}).get(material["segments"][k]["word"]),
                          performer=(record or {}).get("performer_sha256"),
                          frames_native=piece["hi"] - piece["lo"] + 1, frames_planned=plan["cores"][k]))
    labels = frame_labels(plan)
    points, valid = keypoints(state)
    if len(labels) != len(points):
        raise SystemExit("the plan and the rendered state differ in length")
    steps = side_steps(points, valid)
    regions = step_regions(labels, [w["one_handed"] for w in words])
    annotated = annotated_step_regions(labels, [w["annotated_one"] for w in words])
    joins = []
    start = plan["lead"]
    for k in range(len(words) - 1):
        start += plan["cores"][k]
        frames = plan["gaps"][k]
        a, b = start - 1, start + frames              # the last frame of core k and the first of core k + 1
        row = dict(kind=join_kind(words[k]["one_handed"], words[k + 1]["one_handed"]), frames=frames,
                   same_corpus=words[k]["corpus"] == words[k + 1]["corpus"], corpora=[words[k]["corpus"], words[k + 1]["corpus"]],
                   annotated_kind=bridge_kind(words[k]["annotated_one"], words[k + 1]["annotated_one"]),
                   same_performer=words[k]["performer"] is not None and words[k]["performer"] == words[k + 1]["performer"])
        if b < len(points) and a >= 0:
            for name, w in zip(SIDES, WRISTS):
                seen = valid[a, w] and valid[b, w]
                row[name] = dict(travel=float(np.linalg.norm(points[b, w] - points[a, w])) if seen else None,
                                 speed=float(np.linalg.norm(points[b, w] - points[a, w]) / max(frames + 1, 1) * FPS) if seen else None)
        joins.append(row)
        start += frames
    speeds = {name: (float(speed[ok].mean()) if ok.any() else None) for name, (speed, ok) in steps.items()}
    return dict(words=words, joins=joins, paths=path_by_region(steps, regions), speeds=speeds,
                annotated_paths=path_by_region(steps, annotated, ANNOTATED_REGIONS))


# ---- tables -----------------------------------------------------------------------------------------------------------------

def per_entry(words, real_left: float) -> dict:
    """One-handed occurrences, per corpus and per entry: left speed at native and planned time, as ratios to real's left."""
    by_corpus, by_entry = collections.defaultdict(list), collections.defaultdict(list)
    for w in words:
        if not w["one_handed"]:
            continue
        by_corpus[w["corpus"]].append(w)
        by_entry[(w["word"], w["corpus"])].append(w)

    def read(rows):
        left = [r["native"]["left"]["speed"] for r in rows]
        right = [r["native"]["right"]["speed"] for r in rows]
        squeeze = [r["frames_native"] / max(r["frames_planned"], 1) for r in rows]
        return dict(occurrences=len(rows), left_native=summary(left), right_native=summary(right),
                    left_native_over_real=round(float(np.nanmean([x if x is not None else np.nan for x in left])) / real_left, 4)
                    if any(x is not None for x in left) and real_left else None,
                    squeeze=summary(squeeze), left_tracked=summary(r["native"]["left"]["tracked"] for r in rows))
    table = {c: read(v) for c, v in sorted(by_corpus.items())}
    ranked = sorted(((read(v), e) for e, v in by_entry.items() if len(v) >= 3),
                    key=lambda t: -(t[0]["left_native"]["mean"] if t[0]["left_native"] else 0.))
    return dict(by_corpus=table, fastest_left_entries=[dict(entry=e[0], corpus=e[1], **r) for r, e in ranked[:25]],
                entries_with_three_or_more=len(ranked))


def rest_pose_table(words) -> dict:
    """The left wrist's mean position on one-handed words: per corpus, and the distance between consecutive one-handed words of
    one sentence is in the join table. Here: each corpus's centre and spread, and the centres' distances."""
    by = collections.defaultdict(list)
    for w in words:
        p = w["native"]["left"]["position"]
        if w["one_handed"] and p:
            by[w["corpus"]].append(p)
    centre = {c: np.mean(v, 0) for c, v in by.items()}
    return dict(per_corpus={c: dict(words=len(v), centre=[round(float(x), 4) for x in centre[c]],
                                    spread=round(float(np.linalg.norm(np.asarray(v) - centre[c], axis=-1).mean()), 4))
                            for c, v in sorted(by.items())},
                centre_distance={f"{a}|{b}": round(float(np.linalg.norm(centre[a] - centre[b])), 4)
                                 for a in sorted(centre) for b in sorted(centre) if a < b})


def join_table(joins) -> dict:
    out = {}
    groups = collections.defaultdict(list)
    for j in joins:
        groups[j["kind"]].append(j)
        groups[f"{j['kind']}|{'same_corpus' if j['same_corpus'] else 'cross_corpus'}"].append(j)
    for name, rows in sorted(groups.items()):
        out[name] = dict(joins=len(rows), frames=summary(r["frames"] for r in rows),
                         **{f"{s}_{m}": summary((r.get(s) or {}).get(m) for r in rows) for s in SIDES for m in ("travel", "speed")})
    return out


def left_active(words) -> dict:
    by = collections.defaultdict(list)
    for w in words:
        by[w["corpus"]].append(w["left_path"] > w["right_path"])
    return {c: dict(words=len(v), share=round(float(np.mean(v)), 4)) for c, v in sorted(by.items())}


TOUCHING = ("bridge_one_to_one", "bridge_one_to_two", "bridge_one_to_unannotated")


def annotated_tables(words, joins, paths, real: dict, scale: dict) -> dict:
    """The tables on the annotation's label: left against right on annotated one- and two-handed words, per corpus and per
    performer; joins by that label; and the arithmetic bound of each candidate defect."""
    def read(rows):
        return dict(words=len(rows), left_active=round(float(np.mean([w["left_path"] > w["right_path"] for w in rows])), 4) if rows else None,
                    left_over_right_path=summary(w["left_path"] / w["right_path"] for w in rows if w["right_path"] > 0),
                    left_native=summary(w["native"]["left"]["speed"] for w in rows), right_native=summary(w["native"]["right"]["speed"] for w in rows),
                    left_tracked=summary(w["native"]["left"]["tracked"] for w in rows), motion_label_one_handed=round(float(np.mean([w["one_handed"] for w in rows])), 4) if rows else None)
    groups = collections.defaultdict(list)
    for w in words:
        if w["annotated_one"] is not None:
            groups[("one" if w["annotated_one"] else "two", w["corpus"])].append(w)
    performers = collections.defaultdict(list)
    for w in words:
        if w["annotated_one"] and w["performer"]:
            performers[w["performer"]].append(w)
    per_performer = sorted(((read(v), p[:12]) for p, v in performers.items() if len(v) >= 20), key=lambda t: -t[0]["left_active"])
    grouped = collections.defaultdict(list)
    for j in joins:
        grouped[j["annotated_kind"]].append(j)
        grouped[f"{j['annotated_kind']}|{'same_performer' if j['same_performer'] else 'cross_performer'}"].append(j)
        grouped[f"{j['annotated_kind']}|{'same_corpus' if j['same_corpus'] else 'cross_corpus'}"].append(j)
    bound = lambda side, names: what_if(paths, side, names, real[side] / scale[side] if scale[side] else None)                 # noqa: E731
    return dict(by_label_and_corpus={f"{a}|{c}": read(v) for (a, c), v in sorted(groups.items())},
                per_performer_annotated_one=[dict(performer=p, **r) for r, p in per_performer[:12]],
                regions=region_table(paths), joins={k: dict(joins=len(v), frames=summary(r["frames"] for r in v),
                                                            **{f"{s}_{m}": summary((r.get(s) or {}).get(m) for r in v) for s in SIDES for m in ("travel", "speed")})
                                                    for k, v in sorted(grouped.items())},
                what_if={s: {"as_is": bound(s, ()), "no_path_on_annotated_one_handed_cores": bound(s, ("core_annotated_one",)),
                             "no_path_in_one_to_one_bridges": bound(s, ("bridge_one_to_one",)),
                             "no_path_in_one_to_two_bridges": bound(s, ("bridge_one_to_two",)),
                             "no_path_in_bridges_touching_one": bound(s, TOUCHING),
                             "no_path_on_annotated_one_cores_or_bridges_touching_one": bound(s, ("core_annotated_one", *TOUCHING)),
                             "no_path_in_any_bridge": bound(s, (*TOUCHING, "bridge_other"))} for s in SIDES})


def annotation_check(words) -> dict:
    rows = [w for w in words if w["annotated_two_handed"] is not None]
    if not rows:
        return dict(annotated=0)
    agree = [w["one_handed"] == (not w["annotated_two_handed"]) for w in rows]
    return dict(annotated=len(rows), motion_label_agrees=round(float(np.mean(agree)), 4),
                annotated_one_handed_share=round(float(np.mean([not w["annotated_two_handed"] for w in rows])), 4),
                motion_one_handed_share=round(float(np.mean([w["one_handed"] for w in rows])), 4))


def form_report(clips, real_speeds) -> dict:
    """One form's tables from its per-clip reads. Ratios to real use the clip means of the same clips, as the diagnosis."""
    good = [(c, r) for c, r in zip(clips, real_speeds) if c is not None and c["words"]]
    real = {s: float(np.nanmean([r[s] if r[s] is not None else np.nan for _, r in good])) for s in SIDES}
    form = {s: float(np.nanmean([c["speeds"][s] if c["speeds"][s] is not None else np.nan for c, _ in good])) for s in SIDES}
    paths, annotated = empty_paths(), empty_paths(ANNOTATED_REGIONS)
    words, joins = [], []
    for c, _ in good:
        add_paths(paths, c["paths"])
        add_paths(annotated, c["annotated_paths"])
        words += c["words"]
        joins += c["joins"]
    pooled = {s: sum(a for a, _ in paths[s].values()) / max(sum(b for _, b in paths[s].values()), 1) for s in SIDES}
    bounds, scales = {}, {s: (form[s] / pooled[s] if pooled[s] else 1.) for s in SIDES}
    for s in SIDES:
        scale = scales[s]   # clip means against pooled steps: stated, and applied to every bound
        bounds[s] = {"as_is_pooled": what_if(paths, s, (), real[s] / scale if scale else None),
                     "no_path_on_one_handed_cores": what_if(paths, s, ("core_one_handed",), real[s] / scale if scale else None),
                     "no_path_in_bridges": what_if(paths, s, ("bridge",), real[s] / scale if scale else None),
                     "no_path_in_bridges_or_one_handed_cores": what_if(paths, s, ("bridge", "core_one_handed"),
                                                                       real[s] / scale if scale else None),
                     "no_path_in_lead_trail": what_if(paths, s, ("lead_trail",), real[s] / scale if scale else None)}
    return dict(clips=len(good), real_speed={s: round(real[s], 4) for s in SIDES}, form_speed={s: round(form[s], 4) for s in SIDES},
                ratio_to_real={s: round(form[s] / real[s], 4) for s in SIDES}, left_over_right=dict(
                    real=round(real["left"] / real["right"], 4), form=round(form["left"] / form["right"], 4)),
                regions=region_table(paths), what_if=bounds, one_handed_share=round(float(np.mean([w["one_handed"] for w in words])), 4),
                annotation_check=annotation_check(words), left_active=left_active(words), one_handed_entries=per_entry(words, real["left"]),
                rest_pose=rest_pose_table(words), joins=join_table(joins),
                annotated=annotated_tables(words, joins, annotated, real, scales))


def read_operating(state, plan, flags, hold_info=None) -> dict:
    """One clip at an operating point, motion only: per side, contact, paths by the annotation's regions, joins and the left arm's rest depth."""
    labels = frame_labels(plan)
    points, valid = keypoints(state)
    steps = side_steps(points, valid)
    regions = annotated_step_regions(labels, flags)
    sides = side_read(points, valid)
    w = WRISTS[0]
    ones = np.array([k >= 0 and flags[k] is True for k in labels], dtype=bool) & valid[:, w]
    joins, start = [], plan["lead"]
    for k in range(len(flags) - 1):
        start += plan["cores"][k]
        frames = plan["gaps"][k]
        a, b = start - 1, start + frames
        row = dict(annotated_kind=bridge_kind(flags[k], flags[k + 1]), frames=frames)
        if a >= 0 and b < len(points):
            for name, wr in zip(SIDES, WRISTS):
                row[name] = float(np.linalg.norm(points[b, wr] - points[a, wr])) if valid[a, wr] and valid[b, wr] else None
        joins.append(row)
        start += frames
    return dict(paths=path_by_region(steps, regions, ANNOTATED_REGIONS), sides=sides, contact=contact_counts(points, valid, labels, flags), joins=joins,
                left_heights=points[ones, w, 1].tolist(), hold=hold_info)


def operating_report(clips, real_rows, pooled, hold_stats=None) -> dict:
    """The tables of one operating point, on one set of joins (module docstring)."""
    good = [(c, r) for c, r in zip(clips, real_rows) if c is not None]
    ratio = {}
    for key in ("left_speed", "right_speed", "left_spread", "right_spread"):
        pairs = [(c["sides"][key], r[key]) for c, r in good if c["sides"].get(key) is not None and r.get(key) is not None]
        ratio[key] = round(float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])), 4)
    paths = empty_paths(ANNOTATED_REGIONS)
    joins, heights, holds, contact = [], [], [], [0, 0]
    for c, _ in good:
        add_paths(paths, c["paths"])
        joins += c["joins"]
        heights += c["left_heights"]
        contact = [contact[0] + c["contact"][0], contact[1] + c["contact"][1]]
        if c.get("hold"):
            holds.append(c["hold"])
    table = {}
    for kind in ("one_to_one", "one_to_two", "one_to_unannotated", "other"):
        rows = [j for j in joins if j["annotated_kind"] == kind]
        table[kind] = dict(joins=len(rows), frames=summary(j["frames"] for j in rows), left_travel=summary(j.get("left") for j in rows),
                           right_travel=summary(j.get("right") for j in rows))
    def pooled_ratio(k):
        """As ``mean_statistics`` and ``ratio_to``: each mean over the clips that have the number, a word-less clip counted for real only."""
        form_values, real_values = [p[k] for p in pooled if p[k] is not None], [p[k + 1] for p in pooled if p[k + 1] is not None]
        return round(float(np.mean(form_values) / np.mean(real_values)), 4) if form_values and real_values else None
    out = dict(clips=len(good), ratio_to_real=dict(left=ratio["left_speed"], right=ratio["right_speed"]),
               spread_to_real=dict(left=ratio["left_spread"], right=ratio["right_spread"]),
               pooled_wrist_ratio=pooled_ratio(0), pooled_spread_ratio=pooled_ratio(2),
               regions=region_table(paths), joins=table, contact_share=round(contact[0] / contact[1], 4) if contact[1] else None, contact_frames=contact[1],
               left_height_on_annotated_one=summary(heights), left_path={k: v for k, v in paths["left"].items()})
    if holds:
        out["hold"] = {k: int(sum(h[k] for h in holds)) for k in ("stretches", "frames", "minimum_jerk", "constant_speed", "held", "ready_pose", "cap_bound")}
        out["hold"].update(cap=holds[0]["cap"], peak=max(h["peak"] for h in holds))
    return out


def rebuild_check(stored_speed, built_speed) -> dict:
    d = [abs(a - b) for a, b in zip(stored_speed, built_speed) if a is not None and b is not None]
    return dict(clips=len(d), max_abs_difference=round(float(max(d)), 9) if d else None)


# ---- the run ------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--clips", type=int, default=500)
    p.add_argument("--cache-dir", default=str(Path(tempfile.gettempdir()) / "sign-translator-cache"))
    p.add_argument("--hold-stats-file", default=str(ROOT / "reports/comp_a_hold_stats.json"))
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from signtranslator.generation.composition_v2 import fit_duration_model, hold_left_arm, predicted_seconds
    from signtranslator.generation import compositional_grammar as cg
    from signtranslator.generation.compositional import CORE_RULE, isolated_corpora, render, training_vocabulary
    from compositional_baseline import motion_rows
    from evaluate_text_to_motion import at_rate, load_state
    from functools import partial
    from round15_compose_eval import stage1_selection
    from round15_oracle import TEMPLATE_CORPUS, choose_record, word_clip, word_pools
    import round15_stage2 as r2
    import round15_stage2_fit as fit
    from round20_lib import cached_material, load_sentence_set
    from s2_lib import check_read_records, load_manifest_at
    from train_motion_tokens import layout_template
    from train_text_to_motion import make_store
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
    payload = store.get(CACHE_KEY)
    if hashlib.sha256(payload).hexdigest() != CACHE_SHA:
        raise SystemExit(f"{CACHE_KEY} differs from its pin")
    index = {r["entry"]: r for r in json.loads(payload)["entries"]}
    corpora = isolated_corpora(manifest)
    vocab = training_vocabulary(manifest, corpora)
    model, _ = fit.fit_model(fit.agc._store(args.store), EVAL_MIX, EVAL_MIX_SHA, corpora)
    if model["sha256"] != MODEL_SHA:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the declared {MODEL_SHA}")
    fractions = fit.sign_fractions(manifest, vocab)
    margin = .1
    mode = dict(timing="sign-fitted", fractions=fractions, margin=margin, tighten=False)
    cores = partial(r2.sign_core, fractions=fractions, margin=margin, tighten=False)
    rule = dict(CORE_RULE, margin=margin)
    pools = word_pools(manifest, corpora)
    cache = ContentCache(Path(args.cache_dir), 3 * 2 ** 30)
    template = layout_template(manifest, TEMPLATE_CORPUS, cache, store.get)
    load = lambda r: word_clip(r, cache=cache, fetch=store.get, template=template)    # noqa: E731
    annotations = entry_annotations(pools)
    duration = fit_duration_model(manifest)
    if (duration["intercept"], duration["slope"], duration["records"]) != (-.083, .3691, 18289):
        raise SystemExit(f"the duration model is {duration}, not a10's")
    stats = json.loads(Path(args.hold_stats_file).read_bytes())["how2sign_train"]
    cores0 = partial(r2.sign_core, fractions=fractions, margin=0., tighten=False)
    rule0 = dict(CORE_RULE, margin=0.)
    operating = {k: [] for k in ("a9_predicted_margin01", "a9_predicted_margin0", "hold")}
    pooled = {k: [] for k in operating}
    real_rows = []
    cached_states = {}
    stored = {name: {r["sign"]: (r["motion"].get(STORED_FORM[name]) or {}).get("wrist_speed")
                     for r in json.loads(store.get(key))["clips"]} for name, key in STORED.items()}
    texts = [r["text"] for r in val]
    details, words1 = stage1_selection(texts, vocab)
    clips = dict(oracle=[], a9=[])
    real_speeds, built = [], dict(oracle=[], a9=[])
    for i, record in enumerate(val):
        ref = at_rate(load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"]), FPS)
        points, valid = keypoints(ref)
        steps = side_steps(points, valid)
        real_speeds.append({n: (float(sp[ok].mean()) if ok.any() else None) for n, (sp, ok) in steps.items()})
        real_rows.append(side_read(points, valid))
        words = words1[i]
        if not words:
            clips["oracle"].append(None)
            clips["a9"].append(None)
            real_row = motion_rows([ref], [ref], FPS)[0]
            for k in operating:
                operating[k].append(None)
                pooled[k].append((None, real_row["wrist_speed"], None, real_row["wrist_spread"]))
            continue
        seconds = float(ref.timestamps[-1] - ref.timestamps[0]) + 1 / FPS
        plan_words = r2.plan_words(words, model, vocab, rows=details[i])
        job = dict(mode="oracle_stage2", words=list(words), block="matched", weights=plan_words["weights"])
        plan = cg.fitted_plan(job["weights"], cg.gap_ratio(model["timing"]), seconds)
        records = [choose_record(pools, i, w, "matched") for w in words]
        material = r2.oracle_material(job, i, pools, loaded={}, load=load, rule=rule, records=records, **mode)
        state = render(material["parts"], plan, sample_id=f"a2-oracle:{i}")
        clips["oracle"].append(read_clip(material, plan, state, records=records))
        built["oracle"].append((stored["oracle"].get(record["sign"]), motion_rows([state], [ref], FPS)[0]["wrist_speed"]))
        for w in words:
            if w not in cached_states:
                pick = index[w]["chosen"]["16"]
                cached_states[w] = load_state(store, pick["state_key"], pick["state_sha256"])
        job9 = dict(job, mode="core_crossfade")
        material = cached_material(job9, cached_states, cores=cores, rule=rule)
        state = render(material["parts"], plan, sample_id=f"a2-a9:{i}")
        clips["a9"].append(read_clip(material, plan, state, annotations=annotations))
        built["a9"].append((stored["a9"].get(record["sign"]), motion_rows([state], [ref], FPS)[0]["wrist_speed"]))
        predicted = predicted_seconds(duration, record["text"])
        plan_p = cg.fitted_plan(job["weights"], cg.gap_ratio(model["timing"]), predicted)
        flags = [annotations.get(w) for w in words]

        def at_point(name, state_p, plan_used, hold_info=None):
            row = motion_rows([state_p], [ref], FPS)[0]
            real_row = motion_rows([ref], [ref], FPS)[0]
            operating[name].append(read_operating(state_p, plan_used, flags, hold_info))
            pooled[name].append((row["wrist_speed"], real_row["wrist_speed"], row["wrist_spread"], real_row["wrist_spread"]))
        at_point("a9_predicted_margin01", render(material["parts"], plan_p, sample_id=f"a2-p:{i}"), plan_p)
        material0 = cached_material(job9, cached_states, cores=cores0, rule=rule0)
        state0 = render(material0["parts"], plan_p, sample_id=f"a2-p0:{i}")
        at_point("a9_predicted_margin0", state0, plan_p)
        held, info = hold_left_arm(state0, plan_p, flags, ready=stats["ready_pose"], cap=stats["left_speed_p99"])
        at_point("hold", held, plan_p, info)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(val)} clips", flush=True)
    out = dict(kind="comp-a2-left-arm", report_only=True, clips=len(val), set=dict(key=D20_KEY, sha256=D20_SHA),
               cache=dict(key=CACHE_KEY, sha256=CACHE_SHA), note="no hypothesis was read; motion of stored words and clips only",
               rebuild_check={k: rebuild_check([a for a, _ in v], [b for _, b in v]) for k, v in built.items()},
               forms={k: form_report(v, real_speeds) for k, v in clips.items()})
    for name, rows in operating.items():
        out["forms"][name] = operating_report(rows, real_rows, pooled[name])
    out["how2sign_train"] = {k: stats[k] for k in ("clips", "window_frames", "right_over_left_clip_speed", "left_speed_p99")}
    out["how2sign_train"]["left"] = stats["sides"]["left"]
    out["kept_from_two_definitions"] = "joins and regions share comp_a_lib.bridge_kind"
    data = json.dumps(out, indent=1).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
