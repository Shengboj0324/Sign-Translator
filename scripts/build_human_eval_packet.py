"""Build the blinded review packet for Deaf ASL reviewers (plan item 3b; docs/HUMAN_EVALUATION_PROTOCOL.md).

Utterances come from a plan manifest (``build_asllrp_plan_manifest.py``: English, plan, recorded motion). Every
utterance is produced by every ``--system``:

- ``LABEL=reference``: the recorded motion (video estimate) - the human upper reference through the same renderer;
- ``LABEL=plan:KEY@SHA``: a plan-conditioned generator checkpoint ``KEY`` (trained on manifest ``SHA``), conditioned on
  the utterance's plan line - the planner's gold plan, so this scores plan -> motion alone;
- ``LABEL=english:KEY@SHA``: a generator conditioned on the English sentence (the pre-plan baseline);
- ``LABEL=translate:KEY@SHA``: the full chain - the gloss planner (``--planner-run``) plans the English, the generator
  signs the predicted plan;
- ``LABEL=stitch:LEXICON_KEY@SHA``: the citation-stitching baseline (``generation.citation_stitch``) on the gold plan.

Every system signs for the recorded utterance's duration at ``--fps`` (24, the generators' training rate; the reference
is resampled to it), so length is no cue. Only utterances whose recorded hands are well tracked
(``--min-hand-coverage``) are used, so the reference is a fair upper bound (ASLLRP val/test hands are often lost). All videos are
drawn the same way: the four-panel skeleton preview plus the face panel (``face_preview``), with invalid frames of any
channel held from the nearest valid frame (for display only), so tracking dropouts do not reveal the recorded
reference.

The design (``eval_framework.human_study``) is a Latin square over reviewer groups; outputs, under ``--output-dir``
and copied to ``--output-prefix`` in the store:

- ``packet/group-<g>/sheet.csv`` + ``<item>.mp4`` + the reviewer guide (+ ``group-<g>.zip``): what group ``g``'s
  reviewers receive - no system names, no correct answers;
- ``key/key.json``: the private key (systems, correct options, grammatical truth, seed). Never sent to reviewers.

The ``test`` partition is the final study's and requires ``--final-study``; pilots use ``val``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import secrets
import shutil
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.eval_framework.human_study import design_study, key_sha256, write_sheet  # noqa: E402

GUIDE = ROOT / "docs" / "HUMAN_EVALUATION_REVIEWER_GUIDE.md"   # what reviewers read (the protocol stays with the team)


def parse_system(spec: str) -> dict:
    label, _, rest = spec.partition("=")
    if not label or not rest:
        raise ValueError(f"--system {spec!r}: expected LABEL=reference or LABEL=<plan|english|translate>:KEY@SHA")
    if rest == "reference":
        return dict(label=label, kind="reference")
    kind, _, target = rest.partition(":")
    key, _, sha = target.rpartition("@")
    if kind not in ("plan", "english", "translate", "stitch") or not key or len(sha) != 64:
        raise ValueError(f"--system {spec!r}: expected LABEL=<plan|english|translate|stitch>:KEY@<64-hex SHA>")
    return dict(label=label, kind=kind, key=key, manifest_sha256=sha)


def held_for_display(state):
    """Every channel's invalid frames filled from the nearest valid frame (forward, then backward); a channel with no
    valid frame keeps its values. For drawing only - the result is never a training or scoring input."""
    channels = {}
    for name, c in state.channels.items():
        values, valid = c.values.copy(), c.valid.copy()
        for j in range(values.shape[1]):
            ok = valid[:, j].astype(bool)
            if ok.all() or not ok.any():
                continue
            idx = np.where(ok, np.arange(len(ok)), -1)
            idx = np.maximum.accumulate(idx)                       # last valid frame at or before t
            first = int(np.argmax(ok))
            idx[idx < 0] = first
            values[:, j] = values[idx, j]
            valid[:, j] = True
        channels[name] = replace(c, values=values, valid=valid)
    return replace(state, channels=channels)


def render_avatar_video(state, path: Path, *, resolution: int = 384) -> int:
    """The MakeHuman avatar render (``render_avatar.py``: Blender Workbench, front camera) of ``state`` to ``path``."""
    import render_avatar
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "state.npz"
        held_for_display(state).save(npz)
        out = Path(tmp) / "render"
        code = render_avatar.main(["--state", str(npz), "--out-dir", str(out), "--allow-inferred",
                                   "--resolution", str(resolution)])
        if code != 0 or not (out / "avatar.mp4").exists():
            raise RuntimeError(f"avatar render failed ({code}); see {out}")
        shutil.copy(out / "avatar.mp4", path)
    return len(state.timestamps)


def render(state, path: Path, *, panel: int) -> int:
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from signtranslator.generation.face_preview import with_face_panel
    from signtranslator.generation.skeleton_preview import encode_preview, render_preview_frames
    shown = held_for_display(state)
    joints, positions, bones = skeleton_positions(shown)
    frames = with_face_panel(render_preview_frames(positions, joints, bones, panel=panel, bounds_quantile=.02), shown)
    clock = np.asarray(state.timestamps[:len(frames)]) - state.timestamps[0]
    encode_preview(frames, clock, path)
    return len(frames)


def hand_coverage(state) -> float:
    """Fraction of frames on which the better-tracked hand is valid (any joint)."""
    return max(float(state.channels[h].valid.any(axis=1).mean()) for h in ("left_hand", "right_hand"))


def choose_utterances(records, partition: str, count: int | None, min_seconds: float, rng: random.Random) -> list[dict]:
    """``count`` (all, when None: in random order) records of ``partition`` with an English sentence, a plan and a
    state, distinct sentences."""
    pool, seen = [], set()
    for r in sorted(records, key=lambda r: r["sign"]):
        english = (r.get("english") or "").strip()
        if (r.get("partition") != partition or not english or not r.get("plan") or not r.get("text")
                or (r.get("continuous") or {}).get("seconds", 0) < min_seconds):
            continue
        if english.lower() in seen:
            continue
        seen.add(english.lower()); pool.append(r)
    rng.shuffle(pool)
    if count is None:
        return pool
    if len(pool) < count:
        raise SystemExit(f"only {len(pool)} usable utterances in {partition}")
    return sorted(pool[:count], key=lambda r: r["sign"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True, help="plan manifest")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--system", action="append", required=True, help="LABEL=reference | LABEL=<kind>:KEY@SHA")
    parser.add_argument("--planner-run", help="gloss planner run, for translate: systems")
    parser.add_argument("--partition", default="val")
    parser.add_argument("--final-study", action="store_true", help="required to read the test partition")
    parser.add_argument("--count", type=int, default=60, help="utterances (each signed by every system)")
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, help="private design seed; drawn at random and kept in the key if omitted")
    parser.add_argument("--fps", type=float, default=24., help="every system signs, and the reference is shown, at this "
                                                                 "rate (the generators' training rate)")
    parser.add_argument("--min-hand-coverage", type=float, default=.7,
                        help="keep utterances whose better-tracked hand is valid on at least this fraction of the "
                             "recording, so the reference is a fair upper bound")
    parser.add_argument("--guidance", type=float, default=2.)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--panel", type=int, default=240)
    parser.add_argument("--renderer", choices=("skeleton", "avatar"), default="skeleton",
                        help="skeleton + face panel, or the MakeHuman avatar (Blender; slower)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", help="store prefix (private) to copy the packet and key to")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.partition == "test" and not args.final_study:
        raise SystemExit("the test partition is the final study's: pass --final-study")
    systems = [parse_system(s) for s in args.system]
    if len({s["label"] for s in systems}) != len(systems):
        raise SystemExit("system labels must be distinct")
    if any(s["kind"] == "translate" for s in systems) and not args.planner_run:
        raise SystemExit("translate: systems need --planner-run")
    from train_text_to_motion import make_store
    from evaluate_text_to_motion import at_rate, load_generator, load_state
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    candidates = choose_utterances(manifest["records"], args.partition, None, args.min_seconds, random.Random(seed))
    chosen, references = [], {}
    for r in candidates:                                       # in the seeded random order, until enough are kept
        state = load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"])
        if hand_coverage(state) >= args.min_hand_coverage:
            chosen.append(r); references[r["sign"]] = at_rate(state, args.fps)
        if len(chosen) == args.count:
            break
    if len(chosen) < args.count:
        raise SystemExit(f"only {len(chosen)} utterances reach hand coverage {args.min_hand_coverage}")
    chosen.sort(key=lambda r: r["sign"])
    key = design_study([dict(sign=r["sign"], english=r["english"], markers=r["plan"]["markers"]) for r in chosen],
                       [s["label"] for s in systems], seed=seed,
                       distractor_pool=sorted({(r.get("english") or "").strip() for r in manifest["records"]
                                               if r.get("partition") == args.partition and (r.get("english") or "").strip()}))
    key.update(manifest_key=args.manifest_key, manifest_sha256=args.manifest_sha256, partition=args.partition,
               system_specs=systems, planner_run=args.planner_run, guidance=args.guidance, steps=args.steps,
               fps=args.fps, min_hand_coverage=args.min_hand_coverage)
    from signtranslator.generation.citation_stitch import load_stitcher
    samplers = {s["label"]: (load_stitcher(store, s["key"], s["manifest_sha256"]).sample if s["kind"] == "stitch" else
                             load_generator(store, s["key"], s["manifest_sha256"], args.device)[2])
                for s in systems if s["kind"] != "reference"}
    planned = {}
    if args.planner_run:
        from translate_english import load_planner, plan_sentences
        model, words, glosses, best = load_planner(store, args.planner_run, args.device)
        lines = plan_sentences(model, words, glosses, [r["english"] for r in chosen], device=args.device)
        planned = {r["sign"]: p["line"] for r, p in zip(chosen, lines)}
        key["planner_checkpoint"] = best["checkpoint_key"]
    spec = {s["label"]: s for s in systems}
    by_sign = {r["sign"]: r for r in chosen}
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "key").mkdir(parents=True, exist_ok=True)
    for n, item in enumerate(key["items"]):
        record, system = by_sign[item["sign"]], spec[item["system"]]
        reference = references[item["sign"]]
        fps = args.fps
        seconds = len(reference.timestamps) / fps
        condition = dict(plan=record["text"], english=record["english"], translate=planned.get(item["sign"]),
                         stitch=record["text"])
        if system["kind"] == "reference":
            state = reference
        else:
            text = condition[system["kind"]]
            item["condition"] = text
            state = samplers[system["label"]](text, fps, seconds, args.guidance, args.steps, seed + n)
        group_dir = out / "packet" / f"group-{item['group']}"
        group_dir.mkdir(parents=True, exist_ok=True)
        video = group_dir / f"{item['item_id']}.mp4"
        item["frames"] = (render_avatar_video(state, video) if args.renderer == "avatar" else
                          render(state, video, panel=args.panel))
        item["renderer"] = args.renderer
        item["seconds"] = round(seconds, 3)
        print(json.dumps(dict(n=n + 1, of=len(key["items"]), group=item["group"])), flush=True)
    for g in range(key["groups"]):
        group_dir = out / "packet" / f"group-{g}"
        (group_dir / "sheet.csv").write_text(write_sheet(key, g))
        shutil.copy(GUIDE, group_dir / GUIDE.name)
        with zipfile.ZipFile(out / "packet" / f"group-{g}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in sorted(group_dir.iterdir()):
                z.write(f, f"group-{g}/{f.name}")
    key["key_sha256"] = key_sha256(key)
    (out / "key" / "key.json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    if args.output_prefix:
        prefix = args.output_prefix.rstrip("/")
        for f in sorted(p for p in out.rglob("*") if p.is_file()):
            store.put(f"{prefix}/{f.relative_to(out).as_posix()}", f.read_bytes())
    print(json.dumps(dict(utterances=len(chosen), systems=[s["label"] for s in systems], groups=key["groups"],
                          items=len(key["items"]), key=str(out / "key" / "key.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
