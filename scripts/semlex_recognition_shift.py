"""Why does recogniser v2 read converted Sem-Lex poorly? Instrument shift (the adapter's estimator) against data shift.

Stage ``citizen-extract`` (local CPU): for ASL Citizen validation clips (recognition use only; the validation
partition, as ``validate_sign_recognizer.py`` evaluates v2, so the test partition stays unread), run our extraction
(``extract_clip``, ROI mode, as the ASL Citizen corpus was extracted) and the Sem-Lex emulator of
``validate_semlex_adapter.py`` (same Tasks trackers, normalised image landmarks with z, float16) on the same video.

Stage ``citizen-score``: v2 at 24 fps on, per clip,

- ``stored``: the corpus state through ``load_sample`` (usable-channel masking): exactly what v2's validation read;
- ``ours``: our extraction re-solved here, ``canonical_from_landmarks(upright_chest=True)`` (the corpus settings);
- ``ours+adapter_flags``: the same landmarks with the adapter's options (upright root, elbow gate);
- ``adapter``: the adapter on the emulated array (estimated aspect), and ``adapter_true_size``;
- hybrids that swap channel groups between ``ours`` and ``adapter`` (hands, body, face/head), to locate the shift;

top-1/top-5 with bootstrap 95% intervals over clips and over signers, and paired differences against ``ours``.

With ``--regressors`` (learned image-to-world models, ``world_regressor``), also the learned path:

- ``learned_gate``: regressed world points, the pose model's visibility head, solved with the upright chest only
  (the flags ``ours`` uses). **This is the gate path** (declared before it was scored, 2026-10-28): its paired top-5
  difference against ``ours`` must have a 95% bootstrap lower bound above -0.05 (docs/NEXT_PHASE_PLAN.md, 08:21 UTC);
- ``learned_inside_visibility``: the same with the inside-image visibility rule;
- ``learned_adapter_flags``: regressed points with the adapter's flags (upright root, elbow gate, inside-image);
- swaps: ``ours_with_learned_hands``, ``ours_with_learned_body``, and ``ours+adapter_flags`` (flags only).

Stage ``citizen-flags``: our extraction's landmarks on the same clips (MediaPipe's own visibility) solved under each
flag set separately: (a) upright chest only (the corpus setting, the reference), (b) + upright root, (c) + elbow gate,
(d) + both, (e) the inside-image visibility rule instead of MediaPipe's visibility, no other flag. v2 top-1/top-5 with
paired differences against (a), and per body joint the frames each variant removes, adds or changes against (a).

Stage ``gate-v2`` (declared in docs/NEXT_PHASE_PLAN.md, 2026-10-28 09:30 UTC): legacy Holistic arrays of the same
clips (``legacy_holistic_arrays.py``, full frames) converted through ``--regressors`` under the generator convention
(upright chest, upright root, learned visibility, no elbow gate); v2's top-5 paired difference must have its 95%
lower bound above -0.05 against (a) our path under upright chest + upright root and (b) our path with upright chest
only. Also the hands-only and body-only swaps into (a).

Stage ``semlex``: on Sem-Lex sample clips, (a) the label check: whether each clip's ASL-LEX label is the ASL-LEX
entry behind v2's gloss (ASL Citizen's ``asl_lex_code``), and whether the Sem-Lex row's phonology columns equal ASL-LEX
2.0's for that entry, on 30 top-5 misses (and 30 hits as a control); (b) mirror bias: each signer's dominant side
(the hand tracked on more frames and the faster-moving wrist, summed over their clips) and v2's accuracy by side,
as stored and mirrored.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

CITIZEN_MANIFEST = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json"
CITIZEN_MANIFEST_SHA256 = "708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6"
SEMLEX_METADATA_SHA256 = "f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3"
HANDS = ("left_hand", "right_hand")
FACE_HEAD = ("face", "blink", "head_rotation", "head_translation")
# Sem-Lex phonology column -> ASL-LEX 2.0 column
PHONOLOGY = {"Handshape": "Handshape.2.0", "Selected Fingers": "SelectedFingers.2.0", "Flexion": "Flexion.2.0",
             "Flexion Change": "FlexionChange.2.0", "Spread": "Spread.2.0", "Spread Change": "SpreadChange.2.0",
             "Thumb Position": "ThumbPosition.2.0", "Thumb Contact": "ThumbContact.2.0", "Sign Type": "SignType.2.0",
             "Path Movement": "Movement.2.0", "Repeated Movement": "RepeatedMovement.2.0",
             "Major Location": "MajorLocation.2.0", "Minor Location": "MinorLocation.2.0",
             "Second Minor Location": "SecondMinorLocation.2.0", "Contact": "Contact.2.0",
             "Nondominant Handshape": "NonDominantHandshape.2.0", "Wrist Twist": "UlnarRotation.2.0",
             "SignBank Reference ID": "SignBankReferenceID"}


def bootstrap(hits, groups=None, draws: int = 2000, seed: int = 0):
    hits = np.asarray(hits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if groups is None:
        means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(draws)]
    else:
        groups = np.asarray(groups)
        index = {k: np.flatnonzero(groups == k) for k in np.unique(groups)}
        keys = list(index)
        means = [hits[np.concatenate([index[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(draws)]
    return [round(float(v), 4) for v in np.percentile(means, [2.5, 97.5])]


def ranks_of(model, states, labels, *, with_top5: bool = False):
    """Rank of each clip's true gloss under v2 (0 = top-1), and optionally its top-5 class indices."""
    import torch
    from validate_sign_recognizer import logits_of
    logits = logits_of(model, states, torch.device("cpu"))
    y = torch.as_tensor(labels)
    ranks = (logits > logits.gather(1, y[:, None])).sum(1).numpy()
    return (ranks, logits.topk(5, dim=1).indices.numpy()) if with_top5 else ranks


def score(ranks: np.ndarray, groups, reference: np.ndarray | None = None) -> dict:
    out = {}
    for k in (1, 5):
        hits = (ranks < k).astype(np.float64)
        out[f"top{k}"] = round(float(hits.mean()), 4)
        out[f"top{k}_ci_clips"] = bootstrap(hits)
        out[f"top{k}_ci_signers"] = bootstrap(hits, groups) if len(set(groups)) >= 3 else None
        if reference is not None:
            diff = hits - (reference < k)
            out[f"top{k}_minus_ours"] = round(float(diff.mean()), 4)
            out[f"top{k}_minus_ours_ci_clips"] = bootstrap(diff)
    return out


def swap(base, donor, names):
    """``base`` with the channels ``names`` taken from ``donor`` (same frames)."""
    from dataclasses import replace
    if len(base.timestamps) != len(donor.timestamps):
        raise ValueError("states must share frames")
    return replace(base, channels={n: (donor.channels[n] if n in names else c) for n, c in base.channels.items()})


# ----------------------------------------------------------------------------------------------------------------------

def extract_one(job):
    video, out_dir, models_dir = job
    from signtranslator.data_engineering.video_landmarks import extract_clip, verify_models
    from validate_semlex_adapter import CLIP_FIELDS, emulate_holistic
    target = Path(out_dir) / f"{Path(video).stem}.npz"
    if target.exists():
        return dict(name=target.stem, status="present")
    models = verify_models(Path(models_dir))
    try:
        clip = extract_clip(Path(video), models, roi=True)
        array, times, size = emulate_holistic(Path(video), models)
    except (ValueError, OSError) as error:
        return dict(name=target.stem, status="refused", reason=str(error)[:200])
    meta = dict(name=target.stem, size=list(size), names=list(clip.blendshape_names))
    np.savez_compressed(target, meta=np.array(json.dumps(meta)), holistic=array, holistic_times=times,
                        **{f"ref_{k}": np.asarray(getattr(clip, k)) for k in CLIP_FIELDS})
    return dict(name=target.stem, status="extracted", frames=len(times))


def citizen_score(args) -> dict:
    import tempfile
    import torch
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, convert_array, load_face_resources
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    from signtranslator.pose.resample import resample_motion
    from train_text_to_motion import make_store
    from validate_semlex_adapter import load_npz
    from validate_sign_recognizer import load_model
    torch.set_num_threads(1)
    store = make_store("r2")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(CITIZEN_MANIFEST) + b"\n")
        manifest = load_manifest(path, expected_sha256=CITIZEN_MANIFEST_SHA256)
    records = {r["sign"]: r for r in manifest["records"]}
    model, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    blend = BlendshapeModel(resources)
    regressors = None
    if args.regressors:
        from signtranslator.data_engineering.world_regressor import load_regressors
        regressors = load_regressors(Path(args.regressors), expected_sha256=args.regressors_sha256)
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 2 * 2 ** 30)
    kw = dict(source_id="x", source_sha256="0" * 64, clock_id="x", inference_method="x")
    variants = collections.defaultdict(list)
    labels, signers, names, refused, reproduction = [], [], [], [], []
    for path in sorted(Path(args.out_dir).glob("*.npz")):
        meta, clip, hol = load_npz(path)
        record = records.get(meta["name"])
        if record is None or record.get("partition") != "val" or record["gloss"]["label"] not in index:
            continue
        duration = len(hol) * 1000. * float(np.median(np.diff(clip.times)))
        try:
            adapter, _ = convert_array(hol, duration, resources, video_id=meta["name"], source_sha256="0" * 64,
                                       blendshape_model=blend)
            adapter_true, _ = convert_array(hol, duration, resources, video_id=meta["name"], source_sha256="0" * 64,
                                            blendshape_model=blend, image_size=tuple(meta["size"]))
        except Refusal as error:
            refused.append((meta["name"], str(error)))
            continue
        ours, _ = canonical_from_landmarks(clip, sample_id=meta["name"], upright_chest=True, **kw)
        flags, _ = canonical_from_landmarks(clip, sample_id=meta["name"], upright_chest=True, upright_root=True,
                                            elbow_in_image=True, **kw)
        stored = load_sample(record, "continuous", cache=cache, fetch=store.get)
        # does the local re-extraction reproduce the stored corpus state? (frames and body/hand validity)
        sb, ob = stored.channels, ours.channels
        same_frames = len(stored.timestamps) == len(ours.timestamps)
        reproduction.append(dict(same_frames=same_frames, **({n: float(np.abs(
            np.where(sb[n].valid[..., None], sb[n].values, 0) - np.where(sb[n].valid[..., None], ob[n].values, 0)).max())
            for n in ("body", "left_hand", "right_hand", "head_rotation") if sb[n].valid.any()} if same_frames else {})))
        learned = {}
        if regressors is not None:
            common = dict(video_id=meta["name"], source_sha256="0" * 64, blendshape_model=blend, regressors=regressors)
            learned["learned_gate"] = convert_array(hol, duration, resources, learned_visibility=True,
                                                    flags="upright-chest", **common)[0]
            learned["learned_inside_visibility"] = convert_array(hol, duration, resources, flags="upright-chest",
                                                                 **common)[0]
            learned["learned_adapter_flags"] = convert_array(hol, duration, resources, flags="adapter", **common)[0]
            learned["ours_with_learned_hands"] = swap(ours, learned["learned_gate"], HANDS)
            learned["ours_with_learned_body"] = swap(ours, learned["learned_gate"], ("body",))
            learned["learned_gate_with_our_hands"] = swap(learned["learned_gate"], ours, HANDS)
        states = {**learned, "stored": stored, "ours": ours, "ours+adapter_flags": flags, "adapter": adapter,
                  "adapter_true_size": adapter_true,
                  "ours_with_adapter_hands": swap(ours, adapter, HANDS),
                  "adapter_with_our_hands": swap(adapter, ours, HANDS),
                  "ours_with_adapter_body": swap(ours, adapter, ("body",)),
                  "ours_with_adapter_face_head": swap(ours, adapter, FACE_HEAD)}
        for name, state in states.items():
            variants[name].append(resample_motion(state, 24.))
        labels.append(index[record["gloss"]["label"]])
        signers.append(record["performer_sha256"][:12])
        names.append(meta["name"])
    out = dict(run=args.recognizer_run, checkpoint=best["checkpoint_key"], clips=len(labels),
               signers=len(set(signers)), glosses=len(set(labels)), refused=refused, partition="val")
    agree = [r for r in reproduction if r["same_frames"]]
    out["reproduction_of_stored_states"] = dict(
        same_frame_count=len(agree), of=len(reproduction),
        max_abs_6d_difference_median={n: round(float(np.median([r[n] for r in agree if n in r])), 6)
                                      for n in ("body", "left_hand", "right_hand", "head_rotation")
                                      if any(n in r for r in agree)})
    ranks = {name: ranks_of(model, states, labels) for name, states in variants.items()}
    out["scores"] = {name: score(r, signers, None if name == "ours" else ranks["ours"]) for name, r in ranks.items()}
    out["per_signer_top5"] = {s: {name: round(float((r[np.array(signers) == s] < 5).mean()), 4) for name, r in ranks.items()}
                              for s in sorted(set(signers))}
    out["clip_names"] = names
    if regressors is not None:
        gate = out["scores"]["learned_gate"]
        out["gate"] = dict(path="learned_gate", rule="top-5 paired difference vs ours: 95% lower bound > -0.05",
                           difference=gate["top5_minus_ours"], ci=gate["top5_minus_ours_ci_clips"],
                           passed=bool(gate["top5_minus_ours_ci_clips"][0] > -.05), regressors_sha256=regressors.sha256)
    return out


FLAG_SETS = {
    "a_upright_chest": dict(upright_chest=True),
    "b_plus_upright_root": dict(upright_chest=True, upright_root=True),
    "c_plus_elbow_in_image": dict(upright_chest=True, elbow_in_image=True),
    "d_plus_both": dict(upright_chest=True, upright_root=True, elbow_in_image=True),
    "e_inside_image_visibility": dict(upright_chest=True),
}


def inside_image_visibility(clip):
    """The clip with MediaPipe's pose visibility replaced by the adapter's rule (finite and inside the image)."""
    from dataclasses import replace
    xy = np.asarray(clip.pose_image)
    inside = np.isfinite(xy).all(-1) & (xy >= 0).all(-1) & (xy <= 1).all(-1) & np.asarray(clip.pose_found)[:, None]
    return replace(clip, pose_visibility=inside.astype(np.float64))


def flag_states(clip, name: str):
    """Canonical states of one clip under every flag set of ``FLAG_SETS``."""
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    kw = dict(source_id="x", source_sha256="0" * 64, clock_id="x", inference_method="x")
    out = {}
    for variant, flags in FLAG_SETS.items():
        source = inside_image_visibility(clip) if variant.startswith("e_") else clip
        out[variant] = canonical_from_landmarks(source, sample_id=name, **flags, **kw)[0]
    return out


def body_changes(reference, other) -> dict:
    """Per body joint: frames valid in ``reference`` but not ``other`` (removed), the reverse (added), and frames valid in
    both whose stored rotation differs by more than 1e-6 (changed)."""
    r, o = reference.channels["body"], other.channels["body"]
    both = r.valid & o.valid
    changed = both & (np.abs(r.values - o.values).max(-1) > 1e-6)
    return {label: dict(reference=int(r.valid[:, j].sum()), removed=int((r.valid[:, j] & ~o.valid[:, j]).sum()),
                        added=int((o.valid[:, j] & ~r.valid[:, j]).sum()), changed=int(changed[:, j].sum()))
            for j, label in enumerate(r.labels)}


def citizen_flags(args) -> dict:
    import tempfile
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.pose.resample import resample_motion
    from train_text_to_motion import make_store
    from validate_semlex_adapter import load_npz
    from validate_sign_recognizer import load_model
    torch.set_num_threads(1)
    store = make_store("r2")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(CITIZEN_MANIFEST) + b"\n")
        manifest = load_manifest(path, expected_sha256=CITIZEN_MANIFEST_SHA256)
    records = {r["sign"]: r for r in manifest["records"]}
    model, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    variants = {name: [] for name in FLAG_SETS}
    changes = {name: {} for name in FLAG_SETS if name != "a_upright_chest"}
    labels, signers = [], []
    for path in sorted(Path(args.out_dir).glob("*.npz")):
        meta, clip, _ = load_npz(path)
        record = records.get(meta["name"])
        if record is None or record.get("partition") != "val" or record["gloss"]["label"] not in index:
            continue
        states = flag_states(clip, meta["name"])
        for name, state in states.items():
            variants[name].append(resample_motion(state, 24.))
            if name in changes:
                for label, c in body_changes(states["a_upright_chest"], state).items():
                    total = changes[name].setdefault(label, dict(reference=0, removed=0, added=0, changed=0))
                    for k, v in c.items():
                        total[k] += v
        labels.append(index[record["gloss"]["label"]])
        signers.append(record["performer_sha256"][:12])
    ranks = {name: ranks_of(model, states, labels) for name, states in variants.items()}
    reference = ranks["a_upright_chest"]
    arms = ("LeftUpperArm", "LeftForearm", "LeftHand", "RightUpperArm", "RightForearm", "RightHand", "Pelvis", "Chest")
    return dict(run=args.recognizer_run, checkpoint=best["checkpoint_key"], clips=len(labels), signers=len(set(signers)),
                frames_at_24fps=int(sum(len(s.timestamps) for s in variants["a_upright_chest"])),
                scores={name: score(r, signers, None if name == "a_upright_chest" else reference)
                        for name, r in ranks.items()},
                body_changes_against_a_source_frames={name: {j: c[j] for j in arms} for name, c in changes.items()})


def gate_v2(args) -> dict:
    import tempfile
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, convert_array, load_face_resources
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    from signtranslator.data_engineering.world_regressor import load_regressors
    from signtranslator.pose.resample import resample_motion
    from train_text_to_motion import make_store
    from validate_semlex_adapter import load_npz
    from validate_sign_recognizer import load_model
    torch.set_num_threads(1)
    store = make_store("r2")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(CITIZEN_MANIFEST) + b"\n")
        manifest = load_manifest(path, expected_sha256=CITIZEN_MANIFEST_SHA256)
    records = {r["sign"]: r for r in manifest["records"]}
    model, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    blend = BlendshapeModel(resources)
    regressors = load_regressors(Path(args.regressors), expected_sha256=args.regressors_sha256)
    kw = dict(source_id="x", source_sha256="0" * 64, clock_id="x", inference_method="x")
    variants = collections.defaultdict(list)
    labels, signers, refused, misaligned = [], [], [], []
    for path in sorted(Path(args.out_dir).glob("*.npz")):
        meta, clip, _ = load_npz(path)
        record = records.get(meta["name"])
        if record is None or record.get("partition") != "val" or record["gloss"]["label"] not in index:
            continue
        legacy_path = Path(args.holistic_dir) / f"{meta['name']}.npz"
        with np.load(legacy_path, allow_pickle=False) as z:
            holistic = z["holistic"].copy()
        if len(holistic) != len(clip.times):
            misaligned.append(meta["name"])
            continue
        duration = len(holistic) * 1000. * float(np.median(np.diff(clip.times)))
        try:
            learned, _ = convert_array(holistic, duration, resources, video_id=meta["name"], source_sha256="0" * 64,
                                       blendshape_model=blend, regressors=regressors, learned_visibility=True,
                                       flags="generator")
        except Refusal as error:
            refused.append((meta["name"], str(error)))
            continue
        ours_a, _ = canonical_from_landmarks(clip, sample_id=meta["name"], upright_chest=True, upright_root=True, **kw)
        ours_b, _ = canonical_from_landmarks(clip, sample_id=meta["name"], upright_chest=True, **kw)
        states = {"learned_legacy_generator": learned, "ours_a_chest_root": ours_a, "ours_b_chest_only": ours_b,
                  "ours_a_with_learned_hands": swap(ours_a, learned, HANDS),
                  "ours_a_with_learned_body": swap(ours_a, learned, ("body",))}
        for name, state in states.items():
            variants[name].append(resample_motion(state, 24.))
        labels.append(index[record["gloss"]["label"]])
        signers.append(record["performer_sha256"][:12])
    ranks = {name: ranks_of(model, s, labels) for name, s in variants.items()}
    scores = {}
    for name, r in ranks.items():
        scores[name] = dict(score(r, signers),
                            vs_a=score(r, signers, ranks["ours_a_chest_root"]),
                            vs_b=score(r, signers, ranks["ours_b_chest_only"]))
    gate = scores["learned_legacy_generator"]
    passed_a = bool(gate["vs_a"]["top5_minus_ours_ci_clips"][0] > -.05)
    passed_b = bool(gate["vs_b"]["top5_minus_ours_ci_clips"][0] > -.05)
    return dict(run=args.recognizer_run, checkpoint=best["checkpoint_key"], clips=len(labels),
                signers=len(set(signers)), regressors_sha256=regressors.sha256, refused=refused, misaligned=misaligned,
                scores=scores,
                gate=dict(rule="v2 top-5 paired difference: 95% lower bound > -0.05 against (a) and (b)",
                          against_a=dict(difference=gate["vs_a"]["top5_minus_ours"],
                                         ci=gate["vs_a"]["top5_minus_ours_ci_clips"], passed=passed_a),
                          against_b=dict(difference=gate["vs_b"]["top5_minus_ours"],
                                         ci=gate["vs_b"]["top5_minus_ours_ci_clips"], passed=passed_b),
                          passed=passed_a and passed_b, regressors_sha256=regressors.sha256))


# ----------------------------------------------------------------------------------------------------------------------

def norm(value: str) -> str:
    """Phonology values as comparable text: numbers as integers, words lower case without spaces ("Fully Open" and
    "FullyOpen" are the same value), and empty equal to ASL-LEX's "NA"."""
    value = (value or "").strip()
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        value = value.lower().replace(" ", "")
        return "" if value == "na" else value


def semlex_stage(args) -> dict:
    import tempfile
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, convert_array, load_face_resources
    from signtranslator.pose.mirror import mirror_state
    from signtranslator.pose.resample import resample_motion
    from train_text_to_motion import make_store
    from validate_sign_recognizer import load_model
    torch.set_num_threads(1)
    store = make_store("r2")
    payload = Path(args.metadata).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SEMLEX_METADATA_SHA256:
        raise SystemExit("metadata CSV is not the pinned acquisition")
    meta = {r["video_id"]: r for r in csv.DictReader(io.StringIO(payload.decode("utf-8")))}
    asllex_bytes = store.get(args.asllex_key)
    asllex = {r["EntryID"]: r for r in csv.DictReader(io.StringIO(asllex_bytes.decode("utf-8", "replace")))}
    by_code = {r["Code"]: r for r in asllex.values()}
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(CITIZEN_MANIFEST) + b"\n")
        manifest = load_manifest(path, expected_sha256=CITIZEN_MANIFEST_SHA256)
    gloss_code, code_gloss = {}, collections.defaultdict(set)
    for r in manifest["records"]:
        gloss_code.setdefault(r["gloss"]["label"], r["gloss"].get("asl_lex_code"))
        if r["gloss"].get("asl_lex_code"):
            code_gloss[r["gloss"]["asl_lex_code"]].add(r["gloss"]["label"])
    model, glosses, _ = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    blend = BlendshapeModel(resources)
    learned_kw = {}
    if args.regressors:
        from signtranslator.data_engineering.world_regressor import load_regressors
        learned_kw = dict(regressors=load_regressors(Path(args.regressors), expected_sha256=args.regressors_sha256),
                          learned_visibility=True, flags=args.semlex_flags)
    items = []
    for path in sorted(Path(args.npy_dir).glob("*.npy")):
        row = meta.get(path.stem)
        if row is None or row["label_type"] != "asllex":
            continue
        name_gloss = row["label"].upper().replace("_", "")
        entry = asllex.get(row["label"])
        candidates = sorted(code_gloss.get(entry["Code"], ())) if entry else []
        code_gloss_ = candidates[0] if len(candidates) == 1 else None             # ASL-LEX entry -> ASL Citizen gloss
        gloss = code_gloss_ if args.mapping == "code" else name_gloss
        if gloss is None or gloss not in index:
            continue
        data = path.read_bytes()
        array = np.load(io.BytesIO(data), allow_pickle=False)
        try:
            state, report = convert_array(array, float(row["duration"]), resources, video_id=path.stem,
                                          source_sha256=hashlib.sha256(data).hexdigest(), blendshape_model=blend,
                                          **learned_kw)
        except Refusal:
            continue
        # side activity: tracked hand frames and mean image speed of each wrist block while tracked (per second)
        activity = {}
        for side, block in (("Left", slice(511, 532)), ("Right", slice(532, 553))):
            wrist = array[:, block.start, :2].astype(np.float64)
            ok = np.isfinite(wrist).all(1)
            step = np.linalg.norm(np.diff(wrist, axis=0), axis=1)[ok[1:] & ok[:-1]]
            activity[side] = dict(frames=int(ok.sum()), path=float(step.sum()))
        items.append(dict(video_id=path.stem, row=row, gloss=gloss, y=index[gloss], state=resample_motion(state, 24.),
                          activity=activity, name_gloss=name_gloss, code_gloss=code_gloss_))
    labels = [it["y"] for it in items]
    real, top5 = ranks_of(model, [it["state"] for it in items], labels, with_top5=True)
    mirrored = ranks_of(model, [mirror_state(it["state"], sample_id=it["state"].sample_id) for it in items], labels)
    signers = [it["row"]["signer_id"] for it in items]
    from evaluate_text_to_motion import blanked
    blank = ranks_of(model, blanked([it["state"] for it in items]), labels)
    # mismatched control: each clip scored against another sample clip's gloss (a seeded shuffle, never its own gloss)
    shuffle = np.random.default_rng(args.seed)
    other = np.array(labels)
    for _ in range(100):
        order = shuffle.permutation(len(labels))
        if not (np.array(labels)[order] == np.array(labels)).any():
            other = np.array(labels)[order]
            break
    else:
        raise RuntimeError("no derangement of the labels found")
    mismatched = ranks_of(model, [it["state"] for it in items], other.tolist())
    path_name = "learned_gate" if learned_kw else "adapter"
    out_controls = dict(path=path_name, blank=score(blank, signers), mismatched=score(mismatched, signers),
                        real_minus_blank_top5=dict(difference=round(float((real < 5).mean() - (blank < 5).mean()), 4),
                                                   ci=bootstrap((real < 5).astype(float) - (blank < 5))),
                        real_minus_mismatched_top5=dict(
                            difference=round(float((real < 5).mean() - (mismatched < 5).mean()), 4),
                            ci=bootstrap((real < 5).astype(float) - (mismatched < 5))))
    out = dict(mapping=args.mapping, controls=out_controls, flags=learned_kw.get("flags"),
               regressors_sha256=learned_kw["regressors"].sha256 if learned_kw else None, clips=len(items), signers=len(set(signers)), real=score(real, signers),
               mirrored=score(mirrored, signers),
               mapping_disagreements=sum(it["name_gloss"] != it["code_gloss"] for it in items),
               mapping_disagreement_examples=[(it["row"]["label"], it["name_gloss"], it["code_gloss"]) for it in items
                                              if it["name_gloss"] != it["code_gloss"]][:40])

    # (a) labels
    def label_check(it):
        row, gloss = it["row"], it["gloss"]
        code = gloss_code.get(gloss)
        entry = by_code.get(code)
        semlex_entry = asllex.get(row["label"])
        compared = {}
        if entry is not None:
            for s_col, a_col in PHONOLOGY.items():
                a, b = norm(row.get(s_col, "")), norm(entry.get(a_col, ""))
                if a or b:
                    compared[s_col] = dict(semlex=a, asllex=b, same=a == b)
        return dict(video_id=it["video_id"], label=row["label"], v2_gloss=gloss, v2_asl_lex_code=code,
                    v2_entry=entry["EntryID"] if entry else None, label_is_v2_entry=bool(entry and entry["EntryID"] == row["label"]),
                    label_in_asllex=semlex_entry is not None,
                    phonology_agreement=round(float(np.mean([c["same"] for c in compared.values()])), 4) if compared else None,
                    disagreements={k: v for k, v in compared.items() if not v["same"]})
    rng = random.Random(args.seed)
    misses = [i for i, r in enumerate(real) if r >= 5]
    hits = [i for i, r in enumerate(real) if r < 5]
    checked_miss = [label_check(items[i]) | dict(v2_top5=[glosses[g] for g in top5[i]])
                    for i in rng.sample(misses, min(30, len(misses)))]
    checked_hit = [label_check(items[i]) for i in rng.sample(hits, min(30, len(hits)))]
    every = [label_check(it) for it in items]
    def summary(rows):
        agree = [r["phonology_agreement"] for r in rows if r["phonology_agreement"] is not None]
        return dict(clips=len(rows), label_is_v2_entry=sum(r["label_is_v2_entry"] for r in rows),
                    v2_gloss_without_code=sum(r["v2_asl_lex_code"] is None for r in rows),
                    phonology_agreement_median=round(float(np.median(agree)), 4) if agree else None,
                    phonology_identical=sum(a == 1. for a in agree),
                    disagreeing_columns=dict(collections.Counter(k for r in rows for k in r["disagreements"])))
    out["labels"] = dict(misses=summary(checked_miss), hits=summary(checked_hit), all=summary(every),
                         missed_examples=checked_miss)

    # (b) mirror bias per signer
    per = collections.defaultdict(lambda: dict(Left=dict(frames=0, path=0.), Right=dict(frames=0, path=0.), clips=0))
    for it in items:
        p = per[it["row"]["signer_id"]]
        p["clips"] += 1
        for side in ("Left", "Right"):
            for k in ("frames", "path"):
                p[side][k] += it["activity"][side][k]
    dominance = {}
    for s, p in per.items():
        by_frames = "left" if p["Left"]["frames"] > p["Right"]["frames"] else "right"
        by_path = "left" if p["Left"]["path"] > p["Right"]["path"] else "right"
        dominance[s] = dict(clips=p["clips"], by_tracked_frames=by_frames, by_wrist_path=by_path,
                            left_frames=p["Left"]["frames"], right_frames=p["Right"]["frames"],
                            left_path=round(p["Left"]["path"], 2), right_path=round(p["Right"]["path"], 2))
    out["dominance_per_signer"] = dominance
    side = np.array([dominance[s]["by_wrist_path"] if dominance[s]["by_wrist_path"] == dominance[s]["by_tracked_frames"]
                     else "unclear" for s in signers])
    groups = np.array(signers)
    by_side = {}
    for d in ("left", "right", "unclear"):
        mask = side == d
        if mask.any():
            by_side[d] = dict(clips=int(mask.sum()), signers=sorted(set(groups[mask].tolist())),
                              real=score(real[mask], groups[mask]), mirrored=score(mirrored[mask], groups[mask]))
    out["by_dominant_side"] = by_side
    out["per_signer_top5"] = {s: dict(real=round(float((real[groups == s] < 5).mean()), 4),
                                      mirrored=round(float((mirrored[groups == s] < 5).mean()), 4),
                                      clips=int((groups == s).sum())) for s in sorted(set(signers))}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("citizen-extract", "citizen-score", "citizen-flags", "gate-v2", "semlex"))
    p.add_argument("--holistic-dir", help="gate-v2: legacy Holistic arrays (<clip>.npz) of the ASL Citizen clips")
    p.add_argument("--videos", nargs="*", default=[])
    p.add_argument("--out-dir")
    p.add_argument("--npy-dir")
    p.add_argument("--metadata")
    p.add_argument("--asllex-key", default="acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--regressors", help="world regressor bundle (world_regressor.WorldRegressors.save)")
    p.add_argument("--regressors-sha256")
    p.add_argument("--semlex-flags", default="generator", help="semlex stage, learned path: solver convention")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mapping", choices=("name", "code"), default="code",
                   help="Sem-Lex label -> v2 gloss: 'name' upper-cases the label without underscores (cat_3 -> CAT3); "
                        "'code' goes through the label's ASL-LEX Code to the ASL Citizen gloss with that code")
    p.add_argument("--report")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    if args.stage == "citizen-extract":
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(extract_one, [(v, args.out_dir, args.models_dir) for v in args.videos]):
                print(json.dumps(r), flush=True)
        return 0
    report = {"citizen-score": citizen_score, "citizen-flags": citizen_flags, "gate-v2": gate_v2,
              "semlex": semlex_stage}[args.stage](args)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k not in ("clip_names", "labels")}, indent=1)[:8000])
    if "labels" in report:
        print(json.dumps({k: v for k, v in report["labels"].items() if k != "missed_examples"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
