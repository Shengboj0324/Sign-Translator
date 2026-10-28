"""Validate the Sem-Lex adapter (``semlex_pose``) on video we may use, against our own extraction and two-view truth.

**Emulated Holistic.** No legacy MediaPipe Holistic model is installed (MediaPipe 1.1.1 has no ``mp.solutions``; the
Tasks ``HolisticLandmarker`` needs a ``holistic_landmarker.task`` bundle that is not available offline). Sem-Lex-format
arrays are therefore *emulated*: the same Tasks landmarkers and ROI trackers as ``extract_clip`` run over every frame,
and their normalised image outputs (x, y and the models' relative z; crop outputs mapped back to the full image, z
scaled by crop width / image width as MediaPipe's landmark projection does) are stored as ``(T, 553, 3)`` float16 in
Sem-Lex's layout, NaN where a part is not detected. This isolates the **conversion** (pseudo-3D from image coordinates
instead of world landmarks, the inside-image visibility rule, the aspect estimate, the rebuilt face crop, float16).
It does **not** measure how legacy Holistic's models differ from the Tasks models.

**Clips and framing.** How2Sign val front clips (CC BY-NC 4.0), each cropped to Sem-Lex's measured webcam framing (nose
at y = 0.35 and shoulders at y = 0.67 of the crop height, centred on the shoulders) at 4:3 and 16:9 (Sem-Lex's two
frame shapes; a few also at 1:1), so the hips are out of frame and the elbows often are, as in Sem-Lex.

**Four versions of each clip.**
- ``AD``: the adapter on the emulated array of the crop (upright chest, upright root, elbow gate);
- ``MC``: our extraction (``extract_clip`` + ``canonical_from_landmarks``, same options) on the same crop;
- ``MF``: our extraction on the full, uncropped frame (hips in view; How2Sign's own monocular setting);
- ``TV``: the How2Sign two-view state (Sapiens two-view triangulation; on CMU Panoptic 2.95 deg median arm and 6-9 deg
  finger direction error against the dome, against 13.7 / 15-19 deg for MediaPipe monocular,
  DATA_QUALITY_DECISION.md), aligned by the front clip's timestamps.

**Comparisons.**
1. *Equivalence* (AD vs MC, stored local rotations, frames both valid): body joints, fingers, head rotation;
   blendshapes (MAE, Pearson r); validity agreement.
2. *Accuracy* against TV, in the camera-aligned y-up frame by forward kinematics of the stored rotations: upper-arm and
   forearm bone directions and the wrist (palm frame) orientation; fingers by their stored local rotations (rooted at
   the palm, so frame-free). On each method's own valid frames and on the frames AD, MC and TV all hold (common support).
   Also: AD's accuracy on the frames that only AD marks valid (the inside-image visibility rule admits more).
3. *Depth gain.* ``k`` for pose and hand z is chosen on a calibration half (source videos by CRC32 parity) by AD's median
   error against TV; every reported accuracy is on the held-out half.
4. *Aspect.* The face-mesh estimate against the true crop shape on every clip; the held-out comparisons are run with
   the true frame size and with the estimated aspect (the path Sem-Lex takes).

**Acceptance bar** (declared before the full run; the first smoke run, on 2 clips, showed that the in-crop world
landmarks of our own extraction tilt the shoulder line by ~15 deg against the image, so MC is not truth and the bar is
set against TV):
- arms, wrists, fingers: AD's median error against TV at most 2 deg above MC's and above MF's, on common support;
  AD's p90 at most 5 deg above MC's;
- head rotation (AD vs MC): median <= 2 deg, p90 <= 5 deg;
- blendshapes (AD vs MC): mean MAE <= 0.02 and median r >= 0.95 over coefficients whose MC sd >= 0.02;
- validity: AD-only valid frames are no less accurate against TV than common frames (median within 3 deg);
- aspect: the snapped estimate equals the true shape on >= 95% of clips.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHAPES = {"4:3": 4 / 3, "16:9": 16 / 9, "1:1": 1.}
NOSE_Y, SHOULDER_Y = .35, .67
ARM_REST = {"LeftUpperArm": (1., 0., 0.), "LeftForearm": (1., 0., 0.), "RightUpperArm": (-1., 0., 0.),
            "RightForearm": (-1., 0., 0.)}
PARENT = {"Chest": "Pelvis", "LeftUpperArm": "Chest", "LeftForearm": "LeftUpperArm", "LeftHand": "LeftForearm",
          "RightUpperArm": "Chest", "RightForearm": "RightUpperArm", "RightHand": "RightForearm"}


def framing(video: Path, models: dict, aspect: float, *, nose_y: float = NOSE_Y, shoulder_y: float = SHOULDER_Y):
    """Crop box (x0, y0, w, h) in source pixels giving Sem-Lex's framing at ``aspect`` (nose and shoulder midpoint at
    ``nose_y`` and ``shoulder_y`` of the crop height); None when it does not fit."""
    import av
    import mediapipe as mp
    vision, base = mp.tasks.vision, mp.tasks.BaseOptions
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(models["pose_landmarker_heavy.task"]))))
    noses, shoulders, size = [], [], None
    with av.open(str(video)) as c:
        for k, frame in enumerate(c.decode(video=0)):
            if k >= 12:
                break
            rgb = frame.to_ndarray(format="rgb24")
            size = rgb.shape[1], rgb.shape[0]
            r = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if r.pose_landmarks:
                p = np.array([[l.x * size[0], l.y * size[1]] for l in r.pose_landmarks[0]])
                noses.append(p[0])
                shoulders.append((p[11] + p[12]) / 2)
    pose.close()
    if len(noses) < 3:
        return None
    nose, mid = np.median(noses, 0), np.median(shoulders, 0)
    height = int(round((mid[1] - nose[1]) / (shoulder_y - nose_y) / 2)) * 2
    width = int(round(aspect * height / 2)) * 2
    x0, y0 = int(round(mid[0] - width / 2)), int(round(mid[1] - shoulder_y * height))
    if height < 120 or x0 < 0 or y0 < 0 or x0 + width > size[0] or y0 + height > size[1]:
        return None
    return x0, y0, width, height


def write_crop(video: Path, box, out: Path) -> None:
    """The crop of every decoded frame, H.264 CRF 8 at the source's rate."""
    import av
    x0, y0, w, h = box
    with av.open(str(video)) as src, av.open(str(out), "w") as dst:
        stream = src.streams.video[0]
        enc = dst.add_stream("libx264", rate=stream.average_rate)
        enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
        enc.options = {"crf": "8", "preset": "medium"}
        for frame in src.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")[y0:y0 + h, x0:x0 + w]
            for packet in enc.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
                dst.mux(packet)
        for packet in enc.encode():
            dst.mux(packet)


def emulate_holistic(video: Path, models: dict, *, face_scale: float = .9, hand_scale: float = .9):
    """(T, 553, 3) float16 Sem-Lex-layout array, times and frame size, from ``extract_clip``'s trackers (ROI mode):
    the same models, crops and timestamps, recording normalised image landmarks with z instead of world landmarks.
    Hand slot A (511..531) is the left-wrist crop's hand, slot B the right's, as Holistic attaches hands to pose wrists."""
    import av
    import mediapipe as mp
    from signtranslator.data_engineering.video_landmarks import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, square_crop
    vision, base = mp.tasks.vision, mp.tasks.BaseOptions
    mode = vision.RunningMode.VIDEO

    def hand_tracker():
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(models["hand_landmarker.task"])), running_mode=mode, num_hands=1))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=base(model_asset_path=str(models["pose_landmarker_heavy.task"])), running_mode=mode))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(models["face_landmarker.task"])), running_mode=mode,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1))
    trackers = [hand_tracker(), hand_tracker()]
    rows, times, size = [], [], None
    blank = np.zeros((16, 16, 3), np.uint8)

    def image(a):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a))

    def to_full(lm, crop, offset, width, height):
        cw, ch = crop.shape[1], crop.shape[0]
        return np.c_[(lm[:, 0] * cw + offset[0]) / width, (lm[:, 1] * ch + offset[1]) / height, lm[:, 2] * cw / width]
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        last_ms, last_t = -1, -float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * frame.time_base)
            if t <= last_t:
                continue
            last_t = t
            ms = max(int(round(t * 1000)), last_ms + 1)
            last_ms = ms
            rgb = frame.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            size = (width, height)
            row = np.full((553, 3), np.nan)
            p = pose.detect_for_video(image(rgb), ms)
            face_in, hand_in = None, [None, None]
            if p.pose_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in p.pose_landmarks[0]])
                row[478:511] = lm
                px = lm[:, :2] * np.array([width, height], dtype=np.float64)
                shoulder = float(np.linalg.norm(px[L_SHOULDER] - px[R_SHOULDER]))
                face_in = square_crop(rgb, px[0], face_scale * shoulder)
                hand_in = [square_crop(rgb, px[w], hand_scale * shoulder) for w in (L_WRIST, R_WRIST)]
            for k, (tracker, item) in enumerate(zip(trackers, hand_in)):
                crop, offset = item if item is not None else (None, None)
                h = tracker.detect_for_video(image(crop if crop is not None else blank), ms)
                if crop is not None and h.hand_landmarks:
                    lm = np.array([[l.x, l.y, l.z] for l in h.hand_landmarks[0]])
                    row[511 + 21 * k:532 + 21 * k] = to_full(lm, crop, offset, width, height)
            crop, offset = face_in if face_in is not None else (None, None)
            f = face.detect_for_video(image(crop if crop is not None else blank), ms)
            if crop is not None and f.face_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in f.face_landmarks[0]])
                row[:478] = to_full(lm, crop, offset, width, height)
            rows.append(row)
            times.append(t)
    for d in (pose, face, *trackers):
        d.close()
    return np.stack(rows).astype(np.float16), np.array(times), size


CLIP_FIELDS = ("times", "pose_world", "pose_visibility", "pose_image", "pose_found", "hands_world", "hands_image",
               "hands_found", "blendshapes", "face_matrix", "face_found")


def extract_one(job):
    video, shape, out_dir, models_dir = job
    import tempfile
    from signtranslator.data_engineering.video_landmarks import extract_clip, verify_models
    models = verify_models(Path(models_dir))
    name = f"{Path(video).stem}__{shape.replace(':', 'x')}"
    target = Path(out_dir) / f"{name}.npz"
    if target.exists():
        return dict(name=name, status="present")
    if shape == "full":
        clip = extract_clip(Path(video), models, roi=True)
        extra, meta = {}, dict(name=name, video=Path(video).name, shape=shape)
    else:
        box = framing(Path(video), models, SHAPES[shape])
        if box is None:
            return dict(name=name, status="framing-does-not-fit")
        with tempfile.TemporaryDirectory() as t:
            cropped = Path(t) / "crop.mp4"
            write_crop(Path(video), box, cropped)
            clip = extract_clip(cropped, models, roi=True)
            array, times, size = emulate_holistic(cropped, models)
        extra = dict(holistic=array, holistic_times=times)
        meta = dict(name=name, video=Path(video).name, shape=shape, box=list(box), size=list(size))
    meta["names"] = list(clip.blendshape_names)
    np.savez_compressed(target, meta=np.array(json.dumps(meta)), **extra,
                        **{f"ref_{k}": np.asarray(getattr(clip, k)) for k in CLIP_FIELDS})
    return dict(name=name, status="extracted", frames=len(clip.times))


# ----------------------------------------------------------------------------------------------------------------------
# Analysis

def angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between rotation matrices (..., 3, 3)."""
    cos = (np.einsum("...ij,...ij->...", a, b) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1., 1.)))


def direction_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1., 1.)))


def matrices(channel) -> np.ndarray:
    import torch
    from signtranslator.pose.rotations import rotation_6d_to_matrix
    identity = np.array([1., 0., 0., 0., 1., 0.])                     # invalid cells hold zeros: not a rotation
    values = np.where(channel.valid[..., None], channel.values, identity)
    return rotation_6d_to_matrix(torch.from_numpy(np.ascontiguousarray(values))).numpy()


def world_frames(state) -> dict:
    """Forward kinematics of the stored body rotations: {joint: (world (T,3,3), valid (T,))} in the state's root frame."""
    body = state.channels["body"]
    local, valid = matrices(body), body.valid
    out = {"Pelvis": (local[:, body.labels.index("Pelvis")], valid[:, body.labels.index("Pelvis")])}
    for joint, parent in PARENT.items():
        j = body.labels.index(joint)
        pw, pv = out[parent]
        out[joint] = (pw @ local[:, j], pv & valid[:, j])
    return out


def accuracy_items(state, frames: np.ndarray) -> dict:
    """Per quantity, ((T', 3) directions or (T', 3, 3) frames, (T',) valid) on the given frame indices."""
    world = world_frames(state)
    out = {}
    for joint, rest in ARM_REST.items():
        w, v = world[joint]
        out[joint] = ((w @ np.asarray(rest))[frames], v[frames])
    for joint in ("LeftHand", "RightHand"):
        w, v = world[joint]
        out[joint] = (w[frames], v[frames])
    w, v = world["Chest"]
    out["ShoulderLine"] = ((w @ np.array([1., 0., 0.]))[frames], v[frames])    # the chest's x axis: right -> left
    for name in ("left_hand", "right_hand"):
        ch = state.channels[name]
        if ch.valid.any():
            m = matrices(ch)
            for j, label in enumerate(ch.labels):
                out[label] = (m[frames, j], ch.valid[frames, j])
    return out


def rotated(items: dict, rotation: np.ndarray) -> dict:
    """Body directions and wrist frames turned by one global rotation (fingers are local and stay)."""
    out = dict(items)
    for label, (value, ok) in items.items():
        if group_of(label) in ("arms", "shoulder_line"):
            out[label] = (value @ rotation.T, ok)
        elif group_of(label) == "wrists":
            out[label] = (rotation @ value, ok)
    return out


def best_rotation(pairs: np.ndarray) -> np.ndarray:
    """Kabsch rotation R minimising sum |R a - b|^2 over (N, 6) rows of unit directions (a, b)."""
    a, b = pairs[:, :3], pairs[:, 3:]
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1., 1., d]) @ vt


def item_error(a, b) -> np.ndarray:
    return direction_angles(a, b) if a.ndim == 2 else angles(a, b)


def load_npz(path: Path):
    from signtranslator.data_engineering.video_landmarks import ClipLandmarks
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        clip = ClipLandmarks(**{k[4:]: z[k].copy() for k in z.files if k.startswith("ref_")},
                             blendshape_names=tuple(meta["names"]), roi=True)
        holistic = z["holistic"].copy() if "holistic" in z.files else None
    return meta, clip, holistic


def reference_state(clip, *, cropped: bool):
    from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
    flags = dict(upright_chest=True, elbow_in_image=True, upright_root=True) if cropped else {}
    state, _ = canonical_from_landmarks(clip, sample_id="ref", source_id="ref", source_sha256="0" * 64, clock_id="ref",
                                        inference_method="reference", **flags)
    return state


def two_view_frames(tv_state, mf_times: np.ndarray):
    """Indices into the front clip's frames of each two-view frame, or None when the clocks do not match."""
    index = np.searchsorted(mf_times, tv_state.timestamps)
    index = np.clip(index, 0, len(mf_times) - 1)
    if not np.allclose(mf_times[index], tv_state.timestamps, atol=1e-6):
        return None
    return index


def adapter_state(holistic, ref_clip, meta, resources, model, *, mode, pose_k, hand_k, margin=0.):
    from signtranslator.data_engineering.semlex_pose import convert_array
    duration = len(holistic) * 1000. * float(np.median(np.diff(ref_clip.times)))
    kw = dict(image_size=tuple(meta["size"])) if mode == "true-size" else {}
    return convert_array(holistic, duration, resources, video_id="ad", source_sha256="0" * 64,
                         blendshape_model=model, pose_z_scale=pose_k, hand_z_scale=hand_k, visibility_margin=margin,
                         **kw)


def equivalence(ref, ad) -> dict:
    out = {}
    for name in ("body", "left_hand", "right_hand", "head_rotation", "face", "blink"):
        r, a = ref.channels[name], ad.channels[name]
        both = r.valid & a.valid
        entry = dict(validity=dict(ref=int(r.valid.sum()), adapter=int(a.valid.sum()), both=int(both.sum()),
                                   ref_only=int((r.valid & ~a.valid).sum()),
                                   adapter_only=int((a.valid & ~r.valid).sum()), cells=int(r.valid.size)))
        if both.any() and r.labels != a.labels:
            raise AssertionError(f"{name}: label mismatch")
        labels = a.labels
        if name in ("body", "left_hand", "right_hand", "head_rotation") and both.any():
            err = angles(matrices(r), matrices(a))
            entry["angles"] = {lab: err[both[:, j], j].tolist() for j, lab in enumerate(labels) if both[:, j].any()}
            entry["validity_per_label"] = {lab: [int(r.valid[:, j].sum()), int(a.valid[:, j].sum()), int(both[:, j].sum())]
                                           for j, lab in enumerate(labels)}
        if name in ("face", "blink") and both.any():
            entry["pairs"] = {lab: [r.values[both[:, j], j, 0].tolist(), a.values[both[:, j], j, 0].tolist()]
                              for j, lab in enumerate(labels) if both[:, j].any()}
        out[name] = entry
    return out


def stat(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                mean=round(float(v.mean()), 2))


GROUPS = {"arms": ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm"), "wrists": ("LeftHand", "RightHand"),
          "shoulder_line": ("ShoulderLine",)}


def group_of(label: str) -> str | None:
    for group, labels in GROUPS.items():
        if label in labels:
            return group
    return "fingers" if "Hand" in label and label[-1].isdigit() else None


def accuracy(items_by_method: dict, methods: tuple[str, ...]) -> dict:
    """Errors against TV per group: each method on its own valid frames, all methods on common support, and AD on
    the frames only AD (not MC) marks valid."""
    tv = items_by_method["TV"]
    errors = {m: {} for m in methods}
    common = {m: {} for m in methods}
    ad_only = {}
    for label, (truth, tv_ok) in tv.items():
        group = group_of(label)
        if group is None:
            continue
        support = tv_ok.copy()
        for m in methods:
            if label in items_by_method[m]:
                support &= items_by_method[m][label][1]
            else:
                support &= False
        for m in methods:
            if label not in items_by_method[m]:
                continue
            value, ok = items_by_method[m][label]
            err = item_error(value, truth)
            errors[m].setdefault(group, []).extend(err[ok & tv_ok].tolist())
            common[m].setdefault(group, []).extend(err[support].tolist())
        if "AD" in methods and "MC" in methods and label in items_by_method["AD"]:
            value, ok = items_by_method["AD"][label]
            mc_ok = items_by_method["MC"][label][1] if label in items_by_method["MC"] else np.zeros_like(ok)
            only = ok & ~mc_ok & tv_ok
            ad_only.setdefault(group, []).extend(item_error(value, truth)[only].tolist())
    return dict(own=errors, common=common, ad_only=ad_only)


def merge(target: dict, part: dict) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            merge(target.setdefault(key, {}), value)
        else:
            target.setdefault(key, []).extend(value)


def summarise_equivalence(results: list[dict]) -> dict:
    body = {g: stat([x for r in results for lab in labels for x in r["body"].get("angles", {}).get(lab, [])])
            for g, labels in GROUPS.items()}
    body.pop("shoulder_line")
    body["chest"] = stat([x for r in results for x in r["body"].get("angles", {}).get("Chest", [])])
    body["per_joint"] = {lab: stat([x for r in results for x in r["body"].get("angles", {}).get(lab, [])])
                         for group, labels in GROUPS.items() if group != "shoulder_line" for lab in labels}
    fingers = stat([x for r in results for side in ("left_hand", "right_hand")
                    for v in r[side].get("angles", {}).values() for x in v])
    head = stat([x for r in results for v in r["head_rotation"].get("angles", {}).values() for x in v])
    coefficients = {}
    for lab in sorted({lab for r in results for lab in r["face"].get("pairs", {})}):
        pairs = [r["face"]["pairs"][lab] for r in results if lab in r["face"].get("pairs", {})]
        ref, ad = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
        sd = float(ref.std())
        coefficients[lab] = dict(n=int(len(ref)), ref_sd=round(sd, 4), mae=round(float(np.abs(ref - ad).mean()), 5),
                                 r=round(float(np.corrcoef(ref, ad)[0, 1]), 4) if sd > 0 and ad.std() > 0 else None)
    active = [c["r"] for c in coefficients.values() if c["ref_sd"] >= .02 and c["r"] is not None]
    face = dict(mean_mae=round(float(np.mean([c["mae"] for c in coefficients.values()])), 5) if coefficients else None,
                active=len(active), median_r_active=round(float(np.median(active)), 4) if active else None,
                min_r_active=round(float(np.min(active)), 4) if active else None, coefficients=coefficients)
    validity = {}
    for name in ("body", "left_hand", "right_hand", "head_rotation", "face"):
        v = {k: sum(r[name]["validity"][k] for r in results) for k in ("ref", "adapter", "both", "ref_only",
                                                                         "adapter_only", "cells")}
        v["agreement"] = round(1 - (v["ref_only"] + v["adapter_only"]) / max(v["cells"], 1), 4)
        v["adapter_only_per_ref_valid"] = round(v["adapter_only"] / max(v["ref"], 1), 4)
        v["ref_only_per_ref_valid"] = round(v["ref_only"] / max(v["ref"], 1), 4)
        validity[name] = v
    per_label = {}
    for r in results:
        for lab, (rv, av, bv) in r["body"].get("validity_per_label", {}).items():
            e = per_label.setdefault(lab, [0, 0, 0])
            e[0] += rv; e[1] += av; e[2] += bv
    validity["body_per_joint"] = {lab: dict(ref=v[0], adapter=v[1], both=v[2]) for lab, v in per_label.items()}
    return dict(body=body, fingers=fingers, head=head, face=face, validity=validity)


def summarise_accuracy(acc: dict) -> dict:
    return {kind: {m: {g: stat(v) for g, v in groups.items()} for m, groups in by.items()} if kind != "ad_only"
            else {g: stat(v) for g, v in by.items()} for kind, by in acc.items()}


def rigidity_gain(points: np.ndarray, found: np.ndarray, bones, reference_bones, aspect: float, grid) -> dict:
    """Depth gain from rigidity: for each ``k`` in ``grid``, the mean over ``bones`` of the coefficient of variation,
    across frames, of 3D bone length (from ``(x, y / a, k z)``) relative to the mean length of ``reference_bones``.
    A rigid skeleton keeps these ratios constant; the right ``k`` makes the depth agree with the image plane."""
    out = {}
    for k in grid:
        p = np.array(points[found], dtype=np.float64)
        if len(p) < 5:
            return {}
        p[..., 1] /= aspect
        p[..., 2] *= k
        scale = np.mean([np.linalg.norm(p[:, b] - p[:, a], axis=-1) for a, b in reference_bones], axis=0)
        p, scale = p[scale > 1e-9], scale[scale > 1e-9]                    # all-zero blocks carry no hand
        if len(p) < 5:
            return {}
        ratios = np.stack([np.linalg.norm(p[:, b] - p[:, a], axis=-1) / scale for a, b in bones], axis=1)
        out[k] = float(np.mean(ratios.std(0) / ratios.mean(0)))
    return out


HAND_BONES = tuple((a, b) for f in range(5) for a, b in zip(range(1 + 4 * f, 4 + 4 * f), range(2 + 4 * f, 5 + 4 * f)))
PALM = ((0, 5), (0, 17), (5, 17))


def analyse(args) -> int:
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, load_face_resources
    from signtranslator.pose.multichannel import MultichannelMotion
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    model = BlendshapeModel(resources)
    crops, full = [], {}
    for path in sorted(Path(args.out_dir).glob("*.npz")):
        meta, clip, holistic = load_npz(path)
        if meta["shape"] == "full":
            full[meta["video"]] = clip
        else:
            crops.append((meta, clip, holistic))
    two_view = {}
    if args.two_view_dir:
        index = json.loads((Path(args.two_view_dir) / "index.json").read_text())
        for sentence, entry in index.items():
            two_view[f"{sentence}-rgb_front.mp4"] = MultichannelMotion.load(Path(args.two_view_dir) / f"{sentence}.npz",
                                                                            expected_sha256=entry["sha256"])
    report = dict(crops=len(crops), videos=len({m["video"] for m, *_ in crops}), two_view_states=len(two_view))
    det = []
    for meta, clip, hol in crops:
        xy = hol[:, 478:511, :2].astype(np.float64)
        ok = clip.pose_found & np.isfinite(xy).all((1, 2))
        det.append((len(hol) == len(clip.times), bool((np.isfinite(hol[:, 478, 0]) == clip.pose_found).all()),
                    float(np.abs(xy[ok] - clip.pose_image[ok]).max()) if ok.any() else 0.))
    report["emulator_matches_reference_detections"] = dict(frame_counts_equal=all(d[0] for d in det),
                                                          pose_found_equal=all(d[1] for d in det),
                                                          max_pose_xy_difference=max(d[2] for d in det))
    held = [x for x in crops if zlib.crc32(x[0]["video"].encode()) % 2 == 1]
    calibration = [x for x in crops if zlib.crc32(x[0]["video"].encode()) % 2 == 0]
    report["split"] = dict(calibration=len(calibration), held_out=len(held),
                           calibration_videos=sorted({m["video"] for m, *_ in calibration}),
                           held_out_videos=sorted({m["video"] for m, *_ in held}))
    cache = {}

    def references(meta, clip):
        key = meta["name"]
        if key not in cache:
            mc = reference_state(clip, cropped=True)
            mf_clip = full.get(meta["video"])
            mf = reference_state(mf_clip, cropped=False) if mf_clip is not None else None
            tv, frames = two_view.get(meta["video"]), None
            if tv is not None and mf_clip is not None and len(mf_clip.times) == len(clip.times):
                frames = two_view_frames(tv, mf_clip.times)
            cache[key] = (mc, mf, tv, frames)
        return cache[key]

    def run(items, *, mode, pose_k, hand_k, with_equivalence=True, margin=None, rotations=None):
        margin = args.visibility_margin if margin is None else margin
        equiv, acc, refused, aspects, alignment = [], {}, [], [], dict(with_tv=0, without_tv=0)
        pairs = {m: [] for m in ("AD", "MC", "MF")}
        for meta, clip, hol in items:
            mc, mf, tv, frames = references(meta, clip)
            try:
                ad, report_ad = adapter_state(hol, clip, meta, resources, model, mode=mode, pose_k=pose_k, hand_k=hand_k,
                                              margin=margin)
            except Refusal as error:
                refused.append((meta["name"], str(error)))
                continue
            aspects.append(dict(name=meta["name"], true=round(meta["size"][0] / meta["size"][1], 4),
                                raw=report_ad["aspect"].get("raw"), snapped=report_ad["aspect"].get("snapped_to")))
            if with_equivalence:
                equiv.append(equivalence(mc, ad))
            if frames is None:
                alignment["without_tv"] += 1
                continue
            alignment["with_tv"] += 1
            all_frames = np.arange(len(clip.times))[frames]
            per = {"TV": accuracy_items(tv, np.arange(len(tv.timestamps))), "AD": accuracy_items(ad, all_frames),
                   "MC": accuracy_items(mc, all_frames), "MF": accuracy_items(mf, all_frames)}
            for m in pairs:
                for label in GROUPS["arms"] + GROUPS["shoulder_line"]:
                    (a, ok), (b, tv_ok) = per[m][label], per["TV"][label]
                    pairs[m].append(np.c_[a[ok & tv_ok], b[ok & tv_ok]])
            if rotations is not None:
                per.update({m: rotated(per[m], rotations[m]) for m in rotations})
            merge(acc, accuracy(per, ("AD", "MC", "MF")))
        return dict(refused=refused, aspects=aspects, alignment=alignment,
                    pairs={m: np.concatenate(v) if v else np.zeros((0, 6)) for m, v in pairs.items()},
                    equivalence=summarise_equivalence(equiv) if equiv else None,
                    accuracy=summarise_accuracy(acc) if acc else None)

    pose_grid = [float(k) for k in args.pose_grid.split(",")]
    hand_grid = [float(k) for k in args.hand_grid.split(",")]
    scan = dict(pose={}, hand={})
    for k in pose_grid:
        r = run(calibration, mode="true-size", pose_k=k, hand_k=1., with_equivalence=False)
        scan["pose"][str(k)] = r["accuracy"]["common"]["AD"]["arms"] if r["accuracy"] else None
        print("pose k", k, scan["pose"][str(k)], flush=True)
    for k in hand_grid:
        r = run(calibration, mode="true-size", pose_k=1., hand_k=k, with_equivalence=False)
        scan["hand"][str(k)] = r["accuracy"]["common"]["AD"]["fingers"] if r["accuracy"] else None
        print("hand k", k, scan["hand"][str(k)], flush=True)
    report["depth_gain_scan_calibration_half"] = scan

    def best(table, fallback):
        scored = {float(k): v["median"] for k, v in table.items() if v and v.get("n")}
        return min(scored, key=scored.get) if scored else fallback
    pose_k = args.pose_z_scale if args.pose_z_scale is not None else best(scan["pose"], 1.)
    hand_k = args.hand_z_scale if args.hand_z_scale is not None else best(scan["hand"], 1.)
    report["depth_gain_used"] = dict(pose=pose_k, hand=hand_k, chosen_on="calibration half, AD median vs TV, common support")
    margins = {}
    for m in (float(x) for x in args.margin_grid.split(",")):
        r = run(calibration, mode="true-size", pose_k=pose_k, hand_k=hand_k, with_equivalence=False, margin=m)
        margins[str(m)] = dict(ad_only=r["accuracy"]["ad_only"], common=r["accuracy"]["common"]["AD"],
                               own=r["accuracy"]["own"]["AD"])
        print("margin", m, json.dumps(margins[str(m)]), flush=True)
    report["visibility_margin_scan_calibration_half"] = margins
    report["visibility_margin_used"] = args.visibility_margin
    for mode in ("true-size", "estimated-aspect"):
        report[f"held_out_{mode}"] = run(held, mode=mode, pose_k=pose_k, hand_k=hand_k)
        print(mode, "done", flush=True)
    # One global rotation per method (fitted on the calibration half, as the Panoptic validation did per method):
    # the two-view frame and each monocular method may differ by a constant camera-frame rotation.
    fit = run(calibration, mode="estimated-aspect", pose_k=pose_k, hand_k=hand_k, with_equivalence=False)
    rotations = {m: best_rotation(v) for m, v in fit["pairs"].items()}
    report["global_rotation_to_two_view_deg"] = {
        m: round(float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))), 2) for m, r in rotations.items()}
    report["held_out_estimated-aspect_aligned"] = run(held, mode="estimated-aspect", pose_k=pose_k, hand_k=hand_k,
                                                      with_equivalence=False, rotations=rotations)
    for key in [k for k in report if k.startswith("held_out")]:
        report[key].pop("pairs", None)
    every = run(crops, mode="estimated-aspect", pose_k=pose_k, hand_k=hand_k, with_equivalence=False)
    def shape_of(true):          # crops have even pixel sizes, so the true ratio is within 0.2% of the nominal shape
        return min(SHAPES, key=lambda n: abs(np.log(true / SHAPES[n])))
    rel = [a["raw"] / a["true"] - 1 for a in every["aspects"] if a["raw"] is not None]
    report["aspect_all_clips"] = dict(
        clips=len(crops), refused=every["refused"],
        snapped_correct=sum(a["snapped"] == shape_of(a["true"]) for a in every["aspects"]),
        estimated=len(every["aspects"]),
        relative_error_raw=dict(zip(("p10", "median", "p90"), np.percentile(rel, [10, 50, 90]).round(4).tolist())) if rel else None,
        per_clip=every["aspects"])
    # rigidity-based depth gain on the emulated arrays (the same estimator is applied to Sem-Lex)
    grid = [.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.]
    rig = {}
    for meta, clip, hol in crops:
        a = meta["size"][0] / meta["size"][1]
        for block in (slice(511, 532), slice(532, 553)):
            pts = hol[:, block].astype(np.float64)
            g = rigidity_gain(pts, np.isfinite(pts).all((1, 2)), HAND_BONES, PALM, a, grid)
            for k, v in g.items():
                rig.setdefault(k, []).append(v)
    report["hand_rigidity_cv_by_gain_emulated"] = {str(k): round(float(np.median(v)), 4) for k, v in rig.items()}

    def z_ratio(block):
        out = []
        for _, _, h in crops:
            b = h[:, block].astype(np.float64)
            ok = np.isfinite(b).all((1, 2))
            if ok.any():
                out += (np.ptp(b[ok][..., 2], axis=1) / np.maximum(np.ptp(b[ok][..., 0], axis=1), 1e-9)).tolist()
        return round(float(np.median(out)), 4) if out else None
    report["emulated_z_over_x_range"] = dict(face=z_ratio(slice(0, 468)), pose_upper=z_ratio(slice(478, 501)),
                                             left_hand=z_ratio(slice(511, 532)), right_hand=z_ratio(slice(532, 553)))
    report["verdict"] = verdict(report)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["verdict"], indent=1))
    return 0


def verdict(report: dict) -> dict:
    """The declared acceptance bar applied to the held-out estimated-aspect run (the path Sem-Lex takes)."""
    out = {}
    run = report["held_out_estimated-aspect"]
    common = (run["accuracy"] or {}).get("common", {})
    for group in ("arms", "wrists", "fingers"):
        ad, mc, mf = (common.get(m, {}).get(group, {}) for m in ("AD", "MC", "MF"))
        ok = bool(ad.get("n")) and ad["median"] <= mc["median"] + 2 and ad["median"] <= mf["median"] + 2 \
            and ad["p90"] <= mc["p90"] + 5
        out[group] = dict(passed=ok, AD=ad, MC=mc, MF=mf)
    eq = run["equivalence"]
    out["head"] = dict(passed=bool(eq["head"].get("n")) and eq["head"]["median"] <= 2 and eq["head"]["p90"] <= 5,
                       **eq["head"])
    out["blendshapes"] = dict(passed=eq["face"]["mean_mae"] is not None and eq["face"]["mean_mae"] <= .02
                              and (eq["face"]["median_r_active"] or 0) >= .95,
                              mean_mae=eq["face"]["mean_mae"], median_r_active=eq["face"]["median_r_active"])
    only = (run["accuracy"] or {}).get("ad_only", {})
    validity = {}
    for group in ("arms", "wrists", "fingers"):
        o, c = only.get(group, {}), common.get("AD", {}).get(group, {})
        validity[group] = dict(passed=not o.get("n") or (bool(c.get("n")) and o["median"] <= c["median"] + 3),
                               ad_only=o, common=c)
    out["validity"] = dict(passed=all(v["passed"] for v in validity.values()), **validity)
    a = report["aspect_all_clips"]
    out["aspect"] = dict(passed=a["snapped_correct"] >= .95 * a["clips"], snapped_correct=a["snapped_correct"],
                         clips=a["clips"])
    out["accepted"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("extract", "analyse"))
    p.add_argument("--videos", nargs="*", default=[])
    p.add_argument("--shapes", default="4:3,16:9")
    p.add_argument("--extra-shape", default="1:1", help="also crop the first --extra-count videos to this shape")
    p.add_argument("--extra-count", type=int, default=10)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--two-view-dir", help="two-view states (<sentence>.npz) with index.json of their SHA-256")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pose-grid", default="0.1,0.2,0.3,0.4,0.5,0.75,1.0")
    p.add_argument("--hand-grid", default="0.5,1.0,1.5,2.0,3.0")
    p.add_argument("--pose-z-scale", type=float)
    p.add_argument("--hand-z-scale", type=float)
    p.add_argument("--margin-grid", default="0.0,0.02,0.04,0.06,0.1")
    p.add_argument("--visibility-margin", type=float, default=0.)
    p.add_argument("--report")
    args = p.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.stage == "analyse":
        return analyse(args)
    jobs = [(v, "full", args.out_dir, args.models_dir) for v in args.videos]
    jobs += [(v, s, args.out_dir, args.models_dir) for v in args.videos for s in args.shapes.split(",")]
    jobs += [(v, args.extra_shape, args.out_dir, args.models_dir) for v in args.videos[:args.extra_count]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(extract_one, jobs):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
