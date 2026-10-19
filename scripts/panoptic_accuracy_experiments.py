"""Accuracy experiments against CMU Panoptic ground truth (tuning and held-out test).

Stage A (expensive, cached once per sequence/window): MediaPipe pose for crop boxes,
Sapiens full-body heatmaps with and without horizontal-flip averaging, and Sapiens on
high-resolution crops around each hand. Stage B (cheap): every pipeline variant is
evaluated on the same cached keypoints —

    raw            body-pass keypoints, self-calibrated two-view triangulation
    +flip          body pass averaged with its mirror
    +hands         finger keypoints from the hand-crop pass
    +fit(λ)        rigid-bone skeleton fitted to both views with smoothness λ

— against the dome ground truth: arm and finger bone-direction errors (one rotation
per method, scale-free), arm PA-MPJPE, and bone-length CV. Video frame ``f`` is taken
as dome HD frame ``f``; the offset is verified by triangulating with the dome's own
calibration at offsets −2..2 and keeping the best.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.sapiens_keypoints import (  # noqa: E402
    HAND21, crop_box, decode_heatmaps, flip_average, hand_box, preprocess,
)
from signtranslator.data_engineering.skeleton_fit import Skeleton, fit_skeleton  # noqa: E402
from signtranslator.data_engineering.two_view_geometry import (  # noqa: E402
    TwoViewObservations, bone_length_cv, calibrate, triangulate, triangulate_session,
)
from signtranslator.data_engineering.two_view_states import CALIBRATION_ANCHORS, refine_hands, rigid_bones  # noqa: E402
from signtranslator.data_engineering.video_landmarks import extract_clip, verify_models  # noqa: E402
from validate_two_view_on_panoptic import (  # noqa: E402
    GT_ARM, HAND_BONES, SAPIENS_ARM, direction_error_degrees, directions, pa_mpjpe, pose_boxes,
)

SIZE = (1920, 1080)


def ground_truth(root: Path, frames: list[int]):
    body = np.full((len(frames), 19, 3), np.nan)
    hands = {s: np.full((len(frames), 21, 3), np.nan) for s in ("Left", "Right")}
    for i, f in enumerate(frames):
        path = root / "gt/hdPose3d_stage1_coco19" / f"body3DScene_{f:08d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["bodies"]:
                body[i] = np.array(data["bodies"][0]["joints19"]).reshape(19, 4)[:, :3]
        path = root / "gt/hdHand3d" / f"handRecon3D_hd{f:08d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["people"]:
                for side, key in (("Left", "left_hand"), ("Right", "right_hand")):
                    hand = data["people"][0].get(key)
                    if hand and "landmarks" in hand:
                        hands[side][i] = np.array(hand["landmarks"]).reshape(21, 3)
    return body, hands


def decode_window(video: Path, start: int, count: int):
    """Frames ``start .. start+count-1`` (0-based display order), seeking to the preceding keyframe."""
    import av
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        container.seek(max(int((start - 60) / rate / stream.time_base), 0), stream=stream, backward=True)
        for frame in container.decode(stream):
            index = int(round(frame.pts * stream.time_base * rate))
            if index < start:
                continue
            if index >= start + count:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != count:
        raise RuntimeError(f"{video.name}: decoded {len(frames)} of {count} frames from {start}")
    return frames


HAND_SCALES = (1.6, 3.)          # crop side in shoulder widths: tight, and with forearm context


@torch.no_grad()
def sapiens_passes(frames, boxes, model, device, hand_scales=HAND_SCALES):
    """Body keypoints (plain and flip-averaged) and hand-crop keypoints at each scale."""
    def heat(rgb, box):
        tensor = preprocess(rgb, box).to(device)
        return model(tensor).float().cpu().numpy()[0], model(torch.flip(tensor, dims=[3])).float().cpu().numpy()[0]
    keys = ("plain", "flip", *(f"crop{scale:g}" for scale in hand_scales))
    out = {k: np.full((len(frames), 308, 2), np.nan) for k in keys}
    conf = {k: np.zeros((len(frames), 308)) for k in keys}
    for t, rgb in enumerate(frames):
        box = crop_box(boxes[t])
        plain, mirrored = heat(rgb, box)
        out["plain"][t], conf["plain"][t] = decode_heatmaps(plain, box)
        out["flip"][t], conf["flip"][t] = decode_heatmaps(flip_average(plain, mirrored), box)
        shoulder = float(np.linalg.norm(out["flip"][t, 5] - out["flip"][t, 6]))
        for scale in hand_scales:
            key = f"crop{scale:g}"
            for side in ("Left", "Right"):
                hb = hand_box(out["flip"][t], conf["flip"][t], side, shoulder, scale=scale)
                if hb is None:
                    continue
                crop = crop_box(hb, margin=0.)
                points, confidence = decode_heatmaps(flip_average(*heat(rgb, crop)), crop)
                index = list(HAND21[side])
                out[key][t, index], conf[key][t, index] = points[index], confidence[index]
    return out, conf


@torch.no_grad()
def calibration_keypoints(video: Path, frames: list[int], model, device):
    """Flip-averaged Sapiens keypoints on frames spread over the sequence (session-style calibration).

    The signer box comes from a coarse pass on the whole frame, refined by a second pass.
    """
    import av
    wanted, out, conf = set(frames), [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        for frame in container.decode(stream):
            index = int(round(frame.pts * stream.time_base * rate))
            if index > max(frames):
                break
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            box = crop_box((0., 0., float(rgb.shape[1]), float(rgb.shape[0])), margin=0.)
            points, confidence = decode_heatmaps(model(preprocess(rgb, box).to(device)).float().cpu().numpy()[0], box)
            seen = points[confidence >= .3]
            if len(seen) >= 10:
                box = crop_box((*seen.min(0), *seen.max(0)))
            tensor = preprocess(rgb, box).to(device)
            heat = flip_average(model(tensor).float().cpu().numpy()[0],
                                model(torch.flip(tensor, dims=[3])).float().cpu().numpy()[0])
            points, confidence = decode_heatmaps(heat, box)
            out.append(points)
            conf.append(confidence)
    return np.array(out), np.array(conf)


def merged(stage, view, crop_key, mode="confidence"):
    """Flip-averaged body keypoints with hand joints from the crop pass.

    ``confidence``: take the crop joint where it is more confident than the body pass.
    ``position``: take every crop position the crop detected (≥ .3), keeping the body
    pass's confidence — Sapiens' confidences are not comparable across crop contexts.
    """
    points, conf = stage[f"{view}_flip_points"].copy(), stage[f"{view}_flip_conf"].copy()
    crop_points, crop_conf = stage[f"{view}_{crop_key}_points"], stage[f"{view}_{crop_key}_conf"]
    if mode == "confidence":
        better = crop_conf > conf
        points[better], conf[better] = crop_points[better], crop_conf[better]
    else:
        detected = crop_conf >= .3
        points[detected] = crop_points[detected]
    return points, conf


def undistort(points, camera, iterations=20):
    """Pixels with the dome camera's radial-tangential distortion (k1 k2 p1 p2 k3) removed."""
    k = np.array(camera["K"])
    k1, k2, p1, p2, k3 = (list(np.ravel(camera["distCoef"])) + [0.] * 5)[:5]
    flat = points.reshape(-1, 2)
    xd = (flat - k[:2, 2]) / np.diag(k)[:2]
    x = xd.copy()
    for _ in range(iterations):
        r2 = (x ** 2).sum(1)
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        tangential = np.stack((2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2),
                               p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]), axis=1)
        x = (xd - tangential) / radial[:, None]
    return (x * np.diag(k)[:2] + k[:2, 2]).reshape(points.shape)


ARM_JOINTS = (5, 6, 7, 8, 62, 41)


def smooth_joints(points, joints, sigma):
    """Gaussian temporal smoothing of the given joints (NaN-aware)."""
    from scipy.ndimage import gaussian_filter1d
    out = points.copy()
    for j in joints:
        track = points[:, j]
        finite = np.isfinite(track).all(1)
        if finite.sum() < 3:
            continue
        filled = np.stack([np.interp(np.arange(len(track)), np.flatnonzero(finite), track[finite, c]) for c in range(3)], 1)
        out[:, j] = np.where(finite[:, None], gaussian_filter1d(filled, sigma, axis=0, mode="nearest"), np.nan)
    return out


def goliath_skeleton() -> Skeleton:
    joints, parents = [5], [-1]
    def add(j, parent):
        joints.append(j)
        parents.append(joints.index(parent))
    for j, p in ((6, 5), (7, 5), (62, 7), (8, 6), (41, 8)):
        add(j, p)
    for side, wrist in (("Left", 62), ("Right", 41)):
        hand = HAND21[side]
        for f in range(5):
            chain = (wrist, *hand[1 + 4 * f: 5 + 4 * f])
            for a, b in zip(chain, chain[1:]):
                add(b, a)
    return Skeleton(tuple(joints), tuple(parents))


def metrics(points, gt_body, gt_hands):
    arm = direction_error_degrees(directions(points, [SAPIENS_ARM[k] for k in GT_ARM]),
                                  directions(gt_body, list(GT_ARM.values())))
    out = dict(arm_direction_median_deg=arm[0], arm_direction_mean_deg=arm[1])
    for side in ("Left", "Right"):
        hand = direction_error_degrees(directions(points[:, list(HAND21[side])], HAND_BONES),
                                       directions(gt_hands[side], HAND_BONES))
        out[f"{side.lower()}_fingers_median_deg"], out[f"{side.lower()}_fingers_mean_deg"] = hand[0], hand[1]
    joints = [5, 7, 62, 6, 8, 41]
    out["arm_pa_mpjpe_cm"] = pa_mpjpe(points[:, joints], gt_body[:, [3, 4, 5, 9, 10, 11]])
    out["arm_bone_cv"] = bone_length_cv(points, list(SAPIENS_ARM.values()))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--front", required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--mediapipe-models", type=Path, required=True)
    parser.add_argument("--sapiens", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--smoothness", default="0,5,20,50")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", help="comma-separated subset of variant labels to evaluate")
    parser.add_argument("--calibration-frames", type=int, default=0,
                        help="also calibrate on this many frames spread over the whole sequence")
    parser.add_argument("--hinge", default="0", help="comma-separated hinge-prior weights for the hand refit")
    parser.add_argument("--extras", action="store_true", help="also weighted DLT and arm smoothing variants")
    parser.add_argument("--no-full-fit", action="store_true", help="skip the whole-skeleton fit variants")
    parser.add_argument("--hand-scales", default="1.6,3", help="hand-crop scales for Stage A ('' for none)")
    args = parser.parse_args(argv)
    name = args.sequence.name
    cameras = {c["name"]: c for c in json.loads((args.sequence / f"calibration_{name}.json").read_text())["cameras"]
               if c["type"] == "hd"}
    if args.cache.exists():
        cache = dict(np.load(args.cache, allow_pickle=False))
        stage = {k: cache[k] for k in cache}
    else:
        models = verify_models(args.mediapipe_models)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = torch.jit.load(str(args.sapiens), map_location=device).eval()
        stage = {}
        for view, camera in (("front", args.front), ("side", args.side)):
            video = args.sequence / f"hdVideos/hd_{camera}.mp4"
            frames = decode_window(video, args.start, args.frames)
            import tempfile, av
            with tempfile.TemporaryDirectory() as temporary:  # MediaPipe needs a file of just the window
                clip_path = Path(temporary) / "w.mp4"
                with av.open(str(clip_path), "w") as out:
                    stream = out.add_stream("libx264", rate=30)
                    stream.width, stream.height, stream.pix_fmt = SIZE[0], SIZE[1], "yuv420p"
                    stream.options = {"crf": "12"}
                    for rgb in frames:
                        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                            out.mux(packet)
                    for packet in stream.encode():
                        out.mux(packet)
                mp = extract_clip(clip_path, models, roi=True)
            boxes = pose_boxes(mp, SIZE)
            scales = tuple(float(x) for x in args.hand_scales.split(",") if x)
            points, conf = sapiens_passes(frames, boxes, model, device, scales)
            for key in points:
                stage[f"{view}_{key}_points"], stage[f"{view}_{key}_conf"] = points[key], conf[key]
            if view == "front":
                stage["mp_world"] = np.where((mp.pose_visibility >= .5)[..., None], mp.pose_world, np.nan)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **stage)
    count = len(stage["front_plain_points"])
    extra = None
    if args.calibration_frames:
        extra_path = args.cache.with_suffix(f".calib{args.calibration_frames}.npz")
        if extra_path.exists():
            extra = dict(np.load(extra_path))
        else:
            gt_frames = sorted(int(f.stem[-8:]) for f in (args.sequence / "gt/hdPose3d_stage1_coco19").glob("*.json"))
            picks = [int(x) for x in np.linspace(gt_frames[0], gt_frames[-1], args.calibration_frames)]
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            model = torch.jit.load(str(args.sapiens), map_location=device).eval()
            extra = {}
            for view, camera in (("front", args.front), ("side", args.side)):
                extra[f"{view}_points"], extra[f"{view}_conf"] = calibration_keypoints(
                    args.sequence / f"hdVideos/hd_{camera}.mp4", picks, model, device)
            np.savez_compressed(extra_path, **extra)
    cam1, cam2 = cameras[args.front], cameras[args.side]
    p1_true = np.array(cam1["K"]) @ np.c_[np.array(cam1["R"]), np.array(cam1["t"]).ravel()]
    p2_true = np.array(cam2["K"]) @ np.c_[np.array(cam2["R"]), np.array(cam2["t"]).ravel()]
    # Verify the frame offset between the videos and the dome's frame numbering.
    offsets = {}
    for offset in range(-2, 3):
        gt_body, _ = ground_truth(args.sequence, [args.start + i + offset for i in range(count)])
        fp, sp = stage["front_flip_points"], stage["side_flip_points"]
        usable = (stage["front_flip_conf"] >= .3) & (stage["side_flip_conf"] >= .3)
        tri = np.full(fp.shape[:2] + (3,), np.nan)
        t_idx, j_idx = np.nonzero(usable)
        tri[t_idx, j_idx] = triangulate(p1_true, p2_true, undistort(fp, cam1)[t_idx, j_idx],
                                        undistort(sp, cam2)[t_idx, j_idx])
        offsets[offset] = direction_error_degrees(directions(tri, [SAPIENS_ARM[k] for k in GT_ARM]),
                                                  directions(gt_body, list(GT_ARM.values())))[0]
    best_offset = min(offsets, key=offsets.get)
    frames = [args.start + i + best_offset for i in range(count)]
    gt_body, gt_hands = ground_truth(args.sequence, frames)
    results = {"mediapipe_monocular": metrics_mediapipe(stage["mp_world"], gt_body)}
    calibration_report = {}
    variants = {"raw": "plain", "raw+flip": "flip"}
    cached = sorted(float(k.split("_")[1][4:]) for k in stage if k.startswith("front_crop") and k.endswith("_points"))
    variants.update({f"raw+flip+hands{scale:g}": f"crop{scale:g}" for scale in cached})
    variants.update({f"raw+flip+handpos{scale:g}": f"pos{scale:g}" for scale in cached})
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants.split(",")}
    for label, key in variants.items():
        if key.startswith("crop"):
            (fp, fc), (sp, sc) = merged(stage, "front", key), merged(stage, "side", key)
        elif key.startswith("pos"):
            crop = "crop" + key[3:]
            (fp, fc), (sp, sc) = merged(stage, "front", crop, "position"), merged(stage, "side", crop, "position")
        else:
            fp, fc = stage[f"front_{key}_points"], stage[f"front_{key}_conf"]
            sp, sc = stage[f"side_{key}_points"], stage[f"side_{key}_conf"]
        usable = (fc >= .3) & (sc >= .3)
        obs = TwoViewObservations(fp, sp, usable, SIZE, SIZE, rigid_bones())
        calib_obs = obs
        if extra is not None and key == "flip":
            n = min(len(extra["front_points"]), len(extra["side_points"]))
            v1, v2 = np.concatenate((fp, extra["front_points"][:n])), np.concatenate((sp, extra["side_points"][:n]))
            c1, c2 = np.concatenate((fc, extra["front_conf"][:n])), np.concatenate((sc, extra["side_conf"][:n]))
            calib_obs = TwoViewObservations(v1, v2, (c1 >= .3) & (c2 >= .3), SIZE, SIZE, rigid_bones())
        calibrations = {"self": calibrate(calib_obs, anchors=CALIBRATION_ANCHORS),
                        "self+dist": calibrate(calib_obs, distortion=True, anchors=CALIBRATION_ANCHORS)}
        geometry = {}
        for kind, calibration in calibrations.items():
            calibration_report[f"{label} [{kind}]"] = dict(
                focal=[calibration.focal1, calibration.focal2], distortion=list(calibration.distortion),
                true_focal=[float(np.array(cam1["K"])[0, 0]), float(np.array(cam2["K"])[0, 0])],
                true_k1=[float(np.ravel(cam1["distCoef"])[0]), float(np.ravel(cam2["distCoef"])[0])],
                reprojection_px=list(calibration.median_reprojection_px))
            geometry[kind] = (*calibration.matrices(SIZE, SIZE), *calibration.undistort(fp, sp, SIZE, SIZE))
        geometry["true"] = (p1_true, p2_true, undistort(fp, cam1), undistort(sp, cam2))
        for calib, (p1, p2, x1, x2) in geometry.items():
            tri = np.full(fp.shape[:2] + (3,), np.nan)
            t_idx, j_idx = np.nonzero(usable)
            tri[t_idx, j_idx] = triangulate(p1, p2, x1[t_idx, j_idx], x2[t_idx, j_idx])
            tag = f"{label} [{calib}-calib]"
            results[tag] = metrics(tri, gt_body, gt_hands)
            if args.extras:
                weighted = np.full_like(tri, np.nan)
                weighted[t_idx, j_idx] = triangulate(p1, p2, x1[t_idx, j_idx], x2[t_idx, j_idx],
                                                     fc[t_idx, j_idx], sc[t_idx, j_idx])
                results[f"{tag}+wdlt"] = metrics(weighted, gt_body, gt_hands)
                for sigma in (1., 2.):
                    results[f"{tag}+armsmooth(σ={sigma:g})"] = metrics(smooth_joints(tri, ARM_JOINTS, sigma),
                                                                      gt_body, gt_hands)
            for lam in (float(x) for x in args.smoothness.split(",")):
                if not args.no_full_fit:
                    try:
                        fitted, report = fit_skeleton(tri, x1, x2, fc * usable, sc * usable, p1, p2,
                                                      goliath_skeleton(), smoothness=lam)
                        results[f"{tag}+fit(λ={lam:g})"] = dict(
                            metrics(fitted, gt_body, gt_hands), fit_rms_px=[report["start_rms_px"], report["end_rms_px"]])
                    except ValueError as error:
                        results[f"{tag}+fit(λ={lam:g})"] = dict(error=str(error))
                for mu in (float(x) for x in args.hinge.split(",")):
                    hybrid = refine_hands(tri, x1, x2, fc * usable, sc * usable, p1, p2, smoothness=lam, hinge_weight=mu)
                    suffix = f",μ={mu:g}" if mu else ""
                    results[f"{tag}+handfit(λ={lam:g}{suffix})"] = metrics(hybrid, gt_body, gt_hands)
    record = dict(schema_version=1, sequence=name, cameras=[args.front, args.side], start=args.start, frames=count,
                  frame_offset=best_offset, offset_check=offsets, calibration=calibration_report, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    for label, values in results.items():
        if "error" in values:
            print(f"{label:34s} {values['error']}")
            continue
        print(f"{label:34s} arm {values['arm_direction_median_deg']:.2f}°  fingers "
              f"{values.get('left_fingers_median_deg', float('nan')):.2f}/{values.get('right_fingers_median_deg', float('nan')):.2f}°  "
              f"PA {values['arm_pa_mpjpe_cm']:.2f} cm")
    return 0


def metrics_mediapipe(mp_world, gt_body):
    from validate_two_view_on_panoptic import MP_ARM
    arm = direction_error_degrees(directions(mp_world, [MP_ARM[k] for k in GT_ARM]),
                                  directions(gt_body, list(GT_ARM.values())))
    return dict(arm_direction_median_deg=arm[0], arm_direction_mean_deg=arm[1],
                arm_pa_mpjpe_cm=pa_mpjpe(mp_world[:, [11, 13, 15, 12, 14, 16]], gt_body[:, [3, 4, 5, 9, 10, 11]]))


if __name__ == "__main__":
    raise SystemExit(main())
