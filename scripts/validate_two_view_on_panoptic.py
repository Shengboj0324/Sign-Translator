"""Measure the replacement pipeline against multi-camera ground truth (CMU Panoptic).

How2Sign gives no 3D ground truth, so accuracy is measured where it exists: a CMU
Panoptic Studio sequence (non-commercial research data) whose 3D body and hand
keypoints are reconstructed from the dome's many cameras. Two HD views shaped like
How2Sign's frontal/lateral pair are processed exactly as How2Sign will be:

* (a) MediaPipe on the frontal view alone (the monocular fallback);
* (b) Sapiens on both views, self-calibrated with the bone-constrained bundle
  adjustment, triangulated;
* (c) the same Sapiens keypoints triangulated with the dome's own calibration
  (lens distortion ignored), an upper bound for (b).

Metrics, all free of scale and coordinate frame: the angle between each method's
bone directions and the ground truth's after one best rotation per method (what the
canonical rotations encode); per-frame Procrustes-aligned joint error (PA-MPJPE, cm);
bone-length coefficient of variation; and the fraction of frames with a usable hand.
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
from signtranslator.data_engineering.sapiens_keypoints import HAND21, crop_box, decode_heatmaps, preprocess  # noqa: E402
from signtranslator.data_engineering.two_view_geometry import (  # noqa: E402
    Calibration, TwoViewObservations, bone_length_cv, calibrate, triangulate, triangulate_session,
)
from signtranslator.data_engineering.two_view_states import rigid_bones, synchronise, ClipKeypoints  # noqa: E402
from signtranslator.data_engineering.video_landmarks import extract_clip, verify_models  # noqa: E402

# Panoptic coco19: 3 lShoulder, 4 lElbow, 5 lWrist, 9 rShoulder, 10 rElbow, 11 rWrist.
GT_ARM = {"LeftUpperArm": (3, 4), "LeftForearm": (4, 5), "RightUpperArm": (9, 10), "RightForearm": (10, 11)}
MP_ARM = {"LeftUpperArm": (11, 13), "LeftForearm": (13, 15), "RightUpperArm": (12, 14), "RightForearm": (14, 16)}
SAPIENS_ARM = {"LeftUpperArm": (5, 7), "LeftForearm": (7, 62), "RightUpperArm": (6, 8), "RightForearm": (8, 41)}
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def load_ground_truth(root: Path, frames: int):
    body = np.full((frames, 19, 3), np.nan)
    hands = {s: np.full((frames, 21, 3), np.nan) for s in ("Left", "Right")}
    for t in range(frames):
        path = root / "gt/hdPose3d_stage1_coco19" / f"body3DScene_{t:08d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["bodies"]:
                body[t] = np.array(data["bodies"][0]["joints19"]).reshape(19, 4)[:, :3]
        path = root / "gt/hdHand3d" / f"handRecon3D_hd{t:08d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["people"]:
                for side, key in (("Left", "left_hand"), ("Right", "right_hand")):
                    hand = data["people"][0].get(key)
                    if hand and "landmarks" in hand:
                        hands[side][t] = np.array(hand["landmarks"]).reshape(21, 3)
    return body, hands


def directions(points, bones):
    d = np.stack([points[:, b] - points[:, a] for a, b in bones], axis=1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def best_rotation(source, target):
    """Kabsch rotation R minimising |R s - t| over all finite direction pairs."""
    ok = np.isfinite(source).all(-1) & np.isfinite(target).all(-1)
    u, _, vt = np.linalg.svd(source[ok].T @ target[ok])
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def direction_error_degrees(method, truth):
    rotation = best_rotation(method.reshape(-1, 3), truth.reshape(-1, 3))
    aligned = method @ rotation.T
    cosine = np.clip(np.sum(aligned * truth, -1), -1, 1)
    errors = np.degrees(np.arccos(cosine))
    return float(np.nanmedian(errors)), float(np.nanmean(errors)), int(np.isfinite(errors).sum())


def pa_mpjpe(method, truth):
    """Mean per-joint error after per-frame similarity (Procrustes) alignment, in truth units."""
    errors = []
    for m, g in zip(method, truth):
        ok = np.isfinite(m).all(-1) & np.isfinite(g).all(-1)
        if ok.sum() < 3:
            continue
        a, b = m[ok] - m[ok].mean(0), g[ok] - g[ok].mean(0)
        u, s, vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ np.diag([1, 1, d]) @ u.T
        scale = (s * np.array([1, 1, d])).sum() / (a ** 2).sum()
        errors.append(np.linalg.norm(scale * a @ rotation.T - b, axis=1).mean())
    return float(np.mean(errors)) if errors else float("nan")


@torch.no_grad()
def sapiens_view(video, boxes, model, device):
    import av
    points, conf = [], []
    with av.open(str(video)) as container:
        for t, frame in enumerate(container.decode(container.streams.video[0])):
            if t >= len(boxes):
                break
            rgb = frame.to_ndarray(format="rgb24")
            box = crop_box(boxes[t])
            heat = model(preprocess(rgb, box).to(device)).float().cpu().numpy()[0]
            p, c = decode_heatmaps(heat, box)
            points.append(p)
            conf.append(c)
    return np.array(points), np.array(conf)


def pose_boxes(clip, size):
    boxes = []
    for t in range(len(clip.times)):
        if clip.pose_found[t]:
            xy = clip.pose_image[t] * np.array(size)
            boxes.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
        else:
            boxes.append(boxes[-1] if boxes else (0., 0., float(size[0]), float(size[1])))
    return boxes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--front", default="00_21")
    parser.add_argument("--side", default="00_07")
    parser.add_argument("--mediapipe-models", type=Path, required=True)
    parser.add_argument("--sapiens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    size = (1920, 1080)
    name = args.sequence.name
    calibration = {c["name"]: c for c in json.loads((args.sequence / f"calibration_{name}.json").read_text())["cameras"]
                   if c["type"] == "hd"}
    models = verify_models(args.mediapipe_models)
    front_video = args.sequence / f"hdVideos/hd_{args.front}.mp4"
    side_video = args.sequence / f"hdVideos/hd_{args.side}.mp4"
    mp_front = extract_clip(front_video, models, roi=True)
    mp_side = extract_clip(side_video, models, roi=True)
    frames = min(len(mp_front.times), len(mp_side.times))
    gt_body, gt_hands = load_ground_truth(args.sequence, frames)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = torch.jit.load(str(args.sapiens), map_location=device).eval()
        sp_front = sapiens_view(front_video, pose_boxes(mp_front, size)[:frames], model, device)
    except RuntimeError:
        device = torch.device("cpu")
        model = torch.jit.load(str(args.sapiens), map_location=device).eval()
        sp_front = sapiens_view(front_video, pose_boxes(mp_front, size)[:frames], model, device)
    sp_side = sapiens_view(side_video, pose_boxes(mp_side, size)[:frames], model, device)
    times = mp_front.times[:frames]
    sync = synchronise(ClipKeypoints("f", times, *sp_front), ClipKeypoints("s", times, *sp_side))
    usable = (sp_front[1] >= .3) & (sp_side[1] >= .3)
    obs = TwoViewObservations(sp_front[0], sp_side[0], usable, size, size, rigid_bones())
    self_cal = calibrate(obs)
    tri_self = triangulate_session(self_cal, obs)
    cam1, cam2 = calibration[args.front], calibration[args.side]
    p1 = np.array(cam1["K"]) @ np.c_[np.array(cam1["R"]), np.array(cam1["t"]).ravel()]
    p2 = np.array(cam2["K"]) @ np.c_[np.array(cam2["R"]), np.array(cam2["t"]).ravel()]
    t_idx, j_idx = np.nonzero(usable)
    tri_gt = np.full(usable.shape + (3,), np.nan)
    tri_gt[t_idx, j_idx] = triangulate(p1, p2, sp_front[0][t_idx, j_idx], sp_side[0][t_idx, j_idx])

    def arms(points, mapping):
        return directions(points, [mapping[k] for k in GT_ARM])

    gt_arm = arms(gt_body, GT_ARM)
    mp_world = np.where((mp_front.pose_visibility[:frames] >= .5)[..., None], mp_front.pose_world[:frames], np.nan)
    results = dict(arm_direction_error_degrees={}, hand_direction_error_degrees={}, arm_pa_mpjpe_cm={},
                   bone_length_cv={}, hand_frames_fraction={})
    for label, points, mapping in (("mediapipe_monocular", mp_world, MP_ARM),
                                   ("sapiens_two_view_self_calibrated", tri_self, SAPIENS_ARM),
                                   ("sapiens_two_view_true_calibration", tri_gt, SAPIENS_ARM)):
        results["arm_direction_error_degrees"][label] = direction_error_degrees(arms(points, mapping), gt_arm)
        joints = [mapping["LeftUpperArm"][0], mapping["LeftForearm"][0], mapping["LeftForearm"][1],
                  mapping["RightUpperArm"][0], mapping["RightForearm"][0], mapping["RightForearm"][1]]
        results["arm_pa_mpjpe_cm"][label] = pa_mpjpe(points[:, joints], gt_body[:, [3, 4, 5, 9, 10, 11]])
        results["bone_length_cv"][label] = bone_length_cv(points, list(mapping.values()))
    for side in ("Left", "Right"):
        truth = directions(gt_hands[side], HAND_BONES)
        mp_hand = np.full((frames, 21, 3), np.nan)
        from signtranslator.data_engineering.video_landmarks import assign_hands
        assigned = assign_hands(mp_front)
        mp_hand[assigned[side][1][:frames]] = assigned[side][0][:frames][assigned[side][1][:frames]]
        for label, points in (("mediapipe_monocular", mp_hand),
                              ("sapiens_two_view_self_calibrated", tri_self[:, list(HAND21[side])]),
                              ("sapiens_two_view_true_calibration", tri_gt[:, list(HAND21[side])])):
            results["hand_direction_error_degrees"].setdefault(label, {})[side] = direction_error_degrees(
                directions(points, HAND_BONES), truth)
            results["hand_frames_fraction"].setdefault(label, {})[side] = float(np.mean(np.isfinite(points).all((1, 2))))
    record = dict(schema_version=1, sequence=name, cameras=[args.front, args.side], frames=frames, sync=sync,
                  self_calibration=dict(focal=[self_cal.focal1, self_cal.focal2],
                                        true_focal=[float(np.array(cam1["K"])[0, 0]), float(np.array(cam2["K"])[0, 0])],
                                        reprojection_px=list(self_cal.median_reprojection_px),
                                        converged=self_cal.converged),
                  device=str(device), results=results,
                  notes=["direction errors are medians/means over bones and frames after one rotation per method",
                         "PA-MPJPE in cm on shoulders, elbows and wrists",
                         "true-calibration triangulation ignores lens distortion"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, default=float)
        stream.write("\n")
    print(json.dumps(record, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
