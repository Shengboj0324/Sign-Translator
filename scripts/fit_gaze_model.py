"""Calibrate ``gaze.GazeModel`` on MPIIFaceGaze, at How2Sign's face resolution.

Stage 1 (``features``): every MPIIFaceGaze image (15 participants, laptop webcam,
3D gaze target in camera coordinates) is read from the archived zip in R2, scaled so
that its interpupillary distance in pixels equals How2Sign's frontal view, and cut to
the square face crop that ``video_landmarks.extract_clip`` hands MediaPipe (side
``face_scale`` × shoulder width, with shoulder width ≈ ``CROP_PER_IPD`` × IPD). The
Face Landmarker (pinned model) gives the blendshapes and the crop camera's head
rotation. Ground truth per eye is the target minus the eye centre — the undistorted
eye-corner midpoint back-projected to the face centre's depth — expressed, like the
head rotation, in the crop camera's axes (x right, y up, z toward the viewer): the
crop's optical axis is the ray through its centre, not the camera's.

Stage 2 (``fit``): leave-one-participant-out angular errors of the model and of two
baselines (head direction only; head plus a constant eye offset), the anatomical side
of each MediaPipe suffix, and the final model fitted on all participants, written
to R2 and to ``--output``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.gaze import (  # noqa: E402
    SIDES, VARIANTS, GazeModel, angle_degrees, eye_features, fit_eye, head_angles,
)

ZIP_KEY = "acquisition/2026-10-17-v1/gaze-mpiifacegaze/MPIIFaceGaze.zip"
FEATURES_KEY = "acquisition/2026-10-17-v1/derived/gaze-calibration-v1/mpiifacegaze-features.npz"
MODEL_KEY = "acquisition/2026-10-17-v1/derived/gaze-calibration-v1/gaze-model.json"
HOW2SIGN_IPD_PX = 44.6          # median frontal eye-keypoint distance (Sapiens pilot, 1280×720)
CROP_PER_IPD = 206.3 / 44.6     # How2Sign shoulder width per IPD in the same frames
FACE_SCALE = .9                 # video_landmarks.extract_clip default
GL = np.diag([1., -1., -1.])    # OpenCV camera axes -> x right, y up, z toward the viewer


def ray_rotation(ray: np.ndarray) -> np.ndarray:
    """Minimal rotation taking the optical axis (0, 0, 1) onto ``ray`` (OpenCV axes)."""
    z = np.array([0., 0., 1.])
    r = ray / np.linalg.norm(ray)
    axis = np.cross(z, r)
    s, c = np.linalg.norm(axis), float(z @ r)
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def open_zip(store_name: str):
    if store_name == "r2":
        import boto3
        sys.path.insert(0, str(ROOT / "scripts"))
        from extract_video_states import StoreRangeFile
        s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"], region_name="auto")
        bucket = os.environ["R2_BUCKET"]
        provenance = json.loads(s3.get_object(Bucket=bucket, Key=ZIP_KEY + ".provenance.json")["Body"].read())
        if not provenance.get("verified_by_r2_reread"):
            raise SystemExit("the archived zip has no verified provenance")
        return zipfile.ZipFile(io.BufferedReader(StoreRangeFile(s3, bucket, ZIP_KEY), 8 << 20)), provenance["sha256"]
    path = Path(store_name)
    return zipfile.ZipFile(path), hashlib.sha256(path.read_bytes()).hexdigest()


def features(args) -> int:
    import cv2
    import mediapipe as mp
    from scipy.io import loadmat
    from signtranslator.data_engineering.video_landmarks import _polar_rotation, square_crop, verify_models
    archive, zip_sha = open_zip(args.zip)
    models = verify_models(args.mediapipe_models)
    vision = mp.tasks.vision
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(models["face_landmarker.task"])),
        running_mode=vision.RunningMode.IMAGE, output_face_blendshapes=True,
        output_facial_transformation_matrixes=True, num_faces=1))
    rows = dict(subject=[], found=[], scores=[], head=[], gaze=[], ipd_px=[])
    names = None
    for subject in range(15):
        base = f"MPIIFaceGaze/p{subject:02d}"
        camera = loadmat(io.BytesIO(archive.read(f"{base}/Calibration/Camera.mat")))
        K, dist = np.asarray(camera["cameraMatrix"], float), np.asarray(camera["distCoeffs"], float).ravel()
        lines = archive.read(f"{base}/p{subject:02d}.txt").decode().split("\n")
        lines = [l.split() for l in lines if l.strip()][:args.limit or None]
        for fields in lines:
            values = np.array(fields[1:27], dtype=float)
            corners = values[2:14].reshape(6, 2)
            face_centre, target = values[20:23], values[23:26]
            undistorted = cv2.undistortPoints(corners[:4].reshape(-1, 1, 2), K, dist, P=K).reshape(4, 2)
            eyes_px = np.stack((undistorted[:2].mean(0), undistorted[2:].mean(0)))   # image-left eye first
            rays = np.c_[eyes_px, np.ones(2)] @ np.linalg.inv(K).T
            eyes = rays * (face_centre[2] / rays[:, 2:])                                 # at the face centre's depth
            ipd = float(np.linalg.norm(eyes_px[0] - eyes_px[1]))
            image = cv2.imdecode(np.frombuffer(archive.read(f"{base}/{fields[0]}"), np.uint8), cv2.IMREAD_COLOR)
            scale = HOW2SIGN_IPD_PX / ipd
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            centre = (K @ face_centre)[:2] / face_centre[2] * scale
            crop, offset = square_crop(rgb, centre, FACE_SCALE * CROP_PER_IPD * HOW2SIGN_IPD_PX)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            found = bool(result.face_blendshapes)
            crop_centre = (offset + np.array(crop.shape[1::-1]) / 2) / scale                # original pixels
            to_crop = GL @ ray_rotation(np.linalg.inv(K) @ np.r_[crop_centre, 1.]).T
            # anatomical right eye is image-left for a camera facing the participant
            gaze = {"Right": target - eyes[0], "Left": target - eyes[1]}
            rows["gaze"].append(np.stack([to_crop @ (gaze[s] / np.linalg.norm(gaze[s])) for s in SIDES]))
            rows["subject"].append(subject)
            rows["found"].append(found)
            rows["ipd_px"].append(ipd)
            if found:
                current = tuple(c.category_name for c in result.face_blendshapes[0])
                names = names or current
                if current != names:
                    raise SystemExit("blendshape categories changed")
                rows["scores"].append([c.score for c in result.face_blendshapes[0]])
                rows["head"].append(_polar_rotation(np.asarray(result.facial_transformation_matrixes[0])[None, :3, :3])[0])
            else:
                rows["scores"].append(np.zeros(len(names or ()) or 52))
                rows["head"].append(np.eye(3))
        print(json.dumps(dict(subject=subject, images=len(lines), found=int(np.sum(np.array(rows["found"])[
            np.array(rows["subject"]) == subject])))), flush=True)
    landmarker.close()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, names=np.array(names), zip_sha256=zip_sha, **{k: np.asarray(v) for k, v in rows.items()})
    payload = buffer.getvalue()
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    Path(args.features).write_bytes(payload)
    if args.zip == "r2":
        from train_text_to_motion import make_store
        make_store("r2").put(FEATURES_KEY, payload)
    print(json.dumps(dict(features=args.features, sha256=hashlib.sha256(payload).hexdigest())))
    return 0


def fit(args) -> int:
    data = np.load(args.features)
    names, ok = tuple(data["names"]), data["found"]
    subjects, scores, head, gaze = data["subject"][ok], data["scores"][ok], data["head"][ok], data["gaze"][ok]
    side_index = {s: i for i, s in enumerate(SIDES)}
    per_eye = gaze
    bisector = per_eye.sum(1) / np.linalg.norm(per_eye.sum(1), axis=-1, keepdims=True)
    gaze = np.stack((bisector, bisector), axis=1)          # both eye models fit the cyclopean direction
    # Which MediaPipe suffix is which anatomical eye: horizontal gaze toward the
    # participant's left (camera +x in these axes when facing it) is "out" for the left eye.
    local = np.einsum("tji,tj->ti", head, gaze[:, side_index["Left"]])
    corr = {suffix: float(np.corrcoef(eye_features(names, scores, suffix)[:, 1] - eye_features(names, scores, suffix)[:, 0],
                                      local[:, 0])[0, 1]) for suffix in SIDES}
    suffix = {"Left": "Left", "Right": "Right"} if corr["Left"] > corr["Right"] else {"Left": "Right", "Right": "Left"}

    def fitted(train, variant):
        angles = head_angles(head[train])
        limits = tuple(map(tuple, np.percentile(angles, [1, 99], axis=0).T))
        empty = GazeModel(suffix, {}, {}, {}, variant, limits)
        coefficients = {s: fit_eye(empty.design(s, names, scores[train], head[train]), head[train],
                                   gaze[train, side_index[s]]) for s in SIDES}
        return GazeModel(suffix, {s: c[0] for s, c in coefficients.items()}, {s: c[1] for s, c in coefficients.items()},
                         {}, variant, limits)
    errors = {k: {s: [] for s in SIDES} for k in (*VARIANTS, "head_only", "head_plus_offset")}
    per_eye_errors = {s: [] for s in SIDES}
    for subject in np.unique(subjects):
        train, test = subjects != subject, subjects == subject
        for variant in VARIANTS:
            predicted = fitted(train, variant).predict(names, scores[test], head[test])
            for s in SIDES:
                errors[variant][s].append(angle_degrees(predicted[s], gaze[test, side_index[s]]))
                if variant == args.variant:
                    per_eye_errors[s].append(angle_degrees(predicted[s], per_eye[test, side_index[s]]))
        for s in SIDES:
            truth = gaze[test, side_index[s]]
            errors["head_only"][s].append(angle_degrees(head[test][:, :, 2], truth))
            local_train = np.einsum("tji,tj->ti", head[train], gaze[train, side_index[s]])
            offset = local_train.mean(0) / np.linalg.norm(local_train.mean(0))
            errors["head_plus_offset"][s].append(angle_degrees(np.einsum("tij,j->ti", head[test], offset), truth))
    summary = {}
    for kind, per_side in errors.items():
        summary[kind] = {}
        for s, parts in per_side.items():
            per_subject = [float(np.mean(p)) for p in parts]
            values = np.concatenate(parts)
            summary[kind][s] = dict(mean_deg=float(values.mean()), median_deg=float(np.median(values)),
                                    p90_deg=float(np.percentile(values, 90)), worst_participant_mean_deg=max(per_subject))
    chosen = args.variant
    summary["model"] = summary[chosen]
    summary["model_against_each_eye_at_calibration_distance"] = {
        s: dict(mean_deg=float(np.concatenate(v).mean()), median_deg=float(np.median(np.concatenate(v))))
        for s, v in per_eye_errors.items()}
    final = fitted(np.ones(len(subjects), bool), chosen)
    evaluation = dict(protocol="leave-one-participant-out on MPIIFaceGaze, images scaled to How2Sign face resolution; "
                               "errors against the cyclopean direction (both channels carry it)",
                      images=int(ok.sum()), face_found_rate=float(ok.mean()), suffix_correlation=corr,
                      variant=chosen, results=summary,
                      head_range_deg=np.degrees(np.asarray(final.head_range)).round(2).tolist(),
                      feature_percentiles={f"{name}{s}": np.percentile(eye_features(names, scores, s)[:, k], [5, 50, 95]).round(3).tolist()
                                           for s in SIDES for k, name in enumerate(("lookIn", "lookOut", "lookUp", "lookDown"))}, zip_sha256=str(data["zip_sha256"]), features_sha256=hashlib.sha256(
                          Path(args.features).read_bytes()).hexdigest())
    model = GazeModel(final.suffix, final.yaw, final.pitch, evaluation, chosen, final.head_range)
    text = model.to_json() + "\n"
    Path(args.output).write_text(text)
    if args.store == "r2":
        from train_text_to_motion import make_store
        make_store("r2").put(MODEL_KEY, text.encode())
    print(json.dumps(evaluation, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("features", "fit"))
    parser.add_argument("--zip", default="r2", help="'r2' or a local MPIIFaceGaze.zip")
    parser.add_argument("--mediapipe-models", type=Path)
    parser.add_argument("--features", required=True, help="local features .npz (written by 'features', read by 'fit')")
    parser.add_argument("--limit", type=int, default=0, help="images per participant (0: all)")
    parser.add_argument("--output", default=str(ROOT / "signtranslator/data_engineering/gaze_model.json"))
    parser.add_argument("--store", default="r2")
    parser.add_argument("--variant", default="monocular", choices=VARIANTS,
                        help="monocular (the best head-free variant; the cyclopean mean already pools both eyes): "
                             "73%% of How2Sign head poses lie outside MPIIFaceGaze's range, so a head term would "
                             "rest on clipped, unsupported covariates")
    args = parser.parse_args(argv)
    return features(args) if args.stage == "features" else fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
