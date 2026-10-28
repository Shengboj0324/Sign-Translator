"""Run legacy MediaPipe Holistic (``mediapipe.solutions.holistic``) as Sem-Lex did; write Sem-Lex-layout arrays.

Runs in its own venv (``~/.venvs/semlex-legacy-mp``: Python 3.11, mediapipe==0.10.14 and av==12.3.0 from PyPI); it
imports neither this repository nor torch. Only ``model_complexity=1`` (``pose_landmark_full.tflite``, shipped in the
wheel) is allowed: complexity 0 and 2 make mediapipe download a model from Google storage at first use, which this
project does not do. ``refine_face_landmarks=True`` gives the 478-point mesh (468 + 10 iris) of Sem-Lex's layout.

Input: a JSON list of jobs ``{"video": path, "box": [x0, y0, w, h] or null, "out": path}``. Each video is decoded with
PyAV in order (frames with a missing or non-increasing timestamp are skipped, as ``extract_clip`` does), cropped to the
box in memory, and passed to Holistic in video mode (``static_image_mode=False``, the default smoothing). Output per
job: ``.npz`` with ``holistic`` (T, 553, 3) float16 (face 0..477, pose 478..510, left hand 511..531, right hand
532..552; NaN where not detected), ``pose_visibility`` (T, 33) float16 (diagnostic: Sem-Lex stored none), ``times``
(T,) float64 and ``size`` (w, h).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

COMPLEXITY = 1


def model_hashes() -> dict:
    import mediapipe as mp
    root = Path(mp.__file__).parent / "modules"
    names = ("pose_landmark/pose_landmark_full.tflite", "pose_detection/pose_detection.tflite",
             "face_landmark/face_landmark_with_attention.tflite", "hand_landmark/hand_landmark_full.tflite",
             "holistic_landmark/hand_recrop.tflite", "face_detection/face_detection_short_range.tflite",
             "palm_detection/palm_detection_full.tflite")
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}


def run(job: dict) -> dict:
    import av
    import mediapipe as mp
    out = Path(job["out"])
    if out.exists():
        return dict(out=str(out), status="present")
    modules = Path(mp.__file__).parent / "modules"
    if not (modules / "pose_landmark/pose_landmark_full.tflite").exists():
        raise RuntimeError("pose_landmark_full.tflite is missing from the wheel")
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=COMPLEXITY,
                                              refine_face_landmarks=True)
    rows, vis, times, size = [], [], [], None
    box = job.get("box")
    try:
        with av.open(job["video"]) as container:
            stream = container.streams.video[0]
            last = -float("inf")
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * frame.time_base)
                if t <= last:
                    continue
                last = t
                rgb = frame.to_ndarray(format="rgb24")
                if box is not None:
                    x0, y0, w, h = box
                    rgb = rgb[y0:y0 + h, x0:x0 + w]
                rgb = np.ascontiguousarray(rgb)
                size = (rgb.shape[1], rgb.shape[0])
                r = holistic.process(rgb)
                row = np.full((553, 3), np.nan)
                v = np.zeros(33)
                for landmarks, start in ((r.face_landmarks, 0), (r.pose_landmarks, 478),
                                         (r.left_hand_landmarks, 511), (r.right_hand_landmarks, 532)):
                    if landmarks is not None:
                        points = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
                        row[start:start + len(points)] = points
                if r.pose_landmarks is not None:
                    v = np.array([l.visibility for l in r.pose_landmarks.landmark])
                rows.append(row)
                vis.append(v)
                times.append(t)
    finally:
        holistic.close()
    if not rows:
        return dict(out=str(out), status="no-frames")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, holistic=np.stack(rows).astype(np.float16), pose_visibility=np.stack(vis).astype(np.float16),
                        times=np.array(times), size=np.array(size))
    return dict(out=str(out), status="written", frames=len(rows))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jobs", required=True, help="JSON list of {video, box, out}")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--hashes", help="write the mediapipe version and model SHA-256s here")
    args = p.parse_args(argv)
    import mediapipe as mp
    if args.hashes:
        Path(args.hashes).write_text(json.dumps(dict(mediapipe=mp.__version__, model_complexity=COMPLEXITY,
                                                     refine_face_landmarks=True, models=model_hashes()), indent=1))
    jobs = json.loads(Path(args.jobs).read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, jobs, chunksize=2):
            print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
