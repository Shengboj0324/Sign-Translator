"""Do our legacy-Holistic arrays look like Sem-Lex's? Distribution statistics of (T, 553, 3) arrays (local).

Compares three sets in Sem-Lex's layout: real Sem-Lex pose files, legacy Holistic (``legacy_holistic_arrays.py``) on
the How2Sign Sem-Lex-framed crops, and the Tasks emulation (``validate_semlex_adapter.py``) on similar crops. Framing,
signers and signs differ between Sem-Lex and the How2Sign crops, so only statistics that are properties of the
estimator, not of the scene, are informative:

- iris depth: whether the five points of each iris share one z (legacy Holistic's iris model writes a single depth);
- the face mesh's depth spread relative to its width, and its aspect-estimate bias (``estimate_aspect``'s raw value
  over the true shape; Sem-Lex's is measured against the nearest common shape);
- hand depth spread relative to hand width, and the pose's upper-body depth spread relative to its width;
- partial blocks (a block partly NaN) and all-zero face blocks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def stats(arrays: list[np.ndarray], aspects: list[float | None], resources) -> dict:
    from signtranslator.data_engineering.semlex_pose import STANDARD_ASPECTS, aspect_per_frame
    iris_spread, face_depth, hand_depth, pose_depth, bias = [], [], [], [], []
    partial = zero_faces = frames = 0
    for array, true in zip(arrays, aspects):
        a = array.astype(np.float64)
        frames += len(a)
        for block in (slice(0, 478), slice(478, 511), slice(511, 532), slice(532, 553)):
            fin = np.isfinite(a[:, block]).all(-1)
            partial += int((fin.any(1) & ~fin.all(1)).sum())
        face = a[:, :478]
        ok = np.isfinite(face).all((1, 2))
        zero = ok & (np.ptp(np.where(ok[:, None, None], face, 0.)[..., 0], axis=1) == 0)
        zero_faces += int(zero.sum())
        ok &= ~zero
        if ok.any():
            f = face[ok]
            iris_spread += np.concatenate([np.ptp(f[:, 468:473, 2], axis=1), np.ptp(f[:, 473:478, 2], axis=1)]).tolist()
            face_depth += (np.ptp(f[:, :468, 2], axis=1) / np.ptp(f[:, :468, 0], axis=1)).tolist()
            raw = np.nanmedian(aspect_per_frame(f[:, :, :2], resources))
            reference = true if true is not None else min(STANDARD_ASPECTS.values(), key=lambda s: abs(np.log(raw / s)))
            bias.append(raw / reference - 1)
        for block in (slice(511, 532), slice(532, 553)):
            h = a[:, block]
            okh = np.isfinite(h).all((1, 2))
            if okh.any():
                hand_depth += (np.ptp(h[okh][..., 2], axis=1) / np.maximum(np.ptp(h[okh][..., 0], axis=1), 1e-9)).tolist()
        p = a[:, 478:501]
        okp = np.isfinite(p).all((1, 2))
        if okp.any():
            pose_depth += (np.ptp(p[okp][..., 2], axis=1) / np.maximum(np.ptp(p[okp][..., 0], axis=1), 1e-9)).tolist()

    def q(v):
        return [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] if len(v) else None
    return dict(clips=len(arrays), frames=frames, partial_blocks=partial, all_zero_faces=zero_faces,
                iris_z_spread_p10_p50_p90=q(iris_spread), face_z_over_x=q(face_depth), hand_z_over_x=q(hand_depth),
                pose_upper_z_over_x=q(pose_depth), aspect_raw_bias=q(bias))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--semlex-dir", required=True)
    p.add_argument("--legacy-dir", required=True)
    p.add_argument("--tasks-dir", required=True, help="validate_semlex_adapter extract outputs (holistic arrays)")
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--models-dir", default=str(Path("~/.sign-translator-ops/models").expanduser()))
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    from signtranslator.data_engineering.semlex_pose import load_face_resources
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    semlex = [np.load(f, allow_pickle=False) for f in sorted(Path(args.semlex_dir).glob("*.npy"))[:args.limit]]
    legacy, legacy_aspect = [], []
    for f in sorted(Path(args.legacy_dir).glob("*.npz"))[:args.limit]:
        with np.load(f, allow_pickle=False) as z:
            legacy.append(z["holistic"].copy())
            w, h = z["size"].tolist()
            legacy_aspect.append(w / h)
    tasks, tasks_aspect = [], []
    for f in sorted(Path(args.tasks_dir).glob("*.npz"))[:args.limit]:
        with np.load(f, allow_pickle=False) as z:
            if "holistic" not in z.files:
                continue
            meta = json.loads(str(z["meta"]))
            tasks.append(z["holistic"].copy())
            tasks_aspect.append(meta["size"][0] / meta["size"][1])
    report = dict(semlex=stats(semlex, [None] * len(semlex), resources),
                  legacy_holistic=stats(legacy, legacy_aspect, resources),
                  tasks_emulation=stats(tasks, tasks_aspect, resources))
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
