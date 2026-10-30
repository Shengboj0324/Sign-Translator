"""Train the image-to-world landmark regressors (``world_regressor``) on collected pairs, split by signer (local CPU).

Reads the ``.npz`` pairs of ``collect_world_pairs.py``. Validation signers (``--val-signers``) are never trained on.
Reports, on validation frames, the quantities the canonical states carry: our IK's finger rotations (local, rooted at
the palm) and palm orientation from regressed against true world hand points, and the upper-arm/forearm/chest
rotations (upright chest) from regressed against true world pose points, with the visibility head's agreement with
MediaPipe's visibility >= 0.5; the geometric pseudo-world (``semlex_pose.pseudo_world`` at its declared gains) is
scored the same way as the baseline. Writes the model bundle and a JSON report.

With ``--legacy-dir`` (regressors v2) the *inputs* are legacy Holistic's image landmarks on the same frames
(``legacy_holistic_arrays.py``, file names matching the pairs), and the targets stay our Tasks world landmarks and
visibility: a pose frame needs both poses, a hand frame both hands of that pose wrist, with the two wrists within
``HAND_MATCH`` (normalised image units) so that both trackers see the same hand. How2Sign signer 9 (the generator's
final-test signer) is refused outright. The report records the training settings, a manifest of every pairs file and
legacy file used (SHA-256, signer, split) and the bundle SHA-256.

With ``--features xy`` (``world_regressor.FEATURES_XY``, for 2D-only sources such as the UWB YouTube-ASL keypoints) the
same pairs train z-free models: the inputs' z (Tasks or legacy Holistic) is dropped by the feature functions, which
read only ``x, y``; targets, losses and the validation metrics are unchanged. The geometric baseline is then the
pseudo-world with zero depth (``(x, y / a, 0)``: the only geometric reading of 2D points), and a ``--baseline-bundle``
(e.g. regressors v2) is scored on its own feature version, i.e. *with* the z this variant gives up. Default ``v1``
leaves every output of earlier runs unchanged.

``--features xy2`` (``FEATURES_XY2``) drops the xy-v1 pose context (absolute crop framing), so the pose model and its
visibility head cannot read the frame. ``--augment-framing K`` adds K re-framed copies of every *training* pose frame:
the image points (in image widths) are scaled by ``s`` (log-uniform in ``AUGMENT_SCALE``), placed with the shoulder
midpoint uniform in ``AUGMENT_MIDPOINT`` of the new frame, whose shape is drawn from ``AUGMENT_ASPECTS``; world and
visibility targets are unchanged. For xy-v1 it spreads the context inputs over full-frame framings; for xy-v2 the
features are invariant, so it changes nothing (and is refused as a no-op). Validation frames are never augmented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signtranslator.data_engineering import world_regressor as wr  # noqa: E402

ARMS = ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "Chest")
EXCLUDED_SIGNERS = ("h2s-9",)       # How2Sign signer 9: the generator's untouched final-test signer
HAND_MATCH = .08


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pairs(directory: Path, legacy_dir: Path | None = None, manifest: list | None = None):
    hands = {"Left": [], "Right": []}
    pose = []
    skipped = dict(no_legacy=0, frame_count=0)
    for path in sorted(Path(directory).glob("*.npz")):
        if "-9-rgb_front" in path.name:
            raise SystemExit(f"{path.name}: How2Sign signer 9 is excluded from regressor data")
        legacy = None
        if legacy_dir is not None:
            legacy_path = Path(legacy_dir) / path.name
            if not legacy_path.exists():
                skipped["no_legacy"] += 1
                continue
            with np.load(legacy_path, allow_pickle=False) as l:
                legacy = l["holistic"].astype(np.float64)
                legacy_size = l["size"].tolist()
        with np.load(path, allow_pickle=False) as z:
            signer, (w, h) = str(z["signer"]), z["size"].tolist()
            if signer in EXCLUDED_SIGNERS:
                raise SystemExit(f"{path.name}: signer {signer} is excluded from regressor data")
            a = w / h
            found = z["pose_found"].copy()
            pose_img, hand_img, hand_found = z["pose_img"].astype(np.float64), z["hand_img"].astype(np.float64), \
                z["hand_found"].copy()
            if legacy is not None:
                if len(legacy) != len(found) or legacy_size != [w, h]:
                    skipped["frame_count"] += 1
                    continue
                legacy_pose = legacy[:, 478:511]
                found &= np.isfinite(legacy_pose).all((1, 2))
                pose_img = legacy_pose
                hand_img = np.stack((legacy[:, 511:532], legacy[:, 532:553]), axis=1)
                wrist_gap = np.linalg.norm(np.nan_to_num(hand_img[:, :, 0, :2]) - z["hand_img"][:, :, 0, :2], axis=-1)
                hand_found &= np.isfinite(hand_img).all((2, 3)) & (wrist_gap < HAND_MATCH)
            if manifest is not None:
                manifest.append(dict(pairs=path.name, pairs_sha256=sha256_file(path), signer=signer,
                                     legacy_sha256=sha256_file(Path(legacy_dir) / path.name) if legacy_dir else None,
                                     pose_frames=int(found.sum()), hand_frames=[int(v) for v in hand_found.sum(0)]))
            if found.any():
                pose.append(dict(signer=signer, aspect=a, image=pose_img[found],
                                 world=z["pose_world"][found].astype(np.float64),
                                 vis=z["pose_vis"][found].astype(np.float64)))
            for k, side in enumerate(("Left", "Right")):
                ok = hand_found[:, k]
                if ok.any():
                    hands[side].append(dict(signer=signer, aspect=a, image=hand_img[ok, k],
                                            world=z["hand_world"][ok, k].astype(np.float64)))
    return hands, pose, skipped


FEATURES = {"v1": wr.FEATURES_VERSION, "xy": wr.FEATURES_XY, "xy2": wr.FEATURES_XY2}
AUGMENT_SCALE = (.3, 1.2)                  # person size relative to the crop it was collected in
AUGMENT_MIDPOINT = ((.2, .8), (.1, .7))    # shoulder midpoint x (widths), y (heights) in the new frame
AUGMENT_ASPECTS = (16 / 9, 4 / 3, 1., 9 / 16)


def augment_framing(chunks: list, copies: int, seed: int) -> list:
    """``copies`` re-framed versions of every pose chunk (see the module docstring); targets unchanged."""
    rng = np.random.default_rng(seed)
    out = []
    for c in chunks:
        q = np.array(c["image"], dtype=np.float64)
        q[..., 1] /= c["aspect"]                                    # image widths
        mid = (q[:, 11, :2] + q[:, 12, :2]) / 2
        for _ in range(copies):
            n = len(q)
            aspect = rng.choice(AUGMENT_ASPECTS)
            scale = np.exp(rng.uniform(*np.log(AUGMENT_SCALE), n))
            target = np.stack((rng.uniform(*AUGMENT_MIDPOINT[0], n), rng.uniform(*AUGMENT_MIDPOINT[1], n) / aspect), -1)
            moved = q.copy()
            moved[..., :2] = (q[..., :2] - mid[:, None]) * scale[:, None, None] + target[:, None]
            moved[..., 1] *= aspect                                 # back to normalised image y
            out.append(dict(c, image=moved, aspect=float(aspect), augmented=True))
    return out


def hand_arrays(chunks, features: str = wr.FEATURES_VERSION):
    xs, ys, rot, world, image, aspect = [], [], [], [], [], []
    for c in chunks:
        x, r, _, ok = wr.FEATURE_FUNCTIONS[features][0](c["image"], c["aspect"])
        xs.append(x[ok]); ys.append(wr.hand_target(c["world"][ok], r[ok]).reshape(ok.sum(), -1)); rot.append(r[ok])
        world.append(c["world"][ok]); image.append(c["image"][ok]); aspect.append(np.full(ok.sum(), c["aspect"]))
    return [np.concatenate(v) for v in (xs, ys, rot, world, image, aspect)]


def pose_arrays(chunks, features: str = wr.FEATURES_VERSION):
    xs, ys, vis, world, image, aspect = [], [], [], [], [], []
    for c in chunks:
        x, ok = wr.FEATURE_FUNCTIONS[features][1](c["image"], c["aspect"])
        xs.append(x[ok]); ys.append(wr.pose_target(c["world"][ok]).reshape(ok.sum(), -1)); vis.append(c["vis"][ok])
        world.append(c["world"][ok]); image.append(c["image"][ok]); aspect.append(np.full(ok.sum(), c["aspect"]))
    return [np.concatenate(v) for v in (xs, ys, vis, world, image, aspect)]


def fit(x, y, *, bones, points, vis=None, epochs=40, batch=2048, lr=1e-3, direction_weight=1., seed=0, log=""):
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    vt = None if vis is None else torch.from_numpy(vis.astype(np.float32))
    model = wr.MLP(x.shape[1], y.shape[1] + (0 if vis is None else vis.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for epoch in range(epochs):
        order = torch.randperm(len(x))
        total = 0.
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            out = model(xt[idx])
            pred = out[:, :y.shape[1]].reshape(len(idx), points, 3)
            true = yt[idx].reshape(len(idx), points, 3)
            loss = torch.nn.functional.mse_loss(pred, true) + direction_weight * wr.direction_loss(pred, true, bones)
            if vt is not None:
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out[:, y.shape[1]:], vt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 10 == 9 or epoch == epochs - 1:
            print(f"{log} epoch {epoch + 1} loss {total / len(x):.4f}", flush=True)
    return model.eval(), mean.astype(np.float32), std.astype(np.float32)


def angles(a, b):
    cos = (np.einsum("...ij,...ij->...", a, b) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1., 1.)))


def stat(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                mean=round(float(v.mean()), 2)) if len(v) else dict(n=0)


def finger_errors(true_points, other_points, side):
    """Local finger rotation and palm-frame angle errors (deg) between two (N, 21, 3) point sets."""
    from signtranslator.data_engineering.landmark_kinematics import frame_from_axes, solve_chain
    from signtranslator.data_engineering.video_landmarks import finger_joints
    out = {}
    for name, pts in (("true", true_points), ("other", other_points)):
        frame, ok = frame_from_axes(pts[:, 5] - pts[:, 17], pts[:, 9] - pts[:, 0])
        solved = solve_chain(pts, {"Hand": (frame, ok)}, finger_joints(side))
        out[name] = (frame, ok, solved)
    (ft, okt, st), (fo, oko, so) = out["true"], out["other"]
    fingers = []
    for j in finger_joints(side):
        both = st[j.name][2] & so[j.name][2]
        fingers += angles(st[j.name][0][both], so[j.name][0][both]).tolist()
    both = okt & oko
    return fingers, angles(ft[both], fo[both]).tolist()


def arm_errors(true_world, other_world, visible):
    """Upright-chest local rotation errors (deg) of arms and chest between two (N, 33, 3) MediaPipe-axes point sets."""
    from signtranslator.data_engineering.video_landmarks import Y_UP, solve_rotations
    r_true, _ = solve_rotations(true_world @ Y_UP.T, visible, {}, upright_chest=True)
    r_other, _ = solve_rotations(other_world @ Y_UP.T, visible, {}, upright_chest=True)
    out = {}
    for j in ARMS:
        both = r_true[j][1] & r_other[j][1]
        out[j] = angles(r_true[j][0][both], r_other[j][0][both]).tolist()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", required=True)
    p.add_argument("--val-signers", default="h2s-1,h2s-2",
                   help="one source-video component: How2Sign signers 1 and 2 share source videos")
    p.add_argument("--legacy-dir", help="legacy Holistic arrays matching the pairs files (regressors v2 inputs)")
    p.add_argument("--baseline-bundle", help="also score this bundle on the same validation inputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--direction-weight", type=float, default=1.)
    p.add_argument("--max-val-frames", type=int, default=30000)
    p.add_argument("--features", choices=sorted(FEATURES), default="v1",
                   help="v1: (x, y/a, z) inputs (bundles v1, v2); xy: z-free inputs for 2D-only sources; xy2: z-free "
                        "and framing-invariant (no pose context)")
    p.add_argument("--augment-framing", type=int, default=0, help="re-framed copies of every training pose frame")
    p.add_argument("--out", required=True, help="model bundle path (written once)")
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    torch.set_num_threads(1)
    from signtranslator.data_engineering.semlex_pose import HAND_Z_SCALE, POSE_Z_SCALE, pseudo_world
    features = FEATURES[args.features]
    if args.augment_framing and features in (wr.FEATURES_VERSION, wr.FEATURES_XY2):
        raise SystemExit(f"--augment-framing changes nothing a {features!r} model reads; use it with --features xy")
    pose_k, hand_k = (POSE_Z_SCALE, HAND_Z_SCALE) if features == wr.FEATURES_VERSION else (0., 0.)
    val_signers = set(args.val_signers.split(","))
    if val_signers & set(EXCLUDED_SIGNERS):
        raise SystemExit("signer 9 may not be read, so it cannot validate either")
    manifest = []
    hands, pose, skipped = load_pairs(Path(args.pairs), Path(args.legacy_dir) if args.legacy_dir else None, manifest)
    for m in manifest:
        m["split"] = "val" if m["signer"] in val_signers else "train"
    baseline = None
    if args.baseline_bundle:
        baseline = wr.load_regressors(Path(args.baseline_bundle))
    report = dict(inputs="legacy-holistic" if args.legacy_dir else "tasks-image", val_signers=sorted(val_signers),
                  train_signers=sorted({c["signer"] for c in pose} - val_signers), excluded_signers=list(EXCLUDED_SIGNERS),
                  settings=dict(epochs=args.epochs, seed=args.seed, direction_weight=args.direction_weight, lr=1e-3,
                                batch=2048, width=512, depth=3, optimizer="AdamW(weight_decay=1e-4), OneCycleLR",
                                hand_match=HAND_MATCH, features=features),
                  skipped=skipped, pairs_manifest=manifest,
                  baseline_bundle_sha256=baseline.sha256 if baseline else None, hand={}, pose={})
    if features != wr.FEATURES_VERSION:                 # v1 reports stay exactly as before
        report["pseudo_world_depth_gains"] = dict(pose=pose_k, hand=hand_k)
    if args.augment_framing:
        report["settings"]["augment_framing"] = dict(copies=args.augment_framing, scale=list(AUGMENT_SCALE),
                                                     midpoint=[list(v) for v in AUGMENT_MIDPOINT],
                                                     aspects=[round(a, 4) for a in AUGMENT_ASPECTS])
    rng = np.random.default_rng(args.seed)
    models = {}
    for side in ("Left", "Right"):
        train = hand_arrays([c for c in hands[side] if c["signer"] not in val_signers], features)
        val = hand_arrays([c for c in hands[side] if c["signer"] in val_signers], features)
        model = fit(train[0], train[1], bones=wr.HAND_BONES, points=21, epochs=args.epochs,
                    direction_weight=args.direction_weight, seed=args.seed, log=f"hand {side}")
        models[side] = model
        keep = rng.permutation(len(val[0]))[:args.max_val_frames]
        regressors = wr.WorldRegressors(hand={side: model}, pose=None, features=features)
        image, aspect, world = val[4][keep], val[5][keep], val[3][keep]
        predicted = np.concatenate([regressors.hands(image[aspect == a], a, side)[0] for a in np.unique(aspect)])
        order = np.concatenate([np.flatnonzero(aspect == a) for a in np.unique(aspect)])
        world, image, aspect = world[order], image[order], aspect[order]
        pseudo = np.concatenate([pseudo_world(image[aspect == a], a, hand_k) for a in np.unique(aspect)])
        truth = world - world[:, :1]
        fr, palm = finger_errors(truth, predicted, side)
        fb, palm_b = finger_errors(truth, pseudo - pseudo[:, :1], side)
        report["hand"][side] = dict(train_frames=int(len(train[0])), val_frames=int(len(val[0])),
                                    scored_frames=int(len(keep)), regressor=dict(fingers=stat(fr), palm=stat(palm)),
                                    pseudo_world=dict(fingers=stat(fb), palm=stat(palm_b)))
        if baseline is not None:
            other = np.concatenate([baseline.hands(image[aspect == a], a, side)[0] for a in np.unique(aspect)])
            fo, palm_o = finger_errors(truth, other, side)
            report["hand"][side]["baseline_bundle"] = dict(fingers=stat(fo), palm=stat(palm_o))
        print(side, json.dumps(report["hand"][side]), flush=True)
    train_chunks = [c for c in pose if c["signer"] not in val_signers]
    if args.augment_framing:
        train_chunks = train_chunks + augment_framing(train_chunks, args.augment_framing, args.seed)
    train = pose_arrays(train_chunks, features)
    val = pose_arrays([c for c in pose if c["signer"] in val_signers], features)
    pose_model = fit(train[0], train[1], bones=wr.POSE_BONES, points=wr.POSE_POINTS, vis=train[2],
                     epochs=args.epochs, direction_weight=args.direction_weight, seed=args.seed, log="pose")
    regressors = wr.WorldRegressors(hand=models, pose=pose_model, features=features)
    payload = regressors.save()                     # written before the evaluation, so a reporting error loses nothing
    with open(args.out, "xb") as stream:
        stream.write(payload)
    report["bundle_sha256"] = hashlib.sha256(payload).hexdigest()
    keep = rng.permutation(len(val[0]))[:args.max_val_frames]
    image, aspect, world, vis = val[4][keep], val[5][keep], val[3][keep], val[2][keep]
    groups = [np.flatnonzero(aspect == a) for a in np.unique(aspect)]
    order = np.concatenate(groups)
    outs = [regressors.pose_world(image[g], aspect[g[0]]) for g in groups]
    predicted = np.concatenate([o[0] for o in outs])
    predicted_vis = np.concatenate([o[1] for o in outs])
    image, aspect, world, vis = image[order], aspect[order], world[order], vis[order]
    pseudo = np.concatenate([pseudo_world(image[aspect == a], a, pose_k) for a in np.unique(aspect)])
    visible = vis >= .5
    regressed = arm_errors(world, predicted, visible)
    pseudo_errors = arm_errors(world, pseudo, visible)
    agreement = {str(j): round(float(((predicted_vis[:, j] >= .5) == visible[:, j]).mean()), 4) for j in (11, 12, 13, 14, 15, 16)}
    inside = ((image[..., :2] >= 0) & (image[..., :2] <= 1)).all(-1)
    inside_agreement = {str(j): round(float((inside[:, j] == visible[:, j]).mean()), 4) for j in (11, 12, 13, 14, 15, 16)}
    report["pose"] = dict(train_frames=int(len(train[0])), val_frames=int(len(val[0])), scored_frames=int(len(keep)),
                          regressor={j: stat(v) for j, v in regressed.items()},
                          pseudo_world={j: stat(v) for j, v in pseudo_errors.items()},
                          visibility_agreement=agreement, inside_image_agreement=inside_agreement)
    if baseline is not None:
        outs_b = [baseline.pose_world(image[aspect == a], a) for a in np.unique(aspect)]
        report["pose"]["baseline_bundle"] = {j: stat(v) for j, v in arm_errors(
            world, np.concatenate([o[0] for o in outs_b]), visible).items()}
        vis_b = np.concatenate([o[1] for o in outs_b])
        report["pose"]["baseline_visibility_agreement"] = {
            str(j): round(float(((vis_b[:, j] >= .5) == visible[:, j]).mean()), 4) for j in (11, 12, 13, 14, 15, 16)}
    print("pose", json.dumps(report["pose"]), flush=True)
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print("bundle", report["bundle_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
