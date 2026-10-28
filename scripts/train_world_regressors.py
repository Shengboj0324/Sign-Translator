"""Train the image-to-world landmark regressors (``world_regressor``) on collected pairs, split by signer (local CPU).

Reads the ``.npz`` pairs of ``collect_world_pairs.py``. Validation signers (``--val-signers``) are never trained on.
Reports, on validation frames, the quantities the canonical states carry: our IK's finger rotations (local, rooted at
the palm) and palm orientation from regressed against true world hand points, and the upper-arm/forearm/chest
rotations (upright chest) from regressed against true world pose points, with the visibility head's agreement with
MediaPipe's visibility >= 0.5; the geometric pseudo-world (``semlex_pose.pseudo_world`` at its declared gains) is
scored the same way as the baseline. Writes the model bundle and a JSON report.
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


def load_pairs(directory: Path):
    hands = {"Left": [], "Right": []}
    pose = []
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            signer, (w, h) = str(z["signer"]), z["size"].tolist()
            a = w / h
            found = z["pose_found"]
            if found.any():
                pose.append(dict(signer=signer, aspect=a, image=z["pose_img"][found].astype(np.float64),
                                 world=z["pose_world"][found].astype(np.float64),
                                 vis=z["pose_vis"][found].astype(np.float64)))
            for k, side in enumerate(("Left", "Right")):
                ok = z["hand_found"][:, k]
                if ok.any():
                    hands[side].append(dict(signer=signer, aspect=a, image=z["hand_img"][ok, k].astype(np.float64),
                                            world=z["hand_world"][ok, k].astype(np.float64)))
    return hands, pose


def hand_arrays(chunks):
    xs, ys, rot, world, image, aspect = [], [], [], [], [], []
    for c in chunks:
        x, r, _, ok = wr.hand_features(c["image"], c["aspect"])
        xs.append(x[ok]); ys.append(wr.hand_target(c["world"][ok], r[ok]).reshape(ok.sum(), -1)); rot.append(r[ok])
        world.append(c["world"][ok]); image.append(c["image"][ok]); aspect.append(np.full(ok.sum(), c["aspect"]))
    return [np.concatenate(v) for v in (xs, ys, rot, world, image, aspect)]


def pose_arrays(chunks):
    xs, ys, vis, world, image, aspect = [], [], [], [], [], []
    for c in chunks:
        x, ok = wr.pose_features(c["image"], c["aspect"])
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
    p.add_argument("--val-signers", default="h2s-2,h2s-4,h2s-9")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--direction-weight", type=float, default=1.)
    p.add_argument("--max-val-frames", type=int, default=30000)
    p.add_argument("--out", required=True, help="model bundle path (written once)")
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    torch.set_num_threads(1)
    from signtranslator.data_engineering.semlex_pose import HAND_Z_SCALE, POSE_Z_SCALE, pseudo_world
    val_signers = set(args.val_signers.split(","))
    hands, pose = load_pairs(Path(args.pairs))
    report = dict(val_signers=sorted(val_signers),
                  train_signers=sorted({c["signer"] for c in pose} - val_signers), hand={}, pose={})
    rng = np.random.default_rng(0)
    models = {}
    for side in ("Left", "Right"):
        train = hand_arrays([c for c in hands[side] if c["signer"] not in val_signers])
        val = hand_arrays([c for c in hands[side] if c["signer"] in val_signers])
        model = fit(train[0], train[1], bones=wr.HAND_BONES, points=21, epochs=args.epochs,
                    direction_weight=args.direction_weight, log=f"hand {side}")
        models[side] = model
        keep = rng.permutation(len(val[0]))[:args.max_val_frames]
        regressors = wr.WorldRegressors(hand={side: model}, pose=None)
        image, aspect, world = val[4][keep], val[5][keep], val[3][keep]
        predicted = np.concatenate([regressors.hands(image[aspect == a], a, side)[0] for a in np.unique(aspect)])
        order = np.concatenate([np.flatnonzero(aspect == a) for a in np.unique(aspect)])
        world, image, aspect = world[order], image[order], aspect[order]
        pseudo = np.concatenate([pseudo_world(image[aspect == a], a, HAND_Z_SCALE) for a in np.unique(aspect)])
        truth = world - world[:, :1]
        fr, palm = finger_errors(truth, predicted, side)
        fb, palm_b = finger_errors(truth, pseudo - pseudo[:, :1], side)
        report["hand"][side] = dict(train_frames=int(len(train[0])), val_frames=int(len(val[0])),
                                    scored_frames=int(len(keep)), regressor=dict(fingers=stat(fr), palm=stat(palm)),
                                    pseudo_world=dict(fingers=stat(fb), palm=stat(palm_b)))
        print(side, json.dumps(report["hand"][side]), flush=True)
    train = pose_arrays([c for c in pose if c["signer"] not in val_signers])
    val = pose_arrays([c for c in pose if c["signer"] in val_signers])
    pose_model = fit(train[0], train[1], bones=wr.POSE_BONES, points=wr.POSE_POINTS, vis=train[2],
                     epochs=args.epochs, direction_weight=args.direction_weight, log="pose")
    regressors = wr.WorldRegressors(hand=models, pose=pose_model)
    keep = rng.permutation(len(val[0]))[:args.max_val_frames]
    image, aspect, world, vis = val[4][keep], val[5][keep], val[3][keep], val[2][keep]
    groups = [np.flatnonzero(aspect == a) for a in np.unique(aspect)]
    order = np.concatenate(groups)
    outs = [regressors.pose_world(image[g], aspect[g[0]]) for g in groups]
    predicted = np.concatenate([o[0] for o in outs])
    predicted_vis = np.concatenate([o[1] for o in outs])
    image, aspect, world, vis = image[order], aspect[order], world[order], vis[order]
    pseudo = np.concatenate([pseudo_world(image[aspect == a], a, POSE_Z_SCALE) for a in np.unique(aspect)])
    visible = vis >= .5
    regressed = arm_errors(world, predicted, visible)
    baseline = arm_errors(world, pseudo, visible)
    agreement = {str(j): round(float(((predicted_vis[:, j] >= .5) == visible[:, j]).mean()), 4) for j in (11, 12, 13, 14, 15, 16)}
    inside = ((image[..., :2] >= 0) & (image[..., :2] <= 1)).all(-1)
    inside_agreement = {str(j): round(float((inside[:, j] == visible[:, j]).mean()), 4) for j in (11, 12, 13, 14, 15, 16)}
    report["pose"] = dict(train_frames=int(len(train[0])), val_frames=int(len(val[0])), scored_frames=int(len(keep)),
                          regressor={j: stat(v) for j, v in regressed.items()},
                          pseudo_world={j: stat(v) for j, v in baseline.items()},
                          visibility_agreement=agreement, inside_image_agreement=inside_agreement)
    print("pose", json.dumps(report["pose"]), flush=True)
    payload = regressors.save()
    with open(args.out, "xb") as stream:
        stream.write(payload)
    report["bundle_sha256"] = hashlib.sha256(payload).hexdigest()
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print("bundle", report["bundle_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
