"""UWB lifter v2: the YouTube-adapted pose regressor, its selection on lifter-val, and the validity head (CPU only).

Declared in docs/NEXT_PHASE_PLAN.md, "UWB lifter v2 and gate v2: declared" (2026-10-30, 11:12 UTC).

``pose`` — the pose regressor (one arm per run):

- **init** ``xy2`` (default): the xy2 bundle's pose MLP and its input normalisation (``--init-bundle``, SHA-pinned),
  fine-tuned ``--epochs 15`` at ``--lr 3e-4``; ``--from-scratch``: the comparison arm, a new MLP with the same shape
  (normalisation from its training inputs), 30 epochs at the original recipe's 1e-3;
- **batches 50/50**: How2Sign (the v2 pairs of training signers 3, 4, 5, 8, 11 with their legacy Holistic inputs, as
  ``train_world_regressors.load_pairs``, plus the training signers' tight crops: Tasks world and visibility targets),
  with the original loss of ``train_world_regressors.fit`` (position MSE + ``1 - cos`` on ``POSE_BONES`` + visibility
  BCE); and YouTube (the **train** part of the lifter-v2 pairs, ``build_uwb_lifter_pairs.py``): ``1 - cos`` between the
  predicted and our bone directions on the four arm bones and the shoulder line, where ours is valid. One epoch is one
  pass over the larger source; the other is cycled. AdamW (weight decay 1e-4), OneCycleLR, seed 0, batch 2048;
- **hands** stay the xy2 bundle's (frozen); ``--finetune-hands`` is the declared comparison: each side's hand MLP is
  fine-tuned the same way (How2Sign hand pairs, original loss; YouTube: ``1 - cos`` on our 15 finger bone directions
  in the camera frame, turned into the model's in-plane frame);
- **lifter-val** (the split's lifter-val part, never trained on) scores the arm: per bone group the angle between the
  predicted and our directions, beside the init bundle's on the same frames. How2Sign validation signers 1 and 2 and
  signer 9 are never read (1 and 2 are gate v2's How2Sign controls; 9 is the generator's final-test signer).

``select`` — the arm with the lower lifter-val **mean of the upper-arm and forearm medians** (amendment 1, 2026-10-30
11:51 UTC: not forearm-only, which could trade upper-arm accuracy away; ties to the fine-tuned arm). Selection reads
only lifter-val.

``validity`` — the validity head on the selected bundle: an MLP (256 x 2) on ``world_regressor.validity_features``
(xy-v2 pose features, image x, y and outside distance of points 11..16) -> 6 point logits; masked BCE, batches 50/50 of
YouTube train frames (labels: elbow = our upper arm valid, wrist = our forearm valid where the upper arm is, on frames
where our chest is valid; shoulders unlabelled) and the tight crops (Tasks visibility >= 0.5 on points 11..16), 15
epochs at 1e-3. **Thresholds per side on lifter-val**: on frames where our chest is valid, a side's forearm is kept when
shoulder, elbow and wrist all reach the threshold (as the solver chains them); the threshold maximises agreement with
our forearm validity subject to UWB-only <= 0.07 (ties to the higher threshold; a threshold above 1 keeps nothing, so
the constraint can always be met). The head is attached (``WorldRegressors.validity``) and the bundle written once.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

from signtranslator.data_engineering import world_regressor as wr  # noqa: E402

TRAIN_SIGNERS = ("h2s-3", "h2s-4", "h2s-5", "h2s-8", "h2s-11")
NEVER_READ = ("h2s-9",)
XY2_BUNDLE = dict(key="acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1/files/world-regressors-xy2-v1.pt",
                  sha256="02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde")
YT_BONES = ((11, 13), (13, 15), (12, 14), (14, 16), (12, 11))     # the pairs' ARM_TARGETS order; (12, 11): right -> left
BONE_GROUPS = dict(upper_arms=(0, 2), forearms=(1, 3), shoulder_line=(4,))
UWB_ONLY_MAX = .07
THRESHOLDS = tuple(np.round(np.arange(.02, .99, .01), 2)) + (1.01,)
VALIDITY_SHAPE = dict(width=256, depth=2)
SIDE_POINTS = {"Left": (0, 2, 4), "Right": (1, 3, 5)}        # columns of points (11..16): shoulder, elbow, wrist


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Data

def load_youtube(pairs_dir: Path, split: dict, split_sha: str, part: str, *, hands: bool = False) -> dict:
    """Frames of one part (``train`` or ``lifter_val``) of the lifter-v2 pairs; every shard's SHA-256 is checked
    against ``pairs-index.json``, the index must be of this split, and every pair's video must be in its part
    (a held-out video is refused)."""
    from build_uwb_lifter_pairs import load_shard
    if part not in ("train", "lifter_val"):
        raise SystemExit(f"part {part!r}: only train and lifter_val are read")
    index = json.loads((Path(pairs_dir) / "pairs-index.json").read_text())
    if index.get("split_sha256") != split_sha:
        raise SystemExit(f"pairs index of split {index.get('split_sha256')}, expected {split_sha}")
    held = set(split["held_out"]["videos"])
    members = {p: set(split[p]["videos"]) for p in ("train", "lifter_val")}
    keys = ["pose", "arm_dirs", "arm_valid", "chest_valid", "labels", "label_mask", "forearm_valid"]
    keys += ["hands", "hands_found", "finger_dirs", "finger_valid"] if hands else []
    chunks = {k: [] for k in keys + ["aspect", "video"]}
    shards = []
    for entry in index["shards"]:
        z = load_shard(Path(pairs_dir) / entry["file"], entry["sha256"])
        videos = [str(v) for v in z["pair_video"]]
        if held & set(videos):
            raise SystemExit(f"{entry['file']}: held-out videos {sorted(held & set(videos))[:3]} refused")
        for v, p in zip(videos, z["pair_part"]):
            if v not in members.get(str(p), ()):
                raise SystemExit(f"{entry['file']}: {v} is not a {p} video of the split")
        sizes = np.diff(z["pair_offset"])
        frame_part = np.repeat(z["pair_part"], sizes)
        keep = frame_part == part
        for k in keys:
            chunks[k].append(z[k][keep])
        chunks["aspect"].append(np.repeat(z["pair_aspect"], sizes)[keep])
        chunks["video"].append(np.repeat(z["pair_video"], sizes)[keep])
        shards.append(dict(file=entry["file"], sha256=entry["sha256"], frames=int(keep.sum())))
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in chunks.items()}
    out["shards"] = shards
    return out


def by_aspect(function, image: np.ndarray, aspect: np.ndarray):
    """``function(image[g], a)`` for every distinct aspect ``a``, scattered back to the input order."""
    results = None
    for a in np.unique(aspect):
        rows = np.flatnonzero(aspect == a)
        values = function(image[rows], float(a))
        values = values if isinstance(values, tuple) else (values,)
        if results is None:
            results = [np.zeros((len(image),) + v.shape[1:], v.dtype) for v in values]
        for r, v in zip(results, values):
            r[rows] = v
    return tuple(results) if results is not None else ()


def youtube_pose_arrays(yt: dict):
    """(features (N, 50), directions (N, 5, 3) float32, valid (N, 5), video (N,)) of frames with a valid feature row
    and at least one valid target."""
    x, ok = by_aspect(wr.pose_features_xy2, yt["pose"], yt["aspect"])
    keep = ok & yt["arm_valid"].any(1)
    return x[keep], yt["arm_dirs"][keep].astype(np.float32), yt["arm_valid"][keep], yt["video"][keep]


def load_crops(crop_dir: Path, legacy_dir: Path, manifest: list | None = None) -> list[dict]:
    """Pose chunks of the training signers' tight crops: legacy Holistic image points (the inputs), Tasks world and
    visibility (the targets), on frames where both found a pose; a crop whose legacy array does not match its frames
    or size is skipped. Signer 9 and every non-training signer are refused."""
    from uwb_lifter_v2_crops import signer_of
    from validate_semlex_adapter import load_npz
    chunks = []
    for path in sorted(Path(crop_dir).glob("*.npz")):
        signer = signer_of(path.name)
        if f"h2s-{signer}" in NEVER_READ:
            raise SystemExit(f"{path.name}: How2Sign signer 9 is never read")
        if f"h2s-{signer}" not in TRAIN_SIGNERS:
            raise SystemExit(f"{path.name}: signer {signer} is not a training signer")
        legacy_path = Path(legacy_dir) / path.name
        if not legacy_path.exists():
            continue
        meta, clip, _ = load_npz(path)
        with np.load(legacy_path, allow_pickle=False) as z:
            holistic, size = z["holistic"].astype(np.float64), z["size"].tolist()
        if len(holistic) != len(clip.times) or list(size) != list(meta["size"]):
            continue
        pose = holistic[:, 478:511]                      # (x, y, z) as load_pairs keeps it; z-free features ignore z
        ok = np.asarray(clip.pose_found, bool) & np.isfinite(pose).all((1, 2))
        if manifest is not None:
            manifest.append(dict(crop=path.name, crop_sha256=sha256_file(path), legacy_sha256=sha256_file(legacy_path),
                                 signer=f"h2s-{signer}", frames=int(ok.sum())))
        if ok.any():
            chunks.append(dict(signer=f"h2s-{signer}", aspect=size[0] / size[1], image=pose[ok],
                               world=np.asarray(clip.pose_world, np.float64)[ok],
                               vis=np.asarray(clip.pose_visibility, np.float64)[ok]))
    return chunks


def how2sign_chunks(pairs: Path, legacy: Path, manifest: list):
    """(hand chunks, pose chunks) of the v2 pairs restricted to the training signers (validation signers skipped;
    signer 9 refused by ``load_pairs``)."""
    from train_world_regressors import load_pairs
    hands, pose, _ = load_pairs(pairs, legacy, manifest)
    for chunk in pose + hands["Left"] + hands["Right"]:
        if chunk["signer"] in NEVER_READ:
            raise SystemExit("How2Sign signer 9 is never read")
    keep = lambda chunks: [c for c in chunks if c["signer"] in TRAIN_SIGNERS]   # noqa: E731
    return {s: keep(v) for s, v in hands.items()}, keep(pose)


# ----------------------------------------------------------------------------------------------------------------------
# Training

def streams(sizes, steps: int, half: int, generator) -> list:
    """Per source, ``steps * half`` indices from consecutive random permutations (the smaller source is cycled)."""
    out = []
    for n in sizes:
        need = steps * half
        parts = [torch.randperm(n, generator=generator) for _ in range(math.ceil(need / n))]
        out.append(torch.cat(parts)[:need])
    return out


def cosine_loss(vectors: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(vectors, target, dim=-1, eps=1e-6)
    return ((1 - cos) * valid).sum() / valid.sum().clamp(min=1)


def bone_vectors(points: torch.Tensor, bones) -> torch.Tensor:
    return torch.stack([points[:, j] - points[:, i] for i, j in bones], 1)


def train_mixed(model, mean, std, h2s: tuple, yt: tuple, *, h2s_loss, yt_loss, epochs: int, lr: float,
                batch: int = 2048, seed: int = 0, log: str = ""):
    """50/50 batches of the How2Sign arrays ``h2s`` (x, ...) and the YouTube arrays ``yt`` (x, ...); the inputs are
    standardised by (mean, std). Returns the trained model (eval mode) and the per-epoch mean losses."""
    if not len(h2s[0]) or not len(yt[0]):
        raise SystemExit(f"{log}: {len(h2s[0])} How2Sign and {len(yt[0])} YouTube frames; both sources are needed")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    half = batch // 2
    xh = torch.from_numpy(((h2s[0] - mean) / std).astype(np.float32))
    xy = torch.from_numpy(((yt[0] - mean) / std).astype(np.float32))
    rest_h = [torch.from_numpy(np.ascontiguousarray(a)) for a in h2s[1:]]
    rest_y = [torch.from_numpy(np.ascontiguousarray(a)) for a in yt[1:]]
    steps = math.ceil(max(len(xh), len(xy)) / half)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    history = []
    model.train()
    for epoch in range(epochs):
        ih, iy = streams((len(xh), len(xy)), steps, half, generator)
        totals = np.zeros(2)
        for s in range(steps):
            a, b = ih[s * half:(s + 1) * half], iy[s * half:(s + 1) * half]
            lh = h2s_loss(model(xh[a]), *[r[a] for r in rest_h])
            ly = yt_loss(model(xy[b]), *[r[b] for r in rest_y])
            loss = lh + ly
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            totals += [float(lh.detach()), float(ly.detach())]
        history.append(dict(epoch=epoch + 1, how2sign=round(totals[0] / steps, 5), youtube=round(totals[1] / steps, 5)))
        print(f"{log} epoch {epoch + 1}/{epochs} how2sign {history[-1]['how2sign']} youtube {history[-1]['youtube']}",
              flush=True)
    return model.eval(), history


def pose_losses(points: int = wr.POSE_POINTS):
    width = points * 3

    def h2s(out, y, vis):
        pred, true = out[:, :width].reshape(len(out), points, 3), y.reshape(len(y), points, 3)
        return (torch.nn.functional.mse_loss(pred, true) + wr.direction_loss(pred, true, wr.POSE_BONES)
                + torch.nn.functional.binary_cross_entropy_with_logits(out[:, width:], vis))

    def yt(out, target, valid):
        pred = out[:, :width].reshape(len(out), points, 3)
        return cosine_loss(bone_vectors(pred, YT_BONES), target, valid.float())
    return h2s, yt


def hand_losses(side: str):
    from signtranslator.data_engineering.video_landmarks import finger_joints
    bones = [(j.start, j.end) for j in finger_joints(side)]

    def h2s(out, y):
        pred, true = out.reshape(len(out), 21, 3), y.reshape(len(y), 21, 3)
        return torch.nn.functional.mse_loss(pred, true) + wr.direction_loss(pred, true, wr.HAND_BONES)

    def yt(out, target, valid):
        return cosine_loss(bone_vectors(out.reshape(len(out), 21, 3), bones), target, valid.float())
    return h2s, yt


def youtube_hand_arrays(yt: dict, side: str):
    """(features (N, 40), targets (N, 15, 3) in the model's in-plane frame, valid (N, 15)) of one side."""
    k = ("Left", "Right").index(side)
    rows = np.flatnonzero(yt["hands_found"][:, k] & yt["finger_valid"][:, k].any(1))
    image, aspect = yt["hands"][rows, k].astype(np.float64), yt["aspect"][rows]
    x, rotation, _, ok = by_aspect(lambda im, a: wr.hand_features_xy(im, a), image, aspect)
    target = yt["finger_dirs"][rows, k].astype(np.float64)
    target[..., :2] = np.einsum("nij,nkj->nki", rotation, target[..., :2])
    return x[ok], target[ok].astype(np.float32), yt["finger_valid"][rows, k][ok]


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation on lifter-val

def direction_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    cos = (pred * target).sum(-1) / np.maximum(np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1., 1.)))


def stat(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 2), p90=round(float(np.percentile(v, 90)), 2),
                mean=round(float(v.mean()), 2)) if len(v) else dict(n=0)


@torch.no_grad()
def pose_val_errors(model, mean, std, x, target, valid) -> dict:
    """Per bone group: lifter-val angle statistics (degrees) between predicted and our directions."""
    out = model(torch.from_numpy(((x - mean) / std).astype(np.float32))).numpy()
    pred = out[:, :wr.POSE_POINTS * 3].reshape(len(x), wr.POSE_POINTS, 3)
    vectors = np.stack([pred[:, j] - pred[:, i] for i, j in YT_BONES], 1)
    errors = direction_errors(vectors, target)
    report = {}
    for group, columns in BONE_GROUPS.items():
        report[group] = stat(np.concatenate([errors[:, c][valid[:, c]] for c in columns]))
    for k, name in enumerate(("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm", "ShoulderLine")):
        report[name] = stat(errors[:, k][valid[:, k]])
    return report


@torch.no_grad()
def hand_val_errors(model, mean, std, x, target, valid, side: str) -> dict:
    from signtranslator.data_engineering.video_landmarks import finger_joints
    bones = [(j.start, j.end) for j in finger_joints(side)]
    out = model(torch.from_numpy(((x - mean) / std).astype(np.float32))).numpy().reshape(len(x), 21, 3)
    vectors = np.stack([out[:, j] - out[:, i] for i, j in bones], 1)
    return stat(direction_errors(vectors, target)[valid])


# ----------------------------------------------------------------------------------------------------------------------
# Validity head

def validity_arrays(yt: dict):
    """(features (N, 68), labels (N, 6) float32, mask (N, 6)) of YouTube frames with at least one label."""
    x, ok = by_aspect(wr.validity_features, yt["pose"], yt["aspect"])
    keep = ok & yt["label_mask"].any(1)
    return x[keep], yt["labels"][keep].astype(np.float32), yt["label_mask"][keep]


def crop_validity_arrays(chunks: list[dict]):
    """(features, labels: Tasks visibility >= 0.5 on points 11..16, mask: all) of the crops."""
    xs, ys = [], []
    for c in chunks:
        x, ok = wr.validity_features(c["image"], c["aspect"])
        xs.append(x[ok])
        ys.append((c["vis"][ok][:, list(wr.VALIDITY_POINTS)] >= .5).astype(np.float32))
    x = np.concatenate(xs) if xs else np.zeros((0, wr.VALIDITY_INPUTS), np.float32)
    y = np.concatenate(ys) if ys else np.zeros((0, 6), np.float32)
    return x, y, np.ones(y.shape, bool)


def masked_bce(out, labels, mask):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, labels, reduction="none")
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def choose_thresholds(probability: np.ndarray, forearm_ours: np.ndarray, chest: np.ndarray,
                      uwb_only_max: float = UWB_ONLY_MAX, grid=THRESHOLDS) -> dict:
    """Per side: the threshold maximising agreement between the head's forearm (shoulder, elbow and wrist all >= t)
    and ours, on frames where our chest is valid, subject to UWB-only <= ``uwb_only_max`` (ties: the higher t).
    ``probability``: (N, 6) for points 11..16; ``forearm_ours``: (N, 2) Left, Right."""
    out = {}
    for k, side in enumerate(("Left", "Right")):
        p = probability[chest][:, list(SIDE_POINTS[side])].min(1)
        ours = forearm_ours[chest][:, k].astype(bool)
        n = max(len(ours), 1)
        curve, best = [], None
        for t in grid:
            kept = p >= t
            agreement = float((kept == ours).sum() / n)
            uwb_only = float((kept & ~ours).sum() / n)
            curve.append(dict(threshold=float(t), agreement=round(agreement, 4), uwb_only=round(uwb_only, 4)))
            if uwb_only <= uwb_only_max and (best is None or agreement >= best[1]):
                best = (float(t), agreement, uwb_only)
        out[side] = dict(threshold=best[0], agreement=round(best[1], 4), uwb_only=round(best[2], 4), frames=len(ours),
                         ours_valid_share=round(float(ours.mean()), 4) if len(ours) else None, curve=curve)
    return out


# ----------------------------------------------------------------------------------------------------------------------

def load_bundle(path: str, sha: str) -> wr.WorldRegressors:
    return wr.WorldRegressors.load(Path(path).read_bytes(), expected_sha256=sha)


def load_split_file(path: str) -> tuple[dict, str]:
    import uwb_lifter_gate as gate
    return gate.load_split(None, path)


def write_once(path: str, payload: bytes) -> None:
    with open(path, "xb") as stream:
        stream.write(payload)


def peak_rss_mb() -> float:
    """This process's peak resident set size in MB (ru_maxrss: kilobytes on Linux, bytes on macOS)."""
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (2 ** 20 if sys.platform == "darwin" else 2 ** 10), 1)


def run_pose(args) -> int:
    init = load_bundle(args.init_bundle, args.init_sha256)
    if init.features != wr.FEATURES_XY2 or init.validity is not None:
        raise SystemExit("the init bundle must be an xy2 bundle without a validity head")
    split, split_sha = load_split_file(args.split_file)
    manifest, crop_manifest = [], []
    hands_h2s, pose_h2s = how2sign_chunks(Path(args.h2s_pairs), Path(args.h2s_legacy), manifest)
    crops = load_crops(Path(args.crops), Path(args.crops_legacy), crop_manifest) if args.crops else []
    from train_world_regressors import hand_arrays, pose_arrays
    h2s = pose_arrays(pose_h2s + crops, wr.FEATURES_XY2)
    yt_train = load_youtube(Path(args.yt_pairs), split, split_sha, "train", hands=args.finetune_hands)
    yt_val = load_youtube(Path(args.yt_pairs), split, split_sha, "lifter_val", hands=args.finetune_hands)
    xt, dt, vt, _ = youtube_pose_arrays(yt_train)
    xv, dv, vv, video_v = youtube_pose_arrays(yt_val)
    print(json.dumps(dict(how2sign_frames=len(h2s[0]), crop_chunks=len(crops), youtube_train_frames=len(xt),
                          lifter_val_frames=len(xv))), flush=True)
    if args.from_scratch:
        torch.manual_seed(args.seed)
        both = np.concatenate((h2s[0], xt))
        mean, std = both.mean(0).astype(np.float32), (both.std(0) + 1e-6).astype(np.float32)
        model = wr.MLP(wr.POSE_POINTS * 2, wr.POSE_POINTS * 3 + 33)
    else:
        model, mean, std = copy.deepcopy(init.pose[0]), init.pose[1], init.pose[2]
    h2s_loss, yt_loss = pose_losses()
    model, history = train_mixed(model, mean, std, (h2s[0], h2s[1], h2s[2].astype(np.float32)), (xt, dt, vt),
                                 h2s_loss=h2s_loss, yt_loss=yt_loss, epochs=args.epochs, lr=args.lr, batch=args.batch,
                                 seed=args.seed, log="pose")
    hands = dict(init.hand)
    hand_report = {}
    if args.finetune_hands:
        for side in ("Left", "Right"):
            hx = hand_arrays(hands_h2s[side], wr.FEATURES_XY2)
            yx, yd, yv = youtube_hand_arrays(yt_train, side)
            vx, vd, vvalid = youtube_hand_arrays(yt_val, side)
            base, mu, sd = init.hand[side]
            h_loss, y_loss = hand_losses(side)
            tuned, h_hist = train_mixed(copy.deepcopy(base), mu, sd, (hx[0], hx[1]), (yx, yd, yv), h2s_loss=h_loss,
                                        yt_loss=y_loss, epochs=args.epochs, lr=args.lr, batch=args.batch,
                                        seed=args.seed, log=f"hand {side}")
            hands[side] = (tuned, mu, sd)
            hand_report[side] = dict(history=h_hist, how2sign_frames=len(hx[0]), youtube_frames=len(yx),
                                     lifter_val=dict(fingers=hand_val_errors(tuned, mu, sd, vx, vd, vvalid, side),
                                                     init_fingers=hand_val_errors(base, mu, sd, vx, vd, vvalid, side)))
    bundle = wr.WorldRegressors(hand=hands, pose=(model, mean, std), features=wr.FEATURES_XY2)
    payload = bundle.save()
    write_once(args.out, payload)
    report = dict(stage="pose", declaration="docs/NEXT_PHASE_PLAN.md: UWB lifter v2 and gate v2 (2026-10-30 11:12 UTC)",
                  arm="from-scratch" if args.from_scratch else "fine-tuned from xy2",
                  init=dict(path=args.init_bundle, sha256=args.init_sha256),
                  settings=dict(epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed, mix="50/50",
                                optimizer="AdamW(weight_decay=1e-4), OneCycleLR", hands="fine-tuned" if
                                args.finetune_hands else "frozen (xy2)", youtube_bones=[list(b) for b in YT_BONES]),
                  split_sha256=split_sha, train_signers=list(TRAIN_SIGNERS), never_read=list(NEVER_READ),
                  how2sign_pairs_manifest=manifest, crops_manifest=crop_manifest,
                  youtube=dict(train_frames=int(len(xt)), lifter_val_frames=int(len(xv)),
                               lifter_val_videos=int(len(set(video_v.tolist()))),
                               shards=yt_train["shards"]),
                  how2sign_frames=int(len(h2s[0])), history=history,
                  lifter_val=dict(pose=pose_val_errors(model, mean, std, xv, dv, vv),
                                  init_pose=pose_val_errors(init.pose[0], init.pose[1], init.pose[2], xv, dv, vv),
                                  hands=hand_report or None),
                  bundle_sha256=hashlib.sha256(payload).hexdigest(), peak_rss_mb=peak_rss_mb())
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report["lifter_val"]["pose"]), flush=True)
    print(f"peak RSS {report['peak_rss_mb']} MB ({report['arm']}, hands {report['settings']['hands']})", flush=True)
    print("bundle", report["bundle_sha256"])
    return 0


def selection_score(report: dict) -> float:
    """Amendment 1: the lifter-val mean of the upper-arm and forearm medians (degrees)."""
    pose = report["lifter_val"]["pose"]
    return (pose["upper_arms"]["median"] + pose["forearms"]["median"]) / 2


def select(reports: list[dict]) -> dict:
    """The arm with the lowest lifter-val mean of the upper-arm and forearm medians (ties: the fine-tuned arm, then
    the first given)."""
    rows = []
    for k, r in enumerate(reports):
        if r.get("stage") != "pose" or r["settings"]["hands"] != "frozen (xy2)":
            raise SystemExit(f"report {k}: only pose arms with frozen hands are selected between")
        rows.append((selection_score(r), r["arm"] != "fine-tuned from xy2", k))
    if len({r["split_sha256"] for r in reports}) != 1:
        raise SystemExit("the arms were scored on different splits")
    chosen = min(rows)[2]
    return dict(chosen=chosen, bundle_sha256=reports[chosen]["bundle_sha256"], arm=reports[chosen]["arm"],
                rule="lowest lifter-val mean of the upper-arm and forearm medians (ties: the fine-tuned arm); "
                     "amendment 1, 2026-10-30 11:51 UTC",
                candidates=[dict(arm=r["arm"], bundle_sha256=r["bundle_sha256"], score=round(selection_score(r), 4),
                                 lifter_val=r["lifter_val"]["pose"]) for r in reports])


def run_select(args) -> int:
    reports = [json.loads(Path(p).read_text()) for p in args.reports]
    out = select(reports)
    out["reports"] = [dict(path=p, sha256=sha256_file(p)) for p in args.reports]
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: out[k] for k in ("chosen", "arm", "bundle_sha256")}))
    return 0


def run_validity(args) -> int:
    bundle = load_bundle(args.bundle, args.bundle_sha256)
    if bundle.validity is not None:
        raise SystemExit("the bundle already has a validity head")
    split, split_sha = load_split_file(args.split_file)
    yt_train = load_youtube(Path(args.yt_pairs), split, split_sha, "train")
    yt_val = load_youtube(Path(args.yt_pairs), split, split_sha, "lifter_val")
    crop_manifest = []
    crops = load_crops(Path(args.crops), Path(args.crops_legacy), crop_manifest)
    xy, ly, my = validity_arrays(yt_train)
    xc, lc, mc = crop_validity_arrays(crops)
    if not len(xy) or not len(xc):
        raise SystemExit(f"validity head: {len(xy)} YouTube and {len(xc)} crop frames; both sources are needed")
    both = np.concatenate((xy, xc))
    mean, std = both.mean(0).astype(np.float32), (both.std(0) + 1e-6).astype(np.float32)
    torch.manual_seed(args.seed)
    model = wr.MLP(wr.VALIDITY_INPUTS, len(wr.VALIDITY_POINTS), **VALIDITY_SHAPE)
    loss = lambda out, labels, mask: masked_bce(out, labels, mask)          # noqa: E731
    model, history = train_mixed(model, mean, std, (xc, lc, mc), (xy, ly, my), h2s_loss=loss, yt_loss=loss,
                                 epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed, log="validity")
    probe = wr.WorldRegressors(hand=bundle.hand, pose=bundle.pose, features=bundle.features,
                               validity=(model, mean, std, {"Left": 1.01, "Right": 1.01}))
    probability, _ = by_aspect(lambda im, a: probe.point_validity(im, a), yt_val["pose"], yt_val["aspect"])
    chest = yt_val["chest_valid"].astype(bool)          # every frame the gate's forearm row counts
    chosen = choose_thresholds(probability, yt_val["forearm_valid"], chest, args.uwb_only_max)
    thresholds = {side: chosen[side]["threshold"] for side in ("Left", "Right")}
    final = wr.WorldRegressors(hand=bundle.hand, pose=bundle.pose, features=bundle.features,
                               validity=(model, mean, std, thresholds))
    payload = final.save()
    write_once(args.out, payload)
    report = dict(stage="validity", declaration="docs/NEXT_PHASE_PLAN.md: UWB lifter v2 and gate v2 "
                                                "(2026-10-30 11:12 UTC)",
                  base_bundle_sha256=args.bundle_sha256, split_sha256=split_sha, thresholds=thresholds,
                  uwb_only_max=args.uwb_only_max, lifter_val=chosen,
                  settings=dict(epochs=args.epochs, lr=args.lr, batch=args.batch, seed=args.seed, mix="50/50",
                                shape=VALIDITY_SHAPE, features=wr.VALIDITY_FEATURES),
                  frames=dict(youtube_train=int(len(xy)), crops=int(len(xc)),
                              youtube_label_rates={str(p): round(float(ly[my[:, k], k].mean()), 4) if my[:, k].any()
                                                   else None for k, p in enumerate(wr.VALIDITY_POINTS)},
                              crop_label_rates={str(p): round(float(lc[:, k].mean()), 4) if len(lc) else None
                                                for k, p in enumerate(wr.VALIDITY_POINTS)}),
                  crops_manifest=crop_manifest, history=history, bundle_sha256=hashlib.sha256(payload).hexdigest())
    Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({s: {k: v for k, v in chosen[s].items() if k != "curve"} for s in chosen}), flush=True)
    print("bundle", report["bundle_sha256"])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    pose = sub.add_parser("pose")
    pose.add_argument("--split-file", required=True)
    pose.add_argument("--yt-pairs", required=True, help="local directory of the lifter-v2 pairs (pairs-index.json)")
    pose.add_argument("--h2s-pairs", required=True, help="the v2 world pairs (collect_world_pairs.py)")
    pose.add_argument("--h2s-legacy", required=True, help="their legacy Holistic arrays")
    pose.add_argument("--crops", help="the training signers' tight crops (uwb_lifter_v2_crops.py)")
    pose.add_argument("--crops-legacy", help="their legacy Holistic arrays")
    pose.add_argument("--init-bundle", required=True)
    pose.add_argument("--init-sha256", default=XY2_BUNDLE["sha256"])
    pose.add_argument("--from-scratch", action="store_true", help="the comparison arm")
    pose.add_argument("--finetune-hands", action="store_true", help="the hands comparison")
    pose.add_argument("--epochs", type=int)
    pose.add_argument("--lr", type=float)
    pose.add_argument("--batch", type=int, default=2048)
    pose.add_argument("--seed", type=int, default=0)
    pose.add_argument("--out", required=True)
    pose.add_argument("--report", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--reports", nargs="+", required=True)
    sel.add_argument("--out", required=True)
    val = sub.add_parser("validity")
    val.add_argument("--split-file", required=True)
    val.add_argument("--yt-pairs", required=True)
    val.add_argument("--crops", required=True)
    val.add_argument("--crops-legacy", required=True)
    val.add_argument("--bundle", required=True)
    val.add_argument("--bundle-sha256", required=True)
    val.add_argument("--epochs", type=int, default=15)
    val.add_argument("--lr", type=float, default=1e-3)
    val.add_argument("--batch", type=int, default=2048)
    val.add_argument("--seed", type=int, default=0)
    val.add_argument("--uwb-only-max", type=float, default=UWB_ONLY_MAX)
    val.add_argument("--out", required=True)
    val.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    if args.stage == "pose":
        if bool(args.crops) != bool(args.crops_legacy):
            raise SystemExit("--crops and --crops-legacy go together")
        args.epochs = args.epochs or (30 if args.from_scratch else 15)
        args.lr = args.lr or (1e-3 if args.from_scratch else 3e-4)
        return run_pose(args)
    if args.stage == "select":
        return run_select(args)
    return run_validity(args)


if __name__ == "__main__":
    raise SystemExit(main())
