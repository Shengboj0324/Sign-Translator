"""Where a motion tokenizer's reconstruction error comes from, per joint and per hand point.

For the validation clips of a manifest, three reconstructions of each clip's unit-space
motion are compared with the reference:

* **full** — the tokenizer as used (encode, quantise, decode);
* **no quantiser** — encode and decode with the latents passed unquantised: the
  decoder's floor; full − floor is the *quantisation gap*;
* **target noise** — not a reconstruction: the reference's own frame-to-frame jitter,
  the geodesic angle between each rotation and the midpoint (SLERP at ½) of its
  neighbours, and for points the distance to the neighbours' mean. A tokenizer that
  smooths reconstructs below this, and errors of that size are not worth chasing.

Reported: per body/hand joint the median geodesic error in degrees for each, and per
arm/hand point (nominal skeleton, chest frame) the median position error in cm — plus
the overall medians. Works for tokens v1 (``motion_tokenizer``) and v2
(``motion_rvq_tokenizer``) checkpoints.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig  # noqa: E402
from signtranslator.generation.motion_tokens import MotionVQ, MotionVQConfig  # noqa: E402
from signtranslator.generation.signing_space import skeleton_points  # noqa: E402
from signtranslator.generation.text_to_motion import diffusion_targets, unit_to_raw  # noqa: E402
from signtranslator.pose.rotations import rotation_6d_to_matrix  # noqa: E402
from evaluate_text_to_motion import load_state  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402

ROTATIONS = ("body", "left_hand", "right_hand", "head_rotation")


def load_tokenizer(store, key: str):
    saved = torch.load(io.BytesIO(store.get(key)), map_location="cpu", weights_only=False)
    weights = saved.get("ema") or saved["model"]
    if saved["kind"] == "motion_rvq_tokenizer":
        model = MotionRVQ(MotionRVQConfig.from_dict(saved["config"]))
    elif saved["kind"] == "motion_tokenizer":
        model = MotionVQ(MotionVQConfig.from_dict(saved["config"]))
    else:
        raise SystemExit(f"{key} is not a tokenizer checkpoint")
    model.load_state_dict(weights)
    return model.eval(), saved


@torch.no_grad()
def reconstructions(model, unit: torch.Tensor) -> dict:
    frames = unit.shape[1]
    latents = model.encode(unit)
    if isinstance(model, MotionRVQ):
        quantized, _, _ = model.quantize(latents)
        return dict(full=model.decode_latents(quantized, frames), floor=model.decode_latents(latents, frames))
    quantized, _, _ = model.quantize(latents)
    return dict(full=model.decode_codes(quantized, frames), floor=model.decode_codes(latents, frames))


def codes_of(model, unit: torch.Tensor, frame_valid: torch.Tensor):
    """((B, K, J) code indices, (B, K) token validity) of a tokenizer on ``unit``."""
    with torch.no_grad():
        latents = model.encode(unit)
        _, codes, _ = model.quantize(latents)
    codes = codes if codes.dim() == 3 else codes[..., None]
    if hasattr(model, "token_valid"):
        valid = model.token_valid(frame_valid)
    else:
        valid = torch.ones(codes.shape[:2], dtype=torch.bool)
    return codes, valid


def code_statistics(sequences, codebook: int, steps_per_second: float) -> dict:
    """Learnability of a token stream: how much a prior must predict per second.

    ``sequences`` is a list of ((K, J) codes, (K,) valid) per clip. Per code position j:
    the repeat rate P(c_t = c_{t-1}) over valid adjacent pairs, the unigram entropy
    H(c) and the plug-in first-order conditional entropy H(c_t | c_{t-1}) (biased low on
    few samples; a lower estimate of what a bigram model leaves), all in nats. The
    unigram rate steps/s × Σ_j H_j bounds what the prior has to model without context;
    the conditional rate is the same with H(c_t | c_{t-1}). Reconstruction error alone
    does not rank tokenizers for generation: a codec whose codes are near uniform and
    change every step reconstructs well and is hard to predict from text.
    """
    J = sequences[0][0].shape[-1] if sequences else 0
    repeats, pairs = np.zeros(J), 0
    unigram = np.zeros((J, codebook))
    bigram = np.zeros((J, codebook, codebook))
    for codes, valid in sequences:
        c, v = np.asarray(codes), np.asarray(valid, bool)
        for j in range(J):
            np.add.at(unigram[j], c[v, j], 1)
        both = v[1:] & v[:-1]
        pairs += int(both.sum())
        repeats += (c[1:][both] == c[:-1][both]).sum(0)
        for j in range(J):
            np.add.at(bigram[j], (c[:-1][both, j], c[1:][both, j]), 1)

    def entropy(counts):
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum())

    h = [entropy(unigram[j]) if unigram[j].sum() else 0. for j in range(J)]
    hc = []
    for j in range(J):
        rows = bigram[j].sum(1)
        total = rows.sum()
        hc.append(float(sum(rows[a] / total * entropy(bigram[j][a]) for a in np.nonzero(rows)[0])) if total else 0.)
    return dict(codes_per_step=J, steps_per_second=steps_per_second, codebook=codebook,
                repeat_rate=[float(r / max(pairs, 1)) for r in repeats], unigram_entropy=h, conditional_entropy=hc,
                unigram_rate_nats_per_s=steps_per_second * sum(h), conditional_rate_nats_per_s=steps_per_second * sum(hc),
                max_entropy=float(np.log(codebook)))


def geodesic_deg(a6: torch.Tensor, b6: torch.Tensor) -> torch.Tensor:
    ra, rb = rotation_6d_to_matrix(a6), rotation_6d_to_matrix(b6)
    cos = ((ra.transpose(-1, -2) @ rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))


def jitter_deg(r6: torch.Tensor) -> torch.Tensor:
    """(T-2, J) angle between R_t and the geodesic midpoint of R_{t-1} and R_{t+1}."""
    r = rotation_6d_to_matrix(r6)
    prev, nxt, mid = r[:-2], r[2:], r[1:-1]
    rel = prev.transpose(-1, -2) @ nxt                          # R_{t-1}ᵀ R_{t+1}
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    angle = torch.arccos(cos)
    axis = torch.stack((rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0],
                        rel[..., 1, 0] - rel[..., 0, 1]), -1) / (2 * torch.sin(angle).clamp_min(1e-9))[..., None]
    half = angle / 2
    K = torch.zeros(*axis.shape[:-1], 3, 3, dtype=r.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axis[..., 2], axis[..., 1], -axis[..., 0]
    K = K - K.transpose(-1, -2)
    eye = torch.eye(3, dtype=r.dtype).expand_as(K)
    small = (angle < 1e-6)[..., None, None]
    halfway = torch.where(small, eye, eye + torch.sin(half)[..., None, None] * K
                          + (1 - torch.cos(half))[..., None, None] * K @ K)
    midpoint = prev @ halfway
    cos_m = (((midpoint.transpose(-1, -2) @ mid).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos_m))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--tokenizer-key", required=True)
    parser.add_argument("--partition", default="val")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    model, saved = load_tokenizer(store, args.tokenizer_key)
    motion_config = model.config.motion
    records = sorted((r for r in manifest["records"] if r["partition"] == args.partition and r.get("continuous")),
                     key=lambda r: r["sign"])[:args.limit]
    per_joint = {kind: {} for kind in ("full", "floor", "noise")}
    per_point = {kind: {} for kind in ("full", "floor", "noise")}
    sequences, seconds = [], 0.
    for r in records:
        state = load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"])
        batch = collate_multichannel([state])
        unit = diffusion_targets(motion_config, batch)
        recon = reconstructions(model, unit)
        codes, token_ok = codes_of(model, unit, batch.frame_valid)
        sequences.append((codes[0].numpy(), token_ok[0].numpy()))
        seconds += unit.shape[1] * float(np.median(np.diff(np.asarray(state.timestamps)))) if len(state.timestamps) > 1 else 0.
        raws = {k: unit_to_raw(motion_config, v.double()) for k, v in recon.items()}
        valid = {n: batch.channels[n].valid[0] for n in ROTATIONS if n in motion_config.trained_channels}
        identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float64)
        # unavailable rotations are zero payload: identity keeps the maths defined, the masks drop them
        reference = {n: torch.where(valid[n][..., None], batch.channels[n].values[0].double(), identity) for n in valid}
        for name, ref in reference.items():
            labels = batch.channels[name].layout.labels
            for kind, raw in raws.items():
                err = geodesic_deg(raw[name][0], ref)
                for k, label in enumerate(labels):
                    v = err[valid[name][:, k], k]
                    per_joint[kind].setdefault(f"{name}/{label}", []).append(v.numpy())
            if len(ref) >= 3:
                jit = jitter_deg(ref)
                for k, label in enumerate(labels):
                    ok = valid[name][2:, k] & valid[name][1:-1, k] & valid[name][:-2, k]
                    per_joint["noise"].setdefault(f"{name}/{label}", []).append(jit[ok, k].numpy())
        if all(n in reference for n in ("body", "left_hand", "right_hand")):
            labels = [batch.channels[n].layout.labels for n in ("body", "left_hand", "right_hand")]
            target, names, chains = skeleton_points(*(reference[n] for n in ("body", "left_hand", "right_hand")), *labels)
            channels = {n: (valid[n][None], tuple(batch.channels[n].layout.labels)) for n in ("body", "left_hand", "right_hand")}
            from signtranslator.generation.signing_space import _support
            point_ok = _support(channels, chains, torch.ones(1, len(target), dtype=torch.bool))[0]      # (T, P)
            for kind, raw in raws.items():
                pts, _, _ = skeleton_points(*(raw[n][0] for n in ("body", "left_hand", "right_hand")), *labels)
                dist = torch.linalg.vector_norm(pts - target, dim=-1) * 100
                for k, name in enumerate(names):
                    per_point[kind].setdefault(name, []).append(dist[point_ok[:, k], k].numpy())
            if len(target) >= 3:
                noise = torch.linalg.vector_norm(target[1:-1] - (target[:-2] + target[2:]) / 2, dim=-1) * 100
                ok3 = point_ok[1:-1] & point_ok[:-2] & point_ok[2:]
                for k, name in enumerate(names):
                    per_point["noise"].setdefault(name, []).append(noise[ok3[:, k], k].numpy())
    med = lambda parts: float(np.median(np.concatenate(parts))) if parts and sum(map(len, parts)) else None
    joints = {key: {kind: med(per_joint[kind].get(key, [])) for kind in per_joint} for key in per_joint["full"]}
    points = {key: {kind: med(per_point[kind].get(key, [])) for kind in per_point} for key in per_point["full"]}
    overall = {f"{kind}_joint_deg": med([v for parts in per_joint[kind].values() for v in parts]) for kind in per_joint}
    overall.update({f"{kind}_point_cm": med([v for parts in per_point[kind].values() for v in parts]) for kind in per_point})
    hands = {f"{kind}_fingertip_cm": med([v for name, parts in per_point[kind].items() if name.endswith("Tip") for v in parts])
             for kind in per_point}
    steps = sum(len(c) for c, _ in sequences)
    learnability = code_statistics(sequences, model.config.codebook, steps / max(seconds, 1e-9))
    report = dict(tokenizer_key=args.tokenizer_key, kind=saved["kind"], clips=len(records), overall=overall, hands=hands,
                  learnability=learnability,
                  worst_joints=sorted(joints.items(), key=lambda kv: -(kv[1]["full"] or 0))[:12], joints=joints,
                  points=points)
    print(json.dumps(dict(overall=overall, hands=hands, worst_joints=report["worst_joints"][:6],
                          learnability={k: learnability[k] for k in ("codes_per_step", "steps_per_second",
                                                                     "unigram_rate_nats_per_s", "conditional_rate_nats_per_s")}),
                     indent=1, default=float))
    if args.output_key:
        store.put(args.output_key, json.dumps(report, indent=1, sort_keys=True, default=float).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
