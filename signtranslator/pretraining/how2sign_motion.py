"""Quarantined 2D masked-reconstruction experiment for audited How2Sign poses.

This module is intentionally disconnected from the translator, gloss exporter,
3D pose stack, and Stage C.  Its only scientific question is whether a compact
graph-temporal network can reconstruct deliberately hidden, originally observed
2D landmarks better than declared non-neural baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data_engineering.how2sign import (
    OPENPOSE_LANDMARK_PARTS,
    decode_how2sign_openpose,
    openpose_holistic_graph,
)
from ..data_engineering.how2sign_audit import hierarchical_digest, stable_sha256
from ..models.stgcn import STGCNBlock
from .masking import random_point_mask, span_mask, typical_masked_floor


EXPERIMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Pose2DExperimentConfig:
    schema_version: int = EXPERIMENT_SCHEMA_VERSION
    seed: int = 20260803
    sample_count: int = 96
    window_frames: int = 64
    epochs: int = 12
    batch_size: int = 4
    learning_rate: float = 1e-3
    hidden_channels: int = 24
    mask_ratio: float = 0.25
    span_length: int = 8
    tiny_overfit_steps: int = 160

    def validate(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema")
        if self.sample_count < 8 or self.window_frames < 8:
            raise ValueError("sample_count and window_frames are too small")
        if self.epochs < 1 or self.batch_size < 1 or self.hidden_channels < 4:
            raise ValueError("training dimensions must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be strictly between zero and one")
        if self.span_length < 2 or self.span_length >= self.window_frames:
            raise ValueError("span_length must be in [2, window_frames)")
        if self.tiny_overfit_steps < 1:
            raise ValueError("tiny_overfit_steps must be positive")


def source_disjoint_partition(video_ids: Sequence[str], seed: int,
                              ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
                              ) -> dict[str, str]:
    """Deterministically partition complete VIDEO_ID groups, never signer IDs."""
    if any(not value for value in video_ids):
        raise ValueError("video IDs must be non-empty")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video IDs must be unique group identifiers")
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-12:
        raise ValueError("ratios must be non-negative and sum to one")
    names = ("train", "val", "test")
    ordered = sorted(video_ids, key=lambda value: hashlib.sha256(
        f"{seed}\x1f{value}".encode("utf-8")).digest())
    count = len(ordered)
    raw_targets = [ratio * count for ratio in ratios]
    sizes = [int(math.floor(value)) for value in raw_targets]
    for index in sorted(range(3), key=lambda idx: raw_targets[idx] - sizes[idx],
                        reverse=True)[:count - sum(sizes)]:
        sizes[index] += 1
    if count >= sum(ratio > 0 for ratio in ratios):
        for index, ratio in enumerate(ratios):
            if ratio > 0 and sizes[index] == 0:
                donor = max((idx for idx in range(3) if sizes[idx] > 1),
                            key=lambda idx: sizes[idx])
                sizes[donor] -= 1
                sizes[index] += 1
    result = {}
    cursor = 0
    for name, size in zip(names, sizes):
        for value in ordered[cursor:cursor + size]:
            result[value] = name
        cursor += size
    if set(result) != set(video_ids):
        raise AssertionError("source partition is not total")
    return result


def make_artificial_mask(validity: torch.Tensor, strategy: str, *, seed: int,
                         ratio: float = 0.25, span_length: int = 8) -> torch.Tensor:
    """Return masks shaped (N,T,V); true means deliberately hidden."""
    if validity.dtype != torch.bool or validity.ndim != 3:
        raise ValueError("validity must be bool (N,T,V)")
    n, frames, joints = validity.shape
    result = np.zeros((n, frames, joints), dtype=np.bool_)
    for batch in range(n):
        if strategy == "point":
            candidate = random_point_mask(frames, joints, ratio, seed + batch)
        elif strategy == "span":
            spans = max(1, int(round(ratio * frames / span_length)))
            candidate = span_mask(frames, joints, span_length, spans,
                                  seed=seed + batch)
        elif strategy in {"left_hand_tube", "right_hand_tube", "face_tube"}:
            candidate = np.zeros((frames, joints), dtype=np.bool_)
            region = strategy.removesuffix("_tube")
            candidate[:, OPENPOSE_LANDMARK_PARTS[region]] = True
        else:
            raise ValueError(f"unknown masking strategy: {strategy}")
        result[batch] = candidate & validity[batch].cpu().numpy()
    mask = torch.from_numpy(result).to(device=validity.device)
    if not mask.any():
        raise ValueError("artificial mask hides no originally valid observation")
    if not (validity & ~mask).any():
        raise ValueError("artificial mask leaves no visible observation")
    return mask


def build_masked_input(values: torch.Tensor, confidence: torch.Tensor,
                       source_validity: torch.Tensor,
                       artificial_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build five channels without reading deliberately hidden coordinates."""
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values must be (N,2,T,V)")
    expected = (values.shape[0], values.shape[2], values.shape[3])
    if confidence.shape != expected or source_validity.shape != expected \
            or artificial_mask.shape != expected:
        raise ValueError("confidence/validity/mask must align with values")
    if source_validity.dtype != torch.bool or artificial_mask.dtype != torch.bool:
        raise ValueError("validity and artificial mask must be bool")
    if not torch.isfinite(values).all() or not torch.isfinite(confidence).all():
        raise ValueError("values and confidence must be finite")
    if torch.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be in [0,1]")
    target_mask = source_validity & artificial_mask
    visible = source_validity & ~artificial_mask
    xy = torch.where(visible[:, None], values, torch.zeros_like(values))
    conf = torch.where(visible, confidence, torch.zeros_like(confidence))[:, None]
    features = torch.cat((xy, conf, source_validity[:, None].to(values.dtype),
                          artificial_mask[:, None].to(values.dtype)), dim=1)
    return features, target_mask


class Pose2DMaskedReconstructor(nn.Module):
    """Small graph-temporal model over the actual 137-node OpenPose topology."""

    def __init__(self, hidden_channels: int = 24) -> None:
        super().__init__()
        graph = openpose_holistic_graph()
        adjacency = graph.adjacency()
        self.input = nn.Conv2d(5, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden_channels, hidden_channels, adjacency,
                       temporal_kernel=5, residual=True),
            STGCNBlock(hidden_channels, hidden_channels, adjacency,
                       temporal_kernel=5, residual=True),
        ])
        self.output = nn.Conv2d(hidden_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != 5 or features.shape[-1] != 137:
            raise ValueError("features must be (N,5,T,137)")
        hidden = self.input(features)
        for block in self.blocks:
            hidden = block(hidden)
        return torch.sigmoid(self.output(hidden))


def masked_coordinate_error(prediction: torch.Tensor, target: torch.Tensor,
                            target_mask: torch.Tensor) -> torch.Tensor:
    """Mean 2D Euclidean error over artificial, originally valid targets only."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must align as (N,2,T,V)")
    if target_mask.shape != (target.shape[0], target.shape[2], target.shape[3]) \
            or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be bool (N,T,V)")
    if not target_mask.any():
        raise ValueError("no evaluable masked target")
    distance = torch.linalg.vector_norm(prediction - target, dim=1)
    return distance[target_mask].mean()


def masked_velocity_error(prediction: torch.Tensor, target: torch.Tensor,
                          target_mask: torch.Tensor,
                          timestamps: torch.Tensor) -> tuple[torch.Tensor, int]:
    if timestamps.ndim != 2 or timestamps.shape != target_mask.shape[:2]:
        raise ValueError("timestamps must be (N,T)")
    dt = timestamps[:, 1:] - timestamps[:, :-1]
    if not torch.isfinite(dt).all() or torch.any(dt <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    evaluable = target_mask[:, :-1] & target_mask[:, 1:]
    count = int(evaluable.sum())
    if count == 0:
        return target.new_tensor(float("nan")), 0
    pred_velocity = (prediction[:, :, 1:] - prediction[:, :, :-1]) / dt[:, None, :, None]
    target_velocity = (target[:, :, 1:] - target[:, :, :-1]) / dt[:, None, :, None]
    distance = torch.linalg.vector_norm(pred_velocity - target_velocity, dim=1)
    return distance[evaluable].mean(), count


def temporal_interpolation_baseline(values: torch.Tensor, visible: torch.Tensor
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation using visible temporal neighbours only."""
    prediction = torch.zeros_like(values)
    evaluable = torch.zeros_like(visible)
    n, _, frames, joints = values.shape
    for batch in range(n):
        for joint in range(joints):
            indices = torch.where(visible[batch, :, joint])[0]
            if indices.numel() < 2:
                continue
            for left, right in zip(indices[:-1].tolist(), indices[1:].tolist()):
                if right - left <= 1:
                    continue
                weights = torch.arange(1, right - left, device=values.device,
                                       dtype=values.dtype) / (right - left)
                prediction[batch, :, left + 1:right, joint] = (
                    values[batch, :, left, joint, None] * (1 - weights[None])
                    + values[batch, :, right, joint, None] * weights[None]
                )
                evaluable[batch, left + 1:right, joint] = True
    return prediction, evaluable


def last_observation_baseline(values: torch.Tensor, visible: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.zeros_like(values)
    evaluable = torch.zeros_like(visible)
    n, _, frames, joints = values.shape
    for batch in range(n):
        for joint in range(joints):
            last = None
            for frame in range(frames):
                if bool(visible[batch, frame, joint]):
                    last = values[batch, :, frame, joint]
                elif last is not None:
                    prediction[batch, :, frame, joint] = last
                    evaluable[batch, frame, joint] = True
    return prediction, evaluable


def coordinate_mean_baseline(values: torch.Tensor, visible: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.zeros_like(values)
    evaluable = torch.zeros_like(visible)
    for batch in range(values.shape[0]):
        observed = visible[batch]
        if not observed.any():
            continue
        expanded = observed[None].expand(2, -1, -1)
        mean = values[batch][expanded].reshape(2, -1).mean(dim=1)
        prediction[batch] = mean[:, None, None]
        evaluable[batch] = True
    return prediction, evaluable


def _safe_error(prediction: torch.Tensor, target: torch.Tensor,
                target_mask: torch.Tensor, supported: torch.Tensor) -> tuple[float | None, int]:
    evaluable = target_mask & supported
    if not evaluable.any():
        return None, 0
    return float(masked_coordinate_error(prediction, target, evaluable)), int(evaluable.sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_identity() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", os.fspath(repo), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("experiment requires a readable Git revision") from error
    digest = hashlib.sha256()
    sources = (
        Path(__file__), Path(__file__).parents[1] / "models" / "stgcn.py",
        Path(__file__).parents[1] / "data_engineering" / "how2sign.py",
    )
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return {"git_revision": revision, "implementation_sha256": digest.hexdigest()}


def _load_audit_candidates(audit_dir: Path) -> list[dict]:
    manifest = json.loads((audit_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("audit_complete"):
        raise ValueError("motion experiment requires a completed audit")
    expected_database = manifest.get("audit_database", {}).get("sha256")
    if not expected_database or _sha256(audit_dir / "audit.sqlite3") != expected_database:
        raise ValueError("completed audit database hash does not match its manifest")
    with sqlite3.connect(audit_dir / "audit.sqlite3") as connection:
        rows = connection.execute("""
            SELECT sample_id,video_id,filename_code,raw_uri,raw_sha256,openpose_sha256,
                   quality_json,duration
            FROM clips WHERE status IN ('valid','quality_warning')
            ORDER BY sample_id
        """).fetchall()
    names = ("sample_id", "video_id", "filename_code", "raw_uri", "raw_sha256",
             "openpose_sha256", "quality_json", "duration")
    result = []
    for row in rows:
        item = dict(zip(names, row))
        item["quality"] = json.loads(item.pop("quality_json"))
        result.append(item)
    return result


def _select_candidates(candidates: Sequence[dict], count: int, seed: int,
                       minimum_frames: int) -> list[dict]:
    eligible = [item for item in candidates
                if item["quality"]["frame_count"] >= minimum_frames]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} audited clips meet the window length")
    coverages = sorted(item["quality"]["coverage"] for item in eligible)
    strata: dict[tuple[str, int, str], list[dict]] = {}
    for item in eligible:
        rank = int(np.searchsorted(coverages, item["quality"]["coverage"], side="right"))
        decile = min(9, 10 * rank // len(coverages))
        duration = "short" if item["duration"] < 2 else "medium" \
            if item["duration"] < 6 else "long"
        strata.setdefault((item["filename_code"], decile, duration), []).append(item)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _load_window(item: Mapping, root: Path, frames: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video = Path(item["raw_uri"]).resolve(strict=True)
    if root.resolve(strict=True) not in video.parents:
        raise ValueError("audited media path escaped the dataset root")
    json_dir = root / "openpose_output" / "json" / item["sample_id"]
    if stable_sha256(video, root).sha256 != item["raw_sha256"]:
        raise ValueError("selected raw media no longer matches the completed audit")
    frame_paths = sorted(json_dir.iterdir(), key=lambda path: path.name)
    openpose_digest = hierarchical_digest(
        (path.name, stable_sha256(path, root)) for path in frame_paths)
    if openpose_digest != item["openpose_sha256"]:
        raise ValueError("selected OpenPose tree no longer matches the completed audit")
    track, _ = decode_how2sign_openpose(video, json_dir, confidence_threshold=0.0)
    if track.values.shape[1] < frames:
        raise ValueError("selected clip is shorter than the configured window")
    start = (track.values.shape[1] - frames) // 2
    stop = start + frames
    values = torch.from_numpy(track.values[:, start:stop]).float()
    confidence = torch.from_numpy(track.confidence[start:stop]).float()
    validity = torch.from_numpy(track.validity_mask[start:stop]).bool()
    timestamps = torch.from_numpy(track.timestamps[start:stop]).double()
    return values, confidence, validity, timestamps


def _batches(indices: Sequence[int], batch_size: int, seed: int) -> list[list[int]]:
    order = list(indices)
    random.Random(seed).shuffle(order)
    return [order[index:index + batch_size] for index in range(0, len(order), batch_size)]


def _stack(records, indices):
    return tuple(torch.stack([records[index][field] for index in indices]) for field in range(4))


def _strategy(step: int) -> str:
    return ("point", "span", "left_hand_tube", "right_hand_tube")[step % 4]


def run_pose2d_experiment(
    audit_dir: str | os.PathLike[str], root: str | os.PathLike[str],
    output: str | os.PathLike[str], *,
    config: Pose2DExperimentConfig = Pose2DExperimentConfig(),
) -> dict:
    config.validate()
    audit_path = Path(audit_dir).resolve(strict=True)
    source_root = Path(root).resolve(strict=True)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing non-empty motion experiment output")
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    selected = _select_candidates(_load_audit_candidates(audit_path),
                                  config.sample_count, config.seed,
                                  config.window_frames)
    sources = sorted({item["video_id"] for item in selected})
    source_assignment = source_disjoint_partition(sources, config.seed)
    records = [_load_window(item, source_root, config.window_frames) for item in selected]
    split_indices = {
        split: [index for index, item in enumerate(selected)
                if source_assignment[item["video_id"]] == split]
        for split in ("train", "val", "test")
    }
    if any(not values for values in split_indices.values()):
        raise ValueError("bounded selection did not produce three nonempty source splits")

    model = Pose2DMaskedReconstructor(config.hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    curves = []
    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for indices in _batches(split_indices["train"], config.batch_size,
                                config.seed + epoch):
            values, confidence, validity, _ = _stack(records, indices)
            strategy = _strategy(global_step)
            mask = make_artificial_mask(
                validity, strategy, seed=config.seed + global_step,
                ratio=config.mask_ratio, span_length=config.span_length)
            features, targets = build_masked_input(values, confidence, validity, mask)
            prediction = model(features)
            loss = masked_coordinate_error(prediction, values, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                raise FloatingPointError("non-finite training gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            global_step += 1
        curves.append({"epoch": epoch, "train_masked_error": float(np.mean(losses))})

    # Tiny-subset overfit is a separate model so it cannot bias held-out metrics.
    tiny = Pose2DMaskedReconstructor(config.hidden_channels)
    tiny_optimizer = torch.optim.Adam(tiny.parameters(), lr=config.learning_rate * 3)
    values, confidence, validity, _ = _stack(records, [split_indices["train"][0]])
    tiny_mask = make_artificial_mask(validity, "span", seed=config.seed,
                                     ratio=config.mask_ratio,
                                     span_length=config.span_length)
    tiny_features, tiny_targets = build_masked_input(values, confidence, validity, tiny_mask)
    tiny_initial = None
    for step in range(config.tiny_overfit_steps):
        prediction = tiny(tiny_features)
        loss = masked_coordinate_error(prediction, values, tiny_targets)
        if step == 0:
            tiny_initial = float(loss.detach())
        tiny_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        tiny_optimizer.step()
    tiny_final = float(masked_coordinate_error(tiny(tiny_features), values,
                                                tiny_targets).detach())
    tiny_ratio = tiny_final / tiny_initial if tiny_initial else None
    if tiny_ratio is None or not math.isfinite(tiny_ratio) or tiny_ratio >= 0.65:
        raise RuntimeError(
            f"tiny-subset optimization gate failed: final/initial={tiny_ratio!r}")

    metrics: dict[str, dict] = {}
    model.eval()
    for split in ("val", "test"):
        per_strategy = {}
        values, confidence, validity, timestamps = _stack(records, split_indices[split])
        for offset, strategy in enumerate(("point", "span", "left_hand_tube",
                                           "right_hand_tube")):
            mask = make_artificial_mask(
                validity, strategy, seed=config.seed + 10_000 + offset,
                ratio=config.mask_ratio, span_length=config.span_length)
            features, targets = build_masked_input(values, confidence, validity, mask)
            with torch.no_grad():
                prediction = model(features)
                model_error = float(masked_coordinate_error(prediction, values, targets))
                velocity, velocity_count = masked_velocity_error(
                    prediction, values, targets, timestamps)
            visible = validity & ~mask
            interpolation, interpolation_supported = temporal_interpolation_baseline(
                values, visible)
            last, last_supported = last_observation_baseline(values, visible)
            mean, mean_supported = coordinate_mean_baseline(values, visible)
            per_strategy[strategy] = {
                "masked_targets": int(targets.sum()),
                "model_coordinate_error": model_error,
                "model_velocity_error": float(velocity) if velocity_count else None,
                "model_velocity_pairs": velocity_count,
                "interpolation": _safe_error(interpolation, values, targets,
                                             interpolation_supported),
                "last_observation": _safe_error(last, values, targets, last_supported),
                "coordinate_mean": _safe_error(mean, values, targets, mean_supported),
            }
        metrics[split] = per_strategy

    checkpoint = destination / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config)}, checkpoint)
    reload_model = Pose2DMaskedReconstructor(config.hidden_channels)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reload_model.load_state_dict(reloaded["model"])
    reload_model.eval()
    probe_indices = split_indices["test"][:1]
    values, confidence, validity, _ = _stack(records, probe_indices)
    probe_mask = make_artificial_mask(validity, "span", seed=config.seed + 99,
                                      span_length=config.span_length)
    probe_features, _ = build_masked_input(values, confidence, validity, probe_mask)
    with torch.no_grad():
        if not torch.equal(model(probe_features), reload_model(probe_features)):
            raise RuntimeError("checkpoint reload changed inference")

    selection = [{
        "sample_id": item["sample_id"], "video_id": item["video_id"],
        "source_split": source_assignment[item["video_id"]],
        "raw_sha256": item["raw_sha256"], "filename_code": item["filename_code"],
        "openpose_sha256": item["openpose_sha256"],
    } for item in selected]
    (destination / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    (destination / "selected_samples.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    with (destination / "learning_curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_masked_error"))
        writer.writeheader(); writer.writerows(curves)
    result = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_complete": True,
        "scope": "quarantined_2d_masked_reconstruction",
        "translation_claim": False,
        "linguistic_claim": False,
        "three_dimensional_claim": False,
        "production_integration": False,
        "partition": "VIDEO_ID-disjoint; not signer-disjoint",
        "split_counts": {key: len(value) for key, value in split_indices.items()},
        "tiny_overfit": {
            "initial_error": tiny_initial, "final_error": tiny_final,
            "ratio": tiny_ratio, "maximum_accepted_ratio": 0.65,
        },
        "metrics": metrics,
        "interpolation_certificate": {
            "random_point_typical_floor": typical_masked_floor(
                random_point_mask(config.window_frames, 4, config.mask_ratio,
                                  config.seed)),
            "span_typical_floor": typical_masked_floor(
                span_mask(config.window_frames, 4, config.span_length, 2,
                          seed=config.seed)),
        },
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "implementation": _implementation_identity(),
    }
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    artifact_hashes = {
        path.name: _sha256(path) for path in sorted(destination.iterdir()) if path.is_file()
    }
    manifest = {"artifacts": artifact_hashes, "audit_manifest_sha256": _sha256(
        audit_path / "audit_manifest.json"), "result": result}
    (destination / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quarantined How2Sign 2D reconstruction")
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = Pose2DExperimentConfig(sample_count=args.sample_count, epochs=args.epochs)
    result = run_pose2d_experiment(args.audit_dir, args.root, args.output, config=config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
