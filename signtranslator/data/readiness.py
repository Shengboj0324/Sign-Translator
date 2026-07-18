"""Training-readiness audit for a corpus.

Answers the question a practitioner should ask *before* burning GPU hours:
"is this dataset actually fit to train on?" It checks structural and statistical
preconditions that, when violated, produce silently bad models:

  * enough samples, and enough per class
  * every class present in train **and** val (otherwise val metrics are blind)
  * class balance (a skewed corpus inflates accuracy on the majority class)
  * **split leakage** - identical samples appearing in both train and val, which
    makes held-out metrics meaningless
  * CTC feasibility - input frames must exceed target length, else the loss is
    degenerate/infinite
  * normalisation sanity - standardised data should be ~zero-mean/unit-variance
  * pose quality (delegated to :mod:`quality`)

Each check yields a pass/fail with an explanation; the report is gated so a
pipeline can refuse to train on an unfit corpus.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .corpus import CorpusSpec, load_manifest, SignDataset
from .quality import inspect_pose, QualityReport


@dataclass
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ReadinessReport:
    checks: List[ReadinessCheck] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)
    quality: Optional[QualityReport] = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(ReadinessCheck(name, passed, detail))

    def summary(self) -> str:
        lines = ["Training-readiness report", "=" * 62]
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:<28} {c.detail}")
        lines.append("-" * 62)
        lines.append(f"  READY: {'YES' if self.passed else 'NO'}")
        return "\n".join(lines)


def _sample_hashes(pose: torch.Tensor) -> List[int]:
    flat = pose.reshape(pose.shape[0], -1).numpy()
    return [hash(flat[i].tobytes()) for i in range(flat.shape[0])]


def assess_corpus(corpus_dir: str, min_train_samples: int = 32,
                  min_per_class: int = 4, max_imbalance_ratio: float = 10.0,
                  run_quality: bool = True) -> ReadinessReport:
    """Audit an on-disk corpus and return a gated readiness report."""
    manifest = load_manifest(corpus_dir)
    spec = CorpusSpec(**manifest["spec"])
    report = ReadinessReport()

    splits = list(manifest["splits"].keys())
    report.add("splits_present", {"train", "val"}.issubset(set(splits)),
               f"found splits: {splits}")
    if not {"train", "val"}.issubset(set(splits)):
        return report

    train = SignDataset(corpus_dir, "train", normalize=False)
    val = SignDataset(corpus_dir, "val", normalize=False)
    n_train, n_val = len(train), len(val)
    report.stats["train_samples"] = n_train
    report.stats["val_samples"] = n_val

    report.add("sample_count", n_train >= min_train_samples,
               f"train={n_train} (min {min_train_samples}), val={n_val}")

    # ---- class coverage & balance ----------------------------------------
    train_counts = Counter(int(c) for row in train.concepts for c in row)
    val_counts = Counter(int(c) for row in val.concepts for c in row)
    missing_train = [k for k in range(spec.num_concepts) if train_counts.get(k, 0) == 0]
    missing_val = [k for k in range(spec.num_concepts) if val_counts.get(k, 0) == 0]
    report.add("class_coverage_train", not missing_train,
               f"{spec.num_concepts - len(missing_train)}/{spec.num_concepts} classes present"
               + (f", missing {missing_train}" if missing_train else ""))
    report.add("class_coverage_val", not missing_val,
               f"{spec.num_concepts - len(missing_val)}/{spec.num_concepts} classes present"
               + (f", missing {missing_val}" if missing_val else ""))

    least = min(train_counts.values()) if train_counts else 0
    most = max(train_counts.values()) if train_counts else 0
    report.stats["min_class_count"] = least
    report.stats["max_class_count"] = most
    report.add("min_examples_per_class", least >= min_per_class,
               f"rarest class has {least} occurrences (min {min_per_class})")
    ratio = (most / least) if least else float("inf")
    report.stats["imbalance_ratio"] = ratio
    report.add("class_balance", ratio <= max_imbalance_ratio,
               f"max/min class frequency ratio = {ratio:.2f} (limit {max_imbalance_ratio})")

    # ---- split leakage ----------------------------------------------------
    train_hashes = set(_sample_hashes(train.pose))
    overlap = sum(1 for h in _sample_hashes(val.pose) if h in train_hashes)
    report.stats["val_leakage"] = overlap
    report.add("no_split_leakage", overlap == 0,
               f"{overlap} val samples are byte-identical to train samples")

    # ---- CTC feasibility --------------------------------------------------
    max_target = int(train.lengths.max()) if n_train else 0
    frames = spec.num_frames
    report.add("ctc_length_feasible", frames >= max_target,
               f"frames={frames} >= max target length={max_target}")

    # ---- normalisation sanity --------------------------------------------
    norm_pose = SignDataset(corpus_dir, "train", normalize=True).pose
    mean_abs = float(norm_pose.mean().abs())
    var = float(norm_pose.var())
    report.stats["normalized_mean"] = mean_abs
    report.stats["normalized_var"] = var
    report.add("normalization_sane", mean_abs < 0.1 and 0.5 < var < 2.0,
               f"standardised mean|{mean_abs:.4f}| var={var:.3f} (want ~0 / ~1)")

    # ---- pose quality -----------------------------------------------------
    if run_quality:
        q = inspect_pose(train.pose)
        report.quality = q
        report.add("pose_quality", q.is_clean,
                   "clean" if q.is_clean else "; ".join(q.issues))

    return report
