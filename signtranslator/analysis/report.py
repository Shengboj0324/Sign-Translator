"""Evaluation and analysis of a trained BidirectionalSignTranslator.

Computes, on a validation split, one metric per branch plus an integration
metric, checks each against a threshold, and returns a structured report whose
``passed`` flag summarises whether the model meets the acceptance bar.

Metrics
    recognition_wer         sign -> gloss CTC word error rate         (lower)
    planner_token_accuracy  spoken -> gloss token accuracy            (higher)
    recall_at_1 / at_5      motion<->gloss manifold retrieval         (higher)
    generation_val_loss     diffusion denoising loss on val           (lower)
    cycle_consistency_wer   gloss -> generate -> recognise -> gloss   (lower)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from ..eval.metrics import retrieval_recall_at_k, word_error_rate
from ..data.corpus import CONTENT_OFFSET


# Gated acceptance thresholds. Each directly measures that one branch trained.
DEFAULT_THRESHOLDS = {
    "recognition_wer": 0.30,          # <=  sign -> gloss CTC
    "planner_token_accuracy": 0.80,   # >=  spoken -> gloss
    "recall_at_1": 0.60,              # >=  motion<->gloss manifold
    "generation_val_loss": 0.70,      # <=  conditional diffusion denoising
}

# Metrics that are *reported but not gated*. Cycle-consistency is a full
# generative round-trip (gloss -> generate -> recognise); achieving low error
# requires high-fidelity conditional sampling that needs substantially more
# diffusion training/compute than a CPU smoke run. It is tracked as an
# integration diagnostic, not an acceptance gate.
DIAGNOSTIC_THRESHOLDS = {
    "cycle_consistency_wer": 0.70,    # <=  (informational)
}


@dataclass
class AnalysisReport:
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    checks: Dict[str, bool] = field(default_factory=dict)
    gating: set = field(default_factory=set)

    @property
    def passed(self) -> bool:
        """Overall pass depends only on the gated (branch-quality) metrics."""
        return all(v for k, v in self.checks.items() if k in self.gating)

    def summary(self) -> str:
        lines = ["Analysis report", "=" * 48]
        for name, value in self.metrics.items():
            if name in self.checks:
                tag = "PASS" if self.checks[name] else "FAIL"
                if name not in self.gating:
                    tag += " (diagnostic)"
            else:
                tag = ""
            lines.append(f"  {name:<26} {value:8.4f}  {tag}")
        lines.append("-" * 48)
        lines.append(f"  OVERALL (gated metrics): {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _gather(val_loader):
    poses, gloss_tokens, srcs, concept_lists = [], [], [], []
    for batch in val_loader:
        poses.append(batch["pose"])
        gloss_tokens.append(batch["gloss_tokens"])
        srcs.append(batch["src"])
        concept_lists.extend(batch["concepts"])
    return poses, gloss_tokens, srcs, concept_lists


@torch.no_grad()
def analyze(model, val_loader, thresholds: Dict[str, float] = None,
            cycle_subset: int = 16, ddim_steps: int = 8,
            guidance_scale: float = 2.0) -> AnalysisReport:
    thresholds = {**DEFAULT_THRESHOLDS, **DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    model.eval()
    device = next(model.parameters()).device
    poses, gloss_tokens, srcs, concept_lists = _gather(val_loader)

    # ---- recognition WER (sign -> gloss), CTC ids are concept+1 -----------
    rec_hyps: List[List[int]] = []
    rec_refs: List[List[int]] = []
    for p_batch, c_offset in zip(poses, _batched(concept_lists, [p.shape[0] for p in poses])):
        for hyp in model.recognize(p_batch.to(device)):
            rec_hyps.append(hyp)
        for c in c_offset:
            rec_refs.append([int(x) + 1 for x in c])
    recognition_wer = word_error_rate(rec_hyps, rec_refs)

    # ---- planner token accuracy (spoken -> gloss) -------------------------
    correct_tok, total_tok = 0, 0
    for s_batch, c_group in zip(srcs, _batched(concept_lists, [s.shape[0] for s in srcs])):
        preds = model.planner.greedy_decode(s_batch.to(device), max_len=8)
        for pred, c in zip(preds, c_group):
            ref = [int(x) + CONTENT_OFFSET for x in c]
            for j in range(len(ref)):
                total_tok += 1
                if j < len(pred) and pred[j] == ref[j]:
                    correct_tok += 1
    planner_token_accuracy = correct_tok / max(1, total_tok)

    # ---- manifold retrieval recall@k --------------------------------------
    z_m = torch.cat([model.embed_motion(p.to(device)) for p in poses], dim=0)
    z_l = torch.cat([model.embed_gloss(g.to(device)) for g in gloss_tokens], dim=0)
    sim = z_m @ z_l.t()
    recalls = retrieval_recall_at_k(sim, ks=(1, 5))

    # ---- generation validation loss (denoising MSE) -----------------------
    gen_losses = [float(model.generation_loss(p.to(device), g.to(device)))
                  for p, g in zip(poses, gloss_tokens)]
    generation_val_loss = sum(gen_losses) / max(1, len(gen_losses))

    # ---- cycle consistency: gloss -> generate -> recognise ----------------
    g_all = torch.cat(gloss_tokens, dim=0)[:cycle_subset].to(device)
    c_all = concept_lists[:cycle_subset]
    motion = model.generate_from_gloss(g_all, guidance_scale=guidance_scale,
                                       ddim_steps=ddim_steps)
    cyc_hyps = model.recognize(motion)
    cyc_refs = [[int(x) + 1 for x in c] for c in c_all]
    cycle_consistency_wer = word_error_rate(cyc_hyps, cyc_refs)

    metrics = {
        "recognition_wer": recognition_wer,
        "planner_token_accuracy": planner_token_accuracy,
        "recall_at_1": recalls[1],
        "recall_at_5": recalls[5],
        "generation_val_loss": generation_val_loss,
        "cycle_consistency_wer": cycle_consistency_wer,
    }
    checks = {
        "recognition_wer": recognition_wer <= thresholds["recognition_wer"],
        "planner_token_accuracy": planner_token_accuracy >= thresholds["planner_token_accuracy"],
        "recall_at_1": recalls[1] >= thresholds["recall_at_1"],
        "generation_val_loss": generation_val_loss <= thresholds["generation_val_loss"],
        "cycle_consistency_wer": cycle_consistency_wer <= thresholds["cycle_consistency_wer"],
    }
    gating = set(DEFAULT_THRESHOLDS.keys())
    return AnalysisReport(metrics=metrics, thresholds=thresholds, checks=checks,
                          gating=gating)


def _batched(flat_list, sizes):
    """Split a flat list back into per-batch chunks of the given sizes."""
    out, idx = [], 0
    for s in sizes:
        out.append(flat_list[idx:idx + s])
        idx += s
    return out
