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
from typing import Dict, List, Optional

import torch

from ..eval.metrics import retrieval_recall_at_k, word_error_rate
from ..data.corpus import CONTENT_OFFSET


# Gated acceptance thresholds. Each directly measures that one branch trained.
#
# ``cycle_consistency_wer`` is the strictest, fully end-to-end check: gloss ->
# generate 3D motion -> re-recognise. It only passes when generated motion is
# faithful enough that the recogniser reads it about as well as ground-truth
# motion, which requires x0-parameterization, pose standardisation, a velocity
# loss, and high-noise timestep emphasis (see docs/MATH.md).
DEFAULT_THRESHOLDS = {
    "recognition_wer": 0.30,          # <=  sign -> gloss CTC
    "planner_token_accuracy": 0.80,   # >=  spoken -> gloss
    "recall_at_1": 0.60,              # >=  motion<->gloss manifold
    "generation_val_loss": 0.70,      # <=  conditional diffusion objective
    "cycle_consistency_wer": 0.25,    # <=  gloss -> generate -> recognise
    "speech_wer": 0.30,               # <=  audio -> spoken tokens (CTC)
}

# Reserved for metrics that are reported but not gated (currently none).
DIAGNOSTIC_THRESHOLDS: Dict[str, float] = {}


@dataclass
class AnalysisReport:
    metrics: Dict[str, Optional[float]]
    thresholds: Dict[str, float]
    checks: Dict[str, bool] = field(default_factory=dict)
    gating: set = field(default_factory=set)
    protocol: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Overall pass depends only on the gated (branch-quality) metrics."""
        return bool(self.gating) and all(self.checks.get(k, False) for k in self.gating)

    def summary(self) -> str:
        lines = ["Analysis report", "=" * 48]
        for name, value in self.metrics.items():
            if name in self.checks:
                tag = "PASS" if self.checks[name] else "FAIL"
                if name not in self.gating:
                    tag += " (diagnostic)"
            else:
                tag = ""
            rendered = "     N/A" if value is None else f"{value:8.4f}"
            lines.append(f"  {name:<26} {rendered}  {tag}")
        lines.append("-" * 48)
        lines.append(f"  OVERALL (gated metrics): {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


@torch.no_grad()
def analyze(model, val_loader, thresholds: Dict[str, float] = None,
            cycle_subset: int = 24, ddim_steps: int = 20,
            guidance_scale: float = 1.0, *, seed: int = 0,
            planner_max_len: int = 32, checkpoint_identity: str = "in-memory") -> AnalysisReport:
    """Macro-sample generation loss; corpus edit rates; fixed candidate retrieval.

    Each observation is cropped before inference. Randomness is isolated and the
    sequence cap is fixed before looking at reference lengths. A missing EOS is
    reported as truncation and fails the planner gate, even for matching content.
    These synthetic branch diagnostics do not establish ASL comprehension.
    """
    from ..reproducibility import isolated_deterministic_rng
    from ..models.recognition import _levenshtein
    from .observations import observations
    import math
    import hashlib

    if isinstance(cycle_subset, bool) or not isinstance(cycle_subset, int) or cycle_subset < 1:
        raise ValueError("cycle_subset must be a positive integer")
    thresholds = {**DEFAULT_THRESHOLDS, **DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    if any(not math.isfinite(v) for v in thresholds.values()):
        raise ValueError("analysis thresholds must be finite")
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    rec_hyps, rec_refs, plan_hyps, plan_refs = [], [], [], []
    cyc_hyps, cyc_refs, sp_hyps, sp_refs = [], [], [], []
    zm, zl, gen_losses = [], [], []
    completed = []
    model.eval()
    try:
        with isolated_deterministic_rng(seed):
            for batch in val_loader:
                for row in observations(batch):
                    row = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in row.items()}
                    p, g = row['pose'], row['gloss_tokens']
                    support = {k: row[k] for k in ('validity_mask', 'confidence', 'frame_mask')
                               if k in row}
                    p, _ = model.diffusion.motion_support(p, **support)
                    refs = [int(x) + 1 for x in row['concepts'][0]]
                    rec_hyps.extend(model.recognize(p))
                    rec_refs.append(refs)
                    reference = [int(x) + CONTENT_OFFSET for x in row['concepts'][0]]
                    if len(reference) >= planner_max_len:
                        raise ValueError("reference plus EOS exceeds declared planner evaluation cap")
                    hypotheses, ended = model.planner.greedy_decode(
                        row['src'], max_len=planner_max_len, return_status=True)
                    plan_hyps.extend(hypotheses)
                    plan_refs.append(reference)
                    completed.extend(ended)
                    zm.append(model.embed_motion(p))
                    zl.append(model.embed_gloss(g))
                    gen_losses.append(float(model.generation_loss(p, g, **support)))
                    if len(cyc_refs) < cycle_subset:
                        motion = model.generate_from_gloss(
                            g, num_frames=p.shape[2], guidance_scale=guidance_scale,
                            ddim_steps=ddim_steps)
                        cyc_hyps.extend(model.recognize(motion))
                        cyc_refs.append(refs)
                    if 'speech' in row:
                        if 'speech_ctc_targets' not in row:
                            raise ValueError("speech evaluation requires explicit source targets")
                        sp_hyps.extend(model.recognize_speech(row['speech']))
                        sp_refs.append(row['speech_ctc_targets'].tolist())
            if not gen_losses:
                raise ValueError("analysis requires nonempty evaluation observations")
            sim = torch.cat(zm) @ torch.cat(zl).t()
            recalls = retrieval_recall_at_k(sim, ks=(1, 5))
    finally:
        for module, training in modes:
            module.training = training

    edits = sum(_levenshtein(h, r) for h, r in zip(plan_hyps, plan_refs))
    edit_support = sum(max(len(h), len(r)) for h, r in zip(plan_hyps, plan_refs))
    metrics = {
        'recognition_wer': word_error_rate(rec_hyps, rec_refs),
        'planner_token_accuracy': 1.0 - edits / edit_support,
        'planner_wer': word_error_rate(plan_hyps, plan_refs),
        'planner_exact_match': sum(h == r and end for h, r, end in
                                   zip(plan_hyps, plan_refs, completed)) / len(plan_refs),
        'planner_truncation_rate': 1.0 - sum(completed) / len(completed),
        'planner_semantic_field_accuracy': None,
        'recall_at_1': recalls[1], 'recall_at_5': recalls[5],
        'generation_val_loss': sum(gen_losses) / len(gen_losses),
        'cycle_consistency_wer': word_error_rate(cyc_hyps, cyc_refs),
        'speech_wer': word_error_rate(sp_hyps, sp_refs) if sp_refs else None,
    }
    if any(v is not None and not math.isfinite(v) for v in metrics.values()):
        raise ValueError("analysis produced nonfinite metrics")
    checks = {k: metrics[k] is not None and
              (metrics[k] >= thresholds[k] if k in ('planner_token_accuracy', 'recall_at_1')
               else metrics[k] <= thresholds[k]) for k in DEFAULT_THRESHOLDS}
    checks['planner_token_accuracy'] &= all(completed)
    # Partial acoustic coverage cannot pass a whole-corpus speech gate.
    checks['speech_wer'] &= len(sp_refs) == len(rec_refs)
    return AnalysisReport(metrics, thresholds, checks, set(DEFAULT_THRESHOLDS),
                          {'seed': seed, 'replicates': 1, 'checkpoint': checkpoint_identity,
                           'model_state_sha256': digest.hexdigest(),
                           'observations': len(rec_refs), 'speech_observations': len(sp_refs),
                           'cycle_observations': len(cyc_refs), 'planner_max_len': planner_max_len,
                           'generation_estimand': 'mean of per-observation supported losses',
                           'planner_token_accuracy_definition': 'one minus edits / sum(max(hyp, ref) lengths)',
                           'semantic_fields': 'unavailable: no governed SIR field references'})
