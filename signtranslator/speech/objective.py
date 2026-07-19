"""The speech-layer training objective.

Implements the composite loss of ``01_speech_foundation_layer.md``:

    L = L_ASR + lambda_c * L_contrast + lambda_b * L_boundary
        + lambda_cal * L_Brier

Each term exists for a distinct, measurable reason, and Stage 4's tests check
that each one actually moves *its own* metric rather than merely reducing a
scalar:

``L_ASR`` (CTC)
    The transcription objective; streaming-compatible because it needs no
    frame-level labels.

``L_contrast`` (symmetric InfoNCE)
    Aligns pooled speech with target-sign representations. The source document
    is emphatic that this is **not** proof of semantic equivalence -- a high
    batch-level retrieval score can come from speaker, channel or duration
    artefacts rather than meaning. It must be validated by retrieval *and*
    downstream ablation, never assumed. This module therefore only computes the
    loss; :mod:`signtranslator.speech.policy` and the Stage 5 harness judge it.

``L_boundary``
    Per-frame supervision of word starts, which is what makes the Stage 2
    timestamps trustworthy. Boundaries are rare (roughly ``L`` positives among
    ``T`` frames, often under 5%), so an unweighted BCE is minimised almost
    perfectly by predicting "never a boundary". The loss is therefore
    class-balanced by ``pos_weight = #neg / #pos``; without it the term is
    worse than useless because it looks converged while predicting nothing.

``L_Brier``
    The calibration term from Stage 3, applied to frame posteriors against
    forced-alignment labels. Strictly proper, so it pushes the posteriors toward
    the true conditional distribution rather than merely toward the right
    argmax.

Frame-level targets for the boundary and Brier terms come from a CTC forced
alignment of the *known* transcript, computed under ``no_grad``: the alignment
is a target, not a differentiable path, and letting gradients flow through it
would let the model move the goalposts rather than fit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import ctc_forced_alignment, minimum_frames_required
from .calibration import BrierLoss
from ..models.alignment import info_nce_loss


# ---------------------------------------------------------------------------
# Boundary supervision
# ---------------------------------------------------------------------------
def boundary_targets_from_alignment(alignment, num_frames: int) -> torch.Tensor:
    """Binary per-frame targets marking the first frame of each token.

    Token ``j`` (1-based) occupies extended state ``2j - 1``; its boundary is the
    first frame assigned to that state.
    """
    targets = torch.zeros(num_frames, dtype=torch.float32)
    for j in range(1, len(alignment.targets) + 1):
        state = 2 * j - 1
        for t, s in enumerate(alignment.path):
            if s == state:
                if t < num_frames:
                    targets[t] = 1.0
                break
    return targets


def balanced_pos_weight(targets: torch.Tensor, cap: float = 1000.0) -> torch.Tensor:
    """``#negatives / #positives``, so both classes contribute equally.

    Capped because a batch with a single positive would otherwise produce an
    enormous weight and a gradient spike. Returns 1.0 when a class is absent,
    since no balancing is meaningful then.
    """
    pos = float(targets.sum())
    neg = float(targets.numel()) - pos
    if pos <= 0 or neg <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(neg / pos, cap))


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor,
                  balanced: bool = True) -> torch.Tensor:
    """Class-balanced BCE-with-logits over per-frame boundary predictions."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if logits.numel() == 0:
        raise ValueError("empty input")
    pos_weight = balanced_pos_weight(targets) if balanced else None
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight.to(logits.device) if balanced else None)


class BoundaryHead(nn.Module):
    """Per-frame word-boundary detector on encoder hidden states."""

    def __init__(self, hidden_dim: int, inner_dim: Optional[int] = None) -> None:
        super().__init__()
        inner = inner_dim or hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner), nn.GELU(), nn.Linear(inner, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(N, T, H) -> (N, T)`` boundary logits."""
        if hidden.dim() != 3:
            raise ValueError("hidden must be (N, T, H)")
        return self.net(hidden).squeeze(-1)

    @torch.no_grad()
    def predict_boundaries(self, hidden: torch.Tensor,
                           threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(hidden)) >= threshold)


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------
@dataclass
class ObjectiveWeights:
    """The lambda coefficients of the composite loss."""

    contrastive: float = 0.1      # lambda_c
    boundary: float = 0.5         # lambda_b
    brier: float = 0.1            # lambda_cal

    def __post_init__(self) -> None:
        for name in ("contrastive", "boundary", "brier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} weight must be non-negative")


@dataclass
class ObjectiveOutput:
    total: torch.Tensor
    terms: Dict[str, torch.Tensor] = field(default_factory=dict)

    def detached(self) -> Dict[str, float]:
        return {k: float(v.detach()) for k, v in self.terms.items()}


class SpeechTrainingObjective(nn.Module):
    """Composes the four loss terms over a batch.

    Args:
        recognizer: a ``SpeechRecognizer`` exposing ``encode`` and ``forward``.
        boundary_head: optional; created automatically when ``hidden_dim`` given.
        aligner: optional ``ContrastiveAligner`` for the speech<->sign term.
    """

    def __init__(self, recognizer: nn.Module,
                 weights: Optional[ObjectiveWeights] = None,
                 boundary_head: Optional[BoundaryHead] = None,
                 aligner: Optional[nn.Module] = None,
                 contrastive_temperature: float = 0.07,
                 blank: int = 0) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.weights = weights or ObjectiveWeights()
        self.boundary_head = boundary_head
        self.aligner = aligner
        self.contrastive_temperature = contrastive_temperature
        self.blank = blank
        self.brier = BrierLoss(from_logits=False)

    # -- frame targets ------------------------------------------------------
    @torch.no_grad()
    def _frame_targets(self, log_probs: torch.Tensor, targets: torch.Tensor,
                       target_lengths: torch.Tensor,
                       input_lengths: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forced-alignment frame labels and boundary targets.

        Returns ``(frame_labels, boundary_targets, valid_mask)``. Alignment runs
        over each sample's VALID frames only (``input_lengths[i]``), so padding
        frames are never aligned or supervised. Samples whose transcript cannot
        fit in the available frames are marked invalid rather than silently
        dropped or padded, so they cannot skew the loss.
        """
        n, t, _ = log_probs.shape
        frame_labels = torch.zeros(n, t, dtype=torch.long, device=log_probs.device)
        boundaries = torch.zeros(n, t, dtype=torch.float32, device=log_probs.device)
        valid = torch.zeros(n, dtype=torch.bool, device=log_probs.device)

        for i in range(n):
            li = int(input_lengths[i])
            tokens = targets[i, :int(target_lengths[i])].tolist()
            if not tokens or li < minimum_frames_required(tokens):
                continue
            try:
                al = ctc_forced_alignment(log_probs[i, :li], tokens, blank=self.blank)
            except ValueError:
                continue
            frame_labels[i, :li] = torch.tensor(al.state_tokens(), dtype=torch.long,
                                                device=log_probs.device)
            boundaries[i, :li] = boundary_targets_from_alignment(al, li).to(log_probs.device)
            valid[i] = True
        return frame_labels, boundaries, valid

    # -- forward ------------------------------------------------------------
    def forward(self, features: torch.Tensor, targets: torch.Tensor,
                target_lengths: torch.Tensor,
                sign_embeddings: Optional[torch.Tensor] = None,
                feature_lengths: Optional[torch.Tensor] = None) -> ObjectiveOutput:
        hidden = self.recognizer.encode(features)                  # (N, T, H)
        log_probs = F.log_softmax(self.recognizer.classifier(hidden), dim=-1)

        terms: Dict[str, torch.Tensor] = {}

        # --- L_ASR (CTC) ---------------------------------------------------
        # ``feature_lengths`` are the real per-sample valid RAW frame counts;
        # without them CTC would count padding frames as audio and mis-weight the
        # loss on a variable-length batch. They are converted to ENCODED lengths
        # (the recognizer subsamples), matching the log-prob time axis, and used
        # for both CTC and forced alignment. Default recovers the full T.
        n, t, _ = log_probs.shape
        if feature_lengths is None:
            input_lengths = torch.full((n,), t, dtype=torch.long, device=log_probs.device)
        else:
            feature_lengths = feature_lengths.to(dtype=torch.long, device=log_probs.device)
            input_lengths = self.recognizer.output_lengths(feature_lengths).clamp(max=t)
        terms["asr"] = self.recognizer.ctc(
            log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

        frame_labels, boundaries, valid = self._frame_targets(
            log_probs.detach(), targets, target_lengths, input_lengths)

        # --- L_boundary ----------------------------------------------------
        if self.boundary_head is not None and self.weights.boundary > 0:
            if bool(valid.any()):
                logits = self.boundary_head(hidden)[valid]
                terms["boundary"] = boundary_loss(logits, boundaries[valid])
            else:
                terms["boundary"] = log_probs.sum() * 0.0

        # --- L_Brier -------------------------------------------------------
        if self.weights.brier > 0:
            if bool(valid.any()):
                probs = log_probs[valid].exp().reshape(-1, log_probs.shape[-1])
                terms["brier"] = self.brier(probs, frame_labels[valid].reshape(-1))
            else:
                terms["brier"] = log_probs.sum() * 0.0

        # --- L_contrast ----------------------------------------------------
        if (sign_embeddings is not None and self.aligner is not None
                and self.weights.contrastive > 0):
            pooled = hidden.mean(dim=1)                            # (N, H)
            out = self.aligner(pooled, sign_embeddings)
            terms["contrastive"] = out["loss"]

        total = terms["asr"].clone()
        for name, weight in (("contrastive", self.weights.contrastive),
                             ("boundary", self.weights.boundary),
                             ("brier", self.weights.brier)):
            if name in terms:
                total = total + weight * terms[name]
        return ObjectiveOutput(total=total, terms={**terms, "total": total})


# ---------------------------------------------------------------------------
# Retrieval validation for the contrastive term
# ---------------------------------------------------------------------------
@torch.no_grad()
def speech_sign_retrieval(speech_emb: torch.Tensor, sign_emb: torch.Tensor,
                          ks: Sequence[int] = (1, 5)) -> Dict[int, float]:
    """Recall@k for matching pooled speech to its paired sign representation.

    Reported because the specification forbids treating a low InfoNCE loss as
    evidence of semantic alignment. Even this is weak evidence: batch-level
    retrieval can be solved by speaker, channel or duration cues. It is a
    necessary check, not a sufficient one -- downstream ablation decides.
    """
    if speech_emb.shape[0] != sign_emb.shape[0]:
        raise ValueError("paired embeddings must have equal batch size")
    a = F.normalize(speech_emb.double(), dim=-1)
    b = F.normalize(sign_emb.double(), dim=-1)
    sim = a @ b.t()
    n = sim.shape[0]
    ranking = sim.argsort(dim=1, descending=True)
    target = torch.arange(n, device=sim.device).unsqueeze(1)
    hit_pos = (ranking == target).float().argmax(dim=1)
    return {k: float((hit_pos < k).float().mean()) for k in ks}
