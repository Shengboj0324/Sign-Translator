"""Continuous sign-language recognition (sign -> gloss) with CTC.

This is the *recognition* direction that makes the system bidirectional. The
ST-GCN encoder produces a per-frame feature sequence; a linear head emits
per-frame class log-probabilities over the gloss vocabulary plus a blank symbol,
and Connectionist Temporal Classification (Graves et al., 2006) aligns the
unsegmented frame sequence to the (shorter) gloss label sequence without frame-
level annotation -- the standard formulation for continuous sign recognition.

Convention: class index ``0`` is the CTC blank; gloss ids occupy ``1..V``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import STGCNEncoder


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """Best-path (greedy) CTC decoding.

    Collapses consecutive duplicate labels, then removes blanks.

    Args:
        log_probs: ``(N, T, C)`` per-frame log-probabilities.
        blank: blank class index.

    Returns:
        A list of decoded label sequences (one per batch element).
    """
    best = log_probs.argmax(dim=-1)  # (N, T)
    results: List[List[int]] = []
    for seq in best.tolist():
        out: List[int] = []
        prev = blank
        for s in seq:
            if s != prev and s != blank:
                out.append(s)
            prev = s
        results.append(out)
    return results


class SignRecognizer(nn.Module):
    """ST-GCN encoder + CTC head mapping a pose clip to a gloss sequence."""

    def __init__(self, encoder: STGCNEncoder, num_glosses: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_glosses = num_glosses
        self.num_classes = num_glosses + 1  # +1 for blank (index 0)
        self.classifier = nn.Linear(encoder.out_dim, self.num_classes)
        # blank=0 convention; zero_infinity guards degenerate T < target cases.
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """pose (N, C, T, V) -> log-probs (N, T, num_classes)."""
        feats = self.encoder(pose, return_sequence=True)  # (N, T, D)
        logits = self.classifier(feats)
        return F.log_softmax(logits, dim=-1)

    def loss(self, pose: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor,
             input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """CTC loss.

        Args:
            pose: ``(N, C, T, V)``.
            targets: concatenated or padded gloss ids in ``1..V`` (no blanks).
            target_lengths: ``(N,)`` true length of each target.
            input_lengths: ``(N,)`` valid frame counts (defaults to full T).
        """
        log_probs = self.forward(pose)          # (N, T, C)
        n, t, _ = log_probs.shape
        if input_lengths is None:
            input_lengths = torch.full((n,), t, dtype=torch.long,
                                       device=log_probs.device)
        # nn.CTCLoss expects (T, N, C).
        log_probs_tnc = log_probs.permute(1, 0, 2)
        return self.ctc(log_probs_tnc, targets, input_lengths, target_lengths)

    @torch.no_grad()
    def decode(self, pose: torch.Tensor) -> List[List[int]]:
        self.eval()
        return ctc_greedy_decode(self.forward(pose))


def word_error_rate(hypotheses: List[List[int]],
                    references: List[List[int]]) -> float:
    """Mean word (gloss) error rate = Levenshtein distance / reference length."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must align")
    total_dist, total_len = 0, 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += _levenshtein(hyp, ref)
        total_len += max(len(ref), 1)
    return total_dist / max(total_len, 1)


def _levenshtein(a: List[int], b: List[int]) -> int:
    """Classic edit distance (insertions + deletions + substitutions)."""
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=np.int64)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[m, n])
