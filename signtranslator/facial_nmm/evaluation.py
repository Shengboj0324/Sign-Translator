"""Non-manual evaluation batteries (docs/FACIAL_NMM.md §8).

* **Minimal-pair comprehension** — identical manual motion, changed non-manual
  markers must yield a different meaning; a reader of ONLY the non-manual channel
  distinguishes the pair.
* **Scope boundary error** — ``|t_s−t̂_s| + |t_e−t̂_e|`` for a marker's scope.
* **Gaze/locus agreement** — gaze direction vs the direction to the referenced
  locus (cosine similarity).
* **Head-manual synchronisation** — signed temporal offset of the head-movement
  onset from the manual-event onset (~0 = synchronised).
* **Channel ablation** — comprehension drop when a channel is removed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch

from .channels import Marker


# ---------------------------------------------------------------------------
# minimal-pair comprehension
# ---------------------------------------------------------------------------
def minimal_pair_distinguishes(marker_a: Marker, marker_b: Marker,
                               predict: Callable[[Marker], int]) -> bool:
    """True iff a non-manual reader assigns different meanings to the two markers.

    ``predict`` maps a non-manual marker (over identical manual motion) to a
    grammatical label. Two grammatically-distinct markers MUST be distinguished.
    """
    return predict(marker_a) != predict(marker_b)


def minimal_pair_accuracy(pairs: Sequence[Tuple[Marker, Marker]],
                          predict: Callable[[Marker], int]) -> float:
    """Fraction of grammatically-distinct minimal pairs the reader distinguishes."""
    if not pairs:
        return 0.0
    correct = sum(1 for a, b in pairs if minimal_pair_distinguishes(a, b, predict))
    return correct / len(pairs)


# ---------------------------------------------------------------------------
# scope boundary error
# ---------------------------------------------------------------------------
def scope_boundary_error(pred_ts: float, pred_te: float,
                         true_ts: float, true_te: float) -> float:
    """|t_s − t̂_s| + |t_e − t̂_e| for a marker's scope (0 = exact)."""
    return abs(pred_ts - true_ts) + abs(pred_te - true_te)


# ---------------------------------------------------------------------------
# gaze / locus agreement
# ---------------------------------------------------------------------------
def gaze_locus_agreement(gaze_dir: torch.Tensor, to_locus: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between the gaze direction and the direction to the locus.

    1 = gaze points exactly at the referenced locus; -1 = away. ``gaze_dir`` /
    ``to_locus`` (..., 3).
    """
    g = gaze_dir / gaze_dir.norm(dim=-1, keepdim=True).clamp_min(eps)
    d = to_locus / to_locus.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (g * d).sum(-1)


# ---------------------------------------------------------------------------
# head-manual synchronisation
# ---------------------------------------------------------------------------
def head_manual_offset(head_onset: float, manual_onset: float) -> float:
    """Signed temporal offset (seconds): head onset minus manual onset. ~0 = synced."""
    return head_onset - manual_onset


def synchronisation_rate(offsets: Sequence[float], tolerance: float) -> float:
    """Fraction of head/manual pairs synchronised within ``tolerance`` seconds."""
    if not offsets:
        return 0.0
    return sum(1 for o in offsets if abs(o) <= tolerance) / len(offsets)


# ---------------------------------------------------------------------------
# channel ablation
# ---------------------------------------------------------------------------
def channel_ablation_drop(full_score: float,
                          ablated_scores: Dict[str, float]) -> Dict[str, float]:
    """Comprehension drop ``full − ablated`` per removed channel (larger = channel
    carries more of the linguistic signal)."""
    return {name: full_score - s for name, s in ablated_scores.items()}
