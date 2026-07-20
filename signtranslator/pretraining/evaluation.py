"""Evidence battery — loss is not usefulness (Doc-11 §6).

Linear probes for linguistic attributes, low-resource scaling curves, cross-signer
retrieval, signer/background leakage tests, and the loss-vs-usefulness dissociation:
two feature sets with the SAME reconstruction loss but DIFFERENT probe accuracy.
Reuses the Doc-04 `LinearProbe`. A lower pretraining loss alone is not evidence.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..pose.leakage import LinearProbe


def chance_accuracy(labels: Sequence[int]) -> float:
    """Majority-class baseline accuracy."""
    vals, counts = np.unique(np.asarray(labels), return_counts=True)
    return float(counts.max() / counts.sum())


def linear_probe_accuracy(X_tr: torch.Tensor, y_tr: torch.Tensor,
                          X_te: torch.Tensor, y_te: torch.Tensor,
                          num_classes: int, l2: float = 1e-3) -> float:
    """Fit a ridge probe to one-hot labels; report argmax test accuracy."""
    Ytr = F.one_hot(y_tr, num_classes).to(X_tr.dtype)
    probe = LinearProbe(l2=l2).fit(X_tr, Ytr)
    pred = probe.predict(X_te).argmax(dim=-1)
    return float((pred == y_te).float().mean())


def low_resource_scaling_curve(X_tr: torch.Tensor, y_tr: torch.Tensor,
                               X_te: torch.Tensor, y_te: torch.Tensor,
                               num_classes: int,
                               sizes: Sequence[int]) -> List[Tuple[int, float]]:
    """Probe accuracy vs number of labelled training examples."""
    out: List[Tuple[int, float]] = []
    for s in sizes:
        if s < 1 or s > X_tr.shape[0]:
            raise ValueError("size out of range")
        acc = linear_probe_accuracy(X_tr[:s], y_tr[:s], X_te, y_te, num_classes)
        out.append((s, acc))
    return out


def cross_signer_retrieval_recall(embeddings: torch.Tensor,
                                  content_labels: Sequence[int],
                                  signer_ids: Sequence[int], k: int = 1) -> float:
    """Recall@k retrieving same-CONTENT clips from DIFFERENT signers.

    For each query, candidates are restricted to other signers; a hit means a
    top-k candidate shares the query's content label. Tests whether the
    representation encodes content that generalises across signers.
    """
    z = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = z @ z.t()
    content = np.asarray(content_labels)
    signer = np.asarray(signer_ids)
    n = z.shape[0]
    hits = 0
    valid = 0
    for i in range(n):
        cand = np.where(signer != signer[i])[0]
        if cand.size == 0:
            continue
        valid += 1
        order = cand[torch.argsort(sim[i, cand], descending=True).numpy()]
        topk = order[:k]
        if np.any(content[topk] == content[i]):
            hits += 1
    return hits / valid if valid else 0.0


def signer_leakage_accuracy(features: torch.Tensor, signer_ids: Sequence[int],
                            num_signers: int, split: float = 0.5) -> float:
    """Accuracy of a signer classifier probed from frozen features.

    HIGH accuracy (>> chance) means the representation leaks signer identity — the
    document's leakage test. Low is the goal.
    """
    y = torch.as_tensor(list(signer_ids), dtype=torch.long)
    n = features.shape[0]
    cut = int(n * split)
    return linear_probe_accuracy(features[:cut], y[:cut], features[cut:], y[cut:],
                                 num_signers)


def is_leaky(probe_acc: float, chance: float, margin: float = 0.15) -> bool:
    """Leakage flagged when the probe beats chance by more than ``margin``."""
    return probe_acc > chance + margin


# ---------------------------------------------------------------------------
# loss-vs-usefulness dissociation (innovation)
# ---------------------------------------------------------------------------
def loss_usefulness_dissociation(n: int = 200, num_classes: int = 4,
                                 dim: int = 6, seed: int = 0) -> Dict[str, float]:
    """Two feature sets with EQUAL reconstruction loss but UNEQUAL probe accuracy.

    Both feature sets share a block that perfectly reconstructs the input `x` (so a
    reconstruction decoder reading that block has identical loss on both). They
    differ only in a second block: set A encodes the linguistic label (linearly
    decodable), set B encodes noise. A linear probe recovers the label from A but
    not from B — so equal reconstruction loss does NOT imply equal usefulness.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)                     # the input to reconstruct
    labels = torch.randint(0, num_classes, (n,), generator=g)
    label_block = F.one_hot(labels, num_classes).float()     # decodable
    noise_block = torch.randn(n, num_classes, generator=g)   # not decodable

    feat_A = torch.cat([x, label_block], dim=1)
    feat_B = torch.cat([x, noise_block], dim=1)

    # Reconstruction decoder reads ONLY the shared x-block (first `dim` columns):
    # identical reconstruction of x from both feature sets => identical loss.
    recon_A = F.mse_loss(feat_A[:, :dim], x)
    recon_B = F.mse_loss(feat_B[:, :dim], x)

    cut = n // 2
    acc_A = linear_probe_accuracy(feat_A[:cut], labels[:cut], feat_A[cut:],
                                  labels[cut:], num_classes)
    acc_B = linear_probe_accuracy(feat_B[:cut], labels[:cut], feat_B[cut:],
                                  labels[cut:], num_classes)
    return {
        "recon_loss_A": float(recon_A), "recon_loss_B": float(recon_B),
        "probe_acc_A": acc_A, "probe_acc_B": acc_B,
        "chance": chance_accuracy(labels.tolist()),
    }
