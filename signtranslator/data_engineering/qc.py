"""Per-tier inter-annotator agreement + QC stratified sampling (Doc-10 §5).

Agreement is reported PER annotation tier (gloss, non-manual, discourse), never as
a single corpus-wide number — a corpus average can hide a tier where annotators
disagree. QC sampling is stratified so every populated stratum is represented.
Reuses the audited Doc-03 `cohens_kappa`.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from ..grammar.signbleu import cohens_kappa


def per_tier_kappa(
    tier_ratings: Dict[str, Tuple[Sequence[int], Sequence[int]]]
) -> Dict[str, float]:
    """Cohen's kappa computed independently for each annotation tier."""
    return {tier: cohens_kappa(a, b) for tier, (a, b) in tier_ratings.items()}


def pooled_kappa(
    tier_ratings: Dict[str, Tuple[Sequence[int], Sequence[int]]]
) -> float:
    """The (discouraged) corpus-wide kappa: pool every tier's items together.

    Provided ONLY to demonstrate that pooling can mask a low-agreement tier;
    the pipeline reports `per_tier_kappa`, not this.
    """
    a_all: List[int] = []
    b_all: List[int] = []
    # Namespace each tier's labels so identical codes across tiers are distinct.
    for i, (a, b) in enumerate(tier_ratings.values()):
        a_all.extend(x + 1000 * i for x in a)
        b_all.extend(x + 1000 * i for x in b)
    return cohens_kappa(a_all, b_all)


def weakest_tier(per_tier: Dict[str, float]) -> Tuple[str, float]:
    """The tier with the lowest agreement (the one a corpus average would hide)."""
    tier = min(per_tier, key=per_tier.get)
    return tier, per_tier[tier]


# ---------------------------------------------------------------------------
# stratified QC sampling
# ---------------------------------------------------------------------------
def stratify(items: Sequence, key) -> Dict:
    """Group item indices by a stratum key function (signer/skin-tone/…)."""
    strata: Dict = {}
    for idx, it in enumerate(items):
        strata.setdefault(key(it), []).append(idx)
    return strata


def stratified_qc_sample(items: Sequence, key, k_per_stratum: int,
                         seed: int = 0) -> List[int]:
    """Sample up to ``k_per_stratum`` indices from EACH populated stratum.

    Guarantees coverage: every stratum with >=1 item contributes >=1 sample
    (for k_per_stratum >= 1). Deterministic given ``seed``.
    """
    if k_per_stratum < 1:
        raise ValueError("k_per_stratum must be >= 1")
    rng = random.Random(seed)
    chosen: List[int] = []
    for _, idxs in sorted(stratify(items, key).items(), key=lambda kv: str(kv[0])):
        pool = list(idxs)
        rng.shuffle(pool)
        chosen.extend(pool[:k_per_stratum])
    return sorted(chosen)
