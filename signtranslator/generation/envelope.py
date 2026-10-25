"""Real cross-signer envelopes for the exit gate's checks (round 3b gate, docs/NEXT_PHASE_PLAN.md).

Real How2Sign signers judged by one another fail the gate's head and face checks by 2-21x
(``scripts/cross_signer_baseline.py``). The checks measure signer identity there. For a judging signer Y, the band of
check c is therefore the declared band widened to the range of real other signers X judged by Y:
``[min(lo, min_X r_c(X->Y)), max(hi, max_X r_c(X->Y))]``. Generated motion passes c against Y when it is at least as
close to Y as another real signer is. The band is never narrower than the declared one.

``bands(baseline)`` returns ``{judge: {check: (lo, hi)}}`` for:
- every validation group (``cross``: the other groups judged by it);
- ``pool`` (``pooled_others``: each group judged by the others pooled);
- ``test`` when present (``test_cross``: each validation group judged by the test signer).
"""
from __future__ import annotations

ARTICULATION = (.7, 1.3)          # exit_gate's declared bands, unchanged
MARKERS = (.5, 2.)
MARKER_SCORES = ("headshake_rate", "nod_rate", "brow_raise_fraction", "brow_furrow_fraction")


SMOOTHNESS = (0., 1.5)            # R7 (round 3c): >= 6 Hz amplitude at most 1.5x real


def declared(check: str) -> tuple[float, float]:
    if check.startswith("r7_"):
        return SMOOTHNESS
    return MARKERS if check in MARKER_SCORES else ARTICULATION


def _widen(values_by_check: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    out = {}
    for check, values in values_by_check.items():
        lo, hi = declared(check)
        out[check] = (min([lo] + values), max([hi] + values))
    return out


def bands(baseline: dict) -> dict[str, dict[str, tuple[float, float]]]:
    groups = baseline["groups"]
    judges: dict[str, dict[str, list[float]]] = {}
    for y in groups:
        for x in groups:
            if x != y:
                for check, value in baseline["cross"][f"{x}_vs_{y}"]["checks"].items():
                    judges.setdefault(y, {}).setdefault(check, []).append(float(value))
    for x in groups:
        for check, value in baseline["pooled_others"][x]["checks"].items():
            judges.setdefault("pool", {}).setdefault(check, []).append(float(value))
    for x, row in (baseline.get("test_cross") or {}).items():
        for check, value in row["checks"].items():
            judges.setdefault("test", {}).setdefault(check, []).append(float(value))
    return {judge: _widen(values) for judge, values in judges.items()}


def apply(checks: dict, band: dict[str, tuple[float, float]]) -> dict:
    """``{check: (value, ok)}`` re-judged against the envelope (a check without an envelope keeps its declared band)."""
    out = {}
    for check, (value, _) in checks.items():
        lo, hi = band.get(check, declared(check))
        out[check] = (value, value is not None and lo <= value <= hi)
    return out


__all__ = ["bands", "apply", "declared"]
