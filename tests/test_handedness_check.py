"""Handedness check: per-signer majorities, corpus fractions and the declared rule."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import handedness_check as hc  # noqa: E402


def _clip(corpus, unit, side, own=None, height=None):
    return dict(corpus=corpus, unit=unit, clip=f"{unit}-{side}", side=side, side_own_frames=own or side,
                side_height=height or side, shared_pairs=10)


def _corpus(name, right_units, left_units, undecided_units=0):
    out = []
    for k in range(right_units):
        out += [_clip(name, f"r{k}", "Right"), _clip(name, f"r{k}", "Right"), _clip(name, f"r{k}", "Left")]
    for k in range(left_units):
        out += [_clip(name, f"l{k}", "Left"), _clip(name, f"l{k}", None)]
    for k in range(undecided_units):
        out += [_clip(name, f"u{k}", "Left"), _clip(name, f"u{k}", "Right")]
    return out


def test_majorities_fractions_and_the_rule():
    results = _corpus("semlex-all", 1, 9, 2) + _corpus("how2sign-two-view-text-v1", 8, 2) + \
        _corpus("asllvd-text-v1", 4, 0)
    summary = hc.summarise(results)
    s = summary["semlex-all"]["path_shared_frames"]
    assert s["units"] == 12 and s["units_decided"] == 10 and s["right_fraction_units"] == .1
    assert s["clips_decided"] == 3 + 9 + 2 * 2 and summary["semlex-all"]["per_unit"]["u0"]["side"] == "undecided"
    v = hc.verdict(summary)
    assert v["semlex_right_fraction"] == .1 and v["how2sign_right_fraction"] == .8 and v["mirrored"] is True
    # any one condition failing: not mirrored
    assert hc.verdict(hc.summarise(_corpus("semlex-all", 3, 7) + _corpus("how2sign-two-view-text-v1", 8, 2) +
                                   _corpus("asllvd-text-v1", 4, 0)))["mirrored"] is False         # 0.3 >= 0.25
    assert hc.verdict(hc.summarise(_corpus("semlex-all", 1, 9) + _corpus("how2sign-two-view-text-v1", 6, 4) +
                                   _corpus("asllvd-text-v1", 4, 0)))["mirrored"] is False         # How2Sign 0.6
    assert hc.verdict(hc.summarise(_corpus("semlex-all", 1, 9)))["mirrored"] is None               # references missing


def test_sampling_is_capped_and_deterministic():
    rows = [("c", f"u{u}", f"clip{u}-{k}", "key", "sha") for u in range(10) for k in range(20)]
    a, b = hc.sample(rows, 5, 3), hc.sample(list(reversed(rows)), 5, 3)
    assert a == b and len(a) == 15 and len({r[1] for r in a}) == 3
