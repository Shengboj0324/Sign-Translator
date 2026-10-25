from signtranslator.generation.envelope import apply, bands, declared


def baseline():
    return dict(groups=["a", "b", "c"],
                cross={"b_vs_a": dict(checks={"nod_rate": 3.0, "body_speed_ratio": 1.0}),
                       "c_vs_a": dict(checks={"nod_rate": 0.2, "body_speed_ratio": 1.1}),
                       "a_vs_b": dict(checks={"nod_rate": 1.0}), "c_vs_b": dict(checks={"nod_rate": 1.5}),
                       "a_vs_c": dict(checks={"nod_rate": 0.9}), "b_vs_c": dict(checks={"nod_rate": 1.1})},
                pooled_others={"a": dict(checks={"nod_rate": 4.0}), "b": dict(checks={"nod_rate": 1.0}),
                               "c": dict(checks={"nod_rate": 1.0})},
                test_cross={"a": dict(checks={"head_rotation_speed_ratio": 2.5})})


def test_bands_widen_never_narrow():
    b = bands(baseline())
    assert b["a"]["nod_rate"] == (0.2, 3.0)                 # widened both ways by the real other signers
    assert b["a"]["body_speed_ratio"] == declared("body_speed_ratio") == (.7, 1.3)   # real inside: declared band
    assert b["b"]["nod_rate"] == (.5, 2.)                   # real within the declared band: unchanged
    assert b["pool"]["nod_rate"] == (.5, 4.0)
    assert b["test"]["head_rotation_speed_ratio"] == (.7, 2.5)


def test_apply_rejudges_and_keeps_declared_without_band():
    checks = {"nod_rate": (2.5, False), "brow_raise_fraction": (2.5, True), "x_speed_ratio": (None, True)}
    out = apply(checks, {"nod_rate": (0.2, 3.0)})
    assert out["nod_rate"] == (2.5, True)
    assert out["brow_raise_fraction"] == (2.5, False)       # no envelope: declared marker band (0.5, 2)
    assert out["x_speed_ratio"] == (None, False)


def test_gate_row_rejudged_by_envelope():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from multi_signer_gate import enveloped
    row = dict(run="r", setting="s", frechet=1., checks={"nod_rate": (2.5, False), "body_speed_ratio": (1.0, True)},
               passed=1, total=2, all_pass=False, accuracy={})
    assert enveloped(row, None) is row
    out = enveloped(row, {"nod_rate": (0.2, 3.0)})
    assert out["passed"] == 2 and out["all_pass"]


def test_smoothness_checks_keep_their_own_upper_bound_under_the_envelope():
    assert declared("r7_head_translation") == (0., 1.5) and declared("r4_short_body_speed_ratio") == (.7, 1.3)
    out = apply({"r7_head_translation": (1.4, True), "r7_body": (1.6, False), "r4_short_body_speed_ratio": (1.2, True)}, {})
    assert out["r7_head_translation"][1] and not out["r7_body"][1] and out["r4_short_body_speed_ratio"][1]
