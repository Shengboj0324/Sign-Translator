"""The generator round's exit gate: checks, the selection rule and its tie-break."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("exit_gate", Path(__file__).resolve().parents[1] / "scripts/exit_gate.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def result(frechet, speed=1., variation=1., headshake=1., hand_error=20.):
    channel = lambda: dict(geodesic_deg=hand_error, speed_ratio=speed, variation_ratio=variation)
    return dict(frechet=frechet, body=channel(), left_hand=channel(), right_hand=channel(),
                grammar=dict(distributions=dict(headshake_rate=dict(mean_ratio=headshake),
                                                nod_rate=dict(mean_ratio=1.))))


def test_checks_use_the_declared_bounds():
    row = gate.score_setting("r", "g1", result(100, speed=.69, headshake=2.01))
    assert row["checks"]["body_speed_ratio"] == (.69, False) and row["checks"]["headshake_rate"] == (2.01, False)
    assert row["checks"]["body_variation_ratio"][1] and row["total"] == 8 and not row["all_pass"]
    edge = gate.score_setting("r", "g1", result(100, speed=.7, variation=1.3, headshake=.5))
    assert edge["all_pass"]


def test_selection_prefers_passing_then_lowest_frechet_with_the_tie_break():
    rows = [gate.score_setting("fast", "g", result(150, speed=.5)),          # best Fréchet but fails articulation
            gate.score_setting("a", "g", result(200, hand_error=30.)),
            gate.score_setting("b", "g", result(215, hand_error=25.)),        # within 10%: lower hand error wins
            gate.score_setting("c", "g", result(400))]
    assert gate.select(rows)["run"] == "b"
    failing = [gate.score_setting("x", "g", result(300, speed=.5)), gate.score_setting("y", "g", result(250, speed=.5,
                                                                                                          headshake=3.))]
    assert gate.select(failing)["run"] == "x"                     # most checks passed when none passes them all
