"""Multi-signer gate: a candidate must pass on every validation signer."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multi_signer_gate as gate  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402


def result(head_speed, frechet, rh=20.):
    ch = lambda s: dict(speed_ratio=s, variation_ratio=1., geodesic_deg=rh)  # noqa: E731
    return dict(frechet=frechet, body=ch(1.), left_hand=ch(1.), right_hand=ch(1.), head_rotation=ch(head_speed),
                grammar=dict(distributions={m: dict(mean_ratio=1.) for m in gate_markers()}))


def gate_markers():
    from exit_gate import MARKER_SCORES
    return MARKER_SCORES


def test_a_candidate_must_pass_on_every_signer(tmp_path):
    store = DirectoryStore(tmp_path)
    # run A: best Fréchet but over-active head on signer c01; run B: passes on both
    for run, results in (("A", {"c01": result(1.8, 200.), "c03": result(1.0, 210.)}),
                         ("B", {"c01": result(1.1, 260.), "c03": result(1.0, 250.)})):
        for g, r in results.items():
            store.put(f"d/evaluations/{run}-{g}.json", json.dumps(dict(results={"g1": r})).encode())
    rows = gate.candidates(store, "d", ["A", "B"], ["c01", "c03"])
    a = next(r for r in rows if r["run"] == "A")
    assert not a["all_pass"] and a["failing"] == ["c01:head_rotation_speed_ratio"]
    assert gate.select(rows)["run"] == "B"
    assert gate.main(["--store", f"dir:{tmp_path}", "--derived", "d", "--runs", "A", "B", "--groups", "c01,c03",
                      "--output-key", "d/gate.json"]) == 0
    assert json.loads(store.get("d/gate.json"))["selected"]["run"] == "B"
