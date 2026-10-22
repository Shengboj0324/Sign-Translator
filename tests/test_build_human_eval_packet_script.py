"""Review packet builder: system specs, display-only gap holding, the test-partition firewall."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_human_eval_packet import choose_utterances, held_for_display, main, parse_system  # noqa: E402


def test_system_specs():
    assert parse_system("ref=reference") == dict(label="ref", kind="reference")
    s = parse_system("gen=plan:training-runs/x/step-1.pt@" + "a" * 64)
    assert s == dict(label="gen", kind="plan", key="training-runs/x/step-1.pt", manifest_sha256="a" * 64)
    for bad in ("ref", "g=plan:key", "g=other:k@" + "a" * 64, "=reference"):
        with pytest.raises(ValueError):
            parse_system(bad)


@dataclass(frozen=True)
class C:
    values: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class S:
    channels: dict


def test_gaps_are_held_from_the_nearest_valid_frame_for_display():
    v = np.arange(5.)[:, None, None] * np.ones((5, 2, 1))
    ok = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], bool)
    shown = held_for_display(S(dict(face=C(v, ok))))
    np.testing.assert_array_equal(shown.channels["face"].values[:, 0, 0], [1, 1, 1, 3, 3])
    assert shown.channels["face"].valid[:, 0].all()
    np.testing.assert_array_equal(shown.channels["face"].values[:, 1, 0], v[:, 1, 0])   # never valid: unchanged
    assert not shown.channels["face"].valid[:, 1].any()


def test_utterances_are_distinct_usable_sentences():
    import random
    rec = lambda i, e, p="val", s=3.: dict(sign=f"s{i}", english=e, partition=p, plan=dict(markers=[]), text="X",
                                           continuous=dict(seconds=s))
    records = [rec(0, "A b"), rec(1, "a B"), rec(2, "c"), rec(3, "d", "test"), rec(4, "e", s=.5), rec(5, "f")]
    chosen = choose_utterances(records, "val", 3, 1., random.Random(0))
    assert len({r["english"].lower() for r in chosen}) == 3 and all(r["partition"] == "val" for r in chosen)
    with pytest.raises(SystemExit):
        choose_utterances(records, "val", 4, 1., random.Random(0))


def test_test_partition_needs_the_final_study_flag(tmp_path):
    with pytest.raises(SystemExit, match="final study"):
        main(["--manifest-key", "k", "--manifest-sha256", "0" * 64, "--system", "r=reference", "--partition", "test",
              "--output-dir", str(tmp_path)])
