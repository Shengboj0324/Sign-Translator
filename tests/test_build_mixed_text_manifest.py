"""Mixed text -> motion manifests: records kept and tagged by corpus, collisions refused, licences carried."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_mixed_text_manifest import mix, parse_input  # noqa: E402

SHA = "a" * 64


def manifest(corpus, signs, licence=None):
    return dict(corpus_id=corpus, license=licence,
                records=[dict(sign=s, text=f"text {s}", partition=p,
                              continuous=dict(state_key=f"{s}.npz", state_sha256=SHA, available_channels=["body"]))
                         for s, p in signs])


def test_records_are_kept_tagged_and_partitioned_as_given():
    m = mix([("1" * 64, manifest("en", [("a", "train"), ("b", "val")])),
             ("2" * 64, manifest("plan", [("c", "train"), ("d", "test")], "research only"))], corpus_id="mixed")
    assert {r["sign"]: (r["corpus"], r["partition"]) for r in m["records"]} == {
        "a": ("en", "train"), "b": ("en", "val"), "c": ("plan", "train"), "d": ("plan", "test")}
    assert m["inputs"] == {"en": "1" * 64, "plan": "2" * 64}
    assert m["license"] == "plan: research only" and m["split"]["counts"]["plan/test"] == 1


def test_collisions_are_refused():
    with pytest.raises(ValueError, match="occurs in two inputs"):
        mix([("1" * 64, manifest("en", [("a", "train")])), ("2" * 64, manifest("plan", [("a", "train")]))],
            corpus_id="m")
    with pytest.raises(ValueError, match="share corpus_id"):
        mix([("1" * 64, manifest("en", [("a", "train")])), ("2" * 64, manifest("en", [("b", "train")]))],
            corpus_id="m")
    with pytest.raises(ValueError):
        parse_input("key@short")
