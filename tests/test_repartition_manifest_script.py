"""Multi-signer re-partition: partitions follow components, held-out stays, unknown samples are refused."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repartition_manifest import repartition  # noqa: E402

PLAN = dict(sample_components={"a": "c00", "b": "c01", "c": "c02", "d": "c03"})


def test_partitions_follow_signer_components():
    m = dict(records=[dict(sign=s, partition="train") for s in "abcd"] + [dict(sign="h", partition="heldout_official")])
    out = {r["sign"]: r for r in repartition(m, PLAN, val={"c01", "c03"}, test={"c00"})}
    assert [out[s]["partition"] for s in "abcd"] == ["test", "val", "train", "val"]
    assert out["b"]["signer_component"] == "c01" and out["h"]["partition"] == "heldout_official"


def test_a_sample_outside_the_plan_is_refused():
    with pytest.raises(ValueError):
        repartition(dict(records=[dict(sign="z", partition="train")]), PLAN, val={"c01"}, test={"c00"})
