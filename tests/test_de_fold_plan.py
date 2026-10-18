"""Leave-one-component-out fold plans: leakage, quarantine and exact resolution."""
from fractions import Fraction
import itertools
import json
import math

import pytest

from signtranslator.data_engineering.fold_plan import (
    FoldPlan, GroupingRecord, build_leave_one_component_out_plan,
    certify_fold_plan, minimum_resolvable_units, minimum_sign_test_p,
    records_from_rows, resolution_table,
)

MAPPING = "f" * 64


def population():
    """Six components, including a transitive signer/source chain."""
    rows = []
    # Component A: signer s1 in v1, v2; signer s2 also in v2 -> one component.
    rows += [("a1", "s1", "v1"), ("a2", "s1", "v2"), ("a3", "s2", "v2"), ("a4", "s2", "v3")]
    rows += [("b1", "s3", "v4"), ("b2", "s3", "v5")]
    rows += [("c1", "s4", "v6")]
    rows += [("d1", "s5", "v7"), ("d2", "s5", "v7")]
    rows += [("e1", "s6", "v8")]
    rows += [("f1", "s7", "v9"), ("f2", "s8", "v9")]  # two signers, one source
    return [GroupingRecord(*row) for row in rows]


def plan_for(records=None):
    return build_leave_one_component_out_plan(
        population() if records is None else records,
        population_id="fixture", mapping_sha256=MAPPING)


def test_components_close_transitive_signer_and_source_leakage():
    plan = plan_for().to_dict()
    assert plan["component_count"] == 6
    signer_sets = sorted(tuple(c["signers"]) for c in plan["components"])
    assert ("s1", "s2") in signer_sets and ("s7", "s8") in signer_sets
    by_sample = plan["sample_components"]
    assert len({by_sample[s] for s in ("a1", "a2", "a3", "a4")}) == 1


def test_every_fold_partitions_without_any_signer_or_source_crossing():
    plan = plan_for()
    records = {r.sample_id: r for r in population()}
    for fold in range(6):
        partitions = {}
        for sample_id, record in records.items():
            partitions.setdefault(plan.partition(fold, sample_id), []).append(record)
        assert set(partitions) == {"train", "val", "test"}
        for field in ("signer_id_hash", "source_id"):
            owners = {}
            for name, members in partitions.items():
                for record in members:
                    assert owners.setdefault(getattr(record, field), name) == name


def test_each_component_is_tested_exactly_once_with_a_disjoint_validation():
    folds = plan_for().to_dict()["folds"]
    assert len({f["test_component"] for f in folds}) == 6
    assert all(f["test_component"] != f["validation_component"] for f in folds)
    assert all(f["validation_component"] not in f["train_components"] for f in folds)


def test_validation_rule_is_the_exact_integer_argmin_and_spares_a_dominant_component():
    # One dominant component, as in How2Sign, plus five small ones.
    records = [GroupingRecord(f"big{i}", "sbig", f"vbig{i % 7}") for i in range(80)]
    for index, size in enumerate((3, 4, 5, 6, 7)):
        records += [GroupingRecord(f"s{index}_{i}", f"s{index}", f"v{index}") for i in range(size)]
    plan = plan_for(records).to_dict()
    rows = {c["component_id"]: c["rows"] for c in plan["components"]}
    big = max(rows, key=rows.get)
    fraction = Fraction(plan["validation_fraction"])
    for fold in plan["folds"]:
        pool = sum(n for c, n in rows.items() if c != fold["test_component"])
        expected = min((c for c in rows if c != fold["test_component"]),
                       key=lambda c: (abs(rows[c] * fraction.denominator
                                          - fraction.numerator * pool), c))
        assert fold["validation_component"] == expected
        # The dominant component is never consumed as validation.
        assert fold["validation_component"] != big
    starved = [f for f in plan["folds"] if f["test_component"] != big
               and f["partition_rows"]["train"] < rows[big]]
    assert not starved


def test_validation_fraction_must_be_exact():
    with pytest.raises(ValueError, match="exact Fraction"):
        build_leave_one_component_out_plan(population(), population_id="x",
                                           mapping_sha256=MAPPING, validation_fraction=0.15)


def test_plan_is_independent_of_input_order_and_content_addressed():
    records = population()
    first = plan_for(records)
    for permutation in itertools.islice(itertools.permutations(records), 0, 40, 7):
        assert plan_for(list(permutation)).payload == first.payload
    assert len(first.sha256()) == 64


def test_unknown_identities_are_quarantined_not_assumed_unique():
    records = population() + [GroupingRecord("u1", "unknown", "v1"),
                               GroupingRecord("u2", "s9", " "),
                               GroupingRecord("u3", "s10", "NaN")]
    plan = plan_for(records)
    data = plan.to_dict()
    assert data["quarantined_sample_ids"] == ["u1", "u2", "u3"]
    assert data["planned_rows"] == len(population())
    with pytest.raises(ValueError, match="quarantined"):
        plan.partition(0, "u1")
    with pytest.raises(KeyError):
        plan.partition(0, "absent")


def test_exact_sign_test_resolution_matches_closed_form():
    for units in range(1, 12):
        assert minimum_sign_test_p(units) == Fraction(1, 2 ** (units - 1))
    alpha = Fraction(1, 20)
    for family in (1, 2, 3, 5):
        units = minimum_resolvable_units(alpha, family)
        assert units == math.ceil(1 + math.log2(family / alpha))
        assert minimum_sign_test_p(units) <= alpha / family < minimum_sign_test_p(units - 1)
    # The repository's recorded B68 numbers are exact consequences.
    assert minimum_sign_test_p(4) == Fraction(1, 8)
    assert minimum_sign_test_p(6) == Fraction(1, 32)
    assert minimum_resolvable_units(alpha, 2) == 7


def test_resolution_table_for_six_components_is_honest():
    table = {(row["alpha"], row["family_size"]): row for row in resolution_table(6)}
    assert table[("1/20", 1)]["resolvable"] is True
    assert table[("1/20", 2)]["resolvable"] is False
    assert table[("1/20", 2)]["minimum_units_required"] == 7
    assert table[("1/100", 1)]["resolvable"] is False


def test_written_plan_reloads_and_tampering_is_rejected(tmp_path):
    plan = plan_for()
    path = tmp_path / "plan.json"
    digest = plan.write(path)
    assert FoldPlan.load(path, expected_sha256=digest).payload == plan.payload
    with pytest.raises(FileExistsError):
        plan.write(path)
    data = json.loads(path.read_bytes())
    data["folds"][0]["test_component"] = data["folds"][1]["test_component"]
    forged = FoldPlan(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="partition every component|tested exactly once|exact rule"):
        certify_fold_plan(forged)
    data = json.loads(path.read_bytes())
    fold = data["folds"][0]
    swapped = fold["train_components"][0]
    fold["train_components"][0] = fold["validation_component"]
    fold["validation_component"] = swapped
    forged = FoldPlan(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="declared exact rule"):
        certify_fold_plan(forged)
    data = json.loads(path.read_bytes())
    data["sign_test_resolution"][0]["resolvable"] = not data["sign_test_resolution"][0]["resolvable"]
    forged = FoldPlan(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="exact arithmetic"):
        certify_fold_plan(forged)
    data = json.loads(path.read_bytes())
    data["components"][1]["signers"].append(data["components"][0]["signers"][0])
    forged = FoldPlan(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ValueError, match="appears in two components"):
        certify_fold_plan(forged)
    data = json.loads(path.read_bytes())
    data["phase_exit_approved"] = True
    with pytest.raises(ValueError, match="never approve"):
        certify_fold_plan(FoldPlan(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()))


def test_wrong_hash_and_noncanonical_bytes_are_rejected(tmp_path):
    plan = plan_for()
    path = tmp_path / "plan.json"
    plan.write(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        FoldPlan.load(path, expected_sha256="0" * 64)
    pretty = tmp_path / "pretty.json"
    payload = json.dumps(plan.to_dict(), indent=1).encode()
    pretty.write_bytes(payload + b"\n")
    import hashlib
    with pytest.raises(ValueError, match="not canonical"):
        FoldPlan.load(pretty, expected_sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize("records,message", [
    ([GroupingRecord("a", "s1", "v1"), GroupingRecord("b", "s2", "v2")], "at least 3 components"),
    ([GroupingRecord("a", "s1", "v1"), GroupingRecord("a", "s2", "v2")], "unique"),
    ([GroupingRecord("a", "unknown", "v1")], "no record has known"),
])
def test_invalid_populations_are_refused(records, message):
    with pytest.raises(ValueError, match=message):
        plan_for(records)


def test_mapping_rows_require_declared_fields():
    rows = [{"sample": "a", "signer": "s1", "video": "v1"}]
    assert records_from_rows(rows, sample_field="sample", signer_field="signer",
                             source_field="video") == [GroupingRecord("a", "s1", "v1")]
    with pytest.raises(ValueError, match="lacks declared field"):
        records_from_rows(rows, sample_field="sample", signer_field="signer_id",
                          source_field="video")
