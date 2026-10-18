"""Frozen leave-one-component-out evaluation plans with exact resolution.

Why folds rather than one fixed split. Signer and source identities form a
bipartite graph whose edges are samples. Assigning whole connected components
to partitions is sufficient for leakage freedom: if a signer or a source were
shared by two partitions, the samples carrying it would be joined by a path, so
they would lie in one component, contradicting the assignment. When a
population has only a handful of components, a single train/validation/test
split leaves too few independent test units to resolve any confirmatory claim.
Rotating every component through the test role uses each one exactly once as
held-out evidence while keeping each fold leakage-free.

For ``k >= 3`` components, fold ``i`` tests component ``i``. Its validation
component is chosen from the remaining ``k - 1`` by a declared exact rule: with
non-test pool size ``P_i`` and declared rational validation fraction ``f``, pick
the ``j`` minimising ``|n_j - f * P_i|``, ties broken by component identity. The
objective is evaluated in integers as ``|n_j * den(f) - num(f) * P_i|``, so no
floating-point rounding can change a choice, and the stored plan can be
re-derived exactly. Every component is tested exactly once. A dominant
component is not forced into a validation role by position: rotating validation
``i -> i + 1`` would hand it the validation role in one fold and starve that
fold's training set for no statistical benefit.

Statistical resolution is exact and stated, not assumed. If the ``k`` held-out
components are the independent units of a paired two-sided sign test, the
smallest attainable p-value occurs when all ``k`` are non-tied and agree in
sign, giving ``p_min = 2 * 2^-k = 2^(1-k)``. A family of ``m`` primary endpoints
under Bonferroni at level ``alpha`` is resolvable only if
``2^(1-k) <= alpha / m``, i.e. ``k >= 1 + log2(m / alpha)``. Resolution is a
necessary condition for a confirmatory design, never a power calculation, and
whether components are genuinely independent remains a design assumption.

Identities that are missing, blank or declared unknown are quarantined instead
of being treated as unique, because an unknown signer could be any known one.
Nothing in this module selects quality thresholds, opens test data, or approves
an evaluation design.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .splitting import group_samples

FOLD_PLAN_SCHEMA_VERSION = 1
UNKNOWN_IDENTITIES = frozenset({"", "unknown", "none", "null", "nan", "?"})


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class GroupingRecord:
    """One sample's grouping identities; no personal identity is inferred."""

    sample_id: str
    signer_id_hash: str
    source_id: str


def _known(value) -> bool:
    return isinstance(value, str) and value.strip().lower() not in UNKNOWN_IDENTITIES


def minimum_sign_test_p(units: int) -> Fraction:
    """Smallest exact two-sided paired sign-test p-value for ``units`` pairs."""
    if type(units) is not int or units < 1:
        raise ValueError("unit count must be a positive integer")
    return Fraction(2, 2 ** units)


def minimum_resolvable_units(alpha: Fraction, family_size: int) -> int:
    """Least ``k`` with ``2^(1-k) <= alpha / family_size``, computed exactly."""
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1:
        raise ValueError("alpha must be an exact Fraction in (0, 1)")
    if type(family_size) is not int or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    threshold = alpha / family_size
    units = 1
    while minimum_sign_test_p(units) > threshold:
        units += 1
    return units


def resolution_table(units: int, *, alphas=(Fraction(1, 20), Fraction(1, 100)),
                     family_sizes=(1, 2, 3)) -> list[dict]:
    """Exact resolvability of each (alpha, family size) at ``units`` test units."""
    best = minimum_sign_test_p(units)
    rows = []
    for alpha in alphas:
        for family in family_sizes:
            threshold = alpha / family
            rows.append(dict(
                alpha=str(alpha), family_size=family,
                bonferroni_threshold=str(threshold),
                minimum_p=str(best), minimum_p_float=float(best),
                resolvable=best <= threshold,
                minimum_units_required=minimum_resolvable_units(alpha, family)))
    return rows


@dataclass(frozen=True)
class FoldPlan:
    """A frozen, content-addressed fold plan; immutable once written."""

    payload: bytes

    def to_dict(self) -> dict:
        return json.loads(self.payload)

    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def partition(self, fold_index: int, sample_id: str) -> str:
        """Partition of one sample in one fold; quarantined samples have none."""
        plan = self.to_dict()
        if type(fold_index) is not int or not 0 <= fold_index < len(plan["folds"]):
            raise ValueError("fold_index must select an existing fold")
        component = plan["sample_components"].get(sample_id)
        if component is None:
            if sample_id in set(plan["quarantined_sample_ids"]):
                raise ValueError(f"{sample_id} is quarantined and belongs to no partition")
            raise KeyError(f"{sample_id} is not part of the planned population")
        fold = plan["folds"][fold_index]
        if component == fold["test_component"]:
            return "test"
        if component == fold["validation_component"]:
            return "val"
        return "train"

    def write(self, path: Path) -> str:
        """Write once; refuse to overwrite an existing plan."""
        with Path(path).open("xb") as stream:
            stream.write(self.payload + b"\n")
        return self.sha256()

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str) -> "FoldPlan":
        data = Path(path).read_bytes()
        if not data.endswith(b"\n"):
            raise ValueError("fold plan must end with exactly one newline")
        plan = cls(data[:-1])
        if plan.sha256() != expected_sha256:
            raise ValueError("fold plan content hash mismatch")
        if _canonical(plan.to_dict()) != plan.payload:
            raise ValueError("fold plan bytes are not canonical")
        certify_fold_plan(plan)
        return plan


def _choose_validation(rows: Mapping[str, int], ids: Sequence[str], test: str,
                       fraction: Fraction) -> str:
    """Exact argmin of ``|n_j - f * P|`` over non-test components."""
    pool = sum(rows[c] for c in ids if c != test)
    return min((c for c in ids if c != test),
               key=lambda c: (abs(rows[c] * fraction.denominator - fraction.numerator * pool), c))


def build_leave_one_component_out_plan(
        records: Iterable[GroupingRecord], *, population_id: str,
        mapping_sha256: str, validation_fraction: Fraction = Fraction(3, 20)) -> FoldPlan:
    """Freeze components and folds before any windowing or augmentation.

    ``mapping_sha256`` binds the exact identity-mapping bytes the records came
    from. The plan stores each sample's component, not its windows; windows and
    derived views must inherit their sample's partition.
    """
    if not isinstance(population_id, str) or not population_id.strip():
        raise ValueError("population_id must be explicitly named")
    if not isinstance(validation_fraction, Fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be an exact Fraction in (0, 1)")
    if (not isinstance(mapping_sha256, str) or len(mapping_sha256) != 64
            or any(char not in "0123456789abcdef" for char in mapping_sha256)):
        raise ValueError("mapping_sha256 must be a lowercase SHA-256 digest")
    rows = list(records)
    if any(not isinstance(row, GroupingRecord) for row in rows):
        raise ValueError("records must be typed GroupingRecord values")
    identifiers = [row.sample_id for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("every record needs a nonempty sample_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample_id values must be unique")
    known = [row for row in rows if _known(row.signer_id_hash) and _known(row.source_id)]
    quarantined = sorted(row.sample_id for row in rows
                         if not (_known(row.signer_id_hash) and _known(row.source_id)))
    if not known:
        raise ValueError("no record has known signer and source identities")
    groups = group_samples(known)
    components = []
    for (signers, sources), indices in groups.items():
        members = sorted(known[index].sample_id for index in indices)
        membership = hashlib.sha256(_canonical(
            dict(signers=list(signers), sources=list(sources), samples=members))).hexdigest()
        components.append(dict(signers=list(signers), sources=list(sources),
                               rows=len(members), membership_sha256=membership,
                               _members=members))
    # Content-derived order: independent of input order, seeds or hashing salt.
    components.sort(key=lambda item: item["membership_sha256"])
    count = len(components)
    if count < 3:
        raise ValueError(
            f"leave-one-component-out needs at least 3 components; found {count}")
    sample_components = {}
    for index, component in enumerate(components):
        component["component_id"] = f"c{index:02d}"
        for sample in component.pop("_members"):
            sample_components[sample] = component["component_id"]
    ids = [component["component_id"] for component in components]
    rows_by = {component["component_id"]: component["rows"] for component in components}
    folds = []
    for index in range(count):
        validation = _choose_validation(rows_by, ids, ids[index], validation_fraction)
        folds.append(dict(fold_index=index, test_component=ids[index],
                          validation_component=validation,
                          train_components=[c for c in ids if c not in (ids[index], validation)]))
    for fold in folds:
        fold["partition_rows"] = dict(
            train=sum(rows_by[c] for c in fold["train_components"]),
            val=rows_by[fold["validation_component"]],
            test=rows_by[fold["test_component"]])
    plan = dict(
        schema_version=FOLD_PLAN_SCHEMA_VERSION,
        design="leave_one_component_out_exact_validation_rule",
        validation_fraction=str(validation_fraction),
        population_id=population_id, mapping_sha256=mapping_sha256,
        population_rows=len(rows), planned_rows=len(known),
        quarantined_sample_ids=quarantined, component_count=count,
        components=components, folds=folds,
        sample_components=dict(sorted(sample_components.items())),
        sign_test_resolution=resolution_table(count),
        fit_rules=[
            "Normalization, vocabularies, tokenization and QC cutoffs are fit on the "
            "train partition of each fold only.",
            "Model selection uses the validation component; the test component is "
            "evaluated once per fold after all choices for that fold are frozen.",
            "Windows, crops, augmentations and derived motion inherit their source "
            "sample's partition; none is assigned independently.",
        ],
        limitations=[
            "Components are the independence units only by design assumption.",
            "Sign-test resolution is necessary for a confirmatory claim, not a power analysis.",
            "Quarantined samples are excluded, not recovered or reassigned.",
            "This plan selects no quality threshold and approves no evaluation.",
        ],
        phase_exit_approved=False)
    result = FoldPlan(_canonical(plan))
    certify_fold_plan(result)
    return result


def certify_fold_plan(plan: FoldPlan) -> dict:
    """Recheck every structural guarantee from the stored plan alone."""
    data = plan.to_dict()
    if data.get("schema_version") != FOLD_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported fold plan schema")
    if data.get("phase_exit_approved") is not False:
        raise ValueError("a fold plan can never approve a phase")
    components = data["components"]
    ids = [component["component_id"] for component in components]
    if len(set(ids)) != len(ids) or len(ids) != data["component_count"] or len(ids) < 3:
        raise ValueError("component identities are invalid")
    for field in ("signers", "sources"):
        owner: dict[str, str] = {}
        for component in components:
            for identity in component[field]:
                if owner.setdefault(identity, component["component_id"]) != component["component_id"]:
                    raise ValueError(f"{field[:-1]} {identity} appears in two components")
    rows = {component["component_id"]: component["rows"] for component in components}
    assignment = data["sample_components"]
    if (sorted(set(assignment.values())) != sorted(ids)
            or any(sum(1 for value in assignment.values() if value == cid) != rows[cid]
                   for cid in ids)):
        raise ValueError("sample-to-component assignment disagrees with component sizes")
    if set(assignment) & set(data["quarantined_sample_ids"]):
        raise ValueError("a quarantined sample cannot also be assigned")
    if len(assignment) + len(data["quarantined_sample_ids"]) != data["population_rows"]:
        raise ValueError("planned plus quarantined rows must equal the population")
    tested = []
    fraction = Fraction(data["validation_fraction"])
    rows_by = {component["component_id"]: component["rows"] for component in components}
    for index, fold in enumerate(data["folds"]):
        if fold["fold_index"] != index:
            raise ValueError("folds must be indexed contiguously")
        roles = [fold["test_component"], fold["validation_component"], *fold["train_components"]]
        if sorted(roles) != sorted(ids) or len(set(roles)) != len(roles):
            raise ValueError(f"fold {index} must partition every component exactly once")
        tested.append(fold["test_component"])
        if fold["validation_component"] != _choose_validation(
                rows_by, ids, fold["test_component"], fraction):
            raise ValueError(f"fold {index} validation disagrees with the declared exact rule")
    if sorted(tested) != sorted(ids):
        raise ValueError("every component must be tested exactly once")
    expected = resolution_table(len(ids))
    if data["sign_test_resolution"] != expected:
        raise ValueError("recorded statistical resolution disagrees with exact arithmetic")
    return dict(certified=True, components=len(ids), folds=len(data["folds"]),
                quarantined=len(data["quarantined_sample_ids"]))


def summarize_plan(plan: FoldPlan) -> Mapping[str, object]:
    data = plan.to_dict()
    return dict(
        plan_sha256=plan.sha256(), component_count=data["component_count"],
        planned_rows=data["planned_rows"], quarantined=len(data["quarantined_sample_ids"]),
        component_rows=[(c["component_id"], c["rows"], c["signers"]) for c in data["components"]],
        resolvable=[(r["alpha"], r["family_size"], r["resolvable"], r["minimum_units_required"])
                    for r in data["sign_test_resolution"]],
        log2_units_note=f"k >= 1 + log2(m/alpha); for alpha=1/20, m=1: "
                        f"{1 + math.log2(20):.4f}")


def records_from_rows(rows: Sequence[Mapping[str, str]], *, sample_field: str,
                      signer_field: str, source_field: str) -> list[GroupingRecord]:
    """Build typed records from a parsed mapping, naming every field explicitly."""
    result = []
    for row in rows:
        try:
            result.append(GroupingRecord(row[sample_field], row[signer_field], row[source_field]))
        except KeyError as error:
            raise ValueError(f"mapping row lacks declared field {error}") from error
    return result
