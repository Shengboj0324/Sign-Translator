"""Freeze the How2Sign leave-one-component-out fold plan from the certified mapping.

The mapping's bytes must match an explicitly supplied SHA-256 before use. Every
metadata row keeps its component, including rows whose media are missing or
structurally failed, so a later recovery inherits the right partition instead of
being reassigned. Technical exclusions are reported per component, never hidden.
This freezes grouping before windowing; it selects no QC threshold and approves
no evaluation.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signtranslator.data_engineering.fold_plan import (  # noqa: E402
    build_leave_one_component_out_plan, certify_fold_plan, records_from_rows,
)

TRAINABLE_STATUSES = ("valid", "quality_warning")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--mapping-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the plan and its summary")
    args = parser.parse_args(argv)

    payload = args.mapping.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.mapping_sha256:
        raise SystemExit(f"mapping hash {digest} does not match the expected value")
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    records = records_from_rows(rows, sample_field="sample_id", signer_field="signer_id",
                                source_field="video_id")
    plan = build_leave_one_component_out_plan(
        records, population_id="how2sign-green-screen-train-v1", mapping_sha256=digest)
    certification = certify_fold_plan(plan)

    args.output.mkdir(parents=False, exist_ok=False)
    plan_sha256 = plan.write(args.output / "fold-plan.json")
    data = plan.to_dict()
    status = {row["sample_id"]: row["audit_status"] for row in rows}
    per_component = collections.defaultdict(collections.Counter)
    for sample_id, component in data["sample_components"].items():
        per_component[component][status[sample_id]] += 1
    folds = []
    for fold in data["folds"]:
        trainable = {}
        for partition, components in (("train", fold["train_components"]),
                                      ("val", [fold["validation_component"]]),
                                      ("test", [fold["test_component"]])):
            trainable[partition] = sum(per_component[c][s] for c in components
                                       for s in TRAINABLE_STATUSES)
        folds.append(dict(fold_index=fold["fold_index"], test=fold["test_component"],
                          val=fold["validation_component"], trainable_rows=trainable))
    summary = dict(
        plan_sha256=plan_sha256, mapping_sha256=digest, certification=certification,
        components=[dict(component_id=c["component_id"], signers=c["signers"],
                         sources=len(c["sources"]), rows=c["rows"],
                         audit_status=dict(sorted(per_component[c["component_id"]].items())))
                    for c in data["components"]],
        folds=folds, sign_test_resolution=data["sign_test_resolution"],
        trainable_statuses=list(TRAINABLE_STATUSES),
        phase_exit_approved=False)
    (args.output / "fold-plan-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(plan_sha256=plan_sha256, components=certification["components"],
                          folds=certification["folds"],
                          quarantined=certification["quarantined"]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
