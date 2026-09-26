"""Capture W0 Day-1 evidence without modifying data or repairing model code.

Run with the repository's Python 3.12 environment and a NEW output directory.
A successful capture is not mathematical, linguistic, or deployment acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def file_record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a non-symlink regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError(f"file changed during hashing: {path}")
    return {"sha256": digest.hexdigest(), "size": after.st_size}


def snapshot(repo: Path) -> dict:
    paths = [*repo.joinpath("signtranslator").rglob("*.py"),
             *repo.joinpath("tests").rglob("*.py"),
             *repo.joinpath(".github", "workflows").glob("*.yml")]
    paths += [repo / name for name in (
        "conftest.py", "pyproject.toml", "requirements.lock", "requirements.txt",
        "requirements-foundation.txt", "LICENSE", "scripts/w0_day1_baseline.py")]
    return {path.relative_to(repo).as_posix(): file_record(path)
            for path in sorted(set(paths))}


def run_command(argv: list[str], repo: Path, log: Path) -> dict:
    start = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(argv, cwd=repo, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    return {"argv": argv, "returncode": process.returncode,
            "seconds": round(time.monotonic() - start, 3),
            "log": log.name, **file_record(log)}


def verify_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict:
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "status": "missing", "matches": False,
                "expected_sha256": expected_sha256, "expected_size": expected_size}
    record = file_record(path)
    return {"path": str(path), "status": "present", **record,
            "expected_sha256": expected_sha256,
            "matches": record["sha256"] == expected_sha256
            and (expected_size is None or record["size"] == expected_size)}


def local_evidence(data: Path) -> dict:
    """Verify existing small manifests and explicitly named payloads, read-only."""
    signer_dir = data / "how2sign_audit/signer-evidence-v2"
    certificate = json.loads((signer_dir / "certificate.json").read_text())
    audit_dir = data / "how2sign_audit/v1"
    checks = [
        verify_file(signer_dir / "how2sign_train_signers.csv",
                    certificate["mapping"]["sha256"]),
        verify_file(audit_dir / "audit_manifest.json",
                    certificate["source"]["audit_manifest_sha256"]),
        verify_file(audit_dir / "audit.sqlite3",
                    certificate["source"]["audit_database_sha256"]),
        verify_file(data / "how2sign_realigned_train.csv",
                    certificate["source"]["metadata_sha256"]),
    ]
    audit = json.loads((audit_dir / "audit_manifest.json").read_text())
    cokely = data / "cokely_reference/v1"
    inspection = cokely / "audit/i_have_a_dream_v2"
    index = json.loads((inspection / "artifact-index.json").read_text())
    for entry in index["files"]:
        name = entry["name"]
        if Path(name).name != name:
            raise ValueError("inspection artifact name must be a basename")
        checks.append(verify_file(inspection / name, entry["sha256"], entry["size"]))
    manifest = json.loads((inspection / "source-native-manifest.json").read_text())
    payloads = [manifest["eaf_file"], manifest["license_evidence_file"],
                *manifest["auxiliary_evidence_files"],
                *(binding["file"] for binding in manifest["media_bindings"])]
    for entry in payloads:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source payload path must remain inside its source root")
        checks.append(verify_file(cokely / "source/i_have_a_dream" / relative,
                                  entry["sha256"], entry["size"]))
    checks.append(verify_file(cokely / "control/EAFv3.0.xsd",
                              manifest["eaf_schema_sha256"]))
    metrics_path = data / "how2sign_motion_experiment/v1/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    return {
        "verified_files": checks,
        "all_recorded_hashes_match": all(check["matches"] for check in checks),
        "signer_certificate": file_record(signer_dir / "certificate.json"),
        "signer_certified": certificate["certified"],
        "final_split_created": certificate["final_split_created"],
        "audit_counts": audit["status_counts"],
        "metadata_rows": audit["metadata_rows"],
        "review_queue_records": audit["review_queue_records"],
        "quality_threshold_selected": audit["threshold_selection_performed"],
        "cokely_training_target_authorized": manifest["training_target_authorized"],
        "cokely_project_linguistically_validated": manifest["linguistically_validated_by_project"],
        "saved_motion_metrics_identity": file_record(metrics_path),
        "saved_motion_metrics": metrics,
        "limitations": ["missing payloads remain missing; no substitute is accepted",
                        "How2Sign raw clips were not rehashed or decoded",
                        "saved 2D model was not retrained or benchmarked",
                        "hash agreement does not establish rights or ASL validity",
                        "Cokely payloads were hashed, not re-decoded or linguistically reviewed"],
    }


def model_checks() -> dict:
    import torch
    from signtranslator.data.corpus import CorpusSpec, load_manifest
    from signtranslator.eval_framework.statistics import (
        paired_permutation_pvalue, sign_test_pvalue)
    from signtranslator.run import build_model, make_loaders, run_pipeline
    from signtranslator.training import checkpoint_paths

    spec = CorpusSpec.build(num_concepts=12, seq_len=4, num_joints=27,
                            in_channels=3, num_frames=32)
    try:
        build_model(replace(spec, num_joints=137), diff_timesteps=10)
        topology = {"accepted": True, "error": None}
    except ValueError as error:
        topology = {"accepted": False, "error": str(error)}
    with tempfile.TemporaryDirectory(prefix="signtranslator-w0-day1-") as temporary:
        base = Path(temporary)
        corpus = base / "synthetic"
        prefix = base / "smoke.pt"
        start = time.monotonic()
        trained = run_pipeline(str(corpus), epochs=1, diff_timesteps=10,
                               regenerate=True, ckpt_path=str(prefix),
                               do_analyze=False, verbose=False)
        elapsed = time.monotonic() - start
        checkpoints = checkpoint_paths(prefix)
        reloaded = run_pipeline(str(corpus), epochs=1, diff_timesteps=10,
                                do_train=False, ckpt_path=str(checkpoints["last"]),
                                do_analyze=False, verbose=False)
        for result in (trained, reloaded):
            result["model"].eval()
        tokens = torch.tensor([[3, 4, 5]], dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(91)
            expected = trained["model"].generate_from_gloss(tokens, ddim_steps=3)
            torch.manual_seed(91)
            actual = reloaded["model"].generate_from_gloss(tokens, ddim_steps=3)
            batch = next(iter(make_loaders(str(corpus), 32)[0]))
            masked = dict(batch)
            for key in ("validity_mask", "confidence", "frame_mask"):
                masked[key] = torch.zeros_like(batch[key])
            torch.manual_seed(123)
            supported = trained["model"].training_step(batch)["generation"].item()
            torch.manual_seed(123)
            unsupported = trained["model"].training_step(masked)["generation"].item()
        result = {
            "synthetic_smoke": {
                "corpus_readiness_passed": trained["readiness"].passed,
                "optimizer_steps": trained["trainer"].global_step,
                "seeded_last_checkpoint_reload_equal": torch.equal(expected, actual),
                "output_finite": bool(torch.isfinite(actual).all()),
                "output_shape": list(actual.shape),
                "training_seconds": round(elapsed, 3),
                "corpus_manifest": load_manifest(str(corpus)),
                "checkpoint_files": {kind: file_record(path)
                                     for kind, path in checkpoints.items()},
                "temporary_payloads_retained": False,
                "performance_or_asl_acceptance": False,
            },
            "known_defect_observations": {
                "B17_137_joint_build": topology,
                "B18_mask_support": {"supported_loss": supported,
                                     "all_invalid_loss": unsupported,
                                     "equal": supported == unsupported},
                "B23_oversized_batch_count": len(make_loaders(str(corpus), 10000)[0]),
                "B35_sign_test": {"actual": sign_test_pvalue([1.] * 60, [0.] * 60),
                                  "exact_reference": 2. ** -59},
                "B36_nan_permutation_pvalue": paired_permutation_pvalue(
                    [float("nan"), 1.], [0., 0.]),
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.absolute()
    data = args.data_root.resolve(strict=True)
    if output.resolve().is_relative_to(data):
        raise ValueError("output must not be inside the read-only data root")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(repo))
    import signtranslator
    from signtranslator.data_engineering.source_portfolio import CURRENT_PRE_PHASE_2_DECISION
    from signtranslator.planning.phase3c import assess_phase3c_readiness
    from signtranslator.reproducibility import runtime_environment

    if Path(signtranslator.__file__).resolve().parent != repo / "signtranslator":
        raise RuntimeError("baseline must import this checkout")
    before = snapshot(repo)
    baseline = {
        "schema_version": 1,
        "purpose": "W0 Day-1 baseline capture; not readiness approval",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "initial_worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True),
        "runtime": runtime_environment(),
        "installed_distributions": {name: importlib.metadata.version(name)
                                    for name in ("av", "numpy", "torch", "pytest")},
        "source_test_config_inventory": before,
    }
    commands = {}
    for name, argv in (
        ("dependencies", [sys.executable, "-m", "pip", "check"]),
        ("compile", [sys.executable, "-m", "compileall", "-q", "signtranslator", "tests"]),
        ("pytest", [sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
                    "-W", "error", "-p", "no:cacheprovider",
                    f"--junitxml={output / 'pytest.xml'}"]),
    ):
        print(f"Running {name}", flush=True)
        commands[name] = run_command(argv, repo, output / f"{name}.log")
    baseline["commands"] = commands
    xml = ET.parse(output / "pytest.xml").getroot()
    suites = list(xml.iter("testsuite"))
    totals = {key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    baseline["test_totals"] = totals
    baseline["pytest_xml"] = file_record(output / "pytest.xml")
    print("Running model and local-evidence checks", flush=True)
    baseline.update(model_checks())
    baseline["local_evidence"] = local_evidence(data)
    baseline["source_portfolio"] = CURRENT_PRE_PHASE_2_DECISION.to_dict()
    baseline["phase3c_without_external_evidence"] = asdict(assess_phase3c_readiness())
    baseline["source_test_config_unchanged"] = before == snapshot(repo)
    smoke = baseline["synthetic_smoke"]
    baseline["baseline_capture_complete"] = (
        all(command["returncode"] == 0 for command in commands.values())
        and totals["tests"] > 0 and totals["failures"] == totals["errors"] == totals["skipped"] == 0
        and smoke["corpus_readiness_passed"] and smoke["optimizer_steps"] > 0
        and smoke["seeded_last_checkpoint_reload_equal"] and smoke["output_finite"]
        and baseline["source_test_config_unchanged"])
    # A completed measurement may report failed source integrity. Never convert
    # that negative evidence into acceptance or silently substitute another file.
    baseline["data_integrity_passed"] = baseline["local_evidence"]["all_recorded_hashes_match"]
    baseline["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (output / "baseline.json").open("x", encoding="utf-8") as stream:
        json.dump(baseline, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"complete": baseline["baseline_capture_complete"],
                      "tests": totals, "output": str(output)}), flush=True)
    return 0 if baseline["baseline_capture_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
