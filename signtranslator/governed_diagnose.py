"""Read-only development diagnostics for an explicitly bound governed checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .data.governed_manifest import _file, _object, _unique, load_governed_planner_inputs
from .governed_run import load_governed_planner, diagnose_governed_planner
from .governed_train import _read_run_configuration
from .reproducibility import canonical_json_bytes, sha256_file
from .planning.graph_decode import DiagnosticRelationThresholds
from .planning.tensors import EDGE_TYPES


def _threshold_manifest(root, reference, expected_sha256):
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('explicit lowercase threshold-manifest SHA-256 required')
    path = _file(root, reference)
    with path.open('rb') as stream:
        payload = stream.read(4097)
    if len(payload) > 4096 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('threshold manifest exceeds four KiB or hash mismatches')
    value = json.loads(payload, object_pairs_hook=_unique)
    canonical_json_bytes(value)
    _object(value, ('schema_version', 'relation_types', 'negative_below', 'positive_above'), 'threshold manifest')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['relation_types'] != [kind.value for kind in EDGE_TYPES]
            or type(value['negative_below']) is not list or type(value['positive_above']) is not list):
        raise ValueError('schema-1 thresholds with the exact relation codebook required')
    return DiagnosticRelationThresholds(tuple(value['negative_below']), tuple(value['positive_above']))


def diagnose_from_configuration(path: Path, *, expected_sha256: str,
                                checkpoint: str, checkpoint_sha256: str,
                                view: str, model_state: str,
                                sample_indices: tuple[int, ...], permutation: tuple[int, ...],
                                seed: int, max_samples: int,
                                relation_thresholds_manifest: str | None = None,
                                relation_thresholds_sha256: str | None = None) -> bytes:
    """Return canonical diagnostic evidence without training or file writes.

    Uses the original run configuration and full input-manifest admission. The
    checkpoint reference is relative to the configuration directory. Configured
    output destinations are unused. Files must remain stable: before/after digest
    checks are not an atomic filesystem snapshot or a concurrency guarantee.
    """
    if (view not in ('train', 'validation') or model_state not in ('current', 'best_validation')
            or type(sample_indices) is not tuple or not sample_indices
            or any(type(index) is not int or index < 0 for index in sample_indices)
            or len(set(sample_indices)) != len(sample_indices)
            or type(max_samples) is not int or not 1 <= max_samples <= 64
            or len(sample_indices) > max_samples
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(permutation) is not tuple or len(permutation) != len(sample_indices)
            or any(type(index) is not int for index in permutation)
            or set(permutation) != set(range(len(sample_indices)))):
        raise ValueError('explicit bounded development view, model, subset, permutation and uint32 seed required')
    if type(checkpoint_sha256) is not str or re.fullmatch('[0-9a-f]{64}', checkpoint_sha256) is None:
        raise ValueError('explicit lowercase checkpoint SHA-256 required')
    if (relation_thresholds_manifest is None) != (relation_thresholds_sha256 is None):
        raise ValueError('threshold manifest and its SHA-256 must be supplied together')
    value, model, cfg = _read_run_configuration(path, expected_sha256)
    root = path.parent.resolve()
    thresholds = (_threshold_manifest(root, relation_thresholds_manifest, relation_thresholds_sha256)
                  if relation_thresholds_manifest is not None else None)
    checkpoint_path = _file(root, checkpoint)
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError('checkpoint hash mismatch')
    inputs = load_governed_planner_inputs(_file(root, value['input_manifest']),
                                         expected_sha256=value['input_manifest_sha256'])
    run = load_governed_planner(inputs.corpus, inputs.vocabulary, inputs.alphabet,
        model_config=model, trainer_config=cfg, validation=value['validation'], shuffle=value['shuffle'],
        checkpoint_path=checkpoint_path)
    diagnostic = diagnose_governed_planner(run, view=view, model_state=model_state,
        sample_indices=sample_indices, permutation=permutation, seed=seed, max_samples=max_samples,
        relation_thresholds=thresholds)
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError('checkpoint changed during diagnostics')
    result = dict(schema_version=1, scope='governed-checkpoint-development-diagnostics',
        configuration_sha256=expected_sha256, input_manifest_sha256=inputs.manifest_sha256,
        corpus_sha256=inputs.corpus.content_sha256, checkpoint_sha256=checkpoint_sha256,
        diagnostic_sha256=diagnostic.sha256, diagnostic=diagnostic.to_dict(),
        declaration_audit_sha256=run.exposure_audit.sha256,
        declaration_audit=run.exposure_audit.to_dict(), phase_exit_approved=False)
    if thresholds is not None:
        result.update(schema_version=2, relation_thresholds_manifest_sha256=relation_thresholds_sha256)
    return canonical_json_bytes(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--checkpoint', required=True, help='path relative to the configuration directory')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--relation-thresholds-manifest', help='optional relative path to explicit uncalibrated thresholds')
    parser.add_argument('--relation-thresholds-sha256')
    parser.add_argument('--view', required=True, choices=('train', 'validation'))
    parser.add_argument('--model-state', required=True, choices=('current', 'best_validation'))
    parser.add_argument('--sample-indices', type=int, nargs='+', required=True)
    parser.add_argument('--permutation', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-samples', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        payload = diagnose_from_configuration(args.configuration, expected_sha256=args.sha256,
            checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
            view=args.view, model_state=args.model_state, sample_indices=tuple(args.sample_indices),
            permutation=tuple(args.permutation), seed=args.seed, max_samples=args.max_samples,
            relation_thresholds_manifest=args.relation_thresholds_manifest,
            relation_thresholds_sha256=args.relation_thresholds_sha256)
    except (ValueError, PermissionError, OSError) as error:
        parser.error(str(error))
    print(payload.decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
