"""Explicit JSON-configured governed training; no default synthetic fallback."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re

from .config import TrainerConfig
from .data.governed_manifest import load_governed_planner_inputs, _file, _object, _unique
from .governed_run import run_governed_planner
from .planning.text_loci import LocusTextConfig
from .planning.source_intervention import _state_sha256
from .reproducibility import canonical_json_bytes
from .training import checkpoint_paths


def _output_prefix(root, value):
    if (type(value) is not str or not value or '\\' in value
            or any(part in ('', '.', '..') for part in value.split('/'))
            or Path(value).is_absolute()):
        raise ValueError('checkpoint prefix must be a normalized relative POSIX path')
    current = root
    parts = Path(value).parts
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise ValueError('checkpoint output must not traverse symlinks')
        if index < len(parts) - 1 and current.exists() and not current.is_dir():
            raise ValueError('checkpoint parent must be a directory or absent')
    return current


def _read_run_configuration(path: Path, expected_sha256: str):
    """Read and validate shared run settings without loading data or writing files."""
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError('run configuration must be a regular local file')
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('explicit lowercase configuration SHA-256 required')
    with path.open('rb') as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError('run configuration exceeds one MiB')
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('run configuration hash mismatch')
    value = json.loads(payload, object_pairs_hook=_unique)
    canonical_json_bytes(value)
    _object(value, ('schema_version', 'input_manifest', 'input_manifest_sha256', 'model_config',
                    'trainer_config', 'validation', 'shuffle'), 'run configuration')
    if type(value['schema_version']) is not int or value['schema_version'] != 1:
        raise ValueError('run configuration requires schema 1')
    if type(value['validation']) is not bool or type(value['shuffle']) is not bool:
        raise ValueError('explicit validation and shuffle booleans required')
    model = LocusTextConfig(**_object(value['model_config'],
                            (f.name for f in fields(LocusTextConfig)), 'model configuration'))
    trainer = value['trainer_config']
    _object(trainer, ('config_type', 'schema_version', 'values'), 'trainer configuration')
    if type(trainer['schema_version']) is not int or trainer['schema_version'] != 1:
        raise ValueError('trainer configuration requires schema 1')
    _object(trainer['values'], (f.name for f in fields(TrainerConfig)), 'trainer values')
    cfg = TrainerConfig.from_dict(trainer)
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32 or type(cfg.device) is not str or not cfg.device:
        raise ValueError('explicit uint32 seed and device string required')
    return value, model, cfg


def run_from_configuration(path: Path, *, expected_sha256: str,
                           resume_from: str | None = None, max_epochs: int | None = None,
                           allow_checkpoint_replacement: bool = False):
    """Return the governed run and a canonical summary after successful execution.

    Configuration and input references resolve relative to the configuration file.
    Only checkpoint outputs are written. Replacement requires an explicit flag;
    declared input files cannot be replaced even with that flag. Files must remain
    stable during execution; these path checks are not a concurrency-safe sandbox.
    """
    if type(allow_checkpoint_replacement) is not bool:
        raise ValueError('explicit boolean replacement policy required')
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs <= 0):
        raise ValueError('max_epochs must be a positive exact integer')
    value, model, cfg = _read_run_configuration(path, expected_sha256)
    root = path.parent.resolve()
    prefix = _output_prefix(root, cfg.ckpt_path)
    outputs = checkpoint_paths(prefix)
    destinations = tuple(p for target in outputs.values() for p in (target, Path(f'{target}.json')))
    for target in destinations:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError('checkpoint destinations must be regular files or absent')
        if target.exists() and not allow_checkpoint_replacement:
            raise ValueError('checkpoint destination exists; explicit replacement flag required')
    inputs = load_governed_planner_inputs(_file(root, value['input_manifest']),
                                         expected_sha256=value['input_manifest_sha256'])
    protected = set(inputs.declared_files) | {path.resolve()}
    if any(target in protected for target in destinations):
        raise ValueError('checkpoint destination overlaps a declared input file')
    resume = _file(root, resume_from) if resume_from is not None else None
    cfg.ckpt_path = str(prefix)
    run = run_governed_planner(inputs.corpus, inputs.vocabulary, inputs.alphabet,
        model_config=model, trainer_config=cfg, validation=value['validation'], shuffle=value['shuffle'],
        resume_from=resume, max_epochs=max_epochs)
    summary = dict(schema_version=1, configuration_sha256=expected_sha256,
                   input_manifest_sha256=inputs.manifest_sha256,
                   corpus_sha256=inputs.corpus.content_sha256,
                   completed_epochs=run.trainer.completed_epochs, global_step=run.trainer.global_step,
                   model_state_sha256=_state_sha256(run.trainer.model),
                   checkpoint_paths={key: str(target) for key, target in outputs.items()},
                   history=run.trainer.history, exposure=run.trainer.exposure_report().to_dict(),
                   declaration_audit=run.exposure_audit.to_dict(), phase_exit_approved=False)
    return run, canonical_json_bytes(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--resume-from', help='checkpoint path relative to the configuration directory')
    parser.add_argument('--max-epochs', type=int, help='stop after this many additional epochs; retain total horizon')
    parser.add_argument('--allow-checkpoint-replacement', action='store_true')
    args = parser.parse_args(argv)
    try:
        _, summary = run_from_configuration(args.configuration, expected_sha256=args.sha256,
            resume_from=args.resume_from, max_epochs=args.max_epochs,
            allow_checkpoint_replacement=args.allow_checkpoint_replacement)
    except (ValueError, PermissionError, OSError) as error:
        parser.error(str(error))
    print(summary.decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
