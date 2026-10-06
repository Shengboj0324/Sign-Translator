"""Explicit local handoff into existing governed admission; no inferred evidence."""
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import re

from ..data_engineering.phase2_policy import AuthorizationEvidence
from ..data_engineering.schema import ConsentState, DataAuthorization, Sample, validate_sample
from ..data_engineering.source_portfolio import (
    AccessStatus, CapabilityEvidence, EvidenceLevel, RightsStatus, SourceCandidate,
)
from ..planning.label_vocabulary import GovernedLabelVocabulary
from ..planning.loci import LocusAlphabet
from ..planning.supervision import GovernedArtifact, GovernedSIRAnnotation
from ..pseudo_gloss.contracts import WeakGlossCandidateRecord
from ..reproducibility import canonical_json_bytes
from .governed_corpus import GovernedMotionCorpus, GovernedMotionRecord

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _object(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'{label} fields do not match schema 1')
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate governed manifest field: {key}')
        result[key] = value
    return result


def _enum(kind, value):
    if type(value) is not str or value not in kind.__members__:
        raise ValueError(f'explicit {kind.__name__} member name required')
    return kind[value]


def _file(root, reference):
    if (type(reference) is not str or not reference or '\\' in reference
            or any(part in ('', '.', '..') for part in reference.split('/'))):
        raise ValueError('manifest file references require normalized relative POSIX paths')
    path = Path(reference)
    if path.is_absolute():
        raise ValueError('manifest file references must be relative')
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('manifest file references must not traverse symlinks')
    if not current.is_file():
        raise ValueError(f'manifest reference is not a regular local file: {reference}')
    return current


def _source(value):
    value = dict(_object(value, (f.name for f in fields(SourceCandidate)), 'source'))
    capabilities = value['capabilities']
    if not isinstance(capabilities, list):
        raise ValueError('source capabilities must be an explicit list')
    value['capabilities'] = tuple(CapabilityEvidence(
        _object(item, ('capability', 'level'), 'capability')['capability'],
        _enum(EvidenceLevel, item['level'])) for item in capabilities)
    value['access'] = _enum(AccessStatus, value['access'])
    for name in ('research_rights', 'commercial_training_rights', 'commercial_deployment_rights'):
        value[name] = _enum(RightsStatus, value[name])
    return SourceCandidate(**value)


def _sample(value):
    value = dict(_object(value, (f.name for f in fields(Sample)), 'sample'))
    for name in ('sample_id', 'source_id', 'signer_id_hash', 'target_language', 'license',
                 'intended_use', 'smplx_version', 'provenance', 'split'):
        item = value[name]
        if type(item) is not str or not item or item != item.strip():
            raise ValueError(f'explicit nonempty sample {name} required')
    for name in ('dialect', 'video_uri', 'audio_uri'):
        if value[name] is not None and type(value[name]) is not str:
            raise ValueError(f'sample {name} must be a string or null')
    for name in ('calibration', 'frame_transform', 'time_transform'):
        if value[name] is not None and not isinstance(value[name], dict):
            raise ValueError(f'sample {name} must be an object or null')
    if value['transcript_lattice'] is not None and not isinstance(value['transcript_lattice'], list):
        raise ValueError('sample transcript_lattice must be a list or null')
    if (not isinstance(value['annotation_tiers'], dict)
            or any(not isinstance(v, list) for v in value['annotation_tiers'].values())):
        raise ValueError('sample annotation_tiers must map names to lists')
    for name in ('confidence_2d', 'confidence_3d', 'retention_date'):
        if value[name] is not None and type(value[name]) not in (int, float):
            raise ValueError(f'sample {name} must be a number or null')
    if not isinstance(value['weak_gloss_candidates'], list):
        raise ValueError('sample weak_gloss_candidates must be a list')
    value['weak_gloss_candidates'] = tuple(WeakGlossCandidateRecord.from_dict(v)
                                            for v in value['weak_gloss_candidates'])
    value['consent'] = _enum(ConsentState, value['consent'])
    value['authorization'] = DataAuthorization.from_manifest(value['authorization'])
    sample = Sample(**value)
    violations = validate_sample(sample)
    if violations:
        raise ValueError(f'invalid manifest sample: {violations}')
    return sample


@dataclass(frozen=True)
class GovernedPlannerInputs:
    corpus: GovernedMotionCorpus
    vocabulary: GovernedLabelVocabulary
    alphabet: LocusAlphabet
    manifest_sha256: str

    @property
    def phase_exit_approved(self):
        return False


def load_governed_planner_inputs(path: Path, *, expected_sha256: str) -> GovernedPlannerInputs:
    """Load a complete declared population from a byte-bound local JSON manifest.

    References are relative to the manifest directory and cannot traverse symlinks.
    The existing corpus admission checks all declared splits, including test, before
    any training view is exposed. Recorded grants/review claims are not independently
    verified human qualifications or legal interpretations. No files are written.
    """
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError('governed manifest must be a regular local file')
    if type(expected_sha256) is not str or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('explicit lowercase manifest SHA-256 required')
    with path.open('rb') as stream:
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError('governed manifest exceeds size limit')
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('governed manifest hash mismatch')
    value = json.loads(payload, object_pairs_hook=_unique)
    canonical_json_bytes(value)  # Reject NaN, Infinity and overflowed JSON numbers.
    _object(value, ('schema_version', 'scope', 'sources', 'authorizations', 'records',
                    'lexicon', 'convention'), 'governed manifest')
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or value['scope'] != 'research':
        raise ValueError('governed manifest requires schema 1 and explicit research scope')
    if not isinstance(value['sources'], list) or not isinstance(value['authorizations'], dict):
        raise ValueError('explicit source list and authorization map required')
    sources = tuple(_source(item) for item in value['sources'])
    if len({s.source_id for s in sources}) != len(sources):
        raise ValueError('duplicate manifest source identities')
    root = path.parent.resolve()
    authorizations = {}
    for key, item in value['authorizations'].items():
        _object(item, ('source_id', 'authorization', 'consent', 'local_evidence'), 'source authorization')
        if item['source_id'] != key:
            raise ValueError('authorization key and subject differ')
        authorizations[key] = AuthorizationEvidence(DataAuthorization.from_manifest(item['authorization']),
            _enum(ConsentState, item['consent']), _file(root, item['local_evidence']), key)
    if not isinstance(value['records'], list) or not value['records']:
        raise ValueError('nonempty declared record population required')
    records = []
    for item in value['records']:
        _object(item, (f.name for f in fields(GovernedMotionRecord)), 'motion record')
        if not isinstance(item['source_files'], dict):
            raise ValueError('source_files must map source identities to paths')
        arguments = dict(item)
        for name in ('motion_path', 'video_path', 'transcript_path', 'annotation_authorization_path', 'alignment_path'):
            arguments[name] = _file(root, item[name])
        arguments['source_files'] = {key: _file(root, ref) for key, ref in item['source_files'].items()}
        arguments['annotation'] = GovernedSIRAnnotation.from_manifest(item['annotation'])
        arguments['sample'] = _sample(item['sample'])
        records.append(GovernedMotionRecord(**arguments))
    artifacts = {}
    for name in ('lexicon', 'convention'):
        item = _object(value[name], ('artifact', 'path'), name)
        artifact = GovernedArtifact.from_dict(item['artifact'])
        artifact_path = _file(root, item['path'])
        with artifact_path.open('rb') as stream:
            artifact_bytes = stream.read(4 * 1024 * 1024 + 1)
        artifacts[name] = artifact, artifact_bytes
    vocabulary = GovernedLabelVocabulary(artifacts['lexicon'][0], artifacts['convention'][0], artifacts['lexicon'][1])
    alphabet = LocusAlphabet(artifacts['convention'][0], artifacts['convention'][1])
    if any(r.annotation.lexicon != vocabulary.lexicon or r.annotation.convention != vocabulary.convention
           for r in records):
        raise ValueError('manifest artifacts differ from record annotation bindings')
    corpus = GovernedMotionCorpus(records, sources=sources, authorizations=authorizations)
    return GovernedPlannerInputs(corpus, vocabulary, alphabet, expected_sha256)
