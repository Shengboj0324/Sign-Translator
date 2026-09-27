"""Source-parameterized multichannel state; structural validity is not phase approval.

No joint counts, facial convention, rig or observation provenance is inferred.
Unavailable channels must be supplied explicitly with false validity and zero
confidence. Identity shape is deliberately outside this motion-only format.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .rotations import rotation_6d_to_matrix, is_rotation_matrix

# Every required channel is present even when unavailable. Head translation is
# separate from rotation so its SE(3) meaning cannot disappear inside face data.
CHANNELS = {
    'root_translation': ('translation', 3, 'm'),
    'body': ('rotation_6d_columns', 6, 'unitless'),
    'left_hand': ('rotation_6d_columns', 6, 'unitless'),
    'right_hand': ('rotation_6d_columns', 6, 'unitless'),
    'head_rotation': ('rotation_6d_columns', 6, 'unitless'),
    'head_translation': ('translation', 3, 'm'),
    'face': ('coefficient', 1, 'unitless'),
    'left_gaze': ('unit_vector', 3, 'unitless'),
    'right_gaze': ('unit_vector', 3, 'unitless'),
    'blink': ('bounded_scalar', 1, 'unitless'),
    'contact': ('bounded_scalar', 1, 'unitless'),
}
_SHA = re.compile(r'[0-9a-f]{64}')


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be explicitly named')


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate metadata key {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class MotionChannel:
    values: np.ndarray                  # T,J,D
    valid: np.ndarray                   # T,J
    observed: np.ndarray                # T,J; false+valid means inferred
    confidence: np.ndarray              # T,J; reliability, not probability
    labels: tuple[str, ...]             # source-specific joint/feature order
    coordinate_frame: str               # declared right-handed frame identifier
    units: str
    source_id: str
    source_sha256: str
    convention: str                     # versioned semantic/kinematic mapping
    inference_method: str | None = None

    def supervision_weights(self, *, allow_inferred: bool = False) -> np.ndarray:
        """Observed-only reliability by default; inferred supervision is explicit.

        This does not grant permission to use inferred targets. The adapter must
        bind any opt-in to its reviewed training protocol.
        """
        if type(allow_inferred) is not bool:
            raise ValueError('allow_inferred must be an explicit boolean')
        mask = self.valid if allow_inferred else self.valid & self.observed
        return np.where(mask, self.confidence, 0)

    def validate(self, name: str, frames: int) -> None:
        kind, width, units = CHANNELS[name]
        if (not isinstance(self.values, np.ndarray) or self.values.dtype not in (np.float32,np.float64)
                or self.values.ndim != 3 or self.values.shape[0] != frames
                or self.values.shape[1] < 1 or self.values.shape[2] != width):
            raise ValueError(f'{name}: expected nonempty floating (T,J,{width})')
        shape = self.values.shape[:2]
        if name in {'root_translation','head_rotation','head_translation','left_gaze','right_gaze'} and shape[1] != 1:
            raise ValueError(f'{name}: exactly one channel element is required')
        for label, mask in [('valid',self.valid),('observed',self.observed)]:
            if not isinstance(mask,np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape:
                raise ValueError(f'{name}: {label} must be boolean (T,J)')
        if (not isinstance(self.confidence,np.ndarray) or self.confidence.dtype not in (np.float32,np.float64)
                or self.confidence.shape != shape or not np.isfinite(self.confidence).all()
                or np.any((self.confidence < 0) | (self.confidence > 1))):
            raise ValueError(f'{name}: confidence must be finite reliability in [0,1]')
        if np.any(self.confidence[~self.valid] != 0):
            raise ValueError(f'{name}: invalid values cannot carry confidence')
        if not isinstance(self.labels,tuple) or len(self.labels) != shape[1]:
            raise ValueError(f'{name}: labels must match channel order')
        for label in self.labels:
            _text(label,'channel label')
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f'{name}: duplicate labels')
        for label in ('coordinate_frame','source_id','convention'):
            _text(getattr(self,label),label)
        if self.units != units:
            raise ValueError(f'{name}: units must be {units}')
        if not isinstance(self.source_sha256,str) or not _SHA.fullmatch(self.source_sha256):
            raise ValueError(f'{name}: source content hash is required')
        inferred = self.valid & ~self.observed
        if inferred.any():
            _text(self.inference_method, 'inference method for unobserved valid values')
        elif self.inference_method is not None:
            _text(self.inference_method,'inference method')
        supported = self.values[self.valid]
        if not np.isfinite(supported).all():
            raise ValueError(f'{name}: valid values must be finite')
        # Canonical absent payload is zero. No NaNs may leak into another encoder.
        if not np.all(self.values[~self.valid] == 0):
            raise ValueError(f'{name}: unavailable payload must be explicitly zero')
        if kind == 'rotation_6d_columns' and supported.size:
            matrices = rotation_6d_to_matrix(torch.from_numpy(supported.copy()))
            if not bool(is_rotation_matrix(matrices,atol=1e-4).all()):
                raise ValueError(f'{name}: rotation is outside SO(3)')
        elif kind == 'unit_vector' and supported.size:
            if (np.any(np.abs(supported) > 1.00001)
                    or not np.allclose(np.linalg.norm(supported,axis=-1),1.,rtol=0.,atol=1e-5)):
                raise ValueError(f'{name}: gaze must have unit norm')
        elif kind == 'bounded_scalar' and np.any((supported < 0) | (supported > 1)):
            raise ValueError(f'{name}: blink/contact state must be in [0,1]')


@dataclass(frozen=True)
class MultichannelMotion:
    timestamps: np.ndarray              # float64, seconds on a declared clock
    clock_id: str
    sample_id: str
    channels: dict[str, MotionChannel]
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported multichannel schema version')
        _text(self.clock_id,'clock_id'); _text(self.sample_id,'sample_id')
        if (not isinstance(self.timestamps,np.ndarray) or self.timestamps.dtype != np.float64
                or self.timestamps.ndim != 1 or len(self.timestamps) == 0
                or not np.isfinite(self.timestamps).all()):
            raise ValueError('clock must contain finite float64 seconds')
        with np.errstate(over='ignore',invalid='ignore'):
            delta = np.diff(self.timestamps)
        if not np.isfinite(delta).all() or np.any(delta <= 0):
            raise ValueError('clock must strictly increase with finite intervals')
        if not isinstance(self.channels,dict) or set(self.channels) != set(CHANNELS):
            raise ValueError('all multichannel components must be explicitly present')
        for name,channel in self.channels.items():
            if not isinstance(channel,MotionChannel):
                raise ValueError('channels must use MotionChannel contracts')
            channel.validate(name,len(self.timestamps))

    def save(self, path: Path) -> str:
        """Write a new NPZ, return its SHA-256. This does not authorize ingestion."""
        self.validate()  # revalidate mutable numpy payloads at the boundary
        arrays = {'timestamps':self.timestamps}
        metadata = {'schema_version':1,'clock_id':self.clock_id,'sample_id':self.sample_id,'channels':{}}
        for name,ch in sorted(self.channels.items()):
            for key in ('values','valid','observed','confidence'):
                arrays[f'{name}.{key}'] = getattr(ch,key)
            metadata['channels'][name] = {key:getattr(ch,key) for key in (
                'labels','coordinate_frame','units','source_id','source_sha256','convention','inference_method')}
        arrays['metadata'] = np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path, *, expected_sha256: str):
        """Verify the opened bytes and reconstruct without pickle or dtype coercion."""
        import io
        if not isinstance(expected_sha256,str) or not _SHA.fullmatch(expected_sha256):
            raise ValueError('expected content SHA-256 is required')
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError('motion shard content hash mismatch')
        with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
            expected = {'timestamps','metadata'} | {f'{name}.{key}' for name in CHANNELS
                for key in ('values','valid','observed','confidence')}
            if set(archive.files) != expected or len(archive.files) != len(expected):
                raise ValueError('motion shard array inventory mismatch')
            raw = archive['metadata']
            if raw.shape != () or raw.dtype.kind != 'U':
                raise ValueError('metadata must be a Unicode scalar')
            meta = json.loads(str(raw),object_pairs_hook=_strict_object)
            if (not isinstance(meta,dict)
                    or set(meta) != {'schema_version','clock_id','sample_id','channels'}
                    or not isinstance(meta['channels'],dict) or set(meta['channels']) != set(CHANNELS)):
                raise ValueError('unexpected motion metadata fields')
            channels = {}
            fields = {'labels','coordinate_frame','units','source_id','source_sha256','convention','inference_method'}
            for name in CHANNELS:
                entry = meta['channels'][name]
                if not isinstance(entry,dict) or set(entry) != fields or not isinstance(entry['labels'],list):
                    raise ValueError('unexpected channel metadata fields')
                channels[name] = MotionChannel(**{key:archive[f'{name}.{key}'].copy()
                    for key in ('values','valid','observed','confidence')},
                    **{**entry,'labels':tuple(entry['labels'])})
            state = cls(archive['timestamps'].copy(),meta['clock_id'],meta['sample_id'],channels,meta['schema_version'])
        state.validate()
        return state
