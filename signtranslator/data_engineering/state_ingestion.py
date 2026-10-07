"""Scoped, content-bound export of structurally validated Phase-2 motion.

This is the canonical multichannel ingestion boundary, not the legacy gloss
exporter. Human source/channel interpretation and source-to-rig validation remain
separate phase-exit requirements, even if this eligibility check succeeds.
"""
from __future__ import annotations

from pathlib import Path

from ..pose.multichannel import MultichannelMotion
from ..reproducibility import sha256_file
from .phase2_policy import Phase2Scope,AuthorizationEvidence,assess_phase2_scope
from .source_portfolio import SourceCandidate


def validate_phase2_state(state: MultichannelMotion, *,
                        sources: tuple[SourceCandidate,...], scope: Phase2Scope,
                        authorizations: dict[str,AuthorizationEvidence],
                        source_files: dict[str,Path]) -> dict:
    """Validate source eligibility and bytes without writing or approving a phase."""
    if not isinstance(state,MultichannelMotion):
        raise ValueError('a typed multichannel state is required')
    state.validate()
    decision = assess_phase2_scope(sources,scope,authorizations)
    if not decision['portfolio_eligible']:
        raise PermissionError('Phase-2 source portfolio is not eligible for the requested scope')
    eligible_motion = set(next(d for d in decision['decisions']
                              if d['requirement']=='coobserved_motion')['eligible_sources'])
    motion_sources = {channel.source_id for channel in state.channels.values() if channel.valid.any()}
    if not motion_sources or not motion_sources <= eligible_motion:
        raise PermissionError('valid motion channels require an eligible coobserved source')
    # Multiple independently recorded sources cannot masquerade as one synchronized
    # observation. A separately validated synchronization/derivative artifact must
    # be registered as one source before this single-clock boundary can accept it.
    if len(motion_sources)!=1:
        raise ValueError('single-clock state must bind one coobserved motion source')
    hashes = {}
    for name,channel in state.channels.items():
        if not channel.valid.any():
            continue
        source = source_files.get(channel.source_id)
        if not isinstance(source,Path) or source.is_symlink() or not source.is_file():
            raise ValueError(f'{name}: source must be a regular local file')
        if channel.source_id not in hashes:
            hashes[channel.source_id] = sha256_file(source)
        if hashes[channel.source_id] != channel.source_sha256:
            raise ValueError(f'{name}: source hash does not match motion provenance')
    return {'source_sha256': hashes, 'eligibility': decision, 'phase_exit_approved': False}


def export_phase2_state(state: MultichannelMotion, destination: Path, *,
                        sources: tuple[SourceCandidate,...], scope: Phase2Scope,
                        authorizations: dict[str,AuthorizationEvidence],
                        source_files: dict[str,Path]) -> dict:
    """Validate supplied evidence before creating any output; never infer approval."""
    validation = validate_phase2_state(state, sources=sources, scope=scope,
                                      authorizations=authorizations, source_files=source_files)
    digest = state.save(destination)
    return {'schema_version':1,'motion_path':str(Path(destination).resolve()),
            'motion_sha256':digest, **validation,
            'required_next_evidence':['source-specific channel mapping and synchronization',
                                     'authorized rig inverse-transform/render round-trip',
                                     'qualified review of semantic and geometric fidelity']}
