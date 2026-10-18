"""Phonological features for 3D-LEX signs from the SAPA (3D-LEX authors') metadata.

The SAPA repository ships WLASL metadata merged with ASL-LEX expert phonology
(Kezar et al. 2023) for 988 glosses, plus two automatic handshape labelings derived
from 3D-LEX (Euclidean-distance segmentation and k-means). This module parses those
files strictly and joins them to 3D-LEX signs by exact gloss.

Every phonological field must be constant across a gloss's instances (it describes
the lexical sign, not a video); a gloss whose instances disagree is refused. The
join is an English-label match, like the Signbank candidates: it records which
ASL-LEX entry the label names, and needs the gloss review to confirm the performed
sign is that entry. Licences: SAPA code MIT; WLASL metadata C-UDA; ASL-LEX
CC BY-NC 4.0 — research use only.
"""

from __future__ import annotations

import hashlib
import json

PHONOLOGY_FIELDS = (
    "Handshape", "Nondominant Handshape", "Handshape Morpheme 2", "Selected Fingers", "Flexion", "Spread",
    "Spread Change", "Thumb Position", "Thumb Contact", "Major Location", "Minor Location",
    "Second Minor Location", "Contact", "Sign Type", "Path Movement", "Repeated Movement", "Wrist Twist",
    "Sign Onset", "Sign Offset",
)


def parse_sapa_metadata(payload: bytes) -> dict[str, dict]:
    """gloss -> {field: value} for every gloss, with per-gloss constancy enforced."""
    data = json.loads(payload)
    if not isinstance(data, list) or not data:
        raise ValueError("SAPA metadata must be a nonempty list of glosses")
    out = {}
    for entry in data:
        gloss, instances = entry.get("gloss"), entry.get("instances")
        if not isinstance(gloss, str) or not gloss or not isinstance(instances, list) or not instances:
            raise ValueError("every entry needs a gloss and instances")
        if gloss in out:
            raise ValueError(f"duplicate gloss {gloss!r}")
        features = {}
        for field in PHONOLOGY_FIELDS:
            values = {json.dumps(instance.get(field), sort_keys=True) for instance in instances}
            if len(values) != 1:
                raise ValueError(f"{gloss}: {field} differs between instances")
            features[field] = json.loads(values.pop())
        out[gloss] = features
    return out


def join_phonology(signs, *, expert: dict, kmeans: dict | None = None, euclidean: dict | None = None) -> dict:
    """Per sign: expert ASL-LEX features and automatic handshape labels, by exact gloss."""
    joined = {}
    for sign in sorted(set(signs)):
        if sign not in expert:
            joined[sign] = dict(status="no_gloss_entry", requires_review=True)
            continue
        joined[sign] = dict(
            status="gloss_match", requires_review=True, asl_lex=expert[sign],
            handshape_kmeans=(kmeans or {}).get(sign, {}).get("Handshape"),
            handshape_euclidean=(euclidean or {}).get(sign, {}).get("Handshape"))
    return joined


def metadata_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = ["PHONOLOGY_FIELDS", "parse_sapa_metadata", "join_phonology", "metadata_sha256"]
