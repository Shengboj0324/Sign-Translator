"""License/consent gate + Merkle-style provenance chain (Doc-10 §2).

The pipeline gates BEFORE download: no valid license + granted consent + allowed
intended-use => no acquisition. Every preprocessing step is chained into a hash
`h_i = H(h_{i-1} ‖ step_i ‖ output_i)`, so the final root certifies the exact
sequence and any tampering changes the root (a reproduction certificate).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, List, Sequence

from .schema import ConsentState

_GENESIS = "0" * 64  # empty-chain root


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Deterministic content hash of any JSON-serialisable object.

    Canonical serialisation (sorted keys, no incidental whitespace) so that
    logically-equal objects hash equal and any change flips the digest.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return sha256_hex(payload)


# ---------------------------------------------------------------------------
# license / consent gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reasons: tuple  # violated preconditions when not allowed


def gate_download(license: str, consent: ConsentState, intended_use: str,
                  allowed_uses: Sequence[str]) -> GateDecision:
    """Permit acquisition only with license AND granted consent AND allowed use."""
    reasons: List[str] = []
    if not license:
        reasons.append("no_license")
    if consent != ConsentState.GRANTED:
        reasons.append("consent_not_granted")
    if not intended_use:
        reasons.append("no_intended_use")
    elif intended_use not in allowed_uses:
        reasons.append("use_not_permitted")
    return GateDecision(allowed=not reasons, reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# Merkle-style provenance chain
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProvenanceStep:
    name: str
    output_hash: str


@dataclass
class ProvenanceChain:
    """An append-only hash chain over preprocessing steps."""

    steps: List[ProvenanceStep] = field(default_factory=list)
    _root: str = _GENESIS

    @property
    def root(self) -> str:
        return self._root

    @staticmethod
    def _link(prev_root: str, name: str, output_hash: str) -> str:
        # H(prev ‖ step ‖ output); '‖' is an unambiguous field separator.
        return sha256_hex(f"{prev_root}\x1f{name}\x1f{output_hash}".encode("utf-8"))

    def append(self, name: str, output: Any) -> str:
        """Record a step (name + output) and advance the root."""
        oh = content_hash(output)
        self._root = self._link(self._root, name, oh)
        self.steps.append(ProvenanceStep(name, oh))
        return self._root

    def verify(self) -> bool:
        """Recompute the root from the recorded steps; True iff consistent."""
        return self.recompute_root(self.steps) == self._root

    @staticmethod
    def recompute_root(steps: Sequence[ProvenanceStep]) -> str:
        r = _GENESIS
        for s in steps:
            r = ProvenanceChain._link(r, s.name, s.output_hash)
        return r
