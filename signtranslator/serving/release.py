"""A release: every model a deployed translator uses, pinned by store key and SHA-256, with its training provenance.

A release manifest (JSON) names:

- ``planner``: the gloss planner checkpoint (``train_gloss_planner.py``) - optional; without it the generator is
  conditioned on the English directly;
- ``generator``: the motion generator checkpoint, the manifest it was trained on, what it is conditioned on
  (``plan`` - the planner's plan line - or ``english``) and its sampling settings (guidance, per-part temperatures);
- ``fps`` and ``seconds_per_sign`` (the duration rule when the plan is known);
- ``provenance``: one entry per training corpus with its licence and whether it is research-only.

**Licence gate.** ``public_release_allowed`` is false while any research-only corpus (ASLLRP / DAI, ASL Citizen's
research licence) lacks recorded written permission (``permission: "granted ..."``): such a release may run for
research and evaluation, never as a public one, until every such corpus is excluded or cleared (docs/NEXT_PHASE_PLAN.md,
"Licence gate for release").
``commercial_use_allowed`` is also false while any corpus is non-commercial (How2Sign, CC BY-NC 4.0).

Every model is fetched once and checked against its hash before anything loads it; a mismatch refuses the release.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re

_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Pinned:
    key: str
    sha256: str

    def __post_init__(self):
        if not self.key or not _SHA.fullmatch(self.sha256 or ""):
            raise ValueError(f"pinned model needs a key and a 64-hex SHA-256: {self.key!r}")


@dataclass(frozen=True)
class GeneratorSpec:
    checkpoint: Pinned
    manifest_sha256: str
    condition: str = "english"                       # "english" or "plan"
    guidance: float = 1.
    part_temperature: dict = field(default_factory=dict)
    head_smoothing: float = 0.                        # seconds (postprocess.smooth_rotation_channel on head_rotation)
    face_calibration: dict | None = None              # postprocess.estimate_face_calibration, estimated on validation

    def __post_init__(self):
        if self.head_smoothing < 0:
            raise ValueError("head_smoothing must be >= 0 seconds")
        if self.face_calibration is not None and not all(
                k in self.face_calibration for k in ("labels", "center_generated", "center_real", "gain")):
            raise ValueError("face_calibration must be an estimate_face_calibration result")
        if self.condition not in ("english", "plan"):
            raise ValueError("generator condition must be 'english' or 'plan'")
        if not _SHA.fullmatch(self.manifest_sha256 or ""):
            raise ValueError("generator manifest_sha256 must be a 64-hex SHA-256")


@dataclass(frozen=True)
class Corpus:
    name: str
    licence: str
    research_only: bool
    non_commercial: bool = False
    permission: str | None = None       # e.g. "requested 2026-10-22", "granted 2026-11-02 (email from ...)"

    @property
    def cleared(self) -> bool:
        """A research-only corpus is cleared for a public release only by recorded written permission."""
        return not self.research_only or (self.permission or "").startswith("granted")


@dataclass(frozen=True)
class Release:
    name: str
    version: str
    generator: GeneratorSpec
    provenance: tuple[Corpus, ...]
    planner: Pinned | None = None
    fps: float = 24.
    seconds_per_sign: float = .57
    max_text_chars: int = 300

    def __post_init__(self):
        if self.generator.condition == "plan" and self.planner is None:
            raise ValueError("a plan-conditioned generator needs a planner")
        if not self.provenance:
            raise ValueError("a release must list its training provenance")

    @property
    def public_release_allowed(self) -> bool:
        return all(c.cleared for c in self.provenance)

    @property
    def commercial_use_allowed(self) -> bool:
        return self.public_release_allowed and not any(c.non_commercial for c in self.provenance)

    def to_dict(self) -> dict:
        data = asdict(self)
        for k, default in (("head_smoothing", 0.), ("face_calibration", None)):   # unused post-processing is
            if data["generator"].get(k) == default:                              # omitted: older releases keep
                data["generator"].pop(k)                                         # their SHA-256
        data["public_release_allowed"] = self.public_release_allowed
        data["commercial_use_allowed"] = self.commercial_use_allowed
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "Release":
        g = dict(data["generator"])
        g["checkpoint"] = Pinned(**g["checkpoint"])
        planner = Pinned(**data["planner"]) if data.get("planner") else None
        rest = {k: v for k, v in data.items()
                if k not in ("generator", "planner", "provenance", "public_release_allowed", "commercial_use_allowed",
                             "selection")}            # how the setting was chosen: documentation, not part of the spec
        return cls(generator=GeneratorSpec(**g), planner=planner,
                   provenance=tuple(Corpus(**c) for c in data["provenance"]), **rest)


def fetch_pinned(store, pinned: Pinned) -> bytes:
    payload = store.get(pinned.key)
    if hashlib.sha256(payload).hexdigest() != pinned.sha256:
        raise ValueError(f"{pinned.key} does not match its pinned SHA-256")
    return payload


__all__ = ["Pinned", "GeneratorSpec", "Corpus", "Release", "fetch_pinned"]
