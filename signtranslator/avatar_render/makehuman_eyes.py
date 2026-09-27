"""Identity-bound loader for the audited MakeHuman high-poly eye assets."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .makehuman import MakeHumanRig
from .makehuman_attachment import MakeHumanAttachment, _fit_attachment

# Compatibility name for callers of the original eye-only adapter.
MakeHumanEyes = MakeHumanAttachment

_FILES = {
    "makehuman/data/eyes/high-poly/high-poly.mhclo": "b183cfe37120ab726f9b3f2ea6cd3a64c44ce7b4bd91a77c841cf70c04f83a0d",
    "makehuman/data/eyes/high-poly/high-poly.obj": "da2493215b708a344c33dc72f2a9a5b8fa985dcc5a70ad3b208995cf871da8e1",
    "makehuman/data/eyes/materials/brown.mhmat": "4abad93ce50541c08127721e77bce035640c9f16bed16cd0f5524b1cd5908465",
    "makehuman/data/eyes/materials/brown_eye.png": "4659691c7295ad6206c78b003e5fd0e5f91dcd53032fa914a229bb48cabe424b",
}


def load_makehuman_eyes(asset_root: str | Path, rig: MakeHumanRig) -> MakeHumanEyes:
    """Fit pinned eye geometry and retain its texture/UV correspondence."""
    payloads = {}
    for name, expected in _FILES.items():
        payload = (Path(asset_root)/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Eye asset identity mismatch: {name}")
        payloads[name] = payload
    return _fit_attachment(
        payloads["makehuman/data/eyes/high-poly/high-poly.mhclo"],
        payloads["makehuman/data/eyes/high-poly/high-poly.obj"],
        Path(asset_root)/"makehuman/data/eyes/materials/brown_eye.png", rig,
    )
