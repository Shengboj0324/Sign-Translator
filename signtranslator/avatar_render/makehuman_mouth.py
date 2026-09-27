"""Load exact teeth/tongue assets from the audited official system archive."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .makehuman import MakeHumanRig
from .makehuman_attachment import MakeHumanAttachment, _fit_attachment

ARCHIVE_SHA256 = "b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"
_FILES = {
    'teeth/teeth_base/teeth_base.obj': 'f55198069e55d360c4b4cc7ecb1cc292b2c7665753ab8b65eabf46a8783f2875',
    'teeth/teeth_base/teeth_base.mhclo': '9edc3deff3bb97a95c791878064dd069a74a38f95ea6d9707a9ba68d888b3b5c',
    'teeth/teeth_base/teeth.mhmat': '3baeff72d9eb06ca16c3cff7ca2ab304216717dee247e0d88f1dc33037c8320e',
    'teeth/teeth_base/teeth.png': 'd0afb57869c6fbb56b98f5efc4aec629ee7593e70d8dd2bfb9de923acfb65f43',
    'tongue/tongue01/tongue01.obj': '12f4a6a9f85abae2ce3b4aa42d8119e1f679a437e0e89cb928ec68b29a7a1587',
    'tongue/tongue01/tongue01.mhmat': '7769df66c69b8c09a0ec713d70d8381abbb08ac82207369f5413970495cd4340',
    'tongue/tongue01/tongue01_diffuse.png': '3150be398e48e8ba1feac164c3143c16c0b2e959d9ffb6155f0171f59cfb4ea9',
    'tongue/tongue01/tongue01.mhclo': '61d825899cd78ff3146fa1324118060cb44b095fae1427d408c199ade74e3c9e',
}
_SPECS = {
    "teeth": ("teeth/teeth_base/teeth_base.mhclo", "teeth/teeth_base/teeth_base.obj", "teeth/teeth_base/teeth.png"),
    "tongue": ("tongue/tongue01/tongue01.mhclo", "tongue/tongue01/tongue01.obj", "tongue/tongue01/tongue01_diffuse.png"),
}


def load_makehuman_mouth(asset_root: str | Path, rig: MakeHumanRig) -> dict[str, MakeHumanAttachment]:
    """Fit both verified attachments in source units; no speech/ASL claim."""
    payloads = {}
    for name, expected in _FILES.items():
        payload = (Path(asset_root)/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Mouth asset identity mismatch: {name}")
        payloads[name] = payload
    return {
        kind: _fit_attachment(payloads[proxy], payloads[mesh], Path(asset_root)/texture, rig)
        for kind, (proxy, mesh, texture) in _SPECS.items()
    }
