"""Identity-bound loader for the CC0 hair, brows, lashes, top and skin texture of the avatar.

All from MakeHuman's system asset pack; ``source-intake/makehuman-system-assets/intake.json`` records the pack's URL and hash
and each file's CC0 statement. Fixed once for every clip: nothing here takes an argument that could differ by clip.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .makehuman import MakeHumanRig
from .makehuman_attachment import MakeHumanAttachment, _fit_attachment

_ITEMS = {"hair": ("hair/short01", "short01", "short01_diffuse.png"),
          "brows": ("eyebrows/eyebrow002", "eyebrow002", "eyebrow002.png"),
          "lashes": ("eyelashes/eyelashes01", "eyelashes01", "eyelashes01.png"),
          "top": ("clothes/male_casualsuit02", "male_casualsuit02", None)}
SKIN_TEXTURE = "skins/young_caucasian_male/young_lightskinned_male_diffuse.png"


@dataclass(frozen=True)
class Wardrobe:
    attachments: dict                   # object name -> MakeHumanAttachment
    body_deleted_vertices: np.ndarray   # vertices the top covers: the body faces entirely on them are not drawn
    skin_texture: Path
    licence: str
    source: dict


def _deleted_vertices(mhclo: str) -> np.ndarray:
    lines = mhclo.splitlines()
    if "delete_verts" not in lines:
        return np.zeros(0, dtype=np.int64)
    tokens = " ".join(lines[lines.index("delete_verts") + 1:]).split()
    out, i = [], 0
    while i < len(tokens):
        if i + 2 < len(tokens) and tokens[i + 1] == "-":
            out.extend(range(int(tokens[i]), int(tokens[i + 2]) + 1))
            i += 3
        else:
            out.append(int(tokens[i]))
            i += 1
    return np.array(sorted(set(out)), dtype=np.int64)


def load_wardrobe(asset_root: str | Path, rig: MakeHumanRig) -> Wardrobe:
    """Every file is checked against its pinned hash in ``intake.json`` before use."""
    root = Path(asset_root)
    intake = json.loads((root / "intake.json").read_text())
    pinned = {item["path"]: item["sha256"] for item in intake["files"]}
    for name, expected in pinned.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Wardrobe asset identity mismatch: {name}")
    attachments: dict[str, MakeHumanAttachment] = {}
    deleted = np.zeros(0, dtype=np.int64)
    for name, (folder, stem, texture) in _ITEMS.items():
        proxy = (root / folder / f"{stem}.mhclo").read_bytes()
        mesh = (root / folder / f"{stem}.obj").read_bytes()
        attachments[name] = _fit_attachment(proxy, mesh, root / folder / (texture or f"{stem}.mhmat"), rig)
        deleted = np.union1d(deleted, _deleted_vertices(proxy.decode("utf-8")))
    return Wardrobe(attachments, deleted, (root / SKIN_TEXTURE).resolve(), intake["licence"], intake["source"])


def body_corner_uvs(asset_root: str | Path, rig: MakeHumanRig, face_indices) -> np.ndarray:
    """(F, C, 2) corner UVs of the pinned base mesh's listed faces, in the OBJ's own face order."""
    uvs, face_uvs = [], []
    for line in (Path(asset_root) / "makehuman/data/3dobjs/base.obj").read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "vt":
            uvs.append((float(fields[1]), float(fields[2])))
        elif fields[0] == "f":
            face_uvs.append([int(v.split("/")[1]) - 1 for v in fields[1:]])
    if len(face_uvs) != len(rig.faces):
        raise ValueError("base mesh faces and UV faces disagree")
    uvs = np.asarray(uvs, dtype=np.float64)
    corners = max(len(rig.faces[i]) for i in face_indices)
    out = np.zeros((len(face_indices), corners, 2))
    for row, face in enumerate(face_indices):
        out[row, :len(face_uvs[face])] = uvs[face_uvs[face]]
    return out


__all__ = ["Wardrobe", "load_wardrobe", "body_corner_uvs", "SKIN_TEXTURE"]
