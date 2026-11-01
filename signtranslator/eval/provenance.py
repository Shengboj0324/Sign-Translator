"""Which code and which environment produced a read. ``code_digest`` is a SHA-256 over every ``scripts/*.py`` and ``signtranslator/**/*.py`` file, by
relative path and bytes, plus any ``extra_files`` (a job's shell file holds its pins and comparator order). ``environment`` names what can move a
hypothesis without moving any code: python, torch, transformers, tokenizers and the GPU."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]


def code_digest(extra_files=()) -> str:
    """SHA-256 over the code (module docstring). ``extra_files``: paths, relative to the repository or absolute. A missing file raises."""
    files = {p.resolve() for p in (ROOT / "scripts").glob("*.py")} | {p.resolve() for p in (ROOT / "signtranslator").rglob("*.py")}
    files |= {(Path(f) if Path(f).is_absolute() else ROOT / f).resolve() for f in extra_files}
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda p: str(p.relative_to(ROOT))):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    """What can move a hypothesis without moving any code. GPU name None on a machine without one."""
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=_version("transformers"), tokenizers=_version("tokenizers"),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


__all__ = ["code_digest", "environment"]
