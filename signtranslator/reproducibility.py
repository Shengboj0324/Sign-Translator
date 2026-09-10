"""Reproducibility primitives shared by audits, training, and release artifacts.

Git metadata is useful context, but it is not a content identity and is absent from
built wheels and source archives.  The authoritative implementation identity here is
therefore an ordered SHA-256 commitment to the exact source bytes.  A Git revision is
recorded when available and otherwise remains explicitly absent; it is never invented.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import __version__


IDENTITY_SCHEMA_VERSION = 1
_HASH_BLOCK_SIZE = 1024 * 1024


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one non-symlink regular file without following path indirection."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"identity input must be a non-symlink regular file: {source}")
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    after = source.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"file changed while hashing: {source}")
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-domain value uniquely and reject NaN/Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _git_metadata(repo_root: Path) -> tuple[str | None, bool | None]:
    # Do not allow Git to walk into an unrelated parent checkout when an extracted
    # archive happens to live inside one.
    if not (repo_root / ".git").exists():
        return None, None
    try:
        revision = subprocess.run(
            ["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", os.fspath(repo_root), "status", "--porcelain",
             "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("repository advertises Git metadata but it is unreadable") from error
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError("Git returned a malformed commit identity")
    return revision, not bool(status.strip())


def implementation_identity(
    paths: Sequence[str | os.PathLike[str]],
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Commit to an ordered set of implementation files.

    The aggregate is domain-separated and binds each normalized label, byte size, and
    individual file hash.  Directory iteration order therefore cannot alter the result.
    Relative labels are used only when a file is contained by ``repo_root``; external
    files retain their absolute path and cannot collide with repository files.
    """
    if not paths:
        raise ValueError("implementation identity requires at least one source file")
    root = Path(repo_root).resolve() if repo_root is not None else None
    entries: list[dict[str, Any]] = []
    labels: set[str] = set()
    for raw_path in paths:
        source = Path(raw_path)
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"implementation source must be a non-symlink regular file: {source}")
        resolved = source.resolve()
        if root is not None:
            try:
                label = resolved.relative_to(root).as_posix()
            except ValueError:
                label = resolved.as_posix()
        else:
            label = resolved.as_posix()
        if label in labels:
            raise ValueError(f"duplicate implementation source label: {label}")
        labels.add(label)
        entries.append({
            "path": label,
            "sha256": sha256_file(resolved),
            "size": resolved.stat().st_size,
        })
    entries.sort(key=lambda item: item["path"])

    aggregate = hashlib.sha256(b"signtranslator-implementation-identity-v1\0")
    for entry in entries:
        aggregate.update(canonical_json_bytes(entry))
        aggregate.update(b"\0")
    revision, worktree_clean = (_git_metadata(root) if root is not None
                                else (None, None))
    lock = root / "requirements.lock" if root is not None else None
    lock_hash = sha256_file(lock) if lock is not None and lock.is_file() else None
    if revision is None:
        identity_kind = "content-only"
    elif worktree_clean:
        identity_kind = "git-clean+content"
    else:
        identity_kind = "git-dirty+content"
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "package_version": __version__,
        "identity_kind": identity_kind,
        "git_revision": revision,
        "git_worktree_clean": worktree_clean,
        "dependency_lock_sha256": lock_hash,
        "implementation_sha256": aggregate.hexdigest(),
        "sources": entries,
    }


def package_implementation_identity() -> dict[str, Any]:
    """Identity of all importable Python implementation files in this package."""
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    sources: list[Path] = list(package_root.rglob("*.py"))
    sources.sort(key=lambda path: path.as_posix())
    return implementation_identity(sources, repo_root=repo_root)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG family used by the core training path."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def runtime_environment() -> dict[str, str]:
    """Version the numerical runtime that can affect model execution."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError(f"RNG state fields must be exactly {sorted(required)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def seed_all(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_deterministic_rng(seed: int) -> Iterator[None]:
    """Use a fixed RNG stream without perturbing the surrounding training stream."""
    previous = capture_rng_state()
    seed_all(seed)
    try:
        yield
    finally:
        restore_rng_state(previous)
