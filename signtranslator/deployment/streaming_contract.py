"""Streaming contract + display-commit monotonicity (Doc-13 §1).

The system holds a committed prefix C_t, a revisable suffix U_t, and avatar state
q_t; it may revise only U_t. The display-commit monotonicity invariant makes
"already-displayed signs cannot be silently changed" checkable: the committed prefix
is append-only. Reuses the Doc-01 commitment-error accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..speech.revision import commitment_error_count

Tokens = Tuple[int, ...]


class CommitViolationError(RuntimeError):
    """Raised when a new commitment rewrites an already-committed (displayed) sign."""


def is_prefix(a: Sequence[int], b: Sequence[int]) -> bool:
    """True iff ``a`` is a (not necessarily proper) prefix of ``b``."""
    if len(a) > len(b):
        return False
    return all(a[i] == b[i] for i in range(len(a)))


def certify_commit_monotone(history: Sequence[Sequence[int]]
                            ) -> Tuple[bool, Optional[int]]:
    """Certify a sequence of committed prefixes is append-only.

    Returns (True, None) iff every committed prefix extends the previous one;
    otherwise (False, i) with ``i`` the first step whose commit is not an
    extension of step i-1 (a silently-changed displayed sign).
    """
    for i in range(1, len(history)):
        if not is_prefix(history[i - 1], history[i]):
            return False, i
    return True, None


@dataclass
class StreamingContract:
    """Committed prefix C_t / revisable suffix U_t / avatar state q_t."""

    committed: Tokens = ()
    suffix: Tokens = ()
    avatar_state: Optional[object] = None
    history: List[Tokens] = field(default_factory=list)

    def __post_init__(self):
        if not self.history:
            self.history = [tuple(self.committed)]

    def revise_suffix(self, new_suffix: Sequence[int]) -> None:
        """Revising the suffix is always allowed (nothing displayed yet)."""
        self.suffix = tuple(new_suffix)

    def commit(self, new_committed: Sequence[int]) -> None:
        """Advance the committed prefix; must EXTEND the current one, else raise."""
        new_committed = tuple(new_committed)
        if not is_prefix(self.committed, new_committed):
            raise CommitViolationError(
                f"commit {new_committed} does not extend displayed prefix "
                f"{self.committed}")
        self.committed = new_committed
        self.history.append(new_committed)
        # the committed portion is removed from the front of the revisable suffix.

    def commitment_errors(self, reference: Sequence[int]) -> int:
        """Committed tokens disagreeing with a reference transcript (Doc-01 reuse)."""
        return commitment_error_count(self.committed, reference)

    def certify(self) -> Tuple[bool, Optional[int]]:
        """The committed history is append-only (nothing displayed was changed)."""
        return certify_commit_monotone(self.history)
