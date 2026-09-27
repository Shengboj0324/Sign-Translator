"""Pre-registration lock + test-set firewall (Doc-12 §4).

Primary endpoints and minimum effects are hash-locked before test access. The
firewall refuses hyperparameter selection on the test split and refuses to report a
non-registered endpoint as primary; the test set is signer/source-held-out via the
Doc-10 grouped split. The protocol is enforced in code, not merely documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple
import math
from numbers import Real

from ..data_engineering.provenance import content_hash
from ..data_engineering.splitting import (
    grouped_split, certify_no_group_leakage,
)
from .statistics import significant_and_meaningful


class ProtocolError(RuntimeError):
    """Raised on a protocol violation (test peeking, unregistered endpoint)."""


@dataclass(frozen=True)
class PreRegistration:
    """A hash-locked declaration of primary endpoints + minimum meaningful effects."""

    primary_endpoints: Tuple[str, ...]
    min_effects: Tuple[Tuple[str, float], ...]     # sorted (name, min_effect)
    family_alpha: float = 0.05

    def __post_init__(self) -> None:
        if (not isinstance(self.primary_endpoints, tuple) or not self.primary_endpoints
                or any(not isinstance(e, str) or not e.strip() for e in self.primary_endpoints)
                or len(set(self.primary_endpoints)) != len(self.primary_endpoints)):
            raise ValueError('primary endpoints must be unique nonempty names in a tuple')
        if (not isinstance(self.min_effects, tuple)
                or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in self.min_effects)):
            raise ValueError('minimum effects must be an immutable tuple of name/value pairs')
        names = [name for name, _ in self.min_effects]
        if len(names) != len(set(names)) or set(names) != set(self.primary_endpoints):
            raise ValueError('minimum effects must match the exact primary endpoint family')
        for _, value in self.min_effects:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError('minimum improvements must be finite nonnegative real values')
        if (isinstance(self.family_alpha, bool) or not isinstance(self.family_alpha, Real)
                or not math.isfinite(self.family_alpha) or not 0 < self.family_alpha < 1):
            raise ValueError('family_alpha must be finite and strictly between zero and one')

    @staticmethod
    def create(primary_endpoints: Sequence[str],
               min_effects: Dict[str, float], *, family_alpha: float = 0.05) -> "PreRegistration":
        if not primary_endpoints:
            raise ValueError("must register at least one primary endpoint")
        for e in primary_endpoints:
            if e not in min_effects:
                raise ValueError(f"endpoint {e!r} needs a registered min effect")
        # Preserve input types until validation; bool/strings must not be coerced.
        items = tuple(sorted(min_effects.items()))
        return PreRegistration(tuple(primary_endpoints), items, family_alpha)

    @property
    def registration_hash(self) -> str:
        """A content hash locking the pre-registration (tamper-evident)."""
        return content_hash({"endpoints": sorted(self.primary_endpoints),
                             "min_effects": [[name, float(value)] for name, value in sorted(self.min_effects)],
                             "family_alpha": float(self.family_alpha),
                             "multiplicity": "bonferroni", "effect_direction": "positive_improvement"})

    def is_primary(self, name: str) -> bool:
        return name in self.primary_endpoints

    def min_effect(self, name: str) -> float:
        for k, v in self.min_effects:
            if k == name:
                return v
        raise KeyError(name)


@dataclass
class EvaluationFirewall:
    """In-process workflow guard; not a persistent access-control security boundary.

    A study must persist its protocol/access record outside this object. Creating
    another object cannot make an already inspected test population fresh.
    """

    prereg: PreRegistration
    _test_accessed: bool = field(default=False, init=False)
    _opened_registration_hash: str | None = field(default=None, init=False)

    def access_test(self) -> None:
        current = self.prereg.registration_hash
        if self._opened_registration_hash is not None and current != self._opened_registration_hash:
            raise ProtocolError('registration changed after test access')
        self._opened_registration_hash = current
        self._test_accessed = True

    def select_hyperparameters(self, split: str) -> None:
        """Permit tuning on train/val only; selecting on test is a violation."""
        if self._test_accessed:
            raise ProtocolError('test already accessed; tuning requires a new independently reserved test population')
        if split == "test":
            raise ProtocolError(
                "hyperparameter selection on the test split is forbidden")
        if split not in ("train", "val"):
            raise ValueError("split must be train/val/test")

    def report_primary(self, name: str) -> None:
        """A metric may be reported as PRIMARY only if it was pre-registered."""
        if not self.prereg.is_primary(name):
            raise ProtocolError(
                f"{name!r} was not pre-registered as a primary endpoint")
        self.access_test()

    def endpoint_confirmed(self, name: str, effect: float, pvalue: float,
                           alpha: float | None = None) -> bool:
        """A registered endpoint is confirmed iff significant AND >= its min effect."""
        if alpha is not None and (isinstance(alpha, bool) or alpha != self.prereg.family_alpha):
            raise ProtocolError('alpha must equal the preregistered family alpha')
        self.report_primary(name)
        # Bonferroni controls the declared family without an independence assumption.
        meaningful = significant_and_meaningful(
            effect, self.prereg.min_effect(name), pvalue,
            self.prereg.family_alpha / len(self.prereg.primary_endpoints))
        # General magnitude tests admit negative effects; this registration
        # explicitly promises positive improvement, so harm cannot pass.
        return meaningful and effect > 0


def signer_held_out_split(samples, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                          seed: int = 0):
    """Signer/source-held-out split with a leakage certificate (Doc-10 reuse)."""
    assignment = grouped_split(samples, ratios, seed=seed)
    cert = certify_no_group_leakage(samples, assignment)
    if not cert.certified:
        raise ProtocolError(f"test split leaks signers/sources: {cert.offending_groups}")
    return assignment, cert
