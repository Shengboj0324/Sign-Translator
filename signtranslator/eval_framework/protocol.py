"""Pre-registration lock + test-set firewall (Doc-12 §4).

Primary endpoints and minimum effects are hash-locked before test access. The
firewall refuses hyperparameter selection on the test split and refuses to report a
non-registered endpoint as primary; the test set is signer/source-held-out via the
Doc-10 grouped split. The protocol is enforced in code, not merely documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

    @staticmethod
    def create(primary_endpoints: Sequence[str],
               min_effects: Dict[str, float]) -> "PreRegistration":
        if not primary_endpoints:
            raise ValueError("must register at least one primary endpoint")
        for e in primary_endpoints:
            if e not in min_effects:
                raise ValueError(f"endpoint {e!r} needs a registered min effect")
        items = tuple(sorted((k, float(v)) for k, v in min_effects.items()))
        return PreRegistration(tuple(primary_endpoints), items)

    @property
    def registration_hash(self) -> str:
        """A content hash locking the pre-registration (tamper-evident)."""
        return content_hash({"endpoints": sorted(self.primary_endpoints),
                             "min_effects": [list(x) for x in self.min_effects]})

    def is_primary(self, name: str) -> bool:
        return name in self.primary_endpoints

    def min_effect(self, name: str) -> float:
        for k, v in self.min_effects:
            if k == name:
                return v
        raise KeyError(name)


@dataclass
class EvaluationFirewall:
    """Guards the held-out test set against selection bias."""

    prereg: PreRegistration
    _test_accessed: bool = field(default=False)

    def select_hyperparameters(self, split: str) -> None:
        """Permit tuning on train/val only; selecting on test is a violation."""
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
        self._test_accessed = True

    def endpoint_confirmed(self, name: str, effect: float, pvalue: float,
                           alpha: float = 0.05) -> bool:
        """A registered endpoint is confirmed iff significant AND >= its min effect."""
        self.report_primary(name)
        return significant_and_meaningful(effect, self.prereg.min_effect(name),
                                          pvalue, alpha)


def signer_held_out_split(samples, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                          seed: int = 0):
    """Signer/source-held-out split with a leakage certificate (Doc-10 reuse)."""
    assignment = grouped_split(samples, ratios, seed=seed)
    cert = certify_no_group_leakage(samples, assignment)
    if not cert.certified:
        raise ProtocolError(f"test split leaks signers/sources: {cert.offending_groups}")
    return assignment, cert
