"""Governance: consent, retention, policy gates, non-inference guard (Doc-10 §7).

Consent is a state machine (GRANTED → WITHDRAWN); withdrawal removes every record
of a signer and retention removes expired records. Policy gates guard derivative
use, identity, commercial use, and redistribution. The sensitive-trait
non-inference guard makes "do not infer sensitive traits" unbreakable in code:
`infer_sensitive_trait` raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .schema import ConsentState, Sample


class ConsentError(RuntimeError):
    """Raised on an illegal consent transition."""


def transition_consent(current: ConsentState, to: ConsentState) -> ConsentState:
    """Only GRANTED → WITHDRAWN is permitted; withdrawal is terminal."""
    if current == to:
        return current
    if current == ConsentState.GRANTED and to == ConsentState.WITHDRAWN:
        return ConsentState.WITHDRAWN
    raise ConsentError(f"illegal consent transition {current.name} -> {to.name}")


def apply_withdrawal(samples: Sequence[Sample],
                     withdrawn_signer: str) -> List[Sample]:
    """Remove EVERY record of a withdrawn signer (right-to-withdraw)."""
    return [s for s in samples if s.signer_id_hash != withdrawn_signer]


def apply_retention(samples: Sequence[Sample], now: float) -> List[Sample]:
    """Drop records past their retention date (None == no expiry set)."""
    return [s for s in samples
            if s.retention_date is None or s.retention_date > now]


# ---------------------------------------------------------------------------
# policy gates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UsagePolicy:
    allow_derivative_models: bool = False
    allow_identity_use: bool = False
    allow_commercial_use: bool = False
    allow_redistribution: bool = False


def gate_action(policy: UsagePolicy, action: str) -> bool:
    """True iff ``action`` is permitted by the policy."""
    field = {
        "derivative_model": "allow_derivative_models",
        "identity_use": "allow_identity_use",
        "commercial_use": "allow_commercial_use",
        "redistribution": "allow_redistribution",
    }.get(action)
    if field is None:
        raise ValueError(f"unknown action {action!r}")
    return getattr(policy, field)


# ---------------------------------------------------------------------------
# sensitive-trait non-inference guard (innovation)
# ---------------------------------------------------------------------------
SENSITIVE_TRAITS = frozenset({
    "race", "ethnicity", "national_origin", "religion", "age", "sex",
    "sexual_orientation", "gender_identity", "immigration_status", "disability",
    "health_condition", "union_membership",
})


class SensitiveInferenceError(RuntimeError):
    """Raised to structurally forbid inferring a sensitive trait."""


def infer_sensitive_trait(sample: Sample, trait: str):
    """Always raises: inferring a sensitive trait from a record is forbidden.

    The guard is structural — there is no code path that returns a prediction —
    so the policy cannot be silently bypassed downstream.
    """
    raise SensitiveInferenceError(
        f"inference of sensitive trait {trait!r} is structurally prohibited")
