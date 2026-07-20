"""Verification of the parameter-stream interface and contracts.

Proves the handedness certificate (det of the contract basis), contract
self-consistency, the validation rules (timestamp monotonicity, finiteness), and
deterministic replay.
"""

import pytest
import torch

from signtranslator.avatar_render.stream import (
    Handedness, AvatarContract, contract_basis, contract_is_self_consistent,
    ParameterStream, validate_stream, replay,
)


def _stream(T=4, J=3, E=2, seed=0, contract=None):
    g = torch.Generator().manual_seed(seed)
    return ParameterStream(
        contract=contract or AvatarContract(),
        timestamps=torch.arange(T, dtype=torch.float64) / 30.0,
        rot6d=torch.randn(T, J, 6, generator=g, dtype=torch.float64),
        gamma=torch.randn(T, 3, generator=g, dtype=torch.float64),
        expr=torch.randn(T, E, generator=g, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# handedness certificate
# ---------------------------------------------------------------------------
def test_right_handed_basis_has_positive_determinant():
    c = AvatarContract(up_axis="y", forward_axis="z", handedness=Handedness.RIGHT)
    assert float(torch.linalg.det(contract_basis(c))) > 0
    assert contract_is_self_consistent(c)


def test_left_handed_basis_has_negative_determinant():
    c = AvatarContract(up_axis="y", forward_axis="z", handedness=Handedness.LEFT)
    assert float(torch.linalg.det(contract_basis(c))) < 0
    assert contract_is_self_consistent(c)


def test_contract_rejects_bad_axes_and_scale():
    with pytest.raises(ValueError):
        AvatarContract(up_axis="y", forward_axis="y")
    with pytest.raises(ValueError):
        AvatarContract(scale_m_per_unit=0.0)
    with pytest.raises(ValueError):
        AvatarContract(frame_rate=-1.0)


# ---------------------------------------------------------------------------
# stream validation
# ---------------------------------------------------------------------------
def test_valid_stream_passes():
    assert validate_stream(_stream()) == []


def test_non_monotone_timestamps_rejected():
    s = _stream()
    s.timestamps[2] = s.timestamps[1]                        # duplicate -> not strictly increasing
    assert "timestamps_not_strictly_increasing" in validate_stream(s)


def test_non_finite_parameters_rejected():
    s = _stream()
    s.rot6d[0, 0, 0] = float("nan")
    assert "non_finite_parameters" in validate_stream(s)


def test_shape_validation():
    with pytest.raises(ValueError):
        ParameterStream(AvatarContract(), torch.arange(4.0), torch.zeros(4, 3, 6),
                        torch.zeros(4, 2), torch.zeros(4, 2))   # gamma wrong shape


# ---------------------------------------------------------------------------
# deterministic replay
# ---------------------------------------------------------------------------
def test_deterministic_replay():
    s = _stream(seed=1)

    def render(i):
        # a pure render_fn: a deterministic function of the frame parameters
        return s.rot6d[i].sum() + s.gamma[i].mean()

    out1 = replay(s, render)
    out2 = replay(s, render)
    assert torch.equal(out1, out2)                           # byte-identical
    assert out1.shape == (s.num_frames,)
