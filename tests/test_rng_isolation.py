import random

import numpy as np
import pytest
import torch

from signtranslator import reproducibility as rng


def assert_state(expected):
    actual = rng.capture_rng_state()
    assert actual['python'] == expected['python']
    a, b = actual['numpy'], expected['numpy']
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    assert torch.equal(actual['torch_cpu'], expected['torch_cpu'])
    if expected['torch_cuda'] is not None:
        assert all(torch.equal(a, b) for a, b in zip(actual['torch_cuda'], expected['torch_cuda']))


@pytest.mark.parametrize('failure', ['none', 'body', 'seed'])
def test_mps_and_core_rng_restored_after_partial_seeding_or_body_failure(monkeypatch, failure):
    expected = rng.capture_rng_state()
    mps = {'state': torch.tensor([1, 2, 3], dtype=torch.uint8)}
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'get_rng_state', lambda: mps['state'].clone())
    monkeypatch.setattr(torch.mps, 'set_rng_state', lambda state: mps.update(state=state.clone()))
    monkeypatch.setattr(torch.mps, 'manual_seed', lambda seed: None)
    original = rng.seed_all
    def seed(value):
        original(value)
        mps['state'] = torch.tensor([9], dtype=torch.uint8)
        if failure == 'seed':
            raise RuntimeError('seeding failed after mutation')
    monkeypatch.setattr(rng, 'seed_all', seed)
    def invoke():
        with rng.isolated_deterministic_rng(27):
            random.random(); np.random.rand(); torch.rand(2)
            mps['state'] = torch.tensor([8], dtype=torch.uint8)
            if failure == 'body':
                raise RuntimeError('body failed')
    if failure == 'none':
        invoke()
    else:
        with pytest.raises(RuntimeError, match='failed'):
            invoke()
    assert_state(expected)
    assert torch.equal(mps['state'], torch.tensor([1, 2, 3], dtype=torch.uint8))


def test_oversized_seed_restores_python_numpy_and_cpu_state():
    expected = rng.capture_rng_state()
    with pytest.raises((ValueError, RuntimeError, OverflowError)):
        with rng.isolated_deterministic_rng(2**65):
            pytest.fail('invalid torch seed must fail before yielding')
    assert_state(expected)


def test_nested_streams_are_reproducible_and_restore_outer_sequence():
    expected = rng.capture_rng_state()
    def draw():
        return (random.random(), float(np.random.rand()), float(torch.rand(())))
    with rng.isolated_deterministic_rng(51):
        first, second = draw(), draw()
    with rng.isolated_deterministic_rng(51):
        assert draw() == first
        with rng.isolated_deterministic_rng(52):
            draw()
        assert draw() == second
    assert_state(expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS device unavailable')
def test_real_mps_stream_restored_after_success_and_exception():
    saved = torch.mps.get_rng_state().clone()
    try:
        with rng.isolated_deterministic_rng(45):
            first = torch.rand(8, device='mps').cpu()
        assert torch.equal(torch.mps.get_rng_state(), saved)
        with pytest.raises(RuntimeError, match='MPS probe'):
            with rng.isolated_deterministic_rng(45):
                assert torch.equal(torch.rand(8, device='mps').cpu(), first)
                raise RuntimeError('MPS probe')
        assert torch.equal(torch.mps.get_rng_state(), saved)
    finally:
        torch.mps.set_rng_state(saved)
