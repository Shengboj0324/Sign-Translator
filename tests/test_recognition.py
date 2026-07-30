"""Tests for the CTC sign-recognition branch."""

import torch

from signtranslator.skeleton import SkeletonGraph
from signtranslator.models.stgcn import STGCNEncoder
from signtranslator.models.recognition import (
    SignRecognizer, ctc_greedy_decode, word_error_rate, _levenshtein,
)


def _recognizer(num_glosses=10):
    g = SkeletonGraph()
    enc = STGCNEncoder(in_channels=3, adjacency=g.adjacency(), channels=(16, 32))
    return g, SignRecognizer(enc, num_glosses=num_glosses)


def test_encoder_sequence_output_shape():
    g = SkeletonGraph()
    enc = STGCNEncoder(in_channels=3, adjacency=g.adjacency(), channels=(16, 32))
    x = torch.randn(4, 3, 24, g.num_nodes)
    seq = enc(x, return_sequence=True)
    assert seq.shape == (4, 24, 32)


def test_recognizer_logprob_shape_and_normalisation():
    g, rec = _recognizer(num_glosses=10)
    x = torch.randn(3, 3, 20, g.num_nodes)
    lp = rec(x)
    assert lp.shape == (3, 20, 11)  # 10 glosses + blank
    # log-probabilities: exp sums to 1 across classes.
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3, 20), atol=1e-5)


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    # Hand-built path: [a, a, blank, a, b, b] with blank=0, a=1, b=2.
    T, C = 6, 3
    path = [1, 1, 0, 1, 2, 2]
    log_probs = torch.full((1, T, C), -10.0)
    for i, c in enumerate(path):
        log_probs[0, i, c] = 0.0  # argmax at c
    decoded = ctc_greedy_decode(log_probs)[0]
    assert decoded == [1, 1, 2]  # a a b (the two a's are separated by blank)


def test_ctc_greedy_decode_all_blank_is_empty():
    log_probs = torch.full((1, 5, 3), -10.0)
    log_probs[0, :, 0] = 0.0  # always blank
    assert ctc_greedy_decode(log_probs)[0] == []


def test_levenshtein_and_wer():
    assert _levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert _levenshtein([1, 2, 3], [1, 3]) == 1        # one deletion
    assert _levenshtein([1, 2], [1, 2, 4, 5]) == 2     # two insertions
    wer = word_error_rate([[1, 2, 3]], [[1, 3]])
    assert abs(wer - 0.5) < 1e-9  # 1 edit / 2 reference tokens


def test_ctc_loss_is_finite_and_decreases_on_overfit():
    torch.manual_seed(0)
    g, rec = _recognizer(num_glosses=6)
    n, T = 4, 24
    pose = torch.randn(n, 3, T, g.num_nodes)
    targets = torch.randint(1, 7, (n, 3))            # 3 glosses each, ids in 1..6
    target_lengths = torch.full((n,), 3, dtype=torch.long)

    opt = torch.optim.Adam(rec.parameters(), lr=5e-3)
    first = None
    for step in range(40):
        loss = rec.loss(pose, targets, target_lengths)
        assert torch.isfinite(loss)
        if first is None:
            first = loss.detach().item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.detach().item() < first * 0.7  # clearly learning the alignment
