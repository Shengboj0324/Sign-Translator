"""Word retrieval ranks: ties count against the own word; the shuffled control never keeps a clip's own word."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from word_retrieval_check import derangement, ranks, summary  # noqa: E402


def test_ranks_count_ties_against_the_own_word_and_summary_reads_them():
    scores = np.array([[0., -1., -2.], [-1., -1., -3.], [-5., -1., -2.]])
    r = ranks(scores, np.arange(3))
    assert r.tolist() == [1, 2, 2]
    assert summary(r) == dict(top1=round(1 / 3, 4), top5=1.0, mean_rank=round(5 / 3, 2))


def test_derangement_moves_every_index():
    for seed in range(20):
        p = derangement(7, random.Random(seed))
        assert sorted(p.tolist()) == list(range(7)) and all(i != v for i, v in enumerate(p))
