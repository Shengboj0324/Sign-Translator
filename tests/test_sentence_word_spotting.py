"""Fix 5: sentence word spotting's scoring (labels, windows, detections, recall, order) on synthetic predictions."""
import numpy as np
import pytest

from scripts.sentence_word_spotting import (
    detections, order_agreement, recall, sentence_labels, summarise, window_bounds)

LABELS = ["HOUSE", "BOOK", "RED", "DOG", "GO"]
INDEX = {g: i for i, g in enumerate(LABELS)}


def test_sentence_labels_upper_case_known_words_once_in_order():
    assert sentence_labels("The dog's red book, the DOG, a house.", set(LABELS)) == ["RED", "BOOK", "DOG", "HOUSE"]
    assert sentence_labels("nothing here", set(LABELS)) == []


def test_window_bounds_lengths_hop_and_short_clips():
    b = window_bounds(48, 24., windows=(.5,), hop=.25)                # 12-frame windows every 6 frames
    assert [(s, e) for s, e, _ in b] == [(s, s + 12) for s in range(0, 37, 6)]
    assert b[0][2] == pytest.approx(6 / 24.)
    assert [(s, e) for s, e, _ in window_bounds(10, 24., windows=(.6, 1.2), hop=.1)] == [(0, 10)]


def test_detections_take_the_best_scoring_window():
    ids = np.array([[0, 1], [1, 3], [2, 1]])                          # (W=3, k=2)
    scores = np.array([[.5, .2], [.6, .3], [.4, .1]])
    found = detections(ids, scores, np.array([.5, 1., 1.5]), ["BOOK", "DOG", "GO"], INDEX)
    assert found == {"BOOK": 1., "DOG": 1.}                           # BOOK best in window 2 (.6); GO never


def test_recall_and_order():
    words = ["HOUSE", "BOOK", "RED", "DOG"]
    found = {"HOUSE": .5, "BOOK": 1.5, "DOG": 1.}
    assert recall(words, found) == pytest.approx(.75)
    # pairs (HOUSE, BOOK) ok, (HOUSE, DOG) ok, (BOOK, DOG) reversed
    assert order_agreement(words, found) == pytest.approx(2 / 3)
    assert order_agreement(words, {"HOUSE": 1., "BOOK": 1.}) == pytest.approx(.5)   # a tie counts half
    assert order_agreement(words, {"HOUSE": 1.}) is None and recall([], {}) is None
    assert order_agreement(words, {w: t for t, w in enumerate(words)}) == 1.
    assert order_agreement(words, {w: -t for t, w in enumerate(words)}) == 0.


def test_a_real_like_form_beats_blank_and_mismatched_texts():
    """End to end on synthetic windows: signing that shows each word in turn is recalled in order; a constant (blank)
    prediction and another sentence's words are not."""
    words = ["HOUSE", "BOOK", "RED"]
    ids = np.array([[0, 4], [1, 4], [2, 4]]); p = np.array([[.9, .1]] * 3); c = np.array([.5, 1., 1.5])
    real = detections(ids, p, c, words, INDEX)
    assert recall(words, real) == 1. and order_agreement(words, real) == 1.
    blank = detections(np.array([[4, 3]] * 3), p, c, words, INDEX)
    assert recall(words, blank) == 0.
    other = detections(ids, p, c, ["DOG"], INDEX)
    assert recall(["DOG"], other) == 0.


def test_summarise_skips_missing_values():
    s = summarise([1., None, 0., 1.])
    assert s["n"] == 3 and s["mean"] == pytest.approx(2 / 3, abs=1e-4) and s["ci95"][0] <= s["mean"] <= s["ci95"][1]
    assert summarise([None]) == dict(mean=None, ci95=None, n=0)
