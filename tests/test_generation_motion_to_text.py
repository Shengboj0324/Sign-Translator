"""Motion -> English back-translator: vocabulary round trip, BLEU, padding invariance, learning a toy mapping."""
import math

import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.motion_to_text import (EOS, MotionToText, MotionToTextConfig, WordVocabulary,
                                                      corpus_bleu, encode_batch)
from tests.test_generation_text_to_motion import config as motion_config, state


def test_vocabulary_round_trip_with_byte_fallback():
    v = WordVocabulary.build(["the cat sat.", "the dog sat!", "the cat ran"], min_count=2)
    assert v.words == ["the", "cat", "sat"]
    ids = v.encode("The cat sat on a Zebra.")
    assert ids[0] == 1 and ids[-1] == EOS
    assert v.decode(ids) == "the cat sat on a zebra."           # unseen words spelled in bytes, case folded
    assert len(v.encode("the cat sat " * 50, max_tokens=10)) == 10


def test_corpus_bleu_on_known_cases():
    same = corpus_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert math.isclose(same["bleu"], 100.)
    assert corpus_bleu(["dog runs fast today"], ["the cat sat on the mat"])["bleu"] == 0.
    short = corpus_bleu(["the cat sat on"], ["the cat sat on the mat"])          # all n-grams match; brevity penalty
    assert math.isclose(short["bleu"], 100 * math.exp(1 - 6 / 4), rel_tol=1e-9)


def _model(states, vocabulary, **kw):
    return MotionToText(MotionToTextConfig(motion_config(states), len(vocabulary), d_model=32, heads=4,
                                           encoder_layers=1, decoder_layers=1, feedforward=64, dropout=0., **kw))


def test_translation_ignores_padding():
    states = [state(0, 20), state(1, 33)]
    v = WordVocabulary.build(["a b c"] * 2)
    model = _model(states, v).eval()
    alone = collate_multichannel([states[0]])
    padded = collate_multichannel(states)
    tokens = encode_batch(v, ["a b c"])
    m1, v1 = model.encode(alone)
    m2, v2 = model.encode(padded)
    l1 = model.decode(m1, v1, tokens)
    l2 = model.decode(m2[:1], v2[:1], tokens)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_learns_a_toy_motion_to_sentence_mapping():
    torch.manual_seed(0)
    states = [state(i, 24) for i in range(3)]
    texts = ["i want coffee", "where is the library", "thank you very much"]
    v = WordVocabulary.build(texts, min_count=1)
    model = _model(states, v)
    batch, tokens = collate_multichannel(states), encode_batch(v, texts)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, terms = model.loss(batch, tokens)
        opt.zero_grad(); loss.backward(); opt.step()
    assert terms["token_accuracy"] > .99
    model.eval()
    for beam in (1, 3):
        decoded = [v.decode(ids) for ids in model.generate(batch, beam=beam, max_tokens=12)]
        assert decoded == texts, (beam, decoded)
    assert corpus_bleu(decoded, texts)["bleu"] > 99
