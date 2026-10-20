"""Grammatical features of an English sentence that ASL marks non-manually or spatially.

Deterministic, so they are equally available in training and at generation time:

* ``wh_question`` — ends in "?" and its first content word is a wh-word (ASL: brow
  furrow over the wh-phrase/clause, Dachkovsky et al. 2013);
* ``yes_no_question`` — ends in "?" and opens with an auxiliary (ASL: brow raise, head
  forward);
* ``other_question`` — any other "?" (tag, echo or rhetorical);
* ``negation`` — not / n't / never / no / nobody / nothing / none / nowhere / neither /
  nor / cannot (ASL: headshake spreading over the negated VP);
* ``conditional`` — if / unless (ASL: brow raise and head held over the clause);
* ``reported_speech`` — a quotation or a verb of saying with a quotation or colon
  (ASL: role shift / constructed dialogue);
* ``third_person`` — count of third-person pronouns (referential load: loci and
  indexing), capped;
* ``deixis`` — here / there / this / that / these / those (pointing propensity).

Discourse openers ("so", "and", "now", "okay", ...) are skipped before the first-word
tests, so "So what do you do next?" is a wh-question.
"""
from __future__ import annotations

import re

import numpy as np

WH = {"what", "where", "when", "why", "who", "whom", "whose", "which", "how"}
AUX = {"do", "does", "did", "is", "are", "was", "were", "am", "can", "could", "will", "would", "should", "shall",
       "have", "has", "had", "may", "might", "must", "isn't", "aren't", "don't", "doesn't", "didn't", "can't",
       "won't", "wouldn't", "shouldn't", "couldn't", "haven't", "hasn't", "wasn't", "weren't"}
OPENERS = {"so", "and", "but", "now", "okay", "ok", "well", "then", "also", "or", "alright", "right", "um", "uh",
           "oh", "yes", "yeah", "first", "next"}
NEGATION = {"not", "never", "no", "nobody", "nothing", "none", "nowhere", "neither", "nor", "cannot", "noone"}
CONDITIONAL = {"if", "unless"}
SAYING = {"say", "says", "said", "tell", "tells", "told", "ask", "asks", "asked", "reply", "replied", "answer",
          "answered", "exclaim", "exclaimed", "think", "thought"}
THIRD = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs", "himself", "herself",
         "themselves"}
DEIXIS = {"here", "there", "this", "that", "these", "those"}
FEATURES = ("wh_question", "yes_no_question", "other_question", "negation", "conditional", "reported_speech",
            "third_person", "deixis")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words(sentence: str) -> list[str]:
    return _WORD.findall(sentence.lower().replace("’", "'"))


def sentence_features(sentence: str) -> dict:
    w = words(sentence)
    question = sentence.strip().endswith("?")
    content = [x for x in w if x not in OPENERS]
    first = content[0] if content else ""
    wh = question and (first in WH or (len(content) > 1 and first in {"and", "but"} and content[1] in WH))
    yn = question and not wh and first in AUX
    neg = any(x in NEGATION or x.endswith("n't") for x in w)
    quote = bool(re.search(r"[\"“”]", sentence)) or (any(x in SAYING for x in w) and ":" in sentence)
    return dict(wh_question=bool(wh), yes_no_question=bool(yn), other_question=bool(question and not wh and not yn),
                negation=bool(neg), conditional=any(x in CONDITIONAL for x in w), reported_speech=bool(quote),
                third_person=min(sum(x in THIRD for x in w), 4), deixis=min(sum(x in DEIXIS for x in w), 4))


def feature_vector(sentence: str) -> np.ndarray:
    """(8,) float vector in ``FEATURES`` order; counts scaled to [0, 1] by their cap of 4."""
    f = sentence_features(sentence)
    return np.array([float(f[k]) / (4. if k in ("third_person", "deixis") else 1.) for k in FEATURES])


def token_features(tokens, *, pad: int = 256):
    """(B, 8) features of byte-token batches (``generation.text_to_motion.encode_texts``), as a float tensor.

    Bytes (ids < 256) are decoded back to the sentence, so the conditioning depends
    only on the text and is identical in training, at generation, and for the empty
    classifier-free-guidance text (all zeros).
    """
    import torch
    rows = []
    for row in tokens.tolist():
        data = bytes(t for t in row if t < pad).decode("utf-8", errors="ignore")
        rows.append(feature_vector(data) if data.strip() else np.zeros(len(FEATURES)))
    return torch.tensor(np.array(rows), dtype=torch.float32, device=tokens.device)


__all__ = ["FEATURES", "sentence_features", "feature_vector", "token_features", "words"]
