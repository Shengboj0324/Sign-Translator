"""Compositional baseline for sentence generation (report-only diagnostic, 2026-10-28).

The prior signs isolated words (F12: R5 top-5 0.43 with the recogniser) but its sentences carry no content the
back-translator can read (pilot: generated 1.81 BLEU, blank 1.86, real 4.14). This diagnostic asks whether simply
*composing* the prior's own isolated words carries more sentence content than its end-to-end sentences: the evidence
for (or against) a word- or gloss-level path. It changes nothing and trains nothing.

**Clips:** the pilot's (``pilot_content_check.validation_sentences``: How2Sign val, signer groups c01/c03/c05, seed 0,
``--clips``), real states at 24 fps; the reference duration is ``frames / 24`` as in the pilot.

**Forms** (each back-translated by the pilot's instrument, ``back-translator-kp2d-uwb-ft-v1``, beam 4):
- ``real``, ``blank`` (``evaluate_text_to_motion.blanked``: only the duration left; the translator's text prior);
- ``end_to_end``: the pilot's generated form, by the pilot's own path: ``load_generator`` (accepted decoder gains),
  ``sample.batch`` in batches of 32 with seeds 1000 + i, the pilot's part temperatures and guidance, the reference
  duration, then ``pilot_content_check.postprocess``;
- per word-selection variant V (``--variants``):
  - ``V/composed``: the sentence's selected words in English order. Each word occurrence (clip i, word index j in the
    sentence's token list) is sampled in isolation exactly as the pilot's lexicon samples a word,
    ``postprocess(resample_motion(sample(word, 24., None, guidance, 50, seed), 24.))`` (the generator's own predicted
    duration), with ``seed = 100000 + 1000 i + j``. Variants select from the same clips, so they share samples.
    ``--word-workers`` spreads the words over processes (each with its own copy of the prior on ``--device``); every
    word has its own seeded generator, so the clips do not depend on it (tested). The clips are joined by
    ``concatenate`` (overlap crossfade, below);
  - ``V/composed_shuffled``: the same clips in a random non-identity order (seeded per clip and variant): word
    content without word order;
  - ``V/composed_mismatched``: the clips of another sentence (a seeded derangement of the sentences, separate from the
    reference derangement below): the content control;
  - three timings of each: unscaled (above: every word at its own predicted duration); ``*_scaled``: the whole
    unscaled state time-scaled to the reference clip's frame count (``time_scale``); ``*_fitted`` (the PRIMARY form,
    first in the summary): every word sampled again, *conditioned* on its share of the reference duration through the
    prior's duration input (``sample(word, 24., seconds, ...)``, as end_to_end is conditioned on the reference
    duration): ``fitted_seconds`` gives word k ``max(0.3 s, budget * t_k / sum t)``, ``t_k`` the word's typical isolated
    duration (``typical_seconds``: the mean duration of that word's unscaled clips over the run, not equal shares) and
    ``budget`` the reference duration plus the frames the joins overlap, so the joined clip approximates the
    reference duration at the model's own pace for it (words at the 0.3 s floor make it longer: reported). Fitted
    clips are sampled per variant (the shares depend on the selection), seed ``word seed + 10^7 (1 + variant
    index)``. ``composed_shuffled_fitted`` reorders the same fitted clips; ``composed_mismatched_fitted`` is the other
    sentence's fitted composition (fitted to *its* reference), time-scaled to this clip's frame count (the ratio is
    reported as ``mismatched_fitted_time_scale_median``).
  A sentence with no selected word reads as its ``blank`` clip in every composed form of that variant (counted).
- Variants: ``all`` (every token), ``content`` (``STOP_WORDS`` removed: articles, forms of be / do / have, "to",
  "of"), ``recognizer`` (tokens whose upper case is a label of ``--recognizer-run``).
Tokens: the lower-cased text's ``[a-z0-9]+('[a-z]+)*`` matches (contractions kept whole, each occurrence counted).

**Joins (``concatenate``).** Consecutive clips overlap by ``n = round(--crossfade * 24)`` frames (default 0.15 s ->
4 frames; at most half of either clip): clip A's last n frames and clip B's first n frames become n frames
``(1 - a_k) A + a_k B`` with ``a_k = (k + 1) / (n + 1)``, k = 0..n-1 (strictly inside (0, 1), so the joins
interpolate between the last pure frame of A and the first pure frame of B). Per channel kind: translations,
coefficients and bounded scalars linearly; unit vectors linearly then renormalised; 6D rotations linearly in 6D then
Gram-Schmidt re-orthonormalised (the ``pose.resample`` convention), falling back to the heavier endpoint where the blend
degenerates. A frame valid on one side only takes that side's value; valid on neither stays invalid (zero payload).
Observed only where every contributing side is; confidence the weighted mean. The result is on a 24 fps clock;
its length is the sum of the clip lengths minus the overlaps. No post-processing after the join (each word clip was
post-processed as the pilot's lexicon post-processes it).

**Scores** (``sentence_exposure_diagnostic.score_forms``): corpus BLEU with the pilot's bootstrap 95% interval
(exactly ``strict_prior_review.bleu_ci``, from per-sentence counts); a mismatched-reference control (each form's
hypotheses against a seeded derangement of the references, ``--seed``); paired differences (the same resampled
sentences on both sides) against that control, against blank, against ``end_to_end``, and for composed forms against
their shuffled and mismatched-words controls of the same timing and their other two timings. Hypothesis length over reference length is reported (BLEU's
brevity penalty reads it).

**Motion sanity** per form, in the back-translator's own 2D view: duration (and over the reference), wrist speed,
hand-tracking fraction, wrist spread, the fraction of clips over the translator's 20 s training window.

``--reproduce-key`` (optional): a pilot report (``meaning.generated``), a sentence-exposure report
(``whole.free_run``) or a report of this script (``forms.end_to_end``) on the same clips; the fraction of identical
end-to-end hypotheses is reported (the pilot's F12 report has 1000 clips: compare with ``--clips 1000``).

Writes one JSON report to ``--output-key`` (or ``--output-file`` for local checks; both write-once) and prints a
summary.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

FPS = 24.
TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"
TRANSLATOR_WINDOW = 20.            # seconds: train_keypoints_to_text_t5.py --max-seconds default (training clips cut)
WORD_SEED = 100_000                # word clip seed: WORD_SEED + MAX_TOKENS * clip + word index
MAX_TOKENS = 1000
SHUFFLE_SEED = 7_000               # numpy seed sequence (SHUFFLE_SEED, clip, variant index)
FITTED_SEED_BLOCK = 10_000_000     # fitted clip seed: word seed + FITTED_SEED_BLOCK (1 + variant index)
FITTED_FLOOR = .3                  # seconds: the shortest conditioned word duration
VARIANTS = ("all", "content", "recognizer")
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")
# English function words ASL does not usually sign as separate lexical items. Modals (can, will, should...) are kept:
# ASL signs them. Main-verb uses of have / do are dropped too (a known cost of a list).
STOP_WORDS = frozenset({
    "a", "an", "the",                                            # articles
    "am", "is", "are", "was", "were", "be", "been", "being",     # be
    "do", "does", "did",                                         # do
    "have", "has", "had",                                        # have
    "to", "of",
})
MOTION_KEYS = ("seconds", "seconds_over_reference", "wrist_speed", "hands_tracked", "wrist_spread", "over_window")

CAVEATS = [
    "Word order: composed forms sign the English words in English order. How2Sign is interpreted from English and the "
    "translator was fine-tuned on it, so an English-order reading may favour composed forms over real ASL order. "
    "composed_shuffled (same clips, other order) measures how much of any gain is order.",
    "Duration: each word takes the prior's predicted isolated-sign duration, so unscaled composed clips are usually much "
    "longer than the reference (reported: seconds_over_reference). Longer input can draw longer hypotheses, which BLEU's "
    "brevity penalty rewards (reported: hypothesis_over_reference_length). Clips over the translator's 20 s training "
    "window (over_window) are inputs it never saw. The scaled forms remove the duration difference but sign faster than "
    "any real signer (the speed-up is the duration ratio). A local F12 check (3 clips, CPU) gave word clips of 1.7 s, "
    "unscaled composed clips about 5x the reference and scaled wrist speeds about 10x real: neither form is natural "
    "signing speed. Hypotheses are capped at the translator's 64 output tokens whatever the input length.",
    "The translator's text prior: blank reads about 1.86 BLEU on these clips; frequent How2Sign words (I, you, going, "
    "to, the) match whatever the input. BLEU is near its floor: read paired differences and the mismatched controls, "
    "not point gaps.",
    "Joins: a 4-frame linear crossfade between unrelated end and start poses. Isolated-sign clips tend to start and end "
    "near rest, so composed signing dips towards rest between words and has no co-articulation; this may make composed "
    "look worse to the translator than a real word-level path would.",
    "Function words: in the all variant the prior is prompted with words that have no ASL sign (the, of, is); what it "
    "produces for them is arbitrary. The content variant drops them; the recognizer variant keeps only ASL Citizen "
    "glosses, the words the prior learnt in isolation (its best case).",
    "Tokens are passed to the prior as written (lower case, contractions whole, digits as digits); words it never saw "
    "alone get whatever the prior makes of them.",
    "Sampling noise: one sample per word occurrence (repeats sampled separately), seeds independent of the pilot's "
    "lexicon seeds.",
    "composed_mismatched reuses the other sentence's word clips: unscaled, it also has that sentence's duration; the "
    "scaled form matches this clip's duration and is the cleaner content control. composed_mismatched_fitted is the "
    "other sentence's fitted composition time-scaled to this clip (a second, usually milder, speed change that the "
    "matched fitted form does not have).",
    "Fitted words: long sentences give many words little time (How2Sign validation sentences average 18 words; the "
    "3-clip local check had 33 words in 11 s, 0.34 s per word); the prior was trained on isolated signs of about 1-2 s and may compress a sign badly, or not at all, "
    "at 0.3-0.6 s. Words at the 0.3 s floor make the fitted clip longer than the reference (fitted_words_at_floor, "
    "fitted_over_reference_median). The shares come from this run's own unscaled samples of the same words.",
    "end_to_end reproduces the pilot's generated form when checkpoint, clips (--clips) and device arithmetic agree; "
    "--reproduce-key checks it.",
    "Instrument: back-translator-kp2d-uwb-ft-v1 reads arms and hands only (48 2D points via FK, no face) and per-point "
    "validity; generated and composed forms have every hand valid on every frame, real How2Sign hands drop out.",
]


# ---- pure logic --------------------------------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    """The text's words, lower case, in order, every occurrence (curly apostrophes read as straight ones)."""
    return WORD.findall((text or "").lower().replace("’", "'"))


def select_words(words, variant: str, labels=None) -> list[int]:
    """Indices into ``words`` that the variant keeps, in order."""
    if variant == "all":
        return list(range(len(words)))
    if variant == "content":
        return [j for j, w in enumerate(words) if w not in STOP_WORDS]
    if variant == "recognizer":
        if labels is None:
            raise ValueError("the recognizer variant needs the recogniser's labels")
        return [j for j, w in enumerate(words) if w.upper() in labels]
    raise ValueError(f"unknown variant {variant!r}")


def word_seed(clip: int, index: int) -> int:
    """The sampling seed of word ``index`` of clip ``clip``'s token list."""
    if not 0 <= index < MAX_TOKENS:
        raise ValueError(f"word index {index} outside [0, {MAX_TOKENS})")
    return WORD_SEED + MAX_TOKENS * clip + index


def shuffle_order(n: int, clip: int, variant_index: int) -> list[int]:
    """A seeded random order of range(n), never the identity when n >= 2 (rejection sampling)."""
    rng = np.random.default_rng([SHUFFLE_SEED, clip, variant_index])
    if n < 2:
        return list(range(n))
    while True:
        p = [int(x) for x in rng.permutation(n)]
        if p != list(range(n)):
            return p


def crossfade_frames(seconds: float, fps: float = FPS) -> int:
    if seconds < 0:
        raise ValueError("crossfade must be >= 0 seconds")
    return int(round(seconds * fps))


def join_overlaps(lengths, frames: int) -> list[int]:
    """Overlap of each join: ``frames``, capped at half of either neighbour (so no clip is blended at both ends over
    the same frame, and a clip of one frame is hard-cut)."""
    return [min(frames, a // 2, b // 2) for a, b in zip(lengths[:-1], lengths[1:])]


def crossfade_weights(n: int) -> np.ndarray:
    """The second clip's weight on the n overlapping frames: (k + 1) / (n + 1)."""
    return (np.arange(n, dtype=np.float64) + 1) / (n + 1)


def _gram_schmidt(x: np.ndarray, eps: float = 1e-6):
    """(orthonormalised 6D columns, degenerate mask) of (..., 6) values."""
    a, b = x[..., 0:3], x[..., 3:6]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    a = a / np.clip(na, eps, None)
    b = b - (a * b).sum(-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    b = b / np.clip(nb, eps, None)
    return np.concatenate([a, b], -1), (na[..., 0] < eps) | (nb[..., 0] < eps)


def blend_values(kind: str, a: np.ndarray, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(n, J, D) values on each channel's manifold: ``(1 - alpha) a + alpha b``, put back on the manifold; where that
    degenerates (antipodal unit vectors or rotations) the endpoint with the larger weight. ``alpha``: (n,)."""
    w = alpha.reshape(-1, *([1] * (a.ndim - 1)))
    v = (1 - w) * a + w * b
    if kind == "rotation_6d_columns":
        v, bad = _gram_schmidt(v)
    elif kind == "unit_vector":
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        bad = n[..., 0] < 1e-6
        v = v / np.clip(n, 1e-6, None)
    else:
        return np.clip(v, 0., 1.) if kind == "bounded_scalar" else v
    nearer = np.where(w > .5, b, a)
    return np.where(bad[..., None], nearer, v)


def blend_channel(kind: str, ca, cb, alpha: np.ndarray) -> dict:
    """values, valid, observed, confidence of the crossfade of two channels' overlapping frames (same shapes)."""
    va, vb = ca.valid, cb.valid
    w = alpha[:, None]
    valid = va | vb
    values = blend_values(kind, ca.values, cb.values, alpha)
    values = np.where((va & ~vb)[..., None], ca.values, values)          # one side valid: that side, exactly
    values = np.where((vb & ~va)[..., None], cb.values, values)
    values = np.where(valid[..., None], values, 0.)
    wa, wb = (1 - w) * va, w * vb
    confidence = np.where(valid, (wa * ca.confidence + wb * cb.confidence) / np.where(valid, wa + wb, 1.), 0.)
    observed = valid & (~va | ca.observed) & (~vb | cb.observed)
    return dict(values=values, valid=valid, observed=observed, confidence=confidence)


def concatenate(states, overlap: int, *, fps: float = FPS, sample_id: str = "compositional"):
    """One state from ``states`` in order, joined by the overlap crossfade (module docstring)."""
    from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion
    if not states:
        raise ValueError("nothing to concatenate")
    lengths = [len(s.timestamps) for s in states]
    overlaps = join_overlaps(lengths, overlap)
    total = sum(lengths) - sum(overlaps)
    first = states[0]
    channels = {}
    for name, c0 in first.channels.items():
        kind = CHANNELS[name][0]
        group = [s.channels[name] for s in states]
        for c in group[1:]:
            for field in ("labels", "coordinate_frame", "units", "convention"):
                if getattr(c, field) != getattr(c0, field):
                    raise ValueError(f"{name}: clips differ in {field}")
        parts = {k: [] for k in ("values", "valid", "observed", "confidence")}
        for k, c in enumerate(group):
            start = overlaps[k - 1] if k else 0
            stop = lengths[k] - (overlaps[k] if k < len(group) - 1 else 0)
            for f in parts:
                parts[f].append(getattr(c, f)[start:stop])
            if k < len(group) - 1 and overlaps[k]:
                n, nxt = overlaps[k], group[k + 1]
                tail = replace(c, values=c.values[-n:], valid=c.valid[-n:], observed=c.observed[-n:],
                               confidence=c.confidence[-n:])
                head = replace(nxt, values=nxt.values[:n], valid=nxt.valid[:n], observed=nxt.observed[:n],
                               confidence=nxt.confidence[:n])
                for f, v in blend_channel(kind, tail, head, crossfade_weights(n)).items():
                    parts[f].append(v)
        valid = np.concatenate(parts["valid"])
        observed = np.concatenate(parts["observed"]) & valid
        methods = [c.inference_method for c in group if c.inference_method]
        method = methods[0] if methods else ("compositional crossfade" if (valid & ~observed).any() else None)
        channels[name] = replace(
            c0, values=np.where(valid[..., None], np.concatenate(parts["values"]), 0.).astype(c0.values.dtype),
            valid=valid, observed=observed,
            confidence=np.where(valid, np.concatenate(parts["confidence"]), 0.).astype(c0.confidence.dtype),
            source_id=f"compositional:{c0.source_id}",
            source_sha256=hashlib.sha256("|".join(c.source_sha256 for c in group).encode()).hexdigest(),
            inference_method=method)
        if len(valid) != total:
            raise AssertionError("concatenation length")
    timestamps = float(first.timestamps[0]) + np.arange(total, dtype=np.float64) / fps
    return MultichannelMotion(timestamps, first.clock_id, sample_id, channels)


def take_frames(state, index):
    """The frames ``index`` (an index array) of ``state`` as a new state on a uniform 24 fps clock from its start."""
    channels = {n: replace(c, values=c.values[index], valid=c.valid[index], observed=c.observed[index],
                           confidence=c.confidence[index]) for n, c in state.channels.items()}
    t = float(state.timestamps[0]) + np.arange(len(index), dtype=np.float64) / FPS
    return replace(state, timestamps=t, channels=channels)


def time_scale(state, frames: int, fps: float = FPS):
    """``state`` stretched or compressed in time to exactly ``frames`` frames at ``fps`` (``pose.resample``'s
    interpolation on a stretched clock: the first and last frames are kept, everything between is interpolated)."""
    from signtranslator.pose.resample import resample_motion
    T = len(state.timestamps)
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if T == frames:
        return state
    if frames == 1:
        return take_frames(state, np.array([0]))
    if T == 1:
        state = take_frames(state, np.array([0, 0]))
    t0 = float(state.timestamps[0])
    stretched = replace(state, timestamps=t0 + np.linspace(0., (frames - 1) / fps, len(state.timestamps)))
    out = resample_motion(stretched, fps)
    if len(out.timestamps) != frames:
        raise AssertionError(f"time_scale produced {len(out.timestamps)} frames, not {frames}")
    return out


def frames_of(state) -> int:
    return len(state.timestamps)


def fitted_seed(clip: int, index: int, variant_index: int) -> int:
    """The seed of a fitted word clip: the word's seed moved to a block of its own per variant (the fitted durations
    depend on the variant's word selection, so each variant samples its own fitted clips)."""
    return word_seed(clip, index) + FITTED_SEED_BLOCK * (1 + variant_index)


def typical_seconds(words, word_clips: dict) -> dict:
    """{word: mean duration (s) of its unscaled clips} over every sampled occurrence in the run: the word's typical
    isolated duration at the prior's own pace."""
    total, count = {}, {}
    for (i, j), clip in word_clips.items():
        w = words[i][j]
        total[w] = total.get(w, 0.) + frames_of(clip) / FPS
        count[w] = count.get(w, 0) + 1
    return {w: total[w] / count[w] for w in total}


def fitted_seconds(reference_seconds: float, typical, overlap: int, *, floor: float = FITTED_FLOOR,
                   fps: float = FPS) -> list[float]:
    """Conditioned durations of a sentence's words, in order: ``max(floor, budget * share)`` with shares proportional
    to the words' ``typical`` durations and ``budget`` the reference duration plus the frames the joins overlap
    (``overlap`` per join), so that the joined clip approximates the reference duration. Words at the floor make the
    total longer than the reference."""
    typical = [float(t) for t in typical]
    if not typical:
        return []
    if min(typical) <= 0:
        raise ValueError("typical durations must be positive")
    budget = reference_seconds + (len(typical) - 1) * overlap / fps
    return [max(floor, budget * t / sum(typical)) for t in typical]


def compose_forms(word_clips: dict, selected: list, refs: list, blanks: list, content_perm: list, overlap: int,
                  variant_index: int, fitted_clips: dict | None = None) -> tuple[dict, list]:
    """The composed forms of one variant (``word_clips[(i, j)]`` the unscaled clips, ``selected[i]`` the variant's word
    indices of clip i, ``fitted_clips[(i, j)]`` the variant's fitted clips) and per-clip rows (shuffled order,
    mismatched source, frames). The fitted forms come first. ``composed_mismatched_fitted`` is the other sentence's
    fitted composition (fitted to that sentence's reference) time-scaled to this clip's frame count."""
    forms = {k: [] for k in ("composed", "composed_scaled", "composed_shuffled", "composed_shuffled_scaled",
                             "composed_mismatched", "composed_mismatched_scaled")}
    if fitted_clips is not None:
        forms = dict({k: [] for k in ("composed_fitted", "composed_shuffled_fitted", "composed_mismatched_fitted")},
                     **forms)
        fitted = [concatenate([fitted_clips[(i, j)] for j in idx], overlap, sample_id=f"fitted:{i}") if idx else None
                  for i, idx in enumerate(selected)]
    composed, rows = [], []
    for i, idx in enumerate(selected):
        composed.append(concatenate([word_clips[(i, j)] for j in idx], overlap, sample_id=f"composed:{i}")
                        if idx else None)
    for i, idx in enumerate(selected):
        frames = frames_of(refs[i])
        order = shuffle_order(len(idx), i, variant_index)
        other = content_perm[i]
        if idx:
            shuffled = concatenate([word_clips[(i, idx[k])] for k in order], overlap, sample_id=f"shuffled:{i}")
            forms["composed"].append(composed[i])
            forms["composed_scaled"].append(time_scale(composed[i], frames))
            forms["composed_shuffled"].append(shuffled)
            forms["composed_shuffled_scaled"].append(time_scale(shuffled, frames))
        else:
            for k in ("composed", "composed_scaled", "composed_shuffled", "composed_shuffled_scaled"):
                forms[k].append(blanks[i])
        if composed[other] is not None:
            forms["composed_mismatched"].append(composed[other])
            forms["composed_mismatched_scaled"].append(time_scale(composed[other], frames))
        else:
            forms["composed_mismatched"].append(blanks[i])
            forms["composed_mismatched_scaled"].append(blanks[i])
        row = dict(words=list(idx), shuffled_order=order, mismatched_from=other,
                   frames_composed=frames_of(forms["composed"][-1]),
                   frames_mismatched=frames_of(forms["composed_mismatched"][-1]), empty=not idx)
        if fitted_clips is not None:
            if idx:
                forms["composed_fitted"].append(fitted[i])
                forms["composed_shuffled_fitted"].append(concatenate([fitted_clips[(i, idx[k])] for k in order],
                                                                     overlap, sample_id=f"fitted-shuffled:{i}"))
            else:
                forms["composed_fitted"].append(blanks[i])
                forms["composed_shuffled_fitted"].append(blanks[i])
            forms["composed_mismatched_fitted"].append(time_scale(fitted[other], frames) if fitted[other] is not None
                                                       else blanks[i])
            row.update(frames_fitted=frames_of(forms["composed_fitted"][-1]),
                       frames_mismatched_fitted_source=frames_of(fitted[other]) if fitted[other] is not None else None)
        rows.append(row)
    return forms, rows


TIMINGS = {"_fitted": "fitted", "": "unscaled", "_scaled": "scaled"}


def form_pairs(variants) -> dict:
    """``score_forms``'s paired comparisons: every composed form against end_to_end and its two other timings; each
    ``composed`` timing also against its shuffled and mismatched-words control of the same timing."""
    pairs = {}
    for v in variants:
        for base in ("composed", "composed_shuffled", "composed_mismatched"):
            for timing in TIMINGS:
                name = f"{v}/{base}{timing}"
                pairs[name] = {"end_to_end": "end_to_end"}
                if base == "composed":
                    pairs[name].update(shuffled=f"{v}/composed_shuffled{timing}",
                                       mismatched_words=f"{v}/composed_mismatched{timing}")
                for other, label in TIMINGS.items():
                    if other != timing:
                        pairs[name][label] = f"{v}/{base}{other}"
    return pairs


def hypothesis_length_ratio(hypotheses, references) -> float:
    """Total hypothesis words over total reference words (``motion_to_text.words`` tokenisation)."""
    from signtranslator.generation.motion_to_text import words
    ref = sum(len(words(r)) for r in references)
    return round(sum(len(words(h)) for h in hypotheses) / ref, 4) if ref else None


def motion_rows(states, refs, fps: float) -> list[dict]:
    """Per clip: duration, its ratio to the reference, the back-translator-view keypoint statistics, over_window."""
    from signtranslator.data.keypoints2d import from_state, normalise
    from evaluate_text_to_motion import at_rate
    from sentence_exposure_diagnostic import keypoint_statistics
    rows = []
    for s, ref in zip(states, refs):
        points, valid = from_state(at_rate(s, fps))
        k = keypoint_statistics(normalise(points, valid, y_down=False), valid, fps)
        seconds = frames_of(s) / FPS
        rows.append(dict(seconds=seconds, seconds_over_reference=seconds / (frames_of(ref) / FPS),
                         wrist_speed=k["wrist_speed"], hands_tracked=k["hands_tracked"],
                         wrist_spread=k["wrist_spread"], over_window=float(seconds > TRANSLATOR_WINDOW)))
    return rows


def reproduction(earlier: dict, references, hypotheses) -> dict:
    """How many end-to-end hypotheses equal an earlier report's generated ones on the same clips."""
    for path in (("meaning", "generated"), ("whole", "free_run"), ("forms", "end_to_end")):
        node = earlier
        for p in path:
            node = node.get(p, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict) and "hypotheses" in node:
            refs = (earlier.get("meaning", {}).get("references") if path[0] == "meaning"
                    else earlier.get("references"))
            if refs != list(references):
                return dict(source=".".join(path), comparable=False, reason="the earlier report read other clips")
            same = [a == b for a, b in zip(node["hypotheses"], hypotheses)]
            return dict(source=".".join(path), comparable=True, identical=int(sum(same)), clips=len(same),
                        identical_fraction=round(sum(same) / len(same), 4) if same else None,
                        earlier_bleu=node.get("bleu"))
    return dict(comparable=False, reason="no generated hypotheses in the earlier report")


# ---- generation (the pilot's paths) --------------------------------------------------------------------------------

def generate_end_to_end(sample, texts, seconds, guidance: float) -> list:
    """The pilot's generated form: batches of 32, seeds 1000 + i, then the pilot's post-processing."""
    from pilot_content_check import postprocess
    out = []
    for i in range(0, len(texts), 32):
        out += [postprocess(o) for o in sample.batch(texts[i:i + 32], FPS, seconds[i:i + 32], guidance, 1000 + i)]
    return out


def generate_word(sample, word: str, guidance: float, seed: int, seconds: float | None = None):
    """One isolated word, exactly as the pilot's lexicon samples one: at its own predicted duration (``seconds``
    None), or conditioned on ``seconds`` (the fitted form: the prior's duration input, as end_to_end is conditioned on
    the reference duration)."""
    from signtranslator.pose.resample import resample_motion
    from pilot_content_check import postprocess
    return postprocess(resample_motion(sample(word, FPS, seconds, guidance, 50, seed), FPS))


_WORKER = {}


def _init_word_worker(store_spec, checkpoint_key, mix_sha256, device, temperature, threads):
    """A word-sampling process: its own store and generator, loaded as the main process loads them."""
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from evaluate_text_to_motion import load_generator
    from train_text_to_motion import make_store
    torch.set_num_threads(threads)
    _WORKER["sample"] = load_generator(make_store(store_spec), checkpoint_key, mix_sha256, device,
                                       temperature=temperature)[2]


def _word_job(job):
    return generate_word(_WORKER["sample"], *job)


def sample_words(sample, jobs, *, workers: int = 1, worker_args=None, progress=None, every: int = 200) -> list:
    """``generate_word`` for each job (word, guidance, seed[, seconds]), in order: in this process with ``sample``, or
    in ``workers`` spawned processes (``_init_word_worker(*worker_args)``). Each word draws from its own seeded
    generator, so the result does not depend on the number of workers."""
    out = []
    if workers <= 1:
        for k, job in enumerate(jobs):
            out.append(generate_word(sample, *job))
            if progress and ((k + 1) % every == 0 or k + 1 == len(jobs)):
                progress(k + 1)
        return out
    import multiprocessing
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_word_worker,
                                                   initargs=worker_args) as pool:
        for state in pool.imap(_word_job, jobs, chunksize=4):
            out.append(state)
            if progress and (len(out) % every == 0 or len(out) == len(jobs)):
                progress(len(out))
    return out


# ---- the run -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    from pilot_content_check import MIX, MIX_SHA
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", default=MIX, help="the manifest the prior was trained on")
    p.add_argument("--mix-sha256", default=MIX_SHA)
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--part-temperature", default="body=1.1,left_hand=1.1,right_hand=1.1,face=1.0")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--crossfade", type=float, default=.15, help="seconds of overlap at each join")
    p.add_argument("--recognizer-run", default="sign-recognizer-v2", help="labels for the recognizer variant")
    p.add_argument("--translator-run", default=TRANSLATOR_RUN)
    p.add_argument("--seed", type=int, default=0, help="the reference derangement (the content derangement: seed + 1)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--reproduce-key", help="an earlier report on the same clips whose generated hypotheses end_to_end "
                   "should reproduce")
    p.add_argument("--device", default="cuda")
    p.add_argument("--word-workers", type=int, default=1, help="processes sampling word clips (each loads the prior "
                   "on --device; every word has its own seed, so the clips do not depend on this)")
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--output-key", help="the store key of the JSON report (write-once)")
    p.add_argument("--output-file", help="a local path for the JSON report instead (write-once; local checks)")
    args = p.parse_args(argv)
    if not (args.output_key or args.output_file):
        p.error("give --output-key or --output-file")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    variants = [v for v in args.variants.split(",") if v]
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        p.error(f"--variants: distinct names from {','.join(VARIANTS)}")
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from evaluate_text_to_motion import at_rate, blanked, load_generator, load_state
    from pilot_content_check import validation_sentences
    from score_back_translation import load_translator, translate
    from sentence_exposure_diagnostic import derangement, mean_statistics, ratio_to, score_forms
    from train_text_to_motion import make_store
    clock = {"start": time.time()}
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    if args.reproduce_key and not store.exists(args.reproduce_key):
        raise SystemExit(f"--reproduce-key {args.reproduce_key} does not exist")
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    val = validation_sentences(manifest, args.clips)
    texts = [r["text"] for r in val]
    refs = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS) for r in val]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
    blanks = blanked(refs)
    n = len(texts)
    if n < 2:
        raise SystemExit("need at least two clips (the controls are derangements)")
    words = [tokens(t) for t in texts]
    if max(len(w) for w in words) > MAX_TOKENS:
        raise SystemExit(f"a sentence has more than {MAX_TOKENS} words")
    if WORD_SEED + MAX_TOKENS * n >= FITTED_SEED_BLOCK:
        raise SystemExit("too many clips: word seeds would reach the fitted seed block")
    labels = None
    if "recognizer" in variants:
        from validate_sign_recognizer import load_model
        _, labels, _ = load_model(store, args.recognizer_run, "cpu")
        labels = set(labels)
    selected = {v: [select_words(w, v, labels) for w in words] for v in variants}
    needed = sorted({(i, j) for v in variants for i, idx in enumerate(selected[v]) for j in idx})
    print(f"{n} clips, {sum(seconds):.0f} s of reference; {len(needed)} word clips to sample", flush=True)

    temperature = {k: float(v) for k, v in (x.split("=") for x in args.part_temperature.split(","))}
    _, _, sample = load_generator(store, args.checkpoint_key, args.mix_sha256, args.device, temperature=temperature)
    end_to_end = generate_end_to_end(sample, texts, seconds, args.guidance)
    clock["end_to_end"] = time.time()
    print(f"end to end done ({clock['end_to_end'] - clock['start']:.0f} s)", flush=True)
    jobs = [(words[i][j], args.guidance, word_seed(i, j)) for i, j in needed]
    worker = (args.store, args.checkpoint_key, args.mix_sha256, args.device, temperature,
              max(1, torch.get_num_threads() // max(args.word_workers, 1)))
    clips = sample_words(sample, jobs, workers=args.word_workers, worker_args=worker,
                         progress=lambda k: print(f"word clip {k}/{len(jobs)} ({time.time() - clock['start']:.0f} s)",
                                                  flush=True))
    word_clips = dict(zip(needed, clips))
    clock["words"] = time.time()
    overlap = crossfade_frames(args.crossfade)
    # fitted: every selected word conditioned on its share of the reference duration (shares: typical durations)
    typical = typical_seconds(words, word_clips)
    fitted_jobs, fitted_keys, fitted_plan = [], [], {}
    for v in variants:
        vi = VARIANTS.index(v)
        for i, idx in enumerate(selected[v]):
            plan = fitted_seconds(seconds[i], [typical[words[i][j]] for j in idx], overlap)
            fitted_plan[(v, i)] = plan
            for j, s in zip(idx, plan):
                fitted_jobs.append((words[i][j], args.guidance, fitted_seed(i, j, vi), s))
                fitted_keys.append((v, i, j))
    print(f"{len(fitted_jobs)} fitted word clips to sample", flush=True)
    clips = sample_words(sample, fitted_jobs, workers=args.word_workers, worker_args=worker,
                         progress=lambda k: print(f"fitted clip {k}/{len(fitted_jobs)} "
                                                  f"({time.time() - clock['start']:.0f} s)", flush=True))
    fitted_clips = {v: {} for v in variants}
    for (v, i, j), clip in zip(fitted_keys, clips):
        fitted_clips[v][(i, j)] = clip
    clock["fitted_words"] = time.time()

    reference_perm = derangement(n, args.seed)
    content_perm = derangement(n, args.seed + 1)
    forms = {"real": refs, "blank": blanks, "end_to_end": end_to_end}
    per_clip = [dict(sign=r["sign"], text=t, tokens=w, frames_reference=frames_of(ref),
                     frames_end_to_end=frames_of(e), word_frames=[frames_of(word_clips[(i, j)]) if (i, j) in word_clips
                                                                  else None for j in range(len(w))])
                for i, (r, t, w, ref, e) in enumerate(zip(val, texts, words, refs, end_to_end))]
    selection = {}
    for v in variants:
        composed, rows = compose_forms(word_clips, selected[v], refs, blanks, content_perm, overlap, VARIANTS.index(v),
                                       fitted_clips[v])
        for name, states in composed.items():
            forms[f"{v}/{name}"] = states
        for i, (row, extra) in enumerate(zip(per_clip, rows)):
            row[v] = dict(extra, fitted_seconds=[round(s, 4) for s in fitted_plan[(v, i)]])
        counts = [len(idx) for idx in selected[v]]
        ratio = [r["frames_composed"] / frames_of(ref) for r, ref in zip(rows, refs) if not r["empty"]]
        fitted_ratio = [r["frames_fitted"] / frames_of(ref) for r, ref in zip(rows, refs) if not r["empty"]]
        plans = [s for i in range(n) for s in fitted_plan[(v, i)]]
        mism = [r["frames_mismatched_fitted_source"] / frames_of(ref) for r, ref in zip(rows, refs)
                if r["frames_mismatched_fitted_source"]]
        selection[v] = dict(empty_sentences=int(sum(c == 0 for c in counts)), words_total=int(sum(counts)),
                            words_mean=round(float(np.mean(counts)), 3), words_median=float(np.median(counts)),
                            shuffle_unchanged=int(sum(
                                [words[i][j] for j in idx] == [words[i][idx[k]] for k in r["shuffled_order"]]
                                for i, (idx, r) in enumerate(zip(selected[v], rows)))),
                            composed_over_reference_median=round(float(np.median(ratio)), 3) if ratio else None,
                            fitted_over_reference_median=round(float(np.median(fitted_ratio)), 3) if fitted_ratio
                            else None,
                            fitted_over_reference_p90=round(float(np.percentile(fitted_ratio, 90)), 3) if fitted_ratio
                            else None,
                            fitted_words_at_floor=round(float(np.mean([s <= FITTED_FLOOR for s in plans])), 4)
                            if plans else None,
                            fitted_word_seconds_median=round(float(np.median(plans)), 4) if plans else None,
                            mismatched_fitted_time_scale_median=round(float(np.median(mism)), 3) if mism else None)
    clock["compose"] = time.time()

    kp = json.loads(store.get(f"training-runs/{args.translator_run}/best.json"))["checkpoint_key"]
    translator, kind = load_translator(store, kp, MIX_SHA)                  # the translator's own manifest (round 3c)
    translator = translator.to(torch.device(args.device))
    translator_fps = float(getattr(getattr(translator, "config", None), "fps", FPS))
    hyps = {}
    for name, states in forms.items():
        hyps[name] = list(translate(translator, kind, states))
        print(f"translated {name} ({time.time() - clock['start']:.0f} s)", flush=True)
    clock["translation"] = time.time()

    pairs = form_pairs(variants)
    scores = score_forms(hyps, texts, reference_perm, boot=args.bootstrap, pairs=pairs)
    for name, row in scores.items():
        row["hypothesis_over_reference_length"] = hypothesis_length_ratio(hyps[name], texts)
    clock["scoring"] = time.time()

    motion = {name: mean_statistics(motion_rows(states, refs, translator_fps)) for name, states in forms.items()}
    motion_ratio = {name: ratio_to(m, motion["real"]) for name, m in motion.items()}
    reproduced = None
    if args.reproduce_key:
        reproduced = reproduction(json.loads(store.get(args.reproduce_key)), texts, hyps["end_to_end"])
    clock["statistics"] = time.time()

    report = dict(
        checkpoint_key=args.checkpoint_key, mix_key=args.mix_key, mix_sha256=args.mix_sha256,
        translator_run=args.translator_run, translator_key=kp, recognizer_run=args.recognizer_run if labels else None,
        recognizer_labels=len(labels) if labels else None, clips=n, guidance=args.guidance,
        part_temperature=temperature, decoder_gains=getattr(sample, "decoder_gains", None), variants=variants,
        stop_words=sorted(STOP_WORDS), crossfade=dict(seconds=args.crossfade, frames=overlap, weights="(k+1)/(n+1)"),
        fitted=dict(floor_seconds=FITTED_FLOOR, shares="typical_seconds: mean unscaled duration of the word over the run",
                    budget="reference seconds + (words - 1) * crossfade frames / 24", typical_seconds={
                        w: round(s, 4) for w, s in sorted(typical.items())}),
        seeds=dict(end_to_end="1000 + batch start (batches of 32)", word=f"{WORD_SEED} + {MAX_TOKENS} clip + index",
                   fitted=f"word seed + {FITTED_SEED_BLOCK} (1 + variant index)",
                   shuffle=f"default_rng([{SHUFFLE_SEED}, clip, variant index]) non-identity",
                   reference_derangement=args.seed, content_derangement=args.seed + 1),
        bootstrap=args.bootstrap, beam=4, word_clips=dict(
            count=len(word_clips), seconds_mean=round(float(np.mean([frames_of(s) for s in word_clips.values()]) / FPS), 3)
            if word_clips else None,
            seconds_median=float(np.median([frames_of(s) for s in word_clips.values()])) / FPS if word_clips else None),
        selection=selection, references=texts, reference_derangement=reference_perm, content_derangement=content_perm,
        forms=scores, motion=motion, motion_ratio_to_real=motion_ratio, reproduction=reproduced,
        seconds={k: round(v - clock["start"], 1) for k, v in clock.items() if k != "start"},
        caveats=CAVEATS, per_clip=per_clip)
    summary(report)
    payload = json.dumps(report, indent=1, default=float).encode()
    if args.output_file:
        with open(args.output_file, "xb") as stream:                     # write-once
            stream.write(payload)
    if args.output_key:
        if store.exists(args.output_key):
            raise SystemExit(f"{args.output_key} appeared during the run: refusing to overwrite")
        store.put(args.output_key, payload)
    return 0


def summary(report: dict) -> None:
    """BLEU and controls per form, then the motion sanity statistics."""
    def ci(d):
        return f"{d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
    forms = report["forms"]
    print(f"PRIMARY: the fitted composed form (words conditioned on their share of the reference duration), "
          f"{report['clips']} clips; end_to_end {forms['end_to_end']['bleu']:.2f}, blank {forms['blank']['bleu']:.2f}, "
          f"real {forms['real']['bleu']:.2f}", flush=True)
    for v in report["variants"]:
        row = forms[f"{v}/composed_fitted"]
        print(f"  {v + '/composed_fitted':36s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"
              f"  -e2e {ci(row['minus_end_to_end'])}  -blank {ci(row['minus_blank'])}"
              f"  -shuffled {ci(row['minus_shuffled'])}  -mismatched_words {ci(row['minus_mismatched_words'])}"
              f"  -mism {ci(row['minus_mismatched'])}  duration/ref "
              f"{report['selection'][v]['fitted_over_reference_median']}", flush=True)
    print(f"BLEU on {report['clips']} clips (form, BLEU [95% CI], mismatched refs, minus mismatched, minus blank, "
          f"minus end_to_end, hyp/ref length):", flush=True)
    for name, row in report["forms"].items():
        extra = "".join(f"  -{k[6:]} {ci(v)}" for k, v in row.items()
                        if k.startswith("minus_") and k not in ("minus_mismatched", "minus_blank", "minus_end_to_end"))
        print(f"  {name:36s} {row['bleu']:6.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]  mism "
              f"{row['mismatched']['bleu']:5.2f}  -mism {ci(row['minus_mismatched'])}"
              + (f"  -blank {ci(row['minus_blank'])}" if "minus_blank" in row else "")
              + (f"  -e2e {ci(row['minus_end_to_end'])}" if "minus_end_to_end" in row else "")
              + f"  len {row['hypothesis_over_reference_length']}" + extra, flush=True)
    print("motion (" + ", ".join(MOTION_KEYS) + "):", flush=True)
    for name, m in report["motion"].items():
        print(f"  {name:36s} " + "  ".join(f"{m.get(k):7.3f}" if m.get(k) is not None else "    nan"
                                           for k in MOTION_KEYS), flush=True)
    print(json.dumps(dict(selection=report["selection"], word_clips=report["word_clips"],
                          reproduction=report["reproduction"], seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
