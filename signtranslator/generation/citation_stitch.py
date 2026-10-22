"""Citation-form stitching: the retrieval/stitching baseline for English -> ASL (docs/EVALUATION_FRAMEWORK.md §7).

A plan's signs are looked up in a lexicon of recorded citation forms (``scripts/build_citation_lexicon.py``: ASLLVD
studio clips under their Sign Bank glosses, ASL Citizen clips through the exact Sign Bank map), each clip is
time-scaled so the sequence fills the requested duration, and consecutive clips are joined by short interpolated
transitions. What the baseline lacks is exactly what a generator must add: coarticulation, spatial agreement (an
``IX-3p>1`` points wherever the citation clip pointed) and the grammatical non-manuals over phrases (each sign keeps
its own citation face).

**Lookup** (``lookup_names``) tries, in order: the gloss itself; without the handshape prefix (``(25)WHY`` -> ``WHY``)
and repetition / "wiggle" suffixes (``NICE/CLEAN+`` -> ``NICE/CLEAN``, ``MOTHERwg`` -> ``MOTHER``); without a variant
number or ``-indef`` (``WANT_2`` -> ``WANT``, ``GIVE-indef`` -> ``GIVE``); a compound's parts (``PAST__+NIGHT`` ->
``PAST`` ``NIGHT``, ``WANT___+NEG``); each alternative of a slash gloss; the ASL Citizen pointing and possessive
signs for indices (``IX-2p`` -> ``citizen:YOU``, ``IX-3p`` -> ``citizen:HE``, ``IX-loc`` -> ``citizen:THERE``,
``POSS-1p`` -> ``citizen:MY``); and a lexicalised fingerspelling's sign (``#WHEN`` -> ``WHEN``). Fingerspelled words and
name signs (``fs-``, ``ns-``) and classifiers have no citation form: the previous pose is held for their share of the
time, and ``coverage`` reports how many signs were found.

**Stitched state.** Every frame is on one clock at ``fps``; values come from the clips (valid where the clip was);
``observed`` is false and confidence zero, as for generated motion, with ``inference_method="citation-stitch-v1"``.
"""
from __future__ import annotations

from dataclasses import replace
import re

import numpy as np

from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion
from signtranslator.pose.resample import resample_motion

POINTING = {"IX-1p": "citizen:ME", "IX-2p": "citizen:YOU", "IX-3p": "citizen:HE", "IX-loc": "citizen:THERE",
            "POSS-1p": "citizen:MY", "POSS-2p": "citizen:YOUR", "POSS-3p": "citizen:HIS"}
NO_CITATION = re.compile(r"^(?:fs-|ns-|[A-Z]*CL:|\d\")")
INFERENCE = "citation-stitch-v1"


def lookup_names(gloss: str) -> list[tuple[str, ...]]:
    """Alternative lexicon keys for one plan gloss, most faithful first; each alternative is a sequence (a compound
    becomes its parts)."""
    out: list[tuple[str, ...]] = []

    def add(*names):
        names = tuple(n for n in names if n)
        if names and names not in out:
            out.append(names)
    add(gloss)
    base = re.sub(r"^\([^)]*\)", "", gloss)
    base = re.sub(r"(?:\+|wg)+$", "", base)
    add(base)
    plain = re.sub(r"-indef$", "", re.sub(r"_\d+$", "", base))          # WANT_2 -> WANT, GIVE-indef -> GIVE
    add(plain)
    if re.search(r"_+\+", base) or re.search(r"[A-Z]\+[A-Z]", base):
        parts = re.split(r"_+\+|(?<=[A-Z])\+(?=[A-Z])", base)
        add(*[re.sub(r"_\d+$", "", re.sub(r"_+$", "", p)) for p in parts])
    if "/" in base:
        for alternative in base.split("/"):
            add(alternative)
    for prefix, name in POINTING.items():
        if base == prefix or base.startswith(prefix + "-") or base.startswith(prefix + ":"):
            add(name)
    if base.startswith("#"):
        add(base[1:])
    return out


def resolve(gloss: str, lexicon) -> tuple[str, ...] | None:
    """The first alternative whose every part is in ``lexicon`` (a mapping or set of keys); None if there is none."""
    if NO_CITATION.match(gloss):
        return None
    for names in lookup_names(gloss):
        if all(n in lexicon for n in names):
            return names
    return None


def _scaled(state: MultichannelMotion, frames: int, fps: float) -> MultichannelMotion:
    """``state`` played to last ``frames`` frames at ``fps``."""
    t = state.timestamps
    native = max(t[-1] - t[0], 1e-6)
    target = max(frames - 1, 1) / fps
    moved = replace(state, timestamps=t[0] + (t - t[0]) * (target / native))
    out = resample_motion(moved, fps)
    return out


def _transition(a: MultichannelMotion, b: MultichannelMotion, frames: int, fps: float):
    """``frames`` interpolated frames from ``a``'s last frame to ``b``'s first (manifold-aware, via resampling)."""
    if frames <= 0:
        return None
    ends = {}
    for name in a.channels:
        ca, cb = a.channels[name], b.channels[name]
        ends[name] = replace(ca, values=np.concatenate([ca.values[-1:], cb.values[:1]]),
                             valid=np.concatenate([ca.valid[-1:], cb.valid[:1]]),
                             observed=np.concatenate([ca.observed[-1:], cb.observed[:1]]),
                             confidence=np.concatenate([ca.confidence[-1:], cb.confidence[:1]]))
    pair = replace(a, timestamps=np.array([0., (frames + 1) / fps]), channels=ends)
    full = resample_motion(pair, fps)
    return _frames(full, 1, frames + 1)


def _frames(state: MultichannelMotion, start: int, stop: int) -> MultichannelMotion:
    return replace(state, timestamps=state.timestamps[start:stop],
                   channels={n: replace(c, values=c.values[start:stop], valid=c.valid[start:stop],
                                        observed=c.observed[start:stop], confidence=c.confidence[start:stop])
                             for n, c in state.channels.items()})


def _hold(state: MultichannelMotion, frame: int, count: int) -> MultichannelMotion:
    index = np.full(count, frame)
    return replace(state, timestamps=np.arange(count, dtype=np.float64),     # placeholder clock; _concat renumbers
                   channels={n: replace(c, values=c.values[index], valid=c.valid[index], observed=c.observed[index],
                                        confidence=c.confidence[index]) for n, c in state.channels.items()})


def _conform(state: MultichannelMotion, reference: MultichannelMotion) -> MultichannelMotion:
    """``state`` with every channel whose labels differ from ``reference``'s replaced by an invalid copy of the
    reference's layout (so the pieces concatenate)."""
    channels = {}
    for name, ref in reference.channels.items():
        c = state.channels[name]
        if c.labels == ref.labels and c.values.shape[1:] == ref.values.shape[1:]:
            channels[name] = replace(c, inference_method=INFERENCE)   # pieces of different clips join under one
        else:
            T = len(state.timestamps)
            channels[name] = replace(ref, inference_method=INFERENCE,
                                     values=np.zeros((T,) + ref.values.shape[1:], ref.values.dtype),
                                     valid=np.zeros((T, ref.values.shape[1]), bool),
                                     observed=np.zeros((T, ref.values.shape[1]), bool),
                                     confidence=np.zeros((T, ref.values.shape[1]), ref.confidence.dtype))
    return replace(state, channels=channels)


def stitch(clips: list, *, fps: float, seconds: float, source_sha256: str, transition: float = .12,
           sample_id: str = "citation-stitch") -> MultichannelMotion:
    """One state from per-sign citation clips (``None`` for a sign without one: the previous pose is held).

    The covered clips keep their relative durations and are scaled together so that clips, holds and transitions fill
    ``seconds`` exactly (``round(seconds * fps)`` frames)."""
    present = [c for c in clips if c is not None]
    if not present:
        raise ValueError("no sign of the plan has a citation form")
    reference = present[0]
    present = [_conform(resample_motion(c, fps), reference) for c in present]
    total = max(int(round(seconds * fps)), len(clips))
    k = int(round(transition * fps))
    gaps = k * max(sum(1 for c in clips if c is not None) - 1, 0)
    native = [len(c.timestamps) for c in present]
    filler = float(np.median(native))
    lengths = [native.pop(0) if c is not None else filler for c in clips]
    budget = max(total - gaps, len(clips))
    scale = budget / sum(lengths)
    frames = [max(int(round(L * scale)), 1) for L in lengths]
    frames[-1] += budget - sum(frames)                              # exact total
    if frames[-1] < 1:
        frames[-1] = 1
    pieces, previous, clip_iter = [], None, iter(present)
    for clip, n in zip(clips, frames):
        if clip is None:
            source = previous if previous is not None else present[0]
            frame = len(source.timestamps) - 1 if previous is not None else 0
            pieces.append(_hold(source, frame, n))
            continue
        current = _scaled(next(clip_iter), n, fps)
        current = _frames(current, 0, n) if len(current.timestamps) > n else current
        if len(current.timestamps) < n:
            current = _concat([current, _hold(current, len(current.timestamps) - 1, n - len(current.timestamps))])
        if previous is not None and k:
            bridge = _transition(previous, current, k, fps)
            if bridge is not None:
                pieces.append(bridge)
        pieces.append(current)
        previous = current
    out = _concat(pieces)
    out = _frames(out, 0, total) if len(out.timestamps) > total else out
    if len(out.timestamps) < total:
        out = _concat([out, _hold(out, len(out.timestamps) - 1, total - len(out.timestamps))])
    channels = {}
    for name, c in out.channels.items():
        valid = c.valid.copy()
        channels[name] = replace(c, values=np.where(valid[..., None], c.values, 0.).astype(c.values.dtype),
                                 valid=valid, observed=np.zeros_like(valid),
                                 confidence=np.zeros_like(c.confidence), source_id=f"{INFERENCE}:{source_sha256[:12]}",
                                 source_sha256=source_sha256,
                                 inference_method=INFERENCE if valid.any() else c.inference_method)
    return MultichannelMotion(timestamps=np.arange(total, dtype=np.float64) / fps, clock_id=f"{INFERENCE}-{fps:g}fps",
                              sample_id=sample_id, channels=channels)


def _concat(pieces: list[MultichannelMotion]) -> MultichannelMotion:
    first = pieces[0]
    channels = {}
    for name in first.channels:
        cs = [p.channels[name] for p in pieces]
        channels[name] = replace(cs[0], values=np.concatenate([c.values for c in cs]),
                                 valid=np.concatenate([c.valid for c in cs]),
                                 observed=np.concatenate([c.observed for c in cs]),
                                 confidence=np.concatenate([c.confidence for c in cs]))
    stamps = np.arange(sum(len(p.timestamps) for p in pieces), dtype=np.float64)
    return replace(first, timestamps=stamps, channels=channels)


def trim_to_hands(state: MultichannelMotion, *, margin: int = 2) -> MultichannelMotion:
    """The span from the first to the last frame on which either hand is valid (plus ``margin`` frames), so a clip's
    rest pose before and after the sign is not stitched in."""
    active = state.channels["left_hand"].valid.any(1) | state.channels["right_hand"].valid.any(1)
    if not active.any():
        return state
    idx = np.nonzero(active)[0]
    return _frames(state, max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, len(state.timestamps)))


def cut_seconds(state: MultichannelMotion, start: float, stop: float) -> MultichannelMotion:
    t = state.timestamps - state.timestamps[0]
    keep = np.nonzero((t >= start - 1e-9) & (t <= stop + 1e-9))[0]
    if len(keep) < 2:
        return state
    return _frames(state, int(keep[0]), int(keep[-1]) + 1)


class Stitcher:
    """Plan line -> stitched state from a lexicon (``build_citation_lexicon.py``), loading clips through ``fetch``.

    ``sample(text, fps, seconds, guidance, steps, seed)`` matches ``evaluate_text_to_motion.load_generator``'s samplers
    (guidance, steps and seed are unused: the baseline is deterministic). ``last_coverage`` holds the
    (signs found, signs) of the latest call."""

    def __init__(self, lexicon: dict, lexicon_sha256: str, fetch, *, transition: float = .12):
        import hashlib
        import tempfile
        from pathlib import Path
        self.entries, self.sha, self.fetch, self.transition = lexicon["entries"], lexicon_sha256, fetch, transition
        self._clips: dict[str, MultichannelMotion] = {}
        self._hash, self._tmp, self._path = hashlib, tempfile, Path
        self.last_coverage = (0, 0)

    def clip(self, name: str) -> MultichannelMotion:
        if name not in self._clips:
            e = self.entries[name]
            payload = self.fetch(e["state_key"])
            if self._hash.sha256(payload).hexdigest() != e["state_sha256"]:
                raise ValueError(f"{e['state_key']} differs from the lexicon's hash")
            with self._tmp.TemporaryDirectory() as t:
                path = self._path(t) / "s.npz"; path.write_bytes(payload)
                state = MultichannelMotion.load(path, expected_sha256=e["state_sha256"])
            state = trim_to_hands(state) if e["span"] == "hands" else cut_seconds(state, *e["span"])
            self._clips[name] = state
        return self._clips[name]

    def clips_for(self, glosses: list[str]) -> list:
        out, found = [], 0
        for g in glosses:
            names = resolve(g, self.entries)
            if names is None:
                out.append(None)
                continue
            found += 1
            out.extend(self.clip(n) for n in names)
        self.last_coverage = (found, len(glosses))
        return out

    def sample(self, text: str, fps: float, seconds: float, guidance=None, steps=None, seed=None) -> MultichannelMotion:
        from signtranslator.planning.plan_text import plan_from_text
        glosses = plan_from_text(text)[0]
        return stitch(self.clips_for(glosses), fps=fps, seconds=seconds, source_sha256=self.sha,
                      transition=self.transition)


def load_stitcher(store, key: str, sha256: str, **kwargs) -> Stitcher:
    import hashlib
    import json
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise SystemExit(f"{key} differs from its expected SHA-256")
    return Stitcher(json.loads(payload), sha256, store.get, **kwargs)


__all__ = ["POINTING", "INFERENCE", "lookup_names", "resolve", "stitch", "trim_to_hands", "cut_seconds", "Stitcher",
           "load_stitcher"]
