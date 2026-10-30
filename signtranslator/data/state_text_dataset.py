"""Training and selection data of instrument v2 (``generation.state_to_text_t5``): assembled records
(``generation.bt_corpora``) -> (features (T, F) float32, English).

``__getitem__``:
1. ``bt_corpora.check_loadable_bt`` (with ``training`` for the training set: only role ``train``);
2. with probability ``recon_p`` a record with a tokenizer round-trip copy (``_recon``: ``build_recon_states.py``'s
   decoded state, attached by ``recognizer_corpora.attach_recon`` when its load digest matches) is served as that copy,
   under the same text (the generator's form with the clip's content); otherwise ``corpus_manifest.load_sample``
   (usable channels, trim window, mirror flag);
3. conformed to the template layouts (``layout_adapter.conform``);
4. with ``augment``, ``sign_recognizer.augment_sign`` (speed 0.8-1.25, trim up to 10% at each end, face dropout,
   ``validity_dropout`` of arm/torso joint groups), drawn from a per-process generator;
5. resampled to the configuration's rate, featurised (``data.state_features``) and cut to ``max_seconds`` (None: the
   whole clip, as the acceptance reads it).

No control input (blank, static, mismatched) is ever served with a special target: every item is a real clip, or its
round-trip copy, with its own English. ``blank`` (selection and reports only; refused with ``training``) serves the
clip with every channel removed, durations kept: ``evaluate_text_to_motion.blanked``'s control.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..generation.bt_corpora import check_loadable_bt, text_of

_RNG = {}


def _rng():
    """One generator per loader process, seeded from torch (data-loader workers get distinct seeds)."""
    import os
    pid = os.getpid()
    if pid not in _RNG:
        _RNG[pid] = np.random.default_rng(torch.initial_seed() ^ 0x57A7E)
    return _RNG[pid]


class StateTextDataset(Dataset):
    def __init__(self, records, config, *, cache, fetch: Callable[[str], bytes], training: bool,
                 augment: bool = False, recon_p: float = 0., validity_dropout: float = 0., face_dropout: float = .2,
                 speed=(.8, 1.25), trim: float = .1, max_seconds: float | None = 20., blank: bool = False) -> None:
        if not 0. <= recon_p <= 1.:
            raise ValueError("recon_p must lie in [0, 1]")
        if (augment or recon_p) and not training:
            raise ValueError("augmentation and round-trip copies are for training data only")
        if blank and training:
            raise ValueError("a blank (control) input is never trained on")
        self.records = list(records)
        for r in self.records:
            check_loadable_bt(r, training=training)
        from ..generation.state_to_text_t5 import layout_template
        self.config, self.cache, self.fetch, self.training = config, cache, fetch, training
        self.augment, self.recon_p, self.validity_dropout = augment, float(recon_p), float(validity_dropout)
        self.face_dropout, self.speed, self.trim = face_dropout, tuple(speed), trim
        self.max_frames = None if max_seconds is None else int(max_seconds * config.fps)   # None: full length
        self.blank = bool(blank)
        self.template = layout_template(config)

    def __len__(self) -> int:
        return len(self.records)

    def weights(self) -> list[float]:
        return [float(r.get("_weight", 1.)) for r in self.records]

    def state(self, index: int, *, rng=None):
        """The record's state as served (steps 1-4), before resampling."""
        from ..data.layout_adapter import conform
        from ..data_engineering.corpus_manifest import load_sample
        from ..generation.sign_recognizer import augment_sign, load_state
        r = self.records[index]
        check_loadable_bt(r, training=self.training)
        rng = rng if rng is not None else _rng()
        if self.recon_p > 0 and r.get("_recon") and rng.random() < self.recon_p:
            state = load_state(r["_recon"], cache=self.cache, fetch=self.fetch)
        else:
            state = load_sample(r, "continuous", cache=self.cache, fetch=self.fetch)
        state = conform(state, self.template)
        if self.blank:                                     # the selection's control: durations only
            from dataclasses import replace
            state = replace(state, channels={n: replace(c, valid=np.zeros_like(c.valid), observed=np.zeros_like(c.observed),
                                                        confidence=np.zeros_like(c.confidence),
                                                        values=np.zeros_like(c.values), inference_method=None)
                                             for n, c in state.channels.items()})
        if self.augment:
            state = augment_sign(state, rng, speed=self.speed, trim=self.trim, face_dropout=self.face_dropout,
                                 validity_dropout=self.validity_dropout)
        return state

    def __getitem__(self, index: int):
        from ..generation.state_to_text_t5 import at_fps
        from .state_features import state_features
        state = self.state(index)
        if self.augment:                                   # the clock was rescaled: always resample
            from ..pose.resample import resample_motion
            state = resample_motion(state, self.config.fps)
        else:
            state = at_fps(state, self.config.fps)
        features = state_features(state, self.config.motion)[:self.max_frames]
        return torch.from_numpy(features), text_of(self.records[index])


__all__ = ["StateTextDataset"]
