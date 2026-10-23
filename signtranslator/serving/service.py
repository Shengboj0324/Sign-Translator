"""The translation service: English -> plan -> motion (-> preview video), timed per stage.

``TranslatorService.from_release`` fetches every pinned model once (hash-checked, ``release.fetch_pinned``) and keeps
them in memory; ``translate`` then runs, for one sentence:

1. **plan** (when the release has a planner): beam search to glosses, locus slots and markers, written as the plan
   line (``planning.plan_text``);
2. **generate**: the generator samples motion for the plan line (``condition: plan``) or the English, for
   ``len(glosses) x seconds_per_sign`` seconds (or 0.35 s per English word without a planner), at the release's fps,
   guidance and per-part temperatures; the seed is the request's (default 0), so a request is reproducible;
3. **render** (optional): the skeleton + face preview (``generation.skeleton_preview``, ``generation.face_preview``)
   as H.264 bytes.

Every stage's wall time is returned in milliseconds. The service is thread-safe for concurrent requests: the models
are used read-only under ``torch.no_grad`` and one lock serialises the generator (autoregressive sampling holds the
device), so the HTTP front end bounds its queue instead of oversubscribing it.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import sys
import tempfile
import threading
import time

import torch

from .release import Release, fetch_pinned

SECONDS_PER_WORD = .35


class _BytesStore:
    """A read-only store over already verified payloads (so a model is downloaded once)."""

    def __init__(self, payloads: dict):
        self.payloads = payloads

    def get(self, key: str) -> bytes:
        return self.payloads[key]


@dataclass
class Translation:
    text: str
    plan_line: str | None
    glosses: list | None
    seconds: float
    frames: int
    state_npz: bytes
    video_mp4: bytes | None
    timings_ms: dict

    def summary(self) -> dict:
        return dict(text=self.text, plan_line=self.plan_line, glosses=self.glosses, seconds=round(self.seconds, 3),
                    frames=self.frames, timings_ms=self.timings_ms, state_bytes=len(self.state_npz),
                    video_bytes=len(self.video_mp4) if self.video_mp4 else 0)


class TranslatorService:
    def __init__(self, release: Release, planner, sampler, *, device: str = "cpu"):
        self.release, self.planner, self.sampler, self.device = release, planner, sampler, device
        self._generate = threading.Lock()

    @classmethod
    def from_release(cls, store, release: Release, *, device: str = "cpu") -> "TranslatorService":
        root = Path(__file__).resolve().parents[2] / "scripts"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from evaluate_text_to_motion import load_generator
        g = release.generator
        payload = fetch_pinned(store, g.checkpoint)
        temperature = dict(g.part_temperature) if g.part_temperature else 1.
        _, _, sampler = load_generator(_BytesStore({g.checkpoint.key: payload}), g.checkpoint.key, g.manifest_sha256,
                                       device, temperature=temperature)
        planner = None
        if release.planner is not None:
            from signtranslator.generation.motion_to_text import WordVocabulary
            from signtranslator.planning.asllrp_gloss import GlossVocabulary
            from signtranslator.planning.english_gloss import EnglishGlossConfig, EnglishGlossPlanner
            saved = torch.load(io.BytesIO(fetch_pinned(store, release.planner)), map_location="cpu", weights_only=False)
            model = EnglishGlossPlanner(EnglishGlossConfig.from_dict(saved["config"]))
            model.load_state_dict(saved.get("ema") or saved["model"])
            planner = (model.to(device).eval(), WordVocabulary(saved["words"]), GlossVocabulary(saved["glosses"]))
        return cls(release, planner, sampler, device=device)

    def check_text(self, text) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        text = " ".join(text.split())
        if len(text) > self.release.max_text_chars:
            raise ValueError(f"text longer than {self.release.max_text_chars} characters")
        return text

    @torch.no_grad()
    def translate(self, text: str, *, seed: int = 0, render: bool = False, beam: int = 4) -> Translation:
        text = self.check_text(text)
        timings = {}
        started = time.perf_counter()
        plan_line = glosses = None
        if self.planner is not None:
            from signtranslator.planning.english_gloss import encode_sources, plan_from_generation
            from signtranslator.planning.plan_text import plan_to_text
            model, words, gloss_vocab = self.planner
            source = encode_sources(words, [text], max_tokens=model.config.max_source).to(self.device)
            plan = plan_from_generation(gloss_vocab, model.generate(source, beam=beam)[0])
            glosses = plan["glosses"]
            plan_line = plan_to_text(plan["glosses"], plan["loci"], plan["markers"]) if glosses else None
            timings["plan"] = round(1000 * (time.perf_counter() - started), 1)
        r = self.release
        if glosses:
            seconds = max(1, len(glosses)) * r.seconds_per_sign
        else:
            seconds = max(1., SECONDS_PER_WORD * len(text.split()))
        condition = plan_line if (r.generator.condition == "plan" and plan_line) else text
        t = time.perf_counter()
        with self._generate:
            state = self.sampler(condition, r.fps, seconds, r.generator.guidance, 50, seed)
        if r.generator.head_smoothing > 0 and "head_rotation" in state.channels:
            from ..generation.postprocess import smooth_rotation_channel
            state = smooth_rotation_channel(state, "head_rotation", r.generator.head_smoothing)
        if r.generator.face_calibration is not None and "face" in state.channels:
            from ..generation.postprocess import calibrate_face
            state = calibrate_face(state, r.generator.face_calibration)
        timings["generate"] = round(1000 * (time.perf_counter() - t), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.npz"
            state.save(path)
            npz = path.read_bytes()
        video = None
        if render:
            t = time.perf_counter()
            video = render_preview(state)
            timings["render"] = round(1000 * (time.perf_counter() - t), 1)
        timings["total"] = round(1000 * (time.perf_counter() - started), 1)
        return Translation(text=text, plan_line=plan_line, glosses=glosses, seconds=seconds,
                           frames=len(state.timestamps), state_npz=npz, video_mp4=video, timings_ms=timings)


def render_preview(state, *, panel: int = 160) -> bytes:
    """Skeleton + face preview of a generated state as H.264 bytes."""
    import numpy as np
    from signtranslator.generation.canonical_skeleton import skeleton_positions
    from signtranslator.generation.face_preview import with_face_panel
    from signtranslator.generation.skeleton_preview import encode_preview, render_preview_frames
    joints, positions, bones = skeleton_positions(state)
    frames = with_face_panel(render_preview_frames(positions, joints, bones, panel=panel, bounds_quantile=.02), state)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.mp4"
        encode_preview(frames, np.asarray(state.timestamps[:len(frames)]) - state.timestamps[0], path)
        return path.read_bytes()


__all__ = ["TranslatorService", "Translation", "render_preview", "SECONDS_PER_WORD"]
