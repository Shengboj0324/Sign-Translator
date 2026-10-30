"""Road to 3.8, part 2, stage S0: ceilings and the measurement panel on instrument v1 (evaluation only, no training).

Declared in docs/NEXT_PHASE_PLAN.md, "Road to 3.8, part 2" (2026-10-30, 19:32 UTC). Background:
``reports/Route to near real signing.md`` (S0 row and "Measure progress with likelihood and faithfulness").

**Clips.** The first ``--clips`` of ``pilot_content_check.validation_sentences(round10-mixed-v1, --sample)``
(``--sample`` 1000: the 1,000 How2Sign val clips every pilot and the round-15 1,000-clip composition evaluation read),
at 24 fps. A smaller read is always a *prefix* of that declared draw (``declared_sample``), never a draw of its own:
``random.sample``'s choices depend on k (k = 1000 uses its pool method, k = 20 or 32 its set method), so
``validation_sentences(mix, 32)`` agrees with the 1,000-draw only on its first 16 clips (the 2026-10-30 smoke failure).
The report states the draw (``sample``: mix, SHA-256, drawn, clips, signs). With ``--stored`` reports their
references must begin with these texts (checked).

**Forms** (``--forms``; each a list of states of the same clips, read by the same instrument):
- ``real``; ``blank`` (every channel removed, durations kept: ``evaluate_text_to_motion.blanked``);
- ``round_trip``: d2's round trip (``tokenizer_recognisability_check.round_trip``, as ``tokenizer_fidelity_check``:
  no decoder gains; reproduces the 3.365 of round 17's fidelity read); ``round_trip_gains``: the same with the
  tokenizer's accepted decoder gains, which every generated state carries (``load_generator``);
- ``stages1``, ``stages2``, ``stages4``: the round trip decoded from the first 1, 2 or 4 of d2's 8 residual stages
  (``MotionRVQ.quantize(latents, stages)``), no gains;
- ``face_mean``: the face channels (face, blink, left_gaze, right_gaze) held at each clip's mean over its valid frames;
  ``face_absent``: the same channels removed. Instrument v1 reads 48 arm and hand points only (``data.keypoints2d``:
  "Faces are left out"), so on v1 both must read exactly as ``real`` (``identical_to_real`` is reported; a value
  below 1 is a bug). They exist for instrument v2 (``--translator-run``);
- ``warp0.8`` / ``warp1.25``: real played at 0.8× / 1.25× speed (timestamps divided by the speed, resampled to 24 fps;
  durations 1.25× / 0.8×);
- ``lowpass6``: real low-passed by a 4th-order zero-phase Butterworth at 6 Hz on each joint's contiguous valid runs,
  then put back on each channel's manifold (``low_pass``).

**Per form** (``signtranslator.eval.s0_metrics``): hypotheses (beam 4, batch 32, as every pilot), BLEU-4 with bootstrap
CI, chrF (sacreBLEU's chrF2, reimplemented), length ratio, gap fractions (x − blank)/(real − blank), paired bootstraps
against real and against blank (BLEU, chrF, ΔNLL, MRR), ΔNLL over blank (per sentence, token-length normalised),
100-way retrieval (NLL ranks and PMI ranks against blank), ρ_faith in d2's latent space (DTW primary, time-averaged
secondary, per part), and per-clip arrays. ``orderings`` then asks S0's pass question: do ΔNLL and retrieval
reproduce every ordering BLEU resolves (paired CI excluding 0), with tighter intervals (half-widths normalised by
each metric's real − blank range)?

**Stored outputs** (``--stored KEY``, repeatable): pilot reports (``meaning``) and round-15 composition reports
(``forms``) hold hypotheses only, no motion. Their hypotheses get every text metric (BLEU, chrF, length ratio, gap
fractions, paired against this run's real and blank). ΔNLL, retrieval and ρ_faith need motion, so they need
regeneration: ``--regenerate KEY`` (repeatable; end-to-end pilot reports only) regenerates the pilot's generated half
exactly as ``pilot_content_check`` did (its checkpoint, guidance and part temperatures, seeds 1000 + i per block of
32, the round-3e post-processing, the tokenizer's accepted gains) on **the pilot's own clip list**
(``pilot_records``: ``validation_sentences`` of the pilot's generator mix, which is the ``--mix-key`` its pilot ran with,
at the pilot's own clip count, checked text for text against its stored references), whole blocks of 32 even when
``--clips`` is smaller; the regenerated clips must be the ceiling read's clips (sign ids, checked); the report gives
the pilot's mix and then reports how many hypotheses reproduce the stored ones before giving the full panel. Composition forms need
``round15_compose_eval.py``'s renderer (round 20's tooling); they are read on hypotheses only here.

``--forms none`` with ``--stored``: a CPU-only read of stored hypotheses (references, real and blank from the first
stored report that has them); no model is loaded.

Write-once: ``--output-key`` (refused if it exists) or ``--output-file``.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from signtranslator.eval import s0_metrics as M  # noqa: E402

EVAL_MIX = "acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
TOKENIZER = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
TOKENIZER_SHA = "021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de"
TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"                    # instrument v1
TRANSLATOR_MANIFEST_SHA = "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd"   # as pilot_content_check
FACE_CHANNELS = ("face", "blink", "left_gaze", "right_gaze")
ALL_FORMS = ("real", "blank", "round_trip", "round_trip_gains", "stages1", "stages2", "stages4", "face_mean",
             "face_absent", "warp0.8", "warp1.25", "lowpass6")
FPS = 24.
BLOCK = 32                                                            # the pilots' generation and translation batch


# ---- forms ---------------------------------------------------------------------------------------------------------

class _FirstStages:
    """``round_trip``'s ``plan`` hook: the decoder reads the quantised latents of the first ``stages`` stages."""

    def __init__(self, model, stages: int) -> None:
        self.model, self.stages = model, stages

    def eval(self):
        return self

    def reconstruct(self, latents, valid):
        return self.model.quantize(latents, self.stages)[0]


def truncated_round_trip(model, state, stages: int | None):
    """d2's round trip of ``state`` decoded from its first ``stages`` residual stages (None: all, the fidelity read)."""
    from tokenizer_recognisability_check import round_trip
    return round_trip(model, model.config.motion, state, FPS, plan=None if stages is None else _FirstStages(model, stages))


def _on_manifold(name: str, values: np.ndarray) -> np.ndarray:
    from signtranslator.pose.multichannel import CHANNELS
    from signtranslator.pose.resample import _orthonormal_6d
    kind = CHANNELS[name][0] if name in CHANNELS else None
    if kind == "rotation_6d_columns":
        return _orthonormal_6d(values)
    if kind == "unit_vector":
        return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-9, None)
    if kind == "bounded_scalar":
        return np.clip(values, 0., 1.)
    return values


def face_mean(state):
    """The face channels held at the clip's per-joint mean over valid frames (validity unchanged)."""
    channels = dict(state.channels)
    for name in FACE_CHANNELS:
        c = channels.get(name)
        if c is None or not c.valid.any():
            continue
        w = c.valid[..., None].astype(np.float64)
        mean = (c.values * w).sum(0) / np.maximum(w.sum(0), 1.)                     # (J, D)
        values = np.where(c.valid[..., None], _on_manifold(name, np.broadcast_to(mean, c.values.shape).copy()), 0.)
        channels[name] = replace(c, values=values.astype(c.values.dtype))
    return replace(state, channels=channels)


def face_absent(state):
    """The face channels removed (no valid frame, zero payload), as ``blanked`` removes every channel."""
    channels = dict(state.channels)
    for name in FACE_CHANNELS:
        c = channels.get(name)
        if c is None:
            continue
        channels[name] = replace(c, valid=np.zeros_like(c.valid), observed=np.zeros_like(c.observed),
                                 confidence=np.zeros_like(c.confidence), values=np.zeros_like(c.values))
    return replace(state, channels=channels)


def time_warp(state, speed: float, fps: float = FPS):
    """``state`` played at ``speed``× (0.8: slower, 1.25× as long), resampled to ``fps``."""
    from signtranslator.pose.resample import resample_motion
    t = np.asarray(state.timestamps, np.float64)
    return resample_motion(replace(state, timestamps=t[0] + (t - t[0]) / speed), fps)


def low_pass(state, cutoff: float = 6., order: int = 4):
    """Zero-phase Butterworth low-pass of every channel along time, per joint on its contiguous valid runs (runs of
    fewer than 2 frames are kept as they are), then each channel put back on its manifold; validity unchanged."""
    from scipy.signal import butter, sosfiltfilt
    t = np.asarray(state.timestamps, np.float64)
    if len(t) < 3:
        return state
    fs = 1. / float(np.median(np.diff(t)))
    sos = butter(order, cutoff, fs=fs, output="sos")
    default_pad = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    channels = {}
    for name, c in state.channels.items():
        values = c.values.astype(np.float64).copy()
        for j in range(c.valid.shape[1]):
            v = c.valid[:, j]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
            for a, b in zip(edges[::2], edges[1::2]):
                if b - a < 2:
                    continue
                values[a:b, j] = sosfiltfilt(sos, values[a:b, j], axis=0, padlen=min(default_pad, b - a - 1))
        values = np.where(c.valid[..., None], _on_manifold(name, values), 0.)
        channels[name] = replace(c, values=values.astype(c.values.dtype))
    return replace(state, channels=channels)


def build_form(name: str, refs, tokenizer, gains):
    from evaluate_text_to_motion import blanked
    if name == "real":
        return list(refs)
    if name == "blank":
        return blanked(refs)
    if name in ("round_trip", "round_trip_gains"):
        trips = [truncated_round_trip(tokenizer, s, None) for s in refs]
        if name == "round_trip_gains":
            if not gains:
                raise SystemExit("round_trip_gains: the tokenizer has no accepted decoder gains")
            from fit_tokenizer_gains import apply_gains
            trips = [apply_gains(t, gains) for t in trips]
        return trips
    if name.startswith("stages"):
        k = int(name[len("stages"):])
        if not 1 <= k <= tokenizer.config.stages:
            raise SystemExit(f"{name}: d2 has {tokenizer.config.stages} stages")
        return [truncated_round_trip(tokenizer, s, k) for s in refs]
    if name == "face_mean":
        return [face_mean(s) for s in refs]
    if name == "face_absent":
        return [face_absent(s) for s in refs]
    if name.startswith("warp"):
        return [time_warp(s, float(name[len("warp"):])) for s in refs]
    if name.startswith("lowpass"):
        return [low_pass(s, float(name[len("lowpass"):])) for s in refs]
    raise SystemExit(f"unknown form {name}")


# ---- the translator: inputs, likelihoods ----------------------------------------------------------------------------

def translator_items(model, kind: str, states) -> list:
    """Each state's input frames for the translator, exactly as ``score_back_translation.translate`` builds them."""
    import torch
    if kind == "keypoints":
        from evaluate_text_to_motion import at_rate
        from signtranslator.data.keypoints2d import features, from_state, normalise
        out = []
        for state in states:
            points, valid = from_state(at_rate(state, model.config.fps))
            out.append(torch.from_numpy(features(normalise(points, valid, y_down=False), valid)))
        return out
    if kind == "states":
        return [torch.from_numpy(f) for f in model.featurise(states)]
    raise SystemExit(f"likelihood reads need a keypoint or state translator, not {kind!r}")


def candidate_nll(model, items, references, candidates, *, device, group: int = 2) -> np.ndarray:
    """(n, k) token-mean NLL of ``references[candidates[i, c]]`` given clip i's input (teacher forced; the translator's
    tokenisation and truncation, ``MotionToT5._labels``)."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput
    from signtranslator.data.keypoint_shards import collate_keypoints
    n, k = candidates.shape
    out = np.zeros((n, k), np.float64)
    with torch.no_grad():
        for start in range(0, n, group):
            rows = list(range(start, min(n, start + group)))
            (x, valid), _ = collate_keypoints([(items[i], "") for i in rows])
            states, enc_valid = model.encode((x.to(device), valid.to(device)))
            states = states.repeat_interleave(k, 0)
            enc_valid = enc_valid.repeat_interleave(k, 0)
            texts = [references[c] for i in rows for c in candidates[i]]
            labels = model._labels(texts, states.device)
            logits = model.t5(encoder_outputs=BaseModelOutput(last_hidden_state=states),
                              attention_mask=enc_valid.long(), labels=labels).logits
            total, count = M.token_nll(logits, labels)
            out[rows] = (total / count.clamp_min(1)).cpu().numpy().reshape(len(rows), k)
    return out


def d2_latents(tokenizer, states, device) -> list[np.ndarray]:
    """Each state's d2 encoder output (continuous, before quantisation): (frames, parts × code_dim), parts in the
    tokenizer's order, float32."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    out = []
    with torch.no_grad():
        for state in states:
            b = collate_multichannel([state])
            unit = diffusion_targets(tokenizer.config.motion, b).to(device)
            lat = tokenizer.encode(unit)
            z = torch.cat([lat[p] for p, _ in tokenizer.parts], -1)[0, :unit.shape[1]]
            out.append(z.float().cpu().numpy())
    return out


# ---- per-form panel -------------------------------------------------------------------------------------------------

def _resolved(ci) -> int:
    """+1 / −1 when the interval excludes 0 on that side, 0 otherwise."""
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def text_panel(hyps, refs, anchors: dict | None) -> dict:
    """BLEU, chrF, length ratio; with ``anchors`` (real and blank: hypotheses + scores) gap fractions and paired
    bootstraps against both."""
    stats = M.bleu_statistics(hyps, refs)
    bleu, bleu_ci = M.bleu_ci_fast(hyps, refs)
    chrf, chrf_ci = M.chrf_ci(hyps, refs)
    out = dict(bleu=dict(bleu=bleu, ci95=bleu_ci), chrf=dict(chrf=chrf, ci95=chrf_ci), length_ratio=M.length_ratio(stats))
    if anchors:
        real, blank = anchors["real"], anchors["blank"]
        out["gap_fraction"] = dict(bleu=_round(M.gap_fraction(bleu, blank["bleu"], real["bleu"])),
                                   chrf=_round(M.gap_fraction(chrf, blank["chrf"], real["chrf"])))
        out["paired"] = {f"vs_{a}": dict(bleu=M.paired_bleu_fast(stats, anchors[a]["stats"]),
                                         chrf=M.paired_chrf_difference(hyps, anchors[a]["hypotheses"], refs))
                         for a in ("real", "blank")}
        out["identical_to_real"] = round(float(np.mean([h == r for h, r in zip(hyps, real["hypotheses"])])), 4)
    return out


def _round(x, digits: int = 4):
    return None if x is None else round(float(x), digits)


def likelihood_panel(nll, nll_blank, anchors: dict | None) -> dict:
    """ΔNLL (column 0 of the candidate matrices) and retrieval from the (n, k) NLL matrix of a form and of blank."""
    own = nll[:, 0]
    dnll = M.delta_nll(nll_blank[:, 0], own)
    ranks = M.mid_ranks(nll)
    ranks_pmi = M.mid_ranks(nll - nll_blank)
    k = nll.shape[1]
    out = dict(nll=M.bootstrap_mean(own), delta_nll=M.bootstrap_mean(dnll),
               retrieval=dict(nll=M.retrieval_summary(ranks, k), pmi=M.retrieval_summary(ranks_pmi, k),
                              mrr=M.bootstrap_mean(1. / ranks), mrr_pmi=M.bootstrap_mean(1. / ranks_pmi)),
               per_clip=dict(nll=np.round(own, 4).tolist(), delta_nll=np.round(dnll, 4).tolist(),
                             rank=ranks.tolist(), rank_pmi=ranks_pmi.tolist()))
    if anchors:
        real, blank = anchors["real"], anchors["blank"]
        rr, rr_real, rr_blank = 1. / ranks, 1. / real["rank"], 1. / blank["rank"]
        out["gap_fraction"] = dict(delta_nll=M.ratio_of_means(dnll, real["delta_nll"]),
                                   mrr=_round(M.gap_fraction(rr.mean(), rr_blank.mean(), rr_real.mean())))
        out["paired"] = {f"vs_{a}": dict(delta_nll=M.paired_mean_difference(dnll, anchors[a]["delta_nll"]),
                                         mrr=M.paired_mean_difference(rr, 1. / anchors[a]["rank"]))
                         for a in ("real", "blank")}
    return out


def rho_panel(latents, real_latents, mean, std, parts, code_dim, *, K, device, per_part: bool) -> dict:
    norm = lambda zs: [(z - mean) / std for z in zs]
    out_z, ref_z = norm(latents), norm(real_latents)
    dtw = M.rho_faith(out_z, ref_z, K=K, distance="dtw", device=device)
    avg = M.rho_faith(out_z, ref_z, K=K, distance="mean")
    block = dict(dtw={k: v for k, v in dtw.items() if k != "per_clip"},
                 mean={k: v for k, v in avg.items() if k != "per_clip"},
                 per_clip_dtw=dtw["per_clip"])
    if per_part:
        block["parts"] = {}
        for p, part in enumerate(parts):
            sl = slice(p * code_dim, (p + 1) * code_dim)
            r = M.rho_faith([z[:, sl] for z in out_z], [z[:, sl] for z in ref_z], K=K, distance="dtw", device=device)
            block["parts"][part] = {k: v for k, v in r.items() if k != "per_clip"}
    return block


def orderings(table: dict, real: str = "real", blank: str = "blank") -> dict:
    """S0's pass question over every pair of forms with full panels: for each pair, paired BLEU, ΔNLL and MRR (and
    PMI-MRR) differences with 95% intervals; for each metric, whether BLEU's resolved orderings are reproduced (same
    sign resolved), contradicted (opposite sign resolved) or left unresolved, and the median half-width normalised
    by that metric's real − blank range. Kendall's τ of the point estimates against BLEU's."""
    from scipy.stats import kendalltau
    names = sorted(table)
    ranges = dict(bleu=M.bleu_from_statistics(table[real]["stats"].sum(0)) - M.bleu_from_statistics(table[blank]["stats"].sum(0)),
                  delta_nll=float(np.mean(table[real]["delta_nll"]) - np.mean(table[blank]["delta_nll"])),
                  mrr=float(np.mean(1. / table[real]["rank"]) - np.mean(1. / table[blank]["rank"])),
                  mrr_pmi=float(np.mean(1. / table[real]["rank_pmi"]) - np.mean(1. / table[blank]["rank_pmi"])))
    metrics = ("delta_nll", "mrr", "mrr_pmi")
    pairs, widths = [], {m: [] for m in ("bleu",) + metrics}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = dict(a=a, b=b, bleu=M.paired_bleu_fast(table[a]["stats"], table[b]["stats"]))
            row["delta_nll"] = M.paired_mean_difference(table[a]["delta_nll"], table[b]["delta_nll"])
            row["mrr"] = M.paired_mean_difference(1. / table[a]["rank"], 1. / table[b]["rank"])
            row["mrr_pmi"] = M.paired_mean_difference(1. / table[a]["rank_pmi"], 1. / table[b]["rank_pmi"])
            for m in ("bleu",) + metrics:
                lo, hi = row[m]["ci95"]
                row[m]["resolved"] = _resolved(row[m]["ci95"])
                if ranges[m]:
                    widths[m].append((hi - lo) / 2 / abs(ranges[m]))
            pairs.append(row)
    summary = {}
    for m in metrics:
        agree = sum(1 for r in pairs if r["bleu"]["resolved"] and r[m]["resolved"] == r["bleu"]["resolved"])
        opposite = sum(1 for r in pairs if r["bleu"]["resolved"] and r[m]["resolved"] == -r["bleu"]["resolved"])
        unresolved = sum(1 for r in pairs if r["bleu"]["resolved"] and not r[m]["resolved"])
        extra = sum(1 for r in pairs if not r["bleu"]["resolved"] and r[m]["resolved"])
        bleu_points = [M.bleu_from_statistics(table[n]["stats"].sum(0)) for n in names]
        points = [float(np.mean(table[n]["delta_nll"])) if m == "delta_nll" else
                  float(np.mean(1. / table[n]["rank" if m == "mrr" else "rank_pmi"])) for n in names]
        tau = kendalltau(bleu_points, points).statistic if len(names) > 2 else None
        hw, hw_bleu = (float(np.median(widths[m])) if widths[m] else None,
                       float(np.median(widths["bleu"])) if widths["bleu"] else None)
        summary[m] = dict(bleu_resolved=sum(1 for r in pairs if r["bleu"]["resolved"]), reproduced=agree,
                          contradicted=opposite, unresolved=unresolved, resolved_where_bleu_is_not=extra,
                          median_normalised_half_width=_round(hw), bleu_median_normalised_half_width=_round(hw_bleu),
                          kendall_tau_vs_bleu=_round(tau),
                          passes=bool(opposite == 0 and unresolved == 0 and hw is not None and hw_bleu is not None
                                      and hw < hw_bleu))
    return dict(forms=names, ranges={k: _round(v, 5) for k, v in ranges.items()}, pairs=pairs, summary=summary,
                rule="a metric passes when it reproduces every ordering BLEU resolves (none contradicted, none "
                     "unresolved) and its median half-width, normalised by its real − blank range, is below BLEU's")


# ---- stored reports and regeneration ---------------------------------------------------------------------------------

def stored_forms(report: dict) -> tuple[list[str], dict]:
    """(references, {form: hypotheses}) of a stored pilot report (``meaning``) or round-15 composition report."""
    if "meaning" in report:
        meaning = report["meaning"]
        forms = {k: v["hypotheses"] for k, v in meaning.items() if isinstance(v, dict) and "hypotheses" in v}
        for extra in ("meaning_oracle_plan",):
            if isinstance(report.get(extra), dict) and "hypotheses" in report[extra]:
                forms[extra] = report[extra]["hypotheses"]
        return meaning["references"], forms
    if report.get("kind") == "round15-compose-eval" or "forms" in report:
        return report["references"], {k: v["hypotheses"] for k, v in report["forms"].items() if "hypotheses" in v}
    raise SystemExit("not a pilot or composition report")


def short_name(key: str) -> str:
    parts = key[:-5].split("/") if key.endswith(".json") else key.split("/")
    return "/".join(parts[-2:])


def declared_sample(manifest, sample: int, clips: int):
    """The first ``clips`` records of ``validation_sentences(manifest, sample)``: a smaller read is a prefix of the
    declared draw (``random.sample(population, k)`` draws differently for different k)."""
    from pilot_content_check import validation_sentences
    if clips > sample:
        raise SystemExit(f"--clips {clips} exceeds the declared sample {sample}")
    return validation_sentences(manifest, sample)[:clips]


def pilot_records(store, report: dict, load_mix):
    """(records, run info) of a stored pilot's own clips: ``validation_sentences(its generator's training mix,
    len(references))``. ``pilot_content_check`` drew its clips from ``--mix-key`` and gave ``--mix-sha256`` to
    ``load_generator``, which refuses any mix but the checkpoint's training manifest, so that mix is the run's
    ``run.json`` manifest. Checked text for text against the stored references."""
    from pilot_content_check import validation_sentences
    run = report["checkpoint_key"].split("/checkpoints/")[0]
    info = json.loads(store.get(f"{run}/run.json"))
    references = report["meaning"]["references"]
    records = validation_sentences(load_mix(info["manifest_key"], info["manifest_sha256"]), len(references))
    if [r["text"] for r in records] != references:
        raise SystemExit(f"{run}: validation_sentences of its mix does not reproduce the pilot's references")
    return records, info


def regenerate(store, key: str, clips: int, device: str, load_mix, load_states):
    """(generated states of the pilot's first ``clips`` clips, the pilot report, those clips' records, run info):
    the pilot's generated half regenerated as pilot_content_check did, on its own clip list, whole blocks of 32 from
    clip 0 (``load_generator`` with the stored checkpoint, guidance and part temperatures; seeds 1000 + i;
    post-processed)."""
    from evaluate_text_to_motion import load_generator
    from pilot_content_check import postprocess
    report = json.loads(store.get(key))
    if report.get("plan"):
        raise SystemExit(f"{key}: plan-conditioned pilots are not regenerated here")
    if "meaning" not in report or "generated" not in report["meaning"]:
        raise SystemExit(f"{key}: not an end-to-end pilot report")
    records, info = pilot_records(store, report, load_mix)
    if clips > len(records):
        raise SystemExit(f"{key}: the pilot read {len(records)} clips, fewer than --clips {clips}")
    block = records[:min(len(records), math.ceil(clips / BLOCK) * BLOCK)]
    refs = load_states(block)
    texts = [r["text"] for r in block]
    seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in refs]
    _, _, sample = load_generator(store, report["checkpoint_key"], info["manifest_sha256"], device,
                                  temperature=report["part_temperature"])
    gens = []
    for i in range(0, len(texts), BLOCK):
        gens += [postprocess(o) for o in sample.batch(texts[i:i + BLOCK], FPS, seconds[i:i + BLOCK], report["guidance"],
                                                      1000 + i)]
    return gens[:clips], report, records[:clips], info


# ---- main -----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--sample", type=int, default=1000, help="the declared draw: validation_sentences(mix, --sample)")
    p.add_argument("--clips", type=int, default=1000, help="read the first --clips of the declared draw")
    p.add_argument("--forms", default=",".join(ALL_FORMS), help="comma list of forms, or 'none' (stored reads only)")
    p.add_argument("--tokenizer-key", default=TOKENIZER)
    p.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA)
    p.add_argument("--translator-run", default=TRANSLATOR_RUN, help="v1 by default; another run is read through "
                   "score_back_translation.load_translator_run (instrument v2, SHA-checked)")
    p.add_argument("--translator-sha256", help="pin the translator checkpoint's SHA-256 (checked)")
    p.add_argument("--candidates", type=int, default=100, help="retrieval candidates per clip (own included)")
    p.add_argument("--rho-references", type=int, default=10, help="random references per clip for ρ_faith")
    p.add_argument("--no-rho-parts", action="store_true", help="skip the per-part ρ_faith")
    p.add_argument("--stored", action="append", default=[], help="a stored pilot or composition report (hypotheses)")
    p.add_argument("--regenerate", action="append", default=[], help="an end-to-end pilot report to regenerate")
    p.add_argument("--merge-from", action="append", default=[], help="an earlier s0 report (key, or file:/path) of the "
                   "same clips and instrument: its forms join this run's orderings (per-clip arrays and hypotheses)")
    p.add_argument("--nll-group", type=int, default=2, help="clips per teacher-forced batch (× candidates rows)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    if bool(args.output_key) == bool(args.output_file):
        raise SystemExit("give exactly one of --output-key and --output-file")
    import hashlib
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists (write-once)")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists (write-once)")
    forms = [] if args.forms == "none" else [f for f in args.forms.split(",") if f]
    if forms:
        missing = [f for f in ("real", "blank") if f not in forms]
        if missing:
            raise SystemExit(f"--forms must include real and blank (every gap fraction reads them): {missing}")
        forms = ["real", "blank"] + [f for f in forms if f not in ("real", "blank")]
    elif args.regenerate:
        raise SystemExit("--regenerate needs the real and blank forms")
    elif not args.stored:
        raise SystemExit("nothing to do: --forms none and no --stored report")
    started = time.time()
    report = dict(kind="s0-ceilings", settings=dict(vars(args)), forms={}, stored={}, regenerated={})

    stored, stored_sha = {}, {}
    for k in args.stored:
        raw = store.get(k)
        stored[k], stored_sha[k] = stored_forms(json.loads(raw)), hashlib.sha256(raw).hexdigest()
    if not forms:                                       # CPU-only read of stored hypotheses
        anchor = next((k for k, (_, f) in stored.items() if "real" in f and "blank" in f), None)
        if anchor is None:
            raise SystemExit("--forms none: no stored report holds real and blank hypotheses")
        texts = stored[anchor][0][:args.clips]
        report["settings"]["anchor"] = anchor
        hyps = {"real": stored[anchor][1]["real"][:len(texts)], "blank": stored[anchor][1]["blank"][:len(texts)]}
        anchors = {n: dict(hypotheses=h, stats=M.bleu_statistics(h, texts), bleu=M.bleu_ci_fast(h, texts)[0],
                           chrf=M.corpus_chrf(h, texts)) for n, h in hyps.items()}
        for n in ("real", "blank"):
            report["forms"][n] = dict(source=anchor, **text_panel(hyps[n], texts, anchors))
    else:
        import torch
        from signtranslator.data_engineering.corpus_manifest import load_manifest
        from evaluate_text_to_motion import accepted_decoder_gains, at_rate, load_state
        from pilot_content_check import validation_sentences
        from score_back_translation import load_translator, load_translator_run, translate
        import diagnose_tokenizer as dt
        device = torch.device(args.device)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
            manifest = load_manifest(path, expected_sha256=args.mix_sha256)
        mixes = {}

        def load_mix(key, sha):
            if (key, sha) not in mixes:
                with tempfile.TemporaryDirectory() as t:
                    path = Path(t) / "m.json"; path.write_bytes(store.get(key) + b"\n")
                    mixes[(key, sha)] = load_manifest(path, expected_sha256=sha)
            return mixes[(key, sha)]

        def load_states(recs):
            return [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS)
                    for r in recs]
        mixes[(args.mix_key, args.mix_sha256)] = manifest
        records = declared_sample(manifest, args.sample, args.clips)
        texts = [r["text"] for r in records]
        refs = load_states(records)
        report["clips"], report["references"] = len(texts), texts
        report["signs"] = [r["sign"] for r in records]
        report["sample"] = dict(mix_key=args.mix_key, mix_sha256=args.mix_sha256, drawn=args.sample, clips=len(texts),
                                rule="the first --clips records of pilot_content_check.validation_sentences(mix, "
                                     "--sample)")
        payload = store.get(args.tokenizer_key)
        if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
            raise SystemExit(f"{args.tokenizer_key}: SHA-256 differs from --tokenizer-sha256")
        tokenizer, _ = dt.load_tokenizer(store, args.tokenizer_key)
        tokenizer.eval()                                # round trips on the CPU, as the fidelity check
        latent_model = copy.deepcopy(tokenizer).to(device).eval()
        gains = accepted_decoder_gains(store, args.tokenizer_key)
        if args.translator_run == TRANSLATOR_RUN:
            tkey = json.loads(store.get(f"training-runs/{args.translator_run}/best.json"))["checkpoint_key"]
            translator, kind = load_translator(store, tkey, TRANSLATOR_MANIFEST_SHA)
        else:
            translator, kind, tkey, _ = load_translator_run(store, args.translator_run, TRANSLATOR_MANIFEST_SHA)
        tsha = hashlib.sha256(store.get(tkey)).hexdigest()
        if args.translator_sha256 and tsha != args.translator_sha256:
            raise SystemExit(f"{tkey}: SHA-256 {tsha}, expected {args.translator_sha256}")
        translator = translator.to(device).eval()
        report["instrument"] = dict(run=args.translator_run, checkpoint_key=tkey, sha256=tsha, kind=kind)
        report["tokenizer"] = dict(key=args.tokenizer_key, sha256=args.tokenizer_sha256, gains=gains,
                                   stages=tokenizer.config.stages, parts=[p for p, _ in tokenizer.parts],
                                   code_dim=tokenizer.config.code_dim, step=tokenizer.config.step)
        candidates = M.retrieval_candidates(len(texts), args.candidates, seed=0)
        report["retrieval_candidates"] = dict(k=int(candidates.shape[1]), seed=0)
        parts = [p for p, _ in tokenizer.parts]
        anchors, table, rho_ref = {}, {}, {}

        def compute(states):
            """The raw reads of one form: hypotheses, the (n, k) candidate NLL matrix, d2 latents, duration ratio."""
            t0 = time.time()
            hyps = translate(translator, kind, states)
            nll = candidate_nll(translator, translator_items(translator, kind, states), texts, candidates,
                                device=device, group=args.nll_group)
            ratio = float(np.mean([(s.timestamps[-1] - s.timestamps[0]) / max(r.timestamps[-1] - r.timestamps[0], 1e-6)
                                   for s, r in zip(states, refs)]))
            return dict(hyps=hyps, nll=nll, lat=d2_latents(latent_model, states, device), ratio=ratio, t0=t0)

        def panel(name, raw, source):
            entry = dict(source=source, **text_panel(raw["hyps"], texts, anchors))
            entry["hypotheses"] = raw["hyps"]
            entry["seconds_over_reference"] = round(raw["ratio"], 4)
            lik = likelihood_panel(raw["nll"], blank_raw["nll"], anchors)
            rho = rho_panel(raw["lat"], rho_ref["latents"], rho_ref["mean"], rho_ref["std"], parts,
                            tokenizer.config.code_dim, K=args.rho_references, device=str(device),
                            per_part=not args.no_rho_parts)
            per_clip = lik.pop("per_clip")
            per_clip["rho_dtw"] = rho.pop("per_clip_dtw")
            entry.update(lik, rho_faith=rho, per_clip=per_clip, seconds=round(time.time() - raw["t0"], 1))
            table[name] = dict(stats=M.bleu_statistics(raw["hyps"], texts), delta_nll=np.asarray(per_clip["delta_nll"]),
                               rank=np.asarray(per_clip["rank"]), rank_pmi=np.asarray(per_clip["rank_pmi"]))
            print(json.dumps({name: dict(bleu=entry["bleu"]["bleu"], chrf=entry["chrf"]["chrf"],
                                         delta_nll=entry["delta_nll"]["mean"], top1=entry["retrieval"]["nll"]["top1"],
                                         rho_dtw=entry["rho_faith"]["dtw"]["mean"], seconds=entry["seconds"])}),
                  flush=True)
            return entry

        real_raw = compute(build_form("real", refs, tokenizer, gains))
        blank_raw = compute(build_form("blank", refs, tokenizer, gains))
        rho_ref["latents"] = real_raw["lat"]
        rho_ref["mean"], rho_ref["std"] = M.standardiser(real_raw["lat"])
        for name, raw in (("real", real_raw), ("blank", blank_raw)):
            stats = M.bleu_statistics(raw["hyps"], texts)
            anchors[name] = dict(hypotheses=raw["hyps"], stats=stats, bleu=float(M.bleu_from_statistics(stats.sum(0))),
                                 chrf=M.corpus_chrf(raw["hyps"], texts),
                                 delta_nll=M.delta_nll(blank_raw["nll"][:, 0], raw["nll"][:, 0]),
                                 rank=M.mid_ranks(raw["nll"]))
        for name, raw in (("real", real_raw), ("blank", blank_raw)):
            report["forms"][name] = panel(name, raw, "computed")
        del blank_raw["lat"]
        for name in forms[2:]:
            raw = compute(build_form(name, refs, tokenizer, gains))
            report["forms"][name] = panel(name, raw, "computed")
            del raw
        for key in args.regenerate:
            gens, pilot, pilot_recs, info = regenerate(store, key, len(texts), args.device, load_mix, load_states)
            if [r["sign"] for r in pilot_recs] != report["signs"]:
                raise SystemExit(f"{key}: the pilot's clips are not the ceiling read's clips")
            label = f"regen:{short_name(key)}"
            entry = panel(label, compute(gens[:len(texts)]), key)
            stored_hyps = pilot["meaning"]["generated"]["hypotheses"][:len(texts)]
            entry["reproduction"] = dict(report_sha256=hashlib.sha256(store.get(key)).hexdigest(),
                pilot_mix=dict(key=info["manifest_key"], sha256=info["manifest_sha256"],
                               clips=len(pilot["meaning"]["references"])),
                identical=round(float(np.mean([a == b for a, b in zip(entry["hypotheses"], stored_hyps)])), 4),
                stored_bleu=M.bleu_ci_fast(stored_hyps, texts)[0], regenerated_bleu=entry["bleu"]["bleu"],
                checkpoint_key=pilot["checkpoint_key"], guidance=pilot["guidance"],
                part_temperature=pilot["part_temperature"], generated_blocks=math.ceil(len(texts) / BLOCK))
            report["regenerated"][key] = label
            report["forms"][label] = entry
            del gens
        report["merged"] = {}
        for key in args.merge_from:                     # forms of an earlier run join the orderings
            raw = Path(key[5:]).read_bytes() if key.startswith("file:") else store.get(key)
            earlier = json.loads(raw)
            if earlier.get("references") != texts or earlier.get("instrument", {}).get("sha256") != tsha:
                raise SystemExit(f"--merge-from {key}: other clips or another instrument")
            blank_then = np.asarray(earlier["forms"]["blank"]["per_clip"]["nll"])
            drift = float(np.abs(blank_then - blank_raw["nll"][:, 0]).max())
            added = []
            for name, e in earlier["forms"].items():
                if name in table or "per_clip" not in e:
                    continue
                table[name] = dict(stats=M.bleu_statistics(e["hypotheses"], texts),
                                   delta_nll=np.asarray(e["per_clip"]["delta_nll"]),
                                   rank=np.asarray(e["per_clip"]["rank"]), rank_pmi=np.asarray(e["per_clip"]["rank_pmi"]))
                added.append(name)
            report["merged"][key] = dict(forms=added, blank_nll_max_abs_difference=round(drift, 6),
                                         sha256=hashlib.sha256(raw).hexdigest())
        report["orderings"] = orderings({k: v for k, v in table.items()})
    for key, (refs_stored, hyp_forms) in stored.items():
        if refs_stored[:len(texts)] != texts:            # e.g. a 300-clip pilot (another random.sample draw)
            report["stored"][key] = dict(sha256=stored_sha[key], skipped=f"its references are not these {len(texts)} "
                                                                           f"clips ({len(refs_stored)} stored)")
            print(json.dumps({short_name(key): "skipped: other clips"}), flush=True)
            continue
        block = {}
        for form, hyps in hyp_forms.items():
            block[form] = text_panel(hyps[:len(texts)], texts, anchors)
            mine = report["forms"].get(form, {}).get("hypotheses") if form in ("real", "blank") else None
            if mine is not None and report["forms"][form].get("source") == "computed":   # the instrument reproduces
                block[form]["identical_to_computed"] = round(float(np.mean([a == b for a, b in zip(hyps, mine)])), 4)
        report["stored"][key] = dict(sha256=stored_sha[key], forms=block)
        print(json.dumps({short_name(key): {f: v["bleu"]["bleu"] for f, v in block.items()}}), flush=True)
    report["seconds"] = round(time.time() - started, 1)
    payload = json.dumps(report, indent=1, sort_keys=True, default=float).encode()
    if args.output_key:
        store.put(args.output_key, payload)
    else:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
