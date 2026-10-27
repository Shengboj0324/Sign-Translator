"""Where do generated signs stop being readable? Instrument parity and exposure bias (2026-10-27 identification test 2).

The prior reads its text under teacher forcing (word retrieval top-1 0.67-0.91 of 64) and its sampler reproduces
teacher forcing exactly, yet free-run samples of its training words read at about 0 against 0.925 for real clips.
For each of the ``--words`` most frequent one-word training texts (as ``lexicon_diagnostic.py``), one real training
clip is read by the recogniser in six forms:
- ``real``: the conformed training state, as the dataset serves it;
- ``recon_full``: its own tokenizer codes decoded, in the generator's form (every trained channel valid, decoder
  gains, head smoothing): the best a generator can do through this tokenizer, presented as generations are;
- ``recon_masked``: the same, with each channel's validity copied from the real clip (instrument parity: if this is
  far above ``recon_full``, the recogniser misreads the generator's form, e.g. valid legs it never saw);
- ``onestep_sample`` / ``onestep_argmax``: every step's 32 codes drawn (or taken as the mode) from the prior given the
  REAL previous codes and the text (teacher-forced history, no accumulated error), then decoded as ``recon_full``;
- ``free_run``: ordinary sampling at the clip's duration (guidance ``--guidance``).
If one-step samples read well but free-run does not, errors accumulating over the sampled history (exposure bias)
are the cause. Prompts use each record's own text. Writes one JSON report.

``--extra`` adds two free-run variants that locate the drift and test an inference-only lever:
- ``prefix25`` / ``prefix50`` / ``prefix75``: the first 25/50/75% of the steps are the real codes, the rest is
  sampled (guidance ``--guidance``, generation's planned loci and empty-text branch);
- ``hold25`` / ``hold50`` / ``hold75``: the control for the prefixes: the real codes up to the same step, then the
  last real step's codes repeated (a held pose, no generated continuation). A prefix form only shows the model
  continuing the sign where it reads above its hold form;
- ``coarse_sampled``: free run where only the first residual stage of each part is sampled and the finer stages are
  their mode (their sampled noise otherwise enters every later step's history).
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-key", required=True)
    p.add_argument("--mix-key", required=True)
    p.add_argument("--mix-sha256", required=True)
    p.add_argument("--corpus", action="append", required=True)
    p.add_argument("--words", type=int, default=60)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra", action="store_true", help="add the prefix and coarse-sampled free-run variants")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("ignore")
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.data.text_motion_dataset import R2Store, TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig, generate_rvq_motion
    from signtranslator.generation.postprocess import smooth_rotation_channel, smooth_vector_channel
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import (
        PAD, assemble_state, batch_seconds, diffusion_targets, empty_loci, encode_texts, generation_loci, native_values,
        unit_to_raw)
    from signtranslator.grammar_motion.loci_conditioning import state_loci
    from evaluate_text_to_motion import accepted_decoder_gains
    from fit_tokenizer_gains import apply_gains
    from train_motion_tokens import layout_template
    from train_text_to_motion import to_device
    from validate_sign_recognizer import load_model, logits_of
    store = R2Store(os.environ)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    ck = torch.load(io.BytesIO(store.get(args.checkpoint_key)), map_location="cpu", weights_only=False)
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"])); prior.load_state_dict(ck.get("ema") or ck["model"])
    tokenizer = MotionRVQ(prior.config.tokenizer); tokenizer.load_state_dict(ck["tokenizer_state"])
    device = torch.device(args.device)
    prior.to(device).eval(); tokenizer.to(device).eval()
    gains = accepted_decoder_gains(store, ck.get("tokenizer_key"))
    motion_config = prior.config.tokenizer.motion
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 8 * 2 ** 30)
    template = layout_template(manifest, "how2sign-two-view-text-v1", cache, store.get)
    data = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get,
                             max_seconds=motion_config.max_seconds, fps=24., template=template)
    recogniser, labels, _ = load_model(store, args.recognizer_run, args.device)
    index = {g: i for i, g in enumerate(labels)}

    def post(s):
        s = apply_gains(s, gains) if gains else s
        s = smooth_rotation_channel(s, "head_rotation", .1)
        return smooth_vector_channel(s, "head_translation", .1) if "head_translation" in s.channels else s

    @torch.no_grad()
    def decode(codes, frames, text):
        unit = tokenizer.decode_codes(codes, frames)
        raw = unit_to_raw(motion_config, unit.double())
        native = native_values(motion_config, {key: v[0] for key, v in raw.items()})
        return post(assemble_state(motion_config, text, native, frames, 24., "diag:0", "diag-clock",
                                   method="motion_tokens_v2"))

    def masked_like(state, real):
        def mask(c, valid):
            v = valid[..., None] if c.values.ndim == valid.ndim + 1 else valid
            return replace(c, valid=valid, values=np.where(v, c.values, 0.), observed=c.observed & valid,
                           confidence=np.where(valid, c.confidence, 0.))
        channels = {n: mask(c, real.channels[n].valid.copy() if n in real.channels and
                            real.channels[n].valid.shape == c.valid.shape else np.zeros_like(c.valid))
                    for n, c in state.channels.items()}
        return replace(state, channels=channels)

    P = len(tokenizer.parts)

    @torch.no_grad()
    def continue_from(text, prefix, count, seed, greedy_fine=False):
        """Free run of ``count`` steps after the real codes ``prefix`` (1, k, J), as ``generate_rvq_motion`` samples."""
        config = prior.config
        tk, mk = (t.to(device) for t in encode_texts([text]))
        hid, _, mk = prior.encode(tk, mk, generation_loci(config, [text], None, device))
        empty = _empty_like(tk[:1, :2])
        e_hid, _, e_mask = prior.encode(empty, empty != PAD, empty_loci(config, device))
        seconds_out = torch.tensor([count / 24.], device=device)
        clock = (torch.arange(count, device=device, dtype=torch.float32) / 24.)[None]
        g = torch.Generator().manual_seed(seed)
        sequence = torch.cat((torch.full((1, 1, prior.J), prior.K, dtype=torch.long, device=device), prefix), 1)
        for k in range(prefix.shape[1], count):
            h = prior.temporal_states(hid, mk, sequence, clock[:, :k + 1], seconds_out)[:, -1]
            h_free = (prior.temporal_states(e_hid, e_mask, sequence, clock[:, :k + 1], seconds_out)[:, -1]
                      if args.guidance != 1. else None)
            step_codes = torch.zeros((1, 0), dtype=torch.long, device=device)
            for j in range(prior.J):
                logits = prior.depth_logits(h, step_codes)[:, -1]
                if h_free is not None:
                    free = prior.depth_logits(h_free, step_codes)[:, -1]
                    logits = free + args.guidance * (logits - free)
                if greedy_fine and j >= P:                          # stage >= 2 of every part: its mode
                    choice = logits.argmax(-1, keepdim=True)
                else:
                    choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=g).to(device)
                step_codes = torch.cat((step_codes, choice), 1)
            sequence = torch.cat((sequence, step_codes[:, None]), 1)
        return decode(sequence[:, 1:], count, text)

    by_word = collections.defaultdict(list)
    for k, r in enumerate(data.records):
        if r.get("corpus") in args.corpus and r.get("text"):
            label = r["text"].upper().replace(" ", "")
            if label in index:
                by_word[label].append(k)
    words = sorted(by_word, key=lambda w: (-len(by_word[w]), w))[:args.words]
    forms = collections.defaultdict(list)
    y, rows = [], []
    generator = torch.Generator().manual_seed(args.seed)
    for n, w in enumerate(words):
        text, real = data[by_word[w][0]]
        motion = to_device((torch.zeros(0), torch.zeros(0), collate_multichannel([real]), None), device)[2]
        frames = int(motion.frame_valid.shape[1])
        with torch.no_grad():
            unit = diffusion_targets(motion_config, motion)
            _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))                       # (1, S, J)
            tokens, mask = (t.to(device) for t in encode_texts([text]))
            loci = (torch.as_tensor(state_loci(real)[None], dtype=torch.float32, device=device)
                    if prior.config.locus_conditioning else None)
            hidden, _, mask = prior.encode(tokens, mask, loci)
            previous = torch.cat((torch.full_like(codes[:, :1], prior.K), codes[:, :-1]), 1)
            seconds = batch_seconds(motion).float()
            duration = (seconds * motion.frame_valid).amax(1)
            h = prior.temporal_states(hidden, mask, previous, seconds[:, :codes.shape[1]], duration)[0]  # (S, d)
            drawn = torch.zeros((h.shape[0], 0), dtype=torch.long, device=device)
            modes = torch.zeros((h.shape[0], 0), dtype=torch.long, device=device)
            for j in range(prior.J):
                logits = prior.depth_logits(h, drawn)[:, -1]
                choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator)
                drawn = torch.cat((drawn, choice.to(device)), 1)
                modes = torch.cat((modes, prior.depth_logits(h, modes)[:, -1].argmax(-1, keepdim=True)), 1)
            recon = decode(codes, frames, text)
            forms["real"].append(real)
            forms["recon_full"].append(recon)
            forms["recon_masked"].append(masked_like(recon, real))
            forms["onestep_sample"].append(decode(drawn[None], frames, text))
            forms["onestep_argmax"].append(decode(modes[None], frames, text))
        seconds_real = frames / 24.
        forms["free_run"].append(post(generate_rvq_motion(prior, tokenizer, [text], fps=24., durations=[seconds_real],
                                                          guidance=args.guidance, seed=9500 + 10 * n)[0]))
        if args.extra:
            for f in (.25, .5, .75):
                k = int(round(f * frames))
                forms[f"prefix{int(100 * f)}"].append(continue_from(text, codes[:, :k], frames, 9600 + 10 * n))
                held = torch.cat((codes[:, :k], codes[:, k - 1:k].expand(-1, frames - k, -1)), 1)
                forms[f"hold{int(100 * f)}"].append(decode(held, frames, text))
            forms["coarse_sampled"].append(continue_from(text, codes[:, :0], frames, 9800 + 10 * n, greedy_fine=True))
        y.append(index[w])
        rows.append(dict(word=w, text=text, frames=frames,
                         onestep_code_agreement=round(float((drawn == codes[0]).float().mean()), 4),
                         argmax_code_agreement=round(float((modes == codes[0]).float().mean()), 4)))
    target = torch.tensor(y)
    report = dict(checkpoint_key=args.checkpoint_key, recognizer_run=args.recognizer_run, words=len(words),
                  guidance=args.guidance, decoder_gains=gains, top5={}, top1={})
    for name, states in forms.items():
        logits = logits_of(recogniser, states, args.device)
        hits = (logits.topk(5, dim=1).indices == target[:, None]).any(1).float().numpy()
        report["top5"][name] = round(float(hits.mean()), 4)
        report["top1"][name] = round(float((logits.argmax(1) == target).float().mean()), 4)
        for k, r in enumerate(rows):
            r[f"{name}_hit"] = int(hits[k])
    def activity(state, names=("body", "left_hand", "right_hand", "head_rotation")):
        """Mean absolute frame-to-frame change per channel (per second), over frames valid in both."""
        out = {}
        for name in names:
            c = state.channels.get(name)
            if c is None or len(c.values) < 2:
                continue
            ok = c.valid[1:] & c.valid[:-1]
            if not ok.any():
                continue
            d = np.abs(np.diff(c.values, axis=0))
            d = d.reshape(*d.shape[:2], -1).mean(-1) if d.ndim > 2 else d
            out[name] = float(d[ok].mean() * 24.)
        return out
    report["motion_activity"] = {}
    for name, states in forms.items():
        acts = [activity(st) for st in states]
        report["motion_activity"][name] = {ch: round(float(np.mean([a[ch] for a in acts if ch in a])), 4)
                                           for ch in ("body", "left_hand", "right_hand", "head_rotation")
                                           if any(ch in a for a in acts)}
    report["onestep_code_agreement"] = round(float(np.mean([r["onestep_code_agreement"] for r in rows])), 4)
    report["argmax_code_agreement"] = round(float(np.mean([r["argmax_code_agreement"] for r in rows])), 4)
    print(json.dumps({k: v for k, v in report.items() if k != "decoder_gains"}), flush=True)
    report["per_word"] = rows
    store.put(args.output_key, json.dumps(report, indent=1).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
