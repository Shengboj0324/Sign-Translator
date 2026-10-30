"""Road to 3.8, part 2, stage S1: where is the text bottleneck? Per-step KL(p(code | text) ‖ p(code | empty text)) on
a prior's stage-1 codes, teacher forced on real How2Sign val clips (evaluation only, no training).

Declared in docs/NEXT_PHASE_PLAN.md, "Road to 3.8, part 2" (2026-10-30, 19:32 UTC): the round-4 flan-T5 and MiniLM
priors against the byte prior. If the pretrained-encoder priors show the same small text influence as the byte prior,
the decoder (which predicts the next step from its own history) is the bottleneck, not the text encoder.

**Clips:** ``pilot_content_check.validation_sentences(round10-mixed-v1, --clips)`` (the pilots' 1,000), at 24 fps; clips
longer than the prior's ``max_seconds`` (training drops them) are left out and counted.

**Measure.** For each clip, d2's codes of the real clip (the prior's own tokenizer, ``tokenizer_state``) are the
teacher-forced history. The prior's logits for every code position are computed twice with the same history: with the
clip's text (and the loci generation would use: ``generation_loci``, planned from the text; ``--loci``) and with the
empty text (``_empty_like``, ``empty_loci``: classifier-free guidance's unconditional branch). Stage-1 codes are the
first P depth positions (j = q·P + p with q = 0). Reported per part and overall, mean over valid steps:
- ``kl``: KL(p_text ‖ p_empty) in nats per stage-1 code (the full distribution, not just the true code);
- ``gain``: log p_text(true code) − log p_empty(true code) (``text_use_check.py``'s text gain, restricted to stage 1;
  the comparable measure for round 6's "0.02–0.07 nats");
- the entropies of p_text and p_empty (the scale the KL is read against), and the all-stage KL for context;
- the part-valid weighted variants (codes whose part was observed, ``part_code_weights``: the priors' training weights);
- per clip: mean KL per stage-1 code, summed stage-1 KL in nats per clip and per second.
Per-token KL is bounded by the text's information spread over all codes (about H(sentence) / codes), so the per-clip
sum is the scale to compare with the 64-way word retrieval (``word_retrieval_check.py``, run beside this in
``runpod/fleet/s0s1_probes.sh``).

**Side by side:** paired per-clip differences of every prior's mean stage-1 KL against the first ``--checkpoint``
(the reference, by default W14b), and against the byte prior at the matched step (``byte24k``, round 3d's recipe on
the same mix as the round-4 arms).

Write-once: ``--output-key`` or ``--output-file``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
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
D2 = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
DEFAULT_CHECKPOINTS = (
    "w14b=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt"
    "@0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563",
    "byte24k=training-runs/tokens-rvq-d0b-prior-r3d-d2g/checkpoints/step-00024000.pt",
    "flant5=training-runs/tokens-rvq-d0b-prior-r4-flant5/checkpoints/step-00023000.pt",
    "minilm=training-runs/tokens-rvq-d0b-prior-r4-minilm/checkpoints/step-00024000.pt",
)
FPS = 24.


def parse_checkpoint(spec: str) -> tuple[str, str, str | None]:
    """'label=key[@sha256]' -> (label, key, sha256 or None)."""
    label, sep, rest = spec.partition("=")
    if not sep or not label or not rest:
        raise SystemExit(f"--checkpoint {spec!r}: expected label=key[@sha256]")
    key, _, sha = rest.partition("@")
    return label, key, sha or None


def stage_logits(prior, tokenizer, tokens, text_mask, motion, loci=None):
    """Teacher-forced logits of every code position on the frozen tokenizer's codes of ``motion``:
    (logits (N, J, K) of the valid steps, target codes (N, J), row of each step (N,), part-valid weights (N, J)).
    ``rvq_sequence_log_likelihood``'s path: no dropout, no corruption; summing log-softmax at the targets over J and
    steps gives its value (tested)."""
    import torch
    import torch.nn.functional as F
    from signtranslator.generation.motion_rvq import part_code_weights
    from signtranslator.generation.text_to_motion import batch_seconds, diffusion_targets
    config = prior.config
    if config.plan_window:
        raise SystemExit("plan-conditioned priors are not read by S1")
    with torch.no_grad():
        tokenizer.eval()
        unit = diffusion_targets(config.tokenizer.motion, motion)
        latents = tokenizer.encode(unit)
        _, codes, _ = tokenizer.quantize(latents)
        valid = tokenizer.token_valid(motion.frame_valid)
        hidden, _, mask = prior.encode(tokens, text_mask, loci)
        start = torch.full_like(codes[:, :1], prior.K)
        previous = torch.cat((start, codes[:, :-1]), 1)
        seconds = batch_seconds(motion).float()
        step_seconds = seconds[:, ::config.tokenizer.step][:, :codes.shape[1]]
        step_seconds = F.pad(step_seconds, (0, codes.shape[1] - step_seconds.shape[1]))
        duration = (seconds * motion.frame_valid).amax(1)
        h = prior.temporal_states(hidden, mask, previous, step_seconds, duration)
        target = codes[valid]
        logits = prior.depth_logits(h[valid], target[:, :-1])
        row = torch.arange(len(codes), device=codes.device)[:, None].expand_as(valid)[valid]
        weights = part_code_weights(tokenizer, config, motion, valid)
    return logits, target, row, weights


def clip_reads(prior, tokenizer, texts, states, *, loci_mode: str, device):
    """Per valid step of every clip: stage-1 KL, gain, entropies (N, P), all-stage KL (N,), rows (N,), weights (N, P)."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.motion_tokens import _empty_like
    from signtranslator.generation.text_to_motion import PAD, empty_loci, encode_texts, generation_loci
    from signtranslator.grammar_motion.loci_conditioning import state_loci
    from train_text_to_motion import to_device
    P = len(tokenizer.parts)
    tokens, _ = encode_texts(texts)
    motion = collate_multichannel(states)
    tokens, mask, motion = to_device((tokens, tokens != PAD, motion, None), device)[:3]
    config = prior.config
    loci = None
    if config.locus_conditioning:
        if loci_mode == "plan":
            loci = generation_loci(config, texts, None, device)
        elif loci_mode == "state":
            loci = torch.as_tensor(np.stack([state_loci(s) for s in states]), dtype=torch.float32, device=device)
        else:
            loci = empty_loci(config, device).expand(len(texts), -1)
    empty = _empty_like(tokens)
    empty_l = empty_loci(config, device).expand(len(texts), -1) if config.locus_conditioning else None
    lt, target, row, weights = stage_logits(prior, tokenizer, tokens, mask, motion, loci)
    le, target_e, _, _ = stage_logits(prior, tokenizer, empty, empty != PAD, motion, empty_l)
    if not torch.equal(target, target_e):
        raise RuntimeError("the two branches must read the same codes")
    kl = M.categorical_kl(lt[:, :P], le[:, :P])
    gain = M.true_code_gain(lt[:, :P], le[:, :P], target[:, :P])
    ent = lambda x: -(x.double().log_softmax(-1).exp() * x.double().log_softmax(-1)).sum(-1)
    return dict(kl=kl.cpu().numpy(), gain=gain.cpu().numpy(), entropy_text=ent(lt[:, :P]).cpu().numpy(),
                entropy_empty=ent(le[:, :P]).cpu().numpy(), kl_all=M.categorical_kl(lt, le).mean(1).cpu().numpy(),
                row=row.cpu().numpy(), weights=weights[:, :P].float().cpu().numpy())


def summarise(reads: dict, parts, seconds) -> dict:
    """Aggregate the concatenated step reads (``row`` = clip index) into per-part, overall and per-clip numbers."""
    kl, gain, w, row = reads["kl"], reads["gain"], reads["weights"], reads["row"]
    n = len(seconds)
    wmean = lambda x, ww: float((x * ww).sum() / max(ww.sum(), 1e-12))
    per_part = {}
    for p, part in enumerate(parts):
        per_part[part] = dict(kl=round(float(kl[:, p].mean()), 5), kl_part_valid=round(wmean(kl[:, p], w[:, p]), 5),
                              gain=round(float(gain[:, p].mean()), 5),
                              gain_part_valid=round(wmean(gain[:, p], w[:, p]), 5),
                              entropy_text=round(float(reads["entropy_text"][:, p].mean()), 4),
                              entropy_empty=round(float(reads["entropy_empty"][:, p].mean()), 4),
                              observed_fraction=round(float(w[:, p].mean()), 4))
    clip_kl = np.array([kl[row == i].mean() if (row == i).any() else np.nan for i in range(n)])
    clip_sum = np.array([kl[row == i].sum() for i in range(n)])
    clip_gain = np.array([gain[row == i].mean() if (row == i).any() else np.nan for i in range(n)])
    ok = np.isfinite(clip_kl)
    sec = np.asarray(seconds, np.float64)
    return dict(steps=int(len(kl)), stage1_codes=int(kl.size), per_part=per_part,
                kl=round(float(kl.mean()), 5), kl_part_valid=round(wmean(kl, w), 5),
                gain=round(float(gain.mean()), 5), gain_part_valid=round(wmean(gain, w), 5),
                entropy_text=round(float(reads["entropy_text"].mean()), 4),
                entropy_empty=round(float(reads["entropy_empty"].mean()), 4),
                kl_all_stages=round(float(reads["kl_all"].mean()), 5),
                per_clip_summary=dict(mean_kl=M.bootstrap_mean(clip_kl[ok]),
                                      median_kl=round(float(np.median(clip_kl[ok])), 5),
                                      p90_kl=round(float(np.percentile(clip_kl[ok], 90)), 5),
                                      summed_kl_nats=M.bootstrap_mean(clip_sum[ok]),
                                      kl_nats_per_second=M.bootstrap_mean(clip_sum[ok] / sec[ok]),
                                      mean_gain=M.bootstrap_mean(clip_gain[ok])),
                per_clip=dict(mean_kl=np.round(clip_kl, 5).tolist(), summed_kl=np.round(clip_sum, 3).tolist(),
                              mean_gain=np.round(clip_gain, 5).tolist()))


def load_prior(store, key: str, sha: str | None, device):
    import torch
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig
    payload = store.get(key)
    digest = hashlib.sha256(payload).hexdigest()
    if sha and digest != sha:
        raise SystemExit(f"{key}: SHA-256 {digest}, expected {sha}")
    ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)   # our own checkpoint (R2)
    if ck.get("kind") != "motion_rvq_prior":
        raise SystemExit(f"{key} is not an RVQ prior")
    prior = RVQPrior(RVQPriorConfig.from_dict(ck["config"]))
    prior.load_state_dict(ck.get("ema") or ck["model"])
    tokenizer = MotionRVQ(prior.config.tokenizer)
    tokenizer.load_state_dict(ck["tokenizer_state"])
    info = dict(key=key, sha256=digest, weights="ema" if ck.get("ema") else "model", step=ck.get("step"),
                text_model=prior.config.text_model, locus_conditioning=bool(prior.config.locus_conditioning),
                tokenizer_key=ck.get("tokenizer_key"), manifest_sha256=ck.get("manifest_sha256"))
    return prior.to(device).eval(), tokenizer.to(device).eval(), info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--mix-key", default=EVAL_MIX)
    p.add_argument("--mix-sha256", default=EVAL_MIX_SHA)
    p.add_argument("--clips", type=int, default=1000)
    p.add_argument("--checkpoint", action="append", help="label=key[@sha256], repeatable; the first is the reference "
                   f"(default: {', '.join(c.split('=')[0] for c in DEFAULT_CHECKPOINTS)})")
    p.add_argument("--matched-byte", default="byte24k", help="the byte prior every pretrained-encoder prior is also "
                   "paired against (its label)")
    p.add_argument("--loci", choices=("plan", "state", "zero"), default="plan",
                   help="the text branch's locus slots (plan: generation's, from the text)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-key")
    p.add_argument("--output-file")
    args = p.parse_args(argv)
    if bool(args.output_key) == bool(args.output_file):
        raise SystemExit("give exactly one of --output-key and --output-file")
    import torch
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    from evaluate_text_to_motion import at_rate, load_state
    from pilot_content_check import validation_sentences
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists (write-once)")
    if args.output_file and Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists (write-once)")
    specs = [parse_checkpoint(c) for c in (args.checkpoint or DEFAULT_CHECKPOINTS)]
    if len({s[0] for s in specs}) != len(specs):
        raise SystemExit("checkpoint labels must be distinct")
    device = torch.device(args.device)
    started = time.time()
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.mix_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.mix_sha256)
    records = validation_sentences(manifest, args.clips)
    states = [at_rate(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), FPS)
              for r in records]
    report = dict(kind="s1-text-kl", settings=dict(vars(args)), clips_drawn=len(records), priors={})
    per_clip = {}
    for label, key, sha in specs:
        prior, tokenizer, info = load_prior(store, key, sha, device)
        if info["tokenizer_key"] not in (None, D2):
            print(json.dumps({label: f"note: tokenizer {info['tokenizer_key']}"}), flush=True)
        max_s = prior.config.tokenizer.motion.max_seconds
        keep = [i for i, s in enumerate(states) if max_s is None or s.timestamps[-1] - s.timestamps[0] <= max_s]
        reads = {k: [] for k in ("kl", "gain", "entropy_text", "entropy_empty", "kl_all", "row", "weights")}
        t0 = time.time()
        for b in range(0, len(keep), args.batch):
            idx = keep[b:b + args.batch]
            out = clip_reads(prior, tokenizer, [records[i]["text"] for i in idx], [states[i] for i in idx],
                             loci_mode=args.loci, device=device)
            out["row"] = np.asarray(idx)[out["row"]]
            for k, v in out.items():
                reads[k].append(v)
        reads = {k: np.concatenate(v, 0) for k, v in reads.items()}
        seconds = [float(s.timestamps[-1] - s.timestamps[0]) + 1 / FPS for s in states]
        summary = summarise(reads, [p for p, _ in tokenizer.parts], seconds)
        summary["clips"] = len(keep)
        summary["left_out_over_max_seconds"] = len(states) - len(keep)
        summary["seconds"] = round(time.time() - t0, 1)
        report["priors"][label] = dict(checkpoint=info, **summary)
        per_clip[label] = np.asarray(summary["per_clip"]["mean_kl"])
        print(json.dumps({label: {k: summary[k] for k in ("kl", "gain", "entropy_empty", "kl_all_stages")} |
                          {"summed_kl_nats": summary["per_clip_summary"]["summed_kl_nats"]["mean"]}}), flush=True)
        del prior, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    labels = [s[0] for s in specs]
    comparisons = {}
    for other in labels:
        for base in dict.fromkeys([labels[0], args.matched_byte]):
            if other == base or base not in per_clip:
                continue
            a, b = per_clip[other], per_clip[base]
            ok = np.isfinite(a) & np.isfinite(b)
            comparisons[f"{other}-vs-{base}"] = dict(
                mean_kl_difference=M.paired_mean_difference(a[ok], b[ok]),
                mean_kl_ratio=M.ratio_of_means(a[ok], b[ok]), clips=int(ok.sum()))
    report["comparisons"] = comparisons
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
