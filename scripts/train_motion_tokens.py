"""Train the motion-token generator: stage 1 tokenizer (VQ-VAE), stage 2 text prior.

``--architecture vq`` is tokens v1 (one codebook, causal token prior); ``rvq`` is
tokens v2 (part-wise residual VQ with the skeleton motion loss, depth-autoregressive
prior; ``generation/motion_rvq.py``).

**Batch noise.** A batch's loss varies with which clips it drew, so a single step's
loss is not a learning curve. Every ``--val-every`` steps the model is scored on the
same ``--val-batches`` batches of the validation partition (fixed order, fixed noise
seed, no text dropout), with an exponential moving average of the weights
(``--ema-decay``; codebooks, already EMA-updated, are taken from the live model).
The EMA weights are what checkpoints export for generation (``ema`` key), and the
checkpoint with the lowest validation loss is recorded in ``best.json``.

Same data, store and resume conventions as ``train_text_to_motion.py``:
checkpoints under ``training-runs/<run>/checkpoints/step-XXXXXXXX.pt``; the newest is
resumed and must match the configuration and manifest. Stage 2 freezes the stage 1
tokenizer named by ``--tokenizer-key`` (hash-pinned) and stores its weights inside
every prior checkpoint, so one prior checkpoint fully defines the generator.

**Round 17** (docs/NEXT_PHASE_PLAN.md, "Round 17 design"; ``generation/plan_codes.py``). ``--plan-tokenizer-key``
(SHA-pinned, a ``train_plan_tokenizer.py`` checkpoint over ``--tokenizer-key``):
- ``--stage prior``: the plan-conditioned detail prior, trained on the true plans of the batch with
  ``--plan-dropout``; validation also reports ``tokens_no_plan`` and ``plan_gain`` (the informativeness measure: the
  same codes under the empty plan). ``--init-key`` may name a plan-less prior of the same config (W14b): the plan
  modules then start at zero, so the warm-started prior equals it at step 0 (``plan_warm_start``).
- ``--stage plan-prior``: the text -> plan-code prior (kind ``motion_plan_prior``); ``--init-text-key`` copies the
  text encoder of an rvq prior (W14b).
Both store the plan tokenizer's key, SHA-256, config and weights in every checkpoint.
"""
from __future__ import annotations

import argparse
import collections
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy("file_system")   # many small tensors per batch exhaust fd passing ("received 0 items of ancdata")
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import TextMotionDataset, collate_text_motion  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest  # noqa: E402
from signtranslator.generation.canonical_state import channel_bindings_from_state  # noqa: E402
from signtranslator.generation.motion_rvq import (  # noqa: E402
    PLAN_FIELDS, MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig, rvq_loss, rvq_prior_loss,
)
from signtranslator.generation.plan_codes import (  # noqa: E402
    PLAN_PRIOR_KIND, PlanPriorConfig, copy_text_encoder, load_plan_tokenizer, plan_prior_loss,
)
from signtranslator.generation.motion_tokens import (  # noqa: E402
    MotionVQ, MotionVQConfig, TokenPrior, TokenPriorConfig, prior_loss, tokenizer_loss,
)
from signtranslator.generation.text_to_motion import TextToMotionConfig  # noqa: E402
from train_text_to_motion import lr_at, make_store, to_device  # noqa: E402
from signtranslator.training.weight_ema import WeightEMA  # noqa: E402


def kinds(architecture: str) -> tuple[str, str]:
    return ("motion_tokenizer", "motion_token_prior") if architecture == "vq" else \
        ("motion_rvq_tokenizer", "motion_rvq_prior")



def layout_template(manifest, corpus: str, cache, fetch):
    """The first training state of ``corpus`` in a mixed manifest: the layout every other corpus is conformed to."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    record = next((r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") == "train"), None)
    if record is None:
        raise SystemExit(f"no training record of corpus {corpus!r} to take the layout from")
    return load_sample(record, "continuous", cache=cache, fetch=fetch)


LOSS_ONLY_FIELDS = ("quantizer_dropout", "velocity_weights", "codebook_follows_lr")


def warm_start_compatible(saved: dict, new: dict) -> bool:
    """A tokenizer may warm-start another whose config differs only in loss-only fields (round 3c amendment 8): the
    architecture, codebooks and motion layout must be identical."""
    strip = lambda d: {k: v for k, v in d.items() if k not in LOSS_ONLY_FIELDS}
    return strip(saved) == strip(new)


def corpus_sample_weights(records, weights: dict[str, float]) -> list[float]:
    """Per-record sampling weights from per-corpus weights (``record["corpus"]``; corpora not named weigh 1). Round 3c:
    ASLLVD's single-word clips are up-weighted once ASL Citizen's (arms tracked on 14% of frames, so their tokenized
    targets have static hands) are dropped from the prior's mix. Weights must be positive and name known corpora."""
    known = {r.get("corpus") for r in records}
    unknown = set(weights) - known
    if unknown:
        raise SystemExit(f"--corpus-weight names corpora not in the training partition: {sorted(unknown)}")
    if any(w <= 0 for w in weights.values()):
        raise SystemExit("--corpus-weight values must be positive")
    return [float(weights.get(r.get("corpus"), 1.)) for r in records]


# RVQ prior fields that only add or reshape training-loss terms: the network (and so its state_dict) is identical with any
# value, so a warm start may change them (round 13: F12's weights with guided attention). history_stages, the text
# model, locus/grammar conditioning and every size stay part of the comparison: they change the forward pass.
PRIOR_TRAINING_ONLY = ("input_corruption", "scheduled_sampling", "history_dropout", "part_valid_loss", "guided_attention",
                       "guided_attention_width", "alignment_supervision", "rollout_blocks", "rollout_min", "rollout_max",
                       "rollout_onset", "rollout_guidance", "rollout_temperature", "plan_dropout")


def prior_warm_start_compatible(init_config: dict, config: dict) -> bool:
    """A prior may warm-start from ``init_config`` when the two configs agree on everything but training-only fields.

    Round 17: a plan-less prior may also warm-start a plan-conditioned one whose config differs only in the plan
    fields (``motion_rvq.PLAN_FIELDS``). The network then differs by exactly the plan modules, which start at zero, so
    the warm-started prior equals the plan-less one at step 0 (tested); ``plan_warm_start`` loads it."""
    strip = lambda c: {k: v for k, v in c.items() if k not in PRIOR_TRAINING_ONLY}
    if strip(init_config) == strip(config):
        return True
    if not init_config.get("plan_window") and config.get("plan_window"):
        plain = lambda c: {k: v for k, v in strip(c).items() if k not in PLAN_FIELDS}
        return plain(init_config) == plain(config)
    return False


def plan_warm_start(model, state: dict) -> int:
    """Load a plan-less prior's weights into a plan-conditioned ``model``; returns the number of plan keys left at
    their zero initialisation. Refuses unless the missing keys are exactly the plan modules and none is unexpected."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) != model.plan_parameter_names():
        raise SystemExit(f"--init-key: not a plan-less version of this prior (missing {sorted(missing)[:4]}, "
                         f"unexpected {sorted(unexpected)[:4]})")
    return len(missing)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tokenizer", "prior", "plan-prior"), required=True,
                        help="plan-prior (round 17): the text -> plan-code prior over a plan tokenizer "
                             "(--plan-tokenizer-key); a prior with --plan-tokenizer-key is the plan-conditioned detail "
                             "prior")
    parser.add_argument("--architecture", choices=("vq", "rvq"), default="vq", help="tokens v1 (vq) or v2 (rvq)")
    parser.add_argument("--stages", type=int, default=4, help="rvq: residual stages per part")
    parser.add_argument("--codebook-follows-lr", action="store_true",
                        help="rvq: the codebooks' EMA rate follows the learning rate; no dead-code re-seeding below a "
                             "quarter of the peak (round 3d)")
    parser.add_argument("--skip-grad-spikes", type=float, default=0.,
                        help="skip the update (optimizer and EMA) of a step whose gradient norm exceeds this multiple of "
                             "the running median of the last 200 accepted steps (after 500 steps); 0 = off (round 3d)")
    parser.add_argument("--quantizer-dropout", type=float, default=.5,
                        help="rvq: probability a batch trains on a random number of stages (round 3c amendment 8)")
    parser.add_argument("--base-blocks", type=int, default=0, help="rvq: residual blocks at full frame rate")
    parser.add_argument("--depth-layers", type=int, default=2, help="rvq prior: depth transformer layers")
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after this many validations without a new best validation loss (0: never)")
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tokenizer-key", help="prior stage: stage 1 checkpoint")
    parser.add_argument("--tokenizer-sha256", help="prior stage: its SHA-256")
    parser.add_argument("--tokenizer-manifest-sha256",
                        help="prior stage: the manifest the tokenizer was trained on, when it is not --manifest-sha256 "
                             "(e.g. the full-corpus tokenizer under a prior trained on a mixed manifest)")
    parser.add_argument("--init-key", help="warm-start from this checkpoint's EMA weights: a prior of the same config, or "
                                           "(round 3c amendment 8) a tokenizer whose config differs at most in the "
                                           "loss-only fields quantizer_dropout and velocity_weights")
    parser.add_argument("--init-sha256", help="its SHA-256")
    parser.add_argument("--conform-to-corpus", help="mixed manifests: conform every state to this corpus's channel "
                                                    "layouts (data.layout_adapter)")
    parser.add_argument("--fps", type=float, help="resample states whose rate differs from this (mixed-rate corpora)")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--codebook", type=int, default=512)
    parser.add_argument("--down-layers", type=int, default=2)
    parser.add_argument("--locus-conditioning", action="store_true",
                        help="prior: condition on referent-locus slots measured in each clip's pointing")
    parser.add_argument("--no-grammar-features", dest="grammar_features", action="store_false",
                        help="do not condition the prior on the English's grammatical features")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--token-layers", type=int, default=6)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--cfg-dropout", type=float, default=.1)
    parser.add_argument("--text-model", help="prior: a frozen pretrained text encoder (Hugging Face name) instead of the "
                                             "byte embedding (round 4)")
    parser.add_argument("--text-model-revision", help="its pinned revision (commit hash)")
    parser.add_argument("--input-corruption", type=float, default=0.,
                        help="rvq prior: replace previous-code inputs by random codes at a per-sequence rate in [0, this] "
                             "(round 7: the prior must read the text; 0 = off)")
    parser.add_argument("--scheduled-sampling", type=float, default=0.,
                        help="rvq prior: replace previous-code inputs by the prior's own one-step samples at a "
                             "per-sequence rate in [0, this] (round 8 candidate, against exposure bias; 0 = off)")
    parser.add_argument("--history-dropout", type=float, default=0.,
                        help="rvq prior: with this probability per sequence, hide every history step before a random "
                             "cut behind the start code (round 9 candidate, against the copycat shortcut; 0 = off)")
    parser.add_argument("--rollout-blocks", type=float, default=0.,
                        help="rvq prior: with this probability per sequence, a contiguous block of the prior's own "
                             "free-running output replaces the history (round 16 candidate, on-policy; 0 = off)")
    parser.add_argument("--rollout-min", type=int, default=4, help="rvq prior: shortest rollout block (steps)")
    parser.add_argument("--rollout-max", type=int, default=12, help="rvq prior: longest rollout block (steps)")
    parser.add_argument("--rollout-onset", type=float, default=.25,
                        help="rvq prior: probability that a rollout block starts at the first step")
    parser.add_argument("--rollout-guidance", type=float, default=1.5,
                        help="rvq prior: the rollout's guidance against the empty text (as generation's)")
    parser.add_argument("--rollout-temperature", type=float, default=1.1, help="rvq prior: the rollout's temperature")
    parser.add_argument("--part-valid-loss", action="store_true",
                        help="rvq prior: count a code's cross entropy only where its part was observed (round 10)")
    parser.add_argument("--history-stages", type=int, default=0,
                        help="rvq prior: the temporal transformer sees only the first n residual stages of each part "
                             "of the previous step (0 = all)")
    parser.add_argument("--guided-attention", type=float, default=0.,
                        help="rvq prior: weight of the guided-attention loss (Tachibana et al. 2017) on the temporal "
                             "decoder's cross-attention to the text, toward a monotonic diagonal (round 13; 0 = off)")
    parser.add_argument("--guided-attention-width", type=float, default=.2,
                        help="rvq prior: its width g (fraction of the text and of the utterance)")
    parser.add_argument("--alignment-supervision", type=float, default=0.,
                        help="rvq prior: weight of the cross entropy pulling each step's cross-attention to the bytes "
                             "of the word signed there, on records that carry sign_spans (round 13; 0 = off)")
    parser.add_argument("--plan-tokenizer-key", help="round 17: a plan tokenizer checkpoint (train_plan_tokenizer.py) "
                        "over --tokenizer-key; the prior stage conditions on its true plans, the plan-prior stage "
                        "predicts them")
    parser.add_argument("--plan-tokenizer-sha256", help="its SHA-256")
    parser.add_argument("--plan-tokenizer-manifest-sha256",
                        help="the manifest the plan tokenizer was trained on, when it is not --manifest-sha256")
    parser.add_argument("--plan-dropout", type=float, default=0.,
                        help="round 17 detail prior: probability a sequence trains with the empty plan (0 = off)")
    parser.add_argument("--no-plan-memory", dest="plan_memory", action="store_false",
                        help="round 17 detail prior: only the step-rate plan input, no plan cross-attention readers")
    parser.add_argument("--init-text-key", help="round 17 plan prior: copy the text encoder (embedding, projections, "
                        "text layers, duration head) of this rvq prior checkpoint's EMA weights (W14b)")
    parser.add_argument("--init-text-sha256", help="its SHA-256")
    parser.add_argument("--max-seconds", type=float, default=20.)
    parser.add_argument("--velocity-cap", type=float, default=0.,
                        help="rvq tokenizer: log-compress any velocity term above this value (round 10; 0 = off)")
    parser.add_argument("--velocity-weight", action="append", default=[],
                        help="rvq tokenizer: CHANNEL=W (repeatable), per-channel velocity matching "
                             "(motion_rvq.velocity_matching_loss); round 3c: head_rotation, head_translation, hands, body")
    parser.add_argument("--corpus-weight", action="append", default=[],
                        help="CORPUS=W (repeatable): sample that corpus's training records W times as often "
                             "(with replacement, epoch length unchanged); round 3c: asllvd-text-v1=3")
    parser.add_argument("--trained-channels", default="body,left_hand,right_hand,head_rotation,face,blink,root_translation,head_translation,left_gaze,right_gaze")
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-bytes", type=int, default=8 * 2 ** 30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.device == "mps":
        raise SystemExit("MPS is refused: it miscomputes rotation operations at batch scale")
    prior_stage = args.stage in ("prior", "plan-prior")
    if args.stage == "plan-prior" and (args.architecture != "rvq" or not args.plan_tokenizer_key):
        raise SystemExit("--stage plan-prior needs --architecture rvq and --plan-tokenizer-key")
    if args.plan_tokenizer_key and (not prior_stage or args.architecture != "rvq" or not args.plan_tokenizer_sha256):
        raise SystemExit("--plan-tokenizer-key needs an rvq prior stage and --plan-tokenizer-sha256")
    if args.plan_dropout and not (args.stage == "prior" and args.plan_tokenizer_key):
        raise SystemExit("--plan-dropout is for the plan-conditioned detail prior (--stage prior --plan-tokenizer-key)")
    if args.init_text_key and (args.stage != "plan-prior" or not args.init_text_sha256):
        raise SystemExit("--init-text-key is for --stage plan-prior and needs --init-text-sha256")
    store = make_store(args.store)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.json"
        path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    cache = ContentCache(Path(os.environ.get("SIGN_CACHE_DIR", tempfile.gettempdir())) / "sign-translator-cache",
                         args.cache_bytes)
    template = layout_template(manifest, args.conform_to_corpus, cache, store.get) if args.conform_to_corpus else None
    word_spans = prior_stage and args.architecture != "vq" and args.alignment_supervision > 0
    train = TextMotionDataset(manifest, partition="train", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                              fps=args.fps, template=template, word_spans=word_spans)
    if not len(train):
        raise SystemExit("the manifest has no training records")
    val = TextMotionDataset(manifest, partition="val", cache=cache, fetch=store.get, max_seconds=args.max_seconds,
                            fps=args.fps, template=template, word_spans=word_spans)
    tokenizer_kind, prior_kind = kinds(args.architecture)
    if args.stage == "plan-prior":
        prior_kind = PLAN_PRIOR_KIND
    plan, plan_state = None, None                   # round 17: the frozen plan tokenizer
    if args.stage == "tokenizer":
        motion = TextToMotionConfig(bindings=channel_bindings_from_state(train[0][1]),
                                    trained_channels=tuple(args.trained_channels.split(",")), max_seconds=args.max_seconds)
        if args.architecture == "vq":
            config = MotionVQConfig(motion, hidden=args.hidden, code_dim=args.code_dim, codebook=args.codebook,
                                    down_layers=args.down_layers)
            model = MotionVQ(config).to(device)
        else:
            config = MotionRVQConfig(motion, hidden=args.hidden, code_dim=args.code_dim, codebook=args.codebook,
                                     stages=args.stages, down_layers=args.down_layers, base_blocks=args.base_blocks,
                                     quantizer_dropout=args.quantizer_dropout, codebook_follows_lr=args.codebook_follows_lr,
                                     velocity_weights={k: float(v) for k, v in (x.split("=", 1) for x in args.velocity_weight)},
                                     velocity_cap=args.velocity_cap)
            model = MotionRVQ(config).to(device)
        tokenizer, tokenizer_state = None, None
    else:
        if not args.tokenizer_key or not args.tokenizer_sha256:
            raise SystemExit("the prior stage needs --tokenizer-key and --tokenizer-sha256")
        payload = store.get(args.tokenizer_key)
        if hashlib.sha256(payload).hexdigest() != args.tokenizer_sha256:
            raise SystemExit("the tokenizer checkpoint differs from --tokenizer-sha256")
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if saved.get("kind") != tokenizer_kind or saved["manifest_sha256"] != (args.tokenizer_manifest_sha256
                                                                              or args.manifest_sha256):
            raise SystemExit("--tokenizer-key is not a tokenizer trained on this manifest (or on "
                             "--tokenizer-manifest-sha256)")
        tokenizer_state = saved.get("ema") or saved["model"]       # the EMA weights, when the tokenizer has them
        if args.architecture == "vq":
            tokenizer = MotionVQ(MotionVQConfig.from_dict(saved["config"]))
        else:
            tokenizer = MotionRVQ(MotionRVQConfig.from_dict(saved["config"]))
        tokenizer.load_state_dict(tokenizer_state)
        tokenizer.to(device).eval().requires_grad_(False)
        if args.plan_tokenizer_key:
            try:
                plan, _, plan_saved = load_plan_tokenizer(store.get(args.plan_tokenizer_key), args.plan_tokenizer_sha256,
                                                          tokenizer_sha256=args.tokenizer_sha256, device=device)
            except ValueError as error:
                raise SystemExit(f"--plan-tokenizer-key: {error}")
            if plan_saved["manifest_sha256"] != (args.plan_tokenizer_manifest_sha256 or args.manifest_sha256):
                raise SystemExit("--plan-tokenizer-key was not trained on this manifest (or on "
                                 "--plan-tokenizer-manifest-sha256)")
            if plan.config.tokenizer.sha256() != tokenizer.config.sha256():
                raise SystemExit("--plan-tokenizer-key reads another tokenizer config than --tokenizer-key")
            plan_state = plan.state_dict()
        common = dict(d_model=args.d_model, heads=args.heads, text_layers=args.text_layers, feedforward=args.feedforward,
                      dropout=args.dropout, cfg_dropout=args.cfg_dropout, grammar_features=args.grammar_features,
                      locus_conditioning=args.locus_conditioning)
        if args.architecture == "vq":
            config = TokenPriorConfig(tokenizer.config, token_layers=args.token_layers, **common)
            model = TokenPrior(config).to(device)
        else:
            # round 17: the plan prior's "tokenizer" is the plan tokenizer; the detail prior gains the plan fields
            plan_fields = (dict(plan_window=plan.config.window, plan_codes=plan.config.codes_per_step,
                                plan_codebook=plan.config.codebook, plan_memory=args.plan_memory,
                                plan_dropout=args.plan_dropout) if plan is not None and args.stage == "prior" else {})
            config_class = PlanPriorConfig if args.stage == "plan-prior" else RVQPriorConfig
            config = config_class(plan.config if args.stage == "plan-prior" else tokenizer.config,
                                    temporal_layers=args.token_layers, depth_layers=args.depth_layers,
                                    text_model=args.text_model, text_model_revision=args.text_model_revision,
                                    input_corruption=args.input_corruption,
                                    scheduled_sampling=args.scheduled_sampling, history_dropout=args.history_dropout,
                                    history_stages=args.history_stages, part_valid_loss=args.part_valid_loss,
                                    guided_attention=args.guided_attention,
                                    guided_attention_width=args.guided_attention_width,
                                    alignment_supervision=args.alignment_supervision,
                                    rollout_blocks=args.rollout_blocks, rollout_min=args.rollout_min,
                                    rollout_max=args.rollout_max, rollout_onset=args.rollout_onset,
                                    rollout_guidance=args.rollout_guidance,
                                    rollout_temperature=args.rollout_temperature, **plan_fields, **common)
            model = RVQPrior(config).to(device)
    kind = tokenizer_kind if args.stage == "tokenizer" else prior_kind
    plan_detail = plan is not None and args.stage == "prior"

    def loss_of(motion, tokens, mask, loci, train: bool, spans=None):
        if args.stage == "tokenizer":
            return tokenizer_loss(model, motion) if args.architecture == "vq" else rvq_loss(model, motion)
        if args.architecture == "vq":
            return prior_loss(model, tokenizer, tokens, mask, motion, *loci)
        if args.stage == "plan-prior":
            return plan_prior_loss(model, tokenizer, plan, tokens, mask, motion, *loci, train=train, word_spans=spans)
        return rvq_prior_loss(model, tokenizer, tokens, mask, motion, *loci, train=train, word_spans=spans,
                              **({"plan_tokenizer": plan} if plan_detail else {}))

    def split_spans(batch):
        """(batch without the per-frame word spans, the spans on the device or None)."""
        if not word_spans:
            return batch, None
        return batch[:-1], batch[-1].to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    prefix = f"training-runs/{args.run}"
    step = 0
    checkpoints = [k for k in store.list(f"{prefix}/checkpoints/") if k.endswith(".pt")]
    if checkpoints:
        state = torch.load(io.BytesIO(store.get(checkpoints[-1])), map_location=device, weights_only=False)
        if (state.get("kind") != kind or state["config_sha256"] != config.sha256()
                or state["manifest_sha256"] != args.manifest_sha256
                or state.get("plan_tokenizer_sha256") != args.plan_tokenizer_sha256):
            raise SystemExit("the newest checkpoint was trained with a different config or manifest")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        resumed_ema = state.get("ema")
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {checkpoints[-1]} at step {step}", flush=True)
    else:
        if args.init_key:
            if not args.init_sha256 or (args.stage == "tokenizer" and args.architecture != "rvq"):
                raise SystemExit("--init-key needs --init-sha256 (and, for a tokenizer, --architecture rvq)")
            payload = store.get(args.init_key)
            if hashlib.sha256(payload).hexdigest() != args.init_sha256:
                raise SystemExit("the --init-key checkpoint differs from --init-sha256")
            init = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
            same = (prior_warm_start_compatible(init["config"], config.to_dict()) if prior_stage
                    else warm_start_compatible(init["config"], config.to_dict()))
            if init.get("kind") != kind or not same:
                raise SystemExit("--init-key is a different kind or config")
            if plan_detail and not init["config"].get("plan_window"):
                # round 17: a plan-less prior (W14b); the plan modules keep their zero initialisation
                added = plan_warm_start(model, init.get("ema") or init["model"])
                print(f"plan modules zero-initialised ({added} tensors)", flush=True)
            else:
                model.load_state_dict(init.get("ema") or init["model"])
            print(f"initialised from {args.init_key} (step {init.get('step')})", flush=True)
        if args.init_text_key:                         # round 17: the plan prior's text side from W14b
            payload = store.get(args.init_text_key)
            if hashlib.sha256(payload).hexdigest() != args.init_text_sha256:
                raise SystemExit("the --init-text-key checkpoint differs from --init-text-sha256")
            source = torch.load(io.BytesIO(payload), map_location=device, weights_only=False)
            if source.get("kind") != "motion_rvq_prior":
                raise SystemExit("--init-text-key is not an rvq prior checkpoint")
            try:
                copied = copy_text_encoder(model, source.get("ema") or source["model"], source["config"])
            except ValueError as error:
                raise SystemExit(f"--init-text-key: {error}")
            print(f"text encoder from {args.init_text_key} ({len(copied)} tensors)", flush=True)
        store.put(f"{prefix}/run.json", json.dumps(dict(kind=kind, config=config.to_dict(), config_sha256=config.sha256(),
                                                        manifest_key=args.manifest_key,
                                                        manifest_sha256=args.manifest_sha256, args=vars(args),
                                                        train_records=len(train)), indent=2, sort_keys=True).encode())
    sampler = None
    if args.corpus_weight:
        from torch.utils.data import WeightedRandomSampler
        weights = {k: float(v) for k, v in (w.split("=", 1) for w in args.corpus_weight)}
        sampler = WeightedRandomSampler(corpus_sample_weights(train.records, weights), num_samples=len(train),
                                        replacement=True, generator=torch.Generator().manual_seed(args.seed + step))
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        num_workers=args.workers,
                        collate_fn=partial(collate_text_motion, loci=args.locus_conditioning and prior_stage,
                                           word_spans=word_spans),
                        generator=torch.Generator().manual_seed(args.seed + step),
                        persistent_workers=args.workers > 0, drop_last=True)
    log, started = [], time.monotonic()
    ema = WeightEMA(model, args.ema_decay)
    if checkpoints and resumed_ema is not None:
        ema.shadow = {n: resumed_ema[n].to(device).clone() for n in ema.shadow}
        ema.updates = step
    val_batches = []
    if len(val):
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                collate_fn=partial(collate_text_motion, loci=args.locus_conditioning and prior_stage,
                                           word_spans=word_spans))
        for index, batch in enumerate(val_loader):
            if index >= args.val_batches:
                break
            val_batches.append(batch)
    best = json.loads(store.get(f"{prefix}/best.json")) if store.exists(f"{prefix}/best.json") else None

    def validate() -> dict:
        """Mean validation terms of the EMA weights on the fixed batches (fixed noise seed)."""
        live = ema.swap(model)
        model.eval()
        totals, count = {}, 0
        with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(1234)
            for batch in val_batches:
                batch, _ = split_spans(batch)
                tokens_v, mask_v, motion_v, *loci_v = to_device(batch, device)
                loss_v, terms_v = loss_of(motion_v, tokens_v, mask_v, loci_v, train=False)
                if plan_detail:                # round 17 informativeness: the same codes without the plan
                    _, empty_v = rvq_prior_loss(model, tokenizer, tokens_v, mask_v, motion_v, *loci_v, train=False,
                                                plan_tokenizer=plan, plan_mode="empty")
                    terms_v = dict(terms_v, tokens_no_plan=empty_v["tokens"],
                                   plan_gain=empty_v["tokens"] - terms_v["tokens"])
                for key, value in dict(loss=float(loss_v), **terms_v).items():
                    totals[key] = totals.get(key, 0.) + float(value)
                count += 1
        ema.restore(model, live)
        model.train()
        return {f"val/{k}": v / max(count, 1) for k, v in totals.items()}

    pending = []                               # log entries whose values are still device tensors

    def flush():
        """Read the pending entries' values off the device (one wait for the GPU), in place; refuse a non-finite loss.

        Reading a value back synchronises with the GPU, so the training step leaves its
        metrics on the device and they are read here only when they are printed,
        validated or checkpointed. A non-finite loss is therefore reported up to 50 steps
        late (or when the poisoned weights make a later step fail), always before a save.
        """
        cells = [(entry, key) for entry in pending for key, value in entry.items() if isinstance(value, torch.Tensor)]
        if cells:
            values = torch.stack([entry[key].detach().to(torch.float64).reshape(()) for entry, key in cells]).tolist()
            for (entry, key), value in zip(cells, values):
                entry[key] = value
        for entry in pending:
            if not math.isfinite(entry["loss"]):
                raise FloatingPointError(f"non-finite loss at step {entry['step'] - 1}")
        pending.clear()

    def checkpoint(metrics: dict | None = None):
        nonlocal best
        flush()
        buffer = io.BytesIO()
        torch.save(dict(kind=kind, model=model.state_dict(), ema=ema.state_dict(model), optimizer=optimizer.state_dict(),
                        step=step, validation=metrics,
                        torch_rng=torch.get_rng_state(), config=config.to_dict(), config_sha256=config.sha256(),
                        manifest_sha256=args.manifest_sha256, tokenizer_state=tokenizer_state,
                        tokenizer_key=args.tokenizer_key, tokenizer_sha256=args.tokenizer_sha256,
                        **(dict(plan_tokenizer_key=args.plan_tokenizer_key, plan_tokenizer_sha256=args.plan_tokenizer_sha256,
                                plan_tokenizer_config=plan.config.to_dict(), plan_tokenizer_state=plan_state)
                           if plan is not None else {})), buffer)
        key = f"{prefix}/checkpoints/step-{step:08d}.pt"
        items = [(key, buffer.getvalue()), (f"{prefix}/log/step-{step:08d}.json", json.dumps(log, sort_keys=True).encode())]
        if metrics and (best is None or metrics["val/loss"] < best["val_loss"]):
            best = dict(checkpoint_key=key, step=step, val_loss=metrics["val/loss"], validation=metrics)
            items.append((f"{prefix}/best.json", json.dumps(best, indent=2, sort_keys=True).encode()))
        uploader.submit(items)                 # in the background, in order: the checkpoint before best.json

    from signtranslator.training.async_upload import Uploader
    uploader = Uploader(store)
    model.train()
    stale = 0                                  # validations since the last best

    def note(metrics):
        """Early stopping: count validations that do not improve on the best (``best`` set by checkpoint())."""
        nonlocal stale
        best_loss = best["val_loss"] if best else float("inf")
        stale = 0 if metrics["val/loss"] < best_loss else stale + 1
        return args.patience and stale >= args.patience

    accepted, skipped, consecutive = collections.deque(maxlen=200), 0, 0      # --skip-grad-spikes state
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch, spans = split_spans(batch)
            tokens, mask, motion, *loci = to_device(batch, device)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps)
            if getattr(getattr(model, "config", None), "codebook_follows_lr", False):
                # full rate through warmup (the codebooks must form), then the cosine decay's share of the peak
                model.codebook_rate = 1. if step < args.warmup else optimizer.param_groups[0]["lr"] / args.lr
                model.codebook_reseed = model.codebook_rate >= .25
            try:
                loss, terms = loss_of(motion, tokens, mask, loci, train=True, spans=spans)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                skip = False
                if args.skip_grad_spikes > 0:          # one host sync per step, only when enabled
                    value = float(norm)
                    if not math.isfinite(value) or (len(accepted) >= 50 and step >= 500
                                                     and value > args.skip_grad_spikes * statistics.median(accepted)):
                        skip = True
                        skipped += 1; consecutive += 1
                        if consecutive > 100:
                            raise SystemExit(f"over 100 consecutive gradient spikes at step {step}: the run is diverging")
                    else:
                        accepted.append(value); consecutive = 0
                if not skip:
                    optimizer.step()
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                flush()                        # an unread non-finite loss before this failure is its cause
                raise
            step += 1
            entry = dict(step=step, loss=loss.detach(), grad_norm=norm.detach(), **(dict(skipped_steps=skipped) if args.skip_grad_spikes > 0 else {}),
                         seconds=round(time.monotonic() - started, 1), **{f"train/{k}": v for k, v in terms.items()})
            log.append(entry)
            pending.append(entry)
            due_validation = val_batches and (step % args.val_every == 0 or step == args.steps)
            if due_validation or step % 50 == 0 or step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
                flush()
            metrics, stop = None, False
            if due_validation:
                metrics = validate()
                entry.update(metrics)
                stop = note(metrics)
                print(json.dumps({k: v for k, v in entry.items() if k == "step" or k.startswith("val/")}), flush=True)
                if metrics is not None and (best is None or metrics["val/loss"] < best["val_loss"]) \
                        and step % args.checkpoint_every != 0:
                    checkpoint(metrics)                  # a new best between regular checkpoints is kept too
            if step % 50 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps or stop:
                checkpoint(metrics if metrics is not None else (validate() if val_batches else None))
            if stop:
                print(json.dumps(dict(step=step, early_stop=True, best_step=best["step"], best_val_loss=best["val_loss"])),
                      flush=True)
                uploader.drain()
                return 0
    flush()
    uploader.drain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
