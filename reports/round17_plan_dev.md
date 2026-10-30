# Round 17 plan codes: development report (2026-10-30, 05:12 UTC; review fixes 05:48 UTC; second-review fixes 06:40 UTC)

Branch `round17-plan` (from main `8374bd8`, the round 17 design). This implements the design in
docs/NEXT_PHASE_PLAN.md, "Round 17 design: coarse plan codes over the 24 Hz tokenizer". The first commit (c37036b) was
reviewed independently (APPROVE WITH FIXES); the fixes are in the section "Review fixes" and throughout. The second
review (of a8f6da9) found the remaining defects in gate measurement; they are fixed in "Second-review fixes", and the
gates section restates exactly the declared gates (docs/NEXT_PHASE_PLAN.md on main, 7f06613 and amendment 1 16d2e00).

**What ran:**
- No pods ran, and nothing was written to R2.
- Training was limited to local CPU smokes: synthetic states, the tiny test store, and five plan-tokenizer steps (plus one corrupted detail-prior step) on six real How2Sign clips.
- R2 was only read: run configs, the d2 checkpoint, the W14b step-30000 checkpoint and the round14b manifest.

**Owner decision (04:39 UTC):** no training until the UWB data result. The commands below are prepared, not launched.

## What was built

| Piece | File |
|---|---|
| Plan tokenizer `PlanVQ` / `PlanVQConfig`: per-part window encoders over d2's continuous latents, masked mean or learned pooling, residual EMA VQ, shared upsampling decoder; loss, checkpoints, `encode_plan`, `reconstruct`, `oracle_plans`, `decode_plan_states` (plan-only route) | `signtranslator/generation/plan_codes.py` |
| Plan conditioning of the RVQ prior: opt-in config fields (incl. the plan tokenizer's config SHA and plan corruption), `PlanReader`, `empty_plan`, `teacher_forced_plan_samples`, the plan threaded through the loss, rollout, likelihood and generation; plan-less behaviour unchanged | `signtranslator/generation/motion_rvq.py` |
| Plan prior `PlanPriorConfig` (an `RVQPrior` over plan steps), `plan_prior_loss`, text warm start `copy_text_encoder`, `generate_plan_codes_batch`, two-stage `generate_plan_detail_batch` | `signtranslator/generation/plan_codes.py` |
| Plan tokenizer training CLI (store/manifest/resume/EMA/best conventions; frozen d2 pinned by key and SHA; fidelity diagnostic; mean per-batch stage-1 perplexity on validation; dead-code re-seeds persisted in checkpoints and their rate per 1k steps) | `scripts/train_plan_tokenizer.py` |
| `--stage plan-prior`; `--plan-tokenizer-key/--plan-tokenizer-sha256`, `--plan-dropout`, `--plan-corruption*`, `--no-plan-memory`, `--init-text-key`; plan-less warm start (`plan_warm_start`); validation `tokens_no_plan`, `plan_gain`, `plan_gain_stage1` | `scripts/train_motion_tokens.py` |
| Two-stage evaluation: `load_generator(..., plan=...)`, `load_plan_prior` (every checkpoint SHA-pinned and cross-checked), `batch_with_plans` | `scripts/evaluate_text_to_motion.py` |
| Pilot flags `--plan-prior-key/--plan-prior-sha256/--plan-tokenizer-key/--plan-tokenizer-sha256`, `--plan-temperature`, `--plan-guidance`, `--oracle-plan` (adds `meaning_oracle_plan` and the paired `oracle_gap`), `--oracle-only` (the oracle read alone, before the arm's pilot). The default is unchanged | `scripts/pilot_content_check.py` |
| Oracle precondition: `--oracle-precondition` (a report's `meaning_oracle_plan` against a control's generated BLEU, paired on identical references) | `scripts/compare_pilot_lexicon.py` |
| Fidelity hooks `--plan-tokenizer-key/--plan-tokenizer-sha256`: the round trip decodes the plan-reconstructed latents; d2 pinned by the SHA of `--tokenizer-key`'s bytes; the fidelity report adds d2's own round trip of the same clips (ratios, BLEU, paired plan − d2 BLEU) | `scripts/tokenizer_fidelity_check.py`, `scripts/tokenizer_recognisability_check.py` |
| Gate measurements: `plan-route` (with its blank check), `text-gain` (nats per second), `plan-following` (blank-controlled, pairs resampled) | `scripts/round17_diagnostics.py` |
| Tests | `tests/test_generation_plan_codes.py` (30), `tests/test_train_plan_tokenizer_script.py` (1, end to end) |

## Review fixes (review of c37036b)

| # | Finding | Fix |
|---|---|---|
| 1 MAJOR | Padding leaked between residual blocks at `blocks = 2`: only a stack's input was masked, and a block's outputs at invalid positions are not zero | `PlanVQ._stack` re-masks before every block (encoder step/plan stacks, decoder plan/step stacks). Test parametrised over blocks {1, 2} × pooling {mean, learned}, in float64: the continuous plan latents, codes and reconstruction of a clip alone and in a padded batch agree to ≤ 1e-7, and padding filled with 1e3 changes nothing. The test also confirms the reviewed masking leaks (here up to 5.6e5 in float64) |
| 4 MAJOR | The detail prior trains on true plans only but generates from sampled ones | Declared option `plan_corruption` (rows), `plan_corruption_max` (r_max, default 0.5), `plan_corruption_source` (`prior`: P17's teacher-forced samples, one gradient-free P17 pass per batch; `random`: uniform codes). For a fraction of rows, r ~ U(0, r_max) of the valid plan steps are replaced. Zero default: earlier configs unchanged. Training-only (`PRIOR_TRAINING_ONLY`); P17 pinned by key and SHA in the checkpoint, checked on resume; tests. The pilot's `--oracle-plan` now reports the paired oracle − generated gap |
| 5 | No plan-only or text-dependence evaluation | `round17_diagnostics.py plan-route` (P17 → plan decoder → d2 decoder → back-translator, 1,000 pilot clips, with plan round trip, real, blank and paired differences) and `text-gain` (shuffled − true text, teacher-forced, nats/code, all and stage-1 codes, bootstrap CI over batches; for P17 and W14b alike) |
| 3 | No test that the detail follows the plan | `round17_diagnostics.py plan-following` (oracle plan of A, (i) no text, (ii) B's text; back-translation retrieval of A over B, rate with CI, real ceiling) and `plan_gain` / `plan_gain_stage1` (nats/code) at every validation |
| 2 | No content measure for the plan round trip | The fidelity report (with a plan) holds d2's own round trip of the same clips: `r2b.tokenizer_round_trip`, `tokenizer_ratios`, `plan_minus_tokenizer_bleu` (paired, CI). Report-only |
| 7 MINOR | Latent loss counted untracked parts (neutral fill) | Relative latent error over each part's observed steps only (`part_step_observed`, `part_code_weights`' rule per step) |
| 8 | No codebook health diagnostics | Stage-1 perplexity per part and dead-code re-seeds per part (`ResidualQuantizer.reseeded`) in every training and validation log; persisted in checkpoints since the second review |
| 9 | d2 not pinned in the hooks | Both instruments pass the SHA of `--tokenizer-key`'s bytes to `load_plan_tokenizer`, which refuses a plan built on other d2 bytes |
| 13 | Private torch API | `detect_is_causal_mask`: torch's `_detect_is_causal_mask` if importable, else the same rule (equality with the square-subsequent mask in the mask's dtype; for a bool mask that counts the bool causal mask as causal, as torch does). Tested against torch for bool, float and non-causal masks |
| 15 | `upsample_windows` unused; `check_plan_pair` ignored the plan tokenizer | The plan decoder upsamples with `upsample_windows`. The detail config records `plan_tokenizer` (the plan tokenizer's config SHA-256, required); `check_plan_pair` refuses a plan prior over another plan tokenizer. The checkpoints' plan-tokenizer *weights* SHAs are still cross-checked by `load_plan_prior` |
| 6 | d2's own padding leak | Noted below; d2 is not changed |

## Second-review fixes (review of a8f6da9)

| # | Finding | Fix |
|---|---|---|
| 1 | Plan following had no duration control and resampled directions as independent rows | `plan-following` adds **blank at A's duration** on the same directions; reports `a_plan_minus_blank` (paired, CI) and `gate.threshold` = blank + 0.5 × (real − blank); every interval resamples unordered pairs (a pair's two directions together). `--pairing duration` (neighbours within 10% in duration) is a second read. The "chance 0.5" wording is gone: duration alone reads about 0.66 |
| 2 | Text gain per code | `text-gain` reports **nats per second** (each clip's summed part-weighted gain over its duration, mean over clips with a CI over clips), pooled nats per second, and nats per code. Per-clip sums add up to the prior's own loss (tested) |
| 3 | No oracle precondition instrument | `compare_pilot_lexicon.py --oracle-precondition`: `meaning_oracle_plan` minus the control's generated BLEU, paired on identical references. The pilot's `--oracle-only` runs the oracle read alone, so the precondition is judged before the arm's pilot |
| 4 | Gate section held stale proposals | Rewritten below: exactly the declared gates (7f06613 + amendment 1) |
| 5 | Re-seed counter not persisted; perplexity ill-named | `reseeded` saved in every plan tokenizer checkpoint and restored on resume (cumulative, tested); logged as `stability/reseeded_<part>` and `stability/reseed_fraction_per_1k_<part>`. The term is `stage1_perplexity_<part>`; its validation value is the mean per-batch stage-1 perplexity on validation |
| 6 | Plan route's blank unchecked | `plan-route` reports `plan_route_minus_blank` (paired) and `blank_check` (its blank against 1.859, exit 4 when not reproduced); its seed series is documented. Its blank read was run locally on the 1,000 pilot clips (see "Checked against the real models") |
| 7 | Corruption sampling undocumented | The plan-corruption samples are drawn at guidance 1 and temperature 1; the pilot samples P17 at guidance 1.5 and part temperature 1.1. Declared, not changed; noted in the config and below |

**d2's own padding leak (finding 6, not changed).** d2 has no masking: its encoder's convolutions (4 full-rate residual blocks) see the neutral-filled padded frames of a batch, so the latents of the last few valid frames of a short clip differ between batched and alone. This holds for every prior since round 3 (their training targets) and for the plan tokenizer's inputs. The plan tokenizer itself is now padding-independent given its inputs. Training batches and the diagnostics' one-clip encodings (`oracle_plans`, round trips) therefore see slightly different d2 latents at clip ends.

## Checked against the real models (R2 read-only; re-run after the fixes)

**d2** (`tokens-rvq-d0b-r3d-d2` step 5000, sha `021ee00a…08de`):
- `down_layers` 0, so step = 1 frame: one token per frame at 24 Hz.
- 4 parts (body, left hand, right hand, face), 8 stages, so J = 32 codes per step.
- Codebook 512, `code_dim` 128.

**Plan tokenizer at the proposed size:**
- Settings: W 4, 2 stages, codebook 512, hidden 256, `code_dim` 64, 2 blocks.
- 6.04M parameters.
- 8 plan codes per plan step, 4 frames per plan step (6 Hz).
- On 6 real How2Sign val clips of 13 to 609 frames, the plan steps per clip were [153, 47, 4, 11, 63, 9], exactly ceil(frames / 4).
- The oracle plans needed no length adjustment.

**W14b** (step 30000, sha `0a689471…e563`, verified):
- The plan-conditioned W14b adds 41 tensors: 43.30M parameters against 38.02M.
- After `plan_warm_start`, the temporal states equal W14b's **bit for bit** on random inputs, and the evaluation loss equals W14b's **bit for bit** on the real clips (re-checked with the plan-corruption fields set).
- `prior_warm_start_compatible(W14b, W14b + plan fields)` is true. With any other field changed (e.g. `history_stages`) it is false.
- One training step of the warm-started detail prior with plan corruption from a (random) plan prior ran on the real clips (37.6% of plan steps corrupted at rows 1.0, r_max 1.0).

**Plan prior:**
- Its text encoder takes 59 tensors from W14b.
- Its untrained teacher-forced loss on real plans is 6.38 (ln 512 = 6.24).

**Two-stage generation** with the real d2, plan prior and detail prior: 1.0 s and 0.75 s gave 24 and 18 frames, plans of 6 and 5 steps, and detail codes of (24, 32) and (18, 32).

**The plan route's blank reproduces 1.859** (second-review fix 6). `round17_diagnostics.py`'s own clip loading (`pilot_clips`: the pilot's fixed sample from round14b-mixed-v1), blanking and back-translator, run locally on CPU (R2 read-only) on the 1,000 clips: BLEU **1.859 [1.537, 2.175]**, `blank_check.reproduced` true, with references identical (same order) to `round16/pilot-r16c.json`, whose blank is 1.859.

**Earlier configs are unchanged** (re-run after the fixes). The new code was compared with the original `motion_rvq.py` (8374bd8) on every run config stored in R2:
- **53 rvq-prior and 47 rvq-tokenizer configs:** identical `to_dict()` and SHA-256.
- Plan-less priors, tested on a plain config, a W14b-like config (history stages 1, history dropout, part-valid loss, loci) and a rollout config:
  - identical initialisation;
  - bit-identical sampled batch generation;
  - bit-identical training loss, including the rollout path. The new `stage1_tokens` term is computed without gradient and only adds a logged value.
- Eight stored prior configs do not reproduce their recorded SHA under the original code either. That predates this branch, and this branch does not change it.

## Choices and why

### Plan tokenizer

**Input and target.** Input: d2's continuous per-part latents before quantisation, as designed. Target: the same latents.
- Every convolution and every residual block sees zeros at invalid steps and invalid plan steps (re-masked per block, fix 1). So a clip's plan does not depend on the batch's padding, given d2's latents (tested).
- d2's own encoder still sees its padded frames, as in every earlier prior (finding 6 above).

**Windows.** Plan step i covers steps [i·W, (i+1)·W). The last window of a ragged clip holds its remaining valid steps. A plan step is valid iff its window holds a valid step.

**Pooling.** The default is the masked mean, the simplest unbiased summary.
- `learned` is a masked softmax over the window's valid steps, available as an option.
- A strided convolution was not used: it would mix invalid steps into ragged windows.

**Quantiser.** The existing `ResidualQuantizer`, 2 stages × 512 per part: EMA codebooks with dead-code re-seeding, straight-through gradients, commitment 0.25.
- The bit rate is about 1/16 of d2's: 4 × 2 × 9 bits × 6 Hz against 4 × 8 × 9 bits × 24 Hz.
- There is no quantiser dropout: the plan prior models both stages.
- Logged: stage-1 codes used, stage-1 perplexity and dead-code re-seeds per part.

**Decoder.** One shared decoder for all parts, as d2's.
- Convolutions run at plan rate, then nearest upsampling ×W (`upsample_windows`) plus a learned within-window position, then convolutions at step rate.
- Its output is d2-ready per-step latents.

**Loss.** The relative latent error plus d2's own motion loss on the motion decoded from the reconstructed latents.
- Relative latent error: Σ‖ẑ − z‖² / Σ‖z − z̄‖² per part over the steps where the part was observed (fix 7). 1 means the part mean, 0 means exact.
- Motion loss: the per-channel and skeleton position, velocity and acceleration terms of `rvq_loss`, through the frozen decoder.
- Why the motion term: latent regression alone pulls each window towards its mean, which damps motion. The binding fidelity gate is hand speed through d2's decoder, so the loss trains on what the gate measures.

### Detail prior

**Two conditioning paths:**
1. The summed plan-code embeddings, projected by a zero-initialised linear layer, are added to the temporal input at every step of their window (step t reads plan step t // W).
2. A **separate** pre-norm cross-attention reader after every temporal layer. Its zero-initialised output projection reads the whole plan memory (LayerNorm of the plan embeddings plus plan-step position). Its query carries the fractional plan position t / W in the same sinusoid basis.

**Why a separate reader rather than concatenation with a type embedding.** Plan keys appended to the text memory would take softmax mass from the text at step 0. Exact equality with W14b would then be impossible, and the text cross-attention that rounds 13's guided and alignment losses and the attention recording read would change meaning. A zero-initialised residual reader is exactly the identity at step 0. The text attention keeps its meaning.

**Full-plan visibility** (not causal over the plan) is intended. The plan is generated before any detail, so seeing future plan steps gives look-ahead, and it is not a leak.

**Guidance** extrapolates the text only. The conditional and the empty-text pass both read the same plan.

**Plan identity.** The config records `plan_tokenizer`, the plan tokenizer's config SHA-256 (required with `plan_window`). Every checkpoint stores the plan tokenizer's key, checkpoint SHA, config and weights.

**Plan corruption** (training only, fix 4): with probability `plan_corruption` per row, a rate r ~ U(0, `plan_corruption_max`) of the row's valid plan steps is replaced by P17's teacher-forced samples (p(c_t^j | true c_t^{<j}, true c_{<t}, text), temperature 1; one gradient-free P17 pass per batch) or by uniform codes. It draws after every other draw and before plan dropout; rows holding a rollout block are not corrupted; the logged `plan_corrupted` is the corrupted share of valid plan steps. Why teacher-forced samples: they are the plan prior's own errors at each step, at the cost of one forward pass; free-running samples would need ~S_plan passes per batch.

**Plan dropout** (training only, `plan_dropout`):
- It replaces a sequence's plan by the empty plan (the reserved index K_π at every code). The plan mask is kept.
- The draw comes after every other random draw of the loss. So with dropout 0 the warm-started training loss equals W14b's bit for bit, with plan dropout and plan corruption on (tested).
- Rows holding a rollout block keep the plan their block was drawn under.
- `plan_dropout` and the corruption fields are in `PRIOR_TRAINING_ONLY`.

**Warm start.** `prior_warm_start_compatible` accepts a plan-less init for a plan-conditioned config when the configs differ only in `PLAN_FIELDS`. `plan_warm_start` loads with `strict=False` and refuses unless the missing keys are exactly the plan modules (every name starts with `plan_`) and no key is unexpected.

**Exactness scope:**
- **Bit-exact:** in evaluation (loss, gradients, sequence likelihood, sampled generation with a fixed seed), and in training whenever the network has no dropout.
- **With W14b's dropout 0.1:** the readers' dropout and the plan-dropout and corruption draws shift the random stream. A training step's dropout masks then differ from W14b's under the same seed. The function for given masks is the same.
- The temporal layers are called exactly as `nn.TransformerDecoder` calls them, including its causal-mask detection (`detect_is_causal_mask`, fix 13).

### Plan prior

- `PlanPriorConfig` is an `RVQPriorConfig` whose `tokenizer` is the plan tokenizer's config. It exposes `codes_per_step` = 8, `codebook`, `stages` and `step` = 4 frames. So the temporal transformer, depth transformer, history options, duration head and generation are the RVQ prior's own, over plan steps.
- `plan_prior_loss` passes the plan codes as `targets` to `rvq_prior_loss`. Part-valid weights pool `part_code_weights`' per-step rule over each window.
- The text encoder is warm-started from W14b (`--init-text-key`). It refuses any text-side config difference or key/shape mismatch.

### Generation

- `generate_plan_detail_batch`: plan prior (seed + 104729), then the detail prior given the fixed plan (seed), then the d2 decoder.
- Plan lengths are ceil(ceil(frames / step) / W), the same rule on both sides.
- `durations=None` uses the detail prior's predicted duration.
- `plans=` fixes the plan (deterministic, tested). The pilot's `--oracle-plan` uses this with each reference clip's own plan.
- `decode_plan_states`: the plan alone through the plan decoder and d2's decoder (the plan-only route and the plan round trip).
- The plan-less sampling loop was moved into `sample_code_sequence` unchanged (bit-identical to the original, checked above).

## Tests

- **New:**
  - `tests/test_generation_plan_codes.py`, 30 tests: window/upsampling alignment with ragged lengths and learned pooling; **padding-independent plan latents, codes and reconstruction for blocks {1, 2} × pooling {mean, learned}** (float64, and the reviewed masking shown to leak); stage-major code layout; toy-signal reconstruction (relative error < 0.3 × initial) and the motion loss trains with d2 frozen; deterministic checkpoint bytes with SHA, kind and tokenizer pins.
  - The same file continued: opt-in config and SHA compatibility of earlier configs; the script's warm-start rule; **zero-init warm start equals the plan-less prior exactly** (4 variants: plain or W14b-like × with or without plan memory; loss in evaluation and training with plan dropout 0.5, gradients, likelihood, sampled generation); **a plan-dropout row equals the no-plan row**; the window/memory reach of the plan; the plan prior trains, round-trips its config and copies its text encoder; **two-stage generation with a fixed plan is deterministic**; mismatched-pair refusals (layout, tokenizer, plan tokenizer); the fidelity round-trip hook; d2-like (step 1) alignment; oracle plans.
  - Review additions: causal-mask detection equals torch's and its fallback's; **plan corruption** (opt-in; exact at zero init; evaluation never corrupts; only valid steps of selected rows change; the prior source uses the plan prior's teacher-forced samples, refused without it); `stage1_tokens`; the part-observed latent loss; perplexity and re-seed counts; plan-only decoding; the retrieval score.
  - `tests/test_train_plan_tokenizer_script.py` (dir: store), end to end:
    - d2-like tokenizer → plan tokenizer (resume, fidelity diagnostic, refusals) → plan-less prior;
    - plan prior with the text warm start → plan-conditioned detail prior warm-started from the plan-less one (zero-init message, `tokens_no_plan`/`plan_gain`/`plan_gain_stage1` logged, resume) → `load_generator` with the pinned pair (refusals for a missing plan, a plan-less detail and a wrong pin; deterministic batches; predicted duration; fixed-plan batches);
    - plan corruption from random codes and from the trained plan prior (refusals; P17 pinned in the checkpoint; resume under another P17 refused);
    - `round17_diagnostics.py`: `text-gain` for the plan-less prior, the plan prior and the detail prior; `plan-route` and `plan-following` (back-translation replaced by a stand-in).
  - Second-review additions: pairing, pair-resampled intervals, paired rate difference and blank check; per-clip nats equal the loss's tokens (plain, part-valid, detail, plan prior) and the per-second text gain; the oracle precondition; the pilot's oracle and oracle-only modes (stub fixture); the persisted re-seed counter across a resume; plan-route's blank check and plan-following's gate fields end to end.
- **Run:** `OMP_NUM_THREADS=1 nice ~/.venvs/sign-translator/bin/python -W error -m pytest -q`; results in the commit message.

## Pod commands (prepared, not launched)

Common preamble on a pod, as `runpod/fleet/round5_prior.sh`. The bundle is this branch's commit.
```bash
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece        # the back-translator (fidelity, diagnostics, pilot)
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
O=acquisition/2026-10-17-v1/youtube-asl-v1
MIX=$O/round14b-mixed-v1/manifest.json; MSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
TOK=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt; TSHA=021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de
TMIX=$O/round3c-mixed-v1/manifest.json; TMSHA=004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
WEIGHTS="--corpus-weight asllvd-text-v1=3 --corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight semlex-text-v2=2"
DATA="--fps 24 --conform-to-corpus how2sign-two-view-text-v1 --workers 10 --cache-bytes 107374182400"
DIAG="--mix-key $MIX --mix-sha256 $MSHA --clips 1000 --device cuda"
[ "$(sha $MIX)" = "$MSHA" ] && [ "$(sha $TOK)" = "$TSHA" ] && [ "$(sha $W14B)" = "$W14BSHA" ] || { echo "pin mismatch"; exit 1; }
python scripts/prefetch_cache.py $MIX || exit 1
```

**Smoke first.** Every training stage below first runs as `<run>-smoke` with `--steps 300 --val-every 100 --checkpoint-every 100`. The smoke measures seconds per step. A full run is cut, and the cut recorded, if its projected wall time exceeds 8 h.

**Order:** (a) plan tokenizer and its gates → (b) P17 and its diagnostics → (c) the two detail arms (D17, D17-PC, which needs P17) → (d) diagnostics and pilots.

### (a) Plan tokenizer `plan-vq-v1`

The corpus weights are W14b's, so the plan sees the distribution its priors train on.
```bash
python scripts/train_plan_tokenizer.py --run plan-vq-v1 --manifest-key $MIX --manifest-sha256 $MSHA \
  --tokenizer-key $TOK --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA $DATA $WEIGHTS \
  --window 4 --hidden 256 --code-dim 64 --codebook 512 --plan-stages 2 --blocks 2 --pooling mean \
  --latent-weight 1 --motion-weight 1 --steps 20000 --batch-size 64 --lr 3e-4 --warmup 500 \
  --val-every 1000 --val-batches 32 --checkpoint-every 2000 --fidelity-clips 64
P=training-runs/plan-vq-v1/checkpoints/step-00020000.pt; PSHA=$(sha $P)
```
**Produces (PT stability):** in `training-runs/plan-vq-v1/log/*.json`: per validation `val/stage1_perplexity_<part>` (the mean per-batch stage-1 perplexity on validation), `stability/reseeded_<part>` (cumulative, persisted in each checkpoint as `reseeded`) and `stability/reseed_fraction_per_1k_<part>`; also `val/loss`, `val/latent_relative`, `val/motion`, `val/codes_used_<part>`; per checkpoint `fidelity/plan/*` and `fidelity/tokenizer/*` (64 clips, all joints, report-only). A non-finite loss stops the run before any save (no divergence).

The gate instruments use d2's own acceptance settings, with the plan hook. Exit status 4 is a verdict, not an error. The plan gates read the reports' numbers, below.
```bash
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
python scripts/tokenizer_fidelity_check.py --manifest-key $TMIX --manifest-sha256 $TMSHA --tokenizer-key $TOK \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --translator-key "$KP2D" --translator-manifest-sha256 $KPSHA \
  --limit 1000 --device cuda --speed-joints-key $O/round3d/joint-reliability-how2sign-train.json \
  --output-key $O/round17/plan-fidelity-plan-vq-v1.json
python scripts/tokenizer_recognisability_check.py --manifest-key $TMIX --manifest-sha256 $TMSHA --corpus asllvd-text-v1 \
  --gloss-map-key acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json --tokenizer-key $TOK \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --limit 300 --device cuda \
  --output-key $O/round17/plan-recognisability-plan-vq-v1.json
```
**Produces (PT content):**
- `plan-recognisability-plan-vq-v1.json` → `real.top5`, `round_trip.top5`, `ratio` (gate: `ratio` ≥ 0.9).
- `plan-fidelity-plan-vq-v1.json` → `r2b.round_trip.bleu` (plan) and `r2b.tokenizer_round_trip.bleu` (d2, same clips) (gate: plan ≥ 0.8 × d2); `r2b.round_trip.ci95[0]` against `r2b.blank.ci95[1]` (gate: the plan's lower bound above blank); `plan_minus_tokenizer_bleu` (paired, reported).
- Report-only: `ratios.*_speed_ratio` (plan round trip, reliable joints) and `tokenizer_ratios`. The script's own `pass` is d2's acceptance rule, not a round-17 gate.

### (b) Plan prior `tokens-plan-prior-r17-p17`

Settings:
- W14b's sizes and text settings, which the text warm start requires.
- History dropout and part-valid loss as W14b.
- `--history-stages 0`: plan steps are 167 ms apart, so the copy shortcut that `history_stages 1` removes at 24 Hz is much weaker.
- The temporal and depth stacks are new, so the learning rate is 3e-4 with warmup 1000.
```bash
python scripts/train_motion_tokens.py --architecture rvq --stage plan-prior --run tokens-plan-prior-r17-p17 \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key $TOK --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --init-text-key $W14B --init-text-sha256 $W14BSHA $DATA $WEIGHTS \
  --steps 30000 --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --history-dropout 0.5 --part-valid-loss --val-every 1000 --val-batches 32 --checkpoint-every 2000 \
  --patience 0
PP=training-runs/tokens-plan-prior-r17-p17/checkpoints/step-00030000.pt; PPSHA=$(sha $PP)
```
**Produces:** `val/tokens`, `val/stage1_tokens`, `val/token_accuracy` per validation (the plan prior's teacher-forced fit; reported).

P17 gate measurements, with W14b's text gain as the reference:
```bash
for K in "$PP $PPSHA p17" "$W14B $W14BSHA w14b"; do set -- $K
  python scripts/round17_diagnostics.py text-gain --checkpoint-key $1 --checkpoint-sha256 $2 --manifest-key $MIX \
    --manifest-sha256 $MSHA --batches 32 --batch-size 64 --device cuda --output-key $O/round17/text-gain-$3.json; done
python scripts/round17_diagnostics.py plan-route --checkpoint-key $PP --checkpoint-sha256 $PPSHA \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA $DIAG --output-key $O/round17/plan-route-p17.json
```
**Produces:**
- **P17 (a):** `text-gain-p17.json` and `text-gain-w14b.json` → `text_gain_nats_per_second.mean` (gate: P17 ≥ 2 × W14b), with its CI over clips; `text_gain_nats_per_second_pooled`, `text_gain_nats_per_code` and the stage-1 values are reported.
- **P17 (b):** `plan-route-p17.json` → `plan_route_minus_blank.ci95[0]` (gate: > 0) and `blank_check.reproduced` (gate: the script's blank reproduces 1.859; exit status 4 otherwise); `gate.passes` combines both. Also `plan_route`, `plan_round_trip`, `real`, `blank` BLEU and `plan_route_minus_plan_round_trip`. Its plans are drawn with seeds 0 + i per 32-clip block, independent of the pilots' draw.

### (c) Detail priors: D17 (true plans) and D17-PC (plan corruption from P17)

Both are warm-started from W14b step 30000. Every W14b/R16C setting is unchanged: manifest, weights, history dropout 0.5, history stages 1, part-valid loss, lr 1e-4, warmup 200, 10,000 steps at batch 64. Added: the plan and plan dropout 0.2. D17-PC adds plan corruption on half the rows, r_max 0.5, from P17.
```bash
DETAIL="--architecture rvq --stage prior --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key $TOK \
  --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA \
  --plan-dropout 0.2 --init-key $W14B --init-sha256 $W14BSHA $DATA $WEIGHTS --steps 10000 --lr 1e-4 --warmup 200 \
  --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 --locus-conditioning \
  --history-dropout 0.5 --part-valid-loss --history-stages 1 --val-every 1000 --val-batches 32 --checkpoint-every 2000 \
  --patience 0"
python scripts/train_motion_tokens.py $DETAIL --run tokens-rvq-d0b-prior-r17-d17
python scripts/train_motion_tokens.py $DETAIL --run tokens-rvq-d0b-prior-r17-d17pc --plan-corruption 0.5 \
  --plan-corruption-max 0.5 --plan-corruption-source prior --plan-corruption-prior-key $PP --plan-corruption-prior-sha256 $PPSHA
D=training-runs/tokens-rvq-d0b-prior-r17-d17/checkpoints/step-00010000.pt; DSHA=$(sha $D)
DPC=training-runs/tokens-rvq-d0b-prior-r17-d17pc/checkpoints/step-00010000.pt; DPCSHA=$(sha $DPC)
```
- Each log must show `plan modules zero-initialised (41 tensors)` and `initialised from …w14b…`.
- Equivalent through the fleet script: `round5_prior.sh` with R16C's environment and `EXTRA_WEIGHTS` = R16C's plus the plan flags above. Its leading `--corpus-weight asllvd-text-v1=3` repeats a weight already given.

**Produces (D17 informativeness):** per validation `val/plan_gain` (nats/code: empty-plan minus true-plan teacher-forced loss, same codes, 32 fixed batches, EMA weights; gate: ≥ 0.05 by step 4000) and `val/plan_gain_stage1` (reported), with `val/tokens`, `val/tokens_no_plan`, `val/stage1_tokens_no_plan`; D17-PC also logs `train/plan_corrupted`. The corruption samples are P17's teacher-forced draws at guidance 1 and temperature 1 (the pilot samples P17 at guidance 1.5 and part temperature 1.1; declared).

### (d) Diagnostics and pilots

Plan following of each detail arm (the declared gate), plus its text gain (reported):
```bash
for K in "$D $DSHA d17" "$DPC $DPCSHA d17pc"; do set -- $K
  python scripts/round17_diagnostics.py plan-following --checkpoint-key $1 --checkpoint-sha256 $2 \
    --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA $DIAG --output-key $O/round17/plan-following-$3.json
  python scripts/round17_diagnostics.py plan-following --pairing duration --checkpoint-key $1 --checkpoint-sha256 $2 \
    --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA $DIAG --output-key $O/round17/plan-following-$3-duration.json
  python scripts/round17_diagnostics.py text-gain --checkpoint-key $1 --checkpoint-sha256 $2 --manifest-key $MIX \
    --manifest-sha256 $MSHA --batches 32 --batch-size 64 --device cuda --output-key $O/round17/text-gain-$3.json; done
```
**Produces (D17 plan following):** `plan-following-<arm>.json` (500 unordered pairs, 1,000 directions) → `a_plan_minus_blank.ci95[0]` (gate: > 0), `a_plan_no_text.rate` against `gate.threshold` = blank + 0.5 × (real − blank) (gate: ≥), with `blank.rate`, `real.rate` and `a_plan_b_text.rate` (reported); `gate.passes` combines both. The `-duration` report is the second read (duration-matched pairs). The reviewer measured blank 0.6605 [0.631, 0.688] and real 0.757 on the pilots, so the threshold is near 0.709.

Oracle precondition, before the arm's pilot (oracle read only, then the paired comparison with R16C):
```bash
PCMD="python scripts/pilot_content_check.py --mix-key $MIX --mix-sha256 $MSHA \
  --vocab-mix-key $O/round10-mixed-v1/manifest.json --vocab-mix-sha256 ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a \
  --word-split-key $O/round14/r5-word-split.json --word-split-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --word-list-key $O/round14/r5-word-split.json --word-list-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --recognizer-run sign-recognizer-v3d --clips 1000 \
  --plan-prior-key $PP --plan-prior-sha256 $PPSHA --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA"
for A in "$D r17" "$DPC r17pc"; do set -- $A
  $PCMD --checkpoint-key $1 --oracle-plan --oracle-only --output-key $O/round17/oracle-$2.json
  python scripts/compare_pilot_lexicon.py --oracle-precondition --arm-key $O/round17/oracle-$2.json \
    --control-key $O/round16/pilot-r16c.json --output-key $O/round17/oracle-precondition-$2.json; done
```
**Produces (D17 oracle precondition and oracle wrist speed):** `oracle-precondition-<arm>.json` → `oracle_precondition.ci95[0]` (gate: > 0, else that arm's pilot is skipped); `oracle-<arm>.json` → `oracle_wrist_speed_gate.wrist_speed_over_real` (gate: within [0.8, 1.5]; measured as rounds 15–16's wrist speed relative to real, the back-translator's normalised 2D view) with `oracle_motion_ratio_to_real` (every statistic, reported).

Pilots, only for arms whose preconditions passed. Same 1,000 clips, instruments and pins as R16 (`r16_pilots.sh`); each also reads its oracle plan again, for the oracle–generated gap:
```bash
PCMD="python scripts/pilot_content_check.py --mix-key $MIX --mix-sha256 $MSHA \
  --vocab-mix-key $O/round10-mixed-v1/manifest.json --vocab-mix-sha256 ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a \
  --word-split-key $O/round14/r5-word-split.json --word-split-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --word-list-key $O/round14/r5-word-split.json --word-list-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --recognizer-run sign-recognizer-v3d --clips 1000 --oracle-plan \
  --plan-prior-key $PP --plan-prior-sha256 $PPSHA --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA"
$PCMD --oracle-plan --checkpoint-key $D --output-key $O/round17/pilot-r17.json
$PCMD --oracle-plan --checkpoint-key $DPC --output-key $O/round17/pilot-r17pc.json
# on the Mac, paired:
for A in r17 r17pc; do
  python scripts/compare_pilot_lexicon.py --arm-key $O/round17/pilot-$A.json --control-key $O/round16/pilot-r16c.json --per-subset --output-key $O/round17/compare-$A-vs-r16c.json
  python scripts/compare_pilot_lexicon.py --arm-key $O/round17/pilot-$A.json --control-key $O/v3read/pilot-w14b-step30000-v3d.json --output-key $O/round17/compare-$A-vs-w14b.json; done
python scripts/compare_pilot_lexicon.py --arm-key $O/round17/pilot-r17pc.json --control-key $O/round17/pilot-r17.json --output-key $O/round17/compare-r17pc-vs-r17.json
```
**Produces:**
- **R17 rule:** `compare-<arm>-vs-r16c.json` → `bleu_paired.ci95[0]` (> 0) and lexicon `ci95[0]` (> −0.05); `compare-<arm>-vs-w14b.json` → `bleu_paired.ci95[0]` (> 0).
- Reported beside it: `pilot-<arm>.json` → `oracle_gap` (paired oracle − generated, CI); P17's plan-only BLEU from `plan-route-p17.json`; `compare-r17pc-vs-r17.json` (the two arms paired).

## Declared gates (restated exactly: docs/NEXT_PHASE_PLAN.md on main, 7f06613 "Round 17: declared gates" and 16d2e00 "Round 17 gates, amendment 1")

Each stage runs only if the previous stage passes. All measures are on How2Sign val unless stated.

**Plan tokenizer (PT):**
- **Content:** recogniser v2 ASLLVD top-5 on plan-round-trip motion ≥ 0.9 × real. Plan round-trip back-translation BLEU ≥ 0.8 × d2's own round trip on the same clips, with its lower bound above blank.
- **Stability:**
  - stage-1 perplexity ≥ 128 per part, where "perplexity" means the mean per-batch stage-1 perplexity on validation (amendment 1);
  - dead-code re-seeds < 1% of the codebook per 1k steps after step 10k, the rate coming from a counter persisted in checkpoints (amendment 1);
  - no divergence through 20k steps.
- **Report-only:** plan round-trip hand speed (decoded from continuous latents, a path generation never uses).

**Plan prior (P17):** all must hold.
- **(a) Text use:** teacher-forced text gain (shuffled-text loss − true-text loss) ≥ 2× W14b's same measure, measured in **nats per second** (the summed part-weighted gain per clip divided by its seconds), not per code (amendment 1).
- **(b) Plan-only route:** P17 samples → plan decode → d2 → back-translation on the 1,000 clips, gated on `plan_route_minus_blank` with a lower bound > 0; the script's own blank must reproduce 1.859. It samples with its own seeds, which differ from the arms' pilot draw (amendment 1; replaces "BLEU lower bound > 1.859").
- If (b) fails, the detail prior cannot rescue it and **the pilot is not run.**

**Detail prior (D17):** warm-started from W14b.
- **Arms:** plan dropout, plus a declared corruption arm (half the rows have a fraction r ~ U(0, 0.5) of plan steps replaced by P17 teacher-forced samples). The corruption samples are drawn at guidance 1 and temperature 1; the pilot samples P17 at guidance 1.5 and part temperature 1.1 (declared, not changed; amendment 1).
- **Informativeness:** `plan_gain` ≥ 0.05 nats/code by step 4000. The stage-1 gain is reported.
- **Plan following** (amendment 1, replacing "≥ 70%"): both must hold:
  - the paired difference, A-plan-with-no-text minus blank-at-A's-duration on the same pairs, has a lower bound > 0;
  - the A-plan rate ≥ blank + 0.5 × (real − blank).
  - The bootstrap resamples unordered pairs.
- **Oracle precondition** (amendment 1): `meaning_oracle_plan` minus R16C's generated BLEU, paired, lower bound > 0, computed by the comparison option added for this purpose (`compare_pilot_lexicon.py --oracle-precondition`). Otherwise the route cannot work and the pilot is skipped. The oracle-plan pilot is diagnostic only, never the declared metric.
- **Oracle-plan detail generations:** wrist speed within [0.8, 1.5] × real.

**Round-17 pilot (R17):** generated plan → D17, 1,000 clips. All must hold:
- paired BLEU lower bound > 0 against R16C;
- paired BLEU lower bound > 0 against W14b;
- v3 lexicon lower bound > −0.05 against R16C.

P17's plan-only BLEU and the oracle–generated gap are reported beside it.

**Sequencing:**
- PT may train once the review fixes land and are re-reviewed.
- P17 and D17 wait for the UWB gate result, per the owner's O4 decision.

### Where each gate is measured

| Gate | Command (section) | Report field |
|---|---|---|
| PT content: recognition | (a) `tokenizer_recognisability_check.py` | `ratio` ≥ 0.9 |
| PT content: BLEU | (a) `tokenizer_fidelity_check.py` | `r2b.round_trip.bleu` ≥ 0.8 × `r2b.tokenizer_round_trip.bleu`; `r2b.round_trip.ci95[0]` > `r2b.blank.ci95[1]` |
| PT stability | (a) `train_plan_tokenizer.py` log | `val/stage1_perplexity_<part>` ≥ 128; `stability/reseed_fraction_per_1k_<part>` < 0.01 at every validation after step 10000; the run reaches step 20000 without a non-finite loss |
| PT report-only | (a) `tokenizer_fidelity_check.py` | `ratios.*_speed_ratio` |
| P17 (a) | (b) `round17_diagnostics.py text-gain` | `text_gain_nats_per_second.mean` (P17) ≥ 2 × the same (W14b) |
| P17 (b) | (b) `round17_diagnostics.py plan-route` | `plan_route_minus_blank.ci95[0]` > 0 and `blank_check.reproduced` |
| D17 informativeness | (c) training log | `val/plan_gain` ≥ 0.05 at the step-4000 validation |
| D17 plan following | (d) `round17_diagnostics.py plan-following` | `a_plan_minus_blank.ci95[0]` > 0 and `a_plan_no_text.rate` ≥ `gate.threshold` |
| D17 oracle precondition | (d) `compare_pilot_lexicon.py --oracle-precondition` | `oracle_precondition.ci95[0]` > 0 |
| D17 oracle wrist speed | (d) `pilot_content_check.py --oracle-plan --oracle-only` | `oracle_wrist_speed_gate.wrist_speed_over_real` in [0.8, 1.5] |
| R17 | (d) `compare_pilot_lexicon.py` | `bleu_paired.ci95[0]` > 0 against R16C and W14b; lexicon `ci95[0]` > −0.05 against R16C |

Not declared, and so not gates: the hand-speed thresholds, "plan_gain > 0 at every validation", the "≥ 64 codes" stability bar and the tie-break between the two arms proposed in earlier versions of this report. The declaration does not say how a passing corruption arm and a passing plain arm are ranked; that needs the owner before the pilots.

## Risks and open points

- **Learning rate of the new modules.** The plan modules start at zero and train at W14b's continuation rate (1e-4, warmup 200). If `val/plan_gain` stays near 0 by step 2000, the most likely cause is this rate, not the plan. A per-group learning rate is not implemented, deliberately: it would be a declared change.
- **Teacher forcing hides the plan's value.** With the real history, the next step at 24 Hz is nearly determined, so the informativeness gain may be small even when the plan helps free running. Plan following measures it under free running.
- **Teacher-forced corruption is not free running.** D17-PC's corrupting plans are P17's one-step samples from the true plan history. They are closer to generated plans than true plans are, but they do not drift as a free-running plan can.
- **The plan prior's temporal stack is new.** At 6 Hz and 30,000 steps it is the least-tested piece. Its gates (a) and (b) are read before the detail priors' pilots.
- **Dropout and exactness.** Warm-start exactness is bit-level in evaluation and in RNG-free training. Under dropout 0.1 the random stream shifts (see "Exactness scope").
- **d2's padding leak** (finding 6) is unchanged and affects every prior since round 3.
- **Pre-existing.** Eight stored prior configs whose recorded SHA the original code does not reproduce. This branch leaves them unchanged.
