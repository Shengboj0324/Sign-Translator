# Round 17 plan codes: development report (2026-10-30, 05:12 UTC)

Branch `round17-plan` (from main `8374bd8`, the round 17 design). This implements the design in
docs/NEXT_PHASE_PLAN.md, "Round 17 design: coarse plan codes over the 24 Hz tokenizer".

**What ran:**
- No pods ran, and nothing was written to R2.
- Training was limited to local CPU smokes: synthetic states, the tiny test store, and five plan-tokenizer steps on six real How2Sign clips.
- R2 was only read: run configs, the d2 checkpoint, the W14b step-30000 checkpoint and the round14b manifest.

**Owner decision (04:39 UTC):** no training until the UWB data result. The commands below are prepared, not launched.

## What was built

| Piece | File |
|---|---|
| Plan tokenizer `PlanVQ` / `PlanVQConfig`: per-part window encoders over d2's continuous latents, masked mean or learned pooling, residual EMA VQ, shared upsampling decoder; loss, checkpoints, `encode_plan`, `reconstruct`, `oracle_plans` | `signtranslator/generation/plan_codes.py` |
| Plan conditioning of the RVQ prior: opt-in config fields, `PlanReader`, `empty_plan`, the plan threaded through the loss, rollout, likelihood and generation; plan-less behaviour unchanged | `signtranslator/generation/motion_rvq.py` |
| Plan prior `PlanPriorConfig` (an `RVQPrior` over plan steps), `plan_prior_loss`, text warm start `copy_text_encoder`, `generate_plan_codes_batch`, two-stage `generate_plan_detail_batch` | `signtranslator/generation/plan_codes.py` |
| Plan tokenizer training CLI (store/manifest/resume/EMA/best conventions; frozen d2 pinned by key and SHA; fidelity diagnostic) | `scripts/train_plan_tokenizer.py` |
| `--stage plan-prior`; `--plan-tokenizer-key/--plan-tokenizer-sha256`, `--plan-dropout`, `--no-plan-memory`, `--init-text-key`; plan-less warm start (`plan_warm_start`); validation `tokens_no_plan` / `plan_gain` | `scripts/train_motion_tokens.py` |
| Two-stage evaluation: `load_generator(..., plan=...)`, `load_plan_prior` (every checkpoint SHA-pinned and cross-checked) | `scripts/evaluate_text_to_motion.py` |
| Pilot flags `--plan-prior-key/--plan-prior-sha256/--plan-tokenizer-key/--plan-tokenizer-sha256`, `--plan-temperature`, `--plan-guidance`, `--oracle-plan`. The default is unchanged | `scripts/pilot_content_check.py` |
| Fidelity hooks: `--plan-tokenizer-key/--plan-tokenizer-sha256`. The round trip decodes the plan-reconstructed latents | `scripts/tokenizer_fidelity_check.py`, `scripts/tokenizer_recognisability_check.py` |
| Tests | `tests/test_generation_plan_codes.py` (20), `tests/test_train_plan_tokenizer_script.py` (1, end to end) |

## Checked against the real models (R2 read-only)

**d2** (`tokens-rvq-d0b-r3d-d2` step 5000, sha `021ee00a…08de`):
- `down_layers` 0, so step = 1 frame: one token per frame at 24 Hz.
- 4 parts (body, left hand, right hand, face), 8 stages, so J = 32 codes per step.
- Codebook 512, `code_dim` 128.

**Plan tokenizer at the proposed size:**
- Settings: W 4, 2 stages, codebook 512, hidden 256, `code_dim` 64.
- 6.04M parameters.
- 8 plan codes per plan step, 4 frames per plan step (6 Hz).
- On 6 real How2Sign val clips of 13 to 609 frames, the plan steps per clip were [153, 47, 4, 11, 63, 9], exactly ceil(frames / 4).
- The oracle plans needed no length adjustment.

**W14b** (step 30000, sha `0a689471…e563`, verified):
- The plan-conditioned W14b adds 41 tensors: 43.30M parameters against 38.02M.
- After `plan_warm_start`, the temporal states equal W14b's **bit for bit** on random inputs, and the evaluation loss equals W14b's **bit for bit** on the real clips.
- `prior_warm_start_compatible(W14b, W14b + plan fields)` is true. With any other field changed (e.g. `history_stages`) it is false.

**Plan prior:**
- Its text encoder takes 59 tensors from W14b.
- Its untrained teacher-forced loss on real plans is 6.33 (ln 512 = 6.24).

**Two-stage generation** with the real d2, plan prior and detail prior: 1.0 s and 0.75 s gave 24 and 18 frames, plans of 6 and 5 steps, and detail codes of (24, 32) and (18, 32).

**Earlier configs are unchanged.** The new code was compared with the original `motion_rvq.py` (8374bd8) on every run config stored in R2:
- **53 rvq-prior and 47 rvq-tokenizer configs:** identical `to_dict()` and SHA-256.
- Plan-less priors, tested on a plain config, a W14b-like config (history stages 1, history dropout, part-valid loss, loci) and a rollout config:
  - identical initialisation;
  - bit-identical sampled batch generation;
  - bit-identical training loss, including the rollout path.
- Eight stored prior configs do not reproduce their recorded SHA under the original code either. That predates this branch, and this branch does not change it.

## Choices and why

### Plan tokenizer

**Input and target.** Input: d2's continuous per-part latents before quantisation, as designed. Target: the same latents.
- Every convolution sees zeros at invalid steps and invalid plan steps. So a clip's codes do not depend on the batch's padding (tested).
- d2's own encoder still sees its padded frames, as in every earlier prior.

**Windows.** Plan step i covers steps [i·W, (i+1)·W). The last window of a ragged clip holds its remaining valid steps. A plan step is valid iff its window holds a valid step.

**Pooling.** The default is the masked mean, the simplest unbiased summary.
- `learned` is a masked softmax over the window's valid steps, available as an option.
- A strided convolution was not used: it would mix invalid steps into ragged windows.

**Quantiser.** The existing `ResidualQuantizer`, 2 stages × 512 per part: EMA codebooks with dead-code re-seeding, straight-through gradients, commitment 0.25.
- The bit rate is about 1/16 of d2's: 4 × 2 × 9 bits × 6 Hz against 4 × 8 × 9 bits × 24 Hz.
- There is no quantiser dropout: the plan prior models both stages.

**Decoder.** One shared decoder for all parts, as d2's.
- Convolutions run at plan rate, then nearest upsampling ×W plus a learned within-window position, then convolutions at step rate.
- Its output is d2-ready per-step latents.

**Loss.** The relative latent error plus d2's own motion loss on the motion decoded from the reconstructed latents.
- Relative latent error: Σ‖ẑ − z‖² / Σ‖z − z̄‖² per part. 1 means the part mean, 0 means exact.
- Motion loss: the per-channel and skeleton position, velocity and acceleration terms of `rvq_loss`, through the frozen decoder.
- Why the motion term: latent regression alone pulls each window towards its mean, which damps motion. The binding fidelity gate is hand speed through d2's decoder, so the loss trains on what the gate measures.

### Detail prior

**Two conditioning paths:**
1. The summed plan-code embeddings, projected by a zero-initialised linear layer, are added to the temporal input at every step of their window (step t reads plan step t // W).
2. A **separate** pre-norm cross-attention reader after every temporal layer. Its zero-initialised output projection reads the whole plan memory (LayerNorm of the plan embeddings plus plan-step position). Its query carries the fractional plan position t / W in the same sinusoid basis.

**Why a separate reader rather than concatenation with a type embedding.** Plan keys appended to the text memory would take softmax mass from the text at step 0. Exact equality with W14b would then be impossible, and the text cross-attention that rounds 13's guided and alignment losses and the attention recording read would change meaning. A zero-initialised residual reader is exactly the identity at step 0. The text attention keeps its meaning.

**Full-plan visibility** (not causal over the plan) is intended. The plan is generated before any detail, so seeing future plan steps gives look-ahead, and it is not a leak.

**Guidance** extrapolates the text only. The conditional and the empty-text pass both read the same plan.

**Plan dropout** (training only, `plan_dropout`):
- It replaces a sequence's plan by the empty plan (the reserved index K_π at every code). The plan mask is kept.
- The draw comes after every other random draw of the loss. So with dropout 0 the warm-started training loss equals W14b's bit for bit, with plan dropout on (tested).
- Rows holding a rollout block keep the plan their block was drawn under.
- `plan_dropout` is in `PRIOR_TRAINING_ONLY`.

**Warm start.** `prior_warm_start_compatible` accepts a plan-less init for a plan-conditioned config when the configs differ only in `PLAN_FIELDS`. `plan_warm_start` loads with `strict=False` and refuses unless the missing keys are exactly the plan modules (every name starts with `plan_`) and no key is unexpected.

**Exactness scope:**
- **Bit-exact:** in evaluation (loss, gradients, sequence likelihood, sampled generation with a fixed seed), and in training whenever the network has no dropout.
- **With W14b's dropout 0.1:** the readers' dropout and the plan-dropout draw shift the random stream. A training step's dropout masks then differ from W14b's under the same seed. The function for given masks is the same.

### Plan prior

- `PlanPriorConfig` is an `RVQPriorConfig` whose `tokenizer` is the plan tokenizer's config. It exposes `codes_per_step` = 8, `codebook`, `stages` and `step` = 4 frames. So the temporal transformer, depth transformer, history options, duration head and generation are the RVQ prior's own, over plan steps.
- `plan_prior_loss` passes the plan codes as `targets` to `rvq_prior_loss`. Part-valid weights pool `part_code_weights`' per-step rule over each window.
- The text encoder is warm-started from W14b (`--init-text-key`). It refuses any text-side config difference or key/shape mismatch.

### Generation

- `generate_plan_detail_batch`: plan prior (seed + 104729), then the detail prior given the fixed plan (seed), then the d2 decoder.
- Plan lengths are ceil(ceil(frames / step) / W), the same rule on both sides.
- `durations=None` uses the detail prior's predicted duration.
- `plans=` fixes the plan (deterministic, tested). The pilot's `--oracle-plan` uses this with each reference clip's own plan.
- The plan-less sampling loop was moved into `sample_code_sequence` unchanged (bit-identical to the original, checked above).

## Tests

- **New:**
  - `tests/test_generation_plan_codes.py`, 20 tests: window/upsampling alignment with ragged lengths and learned pooling; batch- and padding-independent codes; stage-major code layout; toy-signal reconstruction (relative error < 0.3 × initial) and the motion loss trains with d2 frozen; deterministic checkpoint bytes with SHA, kind and tokenizer pins.
  - The same file continued: opt-in config and SHA compatibility of earlier configs; the script's warm-start rule; **zero-init warm start equals the plan-less prior exactly** (4 variants: plain or W14b-like × with or without plan memory; loss in evaluation and training with plan dropout 0.5, gradients, likelihood, sampled generation); **a plan-dropout row equals the no-plan row**; the window/memory reach of the plan; the plan prior trains, round-trips its config and copies its text encoder; **two-stage generation with a fixed plan is deterministic**; mismatched-pair refusals; the fidelity round-trip hook; d2-like (step 1) alignment; oracle plans.
  - `tests/test_train_plan_tokenizer_script.py` (dir: store), end to end:
    - d2-like tokenizer → plan tokenizer (resume, fidelity diagnostic, refusals) → plan-less prior;
    - plan prior with the text warm start → plan-conditioned detail prior warm-started from the plan-less one (zero-init message, `tokens_no_plan`/`plan_gain` logged, resume) → `load_generator` with the pinned pair (refusals for a missing plan, a plan-less detail and a wrong pin; deterministic batches; predicted duration; fixed-plan batches).
- **Run:** `OMP_NUM_THREADS=1 nice ~/.venvs/sign-translator/bin/python -W error -m pytest -q`
  - every suite that imports a changed module: 21 files, 328 tests, all pass;
  - the full suite (about 3,990 tests, run before the oracle-plan addition, whose suites were re-run after it): exit 0, no failures or errors.

## Pod commands (prepared, not launched)

Common preamble on a pod, as `runpod/fleet/round5_prior.sh`. The bundle is this branch's commit.
```bash
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
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
[ "$(sha $MIX)" = "$MSHA" ] && [ "$(sha $TOK)" = "$TSHA" ] && [ "$(sha $W14B)" = "$W14BSHA" ] || { echo "pin mismatch"; exit 1; }
python scripts/prefetch_cache.py $MIX || exit 1
```

**Smoke first.** Every stage below first runs as `<run>-smoke` with `--steps 300 --val-every 100 --checkpoint-every 100`. The smoke measures seconds per step. A full run is cut, and the cut recorded, if its projected wall time exceeds 8 h.

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

### (c) Detail prior `tokens-rvq-d0b-prior-r17-d17`

Warm-started from W14b step 30000. Every W14b/R16C setting is unchanged: manifest, weights, history dropout 0.5, history stages 1, part-valid loss, lr 1e-4, warmup 200, 10,000 steps at batch 64. Added: the plan and plan dropout 0.2.
```bash
python scripts/train_motion_tokens.py --architecture rvq --stage prior --run tokens-rvq-d0b-prior-r17-d17 \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key $TOK --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --plan-dropout 0.2 --init-key $W14B --init-sha256 $W14BSHA \
  $DATA $WEIGHTS --steps 10000 --lr 1e-4 --warmup 200 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 \
  --depth-layers 2 --locus-conditioning --history-dropout 0.5 --part-valid-loss --history-stages 1 \
  --val-every 1000 --val-batches 32 --checkpoint-every 2000 --patience 0
D=training-runs/tokens-rvq-d0b-prior-r17-d17/checkpoints/step-00010000.pt
```
- The log must show `plan modules zero-initialised (41 tensors)` and `initialised from …w14b…`.
- Equivalent through the fleet script: `round5_prior.sh` with R16C's environment and `EXTRA_WEIGHTS` = R16C's plus `--plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --plan-dropout 0.2`. Its leading `--corpus-weight asllvd-text-v1=3` repeats a weight already given.

### (d) Pilot

Same 1,000 clips, instruments and pins as R16 (`r16_pilots.sh`).
```bash
PCMD="python scripts/pilot_content_check.py --mix-key $MIX --mix-sha256 $MSHA \
  --vocab-mix-key $O/round10-mixed-v1/manifest.json --vocab-mix-sha256 ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a \
  --word-split-key $O/round14/r5-word-split.json --word-split-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --word-list-key $O/round14/r5-word-split.json --word-list-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a \
  --recognizer-run sign-recognizer-v3d --clips 1000 --checkpoint-key $D \
  --plan-prior-key $PP --plan-prior-sha256 $PPSHA --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA"
$PCMD --output-key $O/round17/pilot-r17.json
$PCMD --oracle-plan --output-key $O/round17/pilot-r17-oracle-plan.json      # diagnostic only, never the arm's result
# on the Mac, paired:
python scripts/compare_pilot_lexicon.py --arm-key $O/round17/pilot-r17.json --control-key $O/round16/pilot-r16c.json --per-subset --output-key $O/round17/compare-r17-vs-r16c.json
python scripts/compare_pilot_lexicon.py --arm-key $O/round17/pilot-r17.json --control-key $O/v3read/pilot-w14b-step30000-v3d.json --output-key $O/round17/compare-r17-vs-w14b.json
```
The pilot needs `uv pip install -q 'transformers>=4.40,<5' sentencepiece` on the pod, as R16's.

## Proposed declared gates (for the owner, before anything runs)

### Plan tokenizer `plan-vq-v1`

All on its final step-20000 checkpoint.

| Gate | Requirement | Instrument |
|---|---|---|
| **F1 Fidelity: hand speed** | `left_hand_speed_ratio` ≥ 0.80 and `right_hand_speed_ratio` ≥ 0.80. Reliable joints, plan round trip against real, 1,000 How2Sign val clips. d2's own ratios on the same clips are reported beside them | `tokenizer_fidelity_check.py --plan-tokenizer-key` |
| **F2 Fidelity: recognition** | Plan round-trip top-5 ≥ 0.9 × real top-5. ASLLVD val, 300 clips, recogniser v2 (the instrument d2 was accepted with), gloss map as d2's run. The script's `ratio` and `pass` | `tokenizer_recognisability_check.py --plan-tokenizer-key` |
| **I1 Informativeness** | The detail prior D17's `val/plan_gain` (teacher-forced loss with the empty plan minus with the true plan, same codes, 32 fixed validation batches, EMA weights) > 0 at every validation from step 2000 to step 10000 | D17's training log |
| **S1 Stability** | 20,000 steps with no non-finite loss and every checkpoint written. Validation loss at step 20000 ≤ 1.05 × its value at step 10000. Stage-1 codes used per part on validation ≥ 64 of 512 | the run's log and checkpoints |

**Reported, not gating:**
- body and head speed and variation ratios;
- the plan round trip's R2b BLEU against real and blank;
- the training-time fidelity diagnostic (64 clips, all joints).

**Ordering:**
- If F1 or F2 fails, no prior is trained on this plan. The report names the failing gate and the next lever (`--pooling learned`, 3 stages, or W = 2).
- I1 is judged when D17 has trained. If it fails, R17 is not adopted, whatever the pilot reads.

### Round 17 pilot

**Arm:** R17 = P17 step 30000 → D17 step 10000 → d2.

**Matched control:** R16C. It is W14b + 10,000 steps with the same manifest, weights, history dropout and stages, part-valid loss, lr and warmup. The only differences from R17 are the plan conditioning, plan dropout and the plan prior. Its pilot `round16/pilot-r16c.json` exists.

**Rule:** R17 becomes the end-to-end leader only if all of these hold:
1. paired BLEU R17 − R16C has a 95% lower bound > 0;
2. paired BLEU R17 − W14b step 30000 (`v3read/pilot-w14b-step30000-v3d.json`) has a lower bound > 0;
3. v3 lexicon R17 − R16C has a lower bound > −0.05 (no trading words for BLEU);
4. I1 holds.

**Also reported:**
- R17's point BLEU against blank (1.859), W14b's composition (2.31) and M1 (2.55);
- wrist speed relative to real against the 1.5× bar;
- the **oracle-plan** pilot's BLEU. It is an upper bound on what the plan route can carry with this detail prior:
  - oracle well above R17: the plan prior is the bottleneck;
  - oracle at R17's level: the detail prior does not follow the plan, or the plan carries no sentence content.

The error-discovery checklist of "Road to 3.8" applies to every verdict.

## Risks and open points

- **Learning rate of the new modules.** The plan modules start at zero and train at W14b's continuation rate (1e-4, warmup 200). If `val/plan_gain` stays near 0 by step 2000, the most likely cause is this rate, not the plan. A per-group learning rate is not implemented, deliberately: it would be a declared change.
- **Teacher forcing hides the plan's value.** With the real history, the next step at 24 Hz is nearly determined, so I1 may show a small gain even when the plan helps free running. I1 is a necessary check, not the success criterion; the pilot decides.
- **The plan prior's temporal stack is new.** At 6 Hz and 30,000 steps it is the least-tested piece. Its teacher-forced validation loss and its own free-running behaviour should be read before the pilot.
- **Dropout and exactness.** Warm-start exactness is bit-level in evaluation and in RNG-free training. Under dropout 0.1 the random stream shifts (see "Exactness scope").
- **Pre-existing.** Eight stored prior configs whose recorded SHA the original code does not reproduce. This branch leaves them unchanged.
