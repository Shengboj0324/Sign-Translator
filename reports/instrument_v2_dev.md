# Instrument v2 (state-to-text back-translator): development report

Branch `instrument-v2`, built against the declaration in docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared
gates" (2026-10-30, 09:10 UTC). The gates are implemented as declared; nothing has trained. The plan requires an
**independent review before any pod trains**. Run `back-translator-state-t5-v1`, kind `state_to_text_t5`.

## What was built

| File | What it does |
|---|---|
| `signtranslator/data/state_features.py` | The input. `from_state_3d` gives v1's 48 forward-kinematics points with z kept (x and y equal v1's `keypoints2d.from_state`). `normalise_nd` is v1's normalisation in any dimension, with no y flip. `native_features` gives the native channels in the diffusion space, with root and head translation centred on the clip median and invalid cells zero, plus per-element validity. `state_features` concatenates [3D points 144, point validity 48, native W, element flags E]. A blank state gives all zeros. |
| `signtranslator/generation/state_to_text_t5.py` | `KIND`, `RUN`, `LICENCE`. The frozen `StateToT5Config` holds the native layout (a `TextToMotionConfig` from the template's bindings), the template pin (manifest key and SHA, corpus, sign, state SHA), the feature version, fps 24, the licence tag, and the encoder: d 512, 6 layers, 8 heads, ff 2048, ×2 downsample, flan-t5-base. `StateToT5` uses v1's encoder→T5 path. `layout_template` lets any state be conformed without the stored template. `prepare` runs conform → 24 fps → features. `load_checkpoint` and `load_model` read with `torch.load(weights_only=True)` and an optional SHA pin. |
| `signtranslator/generation/bt_corpora.py` | Roles per corpus and partition: How2Sign train c02 is `train`; c04 is `select:how2sign-c04`; val c01/c03/c05 is `val:how2sign`; test, held-out and c00 are never read. YouTube train is `train` and monitor is `select:youtube-monitor`, a fixed sample of 1,000. STEM train is `train` and holdout is `holdout:asl-stem-wiki`. Every 2M-Flores record is `crossdomain:2m-flores`. `assemble` handles text hygiene and the monitor sample. `check_loadable_bt` is the hard guard, applied on assembly and on every load. |
| `signtranslator/data/state_text_dataset.py` | `StateTextDataset` (described below). `blank=True` serves the selection-only control and is refused with `training`. |
| `scripts/build_video_corpus_manifest.py` | Manifests of `extract_video_corpus.py` output. Held-out groups come from `--holdout-groups` or `--holdout-count`, ordered by SHA-256. `--crossdomain` covers 2M-Flores. Write-once. |
| `scripts/train_state_to_text_t5.py` | Fork of the v1 trainer. Takes `--corpus KEY@SHA[:WEIGHT[:CORPUS,...]]` (repeatable) and `--template`. Augments with speed, trim, face dropout and `--validity-dropout`, plus `--recon-prefix/--recon-index-sha256/--recon-p` and `--monitor-clips`. Selects on `--select` with `--select-on gap` (BLEU(real) − BLEU(blank), pooled over c04 and the monitor sample). Held-out STEM clips are read and reported only. Sampling is weighted per corpus. Resume, patience and EMA work as in v1. Checkpoints are weights-only and carry the licence. |
| `scripts/validate_state_back_translator.py` | Gates G1–G6 with every declared control and report-only form (below). Exit 0 if all gates pass, 4 if a gate failed, 2 if the result is incomplete. |
| `scripts/reread_arms.py` | Regenerates W14b, R16 and R16C with the pilots' blocks and seeds, checks that v1 reproduces the stored hypotheses exactly, then reads with v2. Real and blank get the same treatment. The composition comes from two `round15_compose_eval.py` re-runs (v1 and v2). |
| `runpod/fleet/state_back_translator.sh` | Prepared pod job in three phases (data, train, eval), each with a smoke run before the full run. |

The dataset serves:
- the training role only (every load is guarded);
- the d2 round-trip copy at `recon_p` (default 0.4), under the same text;
- `load_sample` for the real clip;
- states conformed to the template, passed through `augment_sign`, resampled to 24 fps and featurised.

**Modified, additions only (v1 defaults and behaviour unchanged):**
- `score_back_translation.py`: a `state_to_text_t5` branch. `load_translator` checks the config hash and returns kind `"states"`; `translate` featurises through `model.featurise`.
- `pilot_content_check.py`: `--translator-run`, defaulting to v1's run. The report gains `translator_run` only when the flag is not the default. The existing byte-identity test `DEFAULT_REPORT_SHA256` still passes.
- `round15_compose_eval.py`: `--translator-run`, defaulting to v1. The checkpoint was already recorded in `settings.translator_key`.
- `tokenizer_fidelity_check.py`: `check_translator_kind` accepts `"keypoints"` or `"states"`.
- `build_recon_states.py`: `--partition` and `--corpus` select the training records of sentence manifests. The shard and merged index gain these fields only when the flags are given.

## Spec clarifications (decided here; each is for the reviewer to confirm)

1. **Selection is disjoint from training.** The declaration lists How2Sign c04 both as training data and as selection data. Selection data must not be trained on, so:
   - TRAIN = How2Sign c02 + YouTube train + STEM non-held-out participants;
   - SELECTION = c04 + the fixed 1,000 YouTube monitor clips.

   c04 never trains. c02 has 17,945 records and c04 has 344 (round10/round14b mixes).
2. **Text hygiene for every training corpus.** A training sentence is dropped if it is equal or near to a How2Sign val/test/held-out sentence or to any 2M-Flores sentence (`NearDuplicates`, the refiner pairs' rule). This covers YouTube and STEM too, not only How2Sign. A training sentence whose normalised text equals a selection or held-out sentence is also dropped. Every drop is counted in `run.json`.
3. **G4 and G6 use paired CIs.** The gate header says "paired 95% CIs", so "real CI above blank / mismatched" is read as the paired lower bound of real − control > 0. The unpaired interval separation is reported beside it.
4. **G5's "keeps its sign"** means the sign of v2's *point* difference equals v1's, for every pair that v1 resolved. Whether v2 also resolves the pair is reported.
   - Tau is Kendall's tau-a over the six BLEUs.
   - Reproduction means exact equality of every stored v1 hypothesis.
   - Any non-reproduced arm fails G5.
5. **Static-valid.** The first frame is held for the clip's length (`round15_compose_eval.held`), with every channel the instrument reads valid. Each element takes its first valid value, or the neutral value if it is never valid. Blank-at-median is one all-invalid state of the median frame count, read once and scored against every clip.
6. **Mismatched** uses `sentence_exposure_diagnostic.derangement(n, seed 0)` and `compositional.time_scale`, which stretches B to A's frame count.
7. **Shuffled chunks** split the clip into 12-frame (0.5 s) chunks and permute them with `default_rng([seed, clip])`. The length ratio is hypothesis words / reference words, per form.
8. **The d2 round trip in G3** is `build_recon_states`' recon_full form: d2 codes decoded as a generation, the accepted gains, then head smoothing. This is the form used in training.
9. **G2's resolution report:**
   - v1: (real − blank) / half-width of the W14b − R16C paired CI, from the stored v1 hypotheses.
   - v2: G2's gap / half-width of the v2 paired CI.
10. **Bootstrap:**
    - single forms: `strict_prior_review.bleu_ci` (1,000 resamples, seed 0);
    - paired differences: `compare_pilot_lexicon.paired_bleu_difference` (2,000 resamples, seed 0);
    - G2's gap over the maximum control: `paired_gap`, which draws the same indices and takes the maximum control in each resample.
11. **Free choices not fixed by the declaration:**
    - corpus weights: How2Sign 4, YouTube 1, STEM 1 (`H2S_WEIGHT`);
    - `--validity-dropout` 0.3; face dropout 0.2; speed 0.8–1.25; trim ≤10%;
    - 8 heads, ff 2048;
    - lr 3e-4 / 5e-5, 60k steps, patience 6, validation every 2,000 steps;
    - `--recon-min-coverage` 0.9;
    - the STEM held-out participants are the first 4 groups by SHA-256("asl-stem-wiki-v1\n<participant>").
12. **UWB-lifted pretraining** (optional in the declaration) is not implemented. The lifter gate has not passed.

## Deviations and blockers found

- **The 1,000 YouTube monitor clips do not exist yet.** `youtube-asl-v1` in the round10 and round14b mixes has **536** monitor clips. `assemble` refuses fewer than `--monitor-clips` (default 1000), so the train phase stops on today's data. There are two ways forward:
  - a YouTube manifest with ≥1,000 monitor clips (the home-route downloads);
  - an explicit, recorded `MONITOR_CLIPS=536`.

  This is the reviewer's or owner's call.
- **ASL STEM Wiki extraction is incomplete** (about 6k of 64k reports at 2026-10-30). The data phase builds the STEM manifest from whatever reports exist, write-once. Run it after the extraction finishes.
- **The pilot reports omit some reproduction inputs:** the generator mix (supplied as `--mix-key`, default round14b; `load_generator` refuses a wrong mix and the texts must equal the stored references), the code version, and the GPU/CUDA/library versions. Only the exact-equality check shows the regeneration is faithful. If CUDA kernels differ from the pilots' GPUs, the reproduction (and so G5) fails, and reports the clips that differ.
- **The composition (W14b, 2.31) can only be reproduced by re-running it.** The job re-runs `round15_compose_eval.py` twice, 4 shards each (v1 and v2), with the declared command (`job_r15e-w14b-s*.sh`). That is about 2 × the original composition run's GPU time.
- **Some existing paths still unpickle checkpoints.** `score_back_translation.load_translator` keeps its generic `torch.load(weights_only=False)` before dispatching on the kind, because changing it would change the v1 path. New code (trainer resume, validator, reread, `load_model`) reads v2 checkpoints weights-only.

## Pod job (prepared, not launched)

`runpod/fleet/state_back_translator.sh` follows `round17_pt.sh`'s preamble: r2env, venv, the transformers install, SHA pins for the mix, the template mix and the d2 tokenizer, and the smoke run first.

```bash
# 1. data (CPU pods, after the STEM extraction completes): manifests + d2 round-trip copies, 4 shards, then merge
BT_PHASE=data BT_SHARD=0 BT_SHARDS=4 bash runpod/fleet/state_back_translator.sh    # ... shards 1, 2, 3
BT_PHASE=data BT_SHARD=merge BT_SHARDS=4 bash runpod/fleet/state_back_translator.sh
# 2. train (GPU): smoke (300 steps, selection on 64 clips, acceptance on 32 clips), then the full run
BT_PHASE=train MONITOR_CLIPS=1000 bash runpod/fleet/state_back_translator.sh
# 3. eval (GPU): composition re-runs (v1, v2), re-read of the arms, the acceptance G1-G6
BT_PHASE=eval bash runpod/fleet/state_back_translator.sh
```

The underlying commands, with `$O=acquisition/2026-10-17-v1/youtube-asl-v1` and `$A=acquisition/2026-10-30-v1`:

```bash
python scripts/build_video_corpus_manifest.py --prefix $A/asl-stem-wiki/extract-v1 --corpus-id asl-stem-wiki-v1 \
  --text-field "sentence text" --holdout-count 4 --manifest-key $A/asl-stem-wiki/asl-stem-wiki-v1/manifest.json --licence "..."
python scripts/build_video_corpus_manifest.py --prefix $A/2m-flores-asl/extract-v2 --corpus-id 2m-flores-asl-v1 \
  --text-field sentence --crossdomain --manifest-key $A/2m-flores-asl/2m-flores-asl-v1/manifest.json --licence "CC BY-SA 4.0"
python scripts/build_recon_states.py --manifest $MIX@$MSHA --manifest $STEM@$SSHA --corpus how2sign-two-view-text-v1 \
  --corpus youtube-asl-v1 --corpus asl-stem-wiki-v1 --partition train --prefix $O/instrument-v2/recon-d2-v1 \
  --prior-checkpoint-key training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt \
  --shards 4 --shard $S --workers 28 --device cpu            # then --merge --shards 4
python scripts/train_state_to_text_t5.py --run back-translator-state-t5-v1 \
  --corpus $MIX@$MSHA:4:how2sign-two-view-text-v1 --corpus $MIX@$MSHA:1:youtube-asl-v1 --corpus $STEM@$SSHA:1 \
  --corpus $FLORES@$FSHA:1 --template $R10@$R10SHA:how2sign-two-view-text-v1 \
  --recon-prefix $O/instrument-v2/recon-d2-v1 --recon-index-sha256 $RISHA --recon-p 0.4 --validity-dropout 0.3 \
  --monitor-clips 1000 --workers 10 --batch-size 64 --steps 60000 --val-every 2000 --checkpoint-every 4000 --patience 6
python scripts/round15_compose_eval.py run <the declared W14b composition command> [--translator-run back-translator-state-t5-v1] \
  --shard $S --output-key $O/instrument-v2/composition-{v1,v2}-shard$S.json      # 4 shards each, then merge
python scripts/reread_arms.py --translator-key $V2 --translator-sha256 $V2SHA \
  --composition-v1-rerun $O/instrument-v2/composition-v1.json --composition-v2 $O/instrument-v2/composition-v2.json \
  --output-key $O/instrument-v2/reread-arms.json
python scripts/validate_state_back_translator.py --translator-key $V2 --translator-sha256 $V2SHA \
  --reread-key $O/instrument-v2/reread-arms.json --flores-manifest $FLORES@$FSHA --output-key $O/instrument-v2/acceptance.json
```

`$FLORES` is passed to the trainer only so its sentences are excluded from training text. Its records get the cross-domain role, and `check_loadable_bt` refuses them in training.

## Tests

All run with `OMP_NUM_THREADS=1 nice ~/.venvs/sign-translator/bin/python -W error -m pytest -q`.

- **New: `tests/test_instrument_v2.py`, 19 tests, all pass.** They cover:
  - the featurisation round trip and shapes (3D points extend v1's 2D; `normalise_nd` equals v1's normalisation in 2D; every feature block decodes back; native values equal `diffusion_targets × diffusion_valid`; a blank state is all zeros);
  - the validity mask surviving conform (a monocular state with a tracking gap and absent translation/gaze; an undeclared layout is refused; resampling to 24 fps);
  - val, c00, held-out, cross-domain, monitor and c04 records refused in training, and a forged role caught;
  - `assemble` roles, evaluation-text drops, the monitor sample, and refusal when the monitor sample is short;
  - controls built at the reference duration;
  - `paired_gap` equalling `paired_bleu_difference` for one control, Kendall tau, and the G1–G6 gate logic;
  - G5 failing on an unreproduced arm or a flipped resolved order;
  - the reread blocks and seeds, the reproduction check and the composition settings checks;
  - a tiny end-to-end run (tiny random T5, no downloads):
    - it trains, selects on the pooled gap and records the licence;
    - it never fetches a val, c00, test, held-out-official or text-dropped state;
    - its checkpoint loads weights-only and is SHA-pinned;
    - the scorer dispatches on the kind and translates (the same output as the model loaded directly);
    - resume refuses other data;
  - the dataset serving the round-trip copy under the same text, augmentation and the blank control;
  - `--translator-run` defaulting to v1 everywhere, the pilot's default report staying byte-identical while `--translator-run` adds only `translator_run`, and the fidelity check's kind gate;
  - STEM and Flores manifests (holdout groups, cross-domain, write-once);
  - recon jobs limited to the training records of named corpora;
  - the acceptance's `main` end to end, and `reread_arms.main` end to end feeding G5.
- **Existing related suites, all pass:**
  - 31 tests: `test_pilot_content_check`, `test_round15_compose_eval`, `test_tokenizer_fidelity_check_script`, `test_generation_keypoints_to_text_t5`, `test_train_keypoints_to_text_t5_script`, `test_generation_motion_to_text_t5`, `test_data_keypoints2d`.
  - 184 tests: `test_recognizer_v3_pipeline`, `test_recognizer_v3_data`, `test_refiner`, `test_rerank`, `test_composition_space`, `test_data_layout_adapter`, `test_score_saved_generated_script`, `test_score_postprocessing_grid_script`, `test_round15_compositional`, `test_round15_stage2`, `test_round15_oracle`, `test_compare_pilot_lexicon`, `test_sign_recognizer`.
- **Full warning-strict suite** (`python -W error -m pytest`, CI's command; plus `compileall` of signtranslator, tests and scripts): **4,028 passed**, 0 failed, 0 skipped, on the Mac (1 BLAS thread, nice).

## Open questions for the review

1. Should the YouTube monitor selection use 536 clips now, or wait for a manifest with ≥1,000 monitor clips?
2. Should the c04-as-selection-only resolution (clarification 1) be confirmed in the plan?
3. Are the corpus weights (How2Sign 4 : YouTube 1 : STEM 1) and validity dropout 0.3 acceptable as undeclared training choices?
4. G4 and G6 are read as paired lower bounds. Should the stricter unpaired separation be the rule instead? Both are reported.
5. If the pilots' GPU kernels differ, exact reproduction may fail for reasons unrelated to the instrument. The declaration requires reproduction, so G5 then fails. Should a re-generation on the pilots' GPU type be required?
