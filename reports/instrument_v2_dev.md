# Instrument v2 (state-to-text back-translator): development report

Branch `instrument-v2`, built against the declaration in docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared
gates" (2026-10-30, 09:10 UTC), and its amendment 1 (09:56 UTC). The gates are implemented as declared; nothing has
trained. Run `back-translator-state-t5-v1`, kind `state_to_text_t5`. The independent review of 86b8fca returned APPROVE
WITH FIXES; every finding is fixed (section "Review fixes"). Amendment 1 settles the open questions of 86b8fca.

## What was built

| File | What it does |
|---|---|
| `signtranslator/data/state_features.py` | The input. `from_state_3d` gives v1's 48 forward-kinematics points with z kept (x and y equal v1's `keypoints2d.from_state`). `normalise_nd` is v1's normalisation in any dimension, with no y flip. `native_features` gives the native channels in the diffusion space, with root and head translation centred on the clip median and invalid cells zero, plus per-element validity. `state_features` concatenates [3D points 144, point validity 48, native W, element flags E]. A blank state gives all zeros. |
| `signtranslator/generation/state_to_text_t5.py` | `KIND`, `RUN`, `LICENCE`. The frozen `StateToT5Config` holds the native layout (a `TextToMotionConfig` from the template's bindings), the template pin (manifest key and SHA, corpus, sign, state SHA), the feature version, fps 24, the licence tag, and the encoder: d 512, 6 layers, 8 heads, ff 2048, ×2 downsample, flan-t5-base. `StateToT5` uses v1's encoder→T5 path. `layout_template` lets any state be conformed without the stored template. `prepare` runs conform → 24 fps → features. `load_checkpoint` and `load_model` read with `torch.load(weights_only=True)` and an optional SHA pin. |
| `signtranslator/generation/bt_corpora.py` | Roles per corpus and partition: How2Sign train c02 is `train`; c04 is `select:how2sign-c04`; val c01/c03/c05 is `val:how2sign`; test, held-out and c00 are never read. YouTube train is `train` and monitor is `select:youtube-monitor`: all 536 monitor clips (amendment 1). STEM train is `train` and holdout is `holdout:asl-stem-wiki`. Every 2M-Flores record is `crossdomain:2m-flores`. `assemble` handles text hygiene and the monitor sample. `check_loadable_bt` is the hard guard, applied on assembly and on every load. |
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
   - SELECTION = c04 + the 536 YouTube monitor clips (amendment 1; the declaration said 1,000).

   c04 never trains. c02 has 17,945 records and c04 has 344 (round10/round14b mixes).
2. **Text hygiene for every training corpus.** A training sentence is dropped if it is equal or near to a How2Sign val/test/held-out sentence or to any 2M-Flores sentence (`NearDuplicates`, the refiner pairs' rule). This covers YouTube and STEM too, not only How2Sign. A training sentence whose normalised text equals a selection or held-out sentence is also dropped. Every drop is counted in `run.json`.
3. **G4 and G6 use paired CIs.** The gate header says "paired 95% CIs", so "real CI above blank / mismatched" is read as the paired lower bound of real − control > 0. The unpaired interval separation is reported beside it.
4. **G5's "keeps its sign"** means the sign of v2's *point* difference equals v1's, for every pair that v1 resolved. Whether v2 also resolves the pair is reported.
   - Tau is Kendall's tau-a over the six BLEUs.
   - Both instruments read the same regenerated samples (amendment 1).
   - Exact equality with the stored v1 hypotheses is a diagnostic only (review fix 1).
5. **Static-valid.** The first frame is held for the clip's length (`round15_compose_eval.held`), with every channel the instrument reads valid. Each element takes its first valid value, or the neutral value if it is never valid. Blank-at-median is one all-invalid state of the median frame count, read once and scored against every clip.
6. **Mismatched** uses `sentence_exposure_diagnostic.derangement(n, seed 0)` and `compositional.time_scale`, which stretches B to A's frame count.
7. **Shuffled chunks** split the clip into 12-frame (0.5 s) chunks and permute them with `default_rng([seed, clip])`. The length ratio is hypothesis words / reference words, per form.
8. **The d2 round trip in G3** is `build_recon_states`' recon_full form: d2 codes decoded as a generation, the accepted gains, then head smoothing. This is the form used in training.
9. **G2's resolution report:**
   - v1: (real − blank) / half-width of the W14b − R16C paired CI, from v1's reads of the regenerated samples.
   - v2: G2's gap / half-width of the v2 paired CI.
10. **Bootstrap:**
    - single forms: `strict_prior_review.bleu_ci`'s draws (1,000 resamples, seed 0), unrounded;
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

## Review fixes (review of 86b8fca: APPROVE WITH FIXES; amendment 1, 09:56 UTC)

| # | Finding | Resolution |
|---|---|---|
| 1 | BLOCKER: G5 built its orderings from v1's *stored* hypotheses | `gate_g5` judges `v1_rerun` against `v2`: both instruments on the same regenerated samples (amendment 1). Exact reproduction of the stored pilot hypotheses is reported (`reproduced_stored_v1_diagnostic`) and decides nothing. The G2 resolution report also uses `v1_rerun`. The test and the docstrings of the validator and `reread_arms.py` are updated. |
| 2 | MAJOR: the composition's v1 and v2 reads came from two separate runs | Single-run design: `round15_compose_eval.py run --also-translator-run RUN` (repeatable) reads every form with the extra translator too, on the same states. Shards store `extra_hypotheses`; the merge writes `extra_translators`. Absent the flag, runs and reports are unchanged (tested: identical rows and merged report). `reread_arms.py --composition-rerun` takes that one merged report: settings must equal the stored run's (the extra translator aside), and the extra translator must be the pinned instrument checkpoint. The job runs the composition once (4 shards). |
| 3 | MAJOR: monitor clip count | `MONITOR_CLIPS = 536` in `bt_corpora` (the trainer's default) and in the job script. |
| 4 | MINOR: a renamed record could train | `role_of` returns None when `source_partition` differs from `partition`. In training, `check_loadable_bt` requires `source_partition` in (None, "train"). Tested with a renamed How2Sign val record and a renamed YouTube monitor record. |
| 5 | MINOR: 2M-Flores texts were filtered only if given | The trainer refuses to run unless a cross-domain (2M-Flores) manifest is among `--corpus` (tested). |
| 6 | MINOR: trust-on-first-use pins, exists-or-build races | New phases `manifests` (STEM and Flores manifests, write-once; prints their SHAs) and `index` (merges the round-trip shards; prints the index SHA). The `data`, `index`, `train` and `eval` phases refuse unless `STEM_SHA`, `FLORES_SHA` and `RECON_INDEX_SHA` are hard-coded in the script (placeholders `PIN-...` are refused), and they verify the bytes against the pins. No phase builds what another reads. |
| 7 | MINOR: eval-phase exit handling and overwrites | The eval phase exits 1 unless the validator's rc ∈ {0, 2, 4}. It refuses to start if `reread-arms.json` or `acceptance.json` exist; both scripts also refuse an existing `--output-key` (write-once, tested). |
| 8 | MINOR: gates decided on rounded values | Every gate is decided on unrounded values: `paired` no longer rounds; `bleu_ci_raw` reproduces `strict_prior_review.bleu_ci`'s draws without its rounding (tested equal after rounding); the report is rounded to 4 decimals only when written (`rounded`). Tested with a lower bound of 0.0004 (G4, G6) and edge values for G1/G2. |
| 9 | MINOR: selection vs acceptance decoding | Selection now reads every selection and report-only clip at full length (as the acceptance does) and decodes greedily (beam 1), while the acceptance uses beam 4. Recorded in `run.json` (`selection.decoding`). The acceptance reports clips over 20 s (the training cut) separately (`long_clips`: real and blank BLEU over and up to 20 s). |
| 10 | MINOR: `--translator-run` with v2 loaded through the pickle path | `score_back_translation.load_translator_run`: when the run's `best.json` records `checkpoint_sha256` (the v2 trainer writes it), the payload is checked against it and loaded weights-only (`state_to_text_t5.load_model`). Otherwise the run is loaded by `load_translator` exactly as before. The pilot and round-15 scripts call it only for non-default runs; v1's path is untouched. Tested: SHA match, mismatch refused, v1 fallback. |
| 11 | INFO: sampling proportions | `run.json` records `sampling`: per corpus clips, weight and mass share (weight x clips, normalised). See below. |
| — | Worker counts | The job script uses fixed worker counts (`BT_WORKERS`, default 28 on a 32-vCPU CPU pod; `--workers 10` for training), never `nproc`. |

**Effective sampling proportions** (by mass). Measured on round14b-mixed-v1 with this branch's `assemble`, without 2M-Flores texts; Flores filtering can only remove a few more:
- How2Sign c02: 17,585 train clips × 4.
- YouTube-ASL: 99,808 × 1.
- Without STEM, the shares are How2Sign 0.413 and YouTube 0.587.
- With STEM's n training clips × 1, they are 70,340 : 99,808 : n. At n ≈ 60,000 (the full extraction minus 4 participants) that gives about How2Sign 0.31, YouTube 0.43 and STEM 0.26.
- The exact shares are written into `run.json` when the run starts.

Hygiene drops on the same mix:
- How2Sign: 233 exact evaluation texts, 124 near, 3 selection texts.
- YouTube: 3 exact, 101 near, 6 selection texts.

## Deviations and blockers

- **ASL STEM Wiki extraction is incomplete** (about 6k of 64k reports at 2026-10-30). The `manifests` phase runs after it completes (amendment 1). It builds the manifest once.
- **The pilot reports omit some regeneration inputs:**
  - the generator mix (supplied as `--mix-key`, default round14b; `load_generator` refuses a wrong mix, and the texts must equal the stored references);
  - the code version;
  - the GPU/CUDA/library versions.

  Reproduction of the stored hypotheses is a diagnostic (amendment 1). G5 uses both instruments on the same regenerated samples. Run the eval phase on the pilots' GPU type where available.
- **The stored composition's settings must match today's.** The composition re-run is compared with the stored run's settings. If today's code records a settings key the stored run lacked, `reread_arms.py` refuses and names the key.
- **The generic loader still unpickles.** `score_back_translation.load_translator` keeps its generic `torch.load(weights_only=False)` for keys given directly (v1's path, unchanged). Run-based loading of v2 goes through `load_translator_run` (finding 10), and the new scripts read v2 weights-only.

## Pod job (prepared, not launched)

`runpod/fleet/state_back_translator.sh` follows `round17_pt.sh`'s preamble: r2env, venv, transformers install, SHA pins.

```bash
# 0. manifests (CPU, once, after the STEM extraction): prints PIN STEM_SHA=... and PIN FLORES_SHA=...
BT_PHASE=manifests bash runpod/fleet/state_back_translator.sh
#    -> hard-code STEM_SHA and FLORES_SHA in the script, commit, re-bundle
# 1. data (CPU pods, 28 workers each): the d2 round-trip copies of every training record
BT_PHASE=data BT_SHARD=0 bash runpod/fleet/state_back_translator.sh      # ... BT_SHARD=1, 2, 3
# 2. index (CPU, once): prints PIN RECON_INDEX_SHA=...  -> hard-code it, commit, re-bundle
BT_PHASE=index bash runpod/fleet/state_back_translator.sh
# 3. train (GPU): smoke (300 steps, selection on 64 clips, acceptance on 32 clips), then the full run
BT_PHASE=train bash runpod/fleet/state_back_translator.sh
# 4. eval (GPU, the pilots' type where available): the composition once (4 shards, read by v1 and v2), the arms
#    re-read, the acceptance G1-G6
BT_PHASE=eval bash runpod/fleet/state_back_translator.sh
```

The underlying commands, with `$O=acquisition/2026-10-17-v1/youtube-asl-v1` and `$A=acquisition/2026-10-30-v1`:

```bash
python scripts/build_video_corpus_manifest.py --prefix $A/asl-stem-wiki/extract-v1 --corpus-id asl-stem-wiki-v1 \
  --text-field "sentence text" --holdout-count 4 --manifest-key $A/asl-stem-wiki/asl-stem-wiki-v1/manifest.json --licence "..."
python scripts/build_video_corpus_manifest.py --prefix $A/2m-flores-asl/extract-v2 --corpus-id 2m-flores-asl-v1 \
  --text-field sentence --crossdomain --manifest-key $A/2m-flores-asl/2m-flores-asl-v1/manifest.json --licence "CC BY-SA 4.0"
python scripts/build_recon_states.py --manifest $MIX@$MSHA --manifest $STEM@$STEM_SHA --corpus how2sign-two-view-text-v1 \
  --corpus youtube-asl-v1 --corpus asl-stem-wiki-v1 --partition train --prefix $O/instrument-v2/recon-d2-v1 \
  --prior-checkpoint-key training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt \
  --shards 4 --shard $S --workers 28 --device cpu            # then --merge --shards 4
python scripts/train_state_to_text_t5.py --run back-translator-state-t5-v1 \
  --corpus $MIX@$MSHA:4:how2sign-two-view-text-v1 --corpus $MIX@$MSHA:1:youtube-asl-v1 --corpus $STEM@$STEM_SHA:1 \
  --corpus $FLORES@$FLORES_SHA:1 --template $R10@$R10SHA:how2sign-two-view-text-v1 \
  --recon-prefix $O/instrument-v2/recon-d2-v1 --recon-index-sha256 $RECON_INDEX_SHA --recon-p 0.4 --validity-dropout 0.3 \
  --heads 8 --feedforward 2048 --monitor-clips 536 --workers 10 --batch-size 64 --steps 60000 --val-every 2000 \
  --checkpoint-every 4000 --patience 6
python scripts/round15_compose_eval.py run <the declared W14b composition command> \
  --also-translator-run back-translator-state-t5-v1 --shard $S --output-key $O/instrument-v2/composition-shard$S.json
python scripts/round15_compose_eval.py merge --shard-keys <the 4 shards> --output-key $O/instrument-v2/composition.json
python scripts/reread_arms.py --translator-key $V2 --translator-sha256 $V2SHA \
  --composition-rerun $O/instrument-v2/composition.json --instrument-run back-translator-state-t5-v1 \
  --output-key $O/instrument-v2/reread-arms.json
python scripts/validate_state_back_translator.py --translator-key $V2 --translator-sha256 $V2SHA \
  --reread-key $O/instrument-v2/reread-arms.json --flores-manifest $FLORES@$FLORES_SHA \
  --output-key $O/instrument-v2/acceptance.json
```

`$FLORES` must be given to the trainer, and the trainer refuses to run without it. Its sentences are removed from the training text. Its records get the cross-domain role and are never trained on.

## Tests

All run with `OMP_NUM_THREADS=1 nice ~/.venvs/sign-translator/bin/python -W error -m pytest -q`.

- **New: `tests/test_instrument_v2.py`, 22 tests, all pass** (19 at 86b8fca; after the review fixes, the G5 test is rewritten and three tests are added: unrounded gates, the one-run two-instrument composition, and the SHA-checked run loader). They cover:
  - the featurisation round trip and shapes (3D points extend v1's 2D; `normalise_nd` equals v1's normalisation in 2D; every feature block decodes back; native values equal `diffusion_targets × diffusion_valid`; a blank state is all zeros);
  - the validity mask surviving conform (a monocular state with a tracking gap and absent translation/gaze; an undeclared layout is refused; resampling to 24 fps);
  - val, c00, held-out, cross-domain, monitor and c04 records refused in training, and a forged role caught;
  - `assemble` roles, evaluation-text drops, the monitor sample, and refusal when the monitor sample is short;
  - controls built at the reference duration;
  - `paired_gap` equalling `paired_bleu_difference` for one control, Kendall tau, and the G1–G6 gate logic;
  - G5 on the same regenerated samples (`v1_rerun` against `v2`): it fails on a flipped resolved order, and stored-hypothesis reproduction never decides it;
  - gates decided on unrounded values;
  - the reread blocks and seeds, the reproduction check and the composition settings checks;
  - a tiny end-to-end run (tiny random T5, no downloads):
    - it trains, selects on the pooled gap and records the licence;
    - it never fetches a val, c00, test, held-out-official or text-dropped state;
    - its checkpoint loads weights-only and is SHA-pinned;
    - the scorer dispatches on the kind and translates (the same output as the model loaded directly);
    - resume refuses other data;
    - a run without the 2M-Flores manifest is refused;
    - `run.json` records the sampling mass and the selection decoding;
  - the dataset serving the round-trip copy under the same text, augmentation and the blank control;
  - `--translator-run` defaulting to v1 everywhere, the pilot's default report staying byte-identical while `--translator-run` adds only `translator_run`, and the fidelity check's kind gate;
  - STEM and Flores manifests (holdout groups, cross-domain, write-once);
  - recon jobs limited to the training records of named corpora;
  - the acceptance's `main` end to end (write-once, `long_clips`), and `reread_arms.main` end to end feeding G5 (write-once);
  - `round15_compose_eval --also-translator-run`: identical main reads, extra reads of the same states, and a default run and merge unchanged;
  - `load_translator_run`: the SHA check, weights-only loading and the v1 fallback.
- **Existing related suites, all pass:**
  - 31 tests: `test_pilot_content_check`, `test_round15_compose_eval`, `test_tokenizer_fidelity_check_script`, `test_generation_keypoints_to_text_t5`, `test_train_keypoints_to_text_t5_script`, `test_generation_motion_to_text_t5`, `test_data_keypoints2d`.
  - 184 tests: `test_recognizer_v3_pipeline`, `test_recognizer_v3_data`, `test_refiner`, `test_rerank`, `test_composition_space`, `test_data_layout_adapter`, `test_score_saved_generated_script`, `test_score_postprocessing_grid_script`, `test_round15_compositional`, `test_round15_stage2`, `test_round15_oracle`, `test_compare_pilot_lexicon`, `test_sign_recognizer`.
- **Full warning-strict suite** (`python -W error -m pytest`, CI's command; plus `compileall` of signtranslator, tests and scripts): **4,031 passed**, 0 failed, 0 skipped, on the Mac (1 BLAS thread, nice), after the review fixes (4,028 at 86b8fca).

## Open questions

Amendment 1 (09:56 UTC) settled the questions of 86b8fca:
- 536 monitor clips;
- c04 is selection only;
- weights How2Sign 4 : YouTube 1 : STEM 1, validity dropout 0.3, 8 heads, ff 2048;
- G4 and G6 paired;
- G5 on the same regenerated samples;
- the data phase after STEM extraction.

None remain open.
