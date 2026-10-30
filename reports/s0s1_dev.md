# S0 and S1 probes: build report

**Branch:** `probes-s0s1` (from main 5190949), worktree `~/src/st-probes`.
**Spec:** docs/NEXT_PHASE_PLAN.md, "Road to 3.8, part 2" (declared 2026-10-30, 19:32 UTC), stages S0 (the v1 parts) and S1. Background: `reports/Route to near real signing.md`.
**Scope:** evaluation only. Nothing here trains, reranks or selects on an instrument.

## What was built

| File | Purpose |
|---|---|
| `signtranslator/eval/s0_metrics.py` | Pure metric helpers: ΔNLL, retrieval ranks, chrF, fast BLEU/chrF bootstraps, ρ_faith (DTW and time-averaged), categorical KL |
| `scripts/s0_ceilings.py` | S0 ceilings and the measurement panel on instrument v1 (v2 through `--translator-run`) |
| `scripts/s1_text_kl.py` | S1: KL(p(code \| text) ‖ p(code \| empty)) on stage-1 codes, teacher forced on real val clips |
| `runpod/fleet/s0s1_probes.sh` | GPU evaluation pod job: 20-clip smoke with checks, then write-once outputs under `acquisition/2026-10-17-v1/youtube-asl-v1/s0s1/` |
| `tests/test_s0_metrics.py`, `tests/test_s0s1_scripts.py` | 26 tests on synthetic data |

## Clips and instrument

- **Clips:** `pilot_content_check.validation_sentences(round10-mixed-v1, 1000)`, the 1,000 How2Sign val clips that every pilot and the round-15 composition evaluation read, at 24 fps.
- **Smaller reads are prefixes of that draw (fix, 2026-10-30, after the first pod smoke).** `random.sample(val, k)` draws differently for different k: k = 1000 uses its pool method, while k = 20 or 32 uses its set method. So `validation_sentences(mix, 32)` agrees with the 1,000-draw only on its first 16 clips. The first pod smoke drew 32 clips for the W14b regeneration and was refused: its references were not the pilot's. Its 20-clip ceiling numbers (real 3.314, blank 2.332, …) came from that k = 20 draw, not the 1,000's first 20.
  - Now `--sample 1000` is always drawn and `--clips` takes its prefix. The report's `sample` block states the draw.
  - Regeneration uses each pilot's own clip list: `validation_sentences` of its generator's training mix (round14b for W14b, R16 and R16C; round19 for R19 and R19w) at the pilot's clip count, checked against its stored references and against the ceiling read's sign ids.
  - All five pilots resolve to the same 1,000 sign ids as round10's draw (checked on R2).
- **Instrument v1:** `back-translator-kp2d-uwb-ft-v1` step 9000 (SHA-256 2e3ded3f…), beam 4, batch 32, as every pilot.
- **Checked locally (4 clips, CPU):** v1 reproduces the stored eval1000 real hypotheses exactly (`identical_to_computed` = 1.0). The pod smoke repeats this check on 20 clips and stops the job if it fails.

## S0 forms

| Form | Definition |
|---|---|
| `real`, `blank` | The clip; the clip with every channel removed (durations kept) |
| `round_trip` | d2's round trip with no decoder gains, exactly as `tokenizer_fidelity_check` (it reproduces the 3.365 figure) |
| `round_trip_gains` | The same with d2's accepted decoder gains (body 1.17, hands 1.27–1.28), which every generated state carries. This is the ceiling that applies to generated output |
| `stages1`, `stages2`, `stages4` | Decoded from the first 1, 2 or 4 of d2's 8 residual stages (`quantize(latents, stages)`), no gains. `stages1` is S1's "stage-1-only round trip" |
| `face_mean`, `face_absent` | Face, blink and gaze held at the clip mean, or removed |
| `warp0.8`, `warp1.25` | Played at 0.8× or 1.25× speed, then resampled to 24 fps |
| `lowpass6` | 4th-order zero-phase Butterworth at 6 Hz, applied per joint on contiguous valid runs, then projected back onto each channel's manifold. A local helper; round 20's composition Butterworth lives on its own branch |

**v1 is face-blind by construction.** It reads 48 arm and hand points (`data/keypoints2d.py`: "Faces are left out"). So `face_mean` and `face_absent` must read exactly as `real` on v1. A test checks that the 2D features are bit-identical. The smoke stops if `identical_to_real` < 1. The face probe only means something on v2, and the forms are kept for that.

## Metrics: definitions and choices

**ΔNLL over blank**

- `NLL(ref | x)` is the teacher-forced token-mean negative log-likelihood of the reference. It uses the translator's own T5 tokenisation and truncation (`_labels`, 64 tokens) and includes the EOS token.
- ΔNLL = NLL(ref | blank) − NLL(ref | x). It is **positive when the motion helps**, and blank's ΔNLL is 0 by definition.
- The gap fraction is mean ΔNLL_x / mean ΔNLL_real, with a paired bootstrap ratio CI.

**100-way retrieval**

- Each clip's own reference is scored against 99 distractors drawn from the other 999 references, without replacement (seeded per clip). The same candidate sets are used for every form, so ranks are paired across forms.
- Ties take the mid-rank, so a constant score ranks at chance, 50.5. Chance top-1 is 1%.
- **Secondary, PMI ranks:** rank by NLL(c | x) − NLL(c | blank_i), which removes each candidate's template prior. Blank scores exactly chance on this by construction.
- ΔNLL equals column 0 of the same NLL matrix, so one pass gives both.

**ρ_faith**

- **Latent space:** d2's per-part encoder outputs (continuous, before quantisation; body, left hand, right hand, face; 4 × 128 dimensions at 24 Hz), over every frame.
- **Normalisation:** each dimension is standardised by the mean and standard deviation over the *real* clips' frames, so the scale is fixed by real signing.
- **Primary distance, DTW:** frame cost is the Euclidean distance / √D; the accumulated cost is divided by (n + m). This makes different lengths and tempi comparable, which the warped forms need. The DTW is batched over anti-diagonals in torch and tested against the plain recursion.
- **Secondary distance:** between time-averaged latents, which ignores order.
- **Random references:** K = 10 per clip, never the clip itself, the same for every form.
- **Ratio:** ρ_i = d(x_i, ref_i) / mean_k d(x_i, ref_k), averaged with a bootstrap CI. 0 = identical to its reference (real is 0 trivially); 1 = no closer to its reference than to a random one. Per-part ρ is reported too, which is S5's per-part probe on the round trips.
- **Caveats:**
  - ρ_faith is computed in the latent space of the same tokenizer whose round trips are scored. It flatters d2-decoded forms relative to non-d2 forms, so compare only within a family.
  - Blank's ρ is the "no content" control. It should be near 1, but it is not exactly 1, because a constant latent sits closer to low-variance references.

**Text metrics**

- **BLEU-4:** the repository's `corpus_bleu`. The vectorised bootstrap equals `bleu_ci` and `paired_bleu_difference` draw for draw (tested).
- **chrF:** sacreBLEU's chrF2 (character order 6, β = 2, whitespace removed), reimplemented because sacrebleu is not installed.
- **Length ratio:** hypothesis words / reference words.

**Orderings (S0's pass question)**

- Every pair of forms with a full panel (computed forms plus regenerated arms; `--merge-from` joins the two runs) gets paired differences, each with a 95% CI, for BLEU, ΔNLL, MRR and PMI-MRR.
- **A metric passes** when it reproduces every ordering BLEU resolves (none contradicted, none left unresolved) **and** its median half-width, normalised by its own real − blank range, is below BLEU's.
- Kendall's τ against BLEU's point estimates is reported alongside.

## Stored outputs: what can be read and what needs regeneration

All stored pilot and composition outputs in R2 hold **hypotheses only, no motion**.

| Stored output | Text metrics now (BLEU, chrF, length, gap, paired) | ΔNLL / retrieval / ρ_faith |
|---|---|---|
| round15/eval1000: W14b and F12 composition, /shuffled, /mismatched_words, paired, end-to-end | yes | need regeneration through `round15_compose_eval.py`'s renderer (round 20's tooling); **not implemented here**, to avoid duplicating round 20 |
| round15/eval1000: oracle-stage2, oracle-knobs-{all,jerk,space,tight}, oracle-refined-r1 (incl. @refined, @recon, /rest_only) | yes | same as above |
| End-to-end pilots, 1,000 clips: round14b W14b (10k, 30k), round16 R16 and R16C, round19 R19 and R19w, v3read (W14b, F12, control c1000, W14) | yes | **regenerable**: `--regenerate KEY` (the pod regenerates W14b, R16, R16C, R19, R19w) |
| round4 pilots (300 clips; another `random.sample` draw) | skipped (other clips) | regenerable in principle, but not on these 1,000 |
| Plan-conditioned (round 17) pilots | — | refused by `--regenerate` (they need plan priors) |

**How regeneration works:** exactly as `pilot_content_check`.
- Same inputs: the stored checkpoint, guidance and part temperatures.
- Same sampling: `sample.batch` per block of 32 with seed 1000 + i, the round-3e post-processing, and the tokenizer's accepted gains through `load_generator`.
- Always whole blocks of 32, even when `--clips` is smaller.
- The report gives `reproduction.identical`: the fraction of regenerated hypotheses equal to the stored ones, alongside both BLEUs. Stage G5 of the instrument-v2 plan wants reproduction before cross-instrument reads.

**Already read locally (CPU, stored hypotheses, all 1,000 clips; `--forms none`; stored at `s0s1/s0-stored-text-v1.json`, SHA-256 e4382d64…):**

| Form | BLEU | chrF [95% CI] | Length ratio | chrF − blank [95% CI] |
|---|---|---|---|---|
| real | 4.136 | 21.18 [20.64, 21.73] | 1.31 | +4.28 [3.79, 4.82] |
| blank | 1.859 | 16.90 [16.56, 17.24] | 1.43 | 0 |
| W14b composition (core_crossfade) | 2.306 | 17.00 [16.56, 17.46] | 1.68 | +0.10 [−0.39, 0.61] |
| W14b composition / mismatched words | 1.108 | 14.56 | 1.70 | −2.33 [−2.73, −1.93] |
| real-word oracle (oracle_stage2) | 2.170 | 16.98 | 1.65 | +0.08 [−0.41, 0.58] |
| W14b end-to-end | 1.786 | 16.64 | 1.32 | −0.26 [−0.71, 0.17] |
| R16 end-to-end | 1.768 | 17.40 [17.03, 17.77] | 1.45 | **+0.50 [0.08, 0.89]** |
| R16C / R19 / R19w end-to-end | 1.658 / 1.601 / 1.543 | 16.49 / 16.33 / 16.43 | 1.33–1.34 | all below blank (CIs exclude 0) |

**chrF against BLEU on these 12 stored forms (66 pairs):**
- BLEU resolves 38 orderings. chrF reproduces 25, contradicts none and leaves 13 unresolved.
- chrF also resolves 10 pairs that BLEU does not.
- Its median normalised half-width is 0.119, against BLEU's 0.148.
- So chrF would **fail** the S0 rule (unresolved orderings), despite tighter intervals. That is the context for ΔNLL and retrieval.

Two cautions from the same read:
- **Composition's BLEU edge over blank is invisible in chrF** (gap fraction 0.02 against BLEU's 0.20), and composition outputs are 1.65–1.70× the reference length (the translator loops).
- **R16 is the only stored form whose chrF is above blank.** Its length ratio is 1.45 against the 1.32–1.34 of its peers, and chrF2 is recall-weighted (β = 2), so length alone can buy chrF. Any chrF claim needs the length ratio beside it.

## S1 result (run locally on CPU, all 1,000 clips; stored at `s0s1/s1-text-kl-v1.json`, SHA-256 bed8400c…)

984 of the 1,000 clips were read; 16 are longer than the priors' 20 s training limit. Loci follow generation's rule (planned from the text); the empty branch is classifier-free guidance's unconditional input. All values are nats per stage-1 code (4 per 24 Hz step).

| Prior | KL(text ‖ empty) | Per-clip mean KL [95% CI] | Summed KL per clip | True-code gain | Entropy p(·\|empty) |
|---|---|---|---|---|---|
| W14b (byte, round 14b mix, 30k) | **0.0251** | [0.0298, 0.0377] | 12.8 nats | −0.0053 | 2.83 |
| byte24k (r3d-d2g, round3c mix, 24k) | 0.0077 | [0.0085, 0.0100] | 3.9 nats | +0.0080 | 2.34 |
| flan-T5 (round 4, 23k) | 0.0130 | [0.0146, 0.0173] | 6.6 nats | +0.0148 | 2.33 |
| MiniLM (round 4, 24k) | 0.0089 | [0.0101, 0.0122] | 4.5 nats | +0.0086 | 2.36 |

Paired per-clip ratios of mean KL:
- flan-T5 / byte24k = **1.72 [1.66, 1.79]**;
- MiniLM / byte24k = 1.21 [1.17, 1.26];
- W14b / byte24k = 3.62 [3.23, 4.12].

By part, the body carries the most text influence in every prior (byte24k 0.0145, flan-T5 0.0236, W14b 0.0330). The hands and face carry less (0.005–0.027).

**Reading (for the owner's S1 decision; no gate was declared numerically):**
- **In absolute terms the decoder-bottleneck reading holds.** Every prior's text moves its stage-1 code distribution by at most 0.025 nats per code, about 1% of the 2.3–2.8 nats of entropy it predicts with. The pretrained encoders stay in the byte prior's regime: flan-T5 is 0.013 and MiniLM 0.009, both *below* the 0.02–0.07 band quoted earlier. That band was a different measure, the true-code CE gain, on round-6 clips.
- **But the encoder is not irrelevant.** At matched step and mix, flan-T5 roughly doubles the text influence over bytes (×1.72, resolved). So "the decoder ignores every encoder" is too strong. A better encoder raises text use from a very low base, and the decoder still caps it at about 1% of the entropy.
- **More training on more data (W14b) triples the KL, yet its true-code gain on val is slightly negative.** W14b's text moves the distribution, but on held-out clips not toward the true codes. That is consistent with W14b's end-to-end output sitting at blank.
- **Not yet run:** the 64-way word retrieval (part of the pod job). The stage-1-only round trip (`stages1`, the second S1 condition, keeping ≥ 0.8 of d2's gap share) comes from the S0 pod run.

## Pod job

```
# bundle the branch (make_bundle.sh bundles ~/src/Sign-Translator HEAD, so merge probes-s0s1 first, or bundle this worktree the same way)
bash ~/.sign-translator-ops/make_bundle.sh
bash ~/.sign-translator-ops/run_job_pod.sh s0s1 'bash runpod/fleet/s0s1_probes.sh'
```

- **Smoke first, keys timestamped under `s0s1/smoke/`.** On 20 clips, with every form, the stored reports and the W14b regeneration, it runs S1 and an 8-word retrieval. It then checks that real reproduces the stored hypotheses, that v1 is face-blind, and that all four priors loaded. The job stops (ALERT, `S0S1-DONE 2`) if any check fails.
- **Full phases** (`PHASES`, default `s0 regen s1 words`), each skipped if its key exists (`s1-text-kl-v1.json` already exists from the local run, so the pod skips it):
  - `s0-ceilings-v1.json`
  - `s0-regenerated-v1.json` (merged orderings)
  - `s1-text-kl-v1.json`
  - `s1-word-retrieval-{w14b,byte24k,flant5,minilm}.json` (64 ASLLVD words, round6-mixed-v1, as round 7)
- **Splitting across pods:** `PHASES="regen"` on a second pod halves the wall time. The regen file's orderings then cover only real, blank and the regenerated arms, because the merge needs `s0-ceilings-v1.json` to exist first.

**Cost estimate (one L40S at about $0.86/h):**

| Step | Estimate |
|---|---|
| Smoke | ~0.3 h |
| S0 forms | ~1.5 h: 12 forms × 1,000 clips (translation, 100k teacher-forced decodes and DTW per form), with round trips on the CPU |
| Regeneration | ~3 h: 5 × 1,000-clip generations; the R16 pilot job took about 50 min |
| S1 and word retrieval | ~0.5 h |
| **Total** | **about 5–5.5 GPU-h ≈ $5** (well under the plan's "under $50") |

## v2 hooks

- `--translator-run <v2 run>` reads through `load_translator_run` (SHA-checked). The `states` kind is handled by `translator_items`, and the face forms become meaningful there.
- G5's "regeneration must first reproduce v1's stored hypotheses" is covered by `reproduction.identical`.

## Tests

- **Metric tests:**
  - ΔNLL sign convention;
  - token NLL equals cross entropy, ignoring padding;
  - retrieval with a planted match (rank 1), mid-rank ties and random scores at chance;
  - chrF hand-computed values;
  - fast BLEU equal to `bleu_ci` and `paired_bleu_difference`;
  - batched DTW equal to the plain recursion, and a stretched copy aligning at 0;
  - ρ_faith 0 for identical pairs, about 1 for random pairs, low for noisy copies;
  - KL zero for equal logits and matching its definition.
- **Script tests:**
  - stage-truncated round trip with all stages equals the fidelity round trip;
  - face forms, with v1 face-blindness;
  - time-warp durations;
  - the low-pass passes 1 Hz, removes 10 Hz and keeps validity and manifolds;
  - `candidate_nll` column 0 equals the model's loss and is batch-independent;
  - a trained tiny translator retrieves its planted references with ΔNLL > 0 over blank;
  - orderings;
  - the stored-only run end to end, write-once, with skipped mismatched clips;
  - regeneration blocks, seeds and settings, and refusal of plan pilots;
  - the stage-1 logits sum to `rvq_sequence_log_likelihood`;
  - **KL is exactly 0 when the temporal decoder's cross-attention to the text is cut**, and positive otherwise.
- Run with `OMP_NUM_THREADS=1 nice python -W error -m pytest -q tests/test_s0_metrics.py tests/test_s0s1_scripts.py`.
