# Round 20 tooling: development report (2026-10-30)

Branch `round20` (from main `78514c9`). Binding spec: "Round 20: composition v2, the method to M1" (declared
2026-10-30, 19:09 UTC) at the end of `docs/NEXT_PHASE_PLAN.md`. **Nothing was trained.** Nothing ran on a pod. R2 was
read, and written only for the two sentence sets (write-once keys under `round20/`). The local smoke wrote local
files only.

## What was built

| Piece | File |
|---|---|
| D20 / C20 builder (write-once, SHAs printed) | `scripts/build_round20_sets.py` |
| Shared round-20 pieces: sets, Sem-Lex-only entries, coverage, ASLLRP continuous segments, cached material, paired statistics, likelihood summary | `scripts/round20_lib.py` |
| Continuous-pace durations, velocity-bounded bridges, the continuous plan, the Butterworth low-pass, the predicted-duration model | `signtranslator/generation/composition_v2.py` |
| Reference log-likelihood under the back-translator (teacher-forced, per token) | `scripts/score_back_translation.py` (`reference_log_likelihood`) |
| Sem-Lex activity core (inert unless `fractions["activity"]` is given) | `scripts/round15_stage2.py` (`sign_core`) |
| Round-20 options of the composition evaluation | `scripts/round15_compose_eval.py` |
| Best-of-N word cache and word-level recogniser reads (A4, A5) | `scripts/round20_word_cache.py` |
| Paired D20 comparisons: forms (BLEU and likelihood) and word caches | `scripts/round20_compare.py` |
| Phase A pod job | `runpod/fleet/round20_phase_a.sh` |
| Tests (synthetic, offline) | `tests/test_round20_units.py` (19), `tests/test_round20_eval.py` (5) |

**Defaults are unchanged.** Every round-20 option is off by default. A run without one writes no round-20 key. Checked
byte for byte: the HEAD tree (`git archive 78514c9`) and this branch ran the same default runs on the synthetic
round-15 store:
- unsharded, and two shards with `--bridge minimum-jerk --tighten-cores`;
- both merges, one with a rule and an end-to-end comparator.

All five JSON outputs are identical. One shard differs only in its wall-clock `seconds` field.

The line references in the task were verified against the code:
- `round15_compose_eval.py`: vocabulary 278–280, selection 315, translation 443–459, spotting 467–478, guidance and temperature 714–716, reference duration 325;
- `compositional.py`: 152–176, 78–84, 123–149, 490–497, 602–616, 650–678;
- `round15_stage2.py`: `sign_core` 350–361, `word_seeds` 133;
- `round15_stage2_fit.py`: 147–165;
- `compositional_grammar.fitted_plan` at 662.

## 1. Sentence sets (stored)

```
cd ~/src/Sign-Translator && set -a && source ./.env.r2.local && set +a
cd ~/src/st-round20 && OMP_NUM_THREADS=1 ~/.venvs/sign-translator/bin/python scripts/build_round20_sets.py
```

| Set | Key | SHA-256 | Sentences |
|---|---|---|---|
| D20 | `acquisition/2026-10-17-v1/youtube-asl-v1/round20/d20.json` | `d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe` | 500 |
| C20 | `acquisition/2026-10-17-v1/youtube-asl-v1/round20/c20.json` | `24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956` | 1,000 |

**Pool.** The pool is `validation_sentences`' own: How2Sign val, groups c01/c03/c05, with a text, 3,096 clips (round10 mix `ccfd10b8…`). From it:
- 1,011 clips are in the 1,000 or the 300 (11 of the 300 lie outside the 1,000);
- 18 more share a normalised text with the 1,000;
- that leaves 2,067.

**Draws.**
- **D20** = `random.Random(20).sample(remainder, 500)`.
- **C20** comes from the same generator: `sample(remainder minus D20's clips and texts, 1000)`. Removing D20's texts dropped 5 more clips, leaving 1,562 candidates.

No D20 text is in the 300. Only manifest metadata was read: no c00 record and no state.

Each set file records:
- the manifest key and SHA;
- the rule and the counts;
- per clip: sign, text, signer group, and state key/SHA.

`round15_compose_eval.py --sentences-key` refuses a set drawn from another manifest, or a record whose text or state differs.

**D20 coverage** (stage-1 selection, token shares):

| Vocabulary | Entries | D20 entries | exact | fallback | stop | unmatched | entries / sentence | empty |
|---|---|---|---|---|---|---|---|---|
| round10 (4 corpora) | 2,378 | 705 | 0.425 | 0.049 | 0.179 | **0.347** | 9.09 | 6 |
| round14b (+ semlex-text-v2) | 6,150 | 1,086 | 0.615 | 0.041 | 0.171 | **0.174** | 12.16 | 0 |

- 388 of round14b's 1,086 D20 entries are Sem-Lex-only. For round10's D20 entries there are 0, so the activity core matters only for round14b.
- Top unmatched tokens, round10: you 301, it 174, your 146, this 99, so 96, i 89.
- Top unmatched tokens, round14b: this 97, so 84, i 72, just 62, as 49, it's 47.

## 2. Composition options (`round15_compose_eval.py run`)

| Option | Effect |
|---|---|
| `--sentences-key/--sentences-sha256` | A D20 or C20 list instead of `validation_sentences(mix, --clips)`. `--clips N` keeps its first N. |
| `--vocabulary-mix-key/--vocabulary-mix-sha256` | Vocabulary, oracle pools and sign fractions from that mix and its `isolated_corpora`. The stage-2 model stays the eval mix's (round10's corpora, so `--expect-model-sha256 42389ca4…` still holds). |
| `--semlex-core activity [--activity-margin 0]` | Sign-fitted cores of Sem-Lex-only entries (every training record Sem-Lex) by `compositional.core_window`, the existing activity helper, not ASLLVD's pooled median fraction. Applies to generated and oracle words. Reason `activity`. |
| `--oracle-source continuous` | A2: see §3. |
| `--retime continuous --tempo f` | A3: see §4. |
| `--lowpass-hz h` | Butterworth, 4th order, zero-phase (`sosfiltfilt`) on every composed form, controls included (not real/blank). Applied per channel per valid run; 6D rotations re-orthonormalised, unit vectors renormalised, bounded scalars clipped. |
| `--word-cache-key/--word-cache-sha256/--word-cache-n N` | A4: core_crossfade from the cache (no generation; replaces `--checkpoint-key`). Refuses another vocabulary, a missing entry or a missing N. |
| `--likelihood` | Per form and clip: the back-translator's teacher-forced log p(reference \| motion) per token, and the same for the deranged reference. |
| `--predicted-duration` | Report-only `<variant>@predicted`: the main form rendered at a text-predicted duration (§4). |

**Reports.** The shard records:
- `settings.round20`;
- `round20.coverage` (selection statistics plus the entries-per-sentence distribution and top unmatched and stop tokens);
- `round20.vocabulary` and `round20.segments`;
- per clip `plans[v].round20`: each word's core reason and source, the plan fit, cores and gaps, velocity-raised joins and duration sources.

The merge adds:
- `round20.per_variant`: reasons, sources, `continuous_share`, fits and `velocity_raised_share`;
- `likelihood`: per form its mean, `minus_blank` (the selection metric, paired per sentence), `minus_mismatched` (its deranged references), `minus_<pair>` for every BLEU pair, and `minus_paired_<v>` with `--paired-with`;
- `@predicted` against `reference_duration`.

**Spotting guard.** With a cache, `round20.spotting_note` reads "CIRCULAR" when the spotting recogniser is the cache's scorer, and "independent" otherwise. The pod job reads cache arms with v2.

## 3. Continuous-form oracle (A2)

Segments are ASLLRP gloss-aligned signs: `derived/signbank-v1/utterances.json` (frames, occurrence), joined to the gloss manifest `derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json` (`f8c40aea…`, the one `validate_sign_recognizer` G7 uses) by `<collection>:<video stem>`.
- **Partitions:** train and val only. Test (Ben) is never read.
- **Mapping:** each sign maps to an entry by the stage-2 fit's own per-sign rule (`map_sign` tiers with index, exactly one entry), against the vocabulary in use.
- **The clip:** the utterance is loaded, conformed and taken to 24 fps like `word_clip`. The cut runs 0.5 s before to 0.5 s after the annotated span. The core is the span ± 0.1 s, the sign-fitted margin.
- **The draw:** `choose_record` per (clip, word, block), the oracle's own draw.
- **Frame counts:** a state within 3% of its annotation's frame count is mapped proportionally (`span_scale`; the RIT sentence states run about 1% long).
- **Fallback:** words without a segment, or whose utterance frame count differs by more than 3%, take exactly the isolated oracle's piece, with source `isolated` / `isolated_span_mismatch`.
- **Reported:** the share of words served by continuous forms (`continuous_share`).

## 4. Retiming (A3), predicted duration, low-pass

**Option name.** `--timing` already exists: asllrp / stage1 / sign-fitted pick the core rule and plan. The spec's `--timing {reference, continuous}` is therefore **`--retime {reference, continuous}`**. Cores are still cut by `--timing` (sign-fitted on the pod).

**Durations.** Sign k gets its ASLLRP train sentence median (the model's `sentence_seconds`, entries with ≥ 3 signs). Otherwise it gets its isolated annotated seconds (`isolated_sign_seconds`) × `--continuous-ratio` 0.385. The spec's 0.385 is used; the F12 model's own fitted ratio is 0.3926. Both are multiplied by `--tempo`.

**Gaps.** The base gap is the model's ASLLRP median gap (0.1668 s) × tempo. Each join is raised so that a linear bridge stays no faster than its neighbours:
- n ≥ d·24 / v − 1, per side;
- d is the wrist distance in the chest frame (`wrist_from_shoulder`);
- v is the mean of the resampled cores' last and first 3-step wrist speeds, floored at `--bridge-min-speed` 0.3 m/s;
- capped at `--bridge-max-seconds` 0.6.

For the floor: D20 How2Sign wrists in the same frame (30 clips) have a median of 0.37 m/s on the dominant side and 0.24 m/s on the other.

**Fit.** Pad to the reference (lead share as the model's) or compress when longer; `none` keeps the predicted length. Fits and velocity-raised shares are reported.

**Velocity rule.** Our reading of arXiv 2405.07663: the paper names velocity-bounded linear bridges but gives no formula. The bound used is stated in `composition_v2`.

**Predicted duration** (report-only, cheap): least squares of How2Sign *train* seconds on normalised token count, floor 0.5 s. `@predicted` renders the main form's timing with that duration instead of the reference's.

## 5. Word cache (A4) and word-level reads (A5)

`scripts/round20_word_cache.py` takes the entries of the set's stage-1 selection on the chosen vocabulary.
- **Samples:** `--samples 16` per entry, exactly as `generate_word` samples (same `generate_rvq_motion` call, temperatures, resampling and post). Seeds are `rerank_seeds(entry_seed(e), N)`, fixed per entry.
- **Score:** v3d's softmax probability of `entry_label(e)` (upper-cased entry; multi-word joined). Without a label, the prior's `word_pmi`.
- **Choice:** best-of-N for N ∈ {4, 8, 16} from one run, because the seeds are prefixes. Chosen states are stored write-once under the prefix with every score, rank and seed.
- **Merge:** writes `index.json` with word-level summaries: sample 0, all samples, and each best-of-N.

`round20_compare.py words` pairs two priors' indexes entry by entry (A5). `round20_compare.py forms --a KEY:FORM --b KEY:FORM` gives:
- the paired BLEU (the pilot's bootstrap);
- the paired likelihood-minus-blank;
- `both_improve`, the adoption rule.

## 6. Pod job and commands

`runpod/fleet/round20_phase_a.sh` (bash, syntax-checked in the tests).
- **Inputs:** every input is SHA-pinned (round10, round14b, D20, gloss manifest), plus the model SHA. Thread counts come from the cgroup quota or `sched_getaffinity`.
- **Write-once:** each variant is its own invocation with write-once keys under `round20/phase-a/`.
- **Smoke:** `SMOKE=1` runs 5 D20 sentences into `phase-a/smoke/<UTC>/`, with a best-of-{1,4} cache over those 5 sentences' entries.

| Step | Command (on the pod) | Pod | Estimate |
|---|---|---|---|
| A1 | `R20_PHASE=a1 ARM=r10` / `ARM=r14b-activity` (declared) / `ARM=r14b-fraction` (attribution) | CPU ×3 | ≈ 1–1.5 h each |
| A2 | `R20_PHASE=a2 VOCAB=<A1 winner>` | CPU | ≈ 1.5 h |
| A3 | `R20_PHASE=a3 VOCAB=.. ARM=tempo-0.8 / tempo-1.0 (+ @predicted) / tempo-1.25 / tempo-1.5` | CPU ×4 | ≈ 1–1.5 h each |
| A6 (real words) | `R20_PHASE=a6o VOCAB=..` | CPU | ≈ 1–1.5 h |
| generated baseline | `R20_PHASE=a0 VOCAB=..` (W14b, fresh, 4 word workers) | GPU | r14b ≈ 12k word samples, ≈ 3–5 h |
| A6 (generated) | `R20_PHASE=a6 VOCAB=..` (same seeds, 6 Hz) | GPU | as a0 |
| A4 cache | `R20_PHASE=cache ARM=w14b VOCAB=..` (4 shard processes on one GPU) | GPU | 1,086 × 16 ≈ 17k samples, ≈ 2–4 h |
| A5 cache | `R20_PHASE=cache ARM=r16 VOCAB=..` | GPU | as above |
| A4 | `R20_PHASE=a4 ARM=n4 / n8 / n16 VOCAB=..` (v2 spotting) | CPU or GPU | ≈ 1 h each, no generation |
| A5 | `R20_PHASE=a5 VOCAB=..` | CPU | minutes |
| compare | `python scripts/round20_compare.py forms --a $OUT/<x>.json:<form> --b $OUT/<y>.json:<form> --output-key ..` | CPU | seconds |

**Basis of the estimates:**
- The 1,000-clip CPU oracle (`round15/eval1000/oracle-stage2-shard0.json`) took 31 min to compose and 87 min to translate 5 forms; D20 is half of that, plus the likelihood passes.
- W14b generation measured about 1.24 s per word with 4 workers (`w14b-step30000-shard0.json`: 5,469 s for 256 clips, about 4,400 words).

**Totals:**
- About 12–16 CPU-hours: a few dollars.
- About 12–18 GPU-hours: of the order of $10–25 at ≤ $2/h. a0 and a6 can be split with `--shards` across pods (at most 5).

**Smoke first:** `SMOKE=1 R20_PHASE=<phase> ...`. For a4, give `CACHE_KEY=<the smoke cache's index.json>`.

## 7. Local smoke (Mac, CPU, 1 thread, first 5 D20 sentences; local files)

`$SCRATCH/smoke/run.sh`: each arm is `round15_compose_eval.py run` (D20 key/SHA, `--clips 5`, stage-1 selection,
sign-fitted timing, margin 0.1, `--expect-model-sha256 42389ca4…`, `--likelihood`, `--device cpu`, oracle_stage2) then
`merge --bootstrap 200`, local `--output-file`s. About 2–2.5 min per arm (translator, recogniser, model fit, clips
from R2). Every arm ran end to end. **Five sentences decide nothing** (all composed forms read BLEU 0 on 5 clips; real
6.8); the numbers prove the plumbing:

| Arm | words (sources) | core reasons | likelihood − blank (oracle / shuffled / mismatched words) | wrist / real |
|---|---|---|---|---|
| A1 r10 | 81 isolated | entry 38, annotated 37, all 6 | −0.013 / −0.071 / −0.157 | 2.08 |
| A1 r14b-activity | 111 isolated | entry 60, annotated 7, all 6, activity 8, **activity_arm_untracked 30** | −0.121 / −0.156 / −0.083 | 2.05 |
| A2 continuous (r14b) | 111: **72 continuous (0.65)**, 39 isolated | continuous 72, … | −0.060 / −0.099 / −0.153 | 2.11 |
| A3 tempo 1.25 (r14b) | 111 isolated | as A1 r14b | −0.134 / −0.179 / −0.085; @predicted −0.127 | 2.20 |
| A6 6 Hz (r14b) | 111 isolated | as A1 r14b | −0.126 / −0.169 / −0.086 | **1.95** |

Real reads +0.071 above blank on the likelihood. A3: fits padded 3 / compressed 2; 8.5% of joins velocity-raised.
The low-pass lowers wrist speed (2.05 → 1.95) as expected. A2 found 12,524 segments of 951 entries in 2,065
train/val utterances (3,076 signs unmapped).

**Two findings from the smoke, acted on or flagged:**
- **RIT sentence states run about 1% long** against their annotation (150 sampled utterances: every asllrp-continuous
  one exact, 15 of 19 rit-sentences 1.01–1.02×). The first A2 smoke lost 22 of 72 continuous words to
  `isolated_span_mismatch`; segments now map indices proportionally within 3% (`span_scale` recorded), larger
  differences still fall back. After the fix: 72 of 111 words continuous.
- **The activity core mostly cannot cut Sem-Lex clips:** 30 of the 38 Sem-Lex-only words in A1 r14b-activity came out
  `activity_arm_untracked` (whole trimmed clip as core). Sem-Lex states often have the right forearm untracked (elbow
  out of frame; 12 sampled train clips: right-forearm tracked share 0.0 in 8), and where the rule does cut it picks the
  side with the larger wrist-height spread, which can be the left arm. `core_window` was not changed (it is the existing
  helper; its whole-clip fallback is the manifest's trimmed window, about 1 s). This is why A1 carries the attribution
  arm r14b-fraction; the reviewer should decide whether a right-side-only rule is wanted before A1 runs.

## Tests

- `tests/test_round20_units.py` (19) and `tests/test_round20_eval.py` (5) cover:
  - set exclusions, determinism and disjointness;
  - the builder on a 3,096-clip synthetic pool;
  - set loading refusals;
  - Sem-Lex-only entries and coverage;
  - the activity core;
  - continuous seconds;
  - bridge frames against the formula;
  - plan fit, compression, `none` and rendering;
  - the low-pass (1 Hz kept, 10 Hz removed, masks, orthonormality);
  - the duration model;
  - ASLLRP segments (test partition never read) and segment cuts;
  - cache helpers, cached material and the paired statistics;
  - the likelihood equal to the T5's own teacher-forced loss on a tiny model, batched equal to unbatched;
  - end to end on the round-15 synthetic store: a default run with no round-20 rows; sentence set + activity + retime + tempo + low-pass + likelihood + predicted, sharded; the continuous oracle with a synthetic gloss manifest; the cache (shards, merge, write-once, best-of-1 equal to `generate_word`, PMI fallback) and composition from it with both spotting notes; `forms` / `words` comparisons; the pod script's syntax.
- Related round-15 suites pass unchanged: `test_round15_compose_eval.py` and `test_round15_stage2.py`.
- Full suite (warning-strict, `-W error`, 1 thread): **4,147 passed** (14 min 51 s).

## Not implemented exactly as specified, or decided here

1. **`--retime`, not `--timing`.** The latter already exists (§4).
2. **C20 also excludes D20's normalised texts.** A sentence cannot be confirmed on its own development text. This removed 5 candidates.
3. **Best-of-N is nested.** One run of 16 samples, with choices 4 / 8 / 16 over prefix seeds. This is equivalent to separate runs and a quarter of the compute.
4. **The bridge rule is our reading of 2405.07663.** Floor 0.3 m/s, cap 0.6 s.
5. **The activity-core margin is 0.** The activity window already spans about 1.8× the annotated sign. It is an option.
6. **A2 uses ASLLRP train and val utterances (never test).** It is a headroom read and nothing is fitted on them. Signer handedness is not normalised: ASLLRP signers are taken as recorded.
7. **The stage-2 model stays round10's (42389ca4) under the round14b vocabulary.** New entries are timed by the model's trimmed/annotated ratio. The stage-1 selection uses none of its lexicon or structure.
8. **A1 has an extra report-only arm** (r14b-fraction) to attribute the activity core.
9. **A6 runs on both the real-word oracle and the generated W14b composition.**
10. **The pod job is prepared, not launched.** It needs the independent review first.
11. **A2 maps frame counts within 3% proportionally** (RIT sentence states run about 1% long); found in the smoke.
12. **The word cache was not smoke-run on real checkpoints locally** (GPU-scale; no R2 writes beyond the two sets were
    wanted). It is tested end to end on the synthetic store, including best-of-1 equal to `generate_word`. The pod's
    `SMOKE=1 R20_PHASE=cache` is its first real run.
