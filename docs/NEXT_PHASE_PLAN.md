# Next phase: from signing-like motion to deployable ASL (plan log, 2026-10-21)

**Status: planning only.** Nothing here starts until the v2b priors on pod 4 finish and
pass the quality review against v1 and v2 (`scripts/compare_motion_runs.py`). This log
records the user's questions of 2026-10-21, the facts checked, and the plan for each item.

## Where the model stands (inputs to this plan)

| Tokenizer | Joints | Arm/hand points | Fingertips | Wrists |
|---|---|---|---|---|
| v1 (one VQ codebook) | 19.1° | 10.9 cm | 11.7 cm | 36–41° |
| v2 (RVQ + skeleton loss) | 9.6° | 2.2 cm | 2.5 cm | 22° |
| v2b (full frame rate, base blocks) | 8.3° | 1.7 cm | 1.95 cm | 19° |
| data noise floor | 2.0° | 0.9 cm | 1.0 cm | — |

- **Text → motion priors:** v1 was under-articulated (hand speed 0.2–0.5× real). The v2b
  priors are training now.
- **Data:** a 12-hour How2Sign subset, 3,968 training clips.
- **Grammar:** non-manual markers replicate on real data (wh, yes/no, conditional,
  negation). Role shift, pointing → deixis and gaze-follows-point are not validated
  (docs/GRAMMAR_MOTION.md).

## Q1. The sign-to-text back-translation model: train it, or adopt one?

**Recommendation:** train our own, on our own motion representation, with a pretrained
language decoder. Use a published How2Sign translator only as a second opinion.

Reasons:
- **The input must match what the generator emits.** Published How2Sign translators read
  RGB video or 2D keypoints (MediaPipe/OpenPose layouts). Our generator emits canonical 3D
  rotations, face blendshapes and gaze. A pose → text model trained on the same canonical
  states scores generated motion directly, with no rendering step whose own errors would
  be counted against the generator.
- **The cost is moderate.** It is an encoder over the motion channels (it can reuse the
  v2b tokenizer's encoder or codes) plus a pretrained text decoder (T5-class), trained on
  How2Sign pairs.
- **Published How2Sign BLEU is low** (single digits to mid-teens). The metric is therefore
  relative: generated vs real motion through the same translator, never an absolute
  quality claim.
- **Guard against gaming.** The translator is trained on the training partition only and
  frozen before it scores anything. It is never used as a training loss for the
  generator unless a second, independent translator does the scoring.
- **Second opinion.** Project generated motion to 2D keypoints (forward kinematics plus
  camera) and run an openly licensed published translator. Agreement between the two
  raises confidence. Licence and weight availability must be checked per model before
  use.

Is it necessary? For any claim that the output means the English, yes. It is the
standard sign-language-production metric. It is not sufficient: Deaf ASL users'
evaluation (comprehension and naturalness) gates any release.

## Q2–Q4. The BU ASLLRP links: what each is and whether it helps

Facts checked on 2026-10-21. Pages: SignStream 3 download, the ASLLRP index (Sign Bank,
DAI 2) and the DAI terms. Details are in
`research_notes/ASL grammar and spatial reference/datasets_and_licences.md`.

| Link | What it is | Helps? |
|---|---|---|
| SignStream 3 download (`SignStream/3/dnldSS3.html`) | The annotation **software** (SignStream 3.5.1, May 2025), its README and manuals, and an optional "Global Sign Bank" for the app. **No corpus data.** | Only as a tool: its XML documentation (2025) defines the format of the corpus exports, so it is needed for the parser |
| ASLLRP Sign Bank (`#signbank`) | An isolated-sign lexicon: ASLLVD (about 10,000 examples of about 3,000 signs, 1–6 signers each) plus examples segmented from the continuous corpus, with morpho-phonological features (handshapes, etc.). Downloads (citation forms; segmented signs, CSV per Report 24) sit **behind a DAI 2 login**. | **Yes:** a gloss lexicon with video for every sign. It gives lexical supervision for an English → gloss → motion planner, vocabulary for the back-translator, and handshape labels to check hand accuracy |
| DAI 2 (`#dai2`) | The continuous ASLLRP SignStream 3 corpus: 2,048 utterances and 16,659 sign tokens from native signers (2022 report; the site says much more has been added since). Front, side and face close-up views. Frame-level start/end annotation of non-manual markers (brows, gaze, head, eye aperture, mouth) and grammar (wh, yes/no, negation, topic, conditional, relative clause, **role shift**), **locus indices on agreement verbs and pronouns** (`i:GIVE:j`), and English translations. | **Yes, the most valuable item.** It is exactly the gloss and grammar ground truth How2Sign lacks |

**Licence (DAI terms):**
- research and education use only;
- no redistribution without permission;
- commercial use only with explicit permission;
- no patents or copyrights based on the material;
- citation required.

**What this means for us:**
- Storing a copy in our **private** R2 bucket for research is not redistribution and is
  compatible with the terms. The prefix stays private and is never published.
- **A deployed commercial product trained on ASLLRP data needs written permission from
  the ASLLRP (Carol Neidle, Boston University)** before release. Until then, ASLLRP is used
  for training and evaluation in research builds, and the provenance manifest records it,
  so any release build can exclude it or show the permission.
- **Access needs the user:**
  - The Sign Bank downloads need a DAI 2 login, which the user must obtain. Accounts are
    not created on the user's behalf.
  - Downloads go through the site's own cart and download options, at a polite rate,
    without circumventing any access control.

**How each piece plugs in:**
1. **Grammar ground truth.** Validate the motion-grammar detectors on real labels:
   brow/headshake spans against the annotated spans, role shift (not validated on
   How2Sign) against `rs:` spans, gaze-to-locus against gaze-agreement tiers. Report
   per-marker precision and recall instead of English-proxy AUCs.
2. **More 3D data.** The front and side views match our two-view pipeline (Sapiens →
   calibration → triangulation). This produces canonical states with gloss, grammar and
   translation labels, and adds native-signer data to How2Sign.
3. **English → ASL structure.** The (English translation, gloss string with indices, NMM
   spans) triples train the gloss/semantic planner of item 2 below.
4. **Lexicon.** Sign Bank citation forms train isolated-sign motion and give the
   back-translator a gloss vocabulary.

## Additional sources reviewed 2026-10-21 (user-supplied, no login needed)

Read-only review: no download yet. Facts are from the pages and the licence page.

| Source | What it is | Licence | Decision |
|---|---|---|---|
| ASL Citizen: github.com/microsoft/ASL-citizen-code, arXiv 2304.05934, Microsoft Download Center id 105253 (`ASL_Citizen.zip` 42.8 GB, `Demographics.zip` 5 KB, v1.0 of 2024-10-15) | 83,399 webcam videos of 2,731 isolated ASL signs by 52 signers in everyday settings. Collected with consent under IRB approval. Baselines: I3D and pose ST-GCN (code MIT, checkpoints `checkpoints_v1`). Paper: 63% top-1 and 91% recall@10 on unseen signers. | Microsoft Research Licence Terms: "solely for non-commercial, non-revenue generating, research purposes"; "may not distribute the data or your modifications"; use must stay "consistent with the consent". Commercial use: contact ASL_Citizen@microsoft.com. | **Adopt (research builds)** |
| Sign Language MNIST (Kaggle, datamunge) | 28×28 grayscale static images of 24 fingerspelled letters (J and Z excluded: they need motion). 27,455 train + 7,172 test. | CC0, per secondary sources; the Kaggle page itself was not readable | **Reject**: no motion, no 3D, letters only |
| sign-language-translator/sign-language-datasets | **Pakistani** Sign Language: 776 dictionary signs (27 min), a few sentences, MP4 and MediaPipe world/image landmarks | CC BY 4.0 (commercial use allowed with attribution) | **Reject for training**: a different language would contaminate ASL vocabulary and grammar |
| Mendeley "American Sign Language Alphabet Dataset" (DOI 10.17632/jdyksv2jhh.1, 2025-12-04) | Static images of ASL letters (Train/Test, one folder per letter) | CC BY 4.0 | **Reject for training**. Optional later: a probe that the hand pose separates letter handshapes |

**ASL Citizen plan (item 1c; starts only after the current generator review):**
1. **Acquire.** Download from the Download Center into a private R2 prefix (`acquisition/asl-citizen-v1/`): checksums, the licence text archived beside it, never public.
   - **Direct URLs** (read from the page's download data on 2026-10-21):
     - `https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip`: 45,924,134,223 bytes, ETag `0x66C703EABF2B49391CB7B3749773EFF2890BC0510CCD9E0F006D2A34BC975425`;
     - `.../Demographics.zip`: 5,077 bytes, ETag `0xE9872EA784F95815923FDD70903E06A63B317FEBEAAEF620797C3B991D67910B`.

     Both are dated 2025-02-17. No checksum is published, so check whether the 64-hex ETag equals the streamed SHA-256.
   - **Tool:** `scripts/acquisition/acquire_resumable.py` with `--expected-size`, on a pod (HTTP → memory parts → R2 multipart; nothing staged on disk; re-read and re-hashed from R2).
2. **Extract.** Monocular path: Sapiens keypoints → swing IK → canonical states, one isolated sign per clip. Apply the existing quality gates. Monocular 3D is coarser than two-view: arm direction differs by 20–35° on How2Sign. Record the per-clip estimate quality and keep clips that fail the gates out.
3. **Map glosses.** Map ASL Citizen glosses to the ASLLRP Sign Bank and ASL-LEX entries, keeping the mapping table and the unmatched list.
4. **Use for:**
   - (a) an isolated-sign recogniser on our motion representation, for sign-level evaluation of generated motion beside the sentence back-translator;
   - (b) a citation-form lexicon for the English → gloss → motion planner;
   - (c) hand and handshape realism.

   It adds no English-sentence → motion pairs, so How2Sign and DAI 2 remain the training signal for the generator.
5. **Release gate.** Non-commercial terms, like ASLLRP. A commercial build excludes it or holds Microsoft's permission. Signer identity: the raw videos stay private, and the states are used only within the consent terms.

**DAI 2 login:** awaiting the user's confirmation (2026-10-21). The DAI 2 / Sign Bank plan above applies once it is ready.

## Implementation status (2026-10-22)

**Item 2 — English -> ASL structure (code complete; training under way / waits for the exit gate)**
- `signtranslator/planning/asllrp_gloss.py`: ASLLRP surface tokens -> base gloss, agreement/index locus slots
  (renumbered by first placement, the generator's slot order), per-sign multi-hot of the seven grammatical markers;
  split quoted descriptions rejoined (256 pseudo-glosses removed); gloss vocabulary with exact byte fallback.
- `signtranslator/planning/english_gloss.py` + `scripts/train_gloss_planner.py`: English words -> gloss tokens with
  locus and multi-label marker heads; evaluation by gloss WER/BLEU, aligned marker F1 per tier, locus accuracy and
  utterance-level marker presence. Run v1 (old parse) best: gloss BLEU 27.0, WER 0.72, marker F1 0.67 (wh 0.75);
  it memorises the 1,652 training pairs within ~4k steps (data-limited).
- v2 (fixed parse; `gloss-planner-asllrp-v2`, best step 8,000): gloss BLEU 27.7, WER 0.70, marker F1 0.70.
- Locus accuracy is 0.233. On the same predicted glosses, the per-gloss majority locus scores 0.240 and "no locus" 0.
  The locus head has learned only each gloss's usual slot, not the discourse context. Its slots already follow
  canonical order (1 plan in 412 does not), so renumbering gains nothing.
- Referent placement is under-determined from English at this data size. Spatial agreement is a target for the human
  study and the reviewer-driven fine-tuning, not a solved part of the plan.
- `signtranslator/planning/plan_text.py`: the plan as one deterministic text line (`[topic MOTHER ] IX-loc>1
  1>GIVE>2 [neg NOT LIKE ]`), lossless on all 2,403 pairs, so the byte-conditioned generators and their
  classifier-free guidance learn plan -> motion unchanged.
- `scripts/build_asllrp_plan_manifest.py`: plan -> motion manifest v2 (2,403 utterances, train 1,652 / val 412 /
  test 339, SHA `b0187725…`).
- `scripts/translate_english.py`: English -> planner -> plan line -> plan-trained generator -> motion.
- Plan-conditioned generator stage ready (`runpod/fleet/post_gate_plan_generator.sh`, starts after task No.1's exit
  gate):
  - Training: the full-corpus RVQ prior is warm-started (`train_motion_tokens.py --init-key`) on one mixed manifest,
    How2Sign English plus ASLLRP plan lines (`build_mixed_text_manifest.py`), with the full-corpus tokenizer. All
    states are resampled to 24 fps (`--fps`; ASLLRP is 30, and the priors count time in frames).
  - Plan lines take their grammar features from their marker tags, not English words
    (`grammar_motion.text.is_plan_line`). No How2Sign sentence is misread; 2,402 of 2,403 plans are recognised.
  - Evaluation on ASLLRP val: the plan generator vs the English-conditioned full prior on the same utterances.
    Retrieval and back-translation are judged against the English, and both systems are checked against the plan's
    markers.
  - Evaluation on How2Sign val checks for any regression on English input.
- Citation-stitching baseline (EVALUATION_FRAMEWORK §7):
  - Lexicon: `build_citation_lexicon.py`, R2 `citation-lexicon-v1/lexicon.json`, SHA `b326494e…`. 5,439 keys from
    training-partition clips only (2,648 ASLLVD Sign Bank glosses, 2,791 ASL Citizen).
  - ASLLVD clips are cut to the annotated sign: each DAI clip carries exactly 0.833 s of context either side.
  - `generation/citation_stitch.py` normalises gloss decoration, splits compounds and maps indices to the Citizen
    pointing signs. It covers 81.8% of val plan signs; fingerspelling, name signs and classifiers are held.
  - Evaluated beside the plan generator in the post-gate chain, and available as a `stitch:` system in the human
    study.
- ASLLRP data quality: DAI 2 has one camera view per utterance (cam1, 640×480, upscaled to 1280×960). The
  better-tracked hand's validity is 0.94 for Cory and 0.73 for Rachel (train), but 0.65 for Jonathan (val) and
  about 0.6 for Ben (test). Invalid frames are masked in training. Human-study utterances need hand coverage ≥ 0.7,
  so the recorded reference stays a fair upper bound.

**Item 3 — evaluation instruments**
- Back-translator: training on the full corpus (pod C).
- Sign recogniser: v1 failed cross-domain (below). v2 path implemented: `signtranslator/pose/resample.py` (corpora
  are 24 / 25-30 / 60 fps; the recogniser read frames), ASLLVD added through the exact gloss map, speed / trim /
  face-dropout augmentation; v2 training on a pod. `scripts/score_sign_generation.py` scores one-sign plans with a
  recogniser only after it passes validation.

**Item 3b — evaluation by Deaf ASL users (tooling complete; waits for a generator and for reviewers)**
- `docs/HUMAN_EVALUATION_PROTOCOL.md` (team) and `docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md` (sent to reviewers):
  - four-way comprehension choice plus a free translation;
  - blind recovery of sentence type, negation and topic, with ASLLRP markers as the truth;
  - naturalness 1-5.
- Design: a Latin square over reviewer groups, so no reviewer sees a sentence twice and systems are paired by
  utterance. Distractors are drawn from the whole partition and matched in length.
- `signtranslator/eval_framework/human_study.py`:
  - design and blinding: random ids, private hashed key;
  - sheet validation: hashed reviewer keys only, primed reviewers excluded;
  - scores: Wilson intervals, paired permutation against the recorded reference, Fleiss' kappa.
- `scripts/build_human_eval_packet.py` renders every system the same way, on the recording's clock. It shows the
  skeleton plus the new face panel (`generation/face_preview.py`: head turn, brows, eyelids, gaze, mouth). Tracking
  gaps are held for display, so dropouts do not reveal the reference.
- `scripts/score_human_eval.py` scores the returned sheets. Both scripts were smoke-tested end to end on real val
  utterances.

**Item 4 — tuning (sweep ready; runs after the full prior)**
- `runpod/fleet/post_gate_sampling_sweep.sh` evaluates the full prior on validation over:
  - hands temperature {1.0, 1.2, 1.4} × face temperature {0.8, 1.0}, a cooler face because it carries the
    over-active head codes;
  - guidance {1, 1.5, 2}.
- Each setting is scored by retrieval (with its blank control) and back-translation. The setting is selected with
  the same declared exit gate over the core chain's evaluations plus the sweep. The test partition stays untouched.

**Item 6 — serving (path built; measured on the current research release)**
- Pinned, hash-checked releases with a computed licence gate; a translation service; an HTTP API with bounded
  backpressure; a latency benchmark (docs/DEPLOYMENT.md §12).
- Measured on Mac CPU: p50 1.2 s end to end, with generation at 0.4× real time.
- The release is research-only, because the planner is trained on ASLLRP.

**Retrieval evaluator v2 result (2026-10-22, 16:20 PDT).**
- Setup: frozen all-MiniLM-L6-v2 sentence embeddings with a trained projection (`retrieval-minilm-full-v1`), on the
  same data as v1.
- Result: best R@3 **0.256** at step 3,000. v1 (byte text encoder) had 0.264, and chance is 0.094.
- Overfitting moved to the motion side: train batch top-1 0.82 while val R@3 fell to 0.20 by step 7,000. The limit
  is the motion encoder on ~20k clips, not the text encoder.
- v1 stays the instrument.
- v3 (`retrieval-aug-full-v1`) adds motion augmentation (speed 0.85–1.15×, face dropout 0.2) and dropout 0.3 to v1's
  design.
  - Best R@3 **0.264** at step 2,000, the same as v1.
  - Overfitting slowed (train batch top-1 0.31 at step 6k against 0.82 for v2), but validation still plateaued.
- Three designs reach the same ~0.26. That is the ceiling of pose-only retrieval on How2Sign sentences at this data
  scale, not a fixable training fault.
- Generated motion is therefore read relative to real (the ceiling) and blank (chance), never in absolute terms.

## Acceptance rules declared before results (2026-10-22, PDT afternoon)

**Task No.1 review (on top of the declared exit gate).** The exit gate (scripts/exit_gate.py) checks realism,
articulation and marker rates. It does not check meaning. The full-corpus instruments are weak:
- retrieval R@3 is 0.264 (chance 0.094), early-stopped at step 1,500;
- back-translation val BLEU is about 1.5, with val loss flat since step 6k.

So the selected setting is accepted for the next phase only if, on the same validation clips and through the same
frozen instruments:
- generated R@3 is above the duration-only (blank) R@3;
- generated back-translation BLEU is above blank BLEU.

Both are reported beside real, which is the instruments' ceiling. If either fails, the generator round is not done.
The failure is diagnosed (instrument or generator) before anything moves on.

**Strict pass test of the full-corpus prior (declared 2026-10-22, 17:40 PDT, before any prior result).**
`scripts/strict_prior_review.py` runs at the end of the sweep job, on the final gate selection. Task No.1 passes only
if all six rules hold, whatever the gate said:

| Rule | Test | Pass if |
|---|---|---|
| R1 | Retrieval R@3 through the frozen evaluator, per clip, bootstrap 95% CI | generated CI entirely above blank CI |
| R2 | Back-translation corpus BLEU, sentence bootstrap 95% CI | generated CI entirely above blank CI |
| R3 | Nearest-training-clip distance ÷ nearest unseen-real-clip distance (equal set sizes, Fréchet features) | ≤ 5% of generated clips below 0.5. The dry run on v2b gave 0% flagged, median 0.93 against 1.15 for real. A rule based on raw distance would have flagged mode-averaging as copying. |
| R4 | Sentence-length thirds | every third's body, hand and head speed and variation ratios in [0.7, 1.3] |
| R5 | One-word English prompts for the recogniser's reliable glosses | recogniser top-5 CI entirely above blank (real reported) |
| R6 | Test partition, once (92 signer-disjoint clips); heldout_official reported | every gate check that passed on val passes on test |

**Added after the first full-prior validation results (2026-10-22, 20:30 PDT). Disclosed: this is a change made
after seeing results.**
- **Finding:** the full prior's evaluation (guidance 1 and 2, 600 val clips) showed brow furrow at 0.09–0.10× the
  real rate, so the gate's marker check fails. It also showed head speed at 1.53×, which head smoothing already
  targets.
- **Diagnosis:** the tokenizer reconstructs the face blendshapes at 0.89–0.97× their real spread, but the prior
  samples near-neutral faces. Brow-down varies about 3× less than real within a clip.
- **New gate candidates:** `postprocess.calibrate_face`, a per-blendshape affine map estimated on validation that
  never shrinks a range. It was checked on a held-out half of the val clips: brow furrow 0.20× → 0.99×, brow raise
  0.92× → 0.73×, head markers unchanged.
  - The gate rule is unchanged.
  - The calibration is a post-processing setting chosen on validation, like head smoothing.
  - The strict review applies the stored validation calibration unchanged in every rule, and R6 confirms it once on
    the signer-disjoint test partition.
- **Proper fix:** a prior that samples the face's full range, through face-code temperature or a variance-aware loss.
  That belongs to the next generator round.

**Plan-conditioned generator (post_gate_plan_generator.sh), ASLLRP val (Jonathan).** The plan generator is preferred
to the English-conditioned baseline only if all three hold:
1. Retrieval R@3 and back-translation BLEU against the English are not lower than the baseline's, each judged at the
   better guidance of each system.
2. Grammar: for negation and for each question type with at least 10 positives, the marker AUC
   (`grammar.tests[].generated_auc`) is higher than the baseline's. A marker AUC is generated motion's separation of
   marked from unmarked utterances.
3. How2Sign val: Fréchet distance and every articulation ratio stay within 10% of the full prior's own evaluation.

The citation-stitching baseline is reported beside both. The generator should beat it on Fréchet distance and on the
marker AUCs; stitching shows only the citation forms' own non-manuals.

## Mandatory: the sign recogniser must pass as an instrument before it scores generated motion

The first isolated-sign recogniser (`training-runs/sign-recognizer-aslc-v1`, ASL Citizen, best step 27,000: 53.3% top-1 /
76.5% top-5 on a 2,000-clip validation subset) was validated with `scripts/validate_sign_recognizer.py` (rule fixed
before the run; report `training-runs/sign-recognizer-aslc-v1/validation-interim.json`, step-23,000 checkpoint):

| Check | Result | Rule | Pass |
|---|---|---|---|
| In-domain, all ASL Citizen validation signers (10,303 clips, 2,731 classes) | 49.2% top-1, 72.1% top-5 | reference | - |
| Cross-domain, ASLLVD train+val signers, exact Sign Bank map (2,858 clips, 917 glosses) | **13.3% top-1**, 27.7% top-5 | >= 25% top-1 | **No** |
| Blank (duration only) | 0.04% top-1 | <= 2% | Yes |
| No hands | 1.2% top-1 | <= 15% | Yes |
| Hands only | 19.4% top-1 | diagnostic | - |

It reads the hands (no shortcut on duration or face) but does not transfer to another studio, camera set and signers,
so **it must not score generated motion yet**. Required before any sign-level evaluation of the generator:

1. Retrain on ASL Citizen + ASLLVD together (ASLLVD labels mapped through the exact Sign Bank <-> ASL Citizen gloss
   map; ASLLVD test signer Tyler stays unread), with augmentation for camera and signer variation (view rotation,
   scale, speed, small joint noise) - same validation rule, re-run unchanged.
2. Pass all three checks; the published ASL Citizen baselines (~63% top-1 video, ~60% pose) remain the in-domain
   reference.
3. Score generated motion only on the glosses it recognises reliably on real data (top-1 >= 80% over >= 3 clips; 444
   glosses for v1), always beside real-motion and blank scores through the same model.

Until then the generator's meaning-level evaluation rests on the back-translator and the retrieval evaluator.

**v2 result (2026-10-22, 16:57 PDT).**
- Run and report: `sign-recognizer-v2`, best step 27,000; report `training-runs/sign-recognizer-v2/validation.json`.
- Data: ASL Citizen + ASLLVD train signers, 24 fps, augmented.
- Verdict: **passes all three pre-declared checks**. Usable as an instrument on its 474 reliable glosses (top-1 ≥ 0.8
  over ≥ 3 val clips).

| Check | v1 | v2 | Rule |
|---|---|---|---|
| In-domain, all 10,303 ASL Citizen val clips | 49.2% / 72.1% | 50.5% / 72.9% | reference |
| Cross-signer, ASLLVD val signer Naomi (411 clips, 348 glosses) | 13.3%* | **60.8%** / 82.2% | ≥ 25% |
| Blank | 0.04% | 0.03% | ≤ 2% |
| No hands | 1.2% | 1.5% | ≤ 15% |
| In-context ASLLRP signs, unseen studio and signers (diagnostic) | 2.6% | 8.3% / 17.8% (blank 0.17%) | - |

\*v1 was measured on ASLLVD train+val signers with no ASLLVD in training. v2 trains on the ASLLVD train signers, so its
cross check is on the held-out val signer: a new signer in a studio it has seen, not a new studio. The genuinely
unseen domain is the in-context ASLLRP row. It improved 3×, but isolated-sign recognition of coarticulated signs stays
low, so sign-level scoring of the generator stays on one-sign plans, beside real and blank scores.

In-context diagnostic (added 2026-10-22, not part of the rule): `validate_sign_recognizer.py
--continuous-manifest-key` cuts the annotated signs out of ASLLRP continuous utterances of the train/val signers
(Cory, Rachel, Jonathan; unseen by the recogniser), padded by 0.1 s.
- Sample: 5,348 signs from 1,716 utterances, median 0.23 s.
- v1 result: **2.6% top-1, 7.1% top-5** (blank 0.07%).
- Isolated-sign recognisers do not read coarticulated signs inside sentences. Sign-level scoring of the generator
  therefore stays on one-sign plans, and sentence-level meaning rests on back-translation, retrieval and the human
  study.

## Item 7 — YouTube ASL corpora (approved by the owner 2026-10-22, ~18:35 PDT)

**Decision.**
- The owner approved acquiring the YouTube-hosted ASL corpora for research. The candidates are YouTube-ASL (Uthus
  et al. 2023, ~1,000 h, 2,500+ signers) and OpenASL (Shi et al. 2022, ~290 h); the research below confirms or
  amends the list.
- **Release intent:** the full model will eventually be open-sourced as a *research* model. Every corpus therefore
  stays tracked in each release's provenance, and permission to release derived weights is requested from each
  dataset's owners before any public release.
  - Research: `docs/research/YOUTUBE_ASL_RESEARCH.md`, in progress. It covers which corpora and filters give the
    best model, the expected gains, acquisition practice, and contacts and terms for derived-weight permission.
- Data handling is the same as ASLLRP:
  - private R2 prefix `acquisition/2026-10-17-v1/youtube-asl-v1/` and `.../openasl-v1/`;
  - motion states and caption text only; videos are deleted after extraction and never redistributed;
  - per-video provenance (video id, channel, upload date, YouTube licence field, caption kind manual/auto).
  - Because of the licence field, a CC BY-only subset can be separated later if a more permissive release is
    wanted.

**Why.** Pose-only meaning measures are at a data ceiling:
- back-translation BLEU 2.1 (v1), with v2 (T5) training now;
- retrieval R@3 0.26 across three designs.

Published pretraining on YouTube-ASL took pose-based How2Sign translation to about 12 BLEU. Expected effects
(estimates, not results; to be replaced by the research report):
- back-translator BLEU about 6–12;
- retrieval R@3 about 0.35–0.45;
- generator meaning and sign-level scores about 1.5–3×;
- little change to non-manual grammar, which ASLLRP still supplies.

**Research result (2026-10-22, 19:00 PDT; `docs/research/YOUTUBE_ASL_RESEARCH.md`).**
- **Corpora, in priority order:**
  1. YouTube-ASL: 11,093 IDs, 984 h, every video filtered by Deaf annotators; IDs under CC BY 4.0; the list is in R2.
  2. The ASL rows of YouTube-SL-25 add about 410 h but were checked by a non-native signer, so they need stricter
     filters.
  3. DailyMoth-70h and ASL STEM Wiki: consent-based, not on YouTube. They need licence agreements, which only the
     owner can sign. Best suited to the generator.
  4. OpenASL: skipped. Over 72% of its test set is inside YouTube-ASL, and its CC BY-NC-ND licence is the strictest.
- **FLEURS-ASL** (Kaggle, CC BY-SA 4.0) is an independent evaluation set, never training data.
- **Expected gains (pose-only, How2Sign test):**
  - How2Sign alone gives about 1 BLEU, YouTube-ASL pretraining 12.4, all of YouTube-SL-25 15.4, against a human
    ceiling of 19.8.
  - Our target for the back-translator is about 8–14 BLEU, up from 2.1.
  - No published study measures the effect on sign *generation*, so it is A/B-tested here.
- **Filters:** the published ones are implemented in `build_youtube_asl_manifest.py`:
  - 10 s–5 h, at least 360 px high, 15–60 fps, captions covering at least 40% of the video;
  - captions of 300 characters or fewer, at 0.5–4 words per second;
  - face tracked on at least 90% of frames, median head yaw at most 30°, hand coverage at least 0.5;
  - near-duplicates of evaluation sentences removed (exact match, or word-5-gram Jaccard ≥ 0.8);
  - no channel above 5% of the records.
  - Still to calibrate in the pilot: a single-person check and the stricter generation-grade tier, which needs a
    reviewer check of the top channels.
- **Fallback if YouTube blocks the pods** (no circumvention): the UWB YouTube-ASL keypoints (MediaPipe 2D, CC BY 4.0,
  348 GB) or SignVerse-2M (DWPose 2D, CC BY-NC 4.0).
  - Both are 2D, so they can't supply our 3D rotations.
  - They can train a 2D-keypoint back-translator and retrieval evaluator that read generated states projected to 2D.
- **Permission requests (the owner sends them; draft in §4.5 of the report):**
  - Required: Carol Neidle (carol@bu.edu) for ASLLRP, DAI 2, Sign Bank and ASLLVD; Microsoft (ASL_Citizen@microsoft.com)
    for ASL Citizen.
  - Courtesy notices: the YouTube-ASL and YouTube-SL-25 authors at Google, and How2Sign (amanda.duarte@upc.edu).
- **Precedent:** Uni-Sign released YouTube-ASL-pretrained weights under CC BY-NC 4.0.
- **Risk accepted by the owner:** YouTube's Terms restrict downloading.
- **Permission requests sent 2026-10-22, ~19:05 PDT** from the owner's Gmail, in the owner's name (Shengbo Jiang,
  Kehillah High School, https://github.com/Shengboj0324/Sign-Translator), at the owner's instruction:
  - carol@bu.edu (ASLLRP / DAI 2 / Sign Bank / ASLLVD, permission);
  - ASL_Citizen@microsoft.com (permission);
  - the YouTube-ASL / YouTube-SL-25 authors at Google (notice + licence question);
  - amanda.duarte@upc.edu (How2Sign, notice).
  - Replies are tracked by the 20-minute inbox check and recorded in each release's provenance.

**Y2 pilot result (2026-10-22, 20:30 PDT; 95 YouTube-ASL videos, US-TX-3 GPU pod plus one 32-vCPU CPU pod).**
- **Bugs found and fixed during the pilot:**
  - a missing `libEGL`;
  - a broken OpenCV install, where two variants removed `cv2`;
  - English caption tracks filed under the sign-language code `ase` or regional `en-*`;
  - new pods in US-MO-1 and US-NC-1 that never started.
- **No YouTube blocking.** All downloads succeeded from the datacenter.
- **Availability:** 76 of 95 videos converted (80%). Removed and private videos: 11%. 6 videos with no captions
  remain, and 3 have no usable cues.
- **Segments:** 2,633 segments (3.66 h) from 249 min of video. Median hand coverage 0.97, face coverage 1.0.
- **After the manifest filters:** 2,152 segments (3.16 h), 82% kept. The main rejections were caption speed
  (181 too fast, 30 too slow), too few words (98) and hands not visible (83).
- **Throughput:** about 3.4 worker-minutes per minute of video, CPU-bound. Processing all of YouTube-ASL (984 h)
  takes about 3,300 worker-hours: about 2.3 days on two 32-vCPU CPU pods ($0.96/h each), or about 1.2 days on four.
- **Provenance gap:** yt-dlp returned no licence field for any video (the field is null). Channel, upload date and
  caption kind are recorded.

**Blocking event (2026-10-22, ~20:50 PDT).**
- **What happened:**
  - The first full-acquisition pods (two 32-vCPU CPU pods in EU-RO-1) ran 30 parallel downloaders each.
  - YouTube began answering "Sign in to confirm you're not a bot" after about 125 videos.
  - The polite restart still let workers queued on the download lock request once each, about 20 requests.
- **Response, under the no-circumvention rule:**
  - No cookies, sign-in, proxies or address changes.
  - Downloads are serialised to one per machine with a gap between them. The gate re-checks a refusal after taking
    its lock. A refusal makes every worker on the machine wait 1 h, and three refusals stop the shard.
  - Both pods were released during the block.
  - Acquisition resumes at ~22:20 PDT on **one** pod, one download per 24 s (≤150/h, under the ~300/h guest rate).
- **New estimate:**
  - Download-bound: all of YouTube-ASL takes about 3 days at 150 videos/h.
  - Instrument pretraining starts as soon as a useful amount is in. ~200 h of YouTube-ASL arrives in about 0.7 days;
    the published gains already show at a few hundred hours.
  - Fallback for the instruments if refusals continue: the UWB YouTube-ASL keypoints (MediaPipe 2D, CC BY 4.0,
    hosted outside YouTube).

**Stages, cost and time (estimates).**

| Stage | Work | Where | Time | Cost |
|---|---|---|---|---|
| Y0 | Research report: corpora, filters, gains, contacts, terms | agent (running) | ~1 h | - |
| Y1 | Acquisition tool `scripts/acquire_youtube_asl.py` | Mac (code + tests) | ~2 h | - |
| Y2 | Pilot ~50 h: availability rate, caption kinds, hand coverage, extraction throughput, blocking | 1 CPU pod | ~3–6 h | <$10 |
| Y3 | Full acquisition of the selected corpora | ≤4 CPU pods | ~2–4 days | ~$100–300 |
| Y4 | Manifest `youtube-asl-manifest-v1`: quality filters, dedup across corpora, train-only + small channel-disjoint monitor split | Mac | ~2 h | - |
| Y5 | Pretraining and fine-tuning in priority order (meaning instruments first: `runpod/fleet/youtube_pretrain_instruments.sh`, back-translator v2 pre-trained How2Sign+YouTube then fine-tuned on How2Sign, retrieval on the mix; then the generator) | GPU pods (300 GB disk) | ~1–2 days | ~$50–150 |
| Y6 | Same strict rules and gate as task No.1 | GPU pods | ~0.5 day | ~$10 |

Y5 covers, in order:
1. back-translator v2 (T5) and retrieval evaluator, pretrained on YouTube data then fine-tuned on How2Sign;
2. the generator prior on How2Sign + YouTube English (motion tokens from the full-corpus tokenizer, re-checked for
   learnability on the new domain).

**Second blocking event and fallback (2026-10-22, 22:27 PDT).** The polite resume ran on one CPU pod, one
download every 24 s, after a refusal window of more than 80 minutes. YouTube answered its first request with
"Sign in to confirm you're not a bot" on three workers at once. Under the rules, acquisition from YouTube is paused
and the pod was released at once. No cookies, sign-in, proxies or IP hopping. YouTube-ASL stays at the 125 pilot
videos (6.1 h).

Only the owner can unblock it. Acquiring at a low rate from a non-datacenter connection carries Terms-of-Service
implications (research report §4.3), so it is the owner's call. Until then the declared fallback runs:
- **Data:** the **YouTube-ASL Clip Keypoint Dataset** (UWB / JSALT 2024, LINDAT 11234/1-5898, **CC BY 4.0**),
  hosted by LINDAT, not YouTube. It is MediaPipe Holistic 2D, 33 pose + 2×21 hand + 478 face points per frame,
  for 390,547 single-signer sentence clips with English translations (train/dev JSON), 337 GB in 10 zips.
- **Acquisition:** `runpod/fleet/acquire_uwb_keypoints.sh` streams it into R2, MD5-verified, three streams at once.
- **Use:** a **2D "second-opinion" back-translator**, keypoints → English (T5), pretrained on the 390k clips and
  fine-tuned on How2Sign.
  - Both How2Sign and generated motion reach it the same way: canonical 3D state → forward kinematics →
    orthographic front projection → the shared 2D layout (shoulders, elbows, wrists, 2×21 hand points), normalised
    by shoulder centre and width. Fine-tuning and evaluation therefore share one domain.
  - It reads manual channels only. Face meaning stays with the existing instruments.
- **Not covered:** this does not give the generator more 3D training data. That needs the video (blocked).
- **Assumption:** clip names are frame ranges, and the source frame rate is not given, so 30 fps is assumed.
  Pretraining uses time-stretch augmentation of ±25%. The How2Sign fine-tune runs at its known rate.
- **Instrument acceptance before use** (same skepticism as the recogniser): on How2Sign val through the FK path,
  real BLEU must exceed blank and shuffled-text controls by a clear margin. Per-sentence agreement with the T5
  back-translator is reported. If it fails, it is not used.

## Task No.1 verdict: exit gate PASSED on validation, strict review FAILED (2026-10-23, 05:50 PDT)

**Gate.** The sweep over 9 per-part temperature settings × 3 guidance values produced 6 candidates that pass all 12
checks, every one combining head smoothing 0.2 s with face calibration. That fixes both checks the core chain failed
(head speed, brow furrow). The declared rule selected `sweep-h1.4-f1.2-hs0.2-fc` at guidance 2 (Fréchet 280.7): among
passing candidates within 10% of the best Fréchet (257.5), it has the lowest right-hand error.

**Strict review** (rules declared 2026-10-22, 17:40 PDT, before any result): **FAIL.**

| Rule | Result | Pass |
|---|---|---|
| R1 retrieval | generated 0.236 [0.20, 0.27] vs blank [0.07, 0.12]; real 0.290 | yes |
| R2 back-translation | generated 0.83 [0.62, 1.05] vs blank 0; real 0.80 | yes |
| R3 copying | 0% flagged; median ratio 0.92 (real 1.16) | yes |
| R4 length thirds | long and medium pass; short: body variation 1.47, right-hand variation 0.61 | **no** |
| R5 one-word prompts | recogniser top-5: generated 0%, blank 0%, real 99.4% (95 words) | **no** |
| R6 test, once | 92 signer-disjoint clips: 3/11 checks. Head speed 1.83×, nod rate 19.8×, brow raise 3.2×, body variation 1.69×. Official held-out (3,019 clips, reported): 8/12; head 1.35×, nod 2.0×, brow furrow 0.23×. | **no** |

**What this means.** Every partition is signer-disjoint, from fold 0 of the frozen fold plan:
- validation is **one** signer (3, 1,627 clips);
- test is **one** other signer (9, 92 clips);
- training is signers 1, 2, 4, 5, 8 and 11.

So validation was never leaked. It is a single unseen signer, and everything tuned on it (sampling setting, head
smoothing, face calibration) fitted that signer's style. Signer 9 differs. (Correction, 05:58 PDT: a first draft said
validation shares signers with training. It does not.)
- The post-processing tuned on validation did not transfer. Head smoothing brought head speed to 1.05× on val, but
  unseen signers see 1.35–1.83×. The face calibration's brow furrow was 0.67× on val and 0.23× on held-out.
- The carried problem, an over-active head (round 2026-10-21), is not solved; it was masked on validation.
- The generator cannot sign a single word. That is outside its How2Sign sentence-level training, and the recogniser
  reads it from a different capture domain (MediaPipe isolated signs). R5 cannot separate those two causes, but
  generated scores 0 where real scores 99%.
- Meaning at the sentence level is real (R1, R2, and the 2D second opinion).

**The model is not deployable under the declared rules.** The rules are not relaxed; the next generator round
addresses the causes.

**Next generator round (round 3), designed from these failures:**
1. **Multi-signer selection.** Choose sampling and post-processing across several unseen signers, not one.
   - Use the fold plan's leave-one-component-out folds: validation signers 3 and 1/2 (folds 0 and 3), plus signer 11.
   - Require each choice to hold on every validation signer.
   - Estimate face calibration from population statistics of all training signers.
   - R6, test signer 9, stays the untouched final check.
   Then fix head over-activity in the model itself, through head-code temperature and a head-velocity term in the prior
   training, not only by smoothing.
2. **Single signs.** Add ASL Citizen isolated signs (already in R2; research use) to training as one-word English
   prompts. Add ASLLRP citation forms too. R5 then measures a skill that was actually trained.
3. **Mixed tokenizer.** Retrain it on How2Sign + YouTube-ASL + ASL Citizen + ASLLRP. The domain check found ASLLRP
   face and blink encoded 1.5–2.3× worse; ASL Citizen and YouTube are checked the same way.
4. **More signers.** YouTube-ASL home acquisition adds hundreds of signers. This is the direct cure for
   signer-specific style and the main lever for R6.
5. **Short utterances.** Report R4's short-third ratios during training validation.
6. **Recogniser domain.** Validate the recogniser on How2Sign-layout isolated signs before it judges R5, through the
   same FK/conform path, as the plan generator's diagnosis already requires.

**Round 3 addition, declared 2026-10-23 at 11:50 PDT, before any round-3 result: head-excursion damping.**
- **What it is:** `postprocess.damp_rotation_channel` pulls each frame's head rotation toward the clip's mean by a
  factor: R'_t = R̄ · exp(a · log(R̄ᵀ R_t)), a ∈ {0.8, 0.6}. Smoothing removes speed; damping removes amplitude.
- **Why:** the full prior's R6 failure on the unseen signer was amplitude, not only speed (nod rate 19.8×, head
  variation 2.3×).
- **How it is selected:** the variants join the post-processing grid (every head smoothing × damping × pooled face
  calibration). They are chosen by the same multi-signer gate across validation signers, and R6, test signer 9, stays
  the final check.
- **Mechanics:** `runpod/fleet/round3_supplement.sh` runs after stage B. **Round 3's verdict is the strict review of the
  union selection.** Stage B's own strict review, on the undamped grid, is reported as preliminary.
- **Pipeline:** the release spec and service carry `head_damping`, and the strict review applies it in every rule.

**Home route: first soft refusals (2026-10-23, 11:05 PDT).** Two downloads (videos #830 and #956 of about 950) were
answered "Please sign in. This helps protect our community." That is YouTube's soft bot check. It was not in the
detector's patterns, so it was logged as an ordinary error.
- **Fixed:** the wording is now a block pattern.
- **Response, per the rules:** stopped; a 2 h back-off (resume about 13:04 PDT); then half the rate (one download per
  60 s); a 2 h back-off per further refusal; a full stop after 3.
- No sign-in, cookies or address changes, as before.
- 69.1 h collected so far. At the new pace, about 3 days to the 200 h threshold.

**Round 3a: implemented and running (2026-10-23, 06:20 PDT).**
- **Data** (mix `round3a-mixed-v1`, SHA `dc576f4a`; 49,235 training clips):

  | Corpus | Training clips | Manifest |
  |---|---|---|
  | How2Sign, multi-signer re-partition | 18,289 | `how2sign-text-manifest-v2-multisigner`, SHA `dcdb6451` |
  | YouTube-ASL r3a (home route so far; 23.3 h after filters) | 15,560 | |
  | ASL Citizen one-word prompts (capped at 6 per word, 2,289 words) | 13,734 | SHA `50125638` |
  | ASLLRP plans | 1,652 | |

  - Validation is How2Sign signers 3, 1/2 and 11 (3,096 clips), plus the other corpora's own validation splits.
  - Test is signer 9, untouched; held-out is How2Sign's official test.
- **Stage A** (`runpod/fleet/round3_tokenizer.sh`, running): a mixed tokenizer.
  - The full-corpus tokenizer's domain check on the round-3 corpora: ASLLRP fails (blink 1.91×), ASL Citizen passes
    (relative face 2.06×), YouTube passes (blink 1.48×).
  - The new tokenizer must pass the same check on every corpus, or stage B does not start (`after_tok.sh`).
- **Stage B** (`runpod/fleet/round3_prior.sh`): prior from scratch, then:
  - a 3×3 temperature sweep × 3 guidance values, evaluated per validation signer;
  - a post-processing grid with face calibration pooled over signers;
  - the multi-signer gate (every signer must pass the declared checks);
  - the strict review, R1–R6 unchanged, with R6 on test signer 9 once.

  A local dry run (2 signers × 8 clips) exercised every stage.
- **Caveat (instruments):** the retrieval evaluator and back-translator were trained on fold-0 training data, which
  included signers 1/2 and 11, now validation. Their real-clip ceilings on those signers are inflated. The R1 and R2
  rules compare generated with blank, not with real, so they stay valid.

**Second-opinion meaning check of the full prior (2026-10-23, 03:55 PDT).** The accepted 2D back-translator
(`back-translator-kp2d-h2s-v1`, an independent instrument reading FK projections) scored the full prior's core
validation generations on 600 How2Sign val clips:

| Guidance | Generated | Real | Blank | Shuffled | Share of the blank→real gap |
|---|---|---|---|---|---|
| 1 | 1.84 | 2.25 | 1.60 | 1.59 | 37% |
| 2 | **2.16** | 2.25 | 1.60 | 1.59 | **86%** |

Both instruments now find meaning in the generated motion above the text prior, so the signal is not an artefact
of one instrument. The 3D back-translator gave guidance 1 BLEU 0.68 against real 0.80 and blank 0.0. These are
point estimates; the strict review's R2 bootstrap rule uses the 3D translator, as declared.

**Plan-conditioned generator: result against its declared rules (2026-10-23, 03:25 PDT).**
`tokens-rvq-d0b-prior-loci-plan-v1`: warm-started from the full prior and trained on How2Sign + ASLLRP plan lines.
It early-stopped at step 13,000; the best checkpoint is step 8,000 (val loss 3.488).

| Rule | Plan generator | English baseline | Verdict |
|---|---|---|---|
| 1. Retrieval R@3 on ASLLRP val, better guidance | 0.128 | 0.177 | **fail** as declared. The instrument is invalid here: real ASLLRP scores 0.12 against chance 0.094. |
| 1. Back-translation BLEU on ASLLRP | 0.0 | 0.0 | **uninformative.** Real ASLLRP also scores 0.0, because the How2Sign-trained translator does not read ASLLRP. |
| 2. Marker AUC (guidance 2): wh / yes-no / negation | 0.665 / 0.557 / 0.483 | 0.572 / 0.445 / 0.459 | **pass.** Both are below 0.5 on negation (real 0.598). |
| 3. How2Sign Fréchet; articulation within 10% of the full prior | Fréchet 199 vs 290; body speed 0.92 vs 1.06; variation 1.00 vs 1.24 | | **better, but misses the letter:** two ratios moved more than 10%, both toward 1.0 (real). |

Other measurements:
- How2Sign back-translation: 0.71 (guidance 1) against the full prior's 0.68, with real 0.81 and blank 0.0.
- Sign level (recogniser v2, one-sign plans, 219 prompts): top-1 **0.5%**. Real ASLLRP citation clips score 97.3%, the
  stitching pipeline check 88%, blank 0%.
- ASLLRP Fréchet: 7,498 for the plan generator, 6,062 for the English baseline and 219 for stitching. Neither generator
  produces ASLLRP-like motion.

**Verdict: the plan generator is not adopted. The English-conditioned generator stays the selected system.**

**Diagnosis.** On ASLLRP, all three meaning instruments (retrieval, back-translation, recogniser) are either shown
invalid or unvalidated for the generator's output domain.
- **Retrieval and back-translation:** real ASLLRP motion scores at chance and at 0.
- **Recogniser:** validated only on monocular MediaPipe clips. The generators emit How2Sign-layout motion, Sapiens
  two-view, conformed. Stitching passes because it replays real ASLLRP states.
- So the sign-level near-zero cannot yet separate "the generator does not sign the gloss" from "the recogniser does not
  read this domain".
- **The fair sign-level reference is in-context signing.** The plan generator learned glosses from continuous ASLLRP
  utterances, so its one-gloss output resembles coarticulated, in-context signs. On real in-context ASLLRP segments,
  recogniser v2 reached 8.3% top-1, against 60.8% on isolated ASLLVD and 97% on the citation clips. Generated, at
  0.5%, is still about 16× below that in-context reference. That points to the generator as well as the domain.
- The grammar AUCs, which need no instrument, favour the plan generator. The tokenizer domain check found ASLLRP face
  and blink encoded 1.5–2.3× worse, which bounds what any generator on this tokenizer can show on ASLLRP.

**Consequences for the next round:**
- **Tokenizer:** one trained on the mix, as the YouTube round chain already does on a failed domain check.
- **Recogniser:** validated on generator-domain real signs before it judges generated signs. This needs How2Sign-layout
  isolated signs, for example re-estimating ASLLRP or ASL Citizen clips through the two-view-compatible path, or
  testing the recogniser on How2Sign motion conformed to the monocular layout.
- **Meaning on ASLLRP:** the 2D back-translator, which reads both domains through the same FK projection, is the
  candidate meaning instrument. It is only to be used after its own acceptance on ASLLRP real.

**2D back-translator, UWB-pretrained and How2Sign fine-tuned: acceptance PASSED (2026-10-23, 17:11 PDT).**
- **Pretraining** (`back-translator-kp2d-uwb-v2`): 323,978 UWB YouTube-ASL keypoint clips, selected on the UWB
  monitor split. Best step 76,000, monitor BLEU 6.09; it early-stopped at 96,000. Zero-shot How2Sign BLEU stayed
  about 1.1, so the How2Sign fine-tune was needed.
- **Fine-tune** (`back-translator-kp2d-uwb-ft-v1`): initialised from step 76,000 and trained on How2Sign FK
  keypoints. Best val BLEU 5.01, against 2.84 for the How2Sign-only model under the same selection; it early-stopped
  at 15,000. The acceptance scored checkpoint 9,000 on the same 600 val clips and rule as below:

| Control | BLEU | 95% interval |
|---|---|---|
| real | **4.09** | 3.51–4.68 |
| blank (durations only) | 1.95 | 1.56–2.36 |
| shuffled sentences | 1.59 | 1.24–1.94 |

- The real interval lies well above both controls. The gap over blank is 2.1 BLEU, against 1.0 for the How2Sign-only
  model, so the YouTube pretraining made the instrument clearly more sensitive.
- Per-sentence agreement with back-translator v2 (3D T5, BLEU 1.94 on these clips) is Spearman 0.74.
- This model replaces the How2Sign-only one as the 2D second opinion. Its reading of the full prior follows.

**Second opinion of the full prior through the fine-tuned 2D back-translator (2026-10-23, 17:27 PDT).** On the same
600 How2Sign val clips, BLEU is: real **4.26**, blank 1.95, shuffled 1.22, generated **1.43** (guidance 1) and
**1.74** (guidance 2). The full prior's generations score below the duration-only control. The more sensitive
instrument finds no sentence meaning beyond what timing gives, where the 3D back-translator had passed R2 (0.83 vs
real 0.80, within its noise).

**R2b added to round 3's verdict (declared 2026-10-23 17:35 PDT, before any round-3 strict result).** The round-3
union strict review (`round3_supplement.sh`) applies R2's rule through this 2D back-translator as well
(`strict_prior_review.py --second-translator-key`). The generated bootstrap BLEU interval of the selected setting,
pooled over the validation signers, must lie entirely above blank. Like every rule, R2b must pass for round 3 to pass.

**2D back-translator, How2Sign-only model: acceptance PASSED (2026-10-23, 02:32 PDT).**
`back-translator-kp2d-h2s-v1`, best checkpoint step 13,000; it early-stopped at 19,000. Scored on 600 How2Sign val
clips through the FK projection path, beam 4, with sentence-bootstrap 95% intervals:

| Control | BLEU | 95% interval |
|---|---|---|
| real | **2.45** | 2.01–2.90 |
| blank (durations only) | 1.47 | 1.15–1.77 |
| shuffled sentences | 1.53 | 1.19–1.82 |

- The rule passes: the real interval lies entirely above both controls.
- Back-translator v2 (3D T5) scores 1.94 on the same clips.
- Per-sentence agreement with v2 is Spearman 0.85: both instruments find the same sentences legible.
- **Limits:**
  - The margin over the text prior (blank) is modest.
  - Training loss reached 1.4 against a validation loss of 3.9. On How2Sign alone the model overfits early; the UWB
    pretraining stage is meant to fix that.
  - Beam outputs sometimes repeat a phrase.
- It may now be used as the second-opinion meaning instrument. The UWB-pretrained version replaces it only if it is
  better on val BLEU and passes the same rule.

**Home-connection route approved (2026-10-22, about 23:15 PDT).** The owner approved low-rate acquisition from the
home connection. The rules are unchanged: no cookies, sign-in, proxies or CAPTCHA work, and the first bot check
stops the downloader.
- **Home Mac:** `~/.sign-translator-ops/home_youtube.sh` runs `acquire_youtube_asl.py --stage download` in seeded
  random order (spread across channels), one video at a time through the polite gate, 30 s apart. Each video and
  its captions are staged in the private store.
- **CPU pod:** `runpod/fleet/youtube_extract_loop.sh` runs `--stage extract`, writes the reports and deletes each
  staged video. Staged videos are transient: only derived motion and captions are kept.
- **Pace:** Mac upload is 4.1 MB/s and a 720p video is about 85 MB, so roughly 60–80 videos (6–7 h of video) per
  hour. The plan's 200 h threshold for the YouTube pretraining round is about 1.5 days away.
- **Trial:** 3 videos staged, no bot check.

The UWB 2D keypoint instrument continues in parallel. It serves meaning checking, while the home-route video serves
the generator.

**Single-signer filter (2026-10-22, 22:15 PDT).** Acquisition counts people on every 12th frame with a separate
image-mode pose pass. Each segment records `other_person_fraction`, and the manifest drops a segment when a second
person at least half the signer's size appears in more than 10% of its frames (reason `second_person`). In an
interview or group shot the caption may belong to someone other than the tracked signer.
- The signer's own tracker stays single-person. Asking it for two people moved a lone signer's landmarks by a median
  2% of image width (90th percentile 13%).
- Validation: a lone signer scores 0% and a two-person clip 100%. The signer's landmarks are bit-identical, at no
  measurable cost.
- The 125 pilot videos were acquired before this and have no count, so their segments are kept.

**Tokenizer domain check (2026-10-22, 21:58 PDT).** `scripts/tokenizer_domain_check.py`, rule fixed before any
YouTube result: every channel's reconstruction error on a new corpus ≤ 1.5× How2Sign val's. A diagnostic, which
does not change the rule, divides each error by the corpus's own signal scale, separating "moves more" from "encoded
worse". First run, ASLLRP (24 clips) through the frozen full-corpus tokenizer:

| Channel | Error ratio | Signal-scale ratio | Relative-error ratio |
|---|---|---|---|
| body | 0.94 | 1.23 | 0.76 |
| left hand | 0.95 | 1.03 | 0.93 |
| right hand | 0.97 | 0.86 | 1.14 |
| head rotation | 0.94 | 1.00 | 0.93 |
| face | 1.41 | 0.92 | 1.53 |
| blink | **2.09** | 0.93 | **2.25** |

Result: **fail**. Body, hands and head encode as well as How2Sign. Face and blink do not: the motion is the same
size, so this is a real encoding gap, not more movement. Consequences:
- **Plan generator (running):** it uses the frozen tokenizer, so its ASLLRP face and blink output is bounded by
  this. Its grammar results on brows are reported with this limit.
- **YouTube round** (`runpod/fleet/youtube_generator_round.sh`): it checks YouTube-ASL and ASLLRP. A failure on
  either trains a tokenizer on the mix first (How2Sign + YouTube + ASLLRP), then the prior from scratch, rather than
  warm-starting.

**Rules.**
- YouTube data is training-only. No evaluation number may come from it: How2Sign val/test and ASLLRP val stay the
  judges.
- No circumvention of YouTube's bot checks, sign-in requirements or CAPTCHAs. If datacenter downloads are blocked,
  acquisition pauses and the alternatives go to the owner (e.g. pre-extracted public features, a residential
  low-rate path).
- Videos are fetched at ≤720p with English captions. Only segments whose caption timing fits the signing (duration
  limits, hands visible) are kept.

## Round 3 result: the multi-signer gate FAILED (2026-10-23, 22:33 PDT)

- **Union gate** (`tokens-rvq-d0b-prior-r3a-gate-union.json`): 675 candidates (9 temperature settings × 25
  post-processing variants × 3 guidance values). Each was scored on validation signers 1/2 (c01), 3 (c03) and 11
  (c05), with the declared thresholds unchanged.
  - **No candidate passes every check on every signer.**
  - The rule's fallback selects the most checks passed: `sweep-h1.2-f1.0-hs0.2`, guidance 2, **26 of 35 checks**.
    Its mean Fréchet distance is 229.4, against 280.7 for the full prior's selection.
- **Its failures go in opposite directions on different signers:**

  | Signer | Failing checks of the selection |
  |---|---|
  | c01 | brow raise 2.9× |
  | c03 (head too still) | headshake 0.31×, nod 0.26×; hand variation 0.69× / 0.67× |
  | c05 (head too active) | head speed 2.6×, variation 2.9×; headshake 8.9×; brow raise 6.4× |

- Across all 675 candidates, the most common failures are nod, headshake, head speed and variation, and brow raise.
- **Round 3 fails by its declared rules.** This is not reinterpreted after the fact.
- The union strict review (R1–R6 with R2b) on the fallback selection is reported when it finishes.
- **Operational faults fixed on the way:**
  - One sweep evaluation ran out of GPU memory (4 concurrent on 44 GB); it was rerun by the supplement.
  - `GROUPS` is a reserved bash variable: the gates were given signer group `0` (`fbef071`).
  - The strict review lacked `transformers` for R2b (`9d9f08b`).

**Instrument check of the multi-signer gate: real signers fail it (2026-10-23, 22:45 PDT).**
`scripts/cross_signer_baseline.py` scores real validation clips of one signer as if generated, with the gate's own
checks and thresholds, on the clips the evaluation used (`round3a/evaluations/cross-signer-baseline.json`).

| Real clips of | judged by | checks passed |
|---|---|---|
| c01 | c03 / c05 | 6 / 12, 7 / 12 |
| c03 | c01 / c05 | 4 / 11, 2 / 12 |
| c05 | c01 / c03 | 8 / 11, 6 / 12 |
| c01 / c03 / c05 | the other signers pooled | 8 / 12, 4 / 12, 6 / 12 |
| each signer | the other half of its own clips | 11 / 11, 12 / 12, 11 / 12 |

- Between real signers, head speed ratios run 0.29–3.5×, headshake rates 0.05–21×, nod rates 0.09–10.6× and brow
  raise 0–5×.
- Within one signer, nearly every check passes.
- So the per-signer head and face checks measure **who is signing**. A text-conditioned generator cannot pass them on
  signers 3 and 11 at once, and neither can real human signing.
- Round 3's selection (8.7 checks per signer on average) is closer to each signer than the other real signers are
  (5.5 of 12).

**Round 3b gate, declared 2026-10-23 23:07 PDT before any round-3b result.**
- **Envelope rule.** For each validation signer Y and each gate check c, the band is the declared band widened to the
  real cross-signer range: `[min(lo, min r_c(X→Y)), max(hi, max r_c(X→Y))]` over the other real signers X. The
  generated ratio against Y must lie in it.
  - In words: generated motion must be at least as close to each signer as another real signer is.
  - The band is never narrower than the declared one.
- **Population envelope.** The same rule against all validation signers pooled, with the band widened by the real
  signer-vs-others-pooled range.
- A candidate passes only if every check passes in every signer's envelope and in the population envelope.
- Selection, the meaning and copying rules (R1, R2, R2b, R3, R5) and R4 are unchanged.
- **R6** (test signer 9, once): each check that passed on validation must lie in signer 9's envelope. That envelope
  is the declared band widened by the validation signers' real ratios against signer 9.
- The envelopes are computed from real clips only, before any round-3b generation is scored.
- Human Deaf reviewers judge naturalness directly afterwards. The envelope only stops one person's style being
  scored as an error.

**Round 3 union strict review (2026-10-23, 23:36 PDT): FAILED on R2b, R4 and R5.**
The review judged the gate's fallback selection: `sweep-h1.2-f1.0-hs0.2`, guidance 2.

| Rule | Result | Numbers |
|---|---|---|
| R1 retrieval | **pass** | R@3 0.269 [0.241, 0.301] vs blank 0.094 [0.076, 0.113]; real 0.293 |
| R2 3D back-translation | pass | BLEU 0.60 [0.47, 0.74] vs blank 0; real 0.60 |
| R2b 2D back-translation | **fail** | generated 1.67 [1.41, 1.95], **below** blank 2.14 [1.81, 2.46]; real 3.88 |
| R3 copying | pass | 0% flagged |
| R4 length slices | fail | head speed 1.35-1.42x (short, long); head variation 1.46x (short) |
| R5 one-word prompts | fail | top-5 0 / 258; real ASL Citizen 98.9% |
| R6 test signer 9 | pass | no regressions of validation-passing checks |

- Retrieval meaning is close to real (0.269 vs 0.293), up from 0.236 for the full prior.
- The more sensitive 2D instrument still reads generated motion as less informative than durations alone.
- Round 3 is not deployable.

**Why R5 is 0: the ASL Citizen states are mostly missing arms and hands (2026-10-23, 23:40 PDT).**
These were probes on 40 ASL Citizen val clips, conformed to How2Sign's layout as the prior trains on them, with the
validated recogniser `sign-recognizer-v2`:

| Input to the recogniser | top-5 |
|---|---|
| real clips | 0.75 |
| the unit conversion alone (no tokenizer) | 0.75 |
| round-3 tokenizer round trip, quantised or not, 1-4 stages | 0.03-0.05 |
| round trip, with only the hands replaced by the reconstruction | 0.48 |
| round trip, with only the body (arms) replaced | 0.08 |
| **round trip on the frames the real clip observes**, its neutral fill elsewhere | **0.78-0.80** |

- The median ASL Citizen clip has valid **body (arm) rotations on 10%** of its frames and hand rotations on 26-27%.
  These are webcam framings where the elbows and hands leave the MediaPipe pose.
- The recogniser learned to recognise signs from those few observed frames with a neutral fill.
- Complete motion, even a faithful reconstruction of the real clip, is out of its domain: 0.75 -> 0.05.
- So **R5 as declared cannot grade generated motion**, which is always complete. This is the generator-domain
  instrument check that the instrument-skepticism rule demands, and it was missing.
- The tokenizer is not the problem. Where the data exists, its reconstructions are recognised as well as the real
  clips.
- It also means the generator's 13,734 one-word training clips gave it arm supervision on about 10% of their frames.

**Round 3b design, declared 2026-10-23 23:48 PDT before any round-3b result.** It addresses R5 (one-word data
without arms), R2b and R4 (head activity), plus more data.

- **Data** (`round3b-mixed-v1`):

  | Corpus | Training clips | Notes |
  |---|---|---|
  | How2Sign multi-signer | 18,289 | SHA `dcdb6451`, unchanged |
  | YouTube-ASL r3b | about 75k segments, about 140 h after round 3a's unchanged filters | at 200 h captioned |
  | ASL Citizen one-word | 13,734 | SHA `50125638`, kept for hands and face; arm frames are masked as invalid |
  | **ASLLVD one-word** | **5,156** | 2,080 words, 6 signers, arms valid on 66% and hands on 98-100% of frames; publisher partitions; ASLLRP Sign Bank terms; SHA `29a605c4` |
  | ASLLRP plans | 1,652 | |

- **Tokenizer.** Round 3a's tokenizer (`tokens-rvq-d0b-r3a`) is reused only if both checks pass:
  1. the domain check passes on YouTube r3b and ASLLVD (every channel at most 1.5x How2Sign val);
  2. a **recognisability round trip** passes: on ASLLVD val, the recogniser's top-5 on tokenizer reconstructions
     (scored on the frames the real clip observes) is at least 0.9x its top-5 on the real clips.

  Otherwise a tokenizer is trained on the round-3b mix first, and must pass the same two checks.
- **Prior:** from scratch, same architecture and schedule as round 3a, at most 60,000 steps, early stopping
  (patience 5).
- **Sweep:** hands {1.1, 1.2, 1.3} x face {0.8, 1.0, 1.2} x guidance {1, 1.5, 2}, per validation signer, 300 clips each.
  - At most 3 concurrent evaluations, with expandable CUDA segments.
  - Any failed evaluation is rerun before the grid.
- **Post-processing grid:** head smoothing {0, 0.1, 0.2, 0.3} x head damping {1, 0.8, 0.6} x {none, pooled face
  calibration}, per signer and pooled (`--pool-group pool`).
- **Gate:** the envelope rule declared above, per signer and for the population.
- **Strict review:** R1, R2, R2b, R3, R4 and R6 (envelope) as declared.
  - R5's word list, recogniser and pass rule are unchanged.
  - Each one-word prompt is sampled at **the generator's own predicted duration** instead of a fixed 1.2 s. Real
    ASL Citizen word clips run 1.5-5.8 s with their rests, so a fixed 1.2 s was shorter than every real clip.

**Round 3b data built and checked (2026-10-24, 00:25 PDT).**
- **YouTube-ASL r3b** (SHA `55f7e0d3`): 200.7 h captioned, so 97,792 segments and **147.8 h** after round 3a's
  filters. 97,371 train and 421 monitor.
- **Mix `round3b-mixed-v1`** (SHA `13ad9797`): 188,752 records.
  - Training: How2Sign 18,289; YouTube 97,371; ASL Citizen 13,734; ASLLVD 5,156; ASLLRP plans 1,652.
- **Checked:**
  - How2Sign val, test and heldout_official are identical to round 3a's.
  - Each validation signer group's first 300 clips (the evaluated ones) are identical too, so round 3a's real-clip
    baselines carry over.
  - No sign id is in two partitions.
  - Only 2 YouTube training captions of more than two words equal a How2Sign evaluation sentence, both the generic
    "So what does that mean?".
  - All four sources' terms are recorded.
- Launched `runpod/fleet/round3b_all.sh` (smoke run, then the round) at 00:27 PDT.

**Round 3b data rebuilt and every state verified (2026-10-24, 01:30 PDT).**
- **What happened.** The smoke run's prior stopped on a state whose bytes no longer matched `manifest-r3b`.
  - A duplicate extraction started before the claim heartbeat. Its claim had been taken over at the 3 h stale age.
  - It finished at 01:01, after the manifest was built, and rewrote `feXU2ZMBdrI`'s 1,426 states and report.
  - The prefetch had skipped the mismatch silently.
- **Fixed (`1d046c7`):**
  - An extraction never writes over a converted video: states are written only after the "already converted?"
    check, with status `duplicate`.
  - Claims of converted videos are not refreshed.
  - The prefetch exits 3 on any missing or mismatching state, and round 3b stops on it.
  - The last claimless duplicate process was stopped.
- **Rebuilt from the then-consistent reports** (207 h captioned):
  - YouTube `manifest-r3b2` (SHA `f69ad828`): 100,454 segments, **151.8 h** after filters.
  - Mix `round3b-mixed-v2` (SHA `ad8a5876`): 191,414 records, partitions and evaluation clips identical to round 3a.
- **All 100,454 YouTube states were downloaded and SHA-verified** (0 mismatches) before round 3b was relaunched.

## ASLLRP attribution requirements (Carol Neidle, email 2026-10-24 11:14 PDT)

ASLLRP's permission for a non-commercial weights release (2026-10-23) stands. Before any distribution:
1. **Cite each individual dataset used**, with the citation given on its DAI download page:
   - the ASLLRP continuous signing corpus (2,123 plan utterances);
   - the RIT sentence collection (280);
   - ASLLVD via the Sign Bank (5,156 one-word training clips in round 3b, plus the recogniser's validation).

   These citations are to be copied verbatim from the DAI pages (login) before release.
2. **List the DAI and Sign Bank websites:** https://dai.cs.rutgers.edu/dai/s/dai and https://dai.cs.rutgers.edu/dai/s/signbank.
3. **Give the full ASLLVD citation:** C. Neidle, A. Thangali and S. Sclaroff (2012), "Challenges in Development of the American
   Sign Language Lexicon Video Dataset (ASLLVD) Corpus", 5th Workshop on the Representation and Processing of Sign
   Languages: Interactions between Corpus and Lexicon, LREC 2012, Istanbul, 27 May 2012.
4. **The attribution wording is revisited with ASLLRP** when results are ready to distribute.

`scripts/build_model_card.py` now carries the full citations, both URLs and the attribution paragraph, marked as
awaiting ASLLRP's review.

She also points to C. Neidle (2023), "Challenges for Linguistically-Driven Computer-Based Sign Recognition from
Continuous Signing for American Sign Language", arXiv:2311.00762. Her point: recognising signs within continuous
signing is a prerequisite for translation.
- Our sign-level checks today (R5) use isolated one-word prompts.
- Recognition of signs inside generated continuous signing is not measured. It is recorded as a limitation and a
  candidate check for a later round, using the ASLLRP segmented continuous-corpus signs from the Sign Bank.

## Speech input: audio -> English text -> ASL (owner's request, logged 2026-10-24 00:00 PDT; after the generator passes)

The deployed model takes English text (`TranslatorService.translate(text)`). To translate spoken English, such as a
video's soundtrack, a speech-to-text stage goes in front of it:
1. **Audio -> English sentences.** Either the repo's speech front-end (`signtranslator/speech/`: streaming
   features, alignment, decoding, latency model) or a standard ASR model (for example Whisper, licence checked). It
   outputs timed sentences and punctuation for the planner.
2. **Acceptance before integration**, declared when the stage is built:
   - word error rate on held-out English speech against a stated bar;
   - median and p95 streaming latency;
   - sentence segmentation checked against the generator's input limits (`max_text_chars`).
3. **Integration:** `TranslatorService.translate_audio(audio)` runs ASR, then the existing text path. The release spec
   pins the ASR model by SHA-256, like every other component.
4. **End to end:** the strict review's meaning rules run on ASR transcripts of spoken How2Sign-like prompts, and the
   human Deaf reviewers judge audio -> sign output as well.

This starts once a generator round passes its strict review. The text path is the product's core, and speech only
feeds it.

## Plan for the six items (order and gates)

| # | Item | Plan | Gate before it starts |
|---|---|---|---|
| 1 | More data | (c) ASL Citizen isolated signs (section above). (a) The full How2Sign (≈80 h): run the existing Sapiens fleet and two-view chain over the remaining sessions, plus the take-5 re-cut stage (B80). (b) ASLLRP DAI 2 corpus through the same pipeline, under a private R2 prefix with provenance. | v2b priors reviewed; DAI 2 login from the user for (b) |
| 2 | English → ASL structure | Train an English → (gloss with locus indices + NMM spans) planner on ASLLRP. Condition the motion prior on the planned gloss and NMM sequence instead of raw English. The existing semantic planner (docs/SEMANTIC_PLANNER.md) is the base. | Item 1b |
| 3 | Meaning-level evaluation | (a) Pose → text back-translator on our representation (Q1), frozen on the training partition. (b) A protocol for evaluation by Deaf ASL users: comprehension, naturalness, grammar; recruited and paid evaluators; done by people, not automated. | (a) after the v2b review; (b) needs the user to engage evaluators |
| 4 | Model scale and tuning | Larger prior, longer training, sweeps of guidance strength and duration on the larger corpus. Use the merged 2.2× faster step. | Item 1a |
| 5 | Avatar rendering | Retarget body, hands, face blendshapes and gaze onto a rigged signer (docs/AVATAR_RENDER.md). Face and gaze matter for grammar. | Any time; independent |
| 6 | Serving | Export (ONNX/TensorRT), cached text encoding, versioned checkpoints, an API with latency targets, monitoring (docs/DEPLOYMENT.md). | After items 3 and 4 show a model worth serving |

**Licence gate for release:** a release build lists every corpus in its training
provenance. ASLLRP (and any other research-only data) is either excluded or covered by
written permission.

## Execution order after the generator round (recorded 2026-10-21 06:20 UTC)

These start when the generator round's exit gate closes (docs/CONTINUOUS_TRAINING.md). Per the user's instruction of 2026-10-21, they proceed without waiting for input, under the same quality rules:
- every pod stops when its chain ends;
- every stage's log goes to R2;
- data stays in R2 under private prefixes.

1. **ASL Citizen acquisition** (item 1c). One small pod runs `acquire_resumable.py` for both files (≈ 46 GB, about 30 min on a datacentre link). Then:
   - verify the ZIP members;
   - extract monocular states;
   - map glosses.
   - **Status (2026-10-21 12:55 UTC).**
     - Acquired and verified: `ASL_Citizen.zip`, 45,924,134,223 B, SHA-256 `349c0e33…8eab`, re-read from R2.
     - Contents: 83,399 videos, 2,731 glosses, every row with an ASL-LEX code. Splits: train 40,154 (35 signers), val 10,304 (6), test 32,941 (11).
     - Extraction runs on 16 shards (`runpod/fleet/extract_asl_citizen.sh`, about 73 clips/min per 13.6-core pod).
   - **Smoke-test findings (fixed before the full run):**
     - The pod image lacked `libEGL`, and the extractor recorded the failure as a refusal of every clip. Now each worker probes the MediaPipe runtime at start and fails the job.
     - Webcam framing cuts the body above the hips: hips are visible in 2% of frames. The hip-based chest frame then invalidated every arm joint, so the body was available in 3 of 48 clips.
     - The opt-in `--upright-chest` prior takes the chest up axis from the camera vertical. It keeps chest yaw and roll and drops its lean. How2Sign is unchanged.
     - On 512 clips, the body is now available in 84%. 77% of clips have arm rotations in at least 20% of their frames (median 41%), and hands are seen in a median 38% of frames. Isolated-sign clips start and end at rest.
   - **Manifest:** `scripts/build_asl_citizen_manifest.py`.
     - Publisher partitions, with signer-disjointness checked.
     - Participants as pseudonyms.
     - Gloss and ASL-LEX code with status `publisher_label`.
     - Per-split quality counts.
   - **Done 2026-10-21 15:26 UTC.**
     - All 83,399 clips converted: no refusals, no unresolved errors.
     - Nine shards died at 13:29 on a transient R2 failure and were resumed after the retry fix.
     - Manifest: `derived/asl-citizen-v1/manifest-v1/manifest.json`, SHA-256 `708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6`, built on the Mac.
     - Clips with arm rotations in at least 20% of frames: train 77%, val 70%, test 79%. Median arm fraction 0.41, 0.35, 0.41.
     - Clips with hands in at least 20% of frames: 91–93% in every split.
     - Downstream uses filter on the per-clip `quality` counts.
   - **Gloss mapping:** 887 of the 2,731 glosses match SAPA's ASL-LEX phonology entries exactly. Normalised matching is not used, since it would merge variants such as `SOCCER2`. The ASLLRP Sign Bank join waits for DAI 2.
2. **Full How2Sign** (item 1a).
   - **Sapiens:** keypoints for every remaining clip. `runpod/pipeline.sh` without `SUBSET`; R2 claims skip the ≈ 13k frontal and side clips already done. About 57k clips at ≈ 40 clips/min on ten L40S pods: ≈ 24 h, ≈ $260.
   - **Take-5 sessions:** re-cut first (B80, `scripts/recut_how2sign_take5.py`).
   - **3D:** calibration and triangulation per session.
   - **Data:** manifest v2.
   - **Retrain:** the selected generator family and the retrieval evaluator on manifest v2.
3. **Meaning-level evaluation** (item 3a). The retrieval evaluator exists (`text_motion_retrieval.py`; R@3 ≈ 0.29 on real validation motion against chance 0.094 at 4k clips; it overfits early). Retrain it on manifest v2 with stronger regularisation. Score every generator with R-precision and keep their generated states for the human reviewers.
4. **DAI 2 / Sign Bank** (items 1b and 2).
   - **Access:** the user's account was activated on 2026-10-21. The user signs in; Claude never types passwords.
   - **Acquisition** (`~/.sign-translator-ops/dai_uploader.py`): downloads from the signed-in browser session into private R2 at `asllrp-dai2-v1/<collection>/`.
     - Each file is identified by its exact server size.
     - It is SHA-256 hashed while uploading, re-read from R2 and re-hashed, and written with a provenance record (source URL, version, licence).
     - The local copy is deleted once verified.
   - **Scope:** 121 current-version files, about 204 GB. The older archived releases (about 180 GB) are skipped: the site asks that the CSV and videos come from the same version. ASLLVD sign batch 17 is a 281-byte stub on their server.
   - **Done (2026-10-21 22:20 UTC):**
     - every annotation CSV and Excel file;
     - the ASLLRP continuous-corpus sign, utterance and source videos;
     - all RIT sentences files;
     - all RIT citation files.
   - **Remaining:** the ASLLVD videos (about 176 GB). Waiting for the user to sign in again, because the DAI session expired at about 22:05. Downloads then return HTTP 500 (`login_id` null).
   - **Sign Bank outputs** (`scripts/build_signbank_lexicon.py`, under `derived/signbank-v1/`):
     - lexicon: 4,037 main entries;
     - ASL Citizen gloss map, by tier: 977 exact, 17 variant, 147 letters, 421 letters-no-number (non-exact matches need review), 1,169 unmatched;
     - 2,410 ASLLRP and RIT utterances, holding 19,510 signs in time order.
   - **Utterance motion:** extracted on the Mac. The ASLLRP studio framing shows hips, elbows and wrists in every frame, so the upright-chest prior is not used.
   - **Gloss -> motion manifest** (`derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json`, SHA `f8c40aea…`): all 2,407 extracted
     utterances (2,127 ASLLRP, 280 RIT; 3.1 h) joined to their gloss sequences. Signer-disjoint: train 1,655, val 413 (Jonathan),
     test 339 (Ben); 3 rows without a readable signer left out. Median hand coverage 81% of frames (p10 46%), face 100%.
     Some glosses carry classifier descriptions in backslashes (`ICL:X-over-thumb\eating out of a bowl\`); the planner must
     normalise them before use.
   - **DAI 2 transfer complete (2026-10-22 06:20 PDT):** all 121 files (200.8 GB) are in private R2 under
     `asllrp-dai2-v1/`, each re-read and re-hashed after upload with a provenance record (source URL, SHA-256, licence).
     That includes the ASLLVD citation-form source and sign videos (93 files, about 172 GB). Their motion extraction is next.
   - **English translations (acquired 2026-10-22):** the DAI 2 utterance search export (2,407 utterances: English translation and
     SignStream rough-gloss tiers) is in R2 (`asllrp-dai2-v1/asllrp-continuous/utterance-search-2026-10-22.json`, SHA `cc9102d8…`).
     `scripts/build_asllrp_english_gloss.py` wrote the planner's pairs (`derived/asllrp-dai2-v1/english-gloss-v1/pairs.json`,
     SHA `91c9f83f…`): 2,403 English -> gloss pairs (train 1,652 / val 412 / test 339, same signer split), each with its
     grammatical markers mapped to the signs they span: topic/focus 982, negation 600, conditional/when 428, wh 325,
     rhetorical 249, yes-no 238, relative clause 43.
5. **Avatar and serving** (items 5 and 6) follow, with a model selected on R-precision and the realism gates.
6. **Human Deaf reviewers** (on their way): they evaluate the stored generated states and the rendered avatar. Their findings drive fine-tuning.

## Smoothness rule R7 and head-translation smoothing (declared 2026-10-24, 17:55 PDT, before the round-3b verdict)

**Finding.** The owner saw the first round-3b avatar render's head shake. Measured on 30 validation sentences
(sweep h1.3-f0.8, c03, guidance 2) against the same sentences' real clips:
- The share of motion energy at ≥ 6 Hz for head translation was 6.2% vs 0.6% real. The tokenizer's round trip alone gives 2.9%.
- For head rotation it was 3.6% vs 0.8%.
- Body, hands and root were at or below real.
- In absolute amplitude (`motion_metrics.high_frequency_rms`), head translation is 2.0× real and head rotation 1.05×. The excess is broadband above 3 Hz, not a token-rate line.

The post-processing grid smooths and damps only head *rotation*; head translation passes through untouched. No gate
or review check measures jitter, because the articulation ratios compare mean speeds.

**R7 smoothness (new rule, blocks release like R1–R6).**
- For body, both hands, head rotation, head translation and root translation, the median over clips of the generated ≥ 6 Hz amplitude must be at most 1.5× the median of the same sentences' real clips.
- It applies after the candidate's own post-processing, in every validation signer group, and again on test.
- Script: `scripts/smoothness_check.py`.

**Head-translation smoothing.**
- `postprocess.smooth_vector_channel` smooths head translation with the same centred window as the head rotation.
- On the sample above, 0.2 s on both head channels gives at most 0.73× real on every channel, with slow motion kept (unit tests).

**Procedure for round 3b.**
- The pod's chain runs as declared.
- If its strict review passes, the selected candidate gets head-translation smoothing equal to its head-rotation smoothing (at least 0.2 s).
- The strict review is then rerun with it, plus R7. Both must pass before release.
- If the review fails, round 3c adds translation smoothing to the grid and R7 to the review from the start, and treats the tokenizer's head-translation reconstruction (2.9% vs 0.6%) as a training target.

## Round 3b result: FAILED (2026-10-24, 23:45 PDT)

- **Gate (envelope, multi-signer): 46 of 47.** Selected sweep h1.2-f1.0 at guidance 1.5, head smoothing 0.1 s, damping 0.8, face calibration on, Fréchet 226. The one failure is c03 right-hand variation, 0.664 against the 0.70 floor. Head checks pass in every group.
- **Strict review:**
  - R1 retrieval PASS: 0.221 [0.193, 0.248] vs blank 0.094, real 0.293.
  - R2 PASS: BLEU 0.78 vs blank 0.
  - R3 PASS: no copies.
  - **R2b FAIL:** 1.77 vs blank 2.14 (real 3.88).
  - **R4 FAIL:** head speed 1.45 (short sentences) and 1.33 (long); short-sentence body variation 1.38.
  - **R5 FAIL:** 0 of 111 words recognised (real 98.8%).
  - **R6 FAIL:** test brow-raise 2.63 against an envelope maximum of 2.58.
- **R5 root cause (measured on the Mac):**
  - The generated single words have plausible durations (2.3–2.9 s), but the right hand moves at 29–77 °/s against 125–254 °/s real. The recogniser reads the real clips correctly, so it is not at fault.
  - The prior's single-word targets come from two corpora. The 13,734 ASL Citizen clips have the body tracked on 14% of frames, and the tokenizer's neutral fill takes their median right-hand speed to **0 °/s**. The 5,156 ASLLVD clips keep 135 °/s.
  - The prior learned that a single word means static hands. Round 3's 0 of 258 has the same cause.
- **R2 vs R2b:** both translators are weak readers, and R2 rates generated motion above real (0.78 vs 0.60). R2b's instrument is checked before round 3c treats it as a target.

## Round 3c design (declared 2026-10-25, 00:05 PDT, before any round-3c result)

1. **Data:** mix `round3c-mixed-v1` (SHA `004bf6596a23…`, 125,015 training clips). It is round 3b's mix without ASL Citizen's motion: How2Sign, the YouTube r3b2 snapshot (unchanged, so data is not a second variable), ASLLRP plan lines and ASLLVD. ASL Citizen stays only in the recogniser, an evaluation instrument covered by its consent. That also removes it from the generator's provenance.
2. **Prior:** same tokenizer (r3a), architecture, schedule and 60k-step cap as 3b, plus `--corpus-weight asllvd-text-v1=3`, so single words are about 11.5% of the samples.
3. **Sweep:** hands {1.2, 1.25, 1.3} × face {0.8, 1.0} × guidance {1, 1.5, 2}, the hands and face settings 3b found best, plus 1.25 for c03's right-hand variation.
4. **Grid:** head smoothing {0.1, 0.2, 0.3} × damping {1, 0.8, 0.6} × face calibration. Head-translation smoothing equals the head-rotation smoothing (at least 0.1 s). The grid runs on CPU pods.
5. **Gate:** the round-3b envelope gate, plus R4's sentence-length slices as gate checks (per slice, head and body speed and variation in [0.7, 1.3]), plus R7 smoothness, so the selection cannot pick a candidate that the review then fails on those.
6. **Strict review:** R1–R7 as declared (R7 per 2026-10-24 17:55), and the R6 test confirmation once. Every rule must pass.
7. **Operations:** the GPU pod runs only GPU stages; the sweep uses two GPU pods; the review's memory uses `expandable_segments`.

## Round 3c amendment: tokenizer stage (declared 2026-10-25, 00:20 PDT, before any round-3c result)

**Finding.** Real How2Sign validation clips (c01/c03/c05) passed through round 3a's tokenizer and back, measured on the frames each clip observed (`scripts/tokenizer_fidelity_check.py`):
- head rotation speed 1.58–1.93× real;
- hand speed 0.71–0.80× and hand variation 0.67–0.74×;
- body speed 0.67–0.72×;
- R2b on 200 clips: round trip 3.05 [2.27, 3.90] vs blank 1.74 [1.04, 2.32]. The intervals overlap.

This is round 3b's remaining motion failure profile before any prior is involved: head too active (R4), right-hand variation too low (the gate). Head rotation is observed on 100% of frames, so this is the tokenizer's quantisation, not gap filling. Round 3b's prior job was stopped during its smoke run's data check; nothing was trained.

**Amendment.**
- Stage A trains three tokenizer candidates in parallel on `round3c-mixed-v1`, with round 3a's recipe otherwise (hidden 512, code dimension 128, 40,000 steps maximum, early stop):
  - s6: 6 residual stages;
  - s8: 8 residual stages;
  - c1024: 4 stages, 1024-entry codebook.
- Each candidate must pass all three checks:
  1. the domain check (YouTube, ASLLVD);
  2. the recognisability round trip (ASLLVD val);
  3. the new fidelity check (300 val clips): body, both hands and head rotation speed and variation ratios in [0.85, 1.15], and the R2b round trip's interval entirely above blank's.
- The passing candidate with the fewest codes per step is used (c1024 16, s6 24, s8 32).
- If none passes, the round stops before the prior (TOKENIZER-FAILED) and is redesigned.
- Everything else in the round-3c design is unchanged.

## Round 3c amendment 2: a velocity-matched candidate (declared 2026-10-25, 02:45 PDT, before any tokenizer verdict)

**Audit of the step-6000 checkpoints (s8, s6)** on real val clips (`scripts/tokenizer_audit.py`, How2Sign c01/c03/c05):
- head speed 1.8–3.2×;
- hand speed 0.67–0.86×;
- right-hand variation 0.59–0.66×;
- head-translation jitter 2.5–5.4×.

They are no better than round 3a's tokenizer, although their validation loss is lower. The tokenizer's loss is per-frame error, plus velocity and acceleration only through the body and hand skeleton. Head rotation, head translation and the hand rotations carry no velocity term, so minimising per-frame error damps fast motion and leaves noise on near-still channels.

**Amendment:**
- `motion_rvq.velocity_matching_loss` (opt-in `--velocity-weight`) adds a relative frame-to-frame motion error per channel.
- A fourth candidate, **s8v** (the s8 recipe plus velocity weight 1.0 on body, both hands, head rotation and head translation), trains on its own GPU pod (`runpod/fleet/round3c_tokenizer_v.sh`) and is judged by the same three checks.

**Precedence:**
- If any of s6, s8 or c1024 passes, it is used.
- Only if all three fail and s8v passes does s8v's pod train the prior, with the same recipe.
- If no candidate passes, the round stops before the prior.

## Round 3c amendment 3: s8v3 (declared 2026-10-25, 03:25 PDT, before any tokenizer verdict)

The audit of s8v at step 8000 (How2Sign c01/c03/c05):
- head speed 0.97–1.02×, head-translation jitter 0.80–0.93×, now in band;
- right-hand speed 0.81–0.83× and variation 0.73–0.80×;
- body speed 0.64–0.71× and variation 0.55–0.70×, still damped. YouTube and ASLLVD are at 0.83–0.94×.

A fifth candidate, **s8v3**, uses s8v's recipe with velocity weight 3 on body and both hands (1 on head rotation and translation). It runs on its own pod (`runpod/fleet/round3c_tokenizer_v3_all.sh`) with the same three checks. Precedence: c1024, s6, s8, then s8v, then s8v3. A candidate trains the prior only if every earlier one failed and it passes.

## Round 3c amendment 4: s8v5 (declared 2026-10-25, 04:30 PDT, before s8v3's and the original candidates' verdicts)

**s8v failed its domain check**, and correctly. Its per-frame pose error is far above round 3a's:
- How2Sign: body 8.3° vs 3.5°, head 9.8° vs 2.6°;
- YouTube and ASLLVD: body 26–27° vs 4°.

At weight 1, the velocity term, which is relative and of order 1, outweighed the per-frame loss (of order 0.01) by 10–100×. Velocity matching ignores a constant offset, so the tokenizer kept the motion and lost the pose. Speeds and variation were close (head 0.93–1.13, YouTube and ASLLVD 0.83–0.99) while positions drifted. The deep audit (`tokenizer_audit.py`) now also reports absolute per-frame error, so a speed gain cannot hide a pose error.

The per-batch normaliser also diverged on near-still batches (validation loss 14.6 at step 22000). Fixed in 418fb9e: a running energy scale.

**New candidate s8v5** (`runpod/fleet/round3c_tokenizer_v5_all.sh`):
- 8 stages, the stable velocity matching at weight 0.05 on the same five channels;
- the same three checks;
- precedence after c1024, s6, s8, s8v and s8v3.

## Round 3c amendment 5: noise check and a declared amplitude fallback (2026-10-25, 05:00 PDT, before any pending fidelity verdict)

**Measurement check.** Real clips smoothed over 0.1 s, against themselves:
- speed drops to 0.54–0.78 (body 0.54–0.71); variation stays at 0.84–0.98;
- so raw speed is noise-sensitive.

Recomputed with both sides smoothed (How2Sign, 60 clips):

| tokenizer | body speed | hand speed | variation |
|---|---|---|---|
| r3a | 0.80 | 0.74–0.76 | 0.72–0.94 |
| s8 at step 20k | 0.80 | 0.78–0.84 | 0.73–0.92 |
| s8v at step 20k | 0.77 | 0.80–0.83 | 0.69–0.95 |

No verdict would change, so **the declared fidelity rule stands unchanged**. Hand and body amplitude is genuinely damped by about 20% in every candidate.

**Fallback, declared before the verdicts of c1024, s6, s8, s8v3 and s8v5.** It applies only to a candidate whose every failure is speed or variation of the body or hands below the band, while head rotation is in band and the domain and recognisability checks pass.
- **Correction:** a per-channel excursion gain after decoding, `R' = R̄ exp(g log(R̄ᵀ R))` (`postprocess.damp_rotation_channel` with a factor above 1). For each channel, g is fitted on 300 *training* clips as real variation over round-trip variation, capped at [1, 1.5].
- **Rule:** the fidelity check is rerun on validation with the gains applied and must pass in full (all ratios in [0.85, 1.15], R2b above blank). The recognisability check must also still pass.
- **Downstream:** the gains become part of that tokenizer's decoder and are applied to every generated state downstream (sweep, grid, review), recorded in the release spec.
- **Precedence:** a candidate that passes without gains outranks one that needs them.

## Round 3c amendment 6: s8v6 (declared 2026-10-25, 05:30 PDT, before s8's and s8v5's verdicts)

**Verdicts so far:**

| candidate | domain | recognisability | fidelity |
|---|---|---|---|
| c1024 | pass | pass (0.996) | fail: head 2.04, hand speed 0.74–0.76, hand variation 0.64–0.67, body speed 0.68 |
| s6 | pass | pass (1.0) | fail: head 1.90, hands 0.70–0.82, body speed 0.68 |
| s8v | **fail** (pose error) | — | — |
| s8v3 | **fail** (pose error; validation loss diverged to 102 at step 25000, best at 20000) | — | fail, damping only: body 0.76/0.84, hands 0.74–0.79; R2b round trip 3.23 vs blank 2.09 |

**s8v5 audit at step 8000** (How2Sign):
- head speed 1.09–1.30, head-translation jitter 0.72–0.82;
- body pose error 6.8–9.9° (s8 at step 20k: 5.7–8.2°);
- hand variation 0.64–0.76.

**New candidate s8v6:** velocity weight 0.3 on head rotation and head translation, 0.05 on body and both hands, otherwise s8v5's recipe. The head barely moves, so its per-frame loss cannot see jitter and needs a stronger velocity term than the articulators. Precedence: after s8v5.
