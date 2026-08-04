# 09 — Pseudo-Gloss Model Research and Decision Dossier

## 0. Status, scope, and activation boundary

Implementation authorization was granted on 2026-08-04. The local, tool-free hybrid
candidate-lattice subsystem is now implemented under `signtranslator/pseudo_gloss`.
That authorization does not waive the activation gates in this document. No How2Sign
transcript has been submitted to the untrained implementation, no corpus pseudo label
has been generated, and no unreviewed pseudo token may enter `gloss_tokens`.

The gloss-free audit and the quarantined 2D reconstruction experiment are complete.
They establish provenance, structural accounting, pose-quality evidence, and a
limited motion-learning baseline. They do not establish a linguistic label source:

- 31,165 metadata rows were accounted for; the database also records one orphan
  artifact, for 31,166 explicit statuses;
- 2,423 clips are structurally valid, 28,621 have quality warnings, 118 are missing
  source clips, and three have structural failures;
- the three structural failures are a corrupt rendered MP4, non-contiguous OpenPose
  indices, and a multiple-person ambiguity;
- the retained observations are 2D OpenPose coordinates, not the production
  tokenizer's 6D joint rotations;
- the source-disjoint masked-reconstruction model lost to temporal interpolation on
  point and span masks. It demonstrated optimization connectivity, not linguistic,
  anatomical, 3D, or production capability.

Stage B therefore remains unapproved. Authentic gloss, authoritative signer identity,
and qualified signer review are still absent. Stage C remains blocked.

## 1. Vocabulary and epistemic boundaries

| Term | Meaning in this project | May populate `gloss_tokens` now? |
|---|---|---|
| Authentic gloss | A human linguistic annotation created from the signed video under a declared convention, with annotator provenance and source alignment. Gloss is still an annotation convention, not ASL itself. | Only after provenance, mapping, rights, and qualified review pass. |
| Pseudo-gloss | A machine-generated approximation of a gloss sequence. It is a noisy latent hypothesis, even when it resembles gloss notation. | No. It must remain a separate candidate object. |
| English transcript | A spoken/written English rendering aligned to the source material. It is not a transcription of ASL form or order. | No. |
| Semantic concept | A language-neutral or task-specific meaning unit. It need not correspond one-to-one with an ASL lexical sign. | No. |
| SIR | The project's structured semantic intermediate representation: entities, predicates, roles, reference, discourse, and non-manual/spatial requirements. It is a planning interface, not automatically observed ground truth. | No. |

Uppercase tokens, reordered English words, dictionary lemmas, or LLM output do not
become authentic gloss through formatting. A pseudo-gloss can be useful as weak
supervision only when its provenance and uncertainty remain visible to every consumer.

## 2. Evidence review

### 2.1 What the How2Sign evidence establishes

Duarte et al. describe How2Sign as an 80+ hour multimodal, multiview continuous-ASL
dataset with English transcripts, speech, depth, 2D keypoints, and a Panoptic subset.
The paper illustrates gloss annotation, but the locally downloaded release and the
publisher's public download surface do not supply the authentic gloss file required by
this project. The paper is evidence that the annotation modality was used; it is not
provenance for any independently obtained label file.

### 2.2 Pseudo-gloss and gloss-free research

Guo et al. (2025) propose LLM draft glosses followed by weak video-guided ordering and
CTC-based learning. Their reported How2Sign setup uses few ASL examples and still has
high pseudo-gloss error. This is evidence that candidate generation is possible, not
that generated labels are authentic or linguistically reliable.

Sign2GPT uses pretrained vision and language components and pseudo-gloss-like
intermediate supervision. GFSLT-VLP, SignDINO, UniGloR, SAGE, and FVLF instead explore
gloss-free visual-language pretraining, visual tokens, segmentation, or cross-modal
alignment. These works establish viable research alternatives, but adopting them as the
primary runtime would be an architectural change. They also do not remove the need for
source-disjoint evaluation, leakage controls, and qualified human assessment.

Spoken-language glossification research shows that limited parallel examples and
monolingual targets can help generate structured sign-language notation. That result
does not establish transfer to ASL, How2Sign, or this project's annotation convention.
Notation, vocabulary, morphology, word order, and non-manual representation remain
dataset- and language-specific.

### 2.3 Conclusions supported by the literature

1. Text alone can create a linguistic prior, but cannot identify which forms were
   actually signed in a particular video.
2. Video evidence can constrain timing and ordering only if intervention tests prove
   that the model uses it.
3. Gloss-free visual representations are useful baselines and possible future paths,
   not evidence that a pseudo-gloss is correct.
4. BLEU or translation accuracy against English cannot certify gloss correctness,
   because the transcript can leak the target semantics into both labels and metrics.
5. Independent qualified-ASL references are required for linguistic validation.

## 3. Options and recommendation

| Approach | Strength | Fundamental limitation | Decision |
|---|---|---|---|
| From-scratch text-to-gloss | Full local control | Authentic paired supervision is absent; training would learn unverified conventions | Reject now |
| Rule/lexicon transduction | Deterministic, auditable, bounded vocabulary | English-to-ASL is not word replacement; weak handling of morphology, classifiers, reference, and non-manuals | Retain only as a baseline/candidate source |
| Constrained pretrained LLM | Strong text prior and broad lexical coverage | Hallucination, prompt injection, unknown training provenance, transcript-order bias | Eligible only as a sandboxed lattice generator |
| Hybrid text candidates + video evidence | Can combine semantic prior with observed timing/order | Identifiability and circular-learning risks; needs a verified reference set | Recommended research direction after approval |
| Fully gloss-free SLT | Avoids discrete gloss bottleneck | Changes the requested architecture and can hide alignment errors in latent features | Baseline only; no architecture change now |

The recommended future direction is a **constrained hybrid candidate lattice**, not a
single decoded pseudo-gloss. Text proposes a finite, schema-valid candidate set; video
supplies independently trained evidence for feasibility and order; the system abstains
when evidence is insufficient. It remains weak supervision until human correction.

## 4. Formal candidate-lattice model

Let `X` be the English transcript, `V` the signed video or audited visual features, and
`G = (g_1, ..., g_L)` a candidate sequence over a versioned finite lexicon
`A` plus `UNKNOWN`. For a finite allowed lattice `L(X)`, define

\[
q(G\mid X,V)=\frac{
q_\psi(G\mid X)^\alpha
p_\phi(G\mid V)^\beta
\exp[-\lambda C(G,V)]
}{Z(X,V)}, \qquad G\in\mathcal L(X),
\]

where

\[
Z(X,V)=\sum_{G'\in\mathcal L(X)}
q_\psi(G'\mid X)^\alpha p_\phi(G'\mid V)^\beta
\exp[-\lambda C(G',V)].
\]

- `q_psi(G|X)` is a transcript-conditioned proposal distribution. It is a prior, not
  a label.
- `p_phi(G|V)` is video evidence computed without access to `X` or transcript-derived
  candidates during its feature extraction and calibration.
- `C(G,V) >= 0` contains declared hard/soft violations such as impossible CTC length,
  unsupported order, or excessive duration. A term is allowed only if its units and
  validation are specified.
- `alpha`, `beta`, and `lambda` are fixed on a development reference set. They are not
  tuned on the held-out evaluation set.
- `Z` is computed over the retained finite lattice. Dropped mass must be reported; a
  top-k beam is not the full distribution.

Hard constraints use `C=+infinity` or remove a candidate before normalization. Soft
penalties must not masquerade as probabilities. If either model assigns a raw score
rather than a normalized probability, the score must be named and calibrated
separately.

### 4.1 Identifiability limits

`G` is not identifiable from `(X,V)` without assumptions. Several gloss sequences can
express similar meanings; video features may omit fingers, face, or spatial reference;
the transcript may paraphrase; and the gloss convention may collapse morphology.
Scaling `alpha`, `beta`, and `lambda` can also produce the same ranking. Therefore:

- the posterior is a model-relative candidate distribution, not a posterior over true
  ASL;
- the lattice must retain alternatives and abstention;
- no weight is interpretable as trust without calibration against independent labels;
- video dependence must be falsified by interventions, not inferred from architecture.

## 5. CTC alignment and feasibility

For video features of length `T`, CTC introduces the alphabet
`A' = A union {blank}` and paths `pi in (A')^T`. Let `B` remove blanks and collapse
consecutive duplicate tokens. The exact likelihood is

\[
P_\phi(G\mid V)=
\sum_{\pi\in B^{-1}(G)}\prod_{t=1}^{T}p_\phi(\pi_t\mid V).
\]

The sum must be evaluated by the forward-backward dynamic program in log space. Greedy
collapse is not the training likelihood. Let

\[
r(G)=\sum_{i=2}^{L}\mathbf 1[g_i=g_{i-1}].
\]

Then an alignment is possible only if

\[
T \ge L+r(G),
\]

because adjacent repeated labels require an intervening blank. The check uses the
actual encoder output length after temporal subsampling, not raw frame count. Zero
likelihood, non-finite forward values, or infeasible length must reject the candidate;
the system may not shorten or deduplicate it silently.

CTC imposes monotonic order. It cannot represent arbitrary reordering between video and
candidate sequence, and it does not prove that a high-probability alignment corresponds
to linguistic sign boundaries. Alignment entropy and blank dominance must be reported.

## 6. Noisy-label objectives and why they are dangerous

For a retained candidate set `L_i` with normalized candidate weights `w_i(G)`, a future
multi-candidate objective could be

\[
\mathcal J_i(\theta)=-\log\sum_{G\in\mathcal L_i}
w_i(G)P_\theta(G\mid V_i).
\]

A confidence-weighted version,

\[
\mathcal J(\theta)=\frac{\sum_i c_i\mathcal J_i(\theta)}
{\sum_i c_i},
\]

is permissible only when `c_i in [0,1]` is calibrated against independent human
references. LLM token probability, beam score, OpenPose confidence, candidate agreement,
and CTC likelihood are not interchangeable confidence measures.

The objective is circular if the same video model creates `w_i` and is then trained to
match those weights. It can amplify confirmation bias, vocabulary collapse, and
transcript order. At minimum, candidate creation and evaluation require frozen models,
disjoint source groups, separately logged versions, and a human-authored reference set.
Cross-fitting reduces direct self-training leakage but does not make pseudo labels true.

## 7. Abstention and selective risk

Let `s(X,V)` be a separately calibrated acceptance score and `tau` a threshold fixed on
development data. Coverage and selective risk are

\[
\operatorname{coverage}(\tau)=P[s\ge\tau],\qquad
R(\tau)=E[\ell(\hat G,G^*)\mid s\ge\tau].
\]

The complete risk-coverage curve must be reported. Selecting `tau` to minimize held-out
test error is prohibited. Abstention occurs when the lattice is empty, CTC is
infeasible, posterior mass is diffuse, the candidate contains `UNKNOWN`, security
validation fails, or calibrated risk exceeds the predeclared operating bound. Abstained
records remain abstained; there is no fallback to English word order.

Calibration is evaluated with held-out reliability diagrams, expected calibration
error with declared bins, Brier/log loss where applicable, and subgroup/source slices.
Calibration error itself carries uncertainty and is not proof of correctness.

## 8. Failure mechanisms to measure

- **Transcript leakage:** translation metrics can be high even when video is blank.
- **Circular learning:** a model validates labels generated from its own representation.
- **Confirmation bias:** early errors receive higher confidence after self-training.
- **Vocabulary collapse:** frequent generic tokens displace rare or classifier forms.
- **Order hallucination:** English order survives despite incompatible visual order.
- **Candidate deletion:** the correct sequence never enters a small beam.
- **Non-manual loss:** face, mouth, head, and body grammar are absent from lexical tokens.
- **Fingerspelling loss:** names are copied from English without observing handshape.
- **Reference collapse:** loci, directionality, and classifier constructions are flattened.
- **Annotation-convention drift:** model updates silently change token spelling or scope.

Each mechanism needs its own metric and review stratum. Aggregate BLEU cannot absorb
these failures.

## 9. Required falsification tests

These tests are mandatory before any weak label can be used for training:

1. **Text-only:** replace video evidence with its marginal/default value. Performance
   close to the full system proves video non-use.
2. **Blank-video:** preserve dimensions and timestamps but zero visual evidence using a
   declared mask. Accepted outputs must fall sharply.
3. **Shuffled-video:** permute videos across transcripts within duration strata. This
   tests semantic rather than length leakage.
4. **Order corruption:** reverse or segment-shuffle frames while preserving frame-level
   content. Ordering evidence must deteriorate.
5. **Candidate deletion:** remove the top or human-reference candidate and measure
   abstention rather than forced selection.
6. **Vocabulary holdout:** reserve lexical families, fingerspelling patterns, and rare
   constructions; do not let aliases leak across folds.
7. **Source holdout:** partition by official `VIDEO_ID`. This is source-disjoint, not
   signer-disjoint until authoritative signer mapping exists.
8. **Counterfactual sentences:** pair minimally different transcripts with the same
   video and the same transcript with mismatched videos.
9. **Human reference:** compare candidate recall, order, omissions, insertions,
   non-manuals, spatial reference, and acceptability against independent qualified-ASL
   annotations.

All randomization is seeded and persisted. Each test declares a null, effect measure,
uncertainty interval, and stopping rule before evaluation. A system that fails a
video-dependence intervention is a text generator and must be labeled as such.

## 10. Implemented provenance schema

The existing `gloss_tokens` field remains reserved for labels that pass the current
governed exporter. A future, separate candidate record should minimally contain:

```text
annotation_id, source_sample_id, source_video_sha256,
label_type, review_status, lexicon_id, convention_id,
candidate_tokens, candidate_log_score, candidate_rank,
transcript_sha256, visual_feature_sha256,
generator_model_id, model_weight_sha256, tokenizer_sha256,
prompt_or_template_sha256, decoding_config_sha256,
code_revision, random_seed, created_at,
human_annotator_pseudonym, human_review_protocol,
review_attestation_sha256, reviewer_qualified_asl, source_video_reviewed,
parent_annotation_ids, limitations, environment_sha256
```

`label_type` is a closed enum:

- `official_human`: obtained from the dataset authority with verified source mapping;
- `project_human`: newly authored from video under the project's protocol;
- `human_corrected_pseudo`: machine candidate corrected and approved while viewing the
  source video;
- `unreviewed_pseudo`: machine-only candidate, never represented as ground truth.

Missing provenance is an error, not `unknown`. `human_corrected_pseudo` must preserve
its machine parent rather than being relabeled `project_human`. The governed exporter
uses an explicit allowlist and never silently promotes a label type.
Human-authored and human-corrected records require a bound review-attestation hash,
qualified-ASL status, and an affirmative source-video-viewed record; a reviewer name
and protocol string alone are insufficient.

## 11. Security and privacy threat model

### 11.1 Threats

| Threat | Failure | Required control |
|---|---|---|
| Prompt injection in transcript | Data changes instructions, exposes prompt, or escapes schema | Treat transcript as inert length-bounded data; no tools; structured separation; adversarial corpus |
| Arbitrary-token escape | Model emits prose, URLs, commands, or tokens outside the ASL lexicon | Grammar/schema-constrained decoding and exact token allowlist |
| Unicode attack | Confusables, bidi controls, invisible characters, or normalization collisions alter IDs/tokens | Decode strict UTF-8; record original bytes; validate NFC plus UTS #39 confusable/restriction checks; reject controls |
| Oversized input | Memory, latency, or cost exhaustion | Byte, code-point, token, candidate, and output limits before inference |
| Poisoned examples | Retrieved demonstrations steer conventions or embed instructions | Immutable curated examples, signed review, source hashes, no open retrieval |
| Model/dependency substitution | Different weights or code generate labels under the same name | SHA-256 weight/tokenizer/config hashes, lockfile, SBOM, verified provenance, offline verification |
| Data exfiltration | Hosted model retains transcripts or media | Local inference by default; deny network; no tools/secrets; documented retention and deletion |
| Identity inference | Face/voice/video used to infer signer or sensitive traits | No identity objective/API; access controls; privacy review; existing non-inference guard |
| License violation | Data or derived labels are redistributed or used commercially | Action-scoped authorization gate and derivative/redistribution review |
| Unsafe hosted inference | Provider policy or model changes without notice | Prohibit by default; require data-processing terms, region/retention review, immutable model version, and incident plan |

Input sanitization cannot prove prompt-injection immunity. Security therefore depends on
capability removal: the generator has no network, tools, filesystem write access, or
secrets; its only accepted output is a bounded token lattice. Generated strings are
never executed or interpreted as paths, HTML, SQL, or shell commands.

### 11.2 Fail-closed controls

1. Validate source identity and hashes before inference.
2. Enforce strict UTF-8, normalization policy, prohibited controls, and maximum sizes.
3. Pin a versioned ASL lexicon and annotation convention; unknown forms map to explicit
   `UNKNOWN` or abstention, never a guessed nearest token.
4. Constrain decoding to the schema and lexicon; reject trailing text and duplicate
   keys.
5. Enforce output-length, candidate-count, and exact CTC-feasibility bounds.
6. Hash model weights, tokenizer, prompt/template, examples, decoding config, code, and
   environment; verify before and after the run.
7. Require deterministic decoding for corpus generation. If an operation remains
   nondeterministic, record and bound it rather than claiming byte reproduction.
8. Store append-only audit events and parent/child annotation provenance.
9. Render review text with escaping and safe links; never execute generated markup.
10. Stop the batch on schema/security drift. Do not repair output with another model.

NIST AI RMF controls support lifecycle risk ownership, evaluation, third-party incident
planning, and component-risk management. OWASP's prompt-injection guidance supports
instruction/data separation, validation, least privilege, monitoring, and human review.
Unicode UTS #39 defines confusable-detection mechanisms. SLSA provenance supplies a
useful supply-chain vocabulary, but an asserted SLSA level does not establish model
quality, producer trust, or dependency safety.

## 12. Architecture-preserving integration

The current active boundary remains unchanged:

```text
authentic or independently reviewed gloss
    -> governed exporter (`gloss_tokens`)
    -> active training pipeline
```

The implemented research branch adds this parallel boundary:

```text
English transcript -> constrained text lattice -------+
                                                     +-> scored weak_gloss_candidates
audited video -> independent visual/CTC evidence -----+       |
                                                             v
                                             abstain or qualified human review
                                                             |
                                            human-corrected governed annotation
```

The parallel record is named `weak_gloss_candidates`, not `gloss_tokens`.
`promote_reviewed_weak_candidate` is the only promotion boundary: it requires a
qualified-human-corrected record, exact sample/media binding, approved review status,
and reviewer/protocol provenance. The exporter independently rechecks that binding.
`unreviewed_pseudo` remains outside the production loader. No code path converts an
English sentence into gloss by field assignment.

The video encoder for candidate scoring must not reuse a model trained on the same
pseudo labels without cross-fitting and explicit circularity analysis. The current 2D
OpenPose experiment is insufficient for this role: it was a small reconstruction model,
not a sign recognizer, and performed below interpolation on its principal point/span
baselines.

## 13. Validation hierarchy and stop rules

| Level | Evidence | What it permits |
|---|---|---|
| L0 — deterministic safety | Schema, lexicon, hashes, CTC feasibility, reproducibility, hostile-input tests | Candidate-generation testing only |
| L1 — mechanical validity | Finite scores, bounded outputs, candidate diversity/recall, no leakage in IDs | Offline research artifacts only |
| L2 — video reliance | Blank/shuffle/order/counterfactual interventions show predeclared degradation | Claim that video affects selection, not correctness |
| L3 — generalization | Source-held-out evaluation; signer-held-out only after authoritative mapping | Generalization claim within measured scope |
| L4 — linguistic reference | Independent qualified-ASL annotation and blinded evaluation | Calibrated weak-label use under the reviewed protocol |
| L5 — downstream utility | Ablation against no-pseudo, rule, text-only, and gloss-free baselines on untouched data | Limited empirical utility claim |

Stop immediately if any of the following occurs:

- pseudo labels enter `gloss_tokens` without an approved provenance policy;
- a source or signer crosses its declared split;
- transcript-only or shuffled-video performance is not materially worse under the
  preregistered test;
- CTC infeasibility, non-finite scores, schema escape, hash drift, or Unicode ambiguity
  is silently corrected;
- the human reference set is used to generate the candidate it evaluates;
- candidate recall is inadequate for a declared high-stakes construction;
- qualified reviewers find systematic order, classifier, spatial, fingerspelling, or
  non-manual failures;
- license, consent, hosted-processing, or identity-risk scope is unresolved.

No model may be called linguistically validated without an independent qualified-ASL
reference set. Translation metrics, automatic gloss WER against another pseudo system,
or successful optimization do not satisfy L4.

## 14. Pre-registration and statistical requirements

Before real-data activation, define the annotation convention, target phenomena, primary
endpoint, equivalence/non-inferiority margin if used, error taxonomy, strata, and
analysis plan. Reference-set size must follow a power or precision calculation tied to
the primary endpoint and expected clustering by source/signer; choosing an attractive
round number is not a sampling justification.

Report paired uncertainty for candidate recall, token/order error, abstention coverage,
selective risk, and downstream deltas. Resampling must preserve source groups. Multiple
thresholds or variants require a preregistered primary comparison or multiplicity
control. The untouched test set is evaluated once after all design choices freeze.

## 15. Decision and current gate

The implementation study was approved and the architecture has been built. Corpus-wide
generation and training use remain unapproved until the external evidence below passes
the executable activation gate.

Before real-data activation or training, obtain:

1. a versioned ASL gloss convention and closed starting lexicon reviewed by qualified
   ASL expertise;
2. an independent human reference set whose size and strata follow the preregistered
   precision analysis;
3. authoritative signer mapping if a signer-generalization claim is intended;
4. a documented local model/weight/license choice and reproducible supply-chain record;
5. an approved label-provenance policy and review workflow;
6. predeclared falsification thresholds and stop rules.

Calibration fitting and held-out evaluation require separate source-disjoint human
reference artifacts and separate qualified-review attestations. The fit set may not be
reused as evidence of calibration quality.

`python -m signtranslator.pseudo_gloss assess-readiness <charter.json>` verifies exact
artifact hashes and schemas, checkpoint and license-evidence bindings,
action-scoped dataset authorization for derivative creation and model training,
qualified-reference independence, source-disjoint calibration fit/evaluation sets,
held-out ECE/Brier/log-loss evidence with source-cluster uncertainty, calibrator/reference
binding, the exact frozen decoding policy, all nine hash-bound falsification results,
review/provenance policy, dependency lock, SBOM, and optional authoritative signer
mapping. The bundle's training-manifest hash is bound to an inspected `VIDEO_ID` source
manifest, and its local training groups must be disjoint from both calibration-fit and
held-out human-reference groups. Failure is explicit and returns a nonzero status.

If these prerequisites cannot be met, continue pursuing authentic How2Sign annotations
or commissioned human annotation. Do not create pseudo-ground-truth to make Stage B
appear complete.

## 16. Implementation evidence

- `contracts.py`: closed lexicon, separate human and pseudo provenance, immutable weak
  candidate records, and exact label/review state restrictions.
- `security.py`: strict UTF-8/NFC and English-script policy, size limits, duplicate-key
  rejection, and non-finite JSON rejection.
- `mathematics.py`: differentiable exact CTC forward probability, exact repeated-token
  feasibility, path entropy from posterior occupancies, finite log-linear lattice
  normalization, certificate-gated multi-candidate marginal loss, calibrated-confidence
  weighting, and selective risk. CTC class IDs reject boolean or fractional coercion.
- `model.py`: positional text Transformer with lexicon-constrained beam search and
  explicit dropped probability mass; transcript-independent 137-node graph-temporal
  video CTC evidence.
- `training.py`: approved-human-only targets, prohibition of from-scratch text training,
  exact loaded-state/pretrained-checkpoint binding, source-disjoint partitions,
  cross-fit assignments, anti-self-training lineage checks, an exact multi-candidate CTC
  optimization path, finite-gradient enforcement, and text/video optimization paths.
- `calibration.py`: qualified-reference-gated ridge logistic calibration with damped
  Newton optimization, reliability bins, ECE, Brier/log loss, source slices,
  source-cluster bootstrap uncertainty, and fail-closed abstention.
- `pipeline.py`: frozen hybrid inference, exact candidate fusion, provenance hashes,
  explicit `UNKNOWN`/diffuse/uncalibrated/CTC/no-video abstention, and falsification
  interventions.
- `artifacts.py` and `cli.py`: transactional model/candidate bundles, checkpoint and
  license-evidence hashes, hash-chained records, exact checkpoint reload verification,
  exact runtime-environment provenance, before/after input mutation checks, no media
  copying, exact dataset-authorization binding, and offline CLI.
- `inference.py` and `corpus.py`: one-load stable-input inference plus resumable,
  activation-gated corpus generation with exact manifest/checkpoint reconciliation and
  no media duplication.
- `evaluation.py` and `readiness.py`: source-clustered uncertainty, required
  falsification specifications and result hashes, vocabulary-family holdout, deterministic
  shuffled-source construction, token/order/omission/insertion and phenomenon-stratified
  human-reference metrics, and external activation gates.

Synthetic and adversarial tests prove the software contracts; they do not satisfy the
missing qualified-ASL evidence or establish linguistic accuracy.

## 17. Primary and standards references

- Duarte et al., [How2Sign: A Large-Scale Multimodal Dataset for Continuous American
  Sign Language](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html), CVPR 2021.
- Guo et al., [Bridging Sign and Spoken Languages: Pseudo Gloss Generation for Sign
  Language Translation](https://arxiv.org/abs/2505.15438), 2025.
- Wong et al., [Sign2GPT](https://arxiv.org/abs/2405.04164), 2024.
- Zhou et al., [Gloss-Free Sign Language Translation: Improving from Visual-Language
  Pretraining](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html), ICCV 2023.
- Hwang et al., [A Spatio-Temporal Representation Learning as an Alternative to
  Traditional Glosses](https://openaccess.thecvf.com/content/WACV2025/html/Hwang_A_Spatio-Temporal_Representation_Learning_as_an_Alternative_to_Traditional_Glosses_WACV_2025_paper.html), WACV 2025.
- Low et al., [SAGE: Segment-Aware Gloss-Free Encoding for Token-Efficient Sign
  Language Translation](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.html), ICCV Workshop 2025.
- Gan et al., [Learning Effective Sign Features without Text for Gloss-free Sign
  Language Translation](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html), CVPR 2026.
- Yin et al., [Semi-Supervised Spoken Language Glossification](https://aclanthology.org/2024.acl-long.504/), ACL 2024.
- Graves et al., [Connectionist Temporal Classification: Labelling Unsegmented
  Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), ICML 2006.
- Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017.
- Cattelan and Silva, [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v244/cattelan24a.html), UAI 2024.
- NIST, [Artificial Intelligence Risk Management Framework: Generative AI Profile
  (NIST AI 600-1)](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence), 2024.
- OWASP, [LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).
- Unicode Consortium, [UTS #39: Unicode Security Mechanisms](https://www.unicode.org/reports/tr39/).
- SLSA, [Provenance, specification v1.2](https://slsa.dev/spec/v1.2/provenance).
