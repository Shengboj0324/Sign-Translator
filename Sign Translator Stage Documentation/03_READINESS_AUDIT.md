# 03 — Strict Readiness Audit

## Refreshed W1/W2 acceptance audit and source inventory — 2026-09-26

All 15 full W1/W2 requirement rows were reassessed after the rig, rest-shape, native-method,
render and translation work. Their partial implementation evidence has advanced, but none
of the complete rows is yet proven accepted: real source/calibration, qualified review,
eligible population and statistical protocol dependencies remain. This is not a claim of
zero engineering progress. The [refreshed matrix](evidence/w1-w2-2026-09-26/exit-audit-refresh/requirements.json)
replaces the older matrix as the current snapshot while retaining the older evidence.

`CURRENT_SOURCE_CANDIDATES` previously described MakeHuman as publicly downloadable with
publisher-claimed capabilities. Its access/body/hand/face/eye capability snapshot now reflects
local acquisition and execution. The limitation explicitly excludes live file verification,
qualified contact/linguistic acceptance, native full-application parity, blanket community
asset rights and action authorization. No rights status was broadened and no human approval
or authorization object was fabricated.

The legacy capability/rights snapshot now passes its `commercial_render_rig` subcheck;
its overall gate remains closed. That legacy name does **not** establish permission for
an actual commercial action. Both current action-scoped portfolios still reject all four
bundles with the supplied empty authorization inventory, including render rig. The test
that asserted every legacy subcheck failed was stale; its initial failure was diagnosed
and the expectation updated to the exact one-subcheck advance. All 25 schema/policy tests
pass with warnings as errors. Compilation and diff checks pass; the latest full model
regression remains 2,035 passes, not a new full run for this inventory change.

A fresh filename-scoped search for agreement, authorization, consent and reviewer records
in the checkout and `/Users/jiangshengbo/Volumes` found no matching files. This is a bounded
inventory check, not proof about other filenames, formats, locations or external accounts.
Further accepted end-to-end implementation requires the actual eligible motion sample and
its channel/clock/calibration specification, bound permission evidence, qualified review
capacity and accepted independent-unit/study protocol. Static probes cannot supply those
facts. Full W1/W2 acceptance remains unproven and the goal remains active.

---


## Bone-local translation and canonical-state boundary — 2026-09-26

The previous turn integrated rebuilt rest geometry into rendering. Inspection of the
canonical-state boundary found that state includes head translation while the rig accepted
only bone-local rotations and one whole-body transform. `MakeHumanRig.pose` now accepts
optional `(bones,3)` translations with strict finite/dtype/device checks; attachment posing
forwards the same values. Omission retains exactly the rotation-only API semantics.

The declared composition is `global_child = global_parent @ relative_rest @ [R,t]`.
Translation is in that bone's rest-local axes and unchanged source units. It is neither a
world joint position nor automatically the canonical head-translation channel. Own rotation
does not rotate its own translation again; ancestor rotations carry the translation once.
World/root motion is applied once. Meter conversion and source-frame mapping remain explicit
adapter responsibilities. Nonzero translations can stretch the rig and are not biomechanical
permission or an automatic head-motion interpretation.

Tests compare head displacement against an independent descendant-weight skinning formula,
its analytical gradient, parent-rotation transport and whole-world equivariance. Wrong shape,
dtype and nonfinite values fail. An attachment oracle verifies eyes follow the translated
head hierarchy. Existing zero-translation and rest-shape tests remain green. Full warning-
strict regression: **2035 passes in 69.87 seconds**, no failures,
errors or skips; compilation and diff checks passed.

A current [11-channel boundary audit](evidence/w1-w2-2026-09-26/local-translation/boundary-audit.json)
records the exact remaining mappings for root/body/hands/head/face/gaze/blink/contact.
There is still no accepted complete canonical-state-to-rig adapter. Arbitrary source frame
or convention strings are not transformations. Source observations, joint correspondence,
rest bases, head position-versus-displacement, facial basis/combination semantics, eye
calibration, blink timing and contact labels must be bound explicitly. Missing required
channels cannot silently become static bind controls, and gaze/contact cannot be inferred
from body rotations without declared accepted inference.

[Verification](evidence/w1-w2-2026-09-26/local-translation/verification.json) binds the changed
runtime/tests and regression evidence. This closes a mathematical API capability gap while
preserving the real-source integration requirement. W1/W2 remain active and unaccepted.

---


## Rebuilt rest geometry integrated into rendering — 2026-09-26

The render CLI now accepts paired `--rest-mesh-npy` and `--rest-mesh-sha256`. It verifies
SHA-256 against the same bytes decoded with pickle disabled, requires a float64 NPY array,
and delegates shape/finiteness/frame reconstruction to the rig loader. The output JSON
and saved body object retain the rest-file identity and an explicit unaccepted-provenance
label. A hash is identity, not source/anatomical acceptance. Existing raw-base use remains.
Five CLI checks rejected wrong dtype, shape, nonfinite values, unpaired options and wrong
hash before creating output directories. The first launch lacked PYTHONPATH and failed
before rendering; the corrected command and environment requirement are recorded.

The changed rest mesh now has a composed body/eyes/teeth/tongue JawDropStretched render.
All initial body, hand, face and mouth views were inspected. Fixed historical hand cameras
cropped fingertips on the changed shape. The exporter now records anatomical body vertices
with summed left hand/finger skin support greater than .05; a sphere enclosing those
vertices in both frames determines one shared orthographic center and scale with margin.
This is an inspection-framing rule, not a physiological segmentation or occlusion guarantee.
The final hand image includes the fingertips. Projection checks confirm every selected
vertex lies strictly within the view in both frames after updating the camera dependency state.

All six final views replay with identical decoded pixels after a fresh saved-scene load,
independently confirmed by AV decoding. Rest/posed shape keys, body and attachment faces,
UVs, packed texture bytes, nominal units and rest-file properties survive reload. These are
editable baked shape-key diagnostic scenes, not native animated armatures or observed signs.
The first projection check used stale camera state and failed; its log is preserved, and
using the active view-layer update repaired the verification harness. No geometry was changed
for camera framing. Compilation and diff checks passed; no new full-suite run is claimed.

Evidence: [final render verification](evidence/w1-w2-2026-09-26/rig-render-rebuilt-framing/verification.json),
[editable scene](evidence/w1-w2-2026-09-26/rig-render-rebuilt-framing/rig-diagnostic.blend), and
[negative CLI checks](evidence/w1-w2-2026-09-26/rig-render-rebuilt-rest/cli-negative-checks.json).
The earlier cropped render is preserved as diagnostic history. Visible mouth opening and
reproducible appearance do not resolve measured contact or establish linguistic correctness.
W1/W2 remain unaccepted pending source, review, statistical design and rig qualification.

---


## Native rest-frame and FK method comparison — 2026-09-26

The previous goal turn implemented and tested rest-shape skeleton rebuilding. This turn
executes hash-verified upstream `get_normal`, `getMatrix`, `matrix.normalize`/`magnitude`
and `Bone.update` methods on both the raw and default-factor diagnostic rest meshes.
It compares all 163 frames and 61 static cases per mesh. Native inputs/frame arithmetic
use float32; an independent NumPy weighted contraction checks resulting skinning against
the adapter's float64 output. Both paths receive the same local rotations and the same
explicitly normalized adapter weights; native weight mapping and BVH decoding are not
independently tested by this particular comparison.

| Rest mesh | Largest rest-matrix entry difference | Largest posed coordinate difference, source units |
|---|---:|---:|
| Raw base | 1.330335e-5 | 2.454425e-6 |
| Default-factor probe | 4.797099e-6 | 2.474651e-6 |

The maximum posed coordinate difference is approximately 0.2475 micrometers at nominal
asset scale. Rest-matrix differences mix dimensionless rotation entries and source-unit
translations; do not interpret that matrix maximum as a distance or angular error.
These are measured discrepancies, not newly adopted anatomical acceptance tolerances.
The results support rest/FK convention agreement for these inputs; they do not constitute
full native application, default-character, shader, skin-weight, temporal or linguistic parity.
Mesh crossings and qualified-review requirements remain unresolved.

The first harness attempt incorrectly selected both a class and module function named
`get_normal` and stopped before comparisons. Module-level selection fixed the harness;
the failed attempt is preserved. An invalid-escape docstring warning from the unchanged
upstream source was emitted during AST parsing and retained in the successful log. No
runtime adapter code changed. Diagnostic compilation and diff checks passed; the prior
2,029-test full regression remains the latest full-suite result.

Evidence: [native-method comparison](evidence/w1-w2-2026-09-26/native-skeleton-parity-attempt2/verification.json).
The scripts require exact upstream hashes and never import the entire native application.
W1/W2 remain active and unaccepted; remaining work includes contact visibility/intendedness,
real calibrated source integration, accepted independent-unit design and qualified review.

---


## Explicit rest geometry and rebuilt skeleton — 2026-09-26

The rig loader now accepts an explicit fixed float64 CPU rest mesh in pinned vertex order,
topology and source units. It copies caller coordinates, validates shape/finiteness, rejects
trainable or degenerate inputs, and reconstructs all helper joint means and bone frames.
The original no-argument raw-base path is preserved. This is a geometry contract, not
verification of caller-supplied anatomical provenance. Old attachments cannot be rebound to
a changed rig identity; they must be refitted. Source weights remain unchanged explicitly.

Seven new regression cases cover anisotropic rest deformation with independent joint-head
and bone-axis checks, bind and whole-world rigid-motion oracles, caller mutation isolation,
wrong shape/dtype, nonfinite/trainable/degenerate geometry and stale attachment rejection.
The focused rig/eye/mouth/face suite passed 39 cases. The full warning-strict suite passed
**2029 tests in 73.39 seconds**, with zero failures,
errors or skips. Compilation and diff checks passed.

The preceding default-factor diagnostic mesh was loaded through this rebuilt-skeleton path,
not by replacing vertices on the old rig. All 163 frames were reconstructed; the maximum
bone-head movement was 0.61195755 source units. Bind skinning reproduces the new geometry
within 7.11e-15 source units. Fresh identity-bound eye/mouth attachments and face controls
successfully generated all 60 authored cases plus bind pose, followed by a fresh contact audit.

| Pair | Bind strict crossing pairs | Cases with crossings / 61 | Largest count |
|---|---:|---:|---|
| Body / eyes | 1,050 | 61 | 1,180; LeftEyeDown |
| Body / teeth | 0 | 7 | 366; JawDropStretched |
| Body / tongue | 0 | 2 | 278; TongueOut |
| Teeth / tongue | 832 | 61 | 1,161; TongueOut |

Body/teeth cases are lowerLipBackward, UpperLipBackward, UpperLipStretched, JawDrop,
JawDropStretched, MouthLeftPullUp and MouthRightPullUp. Body/tongue cases are TongueOut and
TongueDown. These results narrow contact triage; they do not authorize arbitrary geometric
correction or certify intended anatomy, detector completeness or linguistic fidelity.
Counts remain triangulation-dependent. No real observed motion, native full-application
parity, temporal-transition validation, or qualified review is inferred.

Evidence: [verification](evidence/w1-w2-2026-09-26/rebuilt-rest-rig/verification.json),
[rebuilt-rig measurements](evidence/w1-w2-2026-09-26/rebuilt-rest-rig/rig-measurements.json),
and [all contact cases](evidence/w1-w2-2026-09-26/rebuilt-rest-rig/contact-audit.json).
W1/W2 remain unaccepted. Next rig work is reviewed/visible contact classification and
native rest/pose parity; eligible source, statistical design and reviewer dependencies remain.

---


## Native rest-shape dependency isolated — 2026-09-26

The previous goal turn produced contact localization and native fitting-method parity.
This turn traced the pinned native startup: `mhmain.loadFinish` calls
`updateMacroModifiers()` then `applyAllTargets()`. `Human.applyAllTargets` resets the mesh,
applies the target stack, updates stored rest coordinates for proxy fitting, and updates
skeleton joint positions. Thus the raw OBJ used in the existing diagnostic rig is not
established as the native default character's post-modifier rest geometry.

Eight target files were acquired and verified against pinned Git blob identities and
SHA-256: six young ethnicity/sex macro targets plus two average muscle/weight targets.
The inspected default factors give each of the six macro targets weight 1/6 and the two
universal targets weight 1/2. These are upstream character-control categories, not measured
human biology or source-signer identity. A separate diagnostic rest reconstruction applied
those weighted deltas and refitted attachments. It does not run the full native application
or claim that no other native state contributes. No production geometry was replaced.
Maximum base-vertex shift is 0.9882746 source units (about 9.88 cm at nominal asset scale).

| Pair | Raw-base bind strict crossing pairs | Default-factor rest probe |
|---|---:|---:|
| Body / eyes | 868 | 1,050 |
| Body / teeth | 202 | 0 |
| Body / tongue | 13 | 0 |
| Teeth / tongue | 1,258 | 832 |

This controlled geometry change materially changes contact diagnostics. It does not certify
zero collision where the detector reports zero, and counts remain triangulation-dependent.
Persistent eye and teeth/tongue crossings still require localization and anatomical judgment.
The old skeleton was **not used for any posing of the changed rest mesh**. Adopting a morphed
character requires rebuilding all joint/rest frames and identity-bound attachments, then
native/parity, motion, contact and qualified-review checks; replacing vertices alone is not
a valid retargeting solution. The next rig task is an explicit identity-bound rest-shape
contract and rebuilt-skeleton probe, not arbitrary collision offsets.

Evidence: [target provenance](evidence/w1-w2-2026-09-26/native-rest-probe/provenance.json),
[contact comparison](evidence/w1-w2-2026-09-26/native-rest-probe/contact-audit.json), and
[verification](evidence/w1-w2-2026-09-26/native-rest-probe/verification.json).
B08/B47/B48/B69 remain partially unresolved. W1/W2 remain unaccepted; source and qualified
review dependencies are unchanged. This diagnostic execution is not a new full-suite run.

---


## Contact localization and upstream fitting-method parity — 2026-09-26

The previous turn made verified progress: 61-case contact evidence and 57 exact positive
certificates. This follow-on reproduces **20 pair/case counts across five selected poses**
and records every strict crossing triangle pair, original polygon indices and involved
vertex bounds. Bounds describe triangle vertices, not contact depth. Fifteen mouth-only
orthographic views and five editable Blender scenes were generated; bind front/side,
TongueOut front and JawDropStretched front were visually inspected. The body and eyes are
intentionally hidden in these views, so they do not establish visibility in the final avatar.

Tongue mapping references the base mesh's `helper-tongue` vertices exclusively; that helper
is not among rendered `body` faces. Duplicate tongue-helper rendering therefore does not
explain these crossings. Isolated views expose tooth/gum and tongue overlap, with different
positions under tongue-out and jaw-drop controls. Anatomical intendedness remains unjudged.

A separate diagnostic executes only `Proxy.getCoords` and `TMatrix.getMatrix`, extracted
from the already inspected upstream file after verifying SHA-256
`a83091c0677eb714b04f32e0be5e870241f135ecaf27ab935629319a6204a5ff`.
All eye/teeth/tongue coordinates match the adapter exactly with float64 inputs. Using
native-style float32 base coordinates, proxy coefficients and offsets, maximum per-coordinate
differences are 9.79281e-7, 1.36593e-6 and 2.37275e-7 source units respectively; the largest
vertex-distance discrepancy is approximately 0.137 micrometers at nominal asset scale.
The diagnostic tolerances are numerical checks, not anatomical acceptance limits.

This is **method-level parity on the same unmorphed base input**, not execution of the full
native application, native default-character modifiers, skin-weight/pose parity, subdivision,
shader equivalence or qualified anatomy. It narrows the cause: a large disagreement in
these fitting calculations does not explain the rest overlap. Do not erase intersections
by arbitrary offsets. Next investigate native default/morphed character geometry and intended
contact surfaces, then qualify corrections and transition behavior with appropriate review.
Historical records saying source was inspection-only remain true for their original runs;
this new record explicitly distinguishes the two methods now executed.

Evidence: [localization and verification](evidence/w1-w2-2026-09-26/contact-localization/verification.json),
[native-method results](evidence/w1-w2-2026-09-26/contact-localization/native-fitting-parity.json),
and [all selected crossing indices](evidence/w1-w2-2026-09-26/contact-localization/localization.json).
No model parameters or rig geometry changed. B08/B47/B48/B69 remain open to qualification;
source eligibility, statistical design and qualified-review dependencies still prevent W1/W2
acceptance. No full-suite rerun is claimed for this diagnostic-only addition.

---


## Current blocker disposition and contact audit — 2026-09-26

This current disposition supersedes the original B01–B66 wording below where repairs are
recorded. **Closed** means the specific engineering defect is repaired in its stated scope;
it never means the scientific or release phase passed. **Partial** retains the stated
acceptance dependency. Open findings describe this inspected checkout and evidence, not
unknown private artifacts. W0 is accepted; W1/W2 remain active and unaccepted. The latest
recorded full regression is 2,022 passes with warnings as errors; this diagnostic-only
update does not claim a new full-suite run.

| ID | Current disposition | Remaining requirement or completed scope |
|---|---|---|
| B01 | Open | No accepted co-observed continuous multichannel 3D source. Frontal 2D does not identify depth, palm twist, occluded fingers or gaze. Obtain and inspect an authorized source; preserve unavailable/inferred channels explicitly. |
| B02 | Open | No accepted qualified ASL annotation/review relationship in inspected artifacts. Assign a qualified creator, independent reviewer and disagreement adjudicator with real availability and evidence. |
| B03 | Open | No training-authorized governed text/video→SIR corpus. Populate and independently accept Phase-3B records; English transcripts and native EAF values cannot simply be renamed SIR. |
| B04 | Open | No human-reviewed lexical motion entries. Bind meaning, articulation, form, schema and exact motion bytes; retain unknown/ambiguous refusal. |
| B05 | Partial | Typed multichannel state, serialization and governed ingestion implemented. Still no accepted source-specific real shard, channel calibration or full round-trip. |
| B06 | Open | Research authorization for the final combined supervision/motion bundle is missing. Existing How2Sign research evidence is not absent, but it is not blanket authorization for every new source, annotation, derivative and asset. |
| B07 | Open | Commercial training/deployment permissions are not established. Maintain a separate commercial lineage and signed/action-specific evidence; this is a commercial release blocker, not a reason to prohibit otherwise authorized research. |
| B08 | Partial | Pinned body, eye, teeth, tongue and 60 static face controls acquired and rendered. Contact crossings now measured; native/morphed parity and qualified anatomical/linguistic acceptance remain open. |
| B09 | Partial | Action-scoped research/commercial policy implemented and adversarially tested. Real eligible evidence must be bound at ingestion and later training/export/release; existing rights are not broadened. |
| B10 | Partial | Narrow engineering pilot scope is recorded. Qualified linguistic coverage, usable source population and accepted empirical thresholds still require confirmation. |
| B11 | Partial | Connected-component splitter implemented and deterministic; final eligible population and certified signer/source-disjoint assignments remain unaccepted. |
| B12 | Open | Signer distribution is highly uneven: 12,102 and 14,596 available utterances belong to two codes, about 86% of the local 31,047 clips. Measure feasible grouped partitions, macro scores and subgroup uncertainty; clip count is not independent signer diversity. |
| B13 | Partial | 31,166 ledger rows dispositioned; 122 technical quarantines include 118 missing, 3 structural and 1 unjoinable artifact. Quarantine is implemented, not recovery or accepted QC. |
| B14 | Partial | 28,621 rows await qualified QC and 2,423 await acceptance; development-only thresholds, stratified hand/face/occlusion review and accepted exclusions remain missing. |
| B15 | Open | Decoder clock support exists, but final source synchronization/calibration is unvalidated. Test clip origin, frame rate, dropped frames, EAF milliseconds, motion units and face/audio offsets with real samples. |
| B16 | Open | Existing exporter explicitly requires source and gloss tokens. The gloss-independent roadmap has no direct governed SIR-to-training shard adapter. Implement a distinct typed path without fabricating gloss. |
| B17 | Closed engineering defect | Manifest-bound topology and joint order implemented and tested. |
| B18 | Closed engineering defect | Observed-support losses and required-empty-support rejection implemented. |
| B19 | Partial | Active ST-GCN, denoiser, shared-encoder and acoustic-prefix support repairs tested. Genuine missingness, canonical-source integration, interior acoustic gaps and source-specific policy remain unqualified. |
| B20 | Partial | Timestamp-aware velocity/acceleration and explicit gap exclusion implemented. Actual source clocks, gap/confidence policy, synchronization and physical calibration remain unaccepted. |
| B21 | Partial | Training-only vocabulary fitting and explicit unknown refusal implemented. Real frozen partition, coverage and preprocessing identity remain required. |
| B22 | Open | Shards and datasets are materialized in memory, with corpus-wide temporal padding at export. Measure peak RAM/I/O and adopt bounded shards/bucketing as needed before full-scale runs. This is a scale risk, not a measured OOM. |
| B23 | Closed engineering defect | Empty loaders/zero-step epochs rejected; positive optimizer-step evidence exists. |
| B24 | Closed engineering defect | Evaluation aggregates the declared macro-observation estimand with eligible counts. |
| B25 | Closed engineering defect | Selection metric and exact selected-best checkpoint identity are explicit. |
| B26 | Closed engineering defect | Analysis seed and caller RNG/mode restoration repaired. |
| B27 | Closed engineering defect | Ragged-safe references and length-aware analysis repaired. |
| B28 | Partial | Insertion-aware edit/EOS/truncation metrics repaired. Accepted governed semantic references and field-level scoring remain absent. |
| B29 | Open | Diagonal-only InfoNCE treats semantically equivalent off-diagonal pairs as negatives. Define governed multi-positive grouping and false-negative controls before applying to repeated real meanings. |
| B30 | Open | Joint weighted training starts all branches together; richer curricula are not active. Establish branch competence, gradient scale/flow diagnostics and controlled loss-weight ablations. |
| B31 | Closed engineering defect | Nonfinite loss/gradient abort occurs before optimizer mutation. |
| B32 | Partial | Epoch-boundary CPU/zero-worker continuation is supported. Accelerator, worker, distributed, mid-epoch and secondary-stage continuation remain unproven; implement only required paths. |
| B33 | Closed engineering defect | Bounded checkpointable CLI/API defaults reconciled. |
| B34 | Open | Runtime defaults to CPU and exposes no run-level device parameter. EMA, AMP and gradient accumulation are not integrated into the canonical trainer. Benchmark first; then add required hardware paths with parity/resume tests. |
| B35 | Closed engineering defect | Stable small-tail probability computation independently tested. |
| B36 | Closed engineering defect | Finite statistical input/configuration validation repaired. |
| B37 | Partial | Exact independent-unit resolution diagnosed and draft component aggregation specified. Accepted independence, estimand and real clustered evaluation remain missing. |
| B38 | Open | General bootstrap is IID and seed aggregation accepts one seed. Add appropriate signer/source/reviewer hierarchy and distinguish training-seed variation from population uncertainty. |
| B39 | Partial | Signed-effect/family checks and draft study design implemented. Population, useful effect, pilot-based power/precision, exclusions and accepted preregistration remain open. |
| B40 | Open | No trained direct paired learner and held-out four-intervention score artifact. Evaluator existence is not a passed experiment. Train only with accepted inputs and run all interventions. |
| B41 | Open | Existing 2D pretraining does not establish improvement over interpolation. Recompute on common support with coverage and clustered uncertainty; redesign only if a useful hypothesis survives. |
| B42 | Partial | Typed finger/body/head/face/gaze/blink/contact state and static rig controls exist. Active accepted source-conditioned generation and temporal facial grammar remain absent. |
| B43 | Open | Hand-graph, SIR, duration, spatial reference and non-manual modules are disconnected from active generation. Add typed adapters and intervention tests at each boundary. |
| B44 | Open | No qualified representation comparison among retrieval/interpolation, continuous AE and RVQ; no accepted codebook-use or reconstruction/minimal-pair evidence. Keep simple baselines eligible to win. |
| B45 | Open | Rich DiT/constraints/part-aware generation is separate from active diffusion. Choose one generator, test parameterization/schedule/guidance compatibility, inpainting masks and constraint residuals. |
| B46 | Open | Biomechanical/temporal corrections are not shown to preserve meaning. Test handshape, palm orientation, contact, continuity and non-manual scope before/after projection and chunk stitching. |
| B47 | Partial | Masked affine inverse, target-rig kinematics/LBS, nominal units and replayable static renders verified. Real source calibration/retargeting, contact repair and source-to-render acceptance remain open. |
| B48 | Partial | Editable textured eyes/mouth and selected authored face renders replay correctly. Contact, visible articulation, gaze/blink timing and qualified signer comprehension remain unaccepted. |
| B49 | Open | Cycle consistency uses a related recognizer and can reward shared shortcuts. Add independent baselines and blinded semantic comprehension; cycle WER remains diagnostic. |
| B50 | Open | No frozen multi-seed real held-out generation result, counterfactuals or ablations. Report per-channel failures, coverage and confidence intervals, not only total loss. |
| B51 | Open | Human comprehension, grammaticality and error severity remain unevaluated. Recruit independent reviewers, randomize/blind presentation, retain disagreements and preregister analysis. |
| B52 | Open | Confidence/calibration/refusal primitives are not connected to real semantic failures. Fit on held-out calibration data, measure risk-versus-coverage, test out-of-domain inputs and empty evidence. |
| B53 | Open | Raw audio backend/preprocessing is not active; `translate_audio_to_sign` receives acoustic features. Integrate waveform/sample-rate/channel contracts, real backend assets, tokenizer and timestamps. |
| B54 | Open | Synthetic noise/pitch conditions do not validate real accents, noise, code switching or long speech. Collect eligible real stress cases and characterize actual failure/abstention. |
| B55 | Open | Streaming revision/commit and backpressure contracts are independent primitives. Connect cancellation, committed-prefix immutability, stale-result dropping and temporal seams to the renderer. |
| B56 | Open | No measured first-output/end-to-end p50/p95/p99 latency, throughput, queue behavior, peak memory or target-hardware capacity. Benchmark the actual composed path before optimization promises. |
| B57 | Open | Quantization/static-execution algebra is not optimized production execution. Require numerical AND per-channel/linguistic parity; reject speedups that harm signing. |
| B58 | Open | No self-contained service/app bundle with authorization, retention, logging, model/data identity, failure reporting, rollback and load-tested lifecycle. Build and rehearse a restricted release. |
| B59 | Open | Hashes and typed attestations bind contents but do not authenticate qualifications, permission or observed truth by themselves. Verify issuers/documents and bind accepted evidence to the actual training run. |
| B60 | Open | `run.py` checks corpus structural readiness, not the complete source-portfolio/Phase-3/human release gates. Connect action-specific policy at training/export/release boundaries; do not treat its `passed` flag as project readiness. |
| B61 | Closed engineering defect | Canonical package ownership and legacy/research boundaries documented. |
| B62 | Partial | W0 archive-to-wheel installation and CPU reload smoke passed locally. Selected release platform, accelerators, portability and production data paths remain unverified. |
| B63 | Closed engineering defect | Historical defects retained as history; current engineering acceptance is explicit. |
| B64 | Open | Full ASL→English needs real-video perception, multichannel temporal understanding, an English decoder and independent evaluation. Current sign→gloss recognition does not complete the bidirectional goal. |
| B65 | Open | External acquisition/review turnaround, annotation throughput, compute throughput and budget are unmeasured. Assign owners and benchmark them; engineering days cannot guarantee external evidence arrival. |
| B66 | Partial | Software progress and empirical acceptance are now tracked separately. Neither W1 nor W2 is accepted; every subsequent empirical/release gate still needs its own evidence. |
| B67 | Open, source-specific | Exact Cokely primary HD media remain missing at the recorded binding. Recover the authorized exact payload or separately version/revalidate a new source and clock; never substitute the SD alternate silently. This does not block unrelated eligible sources. |
| B68 | Open, statistical design | Six candidate components leave at most four held-out components after train/validation reservation. The best possible two-sided sign-test p-value is 1/8; even all six yield 1/32, above .025 for two primaries. Obtain defensible independent units and pilot-based power/precision; clips/seeds do not solve this. |
| B69 | New measured geometry finding; qualification open | Static inter-mesh surface crossings occur in target bind and authored face controls. Localize visible versus hidden/intended surfaces, check native attachment/morph behavior, repair unintended intersections and recheck all poses plus temporal transitions before contact acceptance. This refines B08/B47/B48 rather than adding a separate full work package. |

### Contact evidence and limits

All 60 named static face poses plus target bind were audited for four inter-mesh pairs
(244 pair/case comparisons). Blender 5.2.2 LTS triangulated the meshes and supplied BVH
candidates. Float64 segment/triangle tests ran on Blender float32 coordinates. Five analytic
and nine metamorphic crossing checks passed. A separate exact-rational plane-intersection
and Gram-barycentric algorithm certified **57 selected positive crossings** across five
cases. This independently supports those positives; candidate recall is not certified.

| Mesh pair | Target-bind crossing pairs | Cases with crossings / 61 | Maximum crossing pairs; first pose at maximum |
|---|---:|---:|---|
| Body / eyes | 868 | 61 | 910; LeftUpperLidClosed |
| Body / teeth | 202 | 61 | 873; JawDropStretched |
| Body / tongue | 13 | 58 | 306; TongueOut |
| Teeth / tongue | 1,258 | 61 | 1,554; lowerLipUp |

Counts depend on triangulation and are neither penetration depth nor perceptual severity.
At dimensionless tolerance 1e-8, JawDropStretched body/teeth has 872 strict crossings plus
one unresolved candidate; at 1e-10 and 1e-12 it has 873 strict crossings. Other reported
counts are unchanged across those tolerances. Numerical tolerance is not a physical
clearance allowance. No self-intersection, enclosed volume, continuous-time contact,
anatomical correctness or linguistic acceptance is certified. Geometry may include hidden
or intentionally overlapping surfaces; positive findings require localization before repair.
Zero reported crossings cannot establish safety or complete collision freedom.

Evidence: [contact verification](evidence/w1-w2-2026-09-26/face-contact-audit/verification.json),
[all 61 cases](evidence/w1-w2-2026-09-26/face-contact-audit/contact-audit.json), and
[57 exact certificates](evidence/w1-w2-2026-09-26/face-contact-audit/exact-witness-verification.json).
The diagnostic and witness scripts, input mesh archive and runtime-module identities are
hash-bound in the verification record. No model or rig geometry was altered by this audit.

---

## W2 authored face controls — 2026-09-26

Pinned face-pose assets now map 60 named static controls to all 163 target bones. Euler
order/axis/rest-basis conversion has independent trigonometric checks; left/right upper-lid
closure and jaw-drop renders were inspected. The final left-lid probe replays identically
in all six decoded views. Full warning-strict tests: 2,022 passed. Source rest offsets and
small nonzero Rest rotations are preserved as explicit discrepancies, not hidden. Blink
dynamics, contact clearance, calibrated gaze, native runtime parity, real-source supervision
and qualified linguistic acceptance remain unverified; W1/W2 are not complete. See roadmap
and `face-pose-units/verification.json` for exact scope and the metadata-count repair.

## W2 mouth integration and nominal unit convention — 2026-09-26

The prior missing-mouth-interior finding is partially repaired with pinned teeth/tongue
assets, separate affine fitting and bone mapping, and visible inspected rendering. Actual
upper/lower tooth pivot and gradient oracles, direct tongue mapping and rigid invariance
pass. Six rendered views replay identically in decoded pixels from a saved Blender file.
The full warning-strict suite passed 2,015 tests. Native decimeter units and the 0.1 meter
conversion are verified against pinned exporter code and official documentation. Motion
calibration, full facial/blink/contact behavior, native runtime parity and qualified review
remain unaccepted. See roadmap and `mouth-attachment/` evidence; W1/W2 are not complete.

## W2 visible eye attachment — 2026-09-26

B08/B47/B48 advance with pinned, fitted and textured eye assets. Signed affine fitting
coefficients are kept distinct from nonnegative bone weights; actual-mesh rigid/pivot and
gradient oracles pass. Rest and explicit 0.15-radian eye-axis-tilt renders have visible
pupils and replay identically in decoded pixels from the saved same-machine Blender file.
The full warning-strict suite passed 2,010 tests. Gaze calibration, blinking, mouth interior,
native runtime parity and qualified linguistic review remain unaccepted. The prior empty
eye-socket finding is historical for the body-only render, not the current eye-augmented
probe. See the roadmap eye-attachment entry and evidence under `eye-attachment/`.

## W2 actual rendering and replay — 2026-09-26

B47 now includes the acquired mesh rendered through Blender with actual adapter-generated
articulation. Five inspected body/hand/face views replay with identical decoded pixels from
a saved `.blend` on this machine; PNG byte hashes differ due to runtime metadata. B08/B48
remain open with a concrete finding: visible eyeballs/pupils and mouth-interior assets are
missing from the body-only render. Eye bones and body-shell weights do not establish gaze.
The saved mesh is a shape-key diagnostic, not a qualified signing avatar, native rig parity
or observed motion. See the roadmap rendering entry and `rig-render/visual-qa.json`.

## W2 actual-mesh rig adapter — 2026-09-26

B08/B47 now have a pinned-asset loader and full-influence rest/pose/skinning adapter for the
acquired MakeHuman base mesh. Focused checks include real-mesh proximal/distal finger
motion and gradients against independent pivot oracles, whole-body transform application,
and rejection of invalid rotations, weights, hierarchy and asset identities. This does not
supply motion supervision, physical calibration, visible qualified rendering or reviewer
acceptance. W1/W2 remain active; see the roadmap's adapter entry and verification artifacts.

## W2 rig geometry evidence — 2026-09-26

B08/B47/B48 now have an independent float64 geometry audit of the pinned MakeHuman base
assets. All 163 rest frames are proper and nondegenerate; hierarchy/rest-skinning and a
known whole-body rigid-transform oracle pass with maximum coordinate error below 5.33e-15
source units. Native loader inspection explains raw non-unit weights through normalization.
The asset uses up to 12 influences, with 3,665 vertices exceeding four, so influence
truncation requires explicit fidelity evaluation. No native runtime, physical calibration,
render or signer qualification is inferred. See the roadmap's native-semantics subsection
and its reproducible audit; phase acceptance remains incomplete.

## W1/W2 public rig candidate acquired — 2026-09-26

A pinned MakeHuman base mesh/default skeleton/weights bundle is now local with upstream
licenses and file-integrity evidence. Structural inspection found valid references and an
acyclic hierarchy, but the source weight sums are not uniformly one; native normalization
and rest-frame semantics must be verified before skinning/rendering. This advances asset
acquisition under B08/B47/B48, without claiming rig qualification or phase approval. Motion,
linguistic reference and qualified review evidence remain missing. See the
[asset intake and next validation steps](06_IMPLEMENTATION_ROADMAP.md#public-rig-asset-intake-and-source-access-refresh--2026-09-26).

## Current W1/W2 exit audit — 2026-09-26

All 15 full W1/W2 requirement rows were checked against the current implementation and
acceptance record. None is proven complete at its entire required scope. Research and
commercial capability/scoped portfolio checks remain ineligible for all four required
bundles. Verified code repairs and the QC ledger are partial engineering evidence, not
accepted real-state, study or rendering evidence. The filename-scoped local asset search
found no matching rig/agreement files in the searched roots; its limitations are explicit.
See the [exit audit and remaining acceptance sequence](06_IMPLEMENTATION_ROADMAP.md#w1w2-exit-audit-against-current-evidence--2026-09-26).

## W1 source disposition and live technical refresh — 2026-09-26

B13/B14 now have an explicit hash-bound ledger covering all 31,165 metadata samples and
one separate unjoinable artifact. There are 122 logical technical quarantines, 28,621 rows
pending qualified QC and 2,423 rows pending acceptance. Every row remains unapproved for
training. Read-only source checks reproduced the 118 missing samples and three structural
failures; warning/valid rows were not re-decoded or human-approved. Original media, audit
files and mapping were preserved. See the [QC ledger and verification](06_IMPLEMENTATION_ROADMAP.md#w1-source-bound-qc-disposition-ledger--2026-09-26).

## W2 temporal gap-policy implementation — 2026-09-26

B20 now has explicit maximum-gap support in Cartesian derivative utilities and the active
physical-time velocity objective. A supplied cutoff excludes oversized intervals; acceleration
requires two retained intervals, and missing required velocity support raises rather than
passes as zero loss. The policy propagates through collation and the training/evaluation
routes. It is not a selected or accepted real-source cutoff. Physical calibration, reviewed
source-specific policy and its immutable profile binding remain open. See the
[temporal gap audit](06_IMPLEMENTATION_ROADMAP.md#temporal-gap-support-and-derivative-arithmetic--2026-09-26).

## W2 shared encoder and requirement reconciliation — 2026-09-26

Shared text/gloss and speech embeddings now reject empty evidence and exclude unavailable
payloads before projection/pooling. Masked NaN multiplication and empty-denominator fallback
are removed; true vocabulary/feature domains are checked. The older paired-training API now
propagates motion support and optional timestamps instead of leaving these fields outside
its interface. Current W1/W2 matrix entries distinguish these implemented repairs from
unaccepted source-specific transforms, canonical model integration and qualified evaluation.
B19 is not blanket-closed across unintegrated research packages or real missingness regimes.
See the [shared encoder audit and requirement matrix](06_IMPLEMENTATION_ROADMAP.md#shared-encoder-support-and-requirement-reconciliation--2026-09-26).

## W2 acoustic prefix support addendum — 2026-09-26

The active speech branch now excludes prefix padding from each convolutional stage and
Transformer attention, propagates exact ceiling-divided lengths into CTC, and trims decoded
emissions at the true endpoint. Excessive/noninteger/empty lengths are rejected instead of
clamped. Training/evaluation tests verify features, losses and gradients with NaN padding
across subsampling factors 1/2/4 and odd lengths. This repairs the active acoustic-prefix
portion of B19; interior source gaps, other model families and real-source phase acceptance
remain separate requirements. See the [speech support audit](06_IMPLEMENTATION_ROADMAP.md#acoustic-prefix-support-and-subsampling-audit--2026-09-26).

## W2 denoiser support addendum — 2026-09-26

The active pooled and cross-modal denoisers now mask unavailable motion keys throughout
self-attention and sanitize missing coordinates before projection. The actual diffusion
objective supplies that support; a caller cannot override it inconsistently. Cross-modal
conditioning now sanitizes masked/dropped NaN memory before key/value projection and
validates mask/drop contracts. Nonzero-output prediction and gradient tests cover both
training/evaluation and guided `eps`/`x0` objectives. B19 remains open for speech features,
other model families and real-source validation. See the [denoiser support audit](06_IMPLEMENTATION_ROADMAP.md#denoiser-attention-and-conditioning-support--2026-09-26).

## W2 active encoder support addendum — 2026-09-26

B19 is partially repaired in the active ST-GCN: masks now govern normalization moments,
convolutional sources/biases, block outputs and joint/frame pooling. Joint recognition and
alignment, validation, analysis and motion embeddings propagate support. Recognition decoding
excludes padded emissions, and declared CTC lengths must agree with frame support rather than
being silently clamped. Training/evaluation padding, invalid gradients and running-buffer
invariance are tested. The denoiser, speech branch and other model families still require their
own support audits; B19 remains open. See the [encoder implementation and limitations](06_IMPLEMENTATION_ROADMAP.md#st-gcn-support-propagation-and-active-recognition--2026-09-26).

## W2 coordinate inverse addendum — 2026-09-26

The active corpus affine transform now validates its domain and preserves unavailable
support through normalization and inversion, including zero gradients for invalid NaN/Inf
payloads. End-to-end exporter/dataset inverse checks exercise the same path used by loading.
This is a partial repair of B47, not a source-to-render implementation. The inverse returns
source coordinate units; 2D projection, discarded root/scale context, and temporal
resampling cannot be inverted without additional information. The source-specific calibrated
adapter, authorized rig, full training masking (B19), and qualified round-trip review remain
open. See the [coordinate-information audit](06_IMPLEMENTATION_ROADMAP.md#affine-inverse-and-coordinate-information-audit--2026-09-26).

## W1 statistical feasibility addendum — 2026-09-26

**B68 — finite independent-unit resolution for the current candidate sign-test design.**
The hash-verified How2Sign mapping has six connected signer/source components, not 31,165
independent observations. Requiring nonempty training and validation reserves leaves at most
four test components. If these are the independent non-tied paired sign-test units, even all
four favoring the model yield two-sided p=1/8. Thus this design cannot reach alpha=0.05.
Even all six components as test units have p >= 1/32, above the Bonferroni threshold 0.025
for two primaries. Independence remains an assumption to justify, not a property proved by
the component count. This does not rule out every alternative statistical design.

Closure: establish an admissible, sufficiently informative independent-unit sampling/design
plan before final evaluation; verify useful-effect precision/power and multiplicity under
that plan. More windows/clips or training seeds inside existing components are not additional
independent test participants. Owned by W1/W6 statistical design and qualified evaluation
roles; dependent on source acquisition, QC and accepted study protocol. The related B11,
B12, B37–B39 remain open. The blocker register now runs through B68.

The protocol's negative-effect acceptance defect is repaired: positive-improvement
registrations cannot confirm degradation or zero improvement. The general two-sided effect
magnitude helper remains available for uses that do not claim positive benefit.
See the [working study design and exact resolution evidence](06_IMPLEMENTATION_ROADMAP.md#statistical-design-audit-and-working-study-protocol--2026-09-26).

## Full W0 engineering acceptance — 2026-09-25

W0 Days 1–10 are complete against their engineering deliverables. The current
[execution record and metric registry](06_IMPLEMENTATION_ROADMAP.md#w0-days-610-execution-record--2026-09-25)
contains the exact scope and requirement mapping. **1,832 tests passed** with warnings as
errors and no skips. The extracted-archive installed-wheel smoke passed training,
selected-best analysis and seeded checkpoint reload; see
[verification](evidence/w0-complete-2026-09-25-attempt1/verification.json).

In addition to the Days 2–5 repairs below, B24 is repaired for canonical validation and
generator validation using macro-observation support; B25 now freezes branch-based selection
and identifies evaluated weights; B26 isolates analysis randomness; B27 crops ragged inputs
before inference; B28's token-level insertion/cap defects are repaired. Authentic semantic
field evaluation under B28 remains unavailable pending governed references, as documented
for W3/W6. Full training encoder masking (B19), physical-time derivatives (B20), real split
and vocabulary implementation (B11/B21), multi-positive contrast (B29), statistical study
design (B37–B39), the W1 gate-policy dependency (B09), and external blockers retain their
scheduled scope. W0 completion is not empirical or commercial approval. B67 remains open.

Older findings and execution records below are historical; this section and the full W0
record take precedence for the current engineering status.

## W0 Days 2–5 repair status — 2026-09-25

The current repair record is in `06_IMPLEMENTATION_ROADMAP.md`, preceding the frozen Day-1
baseline. B35/B36 (statistical domains/tails), B18 (active masked motion objectives), B17
(declared topology and checkpoint binding), B23/B31 (nonempty/finite training) and B33
(checkpointable defaults) are implemented. **Final verification: 1,816 passed, zero failures/errors/skips; warnings treated as errors.**
The installed-wheel custom-topology smoke completed four optimizer steps and finite,
bit-identical seeded checkpoint reload. See
[the final verification record](evidence/w0-days2-5-2026-09-25-final/verification.json).
The original findings below remain historical evidence of the pre-repair state.

The scope of B18 closure is the active training objective; length-aware analysis and complete
feature masking remain B19/Day 6/W2–W3. B20 physical-time derivatives, B24–B28 evaluation and
selection, and B37–B39 empirical statistical design are not closed by these repairs.
**Empirical readiness remains unapproved.** B67 remains open: no missing Cokely source file was
restored or replaced, and no historical source manifest was modified.

## W0 Day-1 refresh — 2026-09-25

The Day-1 task rechecked checkout `89e2238d371ae8e0af826282520f4c14416bd9c5`, which adds
only the previous audit/roadmap documents over the package/test baseline named below.
Fresh results and exact file identities are in
[evidence/w0-day1-2026-09-25/baseline.json](evidence/w0-day1-2026-09-25/baseline.json).
The [Day-1 execution record](06_IMPLEMENTATION_ROADMAP.md#w0-day-1-execution-record--2026-09-25)
contains the frozen scope, owners, dependencies, and source/reviewer requirements.

**New current blocker B67 — missing Cokely primary HD media (P0 for reuse of that source).**
The historical source-native manifest and binding require
`2_I_Have_a_Dream_720_CokelyAFSParallelCorpus_v1_0.mp4`, expected SHA-256
`8eb18b6b2f01a5a179100b0acd84a639b5f2a0ee95fb0195d477b946217c6bb3`, 650,953,008 bytes.
That file is absent under the recorded local source root
`/Users/jiangshengbo/Volumes/cokely_reference/v1/source/i_have_a_dream/`.
The SD alternate, EAF and publisher-page snapshot remain present. No reason for the absence
is inferred. The original one-item ingestion result is historical, not a currently
reproducible source-bundle guarantee. Source integrity fails even if other hashes match.

Closure: restore the exact authorized, hash-matching HD file, or create a separately
versioned and independently validated binding with explicit timing/source treatment.
Do not silently substitute SD, fabricate the primary hash, modify the historical manifest,
or use that broken binding for review/training. Ownership and due gate are D23 in the
roadmap. This raises the blocker register from the prior B01–B66 to **B01–B67**.

**Day-1 verification status: complete capture.** All 1,726 tests passed with zero
failures/errors/skips; compile/dependency checks passed. Eight synthetic optimizer steps
produced finite output and bit-identical seeded last-checkpoint reload. **Source integrity
FAILS:** eleven recorded files match and the Cokely HD primary is missing. No model or
statistical implementation was repaired; the five reproduced defects remain open.

## Implementation audit snapshot — 2026-09-25

This snapshot supersedes older historical verdicts below; the Day-1 refresh above takes
precedence for current source availability. Scope: checkout
`839acf3bc8d67fef10b0e777a181a086f83a66d8`, active runtime, specialized component
interfaces, tests, and saved evidence under `/Users/jiangshengbo/Volumes`.
This is a comprehensive implementation/readiness audit, not a proof that every line
is defect-free. “Absent” means no accepted artifact was found in these inspected
locations; private agreements and artifacts elsewhere were not inspected.

### Current implementation status

| Layer | Status and actual boundary |
|---|---|
| Phase 1 / Stage A | Substantial execution-integrity implementation: opt-in corpus generation, exact CTC feasibility, schema-v2 checkpoints, best/last separation, epoch-boundary CPU resume, isolated validation RNG, archive CI. Remaining integration/statistical defects are below. |
| Active executable | `run.py` → `models/pipeline.py` → compact speech CTC, synthetic token/gloss planner, Cartesian diffusion, ST-GCN recognition, contrastive alignment. This is an executable research core. |
| Real-data engineering | Decoder/exporter, governed records, masks, timestamps, grouped splitting, source audit and 2D experiment exist. An accepted real multichannel training-to-render path does not. |
| Phase 2 | No accepted source-grounded canonical multichannel state/export/round-trip. Existing pose mathematics is not that completed state. |
| Phase 3A | Source-native EAF ingestion exists; Cokely one-item compatibility evidence exists. It does not authorize linguistic mappings. |
| Phase 3B | Governed queues, review/adjudication and acceptance software exist; an accepted independently reviewed SIR corpus was not found. |
| Phase 3C | Lexical registry, unknown/ambiguous/unauthorized abstention and evidence conjunction exist. Research and industrial empirical exits remain unapproved. |
| Phases 4–5 | Autoencoder/RVQ, transformer, DiT, constraints and facial modules exist independently. No accepted real multichannel representation or trained conditioned generator. |
| Phase 6 | Evaluation/statistical/human-protocol primitives exist, with newly reproduced statistical defects below. No completed independent signer-comprehension study. |
| Phase 7 | Streaming/latency/quantization/runtime primitives exist. No verified production input-to-avatar service, hardware benchmark or release bundle. |
| Reverse direction | Active sign recognition outputs gloss IDs. Full real-video ASL→English translation is a separate unfinished integration goal. |

### Evidence inspected and reproduced

- Fresh warning-strict suite: **1,726 passed**, process exit 0 (`.venv/bin/python -m pytest -q -W error`; quiet output records 1,726 passing test dots).
- `.venv/bin/python -m pip check`: no broken requirements.
- Current runtime observed: Python 3.12.14, Torch 2.13.0; historical experiments used a different Python patch release.
- Fresh bounded one-epoch synthetic/checkpoint smoke: **passed**, 8 optimizer steps; training/saving took 13.82 seconds in this small CPU run (not a real-data throughput estimate).
- A 137-joint `build_model()` reproduces `ValueError: skeleton graph is not connected` because the default edges still describe 27 joints.
- With identical model, batch and random seed, replacing frame/validity/confidence masks with all-zero masks leaves generation loss exactly unchanged: `3.1042861938476562` both times. The active loss ignores observation support.
- A batch size of 10,000 on the default synthetic training corpus yields zero training batches (`drop_last=True`); trainer construction does not reject this.
- General evaluation `sign_test_pvalue([1.]*60, [0.]*60)` returns `0.0`; its exact two-sided value is `2^-59 = 1.734723475976807e-18`. This is cancellation in the general helper, not the separately stabilized Phase-3 dependence helper.
- `paired_permutation_pvalue([NaN, 1], [0, 0])` returns `0.0` instead of rejecting invalid scores. Such a result must never be accepted as evidence of significance.
- An empty-input Phase-3C readiness call returns seven missing-evidence blockers. This verifies default fail-closed behavior; it is not itself a filesystem discovery mechanism.
- The saved signer CSV hash matches its certificate; the audit-manifest hash matches the signer certificate. The entire raw video corpus was not re-decoded or re-hashed in this audit.

The saved full-data audit accounts for **31,165 metadata rows**: 31,047 complete joins,
118 missing sources, 3 structural failures, 28,621 quality warnings, 2,423 valid rows,
and 1 additional orphan artifact. Quality warnings are not automatically unusable data.
There are 1,702 review-queue records and no selected quality threshold in that manifest.
Its old “signer identity absent” note is superseded by the verified pseudonymous-signer
certificate; the final signer-and-source-disjoint split is still explicitly absent.

The saved real 2D experiment has **76/10/10 train/validation/test clips**, source-disjoint
but not signer-disjoint. Test point error is 0.063139 for the model versus 0.004936 for
interpolation; span error is 0.071484 versus 0.010596. Supports differ (21,940 vs 20,989
point targets; 20,351 vs 19,591 span targets), so these are reported metrics, not a valid
paired effect estimate. Tube interpolation has zero support and is unavailable, not zero
error. Tiny-subset fitting succeeded, but useful held-out improvement was not established.

### Numbered blocker register

Priority: **P0** blocks trustworthy next-stage work; **P1** blocks scientific/vertical
integration acceptance; **P2** blocks product or scale. “Confirmed” describes inspected
code or reproduced behavior. “Evidence gap” describes missing accepted project evidence.
Work-package references refer to the updated roadmap. Estimates are allocated there to
avoid counting shared fixes repeatedly.

| ID | Priority / type | Finding, consequence, and required closure | Package |
|---|---|---|---|
| B01 | P0 / evidence | No accepted co-observed continuous multichannel 3D source. Frontal 2D does not identify depth, palm twist, occluded fingers or gaze. Obtain and inspect an authorized source; preserve unavailable/inferred channels explicitly. | W1–W2 |
| B02 | P0 / evidence | No accepted qualified ASL annotation/review relationship in inspected artifacts. Assign a qualified creator, independent reviewer and disagreement adjudicator with real availability and evidence. | W1/W3 |
| B03 | P0 / evidence | No training-authorized governed text/video→SIR corpus. Populate and independently accept Phase-3B records; English transcripts and native EAF values cannot simply be renamed SIR. | W3 |
| B04 | P0 / evidence | No human-reviewed lexical motion entries. Bind meaning, articulation, form, schema and exact motion bytes; retain unknown/ambiguous refusal. | W3–W4 |
| B05 | P0 / integration | Canonical Phase-2 state is not implemented and accepted against a real source. Freeze units, frames, time, joints, expressions, gaze, confidence and availability only after sample inspection. | W2 |
| B06 | P0 / evidence | Research authorization for the final combined supervision/motion bundle is missing. Existing How2Sign research evidence is not absent, but it is not blanket authorization for every new source, annotation, derivative and asset. | W1/W3 |
| B07 | P2 / evidence | Commercial training/deployment permissions are not established. Maintain a separate commercial lineage and signed/action-specific evidence; this is a commercial release blocker, not a reason to prohibit otherwise authorized research. | W1/W9 |
| B08 | P0 / evidence | No qualified, authorized body/hand/face rig and model asset bundle for the target slice. Verify individual assets, expression channels, export permission, articulation and signer review. | W1–W2 |
| B09 | P0 / planning | Old pre-Phase-2 language requires all five research/commercial bundles, whereas Phase-3C separates research and industry. Resolve this as an explicit roadmap/gate-policy change, never a silent relaxation of executable checks. | W0–W1 |
| B10 | P0 / planning | No frozen domain, phrase/phenomenon coverage, audience, target hardware or primary success endpoint. Freeze a narrow pilot; otherwise annotation, architecture and acceptance keep moving. | W0 |
| B11 | P0 / data | Final signer-and-source-disjoint split is absent. Use connected signer/source components, freeze before windowing, certify both separations and report actual counts. | W1 |
| B12 | P1 / data | Signer distribution is highly uneven: 12,102 and 14,596 available utterances belong to two codes, about 86% of the local 31,047 clips. Measure feasible grouped partitions, macro scores and subgroup uncertainty; clip count is not independent signer diversity. | W1/W6 |
| B13 | P1 / data | 118 missing sources, 3 structural failures and an orphan remain. Quarantine and disposition explicitly; do not fabricate missing files or silently repair truth. | W1 |
| B14 | P1 / data | 28,621 quality warnings, 1,702 queued reviews and no frozen QC threshold. Review stratified samples, quantify hand/face/occlusion failure, select thresholds on development data and record accepted exclusions. | W1 |
| B15 | P1 / data | Decoder clock support exists, but final source synchronization/calibration is unvalidated. Test clip origin, frame rate, dropped frames, EAF milliseconds, motion units and face/audio offsets with real samples. | W1–W2 |
| B16 | P0 / integration | Existing exporter explicitly requires source and gloss tokens. The gloss-independent roadmap has no direct governed SIR-to-training shard adapter. Implement a distinct typed path without fabricating gloss. | W3 |
| B17 | P0 / confirmed | `build_model` sizes joints from corpus but uses default 27-joint edges. A 137-joint model fails construction. Bind topology and joint ordering to the manifest and checkpoint. | W0/W2 |
| B18 | P0 / confirmed | Active generation ignores validity, confidence and frame masks. Invalid/padded targets enter MSE and velocity loss; all-invalid support still returns a loss. Carry support through losses and fail unavailable required objectives. | W0/W2 |
| B19 | P1 / integration | CTC lengths exist, but feature convolutions, pooling and decoding are not fully observation/padding aware. Test invariant output under added padding, padded batching, occlusion and variable length; CTC length alone is insufficient. | W2–W3 |
| B20 | P1 / confirmed | Velocity loss uses adjacent frame differences rather than source time intervals. Preserve timestamps; use time-aware derivatives and masks requiring both valid endpoints. | W2 |
| B21 | P1 / confirmed | Exporter builds vocabularies from all records after splitting. Declare a pre-existing closed lexicon or fit only on training data with explicit unseen-token handling; otherwise held-out vocabulary informs preprocessing. | W1/W3 |
| B22 | P1 / integration | Shards and datasets are materialized in memory, with corpus-wide temporal padding at export. Measure peak RAM/I/O and adopt bounded shards/bucketing as needed before full-scale runs. This is a scale risk, not a measured OOM. | W3/W7 |
| B23 | P0 / confirmed | `drop_last=True` can create a zero-step epoch. Reject empty loaders, adapt tiny-overfit batch size and require positive optimizer-step evidence. | W0 |
| B24 | P1 / confirmed | Validation averages batch means equally, including unequal final batches and unequal modality support. Accumulate numerators/denominators under the intended sample/frame estimand. | W0/W6 |
| B25 | P1 / confirmed | Best checkpoint is chosen by weighted total loss; analysis uses final in-memory model without explicitly selecting best. Freeze primary selection metric and name the exact checkpoint being reported. | W0/W5 |
| B26 | P1 / confirmed | Canonical validation RNG is isolated, but `analysis/report.py` independently draws generation noise and does not use that isolated protocol. Bind analysis seeds/replicates and restore caller state. | W0/W6 |
| B27 | P1 / confirmed | Analysis concatenates per-batch padded gloss tensors; different batch widths can fail. Recognition evaluation also omits input lengths. Implement ragged-safe, length-aware evaluation. | W0/W3 |
| B28 | P1 / confirmed | Planner metric decodes only eight tokens and counts reference-position matches without penalizing extra insertions. Add sequence/edit and semantic-field metrics with declared maximum-length/refusal behavior. | W0/W6 |
| B29 | P1 / confirmed | Diagonal-only InfoNCE treats semantically equivalent off-diagonal pairs as negatives. Define governed multi-positive grouping and false-negative controls before applying to repeated real meanings. | W4–W5 |
| B30 | P1 / integration | Joint weighted training starts all branches together; richer curricula are not active. Establish branch competence, gradient scale/flow diagnostics and controlled loss-weight ablations. | W4–W5 |
| B31 | P0 / confirmed | Trainer clips gradients but has no explicit non-finite-loss/gradient abort before optimizer mutation. Reject NaN/Inf, record offending batch provenance and preserve last good checkpoint. | W0 |
| B32 | P1 / confirmed | Exact continuation is scoped to epoch-boundary joint CPU training with zero loader workers. Fine-tune/polish checkpointing is rejected; worker, accelerator, distributed or mid-epoch recovery is not proven. Implement only the paths actually needed and label their guarantee. | W0/W7 |
| B33 | P1 / usability | CLI defaults to 30+175+16 epochs; `--ckpt` conflicts with nonzero secondary stages. API/CLI learning-rate defaults differ. Make bounded, checkpointable runs explicit and test documented commands. | W0 |
| B34 | P1 / integration | Runtime defaults to CPU and exposes no run-level device parameter. EMA, AMP and gradient accumulation are not integrated into the canonical trainer. Benchmark first; then add required hardware paths with parity/resume tests. | W4/W7 |
| B35 | P0 / confirmed | General sign-test helper has upper-tail cancellation; 60 all-positive pairs produce zero instead of 2^-59. Use stable tail/log-probability computation and numerical oracle cases. | W0 |
| B36 | P0 / confirmed | General permutation helper accepts NaN and can return p=0. Validate all score/config domains, finite values, minimum support and resample counts across general statistical helpers. | W0 |
| B37 | P1 / statistical | Video-dependence sign tests count examples; repeated sources/signers are not aggregated into independent units. Predeclare cluster-level tests or justify independence; otherwise significance can be overstated. | W1/W6 |
| B38 | P1 / statistical | General bootstrap is IID and seed aggregation accepts one seed. Add appropriate signer/source/reviewer hierarchy and distinguish training-seed variation from population uncertainty. | W6 |
| B39 | P1 / evidence | No frozen real-data primary endpoint, minimum useful effect, power/precision budget, exclusions, multiplicity family and calibration/test separation. Preregister before inspecting final outcomes. | W1/W6 |
| B40 | P1 / evidence | No trained direct paired learner and held-out four-intervention score artifact. Evaluator existence is not a passed experiment. Train only with accepted inputs and run all interventions. | W3–W5 |
| B41 | P1 / evidence | Existing 2D pretraining does not establish improvement over interpolation. Recompute on common support with coverage and clustered uncertainty; redesign only if a useful hypothesis survives. | W4 |
| B42 | P1 / integration | Active Cartesian motion lacks full finger rotations, facial grammar, head/gaze and contact state. A tensor-size increase alone cannot solve these semantics. | W2/W5 |
| B43 | P1 / integration | Hand-graph, SIR, duration, spatial reference and non-manual modules are disconnected from active generation. Add typed adapters and intervention tests at each boundary. | W3/W5 |
| B44 | P1 / evidence | No qualified representation comparison among retrieval/interpolation, continuous AE and RVQ; no accepted codebook-use or reconstruction/minimal-pair evidence. Keep simple baselines eligible to win. | W4 |
| B45 | P1 / integration | Rich DiT/constraints/part-aware generation is separate from active diffusion. Choose one generator, test parameterization/schedule/guidance compatibility, inpainting masks and constraint residuals. | W5 |
| B46 | P1 / integration | Biomechanical/temporal corrections are not shown to preserve meaning. Test handshape, palm orientation, contact, continuity and non-manual scope before/after projection and chunk stitching. | W5–W6 |
| B47 | P1 / integration | No complete inverse-normalization→body model→retarget→frame-render path. Persist units, handedness, rest pose, scale, face map, camera and timestamps; provide one reload-and-render command. | W2/W5 |
| B48 | P1 / evidence | Renderer functions include mathematical primitives/callbacks, not a qualified final avatar. Validate visible fingers, face/gaze, self-occlusion, mirroring and semantic fidelity with qualified signers. | W5–W6 |
| B49 | P1 / evidence | Cycle consistency uses a related recognizer and can reward shared shortcuts. Add independent baselines and blinded semantic comprehension; cycle WER remains diagnostic. | W6 |
| B50 | P1 / evidence | No frozen multi-seed real held-out generation result, counterfactuals or ablations. Report per-channel failures, coverage and confidence intervals, not only total loss. | W5–W6 |
| B51 | P1 / evidence | Human comprehension, grammaticality and error severity remain unevaluated. Recruit independent reviewers, randomize/blind presentation, retain disagreements and preregister analysis. | W6 |
| B52 | P1 / integration | Confidence/calibration/refusal primitives are not connected to real semantic failures. Fit on held-out calibration data, measure risk-versus-coverage, test out-of-domain inputs and empty evidence. | W6–W8 |
| B53 | P2 / integration | Raw audio backend/preprocessing is not active; `translate_audio_to_sign` receives acoustic features. Integrate waveform/sample-rate/channel contracts, real backend assets, tokenizer and timestamps. | W7 |
| B54 | P2 / evidence | Synthetic noise/pitch conditions do not validate real accents, noise, code switching or long speech. Collect eligible real stress cases and characterize actual failure/abstention. | W7 |
| B55 | P2 / integration | Streaming revision/commit and backpressure contracts are independent primitives. Connect cancellation, committed-prefix immutability, stale-result dropping and temporal seams to the renderer. | W7 |
| B56 | P2 / evidence | No measured first-output/end-to-end p50/p95/p99 latency, throughput, queue behavior, peak memory or target-hardware capacity. Benchmark the actual composed path before optimization promises. | W7–W8 |
| B57 | P2 / integration | Quantization/static-execution algebra is not optimized production execution. Require numerical AND per-channel/linguistic parity; reject speedups that harm signing. | W8 |
| B58 | P2 / integration | No self-contained service/app bundle with authorization, retention, logging, model/data identity, failure reporting, rollback and load-tested lifecycle. Build and rehearse a restricted release. | W8–W9 |
| B59 | P1 / trust | Hashes and typed attestations bind contents but do not authenticate qualifications, permission or observed truth by themselves. Verify issuers/documents and bind accepted evidence to the actual training run. | W1/W3/W9 |
| B60 | P1 / integration | `run.py` checks corpus structural readiness, not the complete source-portfolio/Phase-3/human release gates. Connect action-specific policy at training/export/release boundaries; do not treat its `passed` flag as project readiness. | W3/W9 |
| B61 | P1 / maintainability | Parallel implementations in `models/`, specialized packages, legacy trainers and evaluation can drift. Declare owners/canonical paths; retain explicitly quarantined research paths with scope labels. | W0 |
| B62 | P1 / evidence | CI is defined for macOS; current tests do not prove clean install today on every target, accelerator determinism or production data portability. Run the selected release environment and archive/wheel checks. | W0/W9 |
| B63 | P1 / documentation | Historical audit still lists fixed overwrite, missing-modality, checkpoint, CTC and license issues. Preserve history but make current verdict authoritative; remove these from the new outstanding-work queue. | W0 |
| B64 | P2 / scope | Full ASL→English needs real-video perception, multichannel temporal understanding, an English decoder and independent evaluation. Current sign→gloss recognition does not complete the bidirectional goal. | W10 |
| B65 | P1 / resourcing | External acquisition/review turnaround, annotation throughput, compute throughput and budget are unmeasured. Assign owners and benchmark them; engineering days cannot guarantee external evidence arrival. | W0–W1 |
| B66 | P1 / planning | Completing gate software has been advancing phase labels while empirical dependencies remain unresolved. Track software-complete and evidence-accepted separately; no downstream empirical phase passes on test count alone. | All |

### Minimal reproduction commands for newly found defects

Run from the repository with the existing `.venv`. These are diagnostic probes of the
current code, not assertions that the returned behavior is acceptable:

```python
from dataclasses import replace
from signtranslator.run import build_model
from signtranslator.data.corpus import CorpusSpec
from signtranslator.eval_framework.statistics import (
    sign_test_pvalue, paired_permutation_pvalue,
)

print(sign_test_pvalue([1.] * 60, [0.] * 60), 2. ** -59)
print(paired_permutation_pvalue([float("nan"), 1.], [0., 0.]))
spec = CorpusSpec.build(num_concepts=12, seq_len=4, num_joints=27,
                       in_channels=3, num_frames=32)
try:
    build_model(replace(spec, num_joints=137), diff_timesteps=10)
except ValueError as error:
    print(type(error).__name__, str(error))
```

For the mask test, take a batch from a newly generated temporary synthetic corpus,
put the model in evaluation mode, and compute `training_step(batch)["generation"]`
with seed 123. Repeat with the same seed after replacing `validity_mask`, `confidence`
and `frame_mask` with all-zero tensors. The identical losses above demonstrate that
these fields do not affect the active objective. Never use an existing real-data
folder as a synthetic probe target. No source code was changed during this audit.

### Findings that must not be charged again as unresolved baseline work

Corpus generation is now explicit and guarded; missing speech reports unavailable/failure;
best and last checkpoints are distinct; checkpoint metadata and CPU resume are substantially
implemented; canonical validation has isolated RNG; exact repeat-aware CTC feasibility and
`zero_infinity=False` are active; license and foundation requirements files exist. The
pseudonymous signer grouping key is certified. These are real accomplishments, with the
remaining limitations called out above.

### External-source check

On 2026-09-25 the [How2Sign publisher page](https://how2sign.github.io/) still lists
CC BY-NC 4.0 and research-only availability. The [SMPL-X license page](https://smpl-x.is.tue.mpg.de/modellicense.html)
still distinguishes noncommercial scientific use from separately licensed commercial use.
These publisher statements do not resolve project-specific permissions or authorize new
asset downloads, outreach, purchases or deployment.

---

## Historical audit baseline (retained for traceability; not current status)


## 1. Executive verdict

| Question | Verdict |
|---|---|
| Does the package import and compile? | Yes |
| Are many mathematical primitives tested? | Yes |
| Is the synthetic pipeline executable? | Yes |
| Is the full thirteen-layer architecture integrated? | No |
| Is it ready for a small real-data pilot? | Not yet |
| Is it ready for expensive full training? | No |
| Is it ready for user deployment? | No |

The present codebase is between **verified components** and an **integrated
synthetic research core**. It has not reached real-data pilot training,
scientific validation, or deployment.

## 2. Audit evidence

### Source inventory

- 334 non-directory files
- 311 Python files
- approximately 39,000 Python lines
- 136 test files
- no `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.npz`, or `.npy` artifacts in the
  distributed archive

### Static validation

`python3 -m compileall -q .` completed successfully.

This proves that the files parse under the audited interpreter. It does not
prove that imports, dependencies, training, or inference are correct.

### Test results

Standard run:

```text
1454 passed
10 failed
1 warning
```

Warning-strict run:

```text
11 failed
```

The ten substantive failures occur in:

- `tests/test_speech_stage3_integration.py`;
- `tests/test_speech_stage5_harness.py`.

The synthetic noisy condition remains at 100% accuracy. As a result:

- there are no errors against which confidence can be discriminative;
- abstention cannot improve selective accuracy;
- the fail-closed policy cannot prove that it suppresses errors;
- Brier resolution is zero;
- temperature scaling worsens ECE in the tested split;
- noisy and long-form conditions are classified as degenerate.

These failures do not invalidate all speech mathematics. They invalidate the
claim that the current robustness harness demonstrates useful behavior.

### Synthetic training smoke test

The audited smoke run:

- generated 256 training and 64 validation samples;
- passed the repository’s internal readiness gate;
- constructed a 1,610,109-parameter model;
- completed one training epoch;
- wrote a roughly 20 MB checkpoint;
- reloaded that checkpoint;
- ran acoustic-feature-to-motion inference;
- returned finite tensors of the expected shape.

The one-epoch metrics failed, as expected for an untrained model. Their values
should not be interpreted as estimates of eventual accuracy. The useful result
is that the synthetic path executes mechanically.

## 3. Critical findings

### R1 — Corpus overwrite risk

`signtranslator/run.py` defaults to `regenerate=True`. The documented CLI does
not expose a `--no-regenerate` option. Supplying a real corpus directory to the
default command can replace its standard files with generated synthetic data.

**Required correction:** default to non-destructive behavior. Synthetic
generation should require an explicit flag and should refuse to write into a
non-empty corpus directory without confirmation.

### R2 — Missing modality can pass evaluation

`analysis/report.py` initializes `speech_wer = 0.0`. If a corpus has no speech
features, the speech branch is not evaluated but receives a passing value.

**Required correction:** represent missing evaluation as unavailable and fail
the corresponding gate when that modality is required.

### R3 — “Best” checkpoint is overwritten

The trainer saves a best-validation checkpoint during `fit()`. After fitting,
`run.py` saves the final model to the same path, overwriting the best model.

**Required correction:** maintain distinct `best`, `last`, and optional
milestone checkpoints.

### R4 — Checkpoint is not self-describing

The checkpoint includes:

- model state;
- optimizer state;
- global step;
- best validation loss.

It omits:

- architecture and diffusion configuration;
- scheduler and epoch;
- RNG states;
- vocabulary/tokenizer;
- pose normalization statistics;
- model-format version;
- dataset and preprocessing identifiers;
- source revision;
- evaluation thresholds.

Without those fields, a checkpoint cannot independently reproduce training or
standalone inference.

### R5 — Resume is not a true resume

`resume=True` loads weights with `load_optimizer=False`. Scheduler state is never
saved. The CLI does not expose the resume option. This is a warm restart, not a
continuation of the same optimization trajectory.

### R6 — Validation is stochastic

Diffusion validation samples random timesteps and noise. The composite
validation loss used for checkpoint selection therefore changes with random
draws. Branch losses also have different scales, so their weighted sum is not a
stable scientific selection criterion.

**Required correction:** use fixed validation seeds/noise/timesteps, report
per-branch confidence intervals, and choose a pre-declared primary checkpoint
criterion.

### R7 — CTC feasibility is under-specified

The readiness gate checks only `frames >= target_length`. CTC may require extra
frames when adjacent labels repeat. Speech subsampling further reduces the
usable input length. `zero_infinity=True` can convert impossible alignments into
zero loss, silently removing bad examples from training.

**Required correction:** compute exact per-sample minimum CTC length after
subsampling and reject impossible samples before batching.

### 2026-09-10 remediation status for R3–R7

These findings describe the earlier audit baseline. The current Phase-1 implementation
changes their status as follows:

- **R3 resolved for the canonical joint trainer:** `best` and `last` are distinct;
  explicit milestone saves are supported; the final state no longer overwrites best.
- **R4 substantially resolved:** schema-v2 checkpoints and hash-verified JSON sidecars
  bind model/diffusion/trainer configuration, corpus manifest and shard hashes,
  vocabulary and normalization through that manifest, data-loader contract, numerical
  runtime, source-byte identity, optimizer, scheduler, epoch/step, history, and RNG state.
  Future architecture migrations remain fail-closed rather than implicit.
- **R5 resolved for epoch-boundary joint CPU training with `num_workers=0`:** resume
  restores the same optimization trajectory and is tested bit-for-bit. Worker-local RNG
  and the legacy generator-only fine-tune/polish stages are not yet state-complete;
  checkpointing those paths is explicitly rejected.
- **R6 partially resolved:** validation now uses an isolated fixed RNG stream, produces
  repeatable values, and cannot advance training RNG or leave the model in evaluation
  mode. Per-branch confidence intervals and the preregistered primary selection criterion
  still require real experimental design and remain open.
- **R7 resolved in the active sign, speech, corpus, exporter, and speech-objective paths:**
  the exact adjacent-repeat minimum is enforced after subsampling, blank/out-of-range
  targets are rejected, and active CTC losses use `zero_infinity=False`.

This remediation establishes execution integrity only. It does not change the real-data,
linguistic, human-evaluation, or deployment gates below.

### 2026-09-10 pre-Phase-2 data verdict

One prior limitation is resolved: the How2Sign filename suffix is now certified as the
official pseudonymous signer ID by exact reconciliation of the immutable local audit with
the CVPR supplemental. All 31,047 available clips match published per-signer counts, and
all 118 missing rows reconcile by signer. This is a grouping key, not personal identity.

The new machine-enforced source-portfolio gate still fails every Phase-2 evidence bundle.
There is no locally verified single source with co-observed continuous 3D body, hands,
face, head, and gaze; no locally verified continuous qualified-ASL linguistic reference;
no commercial training authorization; no locally qualified commercial render rig; and no
confirmed qualified-ASL governance relationship. Capabilities from separate corpora are
not composable as if they had been co-observed. Phase 2 therefore remains closed. See
`04_DATA_ENGINEERING_AND_CORPUS.md` §§10–13 for evidence and acquisition treatment.

### R8 — Documentation and code have drifted

Examples include:

- the architecture document describes cycle consistency as diagnostic, while
  the current code gates it;
- the data document reports an older number of test files and a green suite;
- the requirements comment references a missing
  `requirements-foundation.txt`;
- the package declares an Apache license but ships no license file.

Documentation should be versioned and checked in CI against executable
configuration.

## 4. Readiness gates

| Gate | Current state | Evidence needed to pass |
|---|---|---|
| Build and import | Pass | Clean installation in a pinned environment |
| Unit mathematics | Partial | Zero failures under warning-strict tests |
| Synthetic integration | Partial/pass | Reproducible full run and artifact reload |
| Real corpus ingestion | Fail | Versioned real batch through active loader |
| Full architecture integration | Fail | One traced semantics-to-render path |
| Real pilot training | Fail | Tiny-subset overfit and held-out experiment |
| Scientific validation | Fail | Independent baselines and signer-held-out tests |
| Human comprehension | Fail | Blinded qualified-signer evaluation |
| Runtime deployment | Fail | Loadable service/app with measured latency |
| Operational safety | Fail | Integrated abstention, monitoring, and rollback |

## 5. Strict release language

Until the failed gates are resolved, releases should say:

> Research scaffold with synthetic training and independently tested
> mathematical components.

They should not say:

> Production-ready translator, complete speech-to-avatar system, validated
> sign-language generator, or accessibility replacement.
