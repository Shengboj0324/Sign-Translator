# 06 — Implementation Roadmap

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


## Remaining-work rebaseline after contact audit — 2026-09-26

This is the current **proposed capacity plan**, superseding the older calendar and the
R1–R25 task allocation below. W0 is complete: its 10 days/60 hours are not charged again.
Completed W1/W2 adapters, numerical repairs and static asset/render work receive credit;
do not rebuild them. Keep the provisional remaining W1/W2 reservation at **25 workdays /
150 engineer-hours**, because source-specific integration, qualified review and the new
contact findings have not been timed on an eligible pilot. This is a reservation rather
than an assertion that exactly 150 hours of work remain. B69 belongs to the existing rig
qualification/debug allocation; it is not double-counted as another phase.

### Critical path and acceptance

Eligible co-observed source + action evidence + qualified review capacity → accepted QC,
independent units and split → calibrated real canonical state → reviewed source-to-render
round-trip → accepted SIR/lexical supervision → representation baseline → conditioned
model → independent comprehension evaluation. Audio/streaming and release follow the
validated forward path. The reverse direction has its own perception/decoding/evaluation.

The current critical external dependencies are B01/B02/B06 and the eligible population /
statistical-design constraints B11/B14/B39/B68. Contact B69 is a measured geometry issue
requiring classification and qualification, not evidence that every visible render is
wrong. These dependencies cannot be solved by more synthetic training or increasing test
counts. Stop empirical training at an unmet evidence gate, while continuing independent
engineering work that remains useful.

### Capacity calendar and debugging budget

Assume one engineer, six productive hours/day, five weekdays/week. Development allocations
include routine tests, documentation and coordination. Debug allocations below are explicit
repair/integration/acceptance reserves. Specialist hours, queued computation and external
waiting are additional calendar lanes. Dates assume a September 28, 2026 start with all
prerequisites available when needed; weekends are excluded, holidays/leave are not.

| Package | Remaining workdays | Conditional dates | Build / debug days | Engineer-hours | Exit |
|---|---|---|---:|---:|---|
| W1 source/QC/split/study protocol | R1–R10 | Sep 28–Oct 9, 2026 | 8 / 2 | 60 | Eligible action-bound population, accepted QC/split and review/statistical protocol |
| W2 real state/rig round-trip | R11–R25 | Oct 12–30 | 11 / 4 | 90 | Actual calibrated source-to-state-to-render result with qualified review |
| W3 governed language/motion bridge | R26–R40 | Nov 2–20 | 11 / 4 | 90 | Accepted SIR/lexical pilot and direct typed training adapter |
| W4 representation comparison | R41–R55 | Nov 23–Dec 11 | 11 / 4 | 90 | Common-support retrieval/interpolation/AE/optional RVQ comparison |
| W5 conditioned generation | R56–R75 | Dec 14–Jan 8, 2027 | 14 / 6 | 120 | Reloadable conditioned generation/rendering; intervention and constraint evidence |
| W6 independent evaluation | R76–R100 | Jan 11–Feb 12 | 18 / 7 | 150 | Blinded comprehension, grouped uncertainty, failure/coverage and research decision |
| W7 waveform/streaming | R101–R120 | Feb 15–Mar 12 | 15 / 5 | 120 | Real audio-to-avatar, revision/cancellation/seams and measured hardware profile |
| W8 runtime/performance | R121–R130 | Mar 15–26 | 7 / 3 | 60 | Bounded service, load/memory profile and quality-preserving optimizations |
| W9 restricted release | R131–R140 | Mar 29–Apr 9 | 6 / 4 | 60 | Independent bundle reload, monitoring/rollback and action-specific release decision |
| W10 optional ASL-to-English | R141–R165 | Apr 12–May 14 | 18 / 7 | 150 | Real-video perception/English decoding with separate independent acceptance |

Remaining reservations: **100 days/600 hours** to the evaluated text candidate, including
27 debug days/162 hours; **140 days/840 hours** to the audio/restricted-release candidate,
including 39 debug days/234 hours; **165 days/990 hours** including reverse translation,
with 46 debug days/276 hours. These replace the historical 110/150/175-day totals for
remaining scheduling only. At 15 productive hours/week, the engineering workload is
40/56/66 weeks respectively, before external delay. A missed acceptance criterion consumes
buffer, reduces explicitly agreed scope, or extends the plan; the date never grants a pass.

### Next 25 days, six hours per day

Rows describe intended work, not permission to invent unavailable inputs. If source or
review evidence is absent, record the unmet dependency and move only independent work
forward; do not label the dependent row complete. R1 and R3 start with already acquired assets.

| Day | Six-hour workload | Concrete output / check |
|---|---|---|
| R1 | 2h reconcile completed evidence; 3h localize B69 crossing witnesses; 1h issue triage | Visible/hidden/contact-region inventory and reproducible geometric cases |
| R2 | 3h exact source sample/channel inspection; 2h action/evidence binding; 1h acceptance-gap record | Eligible-source candidate disposition; if absent, explicit acquisition dependency |
| R3 | 3h native attachment/rest/morph parity investigation; 2h minimal contact probes; 1h discrepancy report | Distinguish adapter error, source geometry and intended overlap before changing mesh |
| R4 | 2h qualified role/independence evidence; 2h rubric and timing pilot; 2h review process dry-run | Actual review capacity and accepted review procedure, not generated approval |
| R5 | 2h eligible pilot/QC sample; 2h time annotation/adapter work; 2h reforecast | Measured throughput and revised scope/capacity with dependency dates |
| R6 | 3h stratified QC; 2h development-only thresholds; 1h disposition checks | Versioned exclusions, missingness and hand/face/occlusion coverage |
| R7 | 3h eligible component split; 2h leakage/preprocessing checks; 1h identity freeze | Source/signer separation and train-only transforms/vocabulary |
| R8 | 2h unit/estimand/useful-effect decisions; 2h power or precision design; 2h family/test-access checks | Reviewed draft ready for preregistration; finite-resolution failure cannot be waived |
| R9 | 4h QC/split/policy debugging; 2h adverse-case verification | Repair evidence for corrupt, missing, unapproved and crossing-partition inputs |
| R10 | 3h W1 requirement review; 2h remaining repair; 1h decision/reforecast | Accepted W1 evidence or explicit extension with owners and unmet rows |
| R11 | 3h source units/frames/rest conventions; 2h calibration; 1h independent transform oracle | Source-to-canonical mapping, scale/root/camera and handedness evidence |
| R12 | 3h actual channel adapter; 2h availability/inferred provenance; 1h semantic mapping review | Co-observed channel mapping with unsupported fields explicitly unavailable |
| R13 | 3h timestamps/offsets; 2h dropped-frame/synchronization probes; 1h policy record | Validated clock origin, timing tolerances and calibration evidence |
| R14 | 3h real shard export; 2h identity/action binding; 1h reload | Actual eligible source-bound canonical shards |
| R15 | 2h source/state round-trip; 2h support/transform oracles; 2h reforecast | Reversible supported coordinates and honest irreversible/inferred boundaries |
| R16 | 2h source gap/confidence policy; 2h nonuniform derivatives; 2h jitter/missingness checks | Physical velocity/acceleration on declared support; no undeclared gap bridging |
| R17 | 3h physical inverse/retarget; 2h skinning/contact fixtures; 1h state-to-rig checks | Real motion mapping with units and rest frames persisted |
| R18 | 3h deterministic source-sample render; 2h independent reload; 1h evidence packaging | Reproducible real-sample render command and exact asset/model identities |
| R19 | 4h unintended contact/mirroring/weight repair; 2h pose-catalogue re-audit | Explain fixed and remaining B69 regions; preserve intended anatomy/meaning |
| R20 | 3h finger/face/gaze visibility and transitions; 2h temporal/contact debug; 1h reforecast | Static and transition diagnostics; unseen combinations remain explicitly unqualified |
| R21 | 4h support qualified source-versus-render pilot; 2h discrepancy encoding | Independent handshape/orientation/location/contact/non-manual review |
| R22 | 4h development-only adapter/render repairs; 2h focused regression | Fixes linked to reviewed defects without tuning on reserved final examples |
| R23 | 3h review repeats; 2h disagreement/coverage analysis; 1h adjudication record | Accepted scope and unresolved failures; reviewer time budgeted separately |
| R24 | 4h integrated regression/reload; 2h defect buffer | Real pipeline identity, missingness, numerical and render checks |
| R25 | 3h remaining repair; 2h W2 gate audit; 1h freeze/reforecast | Accepted W2 or explicit extension; no phase advancement by elapsed days |

R9/R10 and R19/R20/R24/R25 supply the six dedicated debug/acceptance days. The acquired
rig's nominal units, static controls and basic render replay are already verified; only
source-dependent integration, native parity, contact qualification and relevant regressions
are charged here. Large topology changes, new motion capture or extensive native-runtime
reconstruction require a re-estimate, not silent absorption into these hours.

### Mathematical, statistical and logical work after W2

- **W3:** implement direct governed SIR ingestion, exact field/lexical identity, ambiguity and
  authorized-unknown refusal. Establish meaningful positive/negative pairs and duration/
  non-manual scope before contrastive or generation objectives consume them.
- **W4:** prove tiny-real-subset fit and compare simple baselines on identical supported
  targets. Check geodesic rotation error, position/contact error, nonuniform-time derivatives,
  per-part missingness and coverage. RVQ needs occupancy/collapse/rate-distortion evidence;
  no architecture is selected merely because it is more complex.
- **W5:** bind one generator's target/parameterization/schedule/guidance conventions. Audit
  conditioning interventions, observation/inpainting masks, gradient scales, temporal seams
  and constraint residuals. Any biomechanical/contact correction also needs evidence that
  handshape, spatial reference and non-manual meaning survive the correction.
- **W6:** freeze the independent unit, signed useful effect, endpoint family, exclusion and
  calibration policy before outcomes. Analyze paired components and reviewer/source
  dependence; report seed variation separately. Use blinded semantic recovery and retained
  disagreements, independent baselines and a reserved remediation evaluation. Sign-test
  resolution is a necessary design check, not a power calculation or independence proof.
- **W7–W9:** verify waveform/time contracts, committed-prefix immutability, cancellation and
  queue bounds. Measure p50/p95/p99 latency, memory and throughput on the actual path.
  Optimize only with numerical and linguistic parity; bind action policy and accepted
  source/model identities at training, export and release boundaries.
- **W10:** evaluate real-video ASL-to-English separately; gloss recognition scores cannot
  establish English meaning recovery or non-manual understanding.

The detailed mathematical/statistical/logical acceptance tables later in this document
remain applicable. Their historical defect descriptions are superseded by the current
B01–B69 disposition in `03_READINESS_AUDIT.md`; repaired numeric functions are not reassigned
as new implementation work.

### External capacity and reforecast rules

Retain the provisional **200–350 specialist-hour** forward-program reservation until a
qualified pilot measures actual throughput. The example 300-item annotation workload is
112.5 hours at 12 min creation + 8 min review + 25% × 10 min adjudication per item. It is a
capacity example, not a scientific sample-size recommendation. At 16 combined specialist
hours/week, 200–350 hours require 12.5–21.875 weeks and may run alongside engineering only
when those people and prerequisites actually exist.

Profile accepted data before budgeting training: total device time includes preprocessing,
all epochs/steps, validation, each seed/ablation, checkpoint I/O and restart allowance.
Neither the 76-second unit suite nor overnight availability estimates training throughput.
Every five workdays record hours spent, accepted artifacts, remaining defects, measured
annotation/compute rates and external arrival dates. Recompute each successor's earliest
start as the maximum of predecessor acceptance, required evidence arrival, reviewer slot
and engineering capacity. No defensible unconditional completion date exists before those
external inputs are committed.

---

## Execution-plan revision — 2026-09-25

This is the current proposed execution order and workload baseline. It supersedes old
priority lists that still treat repaired Phase-1 issues as unfinished. It does not claim
that an empirical phase passed, replace external authorizations, or change executable gate
behavior. The corresponding audit contains B01–B66 plus the B67 source-binding and B68
statistical-feasibility addenda in `03_READINESS_AUDIT.md`.

## W1/W2 execution and requirement audit — 2026-09-26

**Status: active; neither phase is accepted.** W0's historical 1,832-test record stays
unchanged. The following work advances the full Days 11–35 objective; it does not redefine
completion around source-independent tests. Absolute correctness is not a finite-test claim.

### Current scheduling interpretation — 2026-09-26

The original D1–D175 table is a whole-program workload baseline, not remaining work or a
guaranteed calendar. W0 has met its recorded engineering exit: do not schedule its 60 hours
again. W1/W2 software repairs also receive credit, but source-specific adaptation and review
turnaround remain unmeasured, so a precise subtraction of remaining effort is not justified.
Reserve **25 workdays / 150 engineer-hours** for remaining W1/W2 integration and acceptance,
and re-estimate after R5. This is a provisional capacity reservation, not measured remaining
effort. Each day below budgets six productive engineer-hours; specialist work and external
waiting are additional. Reuse the verified software rather than rewriting it.

| Relative workdays | Individual daily work and exit evidence |
|---|---|
| R1–R5 | R1: native rig weight/rest/axis semantics. R2: exact source channels, bytes and permitted actions. R3: rig convention and missing-asset qualification plan. R4: qualified review roles, rubric and capacity. R5: eligible pilot timing and dependency reforecast. |
| R6–R10 | R6: stratified QC and development-only thresholds. R7: eligible population and disjoint component split. R8: independent units, endpoints, multiplicity and useful-effect design. R9: QC/split/preprocessing debugging. R10: W1 evidence acceptance and repair buffer. |
| R11–R15 | R11: physical units, handedness, root/camera and rest frames. R12: actual multichannel mapping and missing/inferred provenance. R13: clocks, synchronization and calibration. R14: source adapter and real shards. R15: reload, identity, support and transform oracles. |
| R16–R20 | R16: source-bound gap/confidence/derivative policy. R17: physical inverse, rest/pose transforms and skinning. R18: deterministic real-sample render and reload command. R19: geometric/weight/mirroring debugging. R20: finger/palm/face/eye visibility and temporal debugging. |
| R21–R25 | R21: qualified source-versus-render pilot review. R22: development-only mapping/render repairs and regression cases. R23: review repeats, disagreement and coverage. R24: integration/reload regression and repair buffer. R25: W2 evidence acceptance, final buffer and conditional state freeze. |

R9–R10 and R19–R20/R24–R25 reserve six days (36 hours) explicitly for debugging and
acceptance repair, in addition to routine checks inside other days. R1 can start now.
Source-, permission- and reviewer-dependent rows cannot be completed by elapsed time alone.
Failed dependency gates move their successors. If all inputs and review capacity are
available, a September 28 weekday start places R25 on October 30, 2026, before holiday or
availability adjustments. This is a conditional reservation, not an acceptance deadline.
Afterward the baseline W3–W6 workload is another 75 workdays/450 engineer-hours; W7–W9 adds
40/240, and optional W10 adds 25/150. Do not overlap two full-time engineering rows for one
person. Update actual days spent, remaining artifacts and dependency dates every five days.

### Authored facial-control catalogue and eyelid probes — 2026-09-26

Acquired the pinned upstream face-pose JSON/BVH and license files, verifying Git blob
identity, size and SHA-256. `makehuman_face.py` now decodes all **60 named static poses**
onto all **163 target bones** with exact-name coverage, strict channel layout, proper
rotation checks and target-rest identity binding. The BVH rows are independent pose units,
not observed signing or a recorded blink sequence. The file's 0.041667 frame-time metadata
is not used as a blink-duration claim. Its JSON animation filename is not used for dynamic
file discovery: the native plugin's explicit face-pose file pair is independently pinned.

Source inspection established Z-up to Y-up conversion, declared XYZ Euler order,
translation-disabled expression loading, and conjugation into each target bone's rest
basis. The BVH rest positions differ from this mesh by up to 0.45375 source units, so they
are recorded rather than substituted for the target bind skeleton. Root translation
channels are exactly zero. The named Rest row contains rotations up to 0.000164 degrees;
these are preserved and its resulting mesh displacement is recorded, not silently zeroed.
Native float32/application parity and morphed-character equivalence remain unverified.

Independent explicit trigonometric oracles check every bone for left/right upper-lid
closure, jaw drop and tongue-out poses; identity, naming and mutation failures are tested.
Thirty-two focused tests passed. The full warning-strict suite passed **2,022 tests in
76.09 seconds**, with no failures or skips; compilation and diff checks passed. Source
inspection identities, all 60 control inventories and measured discrepancies are in
[face-pose verification](evidence/w1-w2-2026-09-26/face-pose-units/verification.json).

The render CLI accepts paired `--face-units-root` and `--face-unit` arguments. Inspected
left/right closure renders close the expected anatomical eye with the opposite eye open;
the authored JawDrop produces a visible mouth opening with the existing interior assets.
This verifies selected static controls, not temporal blink dynamics, collision-free contact,
calibrated gaze or facial grammar. All six views of the final left-lid probe replay with
identical decoded pixels from a fresh same-machine saved-scene load, independently checked
via AV. Attachments, shape keys, UVs and packed textures survive reload.

A diagnostic metadata bug counted manual angle entries rather than actual applied rotations
for authored poses. The exporter now counts matrices at an explicit 1e-12 elementwise
threshold, separately recording manual entries. Initial artifacts retain errata; the final
`rig-render-authored-lid-verified` bundle was regenerated and its count verified. This was a
reporting repair; the earlier applied rotations and rendered geometry were correct.

B08/B42/B47/B48 advance at the static rig-control level. Source-derived multichannel states,
actual observation/inference mappings, blink/contact transitions, native-runtime checks and
qualified reviewer acceptance still remain. No facial catalogue is substituted for a
co-observed signing corpus or accepted linguistic supervision. W1/W2 remain active.

### Mouth attachments and verified nominal units — 2026-09-26

The official system archive is now local outside the repository at
`/Users/jiangshengbo/Volumes/makehuman_assets/system-2026-09-26/system-assets.zip`:
280,737,770 bytes, SHA-256 `b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107`.
Eight selected teeth/tongue mesh, attachment, material and texture files were extracted
with member CRC verification and individual SHA-256 records. This is byte identity and
publisher-source provenance, not independent authentication of every permission claim.
The [mouth intake](evidence/w1-w2-2026-09-26/source-intake/makehuman-mouth/intake.json)
records the official pack page and archive binding; original downloaded bytes are preserved.

Shared fitting is now in `makehuman_attachment.py`, with separate pinned eye and mouth
loaders. The eye API remains available. Direct one-vertex mappings and signed three-vertex
mappings are handled distinctly. Teeth have 3,868 vertices/3,560 faces and 2,646 negative
geometry coefficients, preserved during fitting. Their skin support is exactly 1,984
head-bound vertices and 1,884 jaw-bound vertices. The tongue has 226 vertices/224 faces,
zero negative geometry coefficients and exact direct rest-vertex correspondence. Native
rest fitting and bone-weight policies remain distinct; no anatomical/linguistic acceptance
is inferred from agreement with those policies.

Actual-asset tests establish stationary upper teeth under local jaw rotation, rigid lower
teeth about the jaw pivot, derivative agreement, exact tongue rest mapping, whole-body
rigid-transform invariance and altered-byte rejection. Eye/rig regressions remain green.
Twenty-five focused tests passed, and the full warning-strict suite passed **2,015 tests
in 74.78 seconds**, with no failures or skips. Compilation and diff checks passed. See
[mouth verification](evidence/w1-w2-2026-09-26/mouth-attachment/verification.json).

The renderer accepts `--mouth-root` and an explicit `--jaw-angle-radians`. Original 0.10
and wider 0.25-radian probes are preserved separately. The wider probe shows tongue and
teeth inside the mouth in inspected face and close-up views. All six views reproduce with
identical decoded pixels after a fresh same-machine `.blend` reload; independent AV checks
agree. Reload checks preserve all attachment faces, rest/posed shape keys, UV coordinates
and packed texture bytes. Render evidence is in `rig-render-mouth-inspection/`. This is a
synthetic geometric probe, not a mouth morpheme, speech viseme or biomechanics acceptance.
Contacts, clipping over the full intended range, lip seals and tongue articulation remain
requirements for subsequent acceptance.

**Nominal rig units are now verified:** the pinned exporter `guiexport.py:124-128` sets
native decimeter scale to 1 and meter export scale to 0.1. The official
[export documentation](https://static.makehumancommunity.org/makehuman/docs/exports_and_file_formats.html)
independently identifies the internal unit as decimeters. `SOURCE_METERS_PER_UNIT = 0.1`
records that convention; current mesh/pose values remain unchanged in source units. New
Blender probes declare metric display scale 0.1. The
[unit record](evidence/w1-w2-2026-09-26/mouth-attachment/unit-convention.json) binds the pinned
source blob and measures the body extent. This closes uncertainty about this asset's
nominal unit conversion, not calibration of an unacquired motion source or measurement of
an individual signer. Older statements that the unit convention was unknown are historical.

B08/B47/B48 remain partially open: actual motion-to-rig correspondence, native/morphed-mesh
parity, facial and blink controls, gaze calibration and qualified reviewer acceptance have
not been established. W1/W2 remain active. Next focused work is eyelid/eye and mouth/contact
interaction auditing; source eligibility and qualified-review dependencies remain unchanged.

### Fitted eye assets and explicit eye-axis probe — 2026-09-26

The earlier empty-eye-socket finding now has a concrete partial repair. Six source files
(eye mesh, attachment, material, texture and two upstream license files) were acquired from
the same pinned MakeHuman commit and verified against Git blob identity, size and SHA-256.
See [eye intake](evidence/w1-w2-2026-09-26/source-intake/makehuman-eyes/intake.json).
`signtranslator/avatar_render/makehuman_eyes.py` fits the 1,064 eye vertices and 1,020 faces
to the inspected rest rig and preserves 808 UV coordinates and the source texture.

The source OBJ's authoring origin differs from the body, so direct placement is invalid.
Its attachment includes **102 negative affine geometry coefficients**; these are preserved
for rest fitting, not confused with skin weights. Geometry uses the signed three-vertex
combination plus axis-scaled offsets. Bone mapping separately retains positive products
above the inspected native threshold, merges and normalizes them, with explicit final
partition-of-unity normalization. Fit first, then skin: refitting fixed-axis offsets after
posing would fail whole-body rigid-transform equivariance. The fitted object binds its
rest-rig identity and rejects changes after fitting. Native float32/application parity is
still unverified; source code was inspected but not imported or vendored.

Tests cover actual eye inventory/rest placement, common rigid motion, single-eye pivot
motion and gradients, unchanged opposite-eye support, changed-rest rejection, invalid skin
weights and altered asset bytes. Twenty focused tests passed; the full warning-strict
suite passed **2,010 tests in 73.23 seconds**, with no failures or skips. Compilation and
diff checks passed. See [verification](evidence/w1-w2-2026-09-26/eye-attachment/verification.json).

The renderer now accepts `--eyes-root` pointing to the pinned eye asset directory. Texture
bytes are checked and packed into the saved `.blend`. Inspected rest and articulated face
renders show fitted eyeballs/irises/pupils. The initial local-Y rotation was correctly
identified as mainly torsion and retained in `rig-render-eyes`; the final explicit local-Z
probe changes each bone's forward axis by **0.15 radians**, measured geometrically. This
is not calibrated human gaze. Five final views in `rig-render-eye-tilt` replay with identical
decoded pixels in a fresh same-machine Blender process, independently confirmed via AV;
packed texture identity and eye mesh/shape-key storage were also checked.

The mouth still has no visible interior. Eye shell inheritance also includes small weights
from oculi/orbicularis bones; eyelid/blink/eye-shape interactions need their own acceptance.
Material appearance uses a diagnostic Blender shader, not native shader equivalence.
B08/B47/B48 remain open for physical calibration, anatomy, full facial controls, actual
motion correspondence and qualified review. No observed signing or phase approval follows
from this synthetic articulation.

The old downloader's `makehuman-assets` GitHub route returned HTTP 404. A current official
[system asset pack page](https://static.makehumancommunity.org/assets/assetpacks/makehuman_system_assets.html)
lists teeth and tongue assets and a 267 MB download. That pack has not yet been downloaded
or inspected; the [route record](evidence/w1-w2-2026-09-26/eye-attachment/mouth-access-route.json)
keeps acquisition separate from qualification. Next concrete work is mouth-asset intake,
attachment/weight audit and rendering, followed by facial/eyelid interaction checks.

### Real asset rendering and saved-state replay — 2026-09-26

`scripts/w2_makehuman_render_probe.py` now runs the pinned adapter and installed Blender
5.2.2 LTS in a separate factory-startup background process. It saves a JSON mesh/pose bundle,
an editable shape-key diagnostic `.blend`, five body/hand/face PNGs and camera/render/source
identities. This is a baked mesh diagnostic, not a native Blender armature or MakeHuman
runtime-parity claim. The declared proper rotation `(x,y,z)->(x,-z,y)` preserves scale;
units remain source units. The 13,378 body polygons are retained, while 5,108 nonbody/helper
polygons are explicitly excluded from the visible surface.

All five PNGs were visually inspected. Finger and jaw articulation is visible. **The body
mesh has empty eye sockets and no visible mouth interior**, so eye-bone existence cannot
support a visible-gaze claim. Both eye bones influence 141 vertices each, including 40 body
vertices, but that support does not supply eyeballs, pupils or verified gaze. Teeth/tongue,
eye assets and their mapping remain concrete next requirements under B08/B48. The render
uses neutral clay and 24 un-denoised samples; final appearance and biomechanics are not
qualified by this probe. Its invented local rotations are never labeled observed signing.

The saved file was opened in a fresh Blender process and all five views rerendered. Polygon
identity was preserved; maximum float64-to-Blender-float32 vertex storage discrepancy was
4.77e-7 source units or less. Initial byte-hash comparison failed because PNG metadata
contains the file path, date and render timings; that initial result is preserved. Corrected
comparison found identical decoded pixels for all five images, independently confirmed
through AV decoding. This proves same-machine saved-state pixel replay only, not portable
or cross-device determinism.

Artifacts: [render record](evidence/w1-w2-2026-09-26/rig-render/render-record.json),
[replay verification](evidence/w1-w2-2026-09-26/rig-render/reload-verification.json),
[visual findings](evidence/w1-w2-2026-09-26/rig-render/visual-qa.json), and
[editable diagnostic](evidence/w1-w2-2026-09-26/rig-render/rig-diagnostic.blend).
Reproduce with a new output directory:

```sh
PYTHONPATH=. .venv/bin/python scripts/w2_makehuman_render_probe.py \
  --asset-root 'Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake/makehuman' \
  --out-dir /tmp/signtranslator-rig-render-new \
  --blender /Applications/Blender.app/Contents/MacOS/Blender
```

This advances B47 from adapter-only to actual asset→posed mesh→render/reload. It does not
close real-motion→canonical-state→qualified-avatar acceptance. No training/model runtime
changed in this rendering step; validation was actual execution, inspection, saved-state
replay, independent pixel decoding, compilation and diff checks, not a new full-suite run.

### Pinned rig adapter and actual-mesh articulation — 2026-09-26

`signtranslator/avatar_render/makehuman.py` now loads the exact audited base OBJ, default
skeleton and weights, refusing mismatched bytes against fixed SHA-256 identities. It is an
independent implementation and does not import MakeHuman application code. All positive
source influences are retained and normalized per vertex, with no top-k truncation. Bone
ordering is deterministic and topological; source OBJ polygons and their group memberships
are preserved, including helper geometry that must not silently become visible anatomy.

`load_makehuman_rig(asset_root)` returns a float64 CPU rig. `rig.pose(local_rotations,
world_from_source=...)` consumes one matrix rotation per named bone and returns mesh
vertices, posed global bone transforms and rest-removed skin transforms. Local rotations
follow relative rest transforms. The optional rigid world transform is applied once above
the root; translation remains in unscaled source units. Process frames individually to
bound posed-mesh memory. The adapter checks finite proper rotations, homogeneous rows,
shared dtype/device, hierarchy order, weight support and finite outputs on each use.

Actual-asset tests verify rest reconstruction, the known global rigid transform and its
translation gradient, and both proximal and distal finger articulation against an
independent world-pivot oracle, including derivative parity. Invalid matrices, hierarchy,
mutable tensor corruption and altered input bytes are rejected. This advances B08/B47,
but provides neither a source-motion adapter nor a completed rendered/qualified round-trip.
Physical scale, native float32/morphed-character parity, anatomical axis meaning, facial/eye
renderable assets, motion correspondence and qualified review remain unresolved. Asset
identity checks do not replace action authorization or phase acceptance.

Verification is recorded under `evidence/w1-w2-2026-09-26/rig-adapter/`; the dedicated
adapter and existing rig tests passed 22 checks with warnings treated as errors. The full
warning-strict suite then passed 2,004 tests in 70.68 seconds with no failures or skips;
compilation and diff checks also passed. No
synthetic articulation is described as observed or linguistically accepted signing.

### Rig native-semantics and mathematical geometry audit — 2026-09-26

The next acceptance step now has concrete evidence. Inspected the pinned upstream skeleton
and animation loaders as source text, without importing/executing or vendoring their code.
The [native-semantics record](evidence/w1-w2-2026-09-26/source-intake/makehuman/native-semantics.json)
binds source URLs and SHA-256 identities. The raw weights are intentionally interpreted
through per-vertex normalization by the native loader; its threshold is strictly greater
than 1e-4 after normalization. No positive influences of this pinned asset are removed by
that threshold. A requested influence cap can truncate and renormalize: this is a distinct
policy, not evidence that four weights suffice.

An [executable geometry audit](evidence/w1-w2-2026-09-26/source-intake/makehuman/audit_geometry.py)
revalidates all asset hashes, derives joints from specified mesh-vertex means, constructs
right-handed rest frames with the inspected plane winding, reconstructs the hierarchy, and
exercises the repository's LBS function against rest and common rigid-transform oracles.
The [recorded result](evidence/w1-w2-2026-09-26/source-intake/makehuman/geometry-audit.json)
includes script/runtime identities and versions. Float64 results on the unchanged base OBJ:

- All 163 bones have nonzero length and nondegenerate orientation construction. Minimum
  bone length is 0.0392081624 source units; minimum plane cross norm is 0.0223670707.
- Maximum orthogonality and determinant errors are each 4.45e-16 or less; hierarchy and
  rest-removed identity errors are each 4.45e-15 or less.
- Rest mesh and a known common rigid transform each agree with independent point-coordinate
  expectations to 5.33e-15 source units using all retained influences and an explicit
  partition-of-unity normalization policy.
- Maximum influence count is 12; 3,665 vertices exceed four. An export/runtime cap must be
  explicit and measured, rather than silently dropping weights.

This is independent mathematical verification, not native float32/runtime parity. It does
not establish morphed-rest-mesh equivalence, metric scale, anatomical axes, articulated
signing fidelity, visible eye/face controls, rendering or human acceptance. Raw source bytes
remain unchanged. Next: verify native/rest-mesh and physical conventions, implement the
explicit full-influence asset adapter, inspect/render real articulation, and obtain qualified
source-to-render review. W1/W2 and B08/B47/B48 remain open. No runtime code changed and no
new full model-suite result is claimed.

### Public rig asset intake and source access refresh — 2026-09-26

A concrete public rig candidate is now local: MakeHuman's base OBJ, default skeleton and
skin weights, plus the upstream general and asset licenses. Acquisition is pinned to commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482` and tree
`13dc8663461e51f4e8a90389829266b92846d16a`. Every downloaded file matched its Git blob
identity, byte size and recorded SHA-256. The official
[licensing statement](https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.md)
distinguishes bundled CC0 assets from AGPL application code. No application or plugin was
installed or executed, and the original downloaded asset bytes were not normalized or repaired.

The [intake manifest](evidence/w1-w2-2026-09-26/source-intake/makehuman/intake.json) and
[structural audit](evidence/w1-w2-2026-09-26/source-intake/makehuman/structural-audit.json)
record 19,158 mesh vertices, 18,486 faces, 163 bones, 326 joint references and 139 weight
groups. Parent/joint/mesh indices are valid and the hierarchy is acyclic with one root.
Every vertex has positive weight support. Original per-vertex weight sums range from 0.321
to 1.673, so applying a skinning formula that assumes already-normalized weights would be
unjustified. Native loader normalization, rest axes/roll, physical units, mesh/helper groups,
facial controls and separate eye geometry must be established before an adapter/render test.
Eye and finger bone names are inventory evidence, not proof of gaze or signing fidelity.

This changes the local inventory from no matching downloaded rig assets to a pinned **candidate
asset bundle**. It does not qualify the full rig or change portfolio gates. The earlier empty
filename search remains an accurate pre-acquisition snapshot; `.obj`, `.mhskel` and `.mhw`
were also outside its listed formats. Required next steps are native-format semantic audit,
verified rest/pose transforms and skinning, deterministic render tests and qualified review.

The [SignAvatars repository](https://github.com/ZhengdiYu/SignAvatars) still directs annotation
access through a non-commercial research request form. Its documented SMPL-X layout does not
establish explicit gaze channels, and its stated zero root translation needs to remain distinct
from observed displacement. Neither a request nor agreement was submitted. The ASLLRP landing
page did not yield readable access details in this fetch, so no fresh terms claim is made.
See the [access-route record](evidence/w1-w2-2026-09-26/source-intake/access-routes.json).

Verification here is direct asset integrity and structural analysis, not a model-test rerun.
No runtime code changed. W1/W2 remain unaccepted; the real motion source, qualified linguistic
reference, review governance and source-to-qualified-rig round-trip remain required.

### W1/W2 exit audit against current evidence — 2026-09-26

The [requirement-by-requirement snapshot](evidence/w1-w2-2026-09-26/exit-audit/requirements.json)
records all **15 full requirement rows** below. None is proven complete at its full source,
review and acceptance scope; this does not negate the verified software subsets. Research
and commercial portfolio checks both remain ineligible. The underlying capability check,
even before action-authorization validation, has no qualifying source for any of the four
required bundles: co-observed motion, governed linguistic reference, locally qualified rig,
and review governance. Supplying a license file alone would not close missing capabilities.

A current filename inventory searched the repository and `/Users/jiangshengbo/Volumes`,
including hidden/ignored files except Git/virtual environments, for `.blend`, `.fbx`, `.glb`,
`.pkl` and names containing agreement/authorization/consent. It found no matching files.
This is not proof of universal absence: other formats, filenames, roots and external accounts
are outside that search. The search command, scope and empty result are preserved in the
snapshot. The candidate registry is the currently declared portfolio, not a fresh worldwide
survey of available datasets or rigs.

The remaining acceptance sequence is concrete:

1. Establish an admissible co-observed motion source and exact version/bytes; obtain its
   channel/topology conventions, clock/calibration evidence and action-specific permissions.
2. Establish the exact usable rig and its body/hand/face/eye controls, rest pose, coordinate
   conventions, deformation limits and asset permissions. The pinned MakeHuman candidate still requires
   local qualification and any missing assets; a publisher capability claim is insufficient.
3. Establish qualified annotation/review/adjudication roles and a usable review protocol.
   Use those roles to disposition QC, define the bounded linguistic reference and complete
   a pilot. Software cannot supply these human decisions.
4. Freeze an eligible population and defensible independent-unit study design, then bind
   preprocessing, source-specific gap/calibration policy and preregistration to its artifacts.
5. Implement and verify the concrete source adapter, real canonical shards and rig adapter
   against those actual specifications, then complete source→state→inverse→render tests and
   qualified semantic review. Only then freeze W2 and record an explicit phase-exit decision.

Further source-independent hardening and protocol drafting are possible, but they do not
replace this acceptance sequence. No new model code changed in this audit, so no broad test
rerun was needed. The latest full-suite evidence remains 1,976 passes; the subsequent QC
ledger step has its separate 32-test focused record. The roadmap's workload estimates remain
conditional on acquiring evidence and personnel; no guaranteed calendar completion date is
inferred from green tests or the original planned day numbers.

### W1 source-bound QC disposition ledger — 2026-09-26

A reproducible read-only-source command now joins the original How2Sign audit database,
audit manifest, signer certificate/mapping and source metadata by their recorded hashes.
The join requires unique sample IDs, exact source/signer/status agreement, and complete
metadata membership. Unjoinable filesystem artifacts are kept outside the metadata population.
The source database is opened read-only and inputs are rehashed after processing. The output
must be a new directory outside the preserved source tree; no source file is moved, deleted,
renamed or edited.

The resulting [disposition ledger](evidence/w1-w2-2026-09-26/qc-dispositions/dispositions.csv)
has **31,166 records: 31,165 metadata samples plus one unjoinable artifact**:

| Disposition | Count | Meaning and next required evidence |
|---|---:|---|
| Technical quarantine | 122 | 118 missing sources, 3 structural failures, 1 historical unjoinable artifact; repair/reacquire or explicitly adjudicate with new versioned evidence before reconsideration |
| Pending qualified QC | 28,621 | Historical automated quality warnings; source-bound qualified dispositions and development-only threshold selection required |
| Pending acceptance | 2,423 | Historical structural/automated checks passed; this does not establish linguistic quality, permissions, 3D suitability or training acceptance |

All ledger rows explicitly have `qualified_acceptance=false` and `training_eligible=false`.
This is a logical quarantine/review ledger, not a physical relocation or accepted training
manifest. It cannot authorize model fitting, final splitting or loss-threshold selection.

The command freshly checked all **121 metadata-linked missing/structural failures** through
the existing source reader. All 118 remain missing and all three structural failures recur:

- `0sal9F4RXeY_12-8-rgb_front`: rendered OpenPose video fails decoding.
- `CTERDLghzFw_7-8-rgb_front`: OpenPose frame indices are not contiguous from zero.
- `DQ-FXTeLKs8_1-5-rgb_front`: multiple people make signer selection ambiguous.

The 28,621 warning rows and 2,423 historical valid rows were not re-decoded or human-reviewed.
The extra artifact retains its historical unjoinable classification. These scopes are explicit
in the [summary](evidence/w1-w2-2026-09-26/qc-dispositions/summary.json) and
[technical refresh](evidence/w1-w2-2026-09-26/qc-dispositions/technical-refresh.json).

Reproduce into a new destination with `scripts/w1_qc_dispositions.py --source-root
/Users/jiangshengbo/Volumes --audit-root /Users/jiangshengbo/Volumes/how2sign_audit/v1
--signer-root /Users/jiangshengbo/Volumes/how2sign_audit/signer-evidence-v2 --output NEW_DIR`.
The existing output is never overwritten. Review begins with qualified source-level dispositions
and a development-only sampling/threshold protocol; the ledger does not select the final test
population. B13/B14 advance through explicit accounting, but accepted QC remains open.

Verification: **32 focused tests passed in 1.81 seconds**, covering disposition refusal,
join ambiguity/mismatch and existing How2Sign audit behavior. An independent ledger readback
verified unique IDs, population totals, all-false acceptance flags and the 121 unchanged refresh
statuses. Runtime model code did not change in this step; the prior 1,976-test full-suite record
remains historical. See [verification](evidence/w1-w2-2026-09-26/qc-dispositions/verification.json).

### Temporal gap support and derivative arithmetic — 2026-09-26

`pose/temporal.py` now supports an explicit optional `max_gap_seconds` policy. It must be a
finite positive scalar; booleans and unrepresentable values are rejected. The software does
not choose a universal maximum gap. Supported adjacent frames additionally require
`dt <= max_gap_seconds` when a limit is supplied. Equality is included under the supplied
floating clock/limit representation; no hidden tolerance or interpolation is added. Clock
ordering/finiteness is still validated even when a positive interval exceeds the policy.
Rejected intervals carry false support and harmless computational denominators.

Velocity requires valid endpoints **and** an admissible interval; acceleration requires two
adjacent admissible intervals. Rejected pairs/triples are sanitized before subtraction, so
large values across a rejected gap cannot overflow intermediate derivatives or acquire a
gradient. Acceleration uses `delta_velocity / (dt_left/2 + dt_right/2)`, avoiding overflow
from summing two large finite intervals before division. This remains a local Cartesian
finite-difference estimate, not an assertion that sampled motion identifies true acceleration.

The diffusion velocity objective uses the same interval support and confidence intersection.
If a required velocity term has no admissible interval for any sample, the objective rejects
that sample instead of counting an unavailable zero loss. A cutoff without explicit timestamps
is rejected because frame indices do not establish elapsed seconds. Omitting the policy keeps
the existing unlimited-positive-interval behavior; it does not mean a real source's gap policy
has been reviewed or accepted. Gap rejection affects derivative supervision, not coordinate
availability or the attention context of frames on opposite sides of the gap.

Both paired and bidirectional training, generator fine-tuning, generator validation, analysis
and observation-wise validation preserve the policy. The corpus collator preserves explicitly
supplied sample policies only when every sample supplies timestamps and the same nonempty
policy; mixed/partial declarations are rejected rather than silently dropped. This is an
explicit caller/sample field, not an automatically accepted source-profile setting. Its
source/profile identity and immutable study/checkpoint binding remain part of W1/W2 acceptance.

Tests verify an exact-limit boundary, a large excluded gap, hand-computed velocities,
accelerations and coordinate gradients; overflow-safe interval midpoint arithmetic; invalid
policy domains; required-clock enforcement; velocity MSE on the retained intervals; and
all-unavailable rejection through each active objective route. Collation preserves a common
policy and rejects conflicting or missing ones. Focused verification passed 41 tests after
correcting a test fixture to construct Trainer with its required DataLoader before substituting
an adversarial validation batch. No real source cutoff was selected from test outcomes.

This advances B20/W2 Days 26–30. Source clock provenance, synchronization, calibrated units,
jitter/cutoff sensitivity, policy selection on development data, and the real-source
inverse/render acceptance gate remain open.

Verification: **1,976 tests passed in 68.11 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks passed. See the
[verification record](evidence/w1-w2-2026-09-26/temporal-gaps/verification.json).

### Shared encoder support and requirement reconciliation — 2026-09-26

The shared text/gloss encoder and lightweight speech embedding encoder now require nonempty
boolean sequence support, reject entirely unavailable samples and enforce input layout,
vocabulary/feature domains and positional capacity. Missing speech values are sanitized before
projection. Masked pooling uses `where(mask, value, 0)` and divides by the actual available
count, rather than multiplying NaN by zero or clamping an empty denominator into a nominal
embedding. Nonfinite available features are rejected. Text IDs excluded by an explicit mask
are replaced with the padding ID before embedding; available IDs must be in vocabulary and
cannot be the padding ID. Per-token output payloads are zero off support.

The older `SignTranslator` paired-training API now also forwards validity, frame support and
confidence to its motion encoder and diffusion objective, with optional physical timestamps
routed to the objective. These are the same support semantics already used by the active
bidirectional path; the API no longer requires callers to discard those fields. No new
parameter or state-dict tensor is introduced.

Verification includes a hand-computed masked mean and its Jacobian; shared speech/text
training and evaluation feature/parameter-gradient invariance under masked NaN/Inf values
and added padding; masked out-of-range token IDs; zero unavailable sequence payloads; empty
support and malformed-domain rejection; and a legacy paired-training integration case that
checks the derived denoiser support. Focused checks passed 40 tests. As elsewhere, exact
padding comparisons use dropout zero; source-specific missingness semantics and empirical
linguistic utility are not established by these fixtures.

The requirements matrix below has been reconciled with current implementations. In particular:
scoped canonical ingestion, affine inverse support and active encoder propagation are no longer
listed as wholly absent. Their real-source application and phase acceptance remain absent.
This does not move training on accepted canonical multichannel state into the legacy Cartesian
model: that integration still needs the actual source topology, available channels, calibrated
transforms and an approved canonical-state contract.

Verification: **1,962 tests passed in 67.90 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks passed. See the
[verification record](evidence/w1-w2-2026-09-26/shared-encoder-support/verification.json).

### Acoustic prefix support and subsampling audit — 2026-09-26

The active `models/speech.py::SpeechRecognizer` now carries input lengths through the
feature path, not just CTC. It validates a nonempty floating `(N,T,F)` tensor with the
configured feature width and an integer `(N,)` length vector in `[1,T]`. Omitting lengths
means the entire supplied sequence is available; nonfinite values in that support are
rejected. Declared padding is sanitized before convolution. Each convolution updates its
valid prefix and zeroes hidden padding before the next layer, so biases/receptive fields
cannot turn padded positions into input to a later convolution. Transformer key masks and
final hidden masks preserve the resulting support.

For kernel 3, padding 1, dilation 1 and stride s in {1,2}, the convolution length is
`floor((L-1)/s)+1 = ceil(L/s)`. Sequential stride-two layers therefore produce
`ceil(L/subsample)` for supported subsample values 1, 2 and 4. The integer implementation
uses `floor((L-1)/2)+1`, avoiding overflow from `L+1` at the largest signed integer.
Odd-length and small-length cases are verified against actual emitted tensor lengths.
Positional-capacity overflow is explicitly rejected.

CTC receives these exact lengths without silently clamping excessive input declarations.
Greedy decoding slices each sample at its post-subsampling endpoint so classifier bias in
padded hidden frames cannot emit extra tokens. `recognize_speech`, evaluation analysis and
`translate_audio_to_sign` propagate optional true acoustic lengths; joint training already
supplied them and now benefits from feature-level enforcement.

Deterministic paired models verify training/evaluation features, CTC losses, input gradients
and parameter gradients under added NaN padding for subsampling 1/2/4. Ragged batch rows
also agree with individual unpadded encoding. Tests include padding inside the original
batch, zero gradient outside each valid prefix, integer-boundary arithmetic, malformed
length vectors, nonfinite available audio, padded decode emissions and the active training
entry point. Focused verification passed 45 tests.

This is a **prefix availability** contract. Missing or corrupt audio inside a declared valid
segment is rejected when nonfinite; it is not silently interpolated or converted to silence.
Source-specific dropout/gap representation, physical acoustic feature provenance and actual
backend validation remain source-adapter work. Deterministic padding comparisons use dropout
zero; no identical-random-draw claim is made for different-shaped stochastic runs. No model
parameter/state-dict layout changes were needed. B19's active acoustic prefix defect is
repaired, while the complete W2 source/availability and phase-acceptance requirements remain
open.

Verification: **1,955 tests passed in 67.52 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks passed. See the
[verification record](evidence/w1-w2-2026-09-26/speech-support/verification.json).

### Denoiser attention and conditioning support — 2026-09-26

Both repository denoisers now accept explicit boolean `(N,T,V)` motion support.
Coordinates are sanitized before input projection. A frame is an available attention key
iff at least one joint is supported; unavailable frame keys are excluded in every motion
self-attention layer. Output coordinates remain zero off support. Every sample must contain
observed support, supported values must be finite, and positional-capacity overflow is
explicitly rejected. Interior unavailable frames retain their original positions rather than
being packed into a different time sequence. The pooled denoiser disables nested-tensor
packing so this position/support contract uses the dense masked path.

`GaussianMotionDiffusion.p_losses` derives the denoiser mask from the exact positive objective
support, including validity, frame availability and positive confidence. Callers cannot
override it with a conflicting `motion_mask`. Guided diffusion inherits this path while
retaining classifier-free conditioning dropout. The two repository denoisers explicitly
advertise mask support; arbitrary external/custom denoisers without that capability retain
only the objective/input sanitation contract and have **no feature-level padding guarantee**.
Sampling APIs still request fully generated clips; this change is not masked inpainting or
an assertion that generated missing joints were observed.

Cross-modal language memory is also sanitized before projection. Masks and classifier-free
drop vectors require exact boolean shape/device contracts. Nonfinite available memory is
rejected; masked or dropped memory may contain arbitrary nonfinite placeholders without
influencing predictions or gradients. The learned null token remains available even when no
language token is available, preventing an all-masked cross-attention row. This numerical
unconditional path does not override the application's linguistic-evidence refusal gate.

Verification uses nonzero output-projection weights: zero-initialized predictions cannot
trivially pass the tests. Paired model copies test training/evaluation predictions, input and
parameter gradients under NaN padding; masked coordinates receive zero gradient. Guided
`eps` and `x0` objectives are checked with fixed identical supported noise, including the
velocity term for `x0`. Language tests cover masked NaNs, all-dropped nonfinite memory,
null-context parity and zero unavailable-memory gradients. Input/capacity domains and mask
overrides are rejected. Focused verification passed 45 tests.

The deterministic comparisons use dropout zero and fixed noise/timesteps. Changing tensor
shape can change random-number assignment for dropout/noise in a stochastic run; identical
seed alone is not a coupling of those draws, and no such exact-realization claim is made.
No new state-dict tensor is introduced. Existing partial-observation model metrics still need
re-evaluation, since attention no longer treats padded tokens as evidence. This advances B19
for the active denoiser; speech feature support, remaining encoder families and real-source
acceptance are still outstanding.

Verification: **1,937 tests passed in 65.41 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks passed. See the
[verification record](evidence/w1-w2-2026-09-26/denoiser-support/verification.json).

### ST-GCN support propagation and active recognition — 2026-09-26

The active motion encoder now accepts joint validity, frame support and confidence. Boolean
support is `validity AND frame_mask AND confidence>0`; positive confidence magnitudes remain
loss reliability weights, not fractional sample counts in feature normalization. Frame masks
must be contiguous prefixes. Every sample must have observed support. Omitting all support
arguments retains the existing fully-observed encoder path.

Support now controls each stage: input sanitation, per-joint/channel data normalization,
graph-convolution source values **and biases**, graph output locations, temporal/residual
normalization, post-block outputs, joint pooling and frame pooling. Hidden values at unavailable
locations are zeroed at every block; they cannot re-enter a later convolution as observed
features. This is a conservative missing-observation policy, not a claim to recover unseen
articulation. Learned adjacency is not renormalized after missingness; no missing edge is
silently replaced with another anatomical relation.

For each BatchNorm channel, moments use only supported entries. Forward variance uses divisor
n; running variance uses `n/(n-1)` only for n>1. A singleton updates the mean but retains the
running variance; an unsupported channel retains both. Invalid NaN/Inf values are sanitized
before moments and receive zero gradients. Existing BatchNorm parameters/buffers and state-dict
keys are reused; the supported implementation requires fixed momentum and tracked statistics.
It is not a distributed synchronized normalization implementation. Existing stride-one ST-GCN
blocks are supported; masked strided blocks explicitly refuse an undefined support reduction.

Per-frame features average available joint locations, and clip features average frames with
support. Joint training forwards the masks through its shared recognition/alignment encoder;
validation, retrieval embedding and analysis preserve them too. Recognition loss derives lengths
from frame support when needed and rejects inconsistent explicit lengths. Joint training no
longer silently clamps excessive CTC lengths. Recognition decoding trims padded emissions,
including classifier-bias emissions after the true endpoint.

Verification includes hand-computed moments/running-buffer updates, singleton/unsupported
channels, all-true parity with the existing encoder, invalid-value substitution, empty-sample
rejection, pooled estimands and prefix validation. Paired model copies test NaN padding in
training and evaluation: supported outputs, input gradients, parameter gradients and running
buffers remain equal within numerical tolerance. The active recognition training path also
passes a padded-batch loss/gradient test; decoding excludes an injected padded token.

This is progress on **B19**, not its complete closure. BatchNorm training still intentionally
depends on the other observed samples in the batch; padding invariance is not batch-composition
invariance. ST-GCN encoder blocks use zero dropout by default; arbitrary stochastic dropout
streams are not asserted padding-invariant. Denoiser attention/convolutions, speech features,
other encoder families, and real-source missingness/qualified acceptance remain to audit.
Existing checkpoints retain the same ST-GCN tensor layout, but their calibration and metrics
must be re-evaluated under the new partial-observation semantics. No empirical gain is claimed.

Verification: **1,927 tests passed in 65.56 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks passed. See the
[verification record](evidence/w1-w2-2026-09-26/encoder-support/verification.json).

### Affine inverse and coordinate-information audit — 2026-09-26

`data/corpus.py::PoseStandardizer` now validates nonempty `(C,1,V)` finite
statistics with strictly positive standard deviations, exact channel/joint compatibility,
float32/float64 input, mask shape/type/device, representability after dtype conversion,
and finite transformed output. It rechecks mutable statistics at use. Input precision is
preserved; no accidental float64-to-float32 conversion of pose values is introduced.
`SignDataset` now passes validity into the transform, rather than masking only after
normalization. Both normalize and inverse operations sanitize unavailable values before
arithmetic, preserve their zero payload and exclude them from gradients. Omitting the mask
explicitly declares every entry valid; callers must retain the mask through inverse export.

For fixed training statistics, `z=(x-m)/s` and `x=z*s+m` are inverse affine maps on
valid support in exact arithmetic. Floating-point reconstruction is tested within tolerance,
not promised bit-identical. Analytic examples verify the normalized values and Jacobian
`dz/dx=1/s` on supported entries and zero off support. Tests cover nonfinite missing
payloads, all-unavailable tensors, batched padding, dtype conversion underflow/overflow,
mutated statistics, and governed-export → dataset → inverse parity. All-unavailable output
is an unavailable zero payload; it is not a successful loss or usable training example.

The coordinate-information audit distinguishes what is and is not recoverable:

| Operation/path | Information retained or discarded | W2 consequence |
|---|---|---|
| Corpus affine standardization | Fixed training mean/std plus validity are retained | Inverse restores source coordinates on valid support; it does not infer metres, camera calibration or 3D depth |
| `preprocess.PoseNormalizer` root centering + per-sequence RMS | Current return value discards root trajectory and scale; this helper is not the active corpus standardizer | Its output alone cannot reconstruct the original motion; retain explicit transform context before using it in a reversible adapter |
| Temporal resampling | Interpolation/downsampling need not be injective | Original clock and source samples are required evidence; upsampling is not recovery of lost frames |
| Perspective/weak-perspective projection | Depth information is discarded by a 3D-to-2D map | A camera inverse cannot manufacture metric 3D supervision from 2D observations; source calibration and justified fitting provenance remain required |
| Shape-parameter replacement | `MotionSequence.retarget` shares motion tensors but changes body identity | Tensor equality is not proof of preserved contact, geometry or signing meaning; real rig and qualified review remain necessary |

The first full run exposed a diagnostic integration regression: corrupt valid pose values
now correctly raised at normalization, but `assess_corpus` needed to report that rejection.
It now records a failed normalization check and continues pose-quality reporting; malformed
statistics produce a failed corpus-load check. It never substitutes repaired values or
fabricated summary statistics. Focused transform/export/readiness verification passed
42 tests after that repair; the failed first-run evidence is retained alongside the rerun.

This advances the normalization portion of B47 and W2 Days 26–30. It does not close B47,
B42, B19, source-specific temporal-gap policy, or the real source→state→rig acceptance gate.
The legacy model remains Cartesian and source-specific transform profiles are not frozen.
Verification: **1,917 tests passed in 65.89 seconds**, with warnings treated as errors.
Compilation, dependency and diff checks also passed. The first run (1 failure, 1,913 passes)
and successful rerun are both retained in the [verification record](evidence/w1-w2-2026-09-26/affine-inverse/verification.json).

### Statistical design audit and working study protocol — 2026-09-26

**Status: draft design, not preregistered or accepted by qualified reviewers.** Source/QC,
useful-effect thresholds, independence assumptions and study precision are not yet supported
by the required project evidence. Recording proposed choices below does not imply approval.

The endpoint-direction audit found that the general `significant_and_meaningful` helper
uses absolute effect magnitude. The registered positive-improvement protocol now additionally
requires `effect > 0`; significant degradation and zero improvement cannot pass. Effects
must be oriented before comparison: model minus baseline for higher-is-better outcomes,
baseline minus model for lower-is-better outcomes. This repairs the protocol's semantics
without changing the general two-sided magnitude helper for other uses.

`eval_framework/design.py` computes an exact sign-test resolution bound with rational
arithmetic. With n non-tied independent paired units, the smallest two-sided sign-test
p-value is `min(1, 2/2^n)`. The family threshold is alpha/m for m Bonferroni-controlled
primary endpoints. At alpha=0.05, at least six units are needed for one endpoint and seven
for two; these are necessary resolution counts, **not powered sample-size recommendations**.
Ties, mixed signs, small effects and imperfect independence make the situation worse.

The current six-component mapping cannot supply six independent test components while
also reserving nonempty training and validation partitions: at most four remain. Their
best-case p-value is 1/8, above 0.05. Even treating all six as test units yields 1/32,
which exceeds 0.025 for a two-endpoint family. The exact count inputs were rechecked against
the original certificate/mapping hashes. See [resolution audit](evidence/w1-w2-2026-09-26/design/resolution-audit.json).
This limits this declared paired sign-test design; it does not prove that every inference
method is impossible. Choosing an asymptotic test solely to obtain a smaller p-value is
not a remedy for unverified independence or low information.

| Design element | Working proposal / required decision | Acceptance evidence still needed |
|---|---|---|
| Population | Keep the Day-1 bounded meeting/scheduling slice; define eligible speakers/signers, linguistic phenomena, source conditions and intended users | Qualified domain/phenomenon inventory and accepted source coverage; no open-domain claim |
| Sampling frame | Hash the accepted source/QC inventory before partitioning; retain rejected/missing row counts separately | QC dispositions, source rights and actual eligible population |
| Unit and grouping | Use connected signer/source components as provisional independent units; windows, augmented versions and repeat renderings never create new units | Dependence audit and a defensible sampling interpretation; connected-component separation alone is not independence proof |
| Candidate primary endpoint | Unconditional task comprehension from blinded qualified receivers, scored against independently accepted source propositions; count abstention/invalid output as task failure, and report conditional quality plus coverage separately | Accepted proposition rubric, question design, independent reviewer qualifications and refusal policy |
| Aggregation | Score one canonical utterance once; average within each component, then average component differences with equal component weight. Report clip-weighted and signer subgroup summaries as secondary diagnostics | Frozen utterance deduplication, component membership and handling of crossed source/signer/reviewer effects |
| Effect direction | Signed improvement over the frozen baseline; lower-is-better outcomes invert the subtraction | Exact score implementation and independent hand-scored examples |
| Comparator | Freeze admissible retrieval/interpolation and relevant trained baseline outputs using identical eligible support; independent human reference can be a separate ceiling, not training supervision leaked into test | Reviewed baseline suitability, same-support hashes and resource/training-data accounting |
| Minimum useful effect | Do not invent a numerical threshold from current scores. Elicit a domain-relevant threshold before final testing, record its rationale and unit | Qualified/user decision and pilot interpretation; threshold must not be chosen after final outcomes |
| Precision and power | Run a separately labelled pilot; estimate component and crossed-reviewer variance; simulate the actual planned design over prespecified effect scenarios. Check exact finite-sample resolution first | Independent units beyond the current attainable holdout, pilot variance evidence, simulation assumptions/code, power or interval-width target |
| Multiplicity | Prefer one genuinely primary endpoint only if justified before outcomes; otherwise retain the declared family with Bonferroni. Required grammatical/safety failures remain separate acceptance gates | Frozen family and gate hierarchy; do not drop a primary after seeing significance |
| Review and blinding | Randomized presentation order, concealed system identity, independent receiver/reviewer roles, conflict disclosure, adjudication and per-tier disagreement reporting | Actual personnel, availability, consent and a tested review interface; no software-generated human approval |
| Exclusions/missingness | Freeze technical/rights exclusions using development evidence. Do not exclude model failures post hoc. Report excluded, unavailable, abstained and scored denominators by stratum | Source-bound disposition ledger and frozen missing-data sensitivity plan |
| Calibration/selection/test | Fit preprocessing only on training data, select on validation, calibrate on separately assigned data if needed, then lock model/protocol/data hashes before reserved test access | Real partition manifest, adequate independent units, durable access history and fresh evaluation reserve for remediation |
| Uncertainty | Match intervals and comparisons to the sampling hierarchy; distinguish training-seed variation from population and reviewer uncertainty. No IID frame/clip bootstrap masquerading as independent evidence | Implemented and validated estimator, sufficient clusters and explicit small-sample limitations |
| Reproducibility | Preserve versioned inputs, seeds, interventions, row-level scores, aggregation code and hashes; rerun after any accepted protocol change | Immutable accepted protocol and real study artifacts, not this draft |

The immediate design action is to obtain an eligible independent-unit sampling plan and
qualified endpoint/precision decisions. Increasing clip count within the existing dominant
component or counting training seeds as new participants does not meet that requirement.
No final split or QC threshold is frozen by this audit.

Verification: **1,909 tests passed in 65.35 seconds**, with warnings treated as errors.
The directional regression cases and exact finite-sample oracle checks passed; these
validate software contracts, not study power or empirical acceptance. See
[verification record](evidence/w1-w2-2026-09-26/design/verification.json).

### Multichannel state and scoped ingestion implementation — 2026-09-26

`pose/multichannel.py` adds a source-parameterized, motion-only state format. This is
software schema version 1, **not an accepted/frozen real-source representation**. It requires
explicit root translation, body/left-hand/right-hand rotations, head rotation and translation,
face coefficients, left/right gaze, blink and contact. Root/head/gaze elements are singletons;
joint/feature labels and conventions are supplied by the source rather than guessed from a
default 27- or 55-joint layout. Units are metres for translation and unitless for the declared
rotation/scalar representations; coordinate-frame names, convention versions, source IDs and
source SHA-256 values are mandatory. A named frame is not proof of calibration or handedness.

Each channel carries separate validity, observation-origin and confidence arrays. Confidence
is bounded reliability, not a calibrated probability. Observation origin and validity are
independent: an observed measurement may have been rejected as invalid. Valid inferred values
require a named inference method. `supervision_weights()` excludes invalid and inferred values
by default; admitting inferred targets requires an explicit opt-in and a reviewed adapter protocol. Invalid payloads are explicitly zero with zero confidence so they cannot
leak NaNs through another encoder. Zero payload does not make a channel observed or valid.
Geometry checks reject degenerate 6D rotations, non-unit gaze, out-of-range blink/contact,
nonfinite valid payloads, wrong units and malformed labels. A shared float64 clock carries
seconds and must have finite, strictly increasing intervals.

Construction, save and load validate the contract. Serialization uses NPZ without pickle,
strict array/metadata inventories, duplicate-key rejection, explicit dtypes and a required
expected content hash on load. Save refuses existing paths and revalidates mutable NumPy
payloads. Tests check exact canonical-payload reload, inferred/unavailable provenance,
invalid geometry/time/units, changed bytes and structural tampering with a recomputed hash.
This is lossless for the canonical payload; it does not claim recovery of discarded raw
measurements, source-native annotations or sensor packets.

`data_engineering/state_ingestion.py:export_phase2_state` integrates the new state format
with the scoped W1 portfolio assessor. It checks all required source/rig/reference/governance
bundles, source-specific permissions and evidence bytes before writing. Valid motion channels
must bind one eligible co-observed source and its actual local content hash; independently
recorded modalities cannot silently become a synchronized single-clock observation. Returned
records always retain `phase_exit_approved=False` and list the outstanding source mapping,
synchronization, inverse-transform/render and qualified-review evidence.

No real source has been accepted, no human judgment has been fabricated, and no rig/render
round-trip is implied. The active legacy Cartesian model does not become a multichannel
model by the addition of this format. Source-specific kinematics, coefficient meaning/ranges,
clock alignment, inference-model evidence and training/render adapters remain W2 work.

Verification: **1,896 tests passed in 66.35 seconds**, warnings treated as errors, zero failures/errors/skips.
The first full run passed 1,895 tests; the final run adds the observation-origin/validity
regression. Dependency and compile checks passed. [Verification evidence](evidence/w1-w2-2026-09-26/multichannel/verification.json) records source and log hashes.

### Versioned action-scope policy — 2026-09-26

`data_engineering/phase2_policy.py` implements `phase2-action-scope-v1`. New Phase-2
eligibility assessments must explicitly choose typed research or commercial scope through
`assess_phase2_scope`; the historical combined `CURRENT_PRE_PHASE_2_DECISION` remains a
legacy compatibility result, not the new scoped decision. It is unchanged and still closed.

Both scopes require co-observed multichannel motion, qualified governed linguistic reference,
a body/hand/face/eye rig and qualified review governance. The linguistic-reference requirement
uses governed source-native supervision rather than mandating authentic gloss as the only
possible representation; English text or synthetic tokens cannot satisfy it. The entire
co-observed motion bundle must come from one eligible source; no cross-source stitching is
allowed inside a bundle. Research does not require commercial grants, but it still requires
actual research rights, local source evidence and all research capabilities.

For each bundle, eligibility additionally requires a source-bound `DataAuthorization`, its
requested use and actions, consent state and matching local evidence bytes. Motion/reference
use requests derivative creation plus model training; rig/governance use requests derivative
creation. Commercial scope additionally requests commercial use. A generic rights flag cannot
supply a missing action. A permission bound to another source, missing file or mismatched
hash fails. Source binding is an explicit recorded claim; file hashing does not establish
legal authority, the legal meaning of a document or truth of human qualification claims.

Outputs are labelled **portfolio eligibility**, include policy version/scope/action failures,
and always leave `phase_exit_approved=False`. Public deployment, redistribution and identity
use require separate action evidence. No software assertion here substitutes for the W2 real
state/synchronization/render/qualified-review exit. The new assessor must be integrated at the
future real-state ingestion boundary; no current source is newly authorized or promoted.

Focused tests check research/commercial separation, action omission, wrong-source binding,
file tampering, explicit scope and both current real-portfolio failures. Full verification:
**1,876 tests passed in 66.37 seconds**, warnings treated as errors, no failures/errors/skips.
Dependency and compile checks passed. [Verification and current scope decisions](evidence/w1-w2-2026-09-26/action-policy/verification.json) retain source/log hashes and the still-closed real gates.

### Follow-on protocol and temporal implementation — 2026-09-26

The in-process evaluation firewall now rejects train/validation tuning after explicit test
access or primary reporting. It binds the registration hash at first test access and rejects
subsequent replacement. Direct construction and factory construction both validate unique
nonempty endpoint names and an exact family of finite, nonnegative minimum improvements.
The protocol hash now binds the family alpha, positive-improvement direction and Bonferroni
rule; per-endpoint confirmation uses `family_alpha / number_of_primary_endpoints`.
Post-hoc alpha changes are rejected. Existing hashes are historical and are not silently
reinterpreted as hashes of the new protocol. This is an API workflow guard, not a durable
security boundary: recreating Python objects cannot make a previously inspected test set
fresh. Persistent access provenance and the complete scientific study design remain open.

Cartesian temporal utilities now evaluate interval velocities as `delta_x / delta_seconds`
and three-point acceleration as `2*(v_right-v_left)/(dt_left+dt_right)`. Velocity belongs to
an interval; acceleration is the quadratic-interpolant second derivative, not a claim of
continuous physical observation between frames. Support requires both endpoints for velocity
and all three samples for acceleration. Missing/padded targets are sanitized before arithmetic;
unsupported outputs carry false support. Clocks must be finite and strictly increasing on
valid prefixes. Invalid shape, nonpositive interval, nonfinite supported time or overflow
fails explicitly. Source-specific maximum-gap and synchronization contracts remain required.

`frame_timestamps` now propagates through joint training, macro-observation validation,
analysis and generator-only training/validation to the active diffusion objective. With an
explicit clock measured in seconds, velocity loss uses elapsed seconds. Omitted clocks retain
the declared legacy synthetic per-frame convention; validated real v2 batches carry clocks.
Coordinates are still in their supplied (often normalized) units: metric-space interpretation
requires the outstanding calibrated inverse transform. Loss coefficients must be reassessed
before real training because per-second and per-frame penalties have different scales.
Both saved Cokely source-native manifests were also inspected: the v2 manifest retains the
same missing HD primary binding and only adds the SD file as auxiliary evidence. It does
not resolve B67 or authorize substituting SD as the primary timed source.

Tests include an independent irregular-clock quadratic oracle (acceleration exactly six),
linear-motion resampling/time-origin checks, missing-sample gradients, invalid clocks,
a hand-computed diffusion loss and failure injection proving active paths cannot discard
clock metadata. A first focused run caught a misplaced clock-validation block in the noise
sampler; it was moved to the loss entry point and the focused suite rerun successfully.

Verification for these follow-on changes: **1,870 tests passed**, warnings treated as errors, zero failures/errors/skips.
Dependency, compile and whitespace checks passed. See [verification](evidence/w1-w2-2026-09-26/protocol-time/verification.json).
The first full run had 1,869 passes and one independent-oracle comparison differing by
approximately 1e-15 after float64 clock promotion. The independent numerical comparison
now uses 1e-12 relative/absolute tolerance; exact equality across loader partitions remains
required and passed. The original failure log is retained alongside the final rerun.

### Current authoritative findings and repairs

1. **Grouping feasibility is substantially worse than signer counts alone imply.** The
   hash-verified published pseudonymous mapping has 31,165 rows, but transitive signer/source
   connectivity yields only six components. The largest holds 26,806 rows. An exhaustive
   search over all `3^6 = 729` assignments with nonempty partitions proves the smallest
   possible maximum absolute deviation from 70/15/15 is **0.1601315578** (16.013 percentage
   points). A minimax allocation has counts 26,806 / 4,067 / 292. This is a feasibility
   result, not a recommended or accepted evaluation design.
2. Excluding 118 missing-source and three structural-failure rows leaves **31,044 candidate
   rows**, with **28,621 quality-warning rows still awaiting QC acceptance**. The largest
   connected component contains 26,695 rows. Exact minimax counts are 26,695 / 4,057 / 292,
   with maximum proportion error **0.1599085169**. A 292-row partition is not 292 independent
   signers; component/signer-level inference and attainable precision must drive W1 design.
3. **Split reproducibility repaired:** canonical component keys are sorted before seeded
   randomization; input-row permutation no longer changes assignments. Ratios now require
   three finite nonnegative numbers summing to one; seeds require nonnegative integers.
   Empty populations, duplicate sample IDs and blank signer/source IDs cannot be certified.
   Boolean assignment indices cannot masquerade as integer sample indices.
4. **Vocabulary leakage repaired at export:** source and target vocabularies are fitted
   only on the assigned training records. The manifest records the fit split, exact fit
   sample IDs and unknown-token policy. Held-out unknown forms cause explicit export refusal
   before any destination is written. This closed-vocabulary path does not claim to support
   unseen signs; it neither silently substitutes a known sign nor drops held-out records.
   A future external lexicon or meaningful unknown-token model requires its own frozen contract.
5. **Rotation defects reproduced and repaired:** zero 6D inputs previously returned a matrix
   outside SO(3); the identity geodesic returned zero with nonfinite gradients. Gram–Schmidt
   now rejects zero/collinear or numerically unresolved axes, uses scale-safe normalization,
   and rejects zero quaternions and nonfinite vector inputs. Matrix conversions/geodesic
   require SO(3), rejecting reflections and scaled transforms. Angle evaluation now uses
   `atan2(||vee(R-R^T)||/2, (trace(R)-1)/2)` for the relative rotation. Endpoint values are
   preserved without an arbitrary acos floor. Distance is intrinsically nondifferentiable
   at identity and the pi cut locus; a finite autograd convention is not proof of a unique
   mathematical derivative there. Supported vector types are explicitly float32/float64.

Source evidence: [split feasibility](evidence/w1-w2-2026-09-26/split-feasibility.json).
Reproduce with `scripts/w1_split_feasibility.py` and the existing
`/Users/jiangshengbo/Volumes/how2sign_audit/signer-evidence-v2` directory, writing to a new
output file. The script verifies the certificate's mapping hash and row count, exhaustively
checks allocations, records implementation hashes, and leaves source evidence unchanged.
It reuses historical QC statuses; it does not re-decode every video or accept warning rows.

Rotation-reference cross-check: [PyTorch3D transform documentation](https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html)
documents Gram–Schmidt 6D conversion and acos endpoint stability concerns. This repository
uses **columns** for its 6D convention; PyTorch3D describes a row convention. They must not
be mixed without an explicit transpose. The new numerical evidence comes from local
independent angle cases, scale stress, SO(3) invariants and finite-difference gradcheck.

### Full completion requirements and evidence still required

| Requirement / planned days | Current state | Evidence required for completion |
|---|---|---|
| W1 source/rig inventory and action-specific authorization; 11–15 | Existing 2D reference inspected; pinned MakeHuman candidate assets acquired and structurally checked; no accepted multichannel source or qualified rig | Exact source/rig versions, bytes, research actions, distribution/derivative limits and verified admissible samples |
| W1 research/commercial policy reconciliation B09; 11–15 | Versioned scoped assessor and adversarial tests implemented; legacy gate unchanged | Bind real source/rig evidence to the implemented scoped canonical ingestion gate; no commercial approval inferred from research |
| W1 reviewer workflow; 11–15 | Qualified creator/reviewer/adjudicator evidence not supplied | Named qualified roles, independence/conflict handling, actual availability, accepted consent/review process |
| W1 QC and corruption dispositions; 11–20 | Hash-bound ledger created; 121 missing/structural sample failures refreshed unchanged; 122 total technical quarantines including one unjoinable artifact; no QC acceptance | Qualified dispositions for warning/acceptance queues, development-only stratified review, frozen QC thresholds and versioned repair/exclusion decisions |
| W1 grouping/split; 11–20 | Mathematical feasibility audited; deterministic splitter repaired | Scientifically defensible allocation, actual accepted population, source/signer certificate and frozen sample assignments before windows |
| W1 preprocessing/vocabulary; 16–20 | Export fit is training-only; unknown export refused | Real accepted partition applied, source/label coverage report and immutable preprocessing identity; affine support audit implemented |
| W1 pilot annotation protocol; 16–20 | Existing governance primitives; no completed qualified pilot | Frozen domain tiers, admissible labels, timing instructions, double-review/adjudication and per-tier agreement/coverage with independent review |
| W1 estimands/preregistration; 16–20 | Signed endpoint/family lock, exact resolution audit and draft study protocol implemented; accepted full preregistration absent | Population, independent units, primary outcome/direction, useful-effect threshold, confidence/precision or power, exclusions, multiplicity, calibration and final-test access rules |
| W2 typed state and shards; 21–25 | Source-parameterized multichannel contract, strict serialization and scoped ingestion implemented; no real accepted shard | Bind source-specific semantics/kinematics, verify actual provenance and calibration, and produce admissible real shards |
| W2 geometry domains; 21–25 | Rotation domain/stability repairs and multichannel geometry checks tested; source-specific geometry unaccepted | Full domain/invariant audit, degenerate/near-pi cases, unit gaze, local/global handedness, source evidence and typed unavailable channels |
| W2 adapters and synchronization; 26–30 | Timestamped 2D readers exist, canonical multichannel adapter absent | Source-native time origin/rate, exact synchronization/calibration, channel mapping, inference provenance and validated source-specific fixtures |
| W2 derivatives and inverse transforms; 26–30 | Explicit-clock derivatives, caller-declared gap exclusion and masked affine inverse tested; accepted source-specific physical transforms/gap policy absent | Nonuniform-time velocity/acceleration, endpoint support, gap policy, unit conversion, sampling/jitter invariance and inverse-transform parity |
| W2 missingness and padding; 21–30 | Active ST-GCN, denoiser, acoustic-prefix and shared-encoder support implemented; real canonical-state integration and source-specific gaps absent | Attention/convolution/pooling support contract, padding/occlusion gradient tests and genuine availability/inference separation |
| W2 source→state→render round-trip; 31–35 | No accepted real multichannel source-to-authorized-rig result | Real-sample inverse transform/render evidence, left/right fingers/palm/face/gaze/head/contact/time checks and qualified review |
| W2 freeze and exit; 31–35 | Not accepted | All predecessor source/rights/review requirements met, integrated regression, schema freeze, reproducible real round-trip and explicit acceptance decision |

**Protocol audit disposition:** the unused test-access flag and unfrozen multiplicity/alpha
were repaired in the follow-on implementation above. The full statistical design and durable
access provenance remain required before W1 protocol acceptance.

**External-state check:** the hash-bound Cokely primary HD file remains absent at its recorded
path. No source, signer mapping, historical audit, source manifest or human approval was
rewritten. Locating any newly acquired source/rig/authorization/reviewer evidence has been
requested while source-independent implementation continues.

Current verification: **1,848 passed in 68.14 seconds**, warnings treated as errors,
zero failures/errors/skips. Dependency, compile and whitespace checks pass. Current source
hashes, exact command and evidence hashes are recorded in
[progress verification](evidence/w1-w2-2026-09-26/progress-verification.json).
These results verify the repairs above; neither W1 nor W2 has passed its full exit gate.

---

## W0 Days 6–10 execution record — 2026-09-25

**Status: complete and verified through Day 10. W0 engineering exit accepted.** The records below
supersede the historical open W0 findings without changing any empirical gate.

### Days 6–7: evaluation contract and selection

Canonical analysis now unpads each observation before convolution, pooling, acoustic
recognition, target encoding or generation. It preserves each motion/speech length and
splits acoustic CTC references using their own target lengths. No concatenation of ragged
token batches is required. Generation receives observation masks/confidence and cycle
sampling uses the observation's declared duration. This is an evaluation boundary; full
padding/occlusion invariance during joint batched training remains B19 in W2–W3.

Planner decoding has a declared maximum of 32 steps in analysis (configurable up to the
512-position implementation capacity). A reference requiring more than that cap including
EOS is rejected explicitly. Missing EOS is reported as truncation and cannot pass the
planner gate. Reports include corpus edit rate, exact sequence match and insertion-aware
normalized edit accuracy. The historical `planner_token_accuracy` key is retained with its
new definition recorded in the report; old numbers and thresholds are not calibrated to
this new definition. Semantic-field accuracy is explicitly unavailable: the canonical
synthetic token branch has no governed SIR field references. No token match is promoted to
a semantic judgment. Governed field evaluation remains W3/W6 work under B28/B39.

Validation and generation analysis use the **macro-observation estimand**: normalize each
observation over its own valid coordinates and valid adjacent pairs, then give each eligible
observation one vote. Planner loss averages within each sample, including EOS; CTC retains
its target-length-normalized per-sample loss. Each branch has its own eligible sample count.
Alignment uses the full fixed validation cohort as candidates, not the current loader batch.
This fixes unequal final batches and support-dependent batching without claiming independent
signer observations or fixing false negatives between equivalent meanings (B29).
Training epoch loss logs remain optimizer-batch diagnostics, not evaluation estimands.

`TrainerConfig.selection_metric` freezes a branch loss to minimize, defaulting to
`generation`; weighted total loss cannot be selected as the primary checkpoint criterion.
The config is bound into resumable artifacts. Analysis after joint training explicitly loads
best weights, records the artifact path/hash, and restores final optimizer-associated weights.
Without a saved artifact, the selected best in-memory weights are named and content-hashed.
An explicitly loaded checkpoint is reported by its own identity. Secondary-stage analysis
is rejected up front because those stages do not have an accepted selection/resume contract.

Analysis records seed, one stochastic replicate, exact weight hash, support counts, cap and
estimand. Python/NumPy/Torch RNG state and every module's prior train/eval mode are restored
on success and failure. A single seeded replicate is a reproducible engineering diagnostic,
not uncertainty estimation; clustered/multi-seed scientific reporting remains W6.

### Day 8: canonical package and ownership map

The engineering owner for the following mappings is Codex, with the project lead accountable
for scope changes. Qualified ASL specialists own linguistic conventions and acceptance;
software tests cannot take that role. Alternative modules stay available for isolated research
but are not alternate sources of truth for the canonical runtime.

| Responsibility | Canonical implementation now | Alternative / future boundary |
|---|---|---|
| Executable and optimization | `run.py`, `training/trainer.py`, `TrainerConfig` | `train.py` / `SignTranslator` is a legacy synthetic two-branch experiment; not the production or resume entry point |
| Corpus and topology | `data/corpus.py`, governed `data_engineering/exporter.py`, `skeleton/graph.py` | No direct governed SIR shard adapter or accepted multichannel 3D state yet |
| Source-token planning | `models/planner.py` | `planning/` and `grammar/` own governed SIR contracts; no automatic substitution of synthetic token outputs |
| Active motion generation | `models/guided_diffusion.py` on `models/diffusion.py`, wired by `models/pipeline.py` | `diffusion_gen/` and `motion_transformer/` are separately tested candidates; replacement requires W4/W5 comparison and integration evidence |
| Compact acoustic recognition | `models/speech.py` | `speech/` contains advanced contracts/components; raw-waveform service is not wired into the executable |
| Sign recognition and alignment | `models/recognition.py`, `models/stgcn.py`, `models/alignment.py` | Gloss IDs are diagnostic outputs, not full ASL-to-English translation |
| Canonical branch evaluation | `analysis/report.py`, `analysis/observations.py`, `eval/metrics.py` | `eval_framework/` owns scientific contracts, statistical primitives and human-study interfaces; it does not supply missing study results |
| Geometry and rendering | `pose/`, `hand_graph/`, `facial_nmm/`, `avatar_render/` interfaces | Future canonical state and real source-to-rig round-trip remain externally gated |
| Release controls | `deployment/` contracts | No live production service or hardware qualification inferred |

**Leakage design frozen for W1 implementation:** construct a bipartite signer/source graph;
assign whole connected components to partitions before extracting windows. All sessions,
translations, annotation revisions, augmented views and derivative motion from a source stay
with that source. Unknown signer/source identity is quarantined rather than assumed unique.
Fit QC cutoffs, normalization, vocabularies and tokenization only on training/development
partitions as preregistered. A truly external closed lexicon is permitted only when its
version/hash and choice predate held-out inspection. Unseen test forms map to a declared
unknown/refusal path and contribute coverage; never rebuild the vocabulary from all records.
Calibration, model selection and final test are separate; do not tune on test failures and
report the same test as confirmatory. Persist component assignments, seed, source/signer
counts, hashes, overlap assertions and exclusions. B11/B21 implementation and real split
certification remain W1/W3, not completed by this design document.

The research/commercial policy discrepancy B09 remains a named W1 policy dependency:
source-independent W0 repair does not invoke the pre-Phase-2 portfolio gate. Existing
executable gates remain unchanged and closed. Any future action-scoped policy revision must
version requirements and prove research acceptance cannot imply commercial acceptance;
no present authorization is inferred from this mapping.

### Canonical engineering metric registry

| Metric / location | Unit, aggregation and direction | Availability / meaning |
|---|---|---|
| Generation validation / trainer and analysis | Per-observation supported coordinate loss plus supported adjacent-difference term; arithmetic mean; lower | Every required observation/temporal objective must have support; normalized coordinates, not physical-time error |
| Planner validation / trainer | Per-sample teacher-forced CE including EOS, mean over eligible samples; lower | Synthetic token targets; not semantic or ASL accuracy |
| Recognition and speech validation / trainer | Target-length-normalized CTC per sample, mean over branch-eligible samples; lower | Exact CTC feasibility remains required |
| Alignment validation / trainer | Symmetric diagonal InfoNCE on the full validation cohort; lower | Candidate population must stay fixed; repeated meaning false negatives remain B29 |
| Weighted total / trainer | Sum of named branch estimands with frozen config weights | Diagnostic only; excluded from primary selection |
| Recognition, speech and planner WER / analysis | Sum of edit distances divided by reference token count; lower | Insertions count; acoustic coverage is reported and partial coverage fails whole-corpus acoustic gate |
| Planner normalized edit accuracy / analysis | `1 - sum(edit_distance)/sum(max(hyp_length, ref_length))`; higher | Historical token-accuracy key; not comparable to old positional accuracy |
| Planner exact match and truncation / analysis | Fraction of fully matching EOS-terminated sequences; fraction without EOS | Truncation always fails planner acceptance, independent of content match |
| Retrieval R@1/R@5 / analysis | Mean diagonal retrieval success over fixed validation candidates; higher | Embedding diagnostic; no semantic equivalence or intelligibility claim |
| Cycle WER / analysis | Corpus edit rate over first declared subset in loader order; lower | Seed, subset count and duration recorded; shared-recognizer agreement is not independent human validation |
| Semantic-field accuracy / analysis | Unavailable until governed field references and predictions exist | Never filled with zero, token accuracy or fabricated labels |
| Sign/permutation/bootstrap / `eval_framework/statistics.py` | Validated finite-score statistical primitives | W0 tests numerical contracts only; independent units, clustering, power and multiplicity remain W1/W6 |

Current automatic thresholds remain synthetic engineering checks. They are not preregistered
real-ASL endpoints. Registry changes require a versioned protocol and fresh comparisons;
semantic, perceptual and scientific acceptance belongs to the later phase gates.

### Days 9–10: integration and acceptance evidence

**1,832 tests passed in 68.09 seconds**, with warnings treated as errors and zero failures,
errors or skips. Compilation and dependency checks passed. The source/test/config inventory
was unchanged during verification. A no-Git source archive was extracted and built into an
installed wheel; its custom-topology smoke completed four optimizer steps, selected-best
analysis, distinct best/last artifacts, finite output and bit-identical seeded last reload.

Evidence: [full verification](evidence/w0-complete-2026-09-25-attempt1/verification.json),
[warning-strict test log](evidence/w0-complete-2026-09-25-attempt1/pytest.log), and
[completion audit](evidence/w0-complete-2026-09-25-attempt1/completion-verification.json).
`tests/test_w0_analysis.py` adds 16 adversarial/integration cases. The full suite also retains
all Days 2–5 numerical, masking, topology, finite-update and exact-resume regressions.
An early focused invocation named a nonexistent test file; the corrected focused invocation
then exposed a test fixture using a list instead of the trainer's required DataLoader. The
fixture was repaired and all final checks were rerun; neither early attempt was acceptance.

| Day / requirement | Current acceptance evidence |
|---|---|
| 1: baseline, scope, owners, dependencies | Preserved Day-1 baseline and dependency ledger below; historical hashes retained |
| 2: statistical domain and tail repairs | Independent integer probability oracle, invalid-input tests and current full-suite results |
| 3: support-safe objectives | Mask/confidence, zero-support, gradient and active-path tests remain green |
| 4: topology and joint ordering | Alternate topology, manifest/export and mismatched-checkpoint tests; installed-wheel custom graph |
| 5: finite/nonempty updates and CLI | Failure-injection and positive-step tests; checkpointable parser/API defaults |
| 6: ragged and length-aware analysis | Unequal token/motion/speech lengths; padded NaNs; insertion/EOS and ten-token reference adversaries |
| 7: estimands, selection and RNG | Unequal batch partition equality, independent macro-loss oracle, declared branch selection, exact best artifact inspection, RNG/mode restoration |
| 8: package ownership and leakage design | Canonical package map, metric registry and signer/source/vocabulary rules above; implementation dependencies remain explicitly assigned |
| 9: integrated baseline | Entire 1,832-test warning-strict suite, compile and dependency checks |
| 10: archive, install, reload and decision | Extracted source archive → installed wheel → train → selected-best analysis → last reload; hashed evidence and completion audit |

**Decision:** W0 engineering stabilization is accepted. W1 source/QC/split/protocol work is
next; Phase-2 empirical acceptance and production deployment remain unapproved. Completing
scheduled deliverables early does not claim 60 human hours elapsed or automatically advance
external delivery dates.

The existing Days 1–5 evidence remains immutable historical evidence of those exact bytes.
The full-phase runner is `scripts/w0_verify.py`; reproduce into a new output directory.
It checks the complete warning-strict suite, dependency consistency, compilation, source
identity, an extracted no-Git source archive, wheel installation, custom-topology training,
selected-best analysis and last-checkpoint reload. It reuses the pinned local numerical
environment; Linux/GPU/production validation is not claimed.

W0 acceptance does not accept Phase 2, resolve missing source media B67, grant rights,
create human reviews, or close the remaining B19/B20/B29/B37–B39 research work. Those
remain concrete dependencies in the existing ledger and schedule.

---

## W0 Days 2–5 implementation record — 2026-09-25

**Status: implementation and verification complete through W0 Day 5.** Day 1 remains preserved as
historical baseline evidence. This task repairs the numerical and integration defects assigned
to Days 2–5; it does not claim completion of Days 6–10 or real-data/ASL empirical readiness.

### Day 2 — Statistical domains and stable probability tails

`eval_framework/statistics.py` now validates nonempty, one-dimensional finite real score
vectors, equal paired shapes, finite differences, probabilities, effect/alpha values and
integer resampling controls. Invalid values raise rather than becoming significant evidence.
Bootstrap statistics and quantiles must also be finite real scalars. Exact permutation
enumeration is bounded to 20 pairs; the Monte Carlo path requires at least two total
assignments and a nonnegative integer seed. Normalizing differences before the permutation
comparison removes the old absolute-unit tolerance artifact and avoids overflow in the
scale-invariant t statistic.

The two-sided sign test computes the smaller binomial tail directly with integer arithmetic:

`p = min(1, 2 * sum(comb(n,i), i=0..min(k,n-k)) / 2**n)`.

It no longer subtracts a CDF rounded to one. Sixty positive differences now return exactly
`2^-59 = 1.734723475976807e-18`, symmetrically for sixty negative differences. Genuine
probabilities below binary64's subnormal range can still underflow; no arbitrary epsilon
floor is invented. An all-tied, nonempty sample returns p=1; an empty sample is invalid.

Evidence: `tests/test_ef_statistics.py` covers independent `math.comb` oracles, hand-counted
permutation probabilities, scale invariance, invalid domains and overflow. Clustered sampling,
power, independent-unit design and scientific endpoint selection remain W1/W6 work (B37–B39).

### Day 3 — Observed-support motion objectives

The base and guided diffusion paths accept frame masks, per-joint validity and confidence.
The active joint trainer and generator-only fine-tune path propagate them. Targets outside
positive supported weight are replaced before noising/denoising, so changing an unobserved
value—even to NaN—cannot supply a training signal through the denoiser's receptive field.
The observed prediction and target domains are validated.

Coordinate reduction is `sum(w * squared_error) / sum(w)` over supported coordinates. The
confidence weight is declared reliability, not calibrated probability. Every sample must
have positive support. The temporal difference objective uses only adjacent pairs with both
endpoints supported; pair reliability is the minimum endpoint reliability. A required velocity
term with no usable pairs is unavailable and raises, rather than contributing a fabricated
zero. Nonfinite supported loss raises before optimization.

Evidence: `tests/test_w0_motion_support.py` checks hand-calculated weighted coordinate and
velocity losses, exact zero target gradients at invalid positions, invariance under replacing
invalid targets with NaN, all-empty/partially empty batches, missing temporal edges, both
x0/epsilon parameterizations and base/guided models, plus actual joint/fine-tune propagation.

Boundary: this does not yet make every feature pooling/attention path padding invariant
(B19), nor replace index-space temporal differences with physical-time derivatives (B20).
Length-aware evaluation and propagation through the separate analysis aggregator remain Day 6
and W2/W3. Omitted masks explicitly retain the legacy fully-observed direct-call contract;
validated v2 batches already require and carry observation metadata.

### Day 4 — Skeleton topology and joint semantics

`SkeletonGraph` has a strict versioned serialization of node count, unique ordered joint
names, undirected edges and center. It rejects invalid index types, duplicate undirected
edges, disconnected graphs and malformed names/schema. Model construction accepts an explicit
graph and checks its size. Non-default joint counts no longer inherit the 27-joint edge set.

An optional skeleton can be attached by the governed exporter; it must exactly match exported
joint names/order. Corpus validation checks declared topology. The runtime requires topology
for real/non-default corpora; only the explicit legacy 27-joint synthetic format retains its
compatibility default. Storage-only old exports without topology remain readable but cannot
silently become trainable arbitrary skeletons.

The checkpoint model contract includes topology and joint ordering, not merely tensor shapes.
An incompatible ordering is rejected before weights load. Old checkpoints lacking that
contract are not silently migrated; a separately reviewed migration would be required.

Evidence: `tests/test_w0_topology.py` constructs and executes a 137-node engineering graph,
checks manifest/export round-trips, malformed graphs and order mismatches, and verifies a
mismatched checkpoint leaves receiving weights unchanged. The isolated-wheel smoke additionally
trains and reloads a declared five-node synthetic graph. Neither graph is asserted to be a
validated anatomical or ASL representation.

### Day 5 — Nonempty training, finite updates and checkpointable defaults

Runtime loaders preserve partial/tiny batches (`drop_last=False`). Trainer construction and
execution reject empty loaders/iterators, including separately supplied custom loaders.
Every scalar branch/total loss and every available gradient is checked before the optimizer
step; absent gradients and nonfinite clipping norms also fail. Error context includes epoch,
batch, optimizer-step position and available sample IDs. Failed pre-step checks cannot advance
the optimizer, scheduler or step counter or overwrite the previous saved checkpoint.
Validation rejects nonfinite/empty evidence. Generator-only training carries the same support
and gradient safeguards, but its existing exact-resume limitation remains explicit.

Trainer configuration rejects nonfinite numerical values, nonintegral epoch/batch settings,
and empty/negative/nonfinite/all-zero loss-weight configurations. CLI and API now share one
joint epoch, learning rate `3e-4`, and zero generator-fine-tune/polish epochs by default.
Checkpoint use therefore no longer conflicts with the default secondary stages. A one-epoch
quality evaluation can still legitimately fail; mechanical smoke success is not model quality.

Evidence: `tests/test_w0_training_guards.py` injects nonfinite loss and gradient independently,
checks parameter/optimizer/scheduler/step/checkpoint preservation, checks tiny-corpus updates
with oversized batches, rejects invalid configurations and verifies actual CLI parser defaults.
Existing resume/RNG regression tests remain part of the complete suite. Mean-of-batch-means
aggregation and primary checkpoint selection are still Day-7 work (B24/B25), not falsely closed.

### Verification, evidence and acceptance boundary

Final verification: **1,816 passed, zero failures/errors/skips**, with warnings treated as
errors (63.012 seconds for the test command). Dependency and compile checks passed. The
installed wheel completed four optimizer steps with declared custom topology, distinct
best/last checkpoints, finite output and bit-identical seeded last-checkpoint reload.
The recorded source/test/config hashes were unchanged throughout verification.

Machine-readable results and hashed logs:
[verification.json](evidence/w0-days2-5-2026-09-25-final/verification.json).
The earlier 1,812-test run is interim evidence; the final run includes the four additional
integration/parser/export tests. These are engineering checks, not empirical ASL approval.

Reproduction command (new output directory only):

```bash
.venv/bin/python scripts/w0_days2_5_verify.py \
  --output /tmp/signtranslator-w0-days2-5-new-verification
```

The runner records source/test/config hashes before and after verification, executes strict
regression/dependency/compile checks, builds a wheel from the **current bytes in a no-Git
source copy**, installs it into an isolated target, verifies that imports resolve there, then
trains a tiny synthetic custom-topology model and checks finite, bit-identical seeded reload.
The same pinned dependency environment is reused; this is not a fresh production OS/device
qualification. Small synthetic readiness overrides in that geometry smoke are explicit and
cannot pass empirical gates. Temporary source copies, corpora and checkpoints are cleaned.

Day-1 evidence and its runner describe pre-repair behavior. Reproducing that historical
baseline requires its recorded source bytes; executing its old defect probes against repaired
code is not the same experiment. The new runner is the current repair-verification entry point.

| Scheduled deliverable | Result / evidence |
|---|---|
| Day 1 baseline, scope, owners and dependencies | Preserved historical record below; no external approval inferred |
| Day 2 stable small-p/domain checks | B35/B36 repaired; independent numerical oracle and invalid-domain tests |
| Day 3 observed-support loss/gradient tests | B18 repaired in active joint and generator-fine-tune objectives; explicit required-support failure |
| Day 4 topology/joint-order/reload tests | B17 repaired with declared topology, manifest checks and checkpoint binding |
| Day 5 positive training-step and finite-update guards | B23/B31 repaired for the canonical and generator training paths |
| Day 5 bounded checkpointable defaults | B33 repaired; README and actual parser/API agree |
| Empirical readiness decision | **Not approved**: qualified real SIR/motion/reference/rights evidence remains absent; no qualifying human experiment was run |
| Source integrity | B67 remains open: the recorded Cokely HD primary was still missing when checked; real data was not changed |

All other audit findings keep their previous status unless explicitly named above. The next
scheduled work is Day 6: ragged and length-aware evaluation, long-sequence behavior and
insertion-aware planner metrics. Preserving that boundary prevents these W0 engineering
repairs from being mislabeled as successful real translation.

---

## W0 Day 1 execution record — 2026-09-25

**Deliverable boundary:** reproducible baseline, frozen engineering pilot scope,
accountability/ownership, dependency ledger, and source/reviewer requirements. This executes
the first scheduled workload early; it does not claim six human hours elapsed. The six-hour
figure remains the schedule's effort allocation. Day-2 statistical repairs and subsequent
model changes are deliberately separate deliverables. W0 itself is not complete.

### D1-A — Frozen baseline and verification evidence

Baseline checkout: `89e2238d371ae8e0af826282520f4c14416bd9c5`. The changes between the previous
audit's `839acf3...` and this checkout are documentation only; `signtranslator/`, `tests/`,
packaging/lock and CI source have no intervening changes. The Day-1 runner records the actual
code/test/configuration file hashes and requires that they remain unchanged through capture.
The added audit runner is outside the model package and does not change model behavior.
An initial capture failed on an incorrect nested-source path in the new runner; its test
logs and failure note are retained under `evidence/w0-day1-2026-09-25-attempt1/`. Only the
corrected complete capture linked below is the Day-1 evidence. No initial failure is hidden.

Evidence: [baseline JSON](evidence/w0-day1-2026-09-25/baseline.json),
[full test log](evidence/w0-day1-2026-09-25/pytest.log),
[JUnit test evidence](evidence/w0-day1-2026-09-25/pytest.xml), and
[reproduction runner](../scripts/w0_day1_baseline.py). Independent completion checks,
all 18 installed lock-version comparisons, hardware details and blocker coverage are in
[completion verification](evidence/w0-day1-2026-09-25/completion-verification.json).

**Current capture result: complete.** All 1,726 tests passed with zero failures/errors/skips.
Compile/dependency checks passed; eight synthetic optimizer steps produced finite output
and bit-identical seeded last-checkpoint reload. **Source integrity FAILS:** eleven recorded
files match and the Cokely HD primary is missing.

The runner records dependency consistency, package/test compilation, the entire warning-strict
suite, one-epoch synthetic training, best/last checkpoint creation, seeded last-checkpoint
reload equality, output finiteness, the five reproduced defect observations B17/B18/B23/B35/B36,
portfolio/Phase-3C fail-closed results, and the local evidence hashes described below.
Capture completion is not a passing model-quality, mathematical-correctness or ASL gate.
Known wrong behavior stays explicitly wrong even when it is reproducibly measured.

To reproduce into a **new directory** from the repository root:

```bash
.venv/bin/python scripts/w0_day1_baseline.py \
  --output '/tmp/signtranslator-w0-day1-new-capture' \
  --data-root /Users/jiangshengbo/Volumes
```

The destination must not already exist or lie inside the data root. Synthetic corpus and
checkpoints use a fresh temporary directory and are deleted after the run. The runner neither
regenerates nor writes anything under the real data root. Synthetic metrics/latency are not
real-data performance claims. The tests provide the existing resume and validation-RNG
regressions; the separate smoke explicitly checks seeded reload behavior.

Day-1 evidence refresh checks the signer mapping, audit manifest/database, metadata CSV,
Cokely inspection artifacts, exact named EAF/media/publisher-page payloads and EAF schema
against their recorded hashes. The HD primary video is currently missing (B67/D23), so
overall source integrity FAILS; present matching files do not offset that failure. The
capture records the saved 2D experiment without retraining it.
How2Sign's entire raw media population is not rehashed or decoded; Cokely byte matching is
not a fresh linguistic review. These narrower boundaries are part of the evidence record.

### D1-B — Engineering scope freeze: `W0-PILOT-SCOPE-v1`

This is the execution baseline selected under the Day-1 planning task. The optional domain
question was offered; absent a different preference, everyday meeting scheduling is the
engineering assumption. It is not a claim that the user approved ASL labels, appointed an
external reviewer, or accepted any scientific performance threshold. Changes are versioned
with reason and downstream effect instead of silently changing evaluation scope.

| Dimension | Frozen engineering choice / remaining evidence |
|---|---|
| First direction | English text → governed SIR → coordinated multichannel signing → deterministic avatar. Full speech/streaming and reverse ASL→English remain in W7–W10; they have not been removed from the project goal. |
| Domain | Single-turn everyday meeting scheduling. Candidate meanings cover a meeting's time and whether a stated schedule is correct. No calendar transactions or claims about actual appointments. |
| Linguistic contrasts | Declarative versus yes/no-question intent and changes in time reference. The same proposition with different sentence function must remain distinguishable. Exact ASL realization is assigned to qualified reviewers, not inferred from English word order. |
| Language/variety | ASL, limited to the source's documented variety and the qualified lead's reviewed convention. No universal-dialect claim. Exact variety/convention identity is an explicit prerequisite D04; no source-independent dialect or lexicon is fabricated. |
| Candidate input fixtures | “The meeting is tomorrow.” / “Is the meeting tomorrow?” / “The meeting is today.” These are English scope illustrations ONLY: not training pairs, ASL gloss, approved SIR, or accepted render prompts. |
| Vocabulary and coverage | A versioned, qualified-human-reviewed coverage set. Specific lexeme IDs, number/time inventories, spatial conventions and acceptable variants stay unset until D04/D05; the software must abstain outside accepted coverage. |
| First-slice exclusions | Multi-turn discourse, unrestricted proper names/fingerspelling, unrestricted numeric/date expressions, role shift and open-domain translation are outside this pilot acceptance population. They remain potential later scope, not claims of solved capability. |
| Motion | Body, dense left/right hands, head and the face/gaze/eyelid/contact channels required by the accepted linguistic convention. Never omit a required grammatical channel simply because a source lacks it. Non-identifiable inputs retain an unavailable/inferred status. |
| Audience and output | Internal research evaluation with qualified ASL reviewers; timestamped avatar video and a separate inspection record of meaning/SIR/provenance/refusal. No public accessibility or interpreter-replacement deployment. |
| Baseline hardware | Local Apple M4 Max, Mac16,5, 38,654,705,664 bytes = 36 GiB memory (read via `sysctl`). CPU is the reproducibility reference. MPS/CUDA acceleration, service capacity and production latency are unvalidated and not assumed. |
| Primary quality outcome | Blinded recovery of the intended semantic proposition AND correct declarative/question distinction from the rendered signing; collect eligible-trial numerator/denominator, reviewer disagreement and severity. A prettier render does not satisfy this endpoint. |
| Statistical commitment | Split by source/signer before windows; freeze real conventions/coverage and preprocessing; preserve paired comparisons; account for clustered observations. Numerical effect/coverage thresholds and sample size are preregistered after pilot calibration in W1, before confirmatory outcomes. Unset thresholds cannot yield acceptance. |
| Baseline comparison | Reviewed lexical retrieval/composition or another qualified acceptable simple baseline, plus representation-level interpolation on matched support where applicable. A mean-pose baseline alone does not establish semantic usefulness. |
| Success boundary | Accepted source/annotation/rig evidence + reproducible real input-to-render execution + passing independent linguistic and statistical gates. Day-1 scope is a contract for work, not evidence these gates passed. |
| Stop behavior | Unknown/ambiguous/unauthorized forms, missing required modality, invalid coordinates/times, missing observation support or incompatible schema fail closed; no invented gloss, blank-motion fallback or synthetic approval. |

### D1-C — Accountability and ownership register

Roles below assign **work accountability**, not credentials or third-party commitments.
The requester/project lead is the decision and procurement owner by project role; this does
not assert a new agreement, purchase or availability commitment. Codex executes authorized
engineering work in this repository. No external person is represented as hired or assigned.
A specialist role remains `unfilled` until a real person accepts and qualification and
independence evidence are verified. Filling those roles is a later dependency, not a Day-1
claim of staffing completion.

| Workstream | Responsible execution owner | Accountable / acceptance owner | Present status and deadline |
|---|---|---|---|
| Baseline, numerical correctness, reproducibility | Codex engineering lane | Project lead; evidence reviewed against declared contracts | Assigned for this task; Day-1 capture and W0 repair sequence |
| Pilot engineering scope / change control | Codex drafts and maintains | Project lead | `W0-PILOT-SCOPE-v1` planning baseline; user changes create v2 |
| Data inventory, hashes, schema and QC | Codex engineering lane | Project lead; qualified lead for linguistic QC decisions | Technical inspection available; final source/QC acceptance W1 |
| Source access, rights and reviewer recruitment coordination | Project lead/requester | Project lead plus actual source/asset permission issuer | No outreach or purchasing performed by this task; dependencies due D20 |
| ASL convention and primary annotation | Qualified ASL linguistic lead — UNFILLED | Independent qualified reviewer for label acceptance | Identity, qualification, scope and availability required before W3 collection |
| Blind independent review | Distinct qualified ASL reviewer — UNFILLED | Qualified adjudicator for disagreement | Must independently view exact source/motion; not model self-review |
| Disagreement adjudication | Third qualified person — UNFILLED | Recorded adjudication under the frozen protocol | Must differ from primary and blind reviewer |
| Statistical design / implementation | Codex engineering lane | Project lead and qualified study lead; independent methods review if available | No fabricated external statistical sign-off; preregistration before final scores |
| Avatar/source model qualification | Codex for numerical/coordinate validation | Qualified ASL reviewers for articulation; issuer for permission | Technical rig evidence and qualified review absent; W1–W2 dependency |
| Empirical release decision | Codex assembles evidence | Project lead with qualified-human and action-specific rights evidence | Not authorized by the Day-1 baseline; W6/W9 |

Engineering ownership of B01–B67 is exhaustive: Codex maintains technical evidence and
implementation tasks for every item. The project lead coordinates external inputs for
B01–B08, B10–B15, B39–B41, B48–B52, B59, B64–B67. Qualified specialists own linguistic
judgments in B02–B04, B10, B14–B15, B39–B44, B46–B52, B59 and B64. Issuers alone supply
source/asset permissions for B06–B08. No engineering owner may self-certify those external
judgments. The existing 66-item audit plus the Day-1 B67 addendum remain the per-issue closure specification.

### D1-D — Dependency ledger

Statuses are evidence states: `observed`, `missing accepted evidence`, `unfilled`,
`not executed`, or `policy unresolved`. A deadline is a planned gate, not an assurance of
availability. This ledger is for dependency management; it is not a machine authorization
manifest and must never be loaded as one.

| ID | Dependency / audit blockers | Present state | Responsible owner; needed by | Evidence required to close / successor blocked |
|---|---|---|---|---|
| D01 | Continuous co-observed motion source; B01/B06/B15 | Missing accepted multichannel source | Project lead acquisition + Codex inspection; D20 | Exact authorized samples, schema, synchronized channels, provenance and units → W2 |
| D02 | Research permissions by action; B06/B59 | Existing limited source evidence, no final-bundle authorization | Project lead + actual issuer; D20 and each new source | Media, annotations, model derivatives, retention/distribution scope tied to exact artifacts → research training/export |
| D03 | Qualified people; B02/B51/B65 | Primary/reviewer/adjudicator unfilled | Project lead; confirm availability in W1 | Accepted roles, qualification, independence, review time and compensation arrangements → annotation/study |
| D04 | ASL variety, convention and lexicon; B03/B10/B43 | Missing accepted artifacts | Qualified linguistic lead + independent reviewer; before W3 collection | Exact convention and lexical IDs/variants; source-compatible coverage and rejection rules → SIR population |
| D05 | Accepted SIR and lexical motion; B03/B04/B16/B40 | Software exists; approved population absent | Qualified team, Codex adapter; D50 | Hash-bound independently reviewed records and authorized canonical motion → W4/W5 |
| D06 | Canonical state; B05/B17–B20/B42 | Components exist, source-grounded state absent | Codex + source spec owner; D35 | Units/frames/topology/time/availability + real round-trip → W3/W4 |
| D07 | Rig/model assets; B08/B47/B48 | No accepted complete rig bundle | Project lead + issuer + Codex + ASL reviewer; D20 sample, D35 round-trip | Exact asset identity, permission, hand/face/gaze fidelity → rendering |
| D08 | Split and independent units; B11/B12/B21/B37/B38 | Signer key verified; final split absent | Codex + study lead; D20 | Frozen source/signer components, feasible counts, unseen vocabulary policy, estimand → valid evaluation |
| D09 | QC and corruption dispositions; B13/B14/B15 | Saved audit/queue observed; final thresholds absent | Codex + qualified lead; D20 | Reviewed stratified cases, dev-only threshold, explicit missing/corrupt exclusions → eligible corpus |
| D10 | Research/commercial gate reconciliation; B09/B60 | Policy unresolved; current executable gates closed | Codex proposal + project lead; W0/W1 | Explicit versioned policy and adversarial tests, no bypass → stage authorization |
| D11 | Statistical numerical repairs; B35/B36 | Reproduced wrong outputs | Codex; W0 D2 | Finite-domain checks, stable tails and independent oracle cases → trustworthy statistics |
| D12 | Objective/loader/topology correctness; B17–B20/B23/B31 | Reproduced or code-confirmed issues | Codex; W0 D3–D5, full support W2–W3 | Valid support gradients, bound graph identity, nonempty/finite step safeguards → real training |
| D13 | Evaluation semantics; B24–B28/B39/B49 | Code gaps; real protocol not frozen | Codex + study lead; W0 D6–D8 and W1 | Correct ragged/length behavior, weighted estimands, named checkpoint, endpoint/threshold lock → candidate selection |
| D14 | Canonical packages and training strategy; B29/B30/B32–B34/B61/B62 | Partial implementations and bounded guarantees | Codex; W0 D8, W4/W7/W9 | Declared owners, multi-positive/curriculum design, measured hardware path and appropriate resume proof → scale |
| D15 | Representation usefulness; B41/B44 | 2D experiment exists, no demonstrated baseline advantage | Codex; D65 | Common-support baseline comparison, coverage and minimal-pair evidence → generator representation |
| D16 | Generator integration and interventions; B40/B43/B45/B46/B50 | Not executed on accepted real state | Codex + qualified reviewers; D85 | Tiny real overfit, conditioning/video interventions, constraints, multiple seeds, reload/render → study candidate |
| D17 | Study, calibration and sample budget; B37–B39/B49–B52 | Protocol primitives only | Qualified study team + Codex; design W1, results D110 | Preregistered independent units, blind semantic recovery, agreement, risk/coverage and fresh reserved evaluation → research acceptance |
| D18 | Speech, streaming, real acoustic evidence; B53–B55 | Components, no accepted end-to-end path | Codex + source/review owners; D130 | Real waveform backend, timestamps, actual errors, committed-prefix replay → audio scope |
| D19 | Runtime, capacity, release; B22/B56–B58/B62 | Unmeasured/unintegrated | Codex + project lead; D150 | Actual hardware load profile, bundle reproduction, parity, monitoring and rollback → restricted operation |
| D20 | Commercial lineage; B07/B08/B59 | Missing accepted authorization | Project lead + issuer; before any commercial action | Written action-specific data/model/rig evidence → commercial work; not promised by research schedule |
| D21 | Reverse direction; B64 | Sign→gloss exists, full ASL→English absent | Codex + qualified evaluation team; D175 | Real-video baseline/decoder, English meaning recovery, calibration and separate gate → bidirectional claim |
| D22 | Resources, documentation and progress states; B63/B65/B66 | Calendar assumptions; external capacity uncommitted | Project lead + Codex; every five workdays | Actual annotation/compute throughput, remaining artifacts, software/integration/empirical states separately recorded → credible reforecast |
| D23 | Missing Cokely primary HD media; B67 | Absent at manifest/binding path; SD alternate is not a substitute | Project lead acquisition + Codex byte verification; before any Cokely re-ingest/review | Restore the exact hash-bound HD payload through an authorized route, or separately version and validate an explicit new binding/timebase; historical bundle stays unchanged → Phase-3A/3B reuse |
| D24 | Independent-unit resolution and precision; B68/B11/B12/B37–B39 | Current candidate exact sign-test design cannot reach its threshold with at most four held-out components | Study lead + Codex + qualified reviewers; before W1 protocol freeze and W6 study | Eligible independent-unit acquisition/partition, accepted estimand and useful effect, pilot-based power or precision analysis; clips and seeds do not add participants → confirmatory evaluation |

All B01–B67 appear in the dependencies above directly or in a bounded numeric range. Day-1
completion means the ledger exists and has honest owners, inputs, deadlines and stop rules;
it does not mean these later-stage dependencies are resolved.

### D1-E — Source intake requirements, ready for actual acquisition

For every proposed source/asset package, obtain the following before production schema freeze:

1. **Release identity:** source owner, official origin, version/date, exact file inventory,
   hashes, known defects, subset and matching source/transcript/annotation IDs.
2. **Use evidence:** permission issuer and exact permitted actions for media, annotations,
   fitting, training, derivative weights, rendering and redistribution; retention/withdrawal
   constraints where applicable. Keep research and commercial evidence separate.
3. **Co-observation:** per-record table for body, left/right hands, head, face, gaze and
   eyelids/contact where supplied. Label each observed, fitted, generated, interpolated or
   absent. Separate datasets cannot be asserted to co-observe one performance.
4. **Geometry:** joint hierarchy/order, rest pose, local/global transforms, units, handedness,
   rotations, shape parameters, camera intrinsics/extrinsics and calibration uncertainty.
   A `z` field is not metric depth merely because it has that name.
5. **Time:** clock and timestamp units, source time origin, synchronization map, variable frame
   intervals, gaps, dropped frames and drift; annotation endpoint/untimed-slot semantics.
6. **Missingness:** valid/observed masks, confidence interpretation and calibration status,
   occlusion, fitting residuals, interpolation policy and zero-support behavior.
7. **Linguistic identity:** ASL variety, exact native tier conventions, qualified annotation
   provenance, alignment granularity and permitted mapping. English is not gloss; EAF is not
   project SIR until qualified mapping and review pass.
8. **Grouping:** pseudonymous signer, recording/session/source, duplicate lineage and any
   reuse across releases. No personal identity inference; split before windows.
9. **Small compatibility package:** exact licensed source media plus the corresponding
   annotations/motion/schema and known failure examples. Determine support from real files
   before writing a full converter or committing to a final state dimension.
10. **Rig-specific supplement:** mesh/skeleton/blendshape identities, expression/gaze semantics,
    dependencies, deployment/export permission, deterministic renderer and qualified hand/face
    visibility/articulation review.

Decision outcomes: `quarantined-uninspected`, `compatible-for-declared-research-action`,
`needs-clarification`, or `rejected-for-this-use`. Compatibility is not linguistic acceptance.
Do not mark a source accepted if a required channel or permission is unresolved. Preserve raw
files and native tiers; derived outputs are separate and versioned.

The Cokely formal-speech reference is an ingestion compatibility source, not evidence of
meeting-scheduling coverage. Its missing HD file prevents current replay of its historical
binding. No source is promoted into the pilot merely because files or references exist.

Existing source leads are already documented in `04_DATA_ENGINEERING_AND_CORPUS.md` §13.
The Day-1 work reuses those prepared inquiries instead of generating duplicate correspondence.
The public How2Sign #28 and SignAvatars #20 pages were read again during this task; both
retrieved pages display Open and no reply was visible. The How2Sign web result was cached
three days earlier, so no stronger live no-reply assertion is made. No inquiry was sent,
access terms accepted, data downloaded or purchase made by this task.

### D1-F — Qualified-review requirements, ready for recruitment and protocol work

Recruit a qualified ASL linguistic lead and an independently working qualified reviewer;
provide a distinct qualified adjudicator for disputes. Record their actual scope of expertise,
relevant ASL variety, experience in annotation/avatar comprehension, declared conflicts,
independence and available hours. Self-asserted credentials or hash-shaped strings alone do
not establish qualification. Use pseudonymous study identities and separately controlled
qualification/permission evidence.

Required work agreement: purpose and bounded scope, compensation and time, accessible
instructions/consent where relevant, source access, annotation/derivative permission,
confidentiality/retention, withdrawal handling, and authority to decline ambiguous/unviewable
items. Do not assume that institutional human-subject review applies or is satisfied; the
appropriate study owner must determine and document applicable review before recruitment or
new capture. This Day-1 document is not a consent form or institutional approval.

The existing Phase-3B implementation requires **seven real governance artifacts**. Day 1
specifies their contents; it does not create fake approvals or a placeholder directory that
could be mistaken for a governed queue:

| Artifact | Required content / responsible author |
|---|---|
| ASL convention | Variety, native-to-project distinctions, manual/non-manual temporal semantics, unknown and ambiguity treatment; qualified lead + independent review |
| SIR lexicon | Exact IDs, meanings, accepted form variants, coverage/exclusions and versioning; qualified lead + independent review |
| Primary annotation protocol | Exact bound video/EAF input, observed-evidence rule, source clock, abstain/escalate route, no model-generated truth; qualified lead |
| Blind review protocol | Independently view exact source/motion without seeing the primary submission; same scope/timebase, no agreement-by-copying; independent reviewer |
| Third-person adjudication | Preserve both submissions, list disagreements and support, qualified third-person reasoned resolution or unresolved status; adjudicator |
| Exact sampling plan | Source manifest hash and exact tier/annotation/descriptor selections under source/signer split and stratification rules; Codex + study lead |
| Agreement preregistration | Field support, event correspondence, timing tolerance/IoU, disagreements, abstentions, independent units, uncertainty and threshold choices fixed before final review outcomes; study lead + Codex |

Before bulk annotation, time ten representative items and inspect disagreement causes. Report
primary minutes, blind-review minutes, adjudication fraction and minutes separately. The prior
300-item / 112.5-hour example remains only a scheduling illustration. No annotation-rate or
statistical-power claim is adopted until actual pilot measurements exist.

### D1-G — Completion audit and next handoff

| Day-1 requirement | Evidence / result |
|---|---|
| Re-establish reproducible baseline | Fresh complete warning-strict suite, dependency/compile logs, source hashes, temporary synthetic smoke and exact seeded checkpoint reload; result in D1-A |
| Freeze the audit | Existing B01–B66 audit at named baseline commit plus newly verified B67; unchanged package/test bytes and fresh targeted defect measurements; fixes remain scheduled |
| Freeze narrow scope | `W0-PILOT-SCOPE-v1`, explicit input/domain/phenomena/output/population/acceptance and excluded pilot scope; full later goals preserved |
| Accountable owners | D1-C assigns engineering and project accountability; unfilled external roles have named recruitment owner and acceptance evidence, never fictional appointees |
| Dependency ledger | D01–D24 map B01–B68 blockers, due gate, responsible role, closure artifact and blocked successor |
| Source/reviewer preparation | D1-E exact source intake checklist, existing unsent inquiry references, D1-F qualification/independence and seven-artifact requirements |
| Data and mathematical correctness | Real sources read-only; observations distinguished from inferences; five known defects retained; no evidence gate relaxed; numerical acceptance thresholds remain unset until preregistration |

Day 2 is now concretely scoped to B35/B36: repair general statistical finite-domain validation
and small-tail calculations, then test against independent exact/binomial/permutation cases,
including empty/nonfinite inputs, ties and invalid resampling parameters. It must not use
Day-1's reproduced wrong outputs as expected-correct regression targets. Mask/topology/loader
changes remain D3–D5 unless the dependency order must change for a demonstrated reason.

---

### Deliverable and scheduling assumptions

1. First deliverable: a narrow-domain, text→governed SIR→multichannel motion→deterministic
   avatar **research demonstrator**, with independently evaluated comprehension and explicit
   refusal outside supported scope. Include the non-manual channels needed for the chosen
   phenomena; do not defer grammar-critical face/gaze merely to make a manual-only demo.
2. Second deliverable: raw-audio input, streaming and a restricted operational release.
3. Third deliverable, if retaining the repository's full bidirectional goal: real-video
   ASL→English with its own uncertainty, comprehension and release evidence.
4. One dedicated engineer, **6 productive hours/day, 5 weekdays/week**. Code, debugging,
   validation, documentation and coordination all consume these hours. Overnight compute
   does not create additional engineering capacity. No unconfirmed GPU capacity is assumed.
5. A qualified ASL annotator and a distinct qualified reviewer work alongside the engineer;
   an adjudicator is available for disagreements. Their time is additional, not hidden in
   engineering estimates. If the engineer must also do coordination/annotation beyond the
   allowance, the schedule extends; engineering assistance cannot replace qualified review.
6. Conditional calendar starts **Monday 2026-09-28**. Dates count weekdays only and exclude
   no holidays, school commitments, leave or external delays. Add these explicitly before
   treating the calendar as a personal availability commitment.
7. Research source sample, exact permitted actions and usable rig evidence arrive by the
   end of Day 20; accepted pilot labels arrive by Day 50. These are scheduling dependencies,
   not predictions or assertions that access has been granted.
8. All phase lengths below already include debugging/rework days. External wait or scope
   expansion beyond the stated allowance is additional. A failed empirical experiment can
   require redesign; a timetable cannot guarantee a scientifically successful outcome.

### Resolve the phase logic before adding more modules

Keep the existing Phase 1–7 names, but track three columns for every phase: software
implemented, integrated on admissible inputs, and empirically accepted. They must never
collapse into one completion percentage.

- **Phase 1 refresh (W0):** retain achieved checkpoint/CTC/reproducibility work and repair
  newly found support, topology, analysis and statistical defects.
- **Phase 2 (W1–W2):** source contract → canonical multichannel state → validated round-trip.
- **Phase 3 (W3):** actual annotation/adjudication → governed SIR and lexical motion → direct
  training inputs and later intervention evidence. Existing 3A/B/C verifiers are reused.
- **Phase 4 (W4):** real motion representation and simple-baseline comparisons.
- **Phase 5 (W5):** conditioned generation integrated with deterministic rendering.
- **Phase 6 (protocol in W1, execution W6):** preregistration, independent statistical and
  qualified-human evaluation. Design it before training, not after observing results.
- **Phase 7 (W7–W9):** raw audio, streaming, hardware profiling and restricted release.
- **Reverse-direction extension (W10):** separately gated real-video ASL→English.

The previous pre-Phase-2 gate combines research and commercial prerequisites. Proposed
correction: define separate action-scoped research and commercial gates, consistent with
Phase-3C's existing separation. Research still requires source authorization, observed-state
contracts, qualified review and an authorized research rig. Commercial authorization stays
mandatory for commercial training/distribution/deployment. W0–W1 must explicitly reconcile
this policy in documentation and executable tests; until then, existing gates remain closed.
Do not silently bypass the portfolio gate or invent approval artifacts.

Authorized existing-2D QC, numerical repairs, split feasibility, statistical design and
source-independent interface experiments can proceed while access is unresolved. Freezing
source-dependent production semantics, populating approved libraries and claiming successful
multichannel training cannot. Pseudo-gloss is optional and off the critical path; adding a
pseudo-label generator does not supply missing independent references.

### Work packages, effort, dates and exit artifacts

“Build / debug” are engineer-days; debug includes regression, numerical investigation,
integration repair and explicit reruns. Specialist time and external waiting are separate.

| Package / legacy phase | Days | Conditional dates | Build / debug days | Hours | Required exit |
|---|---:|---|---:|---:|---|
| W0: baseline repair / Phase 1 refresh | 1–10 | Sep 28–Oct 9, 2026 | 7 / 3 | 60 | Reproduced defects fixed; warning-strict suite; honest empty-support/loader failures; canonical architecture and metric registry |
| W1: source/QC/split/protocol / pre-Phase 2 | 11–20 | Oct 12–23 | 8 / 2 | 60 | Accepted source/rig action contracts, grouped split, QC dispositions, pilot annotation protocol, preregistered estimands |
| W2: canonical state and round-trip / Phase 2 | 21–35 | Oct 26–Nov 13 | 11 / 4 | 90 | Typed real-state shards, units/frames/timestamps/masks, valid rotations/gaze, source→state→render round-trip |
| W3: supervised bridge / Phase 3 | 36–50 | Nov 16–Dec 4 | 11 / 4 | 90 | Qualified accepted pilot SIR, lexical motion, direct SIR loader, evidence bindings, no fabricated gloss |
| W4: representation learning / Phase 4 | 51–65 | Dec 7–25 | 11 / 4 | 90 | Tiny-real-subset fit, retrieval/interpolation/AE/RVQ comparison, held-out per-channel/minimal-pair report |
| W5: conditioned generation / Phase 5 | 66–85 | Dec 28–Jan 22, 2027 | 14 / 6 | 120 | Reloadable SIR→motion→avatar run, conditioning interventions, constraint/temporal tests and seed experiments |
| W6: independent validation / Phase 6 | 86–110 | Jan 25–Feb 26 | 18 / 7 | 150 | Blinded comprehension results, clustered uncertainty, baselines, coverage/refusal, failure taxonomy and research-demo go/no-go |
| W7: speech and streaming / Phase 7 | 111–130 | Mar 1–26 | 15 / 5 | 120 | Actual waveform→avatar, calibrated real errors, committed-prefix and interruption tests, target-hardware profile |
| W8: runtime/performance / Phase 7 | 131–140 | Mar 29–Apr 9 | 7 / 3 | 60 | Bounded service lifecycle, memory/load/queue profile, justified optimizations and parity evidence |
| W9: release hardening / Phase 7 | 141–150 | Apr 12–23 | 6 / 4 | 60 | Independent bundle reload, archive/wheel checks, monitoring/rollback rehearsal, domain and action-specific release gate |
| W10: reverse direction / extension | 151–175 | Apr 26–May 28 | 18 / 7 | 150 | Real-video ASL→English baseline, English decoding, scoped non-manual evidence, independent evaluation and separate acceptance |

Totals: **110 days / 660 engineer-hours** to the evaluated text research-demonstrator
candidate; **150 days / 900 hours** to the audio-enabled restricted-release candidate;
**175 days / 1,050 hours** including the bidirectional extension. The 150-day plan includes
42 debug days (252 hours); the 175-day plan includes 49 (294 hours). These are candidates
for acceptance, not promised successful or commercially authorized releases.

Planning ranges, not statistical confidence intervals: 120–210 engineer-days for the
forward/audio restricted product; 145–245 for the bidirectional scope. The shorter case
requires readily usable data/rigs, available reviewers and few redesigns. The longer case
allows source mismatch, representation/generator rework and extra human-evaluation rounds.
A new capture program is a separate project and can extend these ranges substantially.

### First ten days: daily execution plan

Each row budgets six productive hours. Source/reviewer preparation begins immediately;
any external messages or purchases still require the user's actual instruction.

| Day | Date | Work allocation | Deliverable |
|---|---|---|---|
| 1 | Sep 28 | 2h reproducible baseline; 2h scope/owners; 2h source and review requirements | Frozen audit, narrow scope, accountable owners and dependency ledger |
| 2 | Sep 29 | 4h statistical-domain/tail repair; 2h oracle/adversarial cases | NaN/Inf rejection and stable small-p tests |
| 3 | Sep 30 | 4h masked objectives and zero-support behavior; 2h gradient/support tests | Invalid targets cannot train motion objectives |
| 4 | Oct 1 | 4h topology/joint-order contracts; 2h graph/reload tests | 27-joint and alternate-topology behavior explicit |
| 5 | Oct 2 | 3h nonempty loader/nonfinite abort; 1h checkpointable CLI; 2h regression | No silent zero-step training or invalid optimizer update |
| 6 | Oct 5 | 4h length-aware/ragged analysis; 2h insertion/long-sequence cases | Evaluation accepts real variable-duration batches correctly |
| 7 | Oct 6 | 3h weighted estimands/selection; 1h analysis RNG; 2h unequal-batch tests | Reproducible named-checkpoint evaluation with correct support |
| 8 | Oct 7 | 2h canonical package mapping; 2h split/vocabulary design; 2h integration checks | Architecture/evaluation ownership and leakage rules |
| 9 | Oct 8 | 4h integration debugging; 2h complete strict regression | First repaired baseline candidate |
| 10 | Oct 9 | 3h archive/reload/smoke; 2h remaining repair; 1h go/no-go report | W0 acceptance or explicit extension with remaining defects |

This allocation is a target. If masked attention/pooling changes exceed W0, complete the
minimal loss safety fix and carry full encoder masking into W2–W3; record the remaining
integration gate rather than claiming all variable-length behavior solved.

### Subsequent five-day execution blocks

| Days | Concrete work and debugging focus |
|---|---|
| 11–15 | Inspect actual source/rig samples; reconcile research-vs-commercial gate; confirm reviewer workflow; disposition corruption; audit signer/source component balance. |
| 16–20 | Freeze QC on development data; create split and train-only preprocessing; define primary outcomes/independent units; approve source and schema requirements. Stop source-dependent work if evidence is missing. |
| 21–25 | Implement real canonical state, units/frames, availability/inferred provenance, rotation/gaze domains; validate numerical edge cases. |
| 26–30 | Build source adapters, synchronization, confidence/validity support, timestamp-aware derivatives and inverse transforms. |
| 31–35 | Round-trip real samples to authorized rig; inspect left/right, finger rotation, palm, face/gaze and timing; repair mismatches and freeze state v1. |
| 36–40 | Calibrate annotation on a small independently reviewed pilot; refine convention before bulk labeling; implement direct SIR shard/loader adapter. |
| 41–45 | Populate reviewed lexical segments and SIR records; measure disagreement/adjudication; bind rights/split/schema hashes to training manifests. |
| 46–50 | Integrate ragged masked batches and evidence checks; audit final pilot population and unseen forms; regression and Phase-3 input acceptance. Paired-learning empirical exit remains pending. |
| 51–55 | Profile data pipeline; fit simple retrieval/interpolation and tiny-subset continuous representation; debug observability and gradient flow. |
| 56–60 | Compare continuous AE and optional RVQ; inspect dead codes, duration error and channel losses; common-support baseline evaluation. |
| 61–65 | Repeat representation runs, test meaning-critical minimal pairs; freeze winning representation and abandon unnecessary complexity. |
| 66–70 | Integrate one generator with typed SIR and duration conditioning; overfit a tiny accepted subset; verify condition swaps affect output appropriately. |
| 71–75 | Add dense hands, required non-manuals, gaze/head/contact and projection; debug cross-channel timing and gradient imbalance. |
| 76–80 | Add inverse transforms and deterministic render command; run blank/shuffled/order-corrupted/text-only and field intervention experiments. |
| 81–85 | Multi-seed generation, ablations, reload parity, seams/collision/minimal-pair repair; freeze evaluation candidate and manifests. |
| 86–90 | Complete blinded study setup and stimulus QA; pilot comprehension tasks and reviewer instructions without opening final test outcomes. |
| 91–95 | Execute independent reviews and baselines; collect proposition/phenomenon scores, failure severity and disagreement. |
| 96–100 | Cluster-aware analysis, calibrated refusal and subgroup diagnostics; debug study data/metric joins and report uncertainty. |
| 101–105 | One budgeted remediation/retraining round on development data; add regression cases for observed failure classes. |
| 106–110 | Evaluate remediated candidate on a fresh reserved evaluation set; publish research-demo acceptance or a concrete failed-gate report. Reusing the previous test to tune is not a clean confirmatory result. |
| 111–115 | Integrate waveform preprocessing and a versioned real ASR/backend; sample-rate/timestamp/tokenizer contract checks. |
| 116–120 | Gather/evaluate eligible real acoustic stress cases; calibrate semantic risk and refusal on a calibration split. |
| 121–125 | Connect revisable output, commit boundary, interruptions, cancellation, stale-result disposal and queue policy to rendering. |
| 126–130 | Profile end-to-end latency/memory under load; debug streaming seams and actual device failures; accept speech/streaming integration. |
| 131–135 | Service lifecycle, request bounds, privacy/retention, trace identities and failure recovery; realistic load replay. |
| 136–140 | Optimize measured bottlenecks only; numerical and signer-relevant parity checks; rehearse overload behavior. |
| 141–145 | Package model/preprocessors/state/rig/rights and evaluation manifests; clean-machine/archive/wheel reproduction; target OS/device checks. |
| 146–150 | Final regression, rollback rehearsal, domain/coverage documentation and separate research/commercial authorization review. No automatic deployment. |
| 151–155 | Reverse-direction real-video perception and multichannel input contract; sentence/source grouping and English evaluation design. |
| 156–160 | Train sign-state→English baseline; preserve non-manual temporal evidence and uncertainty. |
| 161–165 | Held-out comparison, ablations and real-video degradation tests; diagnostic gloss remains optional. |
| 166–170 | Independent English semantic recovery evaluation, calibrated refusal and failure-driven repair. |
| 171–175 | Fresh reserved evaluation, bundle/reload parity, integration regression and bidirectional acceptance decision. |

### Mathematical implementation acceptance

| Contract | Implementation requirement | Falsification/exit evidence |
|---|---|---|
| Masked losses | `L_k = sum(m*w*ell)/sum(m*w)` over declared observed support. If denominator is zero, return unavailable or reject a required objective; never a successful zero loss. Specify whether normalization is per sample or per valid observation. | Invalid targets cannot affect loss/gradient; all-invalid and partially observed batches tested; confidence is not assumed calibrated. |
| Time | Use `(x[t+1]-x[t])/(time[t+1]-time[t])`; both endpoints must be supported. Define acceleration on nonuniform intervals and reject nonpositive deltas. | Resampling-invariance/timestamp-jitter tests; no derivative crosses a missing interval without declared inference. |
| Geometry | Continuous 6D regression mapped to SO(3); evaluate geodesic rotation error with stable near-zero/near-pi handling. Require orthogonality and determinant +1, unit gaze, declared local/global frames and physical units. | Degenerate axes, reflections, unit conversion, handedness and round-trip tests on synthetic edge cases AND real source samples. |
| Kinematics | Preserve articulated finger/palm/head/face/contact structure. Joint limits and collision corrections are constraints whose linguistic effects must be checked. | Forward kinematics/retarget parity; handshape/orientation/contact minimal pairs survive projection and rendering. |
| CTC | Retain `T_after_subsample >= U + adjacent_repeats`; blank/out-of-range labels fail. | Variable-length and repeated-token cases stay green across new backend adapters. |
| Contrastive learning | Typed multi-positive relation and permitted negatives; no test-derived lexicon; measure repeated-meaning false negatives. | Duplicate-meaning batches, signer/source controls and text-only/video interventions. |
| Representation | Retrieval/interpolation → continuous AE → optional RVQ; quantify per-part distortion, temporal/spectral error, code utilization and duration. | Common-support/coverage comparisons and held-out minimal pairs, not training loss alone. |
| Generation | One chosen diffusion/flow parameterization with consistent schedules/targets/guidance; condition, padding and inpainting masks propagate end-to-end. | Tiny overfit, condition swaps, null/shuffle tests, per-channel constraints, seed/reload tests. No advanced sampler adopted solely because its module exists. |
| Numerical optimization | Abort nonfinite loss/gradients before stepping; monitor branch gradients; introduce EMA/AMP/accumulation only with validated step/resume semantics. | Deliberate NaN/Inf injection, no-step loader, freeze/unfreeze and interrupted-run tests. |

### Statistical implementation acceptance

- Define population, experimental unit, primary endpoint, direction, minimum useful effect,
  confidence level, exclusion rules and comparison family before held-out evaluation.
- Split source and signer groups before windows; fit normalization/vocabulary/calibration
  only on their assigned partitions. A fixed external lexicon is allowed if declared in advance.
- Repair all general statistical input validation and small-tail numerics first. Maintain
  exact/log-probability reference cases; `NaN` never becomes a passing p-value.
- Repeated clips from one source/signer do not automatically create independent sign-test
  trials. Aggregate at the preregistered independent unit or use a defensible hierarchical
  resampling/model. Preserve pairing in interventions. For the existing four-comparison
  Bonferroni family at alpha .05, each threshold is .0125; practical effect remains required.
- Compute uncertainty at the correct source/signer/reviewer levels; separately report
  training-seed variability. Plan at least three training seeds initially, but do not call
  three seeds sufficient evidence for population generalization.
- Compare reconstruction methods on common eligible targets and publish coverage. Zero
  baseline support is unavailable. Report missingness and excluded-source sensitivity.
- Measure inter-reviewer agreement plus confusion, field-level support, temporal tolerance
  and adjudication rate; agreement alone cannot establish linguistic correctness.
- Human semantic proposition recovery and phenomenon-level correctness are primary;
  WER/SignBLEU, coordinate error, cycle score and appearance metrics are complementary.
- Select rejection thresholds on calibration data; report selective risk and coverage
  together on held-out data, including OOD/empty evidence and subgroup uncertainty.
- Estimate sample size from a reviewed pilot, meaningful effect and cluster correlation.
  For orientation only, an IID binary proportion near .5 needs about 384 independent
  observations for an approximate 95% interval with ±.05 margin; clustered observations
  need additional design analysis. This is not a prescription for 384 clips or reviewers.
- Keep a reserved final evaluation set for the budgeted remediation round; do not tune
  repeatedly on the same held-out result while still calling it confirmatory.

### Logical and integration acceptance

Use distinct states: implemented, artifact-blocked, empirically failed, accepted-for-research,
and authorized-for-commercial-action. Missing evidence is not “pass with caveat.”

The existing Phase-3 research gate remains the conjunction of accepted supervision,
video-dependence evidence, reviewed lexical motion, canonical state, independent qualified
validation and research rights. Full release additionally needs representation/generation,
human validation, reliability and runtime evidence. Commercial release adds action-specific
commercial permissions. Wire accepted identities into actual training and inference, not
just a disconnected reporting function. Preserve unknown/ambiguous/unauthorized refusal;
reject schema mismatch, stale evidence and unauthorized action. A digest is content identity,
not proof of issuer authority. Streaming cannot revise the committed prefix; queue bounds
must not be advertised as latency bounds without measured service/scheduling assumptions.

### External work and workload budget

| External lane | Target | Effort placeholder / measurement rule | Failure response |
|---|---|---|---|
| Source access and rig | Exact sample/schema/action evidence by D20 | Engineer preparation is budgeted in W0/W1; response/negotiation time is unknown | Continue independent repairs/QC; delay W2 rather than inventing state evidence |
| Qualified ASL team | Available during W1, pilot convention before W3 bulk work | Two distinct reviewers plus adjudicator; confirm actual availability | Annotation and human exit dates slide |
| Pilot annotation | Accepted pilot by D50 | Initial budgeting example: 300 short items × (12 min primary + 8 min review + .25×10 min adjudication) = 112.5 specialist hours | Calibrate these rates on 10 items, then resize; 300 is a workload example, not an adequate statistical sample claim |
| Specialist total | Annotation, calibration, lexical articulation checks, study design/reviews and adjudication | Reserve roughly 200–350 specialist hours across forward phases; estimate is provisional until pilot timing | At 16 combined specialist hours/week this alone occupies 12.5–22 weeks, overlapping engineering |
| New motion capture if required | Separate scoped acquisition project | Recruitment, consent, calibration, recording, synchronization and review need a new work breakdown | Rebaseline; do not bury capture in a 10-day ingestion estimate |
| Compute | Profile accepted mini-corpus in W4 | `training_hours = epochs * ceil(N/B) * measured_step_seconds / 3600`, then add validation, preprocessing, runs/seeds and restart overhead | If scheduled experiment set exceeds allocated wall time, reduce scope or add explicitly authorized capacity |
| Commercial evidence | Before any commercial action | Unbounded external dependency; research access is not commercial authorization | Restrict to authorized research; no promised commercial release date |

Reforecast every five workdays using actual completed artifacts, defect counts, measured
training time and annotation throughput. At D10, D20, D35, D50, D65, D85, D110, D130,
D150 and D175 issue a gate decision. For an unmet dependency, recompute each successor's
start as `max(predecessor completion, required artifact arrival, engineer availability)`.
Do not simply add all parallel waiting periods or pretend waiting consumes no calendar.

For personal scheduling: 900 productive hours at 15 hours/week is approximately 60 weeks
for the forward/audio candidate; 1,050 hours at that rate is 70 weeks for bidirectional,
before external delays. The six-hour-day calendar therefore requires dedicated capacity.

### Deliberately deferred work

Photorealistic NeRF/Gaussian appearance, large foundation pretraining, production pseudo-gloss,
preference optimization and distributed training are not prerequisites for the first scoped
research demonstrator. Reconsider only after a measured bottleneck and accepted evidence
justify them. They are not secretly included in the 150/175-day totals. Open-domain fluency,
interpreter replacement and unrestricted commercial deployment have no credible completion
date from the present evidence.

---

## Earlier roadmap and design rationale (historical; current sequencing above)


## 1. Roadmap principle

The next phase should optimize for a **complete vertical slice**, not additional
package count. Every milestone must produce an artifact that can be inspected,
reloaded, and falsified.

## 1.1 Research-to-deployment program without a required pseudo-gloss model

The deployed product surface does not require gloss: it requires a governed input,
an inspectable linguistic plan, comprehensible multi-channel signing motion, a renderer,
and calibrated refusal. Pseudo-gloss remains one optional weak-supervision experiment;
it is not placed on the critical path. Removing it does not justify a promise of equal
performance. Capability is preserved architecturally, while performance equivalence must
be established by the same held-out and qualified-signer gates as every other route.

The replacement program is divided into the following consecutive phases.

### Phase 1 — reproducible mathematical execution

Implement before any new learned representation:

1. content-addressed implementation identity that remains valid outside Git;
2. distinct `best` and `last` checkpoints with atomic writes and verified sidecars;
3. model/config/corpus/normalization/vocabulary/code bindings in each checkpoint;
4. exact optimizer, scheduler, epoch, history, and Python/NumPy/Torch RNG restoration;
5. deterministic, RNG-isolated validation;
6. exact CTC feasibility, including mandatory blank frames between adjacent repeats;
7. explicit failure instead of `zero_infinity=True` suppression;
8. clean-source-archive CI, checkpoint reload, and deterministic-output comparison.

Exit evidence is a bit-identical interrupted-versus-uninterrupted CPU training test,
repeatable validation that does not advance the training RNG, adversarial checkpoint
tamper/configuration tests, archive execution without `.git`, and the complete
warning-strict suite. This phase is gloss-independent.

**Phase-1 status (2026-09-10): complete within the boundary above.** The source
checkout and a separate no-Git archive each pass all 1,575 tests under warnings-as-errors.
The targeted changed modules pass static type analysis; compilation, YAML parsing, and
diff validation pass. An installed wheel records content-only provenance without inventing
a Git revision, and its trained checkpoint reloads to bit-identical seeded inference.
The resume test proves bit-identical four-epoch CPU results after a two-epoch interruption,
including parameters, optimizer moments, scheduler, history, and step count. These results
certify the Phase-1 software contracts—not ASL validity, real-data model performance, or
deployment readiness.

### Pre-Phase-2 data-evidence gate — 2026-09-10

The official pseudonymous How2Sign signer code is now certified against the immutable
31,165-row audit and the CVPR supplemental counts: all 31,047 available clips reconcile
exactly, and the 118 missing rows reconcile by signer. This resolves the grouping-key
ambiguity, but no final signer-and-`VIDEO_ID`-disjoint split has been created.

The executable source-portfolio gate requires each modality bundle to be co-observed in
one locally verified, authorized source. It currently rejects all five pre-Phase-2 bundles:
continuous 3D research data, continuous linguistic reference, commercial training rights,
a qualified commercial render rig, and qualified-ASL governance. Publisher claims and
capabilities scattered across datasets cannot pass. The exact evidence, acquisition routes,
secure source treatments, and correspondence register are in
`04_DATA_ENGINEERING_AND_CORPUS.md` §§10–13.

**Phase 2 has not begun.** Its schema can be designed only after these empirical source
contracts are known; otherwise the project would risk freezing invented coordinate,
confidence, gaze, or availability semantics into the architecture.

### Phase 2 — canonical observable and generative sign state

Freeze one typed, time-indexed multi-channel representation before training:

\[
Y_t=(R_t^{body},R_t^{left\ hand},R_t^{right\ hand},R_t^{head},
     r_t,f_t,g_t^{left},g_t^{right},b_t,c_t),
\]

where articulated rotations use a continuous 6D parameterization during regression and
SO(3) geodesic error during evaluation; `r` is root translation; `f` is a declared facial
coefficient/action-unit vector; each gaze vector lies on the unit sphere; `b` carries
blink/eyelid state; and `c` carries declared contact or classifier channels. Handshape is
represented by articulated finger rotations/contact, palm orientation by wrist/palm
frames, movement by nonuniform-time derivatives, posture by the body kinematic tree,
and head movement by its own SE(3) channel. These quantities must not be collapsed into
one undifferentiated Cartesian loss.

Every channel carries `observed`, `valid`, `confidence`, `source`, coordinate-frame, and
timestamp fields. A loss is evaluated only on its declared support:

\[
\mathcal L_k=\frac{\sum_{t,j}m^{(k)}_{t,j}w^{(k)}_{t,j}
\ell_k(\hat Y^{(k)}_{t,j},Y^{(k)}_{t,j})}
{\sum_{t,j}m^{(k)}_{t,j}w^{(k)}_{t,j}},
\]

and is unavailable—not zero—when the denominator is zero. Confidence may be used as a
weight only under a declared interpretation; it is not called calibrated probability
without calibration evidence. Phase 2 ends only after source→state→inverse-transform→
render round-trips and left/right, gaze, palm, face, head, and temporal tests pass.

The current frontal 2D OpenPose release cannot uniquely determine this state. Monocular
depth, self-occluded fingers, palm twist, subtle facial action, and gaze are non-identifiable
in many frames. Those channels require licensed multi-view/depth evidence, a validated
fitting model, project-human correction, or an explicit unavailable mask—never synthesis
presented as observation.

### Phase 3 — direct governed language-to-SIR supervision

Replace the pseudo-gloss dependency with three non-substituting evidence routes:

1. qualified project-human text/video→SIR annotations under a frozen ASL convention;
2. direct text/video contrastive and temporal learning on authorized paired media, with
   transcript-only and blank/shuffled-video falsification tests;
3. a versioned lexical motion library for forms that have human-approved meaning and
   articulation, with `UNKNOWN` and abstention for uncovered forms.

Authentic gloss may later enter as an auxiliary observation, but no phase depends on a
full-corpus pseudo-gloss model. English words, uppercase lemmas, retrieval IDs, latent
visual tokens, and SIR fields remain distinct types. Phase 3 ends only when plan fields
are reference-backed, source-disjoint, intervention-sensitive, and independently reviewed.

**Phase-3 software-boundary status (2026-09-12): implemented; empirical exit not
approved.** The repository now provides three fail-closed mechanisms that can accept
future evidence without manufacturing it:

1. a canonical, hash-bound human SIR annotation envelope restricted to
   `official_human` and `project_human`, with a frozen ASL convention and lexicon,
   exact video/transcript/provenance/authorization bindings, distinct qualified
   annotator and reviewer attestations bound to the exact reviewed content plus hashed
   qualification/independence evidence, and signer/source split-leakage checks;
2. strict, versioned SIR parsing and content hashing that reject unknown fields,
   malformed identifiers, non-finite or contradictory intervals, invalid edges,
   duplicate graph elements, and noncanonical payload bytes; and
3. a held-out paired-video dependence evaluator requiring aligned performance to
   exceed blank-video, shuffled-video, order-corrupted-video, and text-only
   interventions under a preregistered positive effect threshold, exact paired sign
   tests, and four-comparison familywise correction.

These mechanisms do not generate annotations, train a language model, produce motion,
or establish ASL correctness. The following blockers are deliberately recorded as
**unsolved**: governed qualified text/video→SIR supervision; a passed direct paired-
learning falsification; a human-approved lexical motion library; the canonical Phase-2
multi-channel state; independent qualified-ASL validation; and commercial training and
deployment rights. No downstream Phase-3/4/5 implementation may treat the software
boundary as evidence that any of those blockers has been resolved.

**Phase-3A linguistic-reference ingestion status (2026-09-13): secure software and one-item
Cokely source compatibility verified; linguistic approval not started.** The fail-closed EAF
reader preserves publisher-native tiers, untimed slots, and reference graphs; binds the exact
EAF, publisher page, primary HD media, and auxiliary SD media; and emits compact canonical
inspection artifacts without creating SIR or training labels. Its 27 focused adversarial
tests and the complete 1,640-test repository suite pass under warnings-as-errors. The
authentic *I Have a Dream* pilot contains 751 source annotations, nine of which have at least
one publisher-unspecified endpoint that remains uninterpolated. Its security contract,
hashes, and qualified-human mapping protocol are in
`10_PHASE_3A_LINGUISTIC_REFERENCE_INGESTION.md`. This bounded result does not establish
corpus-wide compatibility, project SIR validity, ASL correctness, commercial rights, or
readiness for training.

**Phase-3B governed-mapping software status (2026-09-13): implemented; qualified-human
execution not started.** Exact EAF source annotations can now enter a label-empty,
hash-indexed review queue governed by seven real artifacts and a machine-verifiable sampling
plan. Primary and blind-review SIR submissions use an explicit EAF-millisecond timebase;
disagreement requires third-person adjudication; per-field, temporal, event-coverage, and
edge diagnostics retain exact support; and a deterministic batch audit cannot authorize
training, commercial use, or linguistic validity. The complete contract and remaining human
evidence gates are in `11_PHASE_3B_GOVERNED_MAPPING_AND_ADJUDICATION.md`.

**Phase-3C pre-artifact software status (2026-09-14): implemented; empirical Phase-3 exit
not approved.** The project now has a distinct canonical lexical-motion registry, strict
external-evidence identities, fail-closed unknown/ambiguous/unauthorized motion lookup, and
a conjunctive readiness assessment over governed supervision, paired-video intervention
evidence, the canonical Phase-2 state, independent qualified-ASL validation, and separate
research/commercial rights. The runtime report also labels every implementation deferred
because its real source contract is absent. See
`12_PHASE_3C_EVIDENCE_INTEGRATION.md`. This does not authorize Phase 4.

### Phase 4 — multi-channel motion representation learning

Train retrieval/interpolation, continuous autoencoder, and residual-token baselines on the
Phase-2 state. Evaluate hands, palms, body, face, gaze, head, contact, velocity, acceleration,
and temporal scope separately. A learned representation must beat declared simple baselines
on held-out source groups and preserve meaning-critical minimal pairs before a generator is
authorized.

### Phase 5 — constrained text/SIR-to-motion generation

Train body, hand, face, gaze, and head branches in a staged curriculum, then integrate them
through typed SIR conditioning, cross-channel constraints, uncertainty, and abstention.
Require conditioning swaps, field interventions, ablations, multi-seed stability, exact
reload, and deterministic rigged rendering. A low aggregate pose loss is not a pass.

### Phase 6 — linguistic and human validation

Use preregistered, blinded evaluation with qualified ASL signers. Primary endpoints are
semantic proposition recovery and phenomenon-level accuracy; motion and rendering metrics
remain diagnostics. Failures in handshape, orientation, movement, posture, facial grammar,
gaze, head movement, reference, or timing are reported independently rather than averaged
away.

### Phase 7 — restricted product and industrialization

Integrate raw text/audio, rendering, calibrated refusal, traceable model/data identity,
privacy controls, observability, rollback, canaries, and target-hardware load tests. Release
first as a restricted-domain research demonstrator. Commercial deployment additionally
requires data/model/rig rights that do not depend on How2Sign's noncommercial permission.
Medical, legal, emergency, and interpreter-replacement claims remain prohibited until
separately validated and authorized.

### Current implementability boundary

| Can be implemented now | Requires external evidence or assets |
|---|---|
| Phase-1 reproducibility, CTC, checkpoint, and archive controls | Linguistically valid text/video→ASL plans |
| Typed multi-channel schemas and observability rules | Qualified-ASL annotation and blinded review |
| 2D source QC and source-disjoint experimentation; certified pseudonymous signer key | Final signer-and-source-disjoint split and certificate |
| Governed supervision envelopes and paired-video dependence evaluation | Actual qualified human SIR annotations and a trained model passing the intervention gate |
| Render/fitting interfaces and synthetic geometric proofs | Licensed production body/face/rig assets |
| Security, provenance, abstention, and deployment contracts | Commercial data and derivative-model rights |

## 2. Stage A — Stabilize the repository

### Objectives

- obtain a clean, reproducible baseline;
- eliminate destructive and ambiguous behavior;
- designate canonical implementations.

### Required tasks

1. Pin Python, PyTorch, NumPy, and test dependencies in a lock file.
2. Fix all speech integration and warning-strict test failures.
3. Add CI for clean installation, compilation, tests, and a short smoke run.
4. Add the actual license file and correct packaging metadata.
5. Add the missing foundation requirements file or remove the reference.
6. Change corpus regeneration to explicit opt-in.
7. Add non-empty-directory and overwrite guards.
8. Select canonical planner, diffusion, speech, and evaluation packages.
9. Document deprecated duplicate paths.
10. Add typed configuration serialization and schema versions.

### Exit gate

- zero test failures under the pinned warning-strict environment;
- a fresh clone installs and reproduces the smoke result;
- no default command overwrites user data.

## 3. Stage B — Build the real-data bridge

### Objectives

- connect governed records to active training batches;
- preserve uncertainty and variable lengths.

### Required tasks

1. Implement media decoders with timestamp preservation.
2. Integrate body, dense hand, and face extraction.
3. Add optional multi-view triangulation or body-model fitting.
4. Implement an exporter from `data_engineering.Sample` to versioned shards.
5. Carry confidence and validity masks through every transform.
6. Implement signer/source grouped splitting before windowing.
7. Add variable-length collators and input-length propagation.
8. Validate exact CTC feasibility.
9. Persist vocabulary, label definitions, normalization, and coordinate metadata.
10. Produce HTML/video or equivalent human-review reports for sampled records.

### Exit gate

- one licensed, versioned real mini-corpus passes all schema and visual checks;
- every secondary-dataset record binds to a byte-verified local license-evidence
  snapshot and an exact action scope; no license label substitutes for consent;
- the active loader consumes it without synthetic generation;
- every tensor traces back to immutable source media.

### Current gloss-free preparation result (updated 2026-09-10)

The canonical v1 full-corpus audit and the quarantined 2D masked-reconstruction
experiment are complete and reproducible. The audit accounts for all 31,165 metadata
rows plus one orphan artifact without fabricating the 118 missing clips or repairing
the three structural failures. Review queues and `VIDEO_ID` source constraints are
available, but they are not review attestations. The official pseudonymous signer code
is now certified by exact reconciliation with the How2Sign supplemental; it is not a
person's identity. The 2D experiment
is intentionally disconnected from the exporter, active runtime, 6D motion tokenizer,
and Stage C; its held-out point/span model did not beat temporal interpolation.

Detailed evidence is recorded in `docs/DATA_ENGINEERING.md`. The pseudo-gloss
candidate-lattice implementation and its activation requirements are consolidated in
`09_PSEUDO_GLOSS_MODEL_RESEARCH.md`. The software path is implemented, but corpus-wide
generation remains gated on a versioned ASL lexicon/convention, independent qualified
human references, calibrated pretrained weights, frozen preregistration, and partitions
disjoint by both signer and source when generalization is claimed.

**Stage B remains unapproved.** Authentic gloss, a final signer-and-source-disjoint split,
qualified signer review, co-observed production 3D channels, and commercial authorization
are absent. English transcripts, pseudo-glosses, review-queue generation, 2D fits, and
cross-corpus modality composition cannot substitute for those gates. Stage C therefore
remains blocked.

## 4. Stage C — Establish the minimal vertical model

### Recommended first scope

Start with:

```text
validated text or gloss
    -> minimal typed plan
    -> body + dense-hand motion
    -> deterministic rigged avatar
```

Defer raw speech and photorealistic rendering.

### Required tasks

1. Decide the canonical motion representation.
2. Wire minimal semantic/SIR fields into the generator.
3. Replace 27-joint Cartesian output if it cannot express the target phenomena.
4. Integrate variable duration and frame masks.
5. Connect biomechanical constraints to the active loss/sampler.
6. Add a real body/hand model under valid licensing.
7. Build a deterministic retarget-and-render path.
8. Add inverse normalization to inference.
9. Persist readable gloss/plan labels.
10. Add one command that loads a complete artifact and renders a sample.

### Exit gate

- the model overfits a deliberately tiny real subset;
- generated motion can be rendered and visually inspected;
- output quality changes appropriately when conditioning changes;
- shuffled conditioning performs materially worse.

## 5. Stage D — Make training scientifically credible

### Required tasks

- staged training rather than all-branch optimization from initialization;
- pretrained speech/language encoders where justified;
- deterministic validation;
- EMA for the generator;
- mixed precision and gradient accumulation;
- complete checkpoint/resume;
- per-loss and per-module gradient monitoring;
- automatic detection of zero, NaN, or exploding gradients;
- multi-positive contrastive alignment;
- loss-weight and curriculum ablations;
- experiment tracking with immutable configs.

### Recommended curriculum

1. fit and validate the motion representation;
2. train or validate the motion tokenizer;
3. train motion recognition;
4. train semantic planning;
5. train conditional motion generation;
6. add cross-modal alignment;
7. add non-manual and spatial conditioning;
8. jointly polish only after branch competence is established.

### Exit gate

- at least one signer-held-out real-data experiment;
- reproducible result across independent reruns;
- meaningful baselines and ablations;
- no branch passes solely because a modality is missing.

## 6. Stage E — Add speech and streaming

### Required tasks

1. Integrate raw waveform preprocessing with the active model.
2. Add a real ASR or speech representation backend.
3. preserve word/token timestamps and uncertainty;
4. connect calibration and fail-closed policy to actual inference;
5. implement revisable versus committed sign output;
6. test noise, accent, code-switching, long-form speech, and interruptions;
7. measure selective risk versus coverage on real errors.

### Exit gate

- raw audio reaches motion without manually supplied feature tensors;
- confidence predicts semantic failure;
- abstention or clarification reduces harmful assertions;
- streaming revision obeys the committed-prefix contract.

## 7. Stage F — Human evaluation

### Required tasks

- co-design the evaluation with qualified signers;
- pre-register primary endpoints;
- use blinded randomized presentation;
- compare against meaningful baselines;
- measure comprehension, grammaticality, naturalness, and acceptability;
- analyze errors by linguistic phenomenon and signer group;
- record uncertainty and disagreement.

### Exit gate

- the system demonstrates a practically meaningful improvement;
- failures are characterized well enough to define a safe use case;
- the target community considers the presentation and limitations acceptable.

## 8. Stage G — Deployment engineering

### Required tasks

- self-contained model bundle and loader;
- authenticated service or on-device runtime;
- actual optimized kernels and numerical equivalence tests;
- renderer and media output;
- latency and memory profiling on target hardware;
- monitoring, trace IDs, and failure reporting;
- model/data version telemetry;
- rollback and canary strategy;
- privacy, retention, and incident procedures;
- accessibility and user-experience testing.

### Exit gate

- an independently reproducible release candidate meets all quality, safety,
  latency, and operational gates on target hardware.

## 9. Priority matrix

| Priority | Work |
|---|---|
| P0 | Prevent corpus overwrite; fix test failures; connect real corpus exporter |
| P0 | Select canonical architecture and motion representation |
| P0 | Complete checkpoint/config/vocabulary/normalization bundle |
| P1 | Integrate SIR, biomechanical constraints, dense hands, and face |
| P1 | Deterministic training, baselines, ablations, and signer-held-out testing |
| P1 | Independent and human comprehension evaluation |
| P2 | Raw speech, calibration, and streaming revision |
| P2 | Optimized renderer and deployment runtime |
| P3 | Photorealism, large-scale pretraining, and advanced preference optimization |

## 10. What not to do next

- Do not add a fourteenth isolated subsystem.
- Do not run expensive full-corpus training before tiny-real-subset overfitting.
- Do not interpret synthetic cycle consistency as human comprehension.
- Do not merge incompatible sign-language datasets without linguistic mapping.
- Do not optimize rendering appearance before motion is understandable.
- Do not claim production readiness because latency or quantization formulas pass
  unit tests.
