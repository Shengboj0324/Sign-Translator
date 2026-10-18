# Canonical-state generation and declared rig posing

This document specifies `signtranslator/generation/`: the connection from an
admitted governed corpus to canonical multichannel motion, and from canonical
motion to a posed rig. It closes the engineering gap recorded as items 2 and 4
of *Engineering that remains after manual inputs arrive* in the source
acquisition guide, and the canonical-state-to-rig mappings left open by the
11-channel boundary audit.

## 0. Honest scope (read first)

No real co-observed motion, permission, qualified review or rig qualification is
supplied or implied by this layer. Every property below is established against
the existing fictional governance fixtures and the pinned real MakeHuman asset
geometry. What is now implemented is the typed path that a real admitted sample
will travel; what remains missing is the sample, its rights, its calibration and
its reviewers. Blockers B01–B08, B42–B48 and B51 are unchanged, and
`phase_exit_approved` stays false everywhere.

Three consequences are enforced in code rather than asserted in prose:

1. a generated state is emitted `observed=False` with zero confidence, so the
   existing supervision weights are exactly zero and no generated motion can be
   admitted as observation or consumed as a training target;
2. a channel with no observed support produces zero branch support, never a
   measured zero loss, so an absent modality cannot average into a pass;
3. the state-to-rig adapter refuses to pose anything until every rig bone and
   every canonical channel is explicitly declared driven, held at rest, or
   deliberately unmapped.

## 1. Model and objective

`CanonicalStateSIRDecoder` consumes a `GovernedMotionBatch` and predicts all
eleven canonical channels in their own parameterisations. It declares
`governed_batch_schema_version = 1` and `governed_objective_schema_version = 1`,
so the canonical `Trainer` drives it with the existing exposure ledger, support
accounting, schema-5 checkpoints and exact CPU continuation.

### 1.1 Conditioning uses reviewed timing, not inferred alignment

Each frame attends exactly the reviewed events whose half-open interval
`[t_start, t_end)` contains that frame's clock value. That membership is not
learned and not estimated: the governed corpus derives it from the reviewed SIR
and the source clock, and the model cross-checks that the membership rows and
the SIR target rows describe the same events in the same order before using
them. One always-attendable null event keeps the attention distribution defined
for a frame inside no event, which is otherwise a softmax over an empty set.

Conditioning reads event kinds, governed labels, referent presence, loci and
interval geometry. It never reads the channel values it predicts; a test
perturbs the targets and asserts bit-identical predictions.

### 1.2 Per-channel losses

With per-cell reliability weight `w` from `valid & observed` times declared
confidence, each branch reduces as `sum(w*l) / sum(w)` over that example's
supported cells, and examples without support contribute exactly zero.

| Channel kind | Loss `l` | Range | Zero exactly when |
|---|---|---|---|
| `rotation_6d_columns` | `‖R − R̂‖²_F / 8 = sin²(θ/2)` | `[0, 1]` | `R = R̂` |
| `translation`, `coefficient` | Huber on `r = (pred − target)/scale` | `[0, ∞)` | equal values |
| `unit_vector` | `(1 − cos∠)/2` | `[0, 1]` | equal direction |
| `bounded_scalar` | `KL(Bern(t) ‖ Bern(σ(z)))` | `[0, ∞)` | `σ(z) = t` |

The rotation identity is exact: `tr(RᵀR̂) = 1 + 2cos θ`, so
`‖R − R̂‖²_F = 6 − 2tr(RᵀR̂) = 4(1 − cos θ) = 8 sin²(θ/2)`. Optimising
`sin²(θ/2)` rather than `θ` keeps the gradient bounded at `θ ∈ {0, π}`, where
`arccos` is singular, while remaining strictly increasing in the geodesic angle;
the geodesic angle in degrees is reported as a diagnostic instead. The bounded
channel is scored by the Bernoulli divergence, whose exact minimum `0` is
attained only at `σ(z) = t`, so a continuous `[0, 1]` target keeps a calibrated
Bernoulli meaning rather than becoming an unconstrained regression value.

Translation channels add an optional velocity term on first differences divided
by the **actual elapsed seconds** between two supported frames, with
`max_gap_seconds` honoured, so a nonuniform source clock is never treated as a
frame index. A test asserts that stretching the clock by ten lowers the velocity
error for the same displacement, which a frame-index surrogate cannot reproduce.

### 1.3 Head initialisation

Every head is zero-weight with a structured bias, so at initialisation rotation
heads emit exactly the identity rotation, gaze heads a fixed nondegenerate axis,
bounded heads `0.5`, and translation and coefficient heads zero. That removes
degenerate Gram–Schmidt and zero-norm failures at step zero without hiding them
later: a degenerate prediction during training still fails explicitly rather
than being silently regularised.

### 1.4 Exposure declarations

Branch terms declare per-example support membership and **no** cell-level target
declarations. Dense motion cells would grow the optimizer-exposure ledger in
proportion to frames times elements per step, while the annotation and corpus
content hashes already bind the exact target bytes. The exposure audit records
that omission as an unrecorded declaration, which it keeps distinct from an
empty one.

## 2. Declared state-to-rig posing

`StateRigBinding` is a complete, hashable declaration: channel element to bone
rotations, channel element to bone-local translations, the bones deliberately
held at rest, the channels deliberately not driving the rig, the metres per
source unit, the claimed rotation basis, and the rig asset commit. Validation
rejects an incomplete or self-contradictory declaration:

* every rig bone must be either driven or explicitly held at rest;
* every canonical channel must be either driving or explicitly unmapped;
* no bone may take two rotations or two translations, no element may drive two
  bones, and no channel may be both driving and unmapped;
* only the single declared rotation basis `bone_local_after_rest_v1` is
  accepted, because a different source convention needs its own adapter rather
  than a relabelling of this one.

`pose_state_frame` then poses one frame at a time. A required element that is
unavailable at that frame fails; it never falls back to the bind pose. An
element that is valid but unobserved — a generated or fitted value — is posed
only when the caller passes `allow_inferred=True`, and the returned frame
records `posed_inferred_values=True`. Translations convert from canonical metres
into the rig's unscaled source units by dividing by the declared scale, then
apply in that bone's rest-local axes exactly as `MakeHumanRig.pose` defines.

Each posed frame carries its frame index, clock value, sample id, binding hash,
asset commit, whether inferred values were posed, and
`accepted_provenance=False`. Verifying a declaration is complete and applied
exactly is not evidence that it is anatomically or linguistically right: that
remains the source-specific, reviewable question recorded in B47/B48/B69.

## 3. What this does and does not establish

Established against the current fixtures and the pinned rig geometry: identity
rotations reproduce rest geometry through the adapter; a declared rotation moves
only its bone; the declared unit conversion matches an independent float64
oracle; generated states round-trip through the canonical archive with
rotations in SO(3), unit-norm gaze and bounded blink and contact; the real
`Trainer` optimises every supported branch with finite losses and exact
epoch-boundary continuation; and the reviewed-SIR-to-state-to-posed-frame chain
runs frame by frame.

Not established, and not claimable from any of the above: that the generated
motion is intelligible ASL, that the declared bindings match a real source's
conventions, that the rig is qualified for signing, that any branch weight or
scale is calibrated, or that a phase may be approved. This is the W4
conditioned-regression baseline the roadmap requires **before** a tokenised or
diffusion alternative may be preferred (blocker B44), not a demonstration that
it wins.
