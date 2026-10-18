# Native source adapters

The first native adapter converts the public 3D-LEX v1.0 ASL release into the
canonical 11-channel state. It was written against the published glTF 2.0
specification and the actual files, after inspecting them, not from a guessed
schema. Every defect it handles was measured on the real data first.

## 0. Honest scope (read first)

3D-LEX is an **isolated-sign lexicon**. It cannot close B01: it does not co-observe
continuous discourse, sentence grammar or coarticulation. Its values are **solved,
not observed** — an optical-marker and glove solve exported through FBX2glTF — so
every value is `observed=False` with an explicit inference method, and any use as a
training target or comparison reference must opt in explicitly
(`allow_inferred_targets`, `allow_inferred`). After validation only **two
independent performers** remain, so no signer-independent claim can be supported
(the best attainable two-sided sign-test p-value is 1/2). It is an engineering and
descriptive-baseline source, and the first real canonical data the project has.

Every modality of the release is now ingested (§9–§12): face, gaze and eyelid from
Live Link Face as **separate face-only samples on their own clock** (no frame-wise
synchronisation with the skeleton exists, §9); StretchSense glove output as an
independent check that admits or withholds each hand (§10); and ASL Signbank ID
glosses as review candidates, not labels (§11). One content-addressed manifest
joins them and decides what any consumer may load (§12).

## 1. Acquisition

All seven ASL archives (11.84 GB) are public on OSF (`osf.io/3wsed`) under CC BY 4.0
and were streamed into R2 under `acquisition/2026-10-17-v1/3d-lex-v1.0-asl/`. Each
transfer hashed the bytes in flight and was accepted only when both the byte count
and the SHA-256 equalled the publisher's values, and only after the object was
re-read from R2 and re-hashed; any mismatch aborted the multipart upload, so no
unverified object exists. One 3.23 GB transfer dropped at 1.36 GB, was refused by
the size check, and was retried with a verified resume: OSF ignores Range, so the
retry re-reads the already-uploaded prefix and requires its SHA-256 to equal the
saved hash state before continuing. The 4.19 GB handshape part 1 then failed at
completion because network reads are sometimes short, so multipart parts had
unequal lengths, which R2 rejects; parts are now cut at exactly 64 MiB, and the
retry was accepted with the publisher's SHA-256 and an R2 re-read.

## 2. What the files actually are

| Archive | Content | Clock |
|---|---|---|
| `ASL_ANIM_GLB.zip` | 938 glTF 2.0 files, FBX2glTF v0.9.7, 68-joint Mixamo-style skeleton with full fingers, Head and per-eye joints | One shared sampler input per file, 24 fps |
| `ASL_Vicon_MarkerData.zip` | 990 Visual3D CSV exports of 53 raw markers | Frame index only; 120 Hz recovered independently (§5) |
| `ASL_LLF.zip` | 997 Live Link Face CSVs: 52 ARKit coefficients, head and per-eye yaw/pitch/roll | `HH:MM:SS:FF.sub` device timecode |
| `Handshape Data Part 1–3` | 999 sign folders: StretchSense CSVs per hand (32 capacitances, per-joint Euler angles), metadata, calibration, FBX | Device timecode `HHMMSSFFF` at 120 Hz, slaved to the Vicon master |
| `ASL_ANIM_FBX.zip` | The same solves as FBX; take start times are wall-clock | Not adapted: GLB carries the same animation |

The first line of each Vicon CSV is a performer's name. It never leaves the parser:
grouping uses a salted SHA-256 only.

## 3. glTF adapter (`data_engineering/gltf_motion.py`)

* **Container.** Magic `glTF`, version 2, declared length, 4-byte-aligned JSON then
  BIN chunks; strict JSON with duplicate keys and non-finite numbers refused.
  Accessors are bounds-checked against their buffer views; only non-normalised
  float32 is accepted; sparse accessors are refused.
* **Animation.** Exactly one animation; every sampler must share one input clock,
  because merging sampling grids would be resampling; LINEAR or STEP only.
  Quaternions are stored `(x, y, z, w)` and must be unit to `1e-5`; the observed
  worst case is `5.0e-8`.
* **Mapping.** `THREE_D_LEX_MIXAMO68` declares 21 body joints, 15 per hand (three
  per finger, end effectors excluded), and the head. Every skin joint must be
  declared mapped or excluded, so a different skeleton fails instead of silently
  dropping joints. Rotations keep the source convention,
  `gltf-node-local-rotation-including-rest-v1`.
* **Root.** World root position composes the static ancestor chain with the root's
  local translation. A full forward-kinematics check gives toe-to-head-top height
  1.815 m, agreeing with the independent Vicon head-top marker (1.848 m). The
  source's world origin is not on the floor (an exported `actorBP` node offsets it by
  about 0.3 m), so the frame is declared, not corrected.
* **Unavailable, explicitly.** Head translation, face, both gaze channels, blink and
  contact. Eye joints are not animated. Face data exists but on a separate device
  clock with no declared synchronisation, so placing it on the skeleton clock would
  be an unverified alignment. Hips scale (about 1.04, varying 0.1%) is reported as
  unrepresented.

## 4. Measured source defects and the rules that handle them

**Terminal reset keys.** Every clean-performer file ends with a block that moves the
hips about 1.36 m in 1/24 s (≈33 m/s): one key in 263 files, two keys in 163. The
185 salvaged files, from a different export batch, have none. The
adapter keeps the **longest physically plausible subsequence**: frame `i` may
precede `j` iff `|p_j − p_i| ≤ v_max (t_j − t_i)` on the real clock, and a longest
path in that DAG is found exactly by dynamic programming. With `v_max = 5 m/s`,
genuine motion keeps every frame. Files that would lose more than a declared 10% of
frames are refused as ambiguous. An exhaustive brute-force test confirms the
programme returns a true longest chain. An earlier isolated-frame rule missed the
two-key case and wrongly refused 176 files; that run is preserved as superseded.

**Keyed axis correction.** One performer's 187 files key the `Armature` up-axis node
instead of leaving it static, sweeping 84–90°; four key it to identity, 90° from its
static value. Composing faithfully would tip the body; in fact the world hips drift
by exactly 37.1° in every such file, a systematic artifact. Elbow flexion is
invariant to any whole-body rigid motion and is identical to `1e-9` with and without
the keyed node, so parent-relative joints are unaffected. With
`salvage_reoriented_root=True` these files are converted with the root rotation and
root translation unavailable, and only if they pass §5.

**No animation.** 203 files contain no animation and are refused.

**Rest-pose end frames (found by the glove check, §10).** In 181 of the 185 salvaged
files the last key snaps every joint to the bind pose: the median animated joint is
0.39° (at most 0.97°) from its node rest rotation, against 11–38° in genuine signing
frames.
These files' root is unavailable, so the root-speed rule could not see it. The
adapter now removes leading or trailing frames whose median animated joint lies
within 2° of rest (`rest_pose_tolerance_degrees`), never interior frames, and the
removal counts toward the 10% exclusion limit. Conversion v4 removed 184 such
trailing frames in 183 files (181 of them validated); the same 611 files pass, now
with 55,358 valid frames.

## 5. Independent marker validation (`data_engineering/marker_validation.py`)

The solved skeleton is checked against the raw markers it was fitted to, using
elbow flexion: the angle at the elbow between upper arm and forearm. Marker joint
centres are midpoints of lateral/medial pairs; missing markers stay missing. One
rate and offset are searched per take, jointly for both elbows, maximising the
correlation of the per-series-centred concatenation; the same clock must explain
both arms.

Thresholds were fixed from a 20-file pilot before the full run: per elbow, centred
error ≤ 10°, and correlation ≥ 0.90 wherever the elbow moves at least 5°. The pilot
revision is recorded: correlation carries no information at near-zero variance, and
two static elbows with 0.1–0.2° error had been failing on noise. The capture rate
is recovered as **120 Hz in 605 of 611** validated files.

## 6. Result

| Stage | Files |
|---|---:|
| GLB files | 938 |
| No animation | 203 |
| Converted, all joints | 426 |
| Converted, salvaged (root unavailable) | 185 |
| Failed validation: no single clock aligns the take | 106 |
| Failed validation: thresholds | 17 |
| Failed validation: no marker file | 1 |
| **Validated canonical states** | **611** (55,358 valid frames, ≈38.4 min, v4) |

By performer: A contributes 426 validated signs, B 185 salvaged signs, and C none:
all 72 of C's animated files fail validation, 69 because no single clock aligns the
take. In a 40-file sample of such failures the marker trial was shorter than the
animation in 65% (median 1.82 s against 5.33 s), whereas in passing files it always
contained the animation (median 4.27 s against 3.50 s). States are in R2 under
`derived/3d-lex-canonical-v4/states/`; earlier v1–v3 conversions remain as history.

## 7. First real W4 baseline

One seeded interior 0.25 s span (6 frames) per sign is hidden and reconstructed;
methods are scored on identical cells with the solved skeleton declared as reference.
Numbers are on the v4 states (§4, rest-pose end frames removed); the v3 table, kept
in the evidence, was 0.78°/0.36°/1.41°/1.19°, slightly worse because spans touching
the contaminated end frames were scored against a reset pose.

| Channel | Geodesic interpolation | Zero-order hold | Interpolation better |
|---|---:|---:|---:|
| Body | 0.75° | 1.74° | 85.2% of signs |
| Head | 0.35° | 0.76° | 86.2% |
| Left hand | 1.39° | 2.68° | 79.8% |
| Right hand | 1.17° | 2.39° | 82.9% |
| Root | 0.6 mm | 1.4 mm | 87.1% |

Any learned representation must beat these numbers on the same support before it is
preferred (B44). Signs are not independent units: they share two performers, so
these are descriptive. A constant identity reference is reported in the evidence but
is a strawman here, because the source convention includes the rest pose.

## 8. First learned representation: a negative result

`generation/masked_motion.py` trains a residual bidirectional GRU (506,616
parameters) that predicts a correction to the geodesic interpolation of each hidden
span. Its correction layer is zero-initialised, so it starts exactly at
interpolation and can only differ by learning; a test asserts that perturbing hidden
targets leaves its predictions unchanged.

Design: train on the performer with more validated signs (385 signs, six seeded span
variants each, 2,310 examples; 41 held out for selection), select the epoch on a held-out tenth of that performer's signs, and
test once on the other performer's 185 signs, using the same span rule as §7.

Result (v4 states): training loss fell from 0.000905 after the first epoch to a
minimum of 0.000719, but validation error was worse than interpolation at every one of
30 epochs (1.41–1.64° against 1.27°). Selection therefore kept epoch 0, which is
interpolation, and the held-out test equals it: 0.7391° on 56,457 cells, with all 183
test signs tied within 0.001° (the v3 run gave 0.8130°; its "fraction better" counted
float32 rounding as wins and is superseded by explicit better/worse/tied counts). For
0.25 s gaps in 24 fps solved motion, interpolation is near the noise floor and this
model overfits its training performer instead of
generalising. Simple baselines stay eligible to win, and here interpolation wins.
The pre-declared follow-up asked whether learning helps on longer gaps, where
interpolation degrades. It was run with the identical design and seed rule:

| Hidden span | Interpolation, validation | Model, validation (30 epochs) | Selected | Test (other performer) |
|---|---:|---:|---|---:|
| 0.25 s | 1.27° | 1.41–1.64° | epoch 0 | 0.739°, 183 signs tied |
| 0.5 s | 2.46° | 2.63–3.66° | epoch 0 | 1.450°, 182 signs tied |
| 1.0 s | 4.12° | 4.22–5.03° | epoch 0 | 2.802°, 180 signs tied |

Interpolation error grows roughly linearly with the gap (body 0.75°, 1.47°, 2.60°;
`baseline-comparison-v4-span*.json`), yet this model never beats it on validation at
any length. The negative result therefore holds across the pre-declared gap lengths
for this model and this data: about 400 isolated signs from one performer do not
teach a residual model structure that interpolation lacks. One held-out performer is
one unit, so this is descriptive.

## 9. Face, gaze and eyelid: Live Link Face (`data_engineering/llf_face.py`)

Each of the 997 CSVs was measured before the adapter was written.

* **Clock.** `HH:MM:SS:FF.fff` is device timecode at 60 frames per second with a
  fractional subframe; time is computed in exact rationals and rounded once to
  float64, must strictly increase, and is never resampled (dropped frames stay
  dropped).
* **Truncated recordings.** 824 files end inside a row without a final newline, which
  is why most recordings are short (median 0.17 s; 184 last at least 1 s). A file
  without a terminating newline always loses its last line, even if it parses,
  because the cut may fall inside a number. Any other malformed row refuses the file.
* **Head-worn device.** Head yaw/pitch/roll is exactly zero in every measured file, so
  head pose was not recorded; `head_rotation` stays unavailable rather than becoming
  an identity. The 11 files with nonzero head values are refused, because their
  frame convention would then matter and is not declared.
* **Channels.** `face` holds all 52 ARKit coefficients in `[0, 1]`; `left_gaze` and
  `right_gaze` are exact unit vectors `(cos p sin y, sin p, cos p cos y)` of the eye
  yaw and pitch (eye roll is unrepresentable in a direction and is reported); `blink`
  holds the two `EyeBlink` coefficients as eyelid closure. All are `observed=False`
  ARKit solves.
* **No synchronisation with the skeleton.** The face timecode is the phone's; the
  skeleton's take clock is Shogun's. FBX take start times differ from the face
  recordings by an hour plus seconds that vary between takes, and no observable is
  shared (the head is not tracked by the face device). Face states therefore carry
  their own clock and are separate samples; they are never merged frame-wise.

Result: 911 face-only states in R2 under `derived/3d-lex-face-llf-v1/states/`; 86
refused (63 with no complete row, 12 whose header is another sign's junk marker, 11
with recorded head rotation).

## 10. Independent hand validation: StretchSense gloves (`data_engineering/glove_validation.py`)

The gloves report per-joint flexion as a local z angle at 120 Hz on a device
timecode slaved to the Vicon master. For finger joints 1–3 of index, middle, ring and
little fingers, the skeleton-side quantity is the **twist about the declared local
`+x` axis** of `Q = R(t₀)ᵀR(t)`: with `Q = (w, v)`, `w ≥ 0`, twist `= 2 atan2(v·x̂, w)`,
unwrapped over consecutive frames. If `R(t) = R_rest exp(φ(t)[x̂]×)` then the twist
equals `φ(t) − φ(t₀)` exactly; tests confirm `1e-9°` through arbitrary rest rotations.
The axis is declared, not fitted, so no sign is chosen from the data; a 20-sign pilot
fixed the axis and the side pairing before the full run, and excluded thumbs, whose
flexion is not about local `x`. One 120 Hz offset must explain all 24 joints of both
hands. The thresholds are the elbows' (§5), reused unchanged.

Glove clock defects, each handled explicitly: the device counter restarts a second's
frame field (`000 001 002 000 002 …`), so every row whose label lies in an overlap
interval `[cᵢ, max_{j<i} c_j]` is ambiguous and dropped from both sides; repeated
labels with different values are dropped; a file needing more than 10% of rows
dropped is refused.

| Hand | Validated | Failed | No glove export | Other |
|---|---:|---:|---:|---:|
| Left | 362 | 45 | 195 | 9 |
| Right | 541 | 63 | 0 | 7 |

Validated joints agree at median r 0.9993 with median centred error 0.18° (p95 2.96°).
Failures concentrate at joint 1 (MCP), which also abducts; a single-axis twist is a
weaker proxy there. The 195 missing left gloves are absent from the published archive
(plausibly one-handed signs; unverified). A failed or missing hand is withheld by the
manifest, not repaired.

## 11. Gloss candidates: ASL Signbank ECV (`data_engineering/signbank_ecv.py`)

The ELAN ECV snapshot (4,527 ID glosses, SHA-256 `7f51c84d…1044`, CC BY-NC-SA 4.0) is
parsed strictly: no DTD, entity, CDATA, comment or processing instruction; unique ids
and glosses; every entry's external reference declared. For each English sign label,
**stem** candidates are ID glosses whose variant-free stem equals it (`ABOUTb`,
`CAT8`, `CHAMPb5` → `ABOUT`, `CAT`, `CHAMP`; a hyphenated numeral stays in the stem)
and **keyword** candidates list it among their English keywords. Of 1,001 labels: 744
have one stem candidate, 195 several (lexical variants a person must choose between),
59 keywords only, 3 none. Nothing selects a candidate, and every record requires
review: a shared English label does not show that the performed sign is the Signbank
sign. Derived files carry the share-alike licence and citation.

## 12. Corpus manifest and loading (`data_engineering/corpus_manifest.py`, `data/manifest_dataset.py`)

`scripts/w4_ingest_3dlex.py` joins, per sign, the skeleton state and its marker
result, both hand results, the face state and the gloss candidates into one
canonical, write-once manifest (`derived/3d-lex-corpus-manifest-v2/manifest.json`,
SHA-256 `c7617289…61e7`). Admission is a pure function of the record, rechecked on
every load:

* skeleton sample: marker validation passed; its own available channels;
* each hand channel: that hand's glove validation passed;
* face sample: a converted face state, on its own clock.

`ContentCache` keeps at most a declared number of bytes locally, named by SHA-256, and
re-hashes every payload whether it came from R2 or disk; a corrupted file is refetched.
`ManifestMotionDataset` serves only admitted samples of one kind and one partition,
with every unadmitted channel masked, and the manifest's performer split makes the
partitions disjoint by construction. An end-to-end run on the real manifest served
426 training and 185 test skeleton samples with no overlap and 911 face samples,
every collated mask matched the manifest, and the cache stayed within 5 MiB.

With two performer groups no leave-one-group-out plan exists; the manifest records a
descriptive train-on-one, test-on-the-other split and `confirmatory: false`.

## 13. Prepared adapter for the continuous source: BVH (`data_engineering/bvh_motion.py`)

The RIT/CUNY ASL Motion-Capture Corpus (first release: 98 multi-sentence stories from
3 signers, 30 s to 4 min each) ships per story a BVH and an FBX file, gloss start/end
keyframes, a referent list, an English translation and front, side and face videos.
Access is by request, so the adapter is written against the BVH format and is ready
before the files arrive:

* strict parsing of the hierarchy and motion block — one root, unique names, balanced
  braces, known non-repeated channels, row widths and frame count checked, finite
  values; `Frame Time` is parsed as an exact decimal and frame `i` is at `i·dt`;
* rotations compose in listed channel order, `R = R_a(θ₁) R_b(θ₂) R_c(θ₃)`, from
  standard right-handed elementary rotations (checked against Rodrigues to 1e-14), and
  forward kinematics `p_child = p_parent + R_parent (offset + T)` is checked exactly;
* a declared `BVHMapping` must cover every joint as mapped or excluded and name the
  root with position channels; the metre conversion is declared, not inferred;
* gloss keyframes convert to seconds exactly on the file's own clock.

The mapping for the real skeleton, and any face or gaze channels (the release lists a
face video, not face tracking), can only be declared after the first file is inspected.

## 14. What remains open for this source

* Face-to-skeleton synchronisation: unrecoverable from the release; would need a
  shared event or timecode from the publisher.
* Gloss identity: 744 single candidates and 195 variant choices await qualified
  review; none is a label.
* Performer diversity: two validated performers; any claim stays descriptive.
