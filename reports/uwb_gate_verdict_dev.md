# UWB lifter gate: the two missing rows and a row-by-row verdict (development report, 2026-10-30 10:18 UTC)

Branch `uwb-verdict`, from main `ecaa490`. The declared gate is in docs/NEXT_PHASE_PLAN.md: "UWB lifter: declared selection and gate" (03:34 UTC), amendment 1 (03:59), owner declaration (04:02), and the selection result `inside` (04:20).

`uwb_lifter_gate.py` leaves two declared rows uncomputed:

- (A) the relative row against How2Sign validation;
- (B) the downstream row.

This branch adds both, plus a verdict over every row.

## What was built

| File | Purpose |
|---|---|
| `scripts/uwb_how2sign_val_medians.py` | (A) the xy2 bundle's How2Sign-val median per gate group, computed by the gate's own code path |
| `scripts/uwb_downstream_check.py` | (B) the frozen `back-translator-kp2d-uwb-ft-v1` reads lifted, ours and blank on the gate's pairs |
| `scripts/uwb_gate_verdict.py` | every declared row with its value, bar and status, the overall verdict, and a cross-check against `proposal_check` |
| `runpod/fleet/uwb_downstream.sh` | pod job: (A), then (B), then the verdict. Takes the gate report key as `$1` |
| `tests/test_uwb_downstream_check.py`, `tests/test_uwb_gate_verdict.py` | 18 tests on synthetic data |

Tests: these 18 pass together with the existing UWB suites (`test_uwb_lifter_gate`, `test_convert_uwb_clips_script`, `test_uwb_outside_frame_accuracy_script`, `test_de_uwb_keypoints`), 61 in total, under warning-strict pytest.

## (A) Relative to How2Sign validation

**The bundle's own validation metrics do not cover the gate's groups**, so the medians were recomputed. The training report (`world-regressors-xy2-v1-training.json`, sha `79b8d5f0…`) differs from the gate in four ways:

- **Arms:** it records per-side angles between *local joint rotations*, solved with the upright chest. The gate's upper arms and forearms are angles between *world bone directions*, from forward kinematics.
- **Fingers:** it records local IK *rotation* angles. The gate uses finger bone *directions* in the palm frame.
- **Wrists:** it records the palm frame of the world hand points. The gate uses the state's hand frame in the camera frame.
- **Pooling:** it records per side only, on 30,000 sampled frames, with no common-frame rule. A pooled median cannot be derived from two per-side medians.

**How the medians were recomputed.** They use the same validation data the bundle was validated on:

- **Data:** the SHA-pinned v2 world pairs plus their legacy Holistic arrays, signers h2s-1 and h2s-2 only. Signer 9 is refused by file name and by signer field.
- **Lifted states:** `uwb_keypoints.convert_clip` for exactly the gate run's methods: `lifter[inside]`, the learned rule, `margin:0.1`, `margin:0.25`, and pseudo. Together they define the common frames.
- **Truth:** Tasks world landmarks through `canonical_from_landmarks`.
- **Scoring:** `uwb_lifter_gate.evaluate_pair` (lag pinned to 0, since both states share one clock), the gate's per-pair cap of 600, then `group_stats`. Confidence intervals resample How2Sign source videos.

**Local result.** 399 validation clips, 396 evaluated. Three were refused because state construction hit a non-rotation matrix (ValueError "must belong to SO(3)"). Primary method `lifter[inside]`, common frames:

| Group | n | Videos | How2Sign-val median [95% CI] | 1.6× (YouTube bar) |
|---|---|---|---|---|
| upper arms | 1,159 | 48 | 6.09° [4.92, 7.80] | 9.74° |
| forearms | 1,032 | 45 | 7.17° [5.57, 9.14] | 11.47° |
| wrists | 759 | 43 | 13.55° [9.55, 17.91] | 21.68° |
| fingers | 224,248 | 97 | 10.25° [9.93, 10.62] | 16.40° |

Readings:

- **For arms, the relative bar binds before the absolute one.** YouTube needs upper arms ≤ 9.74° (absolute bar 12°) and forearms ≤ 11.47° (absolute bar 18°).
- **Arm counts are small.** The chest-up crops put most elbows outside the frame, and the inside rule (and pseudo) then drop the arm. Each arm group still clears the verdict's "measured" rule (≥ 200 values from ≥ 5 videos).
- **The pod job recomputes this** and stores it as `gate-rerun/how2sign-val-medians-v1.json`.

## Chest-up forearm survival (Validity row), from the selection report

This is measured by `uwb_outside_frame_accuracy.py` (`lifter-v1/selection/outside-frame-xy2-v1.json`, sha `3e26180a…`). The value is the selected rule's forearm survival over MediaPipe's own keep decision, on the How2Sign validation chest-up crops (all distance bins, 3,182 frames per side):

| Side | `inside` survival | MediaPipe survival | Ratio | Bar ≥ 0.8 |
|---|---|---|---|---|
| Left | 0.000 | 0.286 | 0.00 | **fail** |
| Right | 0.0233 | 0.0069 | 3.38 | pass (MediaPipe keeps almost nothing) |

**The declared gate therefore fails on this row (Left) whatever the YouTube numbers are.**

- On these crops no left forearm lies inside the frame: the inside bin has 0 frames.
- `inside` keeps none of them. MediaPipe keeps 28.6%.

The UWB-weighted survival tells the same story: 0.0 against 0.455 on the left.

## (B) Downstream, `scripts/uwb_downstream_check.py`

**Pairs.** The gate report keeps `video_id`, `clip_id`, `sample_id`, `cue_start`, `iou`, `fps`, `zip`, `member_sha256`, `alignment` and `aligned_frames`. It does not keep our state key, its SHA or the cue text. Each pair is reconstructed from the same inputs the gate read, by rerunning `pair_clips` at the gate's fps. Any of these differences refuses the pair, and refusals are counted:

- a different clip, cue or IoU;
- a different cue start;
- an fps that differs from the report's fps row;
- a different member SHA;
- a different aligned-frame count.

**Sample.** At most 2,000 pairs. One pair per video is taken first (seed 0), then a seeded draw fills the rest.

**Reads.** Three states per pair, all against the same reference (our cue's English text):

- **lifted:** produced by `convert_uwb_clips.convert_member` itself. The converter's worker state is set as a conversion run sets it: xy2, `inside`, `generator` flags, estimated fps.
- **ours:** our state of the cue.
- **blank:** `blanked(ours)`, which keeps the duration.

Each read is cut to the gate's frame map (lag respected), resampled with `at_rate(·, 24)` and passed to `load_translator`/`translate`, exactly as the pilots read motion.

**Statistics.**

- Corpus BLEU per read, and the lifted/ours ratio.
- A paired bootstrap clustered by video (2,000 resamples, seed 0) for lifted − ours, lifted − blank and ours − blank. It is exact: BLEU is recomputed from summed segment statistics, and a test checks the result against `corpus_bleu` on hand-resampled videos.
- Exact-match rates.
- A breakdown by UWB partition.

**Verdict.** Pass when the ratio is ≥ 0.9 **and** the lifted − blank lower bound is > 0. If the ours − blank lower bound is ≤ 0, the row is not-evaluable.

## Verdict, `scripts/uwb_gate_verdict.py`

- **Rows:** every declared row with its declared bar. The script also checks the declared sample: rule `inside`, the xy2 SHA, 5 pairs per video, 2,000 resamples, 70 development videos excluded.
- **Not gating:** input p99 |z| and drift (diagnostics in the declaration). Stream-mismatch strata and frame-count difference "other" are reported but not gated.
- **Overall:** FAIL if any gating row fails. Otherwise PASS only if every gating row passes; any not-evaluable row gives NOT-EVALUABLE.
- **Cross-check:** all 22 `proposal_check` booleans are recomputed by the proposal's own definitions. Any difference is listed as a disagreement. The declared-versus-proposal differences (strata scope, drift gating) are listed separately.

## Pod command

The gate rerun stores its report through `store_local_files.py` under `$GATE_PREFIX/files/`. With `GATE_PREFIX=…/lifter-v1/gate-rerun`, the command is:

```
runpod/fleet/uwb_downstream.sh acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1/gate-rerun/files/gate-xy2-v1.json
```

It needs a GPU pod with `/root/work/models/face_landmarker.task` (as the gate pods have). It writes:

- `lifter-v1/gate-rerun/how2sign-val-medians-v1.json`
- `lifter-v1/gate-rerun/downstream-v1.json`
- `lifter-v1/gate-rerun/gate-verdict-v1.json` and `.md`

Each script refuses to overwrite an existing key.

## Not implemented exactly as declared or briefed

1. **BLEU:** the repo has no sacrebleu. Every pilot and helper uses `motion_to_text.corpus_bleu`, so that is used here.
2. **Windows:** each state is cut to the frame range of its matched frames, `[min, max]`, not to the matched frames alone. A state needs a strictly increasing, contiguous clock. Both windows cover the same time span, within a frame.
3. **Lifted face and head:** the lifted state uses the converter's defaults (blendshapes, head none). The gate used affine heads. The 2D instrument reads only the body and hands through its FK projection, so this does not change the read.
4. **Instrument contamination:** `back-translator-kp2d-uwb-ft-v1` was pretrained on 323,978 UWB train-split clips with these sentences. The per-partition reads are a diagnostic for this, not a gate.
5. **Instrument checkpoint SHA:** not pinned (3.6 GB, not downloaded here). It is recorded at run time; `--instrument-sha256` can pin it.
6. **(A) truth solver:** the truth uses `generator` flags (upright chest). The gate's "ours" uses the original solver. Arm directions are convention-free; only which frames hold an arm differs.
7. **Hip-visible forearm agreement:** this is the gate's "where our chest is valid" (both hips). UWB-only valid per side is the larger of the hand and forearm rates, as `proposal_check` computes it.
8. **No real verdict yet:** neither the gate rerun report nor (B) exists, so the full verdict has not been run on real data.
