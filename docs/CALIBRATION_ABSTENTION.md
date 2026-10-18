# Calibrated error bounds and abstention

`generation/conformal_abstention.py` gives gap reconstruction a calibrated error bound
with a finite-sample guarantee, and uses the bound to decide when to abstain. It is
item 5 of the acquisition guide's remaining engineering — uncertainty, calibration and
declared abstention — implemented and run on the real 3D-LEX manifest.

## 0. Honest scope (read first)

The reconstructed quantity is solved motion inside a hidden span, not translation
quality. The guarantee is the split-conformal theorem's and needs calibration and test
units to be exchangeable. Calibrating on one performer and testing on another is a
**check of transfer**, not an instance of the theorem. Two performers make every
cross-performer statement descriptive.

## 1. Bound

Nonconformity of a unit is its error divided by a difficulty computed from visible
data, `s = e / (floor + d)`. With calibration scores `s_1..s_n` and
`k = ⌈(n + 1)(1 − α)⌉`, the bound for a new unit is `q (floor + d)` with `q = s_(k)`
(infinite if `k > n`). Exchangeability gives `P(e ≤ bound) ≥ 1 − α`, and with distinct
scores coverage is at most `1 − α + 1/(n + 1)`. The rank is computed in integers, so
floating error cannot move it.

* **Unit.** One hidden span of one sign. Its error is the largest geodesic error over
  the span's scored cells, so the bound holds for every cell of the span at once.
* **Difficulty.** The largest geodesic angle, over elements, between the visible
  rotations immediately before and after the span: the arc the interpolation must
  bridge. A test perturbs hidden targets and asserts the difficulty is unchanged.

## 2. Abstention

Reconstruct only when the bound is at most a declared tolerance `τ`. Then
`P(accept ∧ e > τ) ≤ P(e > bound) ≤ α`: the rate of accepted reconstructions that miss
the tolerance is controlled whatever the abstention rate is. Abstention is the honest
output when the bound is too wide.

## 3. Tests

The rank is exact, including the infinite case; a 4,000-trial simulation on
exchangeable heteroscedastic data lands coverage inside
`[1 − α, 1 − α + 1/(n + 1)]` within three standard errors and keeps the accepted
violation rate at most `α`; abstention summaries count exactly; difficulty reads no
hidden value.

## 4. Real run (`scripts/w5_conformal_abstention.py`)

Declared before the run: `α = 0.1`, floor 0.1°, `τ = 2°`, spans 0.25, 0.5 and 1.0 s,
geodesic interpolation (the method the W4 comparison selected). Samples come through the
manifest dataset, so an unvalidated hand is never scored. Per channel, the bound is fit
on a seeded half of the training performer and evaluated on the other half (the
theorem applies), and refit on all of that performer and evaluated on the other
performer (transfer).

| Span | Channel | Same performer: coverage | abstain | accepted miss | Other performer: coverage | abstain | accepted miss |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25 s | body | 0.886 | 0.80 | 0.005 | 0.962 | 0.60 | 0.011 |
| 0.25 s | left hand | 0.937 | 0.82 | 0.009 | **0.662** | 0.38 | 0.022 |
| 0.25 s | right hand | 0.910 | 0.54 | 0.016 | 0.936 | 0.57 | 0.000 |
| 0.25 s | head | 0.857 | 0.28 | 0.005 | 0.978 | 0.17 | 0.000 |
| 0.5 s | body | 0.847 | 0.92 | 0.000 | 0.951 | 0.70 | 0.005 |
| 0.5 s | left hand | 0.908 | 0.94 | 0.000 | **0.659** | 0.48 | 0.030 |
| 0.5 s | right hand | 0.904 | 0.72 | 0.005 | 0.929 | 0.68 | 0.000 |
| 0.5 s | head | 0.938 | 0.54 | 0.014 | 0.956 | 0.35 | 0.005 |
| 1.0 s | body | 0.923 | 1.00 | 0.000 | 0.933 | 0.75 | 0.000 |
| 1.0 s | left hand | 0.811 | 1.00 | 0.000 | **0.649** | 0.51 | 0.075 |
| 1.0 s | right hand | 0.899 | 0.81 | 0.000 | 0.929 | 0.77 | 0.000 |
| 1.0 s | head | 0.900 | 0.80 | 0.014 | 0.911 | 0.50 | 0.033 |

Reading it:

* **Within a performer the guarantee holds as stated.** Mean held-out coverage over the
  twelve rows is 0.893 against 0.90; individual rows vary by the expected sampling error
  of about ±0.03 for 110–210 units.
* **Transfer works for body, right hand and head** (0.91–0.98, conservative) **and fails
  for the left hand** (0.65–0.66): the other performer's left hand is harder than the
  calibration performer's. A left-hand bound must not be carried across performers.
* **The controlled quantity stays controlled.** The accepted-miss rate is at most 0.075
  in every row, including the left-hand transfer rows, because abstention absorbs much
  of the miscalibration.
* **A 2° tolerance on a span's largest error is strict**, so abstention is common and
  rises with span length. The tolerance is a declared operating point, not a claim
  about acceptable signing quality, which needs qualified review.

Evidence: `evidence/w5-calibration-abstention-2026-10-18/report.json`.
