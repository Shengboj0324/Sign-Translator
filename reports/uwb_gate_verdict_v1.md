**Overall: FAIL**

| Check | Row | Value | Bar | Status | Gating | Note |
|---|---|---|---|---|---|---|
| Setup | `visibility_is_selected_rule` | inside | inside | pass | yes |  |
| Setup | `bundle_is_xy2` | 02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde | 02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde | pass | yes |  |
| Setup | `pairs_per_video` | 5 | 5 | pass | yes |  |
| Setup | `bootstrap_resamples` | 2000 | 2000 | pass | yes |  |
| Setup | `development_videos_excluded` | 70 | 70 | pass | yes |  |
| Accuracy, medians | `median.upper_arms<=12.0` | 9.56 | 12 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median_ci_upper.upper_arms<=13.0` | 9.77 | 13 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median.forearms<=18.0` | 12.1 | 18 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median_ci_upper.forearms<=19.0` | 12.44 | 19 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median.wrists<=15.0` | 14.96 | 15 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median_ci_upper.wrists<=16.0` | 15.43 | 16 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median.fingers<=16.0` | 14.79 | 16 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, medians | `median_ci_upper.fingers<=17.0` | 14.97 | 17 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, p90 | `p90.upper_arms<=25.0` | 20.5 | 25 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Accuracy, p90 | `p90.forearms<=40.0` | 37.15 | 40 | pass | yes | rounded source: gate summary, 2 dp (uwb_lifter_gate.stat) |
| Relative to How2Sign validation | `relative.upper_arms<=1.6x` | 1.56979 | 1.6 | pass | yes | rounded source: ratio of two 2-dp medians |
| Relative to How2Sign validation | `relative.forearms<=1.6x` | 1.68759 | 1.6 | fail | yes | rounded source: ratio of two 2-dp medians |
| Relative to How2Sign validation | `relative.wrists<=1.6x` | 1.10406 | 1.6 | pass | yes | rounded source: ratio of two 2-dp medians |
| Relative to How2Sign validation | `relative.fingers<=1.6x` | 1.44293 | 1.6 | pass | yes | rounded source: ratio of two 2-dp medians |
| Global offset | `rotation_gain.upper_arms<=3.0` | 1.07 | 3 | pass | yes | rounded source: difference of two 2-dp medians |
| Global offset | `rotation_gain.forearms<=3.0` | 0.11 | 3 | pass | yes | rounded source: difference of two 2-dp medians |
| Strata | `strata_within_3deg_of_bar` | none over | bar + 3.0 (strata with >= 200 pairs) | pass | yes |  |
| Strata | `stratum.fc_diff=-1` | upper_arms 9.69, forearms 12.74, wrists 16.31, fingers 15.54 | bar + 3.0 | pass | no |  |
| Strata | `stratum.fc_diff=0` | upper_arms 9.55, forearms 12.02, wrists 14.77, fingers 14.65 | bar + 3.0 | pass | no |  |
| Strata | `stratum.fc_diff=other` | upper_arms 8.68, forearms 11.71, wrists 16.2, fingers 15.87 | bar + 3.0 | not-evaluable | no |  |
| Strata | `stratum.fps30=no` | upper_arms 9.91, forearms 13.25, wrists 13.4, fingers 13.96 | bar + 3.0 | pass | no |  |
| Strata | `stratum.fps30=yes` | upper_arms 9.44, forearms 11.78, wrists 15.61, fingers 15.12 | bar + 3.0 | pass | no |  |
| Strata | `stratum.hips=not_visible` | upper_arms 9.84, forearms 12.72, wrists 14.09, fingers 14.25 | bar + 3.0 | pass | no |  |
| Strata | `stratum.hips=visible` | upper_arms 9.29, forearms 11.61, wrists 15.92, fingers 16.91 | bar + 3.0 | pass | no |  |
| Strata | `stratum.stream_match=no` | upper_arms 16.23, forearms 30.78, wrists 71.88, fingers 33.5 | bar + 3.0 | fail | no |  |
| Strata | `stratum.stream_match=yes` | upper_arms 9.13, forearms 11.25, wrists 13.53, fingers 13.95 | bar + 3.0 | pass | no |  |
| Validity | `hand_agreement.Left>=0.88` | 0.90248 | 0.88 | pass | yes |  |
| Validity | `hand_agreement.Right>=0.88` | 0.904576 | 0.88 | pass | yes |  |
| Validity | `forearm_agreement_hips_visible.Left>=0.85` | 0.951078 | 0.85 | pass | yes |  |
| Validity | `forearm_agreement_hips_visible.Right>=0.85` | 0.889423 | 0.85 | pass | yes |  |
| Validity | `uwb_only_rate.Left<=0.1` | 0.0442524 | 0.1 | pass | yes |  |
| Validity | `uwb_only_rate.Right<=0.1` | 0.100612 | 0.1 | fail | yes |  |
| Validity | `chest_up_forearm_survival.Left>=0.8x_mediapipe` | 0 | 0.8 | fail | yes | fails by construction: 'inside' drops every point outside the image; rounded source: ratio of two 4-dp survivals (uwb_outside_frame_accuracy) |
| Validity | `chest_up_forearm_survival.Right>=0.8x_mediapipe` | — | 0.8 | not-evaluable | yes | unmeasured: MediaPipe keeps 18 outside-frame frames from >= 2 videos (needs 200 from 5); rounded source: ratio of two 4-dp survivals (uwb_outside_frame_accuracy) |
| Timing | `mirror_better_share<=0.03` | 0.0203405 | 0.03 | pass | yes |  |
| Timing | `lag_within_1_share>=0.9` | 0.726805 | 0.9 | fail | yes |  |
| Downstream | `downstream_back_translator` | — | >= 0.9 x ours and lifted - blank lower bound > 0 (ours - blank > 0), pooled and on UWB dev | not-evaluable | yes | no downstream report given (uwb_downstream_check.py) |
| Diagnostics | `input_p99_abs_z` | 6.07 | reference 4.0 | reported | no |  |
| Diagnostics | `drift_below_1_share>=0.95` | — | 0.95 | reported | no |  |

Cross-check against proposal_check: 22 booleans compared, 0 disagreements, not recomputed: none.
- declared differs from proposal `strata_within_3deg_of_bar`: proposal False, declared pass (the proposal judges every stratum (stream match, frame-count difference 'other'); the declaration gates hips, fps 30 and frame-count difference 0 / -1 and reports stream mismatch)

Values are compared unrounded wherever the inputs carry them; a row marked 'rounded source' compares the value as the gate's summary rounds it (a value exactly at the bar then passes).
