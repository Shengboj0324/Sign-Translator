# First Deaf-rater read: package plan

2026-10-31, 22:34 UTC; power and status updated 2026-11-01. Diagnostic: cannot clear the release bar.

## 1. Sentences
Draw 200 plus 60 pilot from D20's eligible (427 before leaving out the six with no selected word; distinct, ≥1.5 s, hand tracked ≥70%). Distractors: D20's other sentences only (tested); the builder's whole-partition pool would read C20 and fresh text.

D20, not the 1,000, because protocol §7 feeds misunderstood sentences into tuning. D20 only chose global levers by instrument, so human optimism should be small; disclose it.

Bar conditions failed: val signers untrained (plan text) but used for selection; wrist speed 2.60× and 2.39×. This read is at predicted timing (`core_crossfade@predicted`, command in `docs/NEXT_PHASE_PLAN.md`), not reference length.

## 2. Systems and power
First arm: W14b, gains on, best of 16, the arm amendment 6 kept (not the no-gains stack), at predicted timing; reference through the same avatar. Three groups; each rater sees 200 sentences (~67 per system). At an assumed 2 min per item, stage 1: 6.7 h, nine 45-minute sessions; stage 2 at an assumed 30 s per row adds 1.7 h.

Power by the method declared 2026-11-01, 03:28 UTC (Wilson plus method D; 200 sentences, three groups, sentence SD 1.0, rater SD 0.5; simulation only):

| Raters | Ratings per system | True 80% | True 75% | True gap 0 | True gap 5 |
|---|---|---|---|---|---|
| **9** | 600 | about 85% | about 33% | about 100% | about 65% |

About 15 raters give 80% power with a true 5-point gap, and about 45 give 80% at a true 75%. False pass for six raters or more: .036–.045 at a true 70%, .024–.055 at a true 10-point gap. This replaces the earlier table, whose power was Wilson's alone.

The build's own run, with the three-strata weights and 1,500 simulated studies (375 at 30 or more raters): nine raters 82% at true 80%, 31% at true 75%, 99% at gap 0, 58% at gap 5; 15 raters 81% at gap 5; 45 raters 77% at true 75%.

Nine raters, about 60 hours of stage 1, about 75 with stage 2. Three raters cannot give kappa. $500 per system implies $25/h on the 60 hours; $40/h gives about $2,400, or about $3,000 with stage 2. The pilot (60 sentences, 8 raters, `reviewer_package/OPERATOR_GUIDE.md` §10) adds 20 rater hours, about $800.

## 3. Missing
- **Composed states** (`round15_compose_eval.py --export-prefix`): built, synthetic only; not run on D20.
- **How2Sign packet** (`build_human_eval_packet.py`, `how2sign-draw`, `how2sign-packet`, `how2sign-stage2`): built, two-stage, synthetic only.
- **Negation truth:** English rule and checker's sheet built; the Deaf checker is needed. 26 of 80 negated sentences have no negation word cached.
- **Bar scorer** (`score_human_eval.py --bar`): built, on synthetic sheets.
- **Design, sheets** (`human_study.py`): passed, synthetic IDs.
- **Avatar** (`render_avatar.py`): reworked, `docs/AVATAR_RENDER.md` section 13: filter, hands clear of the drawn body (forearms may stay inside), one frame for every clip, hair and a plain top. Not yet seen by a Deaf signer. Pilot decides.
- **Hosting, blinding:** no hosting: the owner plays the videos or delivers them, and enters the answers (owner's flow, 2026-11-02); MP4 tags clean; speed may reveal the reference.
- **Consent, payment** (`reviewer_package/`): drafts; pay blank. 0.5 day.

## 4. Render cost
Measured: 26 s per 104-frame clip at 768 px after the avatar rework (was 7.9 s). 240 clips at a mean 170 frames: about 3 h, one Mac process; 600 clips: about 7 h. Pod untested. Use the Mac. R2, all files: about 65 GB, ~$1/month. The Mac has 39 GB free: upload and delete per clip.

## 5. Licensing (not legal advice)
How2Sign: CC BY-NC 4.0, research consent; attribute and state changes (done on `reviewer_package/FOR_REVIEWERS.md`). English options derive from How2: contracted reviewers only. No ASL Citizen or STEM Wiki clip appears, and the generator mix excludes them. v3d, trained on ASL Citizen, chooses both arms' best-of-16 words. The owner deferred consent until after the model finishes (22:39 UTC); this read keeps best of 16.

## 6. Owner's actions
Full notes for the owner: `reviewer_package/OPERATOR_GUIDE.md` §10.
- Recruit 9 Deaf raters plus 8 for the pilot.
- Set and pay rates.
- Get ethics or IRB review; sign consent.
- Arrange ASL instructions or an interpreter.
- Choose delivery; it must hold stage 2 back until stage 1 is returned. Raters never use a terminal or compute a hash: they answer on a form or in ASL, and the owner enters the answers.
- Find one Deaf checker who is not a rater.
