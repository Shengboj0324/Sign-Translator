# First Deaf-rater read: package plan

2026-10-31, 22:34 UTC. Diagnostic: cannot clear the release bar.

## 1. Sentences
Draw 200 plus 20 pilot from D20's 427 eligible (distinct, ≥1.5 s, hand tracked ≥70%). Distractors: D20's other sentences only (tested); the builder's whole-partition pool would read C20 and fresh text.

D20, not the 1,000, because protocol §7 feeds misunderstood sentences into tuning. D20 only chose global levers by instrument, so human optimism should be small; disclose it.

Bar conditions failed: reference-length timing (`@predicted` waits for the next read); val signers untrained (plan text) but used for selection; wrist speed 2.60× and 2.39×.

## 2. Systems and power
First arm, no-gains stack, reference through the same avatar. Three groups; each rater sees 200 sentences (~67 per system). At an assumed 2 min per item: 6.7 h, nine 45-minute sessions.

| Raters | Ratings per system | Wilson ± at 75% | Power, true 75% | Detect 10 points |
|---|---|---|---|---|
| 3 | 200 | 6.0 | 35% | 66–70% |
| 6 | 400 | 4.2 | 62% | 93% |
| **9** | 600 | 3.5 | 76% | 98% |

Nine raters, about 67 hours; the bound needs 73.8% observed. With assumed clustering, power is 65–67% at true 75%, ≥98% at 80%, and Wilson falsely passes a true 70% 8–14% of the time, so add a clustered bound. Three raters cannot give kappa. $500 per system implies $25/h; $40/h gives about $2,700.

## 3. Missing
- **Composed states** (`round15_compose_eval.py`): D20 caches exist, no export. 1 day.
- **How2Sign packet** (`build_human_eval_packet.py`): no composition kind; `design_study` raises KeyError without markers. 1 day.
- **Negation truth:** English rule plus signer check. 26 of 80 negated sentences have no negation word cached. 0.5 day.
- **Bar scorer** (`score_study`): no paired interval or clustered bound. 0.5 day.
- **Design, sheets** (`human_study.py`): passed, synthetic IDs.
- **Avatar** (`render_avatar.py`): works. Bald, flat, hands through body, shake unfixed; v2 unstarted. Pilot decides.
- **Hosting, blinding:** nothing built; MP4 tags clean; speed may reveal the reference. 2 days.
- **Consent, payment** (`reviewer_package/`): drafts; pay blank. 0.5 day.

## 4. Render cost
Measured: 13 s per 147-frame clip at 768 px (docs' 45 s is stale). 600 clips: about 2.2 h, one Mac process. Pod untested. Use the Mac. R2, all files: about 65 GB, ~$1/month. The Mac has 39 GB free: upload and delete per clip.

## 5. Licensing (not legal advice)
How2Sign: CC BY-NC 4.0, research consent; attribute and state changes. English options derive from How2: contracted reviewers only. No ASL Citizen or STEM Wiki clip appears, and the generator mix excludes them. **But v3d, trained on ASL Citizen, chooses both arms' best-of-16 words**, a consent question.

## 6. Owner's actions
- Recruit 9 Deaf raters plus pilot.
- Set and pay rates.
- Get ethics or IRB review; sign consent.
- Arrange ASL instructions or an interpreter.
- Rule on v3d.
- Define "within 10 points" and "negation recovered".
- Choose delivery.
