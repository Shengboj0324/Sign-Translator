# Sources this package reuses

Surveyed on 2026-10-29 from `main` at `c7abc24`. This package adds plain-language reviewer material and an operator
guide. It does not replace the study protocol or its code, which stay where they are.

## Human evaluation (plan item 3b): protocol, code and tests

| Path | What it is | How this package uses it |
|---|---|---|
| `docs/HUMAN_EVALUATION_PROTOCOL.md` | The team's protocol. Systems (reference, plan, translate, english, stitch), Latin square over reviewer groups, blinding, analysis table (Wilson CI, paired permutation, Fleiss' kappa), consent and payment rules | The study design and the analysis in `OPERATOR_GUIDE.md` follow it |
| `docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md` | The current reviewer guide, copied into every packet zip | Too technical for these reviewers: it asks them to compute a SHA-256 in a terminal and to edit a CSV. `FOR_REVIEWERS.md` and `RATING_SHEET.md` are the plain-language versions. |
| `signtranslator/eval_framework/human_study.py` | `design_study`, `write_sheet`, `parse_sheet`, `score_study`, `wilson`, `SHEET_COLUMNS` | Defines the `sheet.csv` columns that `score_human_eval.py` accepts. `rating_sheet.csv` maps onto them. |
| `scripts/build_human_eval_packet.py` | Builds the blinded packets (`packet/group-<g>/`, private `key/key.json`). `--renderer avatar` renders on the MakeHuman avatar. `--final-study` is required for `test`. | Clip production, step 2 of `OPERATOR_GUIDE.md` |
| `scripts/score_human_eval.py` | Scores returned `sheet.csv` files against the key | Scoring, step 7 |
| `tests/test_eval_human_study.py`, `tests/test_build_human_eval_packet_script.py` | Tests for the code above | None |
| `docs/EVALUATION_FRAMEWORK.md` §4 and §6 | Pre-registration lock, test-set firewall, blinded comprehension versus preference | Analysis rules |

## Rendering

| Path | What it is |
|---|---|
| `scripts/render_avatar.py` | Renders one canonical state (`.npz` or R2 key with SHA) on the MakeHuman rig. Blender Workbench, fixed front camera, `avatar.mp4` plus `verification.json` |
| `docs/AVATAR_RENDER.md` §12 | Retarget, face mapping, render CLI and measurements |
| `Sign Translator Stage Documentation/evidence/avatar-retarget-2026-10-22/` | Verification records of the first avatar renders |
| `scripts/render_text_to_motion_samples.py` | Side-by-side skeleton previews of real and generated motion. An inspection aid, not for reviewers |
| `signtranslator/generation/skeleton_preview.py`, `generation/face_preview.py` | Skeleton renderer with a face panel (the packet's default renderer) |
| `scripts/round15_stage2.py` (`render_form`), `scripts/round15_compose_eval.py` | Round-15 compositions (core_crossfade, oracle) and their controls (`/mismatched_words`, `/shuffled`, `/rest_only`). The states are built in memory only and are never saved or rendered to video. |
| `tests/test_render_avatar.py`, `tests/test_avatar_retarget.py` | Tests for the renderer |

## Earlier review packet (gloss review)

| Path | What it is |
|---|---|
| `docs/GLOSS_REVIEW_PROTOCOL.md` | Reviewer qualification, two independent reviewers, conflicts, consent and payment, SHA-256 reviewer key (§5) |
| `Sign Translator Stage Documentation/evidence/gloss-review-packet-2026-10-18/` | `README.md`, `review-packet.csv`, `previews-index.json`. The model for packet layout and attribution |
| `scripts/w6_render_review_previews.py` | Skeleton previews for the gloss reviewers |
| `Sign Translator Stage Documentation/outreach-drafts-2026-10-18.md` §3 | Draft request for qualified gloss review |

## Plan, licensing and model card

| Path | What it says that matters here |
|---|---|
| `docs/NEXT_PHASE_PLAN.md`, "Item 3b" | Tooling is complete. The study waits for a generator and for reviewers. |
| `docs/NEXT_PHASE_PLAN.md`, "Owner directive: the 2.55 milestone" | After M1: evaluation in realistic translation scenarios, then human sign reviewers |
| `docs/NEXT_PHASE_PLAN.md` (round-3 split, R6) | How2Sign **test signer 9** is the untouched final check. Validation signers are 3, 1/2 and 11. |
| `docs/NEXT_PHASE_PLAN.md` (sources sections) | ASL Citizen is recognition-only. Sem-Lex's terms are restrictive (public statements need approval). |
| `releases/research-v2b-planner-v2.MODEL_CARD.md`, `scripts/build_model_card.py` | Intended use: research and evaluation by Deaf signers; outputs presented as machine-generated. Human evaluation: pending. |
| `reports/Model weights licensing review.md` (only in the owner's main checkout, not tracked in git) | Per-dataset verdicts: How2Sign CC BY-NC 4.0; ASLLRP research-only under DAI terms; ASL Citizen high risk (recognition consent); YouTube-ASL medium risk |
