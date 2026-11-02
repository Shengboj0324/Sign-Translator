# Reviewer package

This folder holds the materials for the testing phase with Deaf/ASL signer reviewers. The reviewers are not
technical, so the owner handles every technical step: rendering, delivering the videos, and collecting and entering
the answers. The study design and the scoring code are unchanged: see `docs/HUMAN_EVALUATION_PROTOCOL.md`.

| File | For | What it is |
|---|---|---|
| `FOR_REVIEWERS.md` | Reviewers (print large) | For the first rater read: what the project is, the two stages, about 2.5 hours, that some videos may be hard or poor on purpose, that taking part is voluntary, privacy, where the material comes from, contact |
| `FOR_REVIEWERS_ORIGINAL_SESSION.md` | Kept aside | The earlier version, for the original 30 to 45 minute session format. Not for this read. |
| `RATING_SHEET_TWO_STAGE.md` | Reviewers (print one page per video) | First rater read: the stage 1 questions, then the stage 2 choice of four sentences |
| `RATING_SHEET.md` | Reviewers (print one page per video) | Original session format. Five questions: what was signed, clarity 1-5, naturalness 1-5, hand shape / movement / face, comments |
| `CONSENT_FORM.md` | Reviewers and owner | Plain-language consent, with optional separate consents (filming, quotes, training use, recontact). Check it with an ethics advisor before use. |
| `CHECKER_GUIDE.md` | Deaf checker and owner (print) | One page, for a session the owner leads: watch the real recording first, then read the English and the flagged word, then answer |
| `CHECKER_CONSENT_FORM.md` | Deaf checker and owner | A consent draft that fits the checker's job, for the board |
| `rating_sheet.csv` | Owner | Blank entry template, one row per rating page |
| `DEBRIEF.md` | Reviewers, only after all sheets return; the review board first | What the pilot withheld and why. Draft. Never in a packet. |
| `OPERATOR_GUIDE.md` | Owner | Producing clips, study design, sessions, data entry, storage, scoring, accessibility checklist, what is not built yet |
| `SOURCES.md` | Owner | Existing repo material this package reuses, with paths |
| `.gitignore` | Owner | Keeps returned answers, keys, consent forms and recordings out of git |

**First rater read (2026-11-01):** start at `OPERATOR_GUIDE.md` §10. It uses two stages with `docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md` and `RATING_SHEET_TWO_STAGE.md`, in place of `RATING_SHEET.md`. Raters never use a terminal, compute a hash or edit a CSV: they watch, and answer on the form or in ASL, and the owner enters the answers into the generated sheets. Its pilot is 60 sentences and 8 raters, with two extra report-only arms, described in §10; nothing a rater reads while rating may mention them or word order. The review board sees all of it.

**Constraints:**
- Research use only, non-commercial.
- ASL Citizen, ASL STEM Wiki and YouTube-ASL material is never shown to reviewers.
- ASLLRP-derived clips go only to consenting reviewers, under the DAI terms.
- How2Sign test signer 9 and all `test` partitions stay unseen until the final study, unless the owner decides
  otherwise and records it.
- Filled sheets, keys and consent forms are kept in private storage, never in this repository.
