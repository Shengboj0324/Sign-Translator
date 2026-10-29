# Reviewer package

This folder holds the materials for the testing phase with Deaf/ASL signer reviewers. The reviewers are not
technical, so the owner handles every technical step: rendering, delivering the videos, and collecting and entering
the answers. The study design and the scoring code are unchanged: see `docs/HUMAN_EVALUATION_PROTOCOL.md`.

| File | For | What it is |
|---|---|---|
| `FOR_REVIEWERS.md` | Reviewers (print large) | What the project is, what they do, how long it takes, that taking part is voluntary, privacy, contact |
| `RATING_SHEET.md` | Reviewers (print one page per video) | Five questions: what was signed, clarity 1-5, naturalness 1-5, hand shape / movement / face, comments |
| `CONSENT_FORM.md` | Reviewers and owner | Plain-language consent, with optional separate consents (filming, quotes, training use, recontact). Check it with an ethics advisor before use. |
| `rating_sheet.csv` | Owner | Blank entry template, one row per rating page |
| `OPERATOR_GUIDE.md` | Owner | Producing clips, study design, sessions, data entry, storage, scoring, accessibility checklist, what is not built yet |
| `SOURCES.md` | Owner | Existing repo material this package reuses, with paths |
| `.gitignore` | Owner | Keeps returned answers, keys, consent forms and recordings out of git |

**Constraints:**
- Research use only, non-commercial.
- ASL Citizen and YouTube-ASL material is never shown to reviewers.
- ASLLRP-derived clips go only to consenting reviewers, under the DAI terms.
- How2Sign test signer 9 and all `test` partitions stay unseen until the final study, unless the owner decides
  otherwise and records it.
- Filled sheets, keys and consent forms are kept in private storage, never in this repository.
