# Gloss review protocol

This protocol turns the English-label matches between 3D-LEX signs and ASL Signbank
ID glosses into reviewed labels. A label exists only when two independent qualified
reviewers choose the same Signbank entry for a sign. Answers are read by
`data_engineering/gloss_review.py`, which accepts nothing that does not follow this
protocol.

## 1. What is being decided

For each 3D-LEX sign (one row of `review-packet.csv`), the reviewer watches the
performed sign and decides which, if any, offered Signbank entry is the same sign.
The English label is a hint, not evidence: two entries can share an English word
and be different signs, and a label can name a sign the performer did not produce.

## 2. Reviewers

* **Qualification.** Deaf signers fluent in ASL, or hearing signers with an
  equivalent recognised qualification (for example NIC/RID certification or ASL
  teaching credentials). Record the basis privately; it is not entered in the sheet.
* **Two per sign, independent.** Each reviewer works alone on their own copy. No
  discussion until both sheets are returned.
* **Conflicts.** A reviewer who appears in 3D-LEX or in the Signbank entries' videos
  declares it and does not review those rows.
* **Consent and payment.** Reviewers are told the purpose, how answers are used, and
  that they may stop at any time; they are paid for the work, including the pilot.

## 3. Materials

* `review-packet.csv` — one row per sign, hardest first (variant choices, then
  keyword-only matches, then single candidates). Each `candidate_N` cell is
  `ID|GLOSS|URL`; the URL opens that entry on aslsignbank.com.
* `review-previews.zip` (R2: `derived/3d-lex-review-previews-v1/`) — one skeleton
  preview video per animated sign, named `<sign>.mp4`. Four panels: front and side
  of the upper body, and close-ups of the left (blue) and right (red) hand. These
  are skeleton renders of the solved motion, not video of the person. Signs whose
  file has no animation have no preview; mark them `unsure`.

## 4. Procedure for each row

1. Watch the preview at least twice, including the hand close-ups.
2. Open every candidate link and watch the Signbank video.
3. Fill the row:

| `decision` | When | `chosen_cve_id` |
|---|---|---|
| `confirm` | The row has one candidate and it is the performed sign | that candidate's ID |
| `choose` | The row has several candidates and one is the performed sign | the chosen ID |
| `none_match` | No candidate is the performed sign | empty |
| `unsure` | You cannot tell (unclear preview, no preview, regional variant) | empty |

   The ID is the number before the first `|` in the candidate cell. A `confirm` or
   `choose` without an ID is incomplete, even if the right answer seems obvious.
4. Put your reviewer key in `reviewer_sha256` on every answered row (§5).
5. Use `notes` freely; leave every other column unchanged.

Rows may be left blank; blank rows are simply unreviewed.

## 5. Reviewer key (never your name)

Choose a private passphrase and use its SHA-256 as your key on every row, for
example with `printf '%s' 'your passphrase' | shasum -a 256`. The key lets two
sheets be matched and agreement measured without any name entering the data. Keep
the passphrase; the same key must be used for all your rows.

## 6. Pilot, agreement and adjudication

1. **Pilot.** Both reviewers first answer the same 50 rows (the first 50 of the
   packet). Agreement is measured with Cohen's kappa on the chosen outcome
   (`review_agreement`). If kappa is below 0.6, the instructions are clarified with
   both reviewers before the main round; pilot answers are kept.
2. **Main round.** Both reviewers answer all remaining rows independently.
3. **Labels.** A sign is labelled only when both reviewers chose the same ID
   (`agreed_labels`). Any disagreement, `none_match` or `unsure` leaves it
   unlabelled.
4. **Adjudication.** Disagreements may go to a third qualified reviewer, who sees
   both answers. The adjudicated answer is recorded as that reviewer's own row, and
   a label then needs two matching choices among the three.
5. **Batch acceptance.** The labelled set is accepted for use if kappa over all
   doubly answered rows is at least 0.8; otherwise it is reported with its kappa
   and used only descriptively.

## 7. Returning sheets

Return each reviewer's CSV unchanged in structure. Every sheet is audited by
`audit_review_sheet`, which lists every problem row at once (missing ID, invalid
key, unknown decision, duplicate answer) without guessing. Only problem-free rows
count. The audit of each returned sheet is stored next to the packet.

## 8. Data handling and licence

The sheets contain no personal data. The packet is derived from ASL Signbank
(CC BY-NC-SA 4.0): non-commercial use, derivatives shared alike, citation required
(Hochgesang, Crasborn and Lillo-Martin, ASL Signbank,
doi:10.6084/m9.figshare.9741788.v2). Reviewed labels inherit those terms.

## 9. Status of the first returned sheet (2026-10-18)

Six rows were answered `Confirm`: agreement, and, apple, ask, balance, balloon. All
six are multi-candidate rows, so each needs `choose` with the chosen ID, and none
carries a reviewer key. No label is admitted yet. The row-by-row audit is in
`evidence/gloss-review-packet-2026-10-18/audit-2026-10-18.json`.
