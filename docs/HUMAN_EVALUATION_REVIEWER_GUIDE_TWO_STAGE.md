# Reviewer guide: judging ASL signing videos, in two stages

Thank you for reviewing. You may stop at any time. You are paid for the time you spend, including if you stop.

You will work in two stages. Finish and return stage 1 before you get stage 2. Once returned, stage 1 is final.

## What you will see

Your packet uses one of two kinds of video (the study team will tell you which).

**A 3D avatar:** a plain human figure in a dark top, seen from the front, signing. The face shows eyebrow movements,
blinks, eye gaze and mouth shapes. The figure is not a real person and has no facial hair or hairstyle on purpose.

**A simple stick figure:**
- Top two panels: the upper body from the front and from the side. The signer's left arm is blue and the right arm
  is red.
- Bottom two panels: close-ups of the left and right hand.
- Right-hand panel: a simple face. It shows head turns and nods, eyebrows, eyelids, eye gaze and the mouth. A grey
  outline with no features means the face could not be shown in those frames.

The videos come from different sources, and some may be clearer than others. Please judge each one on its own. You
will not see the same sentence twice.

## Stage 1: watch and answer (`sheet.csv`)

Watch the video named in the `video` column as many times as you need. Then fill in:

| Column | What to write |
|---|---|
| `viewable` | `yes`, or `no` if the video does not play or you cannot judge it at all. If `no`, leave the rest of the row empty. |
| `translation` | What was signed, in your own English words |
| `sentence_type` | `statement`, `yes_no_question`, `wh_question` or `rhetorical_question` |
| `negated` | `yes` if the signed sentence is negative (for example a headshake or NOT), otherwise `no` |
| `topic` | `yes` if a topic is set up at the start (for example with raised brows), otherwise `no` |
| `naturalness` | 1 = not signing; 2 = hard to follow; 3 = understandable but unnatural; 4 = mostly natural; 5 = fluent and natural |
| `reviewer_sha256` | Your reviewer key, the same on every row (see below) |
| `notes` | Anything else: what looked wrong, which sign was unclear, or what was missing on the face |

- Please answer every row, in order. If a video cannot be judged, mark it `viewable` = `no` instead of leaving it blank.
  If you need to stop, tell the study team; rows you have not reached stay empty.
- Sentence type and topic help the team describe the signing; they are not graded in this study.
- Do not change the `item_id` or `video` columns.
- Please work alone and do not discuss the videos with other reviewers until everyone has returned both stages.

Return the sheet to the study team as it is. Please do not edit it after returning it.

## Stage 2: choose the sentence (`stage2-….csv`)

After stage 1 comes back, the team sends you a second sheet. It lists each video you could view, with four English
sentences. Your key and a code for your stage 1 sheet are already filled in. You may watch the videos again.

| Column | What to write |
|---|---|
| `choice` | Which of `option_a` .. `option_d` was signed: `a`, `b`, `c` or `d`. If you are unsure, pick the closest one. |

- Judge the signing you see.
- Do not go back to change stage 1, and do not change any other column.

## Your reviewer key (never your name)

1. Choose a private passphrase and keep it.
2. Compute its SHA-256 (for example `printf '%s' 'your passphrase' | shasum -a 256`).
3. Paste the 64-character result into `reviewer_sha256` on every row of stage 1. Give the team the same key before you
   start, so they know which sheet to expect from you.

The study team never sees your name on the sheet.
