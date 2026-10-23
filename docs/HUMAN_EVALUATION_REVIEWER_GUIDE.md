# Reviewer guide: judging ASL signing videos

Thank you for reviewing. You may stop at any time. You are paid for the time you spend, including if you stop.

## What you will see

Your packet uses one of two kinds of video (the study team will tell you which).

**A 3D avatar:** a plain human figure in a dark top, seen from the front, signing. The face shows eyebrow movements,
blinks, eye gaze and mouth shapes. The figure is not a real person and has no facial hair or hairstyle on purpose.

**A simple stick figure** (older packets):
- Top two panels: the upper body from the front and from the side. The signer's left arm is blue and the right arm
  is red.
- Bottom two panels: close-ups of the left and right hand.
- Right-hand panel: a simple face. It shows head turns and nods, eyebrows, eyelids, eye gaze and the mouth. A grey
  outline with no features means the face could not be shown in those frames.

The videos come from different sources, and some may be clearer than others. Please judge each one on its own. You
will not see the same sentence twice.

## For each row of your sheet

Watch the video named in the `video` column as many times as you need. Then fill in:

| Column | What to write |
|---|---|
| `viewable` | `yes`, or `no` if the video does not play or you cannot judge it at all. If `no`, leave the rest of the row empty. |
| `translation` | What was signed, in your own English words. Write it **before** reading the four options. |
| `choice` | Which of `option_a` .. `option_d` was signed: `a`, `b`, `c` or `d`. If you are unsure, pick the closest one. |
| `sentence_type` | `statement`, `yes_no_question`, `wh_question` or `rhetorical_question` |
| `negated` | `yes` if the signed sentence is negative (for example a headshake or NOT), otherwise `no` |
| `topic` | `yes` if a topic is set up at the start (for example with raised brows), otherwise `no` |
| `naturalness` | 1 = not signing; 2 = hard to follow; 3 = understandable but unnatural; 4 = mostly natural; 5 = fluent and natural |
| `reviewer_sha256` | Your reviewer key, the same on every row (see below) |
| `notes` | Anything else: what looked wrong, which sign was unclear, or what was missing on the face |

- Judge the signing you see, not the English options.
- Do not change the `item_id`, `video` or `option` columns.
- You may leave rows blank; blank rows count as not reviewed.
- Please work alone and do not discuss the videos with other reviewers until everyone has returned their sheet.

## Your reviewer key (never your name)

1. Choose a private passphrase and keep it.
2. Compute its SHA-256 (for example `printf '%s' 'your passphrase' | shasum -a 256`).
3. Paste the 64-character result into `reviewer_sha256` on every row you answer.

The study team never sees your name on the sheet.
