# Word-level alignment supervision of the RVQ prior: data and derivation (2026-10-27)

`RVQPriorConfig.alignment_supervision` (opt-in, default 0, earlier SHAs unchanged) adds to `rvq_prior_loss`, on
text-conditioned training rows, the cross entropy `−Σ_n q[t, n] log Ā[t, n]` of the temporal decoder's
layer- and head-averaged cross-attention Ā at step t against q = uniform over the memory positions of the word
signed at t (`motion_rvq.alignment_loss`). Steps and rows without a word contribute nothing. The loss, the
per-frame span threading (`TextMotionDataset(word_spans=True)` → `collate_text_motion(word_spans=True)` → last
batch element (B, T, 2), −1 = none → `rvq_prior_loss(word_spans=...)`, each token step takes its first frame's span)
and the CLI flag (`train_motion_tokens.py --alignment-supervision`) are implemented and tested. **What is missing is
the per-record input field `sign_spans`**; no stored manifest has it yet.

## The input the code expects

Each manifest record that should be supervised carries

    "sign_spans": [{"gloss": "<label>", "seconds": [start, end]}, ...]

the record's time-segmented signs in order, in **seconds relative to the record's state's first frame** (end
inclusive of the sign's last frame). Seconds, not frames, because the trainer resamples ASLLRP (30 fps) to 24 fps
(`--fps 24`). `motion_rvq.frame_word_spans` turns this into per-frame memory spans on the final, resampled
timestamps. It matches text words to signs by an order-preserving alignment (`difflib.SequenceMatcher`) of their base
glosses (`planning.asllrp_gloss.parse_surface_token(...).base`; plan-line loci `a>…>b` and the `[yn`/`]` brackets
are stripped or skipped), so a sign whose label does not agree with any text word gets no span.

## What exists (ASLLRP DAI 2)

- `scripts/build_signbank_lexicon.py` (`build_utterances`), from the DAI sentence-sign CSVs, writes
  `derived/signbank-v1/utterances.json`. Each utterance has `frames = [u0, u1]` (CSV "Start/End frame of the
  containing utterance") and `signs[] = {gloss (Main entry gloss label), variant, occurrence, sign_type,
  frames = [a, b] (Start/End frame of the sign video), handshapes, sign_video}`, sorted by start frame. The
  frames are integers on the source video's 30 fps clock, inclusive, and counted on the same clock as `[u0, u1]`,
  not from the utterance start.
- `scripts/build_asllrp_gloss_manifest.py`: `text = " ".join(non-empty glosses)` and
  `gloss_sequence = [{gloss, sign_type, frames}]` (all signs, including those with an empty gloss). `source` keeps
  `collection, source, utterance, utterance_video`, **but not `[u0, u1]`**.
- `scripts/build_asllrp_plan_manifest.py` (corpora `asllrp-dai2-plan-v1/-v2`): copies the gloss record, so
  `gloss_sequence` survives, and replaces `text` with the plan line (`plan_to_text`, e.g. `[yn IX-3p>1 BOOK ]`).
  `plan.glosses` comes from the dominant-hand **surface tier** of the utterance export (quoted tokens merged), and
  `gloss_sequence` from the **time-segmented CSV**. No index links the two, and their lengths can differ.
- `build_mixed_text_manifest.py` keeps records as they are. `TextMotionDataset` previously read only `text` and
  the state. English-text corpora (How2Sign, YouTube-ASL, the `english` field of ASLLRP) have no word-to-time
  alignment at all.

## What is missing, and how to derive `sign_spans`

1. **The utterance start u0.** It is not in the manifests. Join each record to `utterances.json` on
   `(source.collection, source.source, source.utterance)` and take `frames[0]`. As `validate_sign_recognizer.py`
   already does, check that `u1 − u0 + 1 == len(state.timestamps)` (the stored state covers exactly the utterance's
   frames). Drop the spans of records that fail this check, and report how many fail.
2. **Seconds.** `seconds = [(a − u0) / fps, (b − u0) / fps]` with fps the state's own rate
   (`1 / median(diff(timestamps))`, 30 for ASLLRP), before any resampling. Keep only signs with a non-empty gloss
   and `u0 ≤ a ≤ b ≤ u1`.
3. **Word mapping.** For the gloss manifest, the text words are exactly the non-empty glosses in order, so the alignment is
   one-to-one. The exception is a main-entry label that contains a space (not checked here): it splits into several
   words, and only its agreeing parts keep a span. For plan lines, `frame_word_spans`' alignment of base glosses
   keeps the agreeing signs. Measure, per corpus, the share of plan words that get a span (the surface tier and the
   CSV may differ in label form, classifier decoration or merged quotes) before turning the option on.
4. **Rebuild, never overwrite.** Write the new field into a **new** manifest id (for example `asllrp-dai2-plan-v3`
   and a matching mixed manifest) with a new SHA. The builders refuse to overwrite existing keys, and every earlier
   run stays reproducible.

## Caveats

- Signs do not cover every frame. Transitions, holds and the rest pose between signs have no span and are not
  supervised. The optional `guided_attention` diagonal prior still covers them.
- English-text rows cannot be supervised this way. They would need an English-to-gloss alignment that does not
  exist in the repository.
- The memory positions are byte positions, so the option refuses a prior with a pretrained `text_model`. That
  prior's memory is the language model's own tokenisation.
