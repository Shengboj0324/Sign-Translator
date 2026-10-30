"""Training-record selection for converted UWB clips: every manifest builder over ``convert_uwb_clips.py`` output must
pass its records through :func:`select_training_records`.

It **refuses** (``ValueError``) a converted record that was not checked for evaluation leakage (no ``eval_leak`` field,
or no ``eval_index_sha256``) or not checked against our own YouTube cues (no ``duplicate_of_our_cue`` field), and it
**drops**:

- records that are not ``converted`` or have no English text;
- evaluation leaks (``eval_leak``: exact or >= 0.8 5-gram Jaccard match with How2Sign / ASLLRP val or test sentences,
  ``build_youtube_asl_manifest.EvalIndex``);
- cues we already hold as our own Tasks-extracted state (``duplicate_of_our_cue``: one source per cue, ours preferred).

The partition is the record's ``partition``: our channel-hashed train/monitor split for videos we extracted (so a
video's UWB clips never cross it), the UWB train/dev split otherwise.
"""
from __future__ import annotations

import collections

REQUIRED = ("eval_leak", "eval_index_sha256", "duplicate_of_our_cue", "partition")


def select_training_records(records, *, eval_index_sha256: str | None = None) -> tuple[list[dict], dict]:
    """(kept records, counts by reason). ``eval_index_sha256``, when given, must equal every record's (one index)."""
    kept, why = [], collections.Counter()
    for r in records:
        if r.get("status") != "converted":
            why[f"status_{r.get('status')}"] += 1
            continue
        missing = [k for k in REQUIRED if k not in r]
        if missing or not r.get("eval_index_sha256"):
            raise ValueError(f"{r.get('clip_id')}: converted without the leak and overlap checks ({missing or 'eval_index_sha256'}); "
                             "reconvert with convert_uwb_clips.py --eval-manifest and --overlap-table")
        if eval_index_sha256 is not None and r["eval_index_sha256"] != eval_index_sha256:
            raise ValueError(f"{r.get('clip_id')}: checked against another evaluation index")
        if not r.get("text"):
            why["no_text"] += 1
        elif r["eval_leak"]:
            why["evaluation_leak"] += 1
        elif r["duplicate_of_our_cue"]:
            why["duplicate_of_our_state"] += 1
        else:
            kept.append(r)
            why["kept"] += 1
    return kept, dict(why)


__all__ = ["REQUIRED", "select_training_records"]
