"""Training-record selection for converted UWB clips: every manifest builder over ``convert_uwb_clips.py`` output must
pass its records through :func:`select_training_records`.

**Refusals** (``ValueError``): a converted record without the leak check (``eval_leak``, ``eval_index_sha256``) or the
overlap fields (``our_video``, ``duplicate_of_our_video``, ``duplicate_of_our_cue``, ``our_cue_overlaps``); a record
checked against another evaluation index than the builder's (``eval_index_sha256`` is required); a record whose flag
disagrees with the builder's own ``EvalIndex`` when one is given (a leak can never pass on a stale flag); our manifest
records that put one video or channel in two partitions; any partition outside ``{train, monitor}``.

**Drops** (counted by reason): records not ``converted`` or without text; evaluation leaks (exact or >= 0.8 5-gram
Jaccard against How2Sign / ASLLRP val and test, ``build_youtube_asl_manifest.EvalIndex``); and **cues we already hold
in a selected manifest record**. Ours is preferred only where our cue is actually selected (many of our cues are
dropped by our own manifests, e.g. arm-untracked ones), so the decision is made here, against ``our_records`` (the
records of the YouTube manifests the build also uses), not at conversion time:

- a timed video: the clip is dropped when a *selected* cue of ours is its pair (IoU >= 0.5) or covers at least
  ``max_cover`` of the clip's span (``our_cue_overlaps``: near-duplicates the one-to-one pairing misses);
- a video we hold but could not time (``duplicate_of_our_video``): every clip is dropped when any cue of that video
  is selected, since which one overlaps is unknown.

**Partition.** For a video we extracted, the partition is read from our manifest records of that video, else of its
channel (our split is channel-disjoint); a clip of our video whose channel has no selected record of ours is dropped
(``our_partition_unknown``) rather than re-hashed. Other clips keep UWB's train/dev split (``train``/``monitor``).
"""
from __future__ import annotations

import collections

REQUIRED = ("eval_leak", "eval_index_sha256", "our_video", "duplicate_of_our_video", "duplicate_of_our_cue",
            "our_cue_overlaps", "partition")
PARTITIONS = ("train", "monitor")
MAX_COVER = .5


def our_selection(our_records) -> tuple[set, dict, dict, dict]:
    """(selected sample ids, video -> partition, channel -> partition, video -> selected count) from our YouTube
    manifest records (``sign`` = ``<corpus>:<sample id>``, ``source.video_id``, ``source.channel_id``)."""
    selected, by_video, by_channel, counts = set(), {}, {}, collections.Counter()
    for r in our_records:
        sample = r["sign"].split(":", 1)[-1]
        source = r.get("source") or {}
        partition = r.get("partition")
        if partition not in PARTITIONS:
            raise ValueError(f"our record {r['sign']}: partition {partition!r} not in {PARTITIONS}")
        selected.add(sample)
        for table, key in ((by_video, source.get("video_id")), (by_channel, source.get("channel_id"))):
            if key is None:
                continue
            if table.setdefault(key, partition) != partition:
                raise ValueError(f"our manifest records put {key} in two partitions")
        counts[source.get("video_id")] += 1
    return selected, by_video, by_channel, counts


def select_training_records(records, *, eval_index_sha256: str, our_records, eval_index=None,
                            max_cover: float = MAX_COVER) -> tuple[list[dict], dict]:
    """(kept records, each with its resolved ``partition`` and ``partition_source``; counts by reason)."""
    if not eval_index_sha256:
        raise ValueError("eval_index_sha256 is required: the builder must name the evaluation index it trusts")
    selected, by_video, by_channel, our_counts = our_selection(our_records)
    kept, why = [], collections.Counter()
    for r in records:
        if r.get("status") != "converted":
            why[f"status_{r.get('status')}"] += 1
            continue
        missing = [k for k in REQUIRED if k not in r]
        if missing or not r.get("eval_index_sha256"):
            raise ValueError(f"{r.get('clip_id')}: converted without the leak and overlap checks "
                             f"({missing or 'eval_index_sha256'}); reconvert with convert_uwb_clips.py "
                             "--eval-manifest and an uwb-overlap-table-v2 --overlap-table")
        if r["eval_index_sha256"] != eval_index_sha256:
            raise ValueError(f"{r.get('clip_id')}: checked against another evaluation index")
        text = r.get("text")
        if eval_index is not None and text and bool(eval_index.leaks(text)) != bool(r["eval_leak"]):
            raise ValueError(f"{r.get('clip_id')}: its eval_leak flag disagrees with the builder's evaluation index")
        if not text:
            why["no_text"] += 1
            continue
        if r["eval_leak"]:
            why["evaluation_leak"] += 1
            continue
        partition, source = r["partition"], "uwb"
        if r["our_video"]:
            video = r.get("video_id")
            if r["duplicate_of_our_video"] and our_counts.get(video):
                why["duplicate_of_our_video"] += 1
                continue
            if r["duplicate_of_our_cue"] in selected:
                why["duplicate_of_our_cue"] += 1
                continue
            if any(o["sample_id"] in selected and (o.get("cover", 0.) >= max_cover or o.get("iou", 0.) >= .5)
                   for o in r["our_cue_overlaps"]):
                why["near_duplicate_of_our_cue"] += 1
                continue
            if video in by_video:
                partition, source = by_video[video], "our-video"
            elif r.get("our_channel_id") in by_channel:
                partition, source = by_channel[r["our_channel_id"]], "our-channel"
            else:
                why["our_partition_unknown"] += 1
                continue
        if partition not in PARTITIONS:
            raise ValueError(f"{r.get('clip_id')}: partition {partition!r} not in {PARTITIONS}")
        kept.append(dict(r, partition=partition, partition_source=source))
        why["kept"] += 1
    return kept, dict(why)


__all__ = ["REQUIRED", "PARTITIONS", "MAX_COVER", "our_selection", "select_training_records"]
