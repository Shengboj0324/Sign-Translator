"""YouTube ASL manifest: caption filters, channel-disjoint monitor split, provenance kept."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_youtube_asl_manifest import build, clean_text, keep_segment  # noqa: E402

ARGS = types.SimpleNamespace(min_words=2, min_seconds=1., max_seconds=20., max_words_per_second=5.,
                             min_hand_coverage=.5, monitor_share=0., allow_auto=False)


def seg(text, start=0., end=3., cov=.9, k=0):
    return dict(cue=k, sample_id=f"vid-{k:04d}", start=start, end=end, text=text, hand_coverage=cov,
                state_key=f"s/{k}.npz", state_sha256="a" * 64, seconds=end - start, available_channels=["body"])


def test_filters():
    assert keep_segment(seg("[demonstrates name sign]"), args=ARGS) == "note"
    assert keep_segment(seg("♪ music ♪"), args=ARGS) == "note"
    assert keep_segment(seg("Harry"), args=ARGS) == "few_words"
    assert keep_segment(seg("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen", end=2.), args=ARGS) == "too_fast"
    assert keep_segment(seg("my name is Ana", cov=.2), args=ARGS) == "hands_not_visible"
    assert keep_segment(seg("my name is Ana", end=30.), args=ARGS) == "duration"
    assert keep_segment(seg("my name is Ana"), args=ARGS) is None
    assert clean_text("NARRATOR: Hello there [laughs] friend") == "Hello there friend"


def test_build_keeps_manual_captions_with_provenance_and_drops_repeats():
    video = dict(video_id="abcdefghijk", status="converted", caption_kind="manual",
                 meta=dict(channel_id="UC1", upload_date="20200101", license="Creative Commons Attribution license (reuse allowed)"),
                 segments=[seg("Welcome to my channel", k=0), seg("my name is Ana", k=1), seg("Welcome to my channel", k=2)])
    auto = dict(video, video_id="bbbbbbbbbbb", caption_kind="auto")
    gone = dict(video_id="ccccccccccc", status="removed")
    records, why = build([video, auto, gone], args=ARGS, corpus_id="yt")
    assert [r["text"] for r in records] == ["Welcome to my channel", "my name is Ana"]
    assert why["repeated_text"] == 1 and why["video_auto_captions"] == 3 and why["video_removed"] == 1
    r = records[0]
    assert r["partition"] == "train" and r["source"]["youtube_license"].startswith("Creative Commons")
    monitored, _ = build([video], args=types.SimpleNamespace(**{**vars(ARGS), "monitor_share": 1.}), corpus_id="yt")
    assert {x["partition"] for x in monitored} == {"monitor"}
