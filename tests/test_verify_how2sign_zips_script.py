"""How2Sign zip verification: CRC-checked members, names against the CSV, manifests."""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_how2sign_zips import expected_clips, verify_archive  # noqa: E402

CSV = ("VIDEO_ID\tVIDEO_NAME\tSENTENCE_ID\tSENTENCE_NAME\tSTART_REALIGNED\tEND_REALIGNED\tSENTENCE\n"
       "a\ta-1-rgb_front\ta_0\ta_0-1-rgb_front\t1.0\t2.0\tHe said \"hi\".\n"
       "a\ta-1-rgb_front\ta_1\ta_1-1-rgb_front\t2.0\t3.0\tSecond.\n"
       "b\tb-1-rgb_front\tb_0\tb_0-1-rgb_front\t0.5\t1.5\tThird.\n")


def archive(entries, corrupt=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    data = bytearray(buffer.getvalue())
    if corrupt:
        at = data.find(entries[corrupt])
        data[at] ^= 0xFF                                  # flip one stored byte; CRC no longer matches
    return io.BytesIO(bytes(data))


def test_expected_clip_names_follow_the_view():
    assert expected_clips(CSV, "front") == {"a_0-1-rgb_front.mp4", "a_1-1-rgb_front.mp4", "b_0-1-rgb_front.mp4"}
    assert expected_clips(CSV, "side") == {"a_0-1-rgb_side.mp4", "a_1-1-rgb_side.mp4", "b_0-1-rgb_side.mp4"}


def test_members_are_hashed_and_names_reconciled():
    entries = {"raw_videos/a_0-1-rgb_front.mp4": b"clip-a0" * 100, "raw_videos/b_0-1-rgb_front.mp4": b"clip-b0" * 90,
               "raw_videos/z_9-1-rgb_front.mp4": b"stray"}
    report = verify_archive(archive(entries), expected_clips(CSV, "front"))
    assert report["passed"] and report["clips"] == 3 and not report["corrupt"]
    assert report["missing"] == ["a_1-1-rgb_front.mp4"] and report["unexpected"] == ["z_9-1-rgb_front.mp4"]
    by_name = {m["name"]: m for m in report["members"]}
    assert by_name["raw_videos/a_0-1-rgb_front.mp4"]["sha256"] == hashlib.sha256(b"clip-a0" * 100).hexdigest()
    assert by_name["raw_videos/a_0-1-rgb_front.mp4"]["size"] == 700


def test_a_corrupted_member_fails_verification():
    entries = {"a_0-1-rgb_front.mp4": b"A" * 4096, "b_0-1-rgb_front.mp4": b"B" * 4096}
    report = verify_archive(archive(entries, corrupt="b_0-1-rgb_front.mp4"), expected_clips(CSV, "front"))
    assert not report["passed"]
    assert [c["name"] for c in report["corrupt"]] == ["b_0-1-rgb_front.mp4"]
