"""UWB keypoint shards: translations joined, leaks/short/no-hand clips dropped, shards round-trip, re-runs skip."""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_uwb_keypoint_shards as shards  # noqa: E402
from test_data_keypoints2d import holistic_frame  # noqa: E402

P = "uwb"


def write(tmp_path):
    root = tmp_path / "store"; (root / P).mkdir(parents=True)
    clips = {"vidA.000010-000040": 31, "vidA.000050-000055": 6, "vidB.000000-000030": 31, "vidC.000000-000030": 31,
             "vidD.000000-000030": 31}
    with zipfile.ZipFile(root / P / "raw_keypoints_1.zip", "w") as z:
        for clip, n in clips.items():
            frames = [holistic_frame(shift=k) for k in range(n)]
            if clip.startswith("vidD"):
                frames = [dict(f, left_hand_landmarks=None) for f in frames]
            z.writestr(f"{clip}.json", json.dumps(dict(keypoints=frames)))
    train = {"vidA": {"clip_order": ["vidA.000010-000040", "vidA.000050-000055"],
                      "vidA.000010-000040": {"translation": "Hello, my name is Ana."},
                      "vidA.000050-000055": {"translation": "Too short to keep."}},
             "vidB": {"clip_order": ["vidB.000000-000030"], "vidB.000000-000030": {"translation": "the evaluation sentence itself"}},
             "vidD": {"clip_order": ["vidD.000000-000030"], "vidD.000000-000030": {"translation": "no hands are visible here"}}}
    dev = {"vidC": {"clip_order": ["vidC.000000-000030"], "vidC.000000-000030": {"translation": "Welcome back  to the channel"}}}
    (root / P / "YT.translations.train.json").write_text(json.dumps(train))
    (root / P / "YT.translations.dev.json").write_text(json.dumps(dev))
    return root


def test_shards_keep_good_clips_with_text_and_partition(tmp_path):
    root = write(tmp_path)
    from signtranslator.data.text_motion_dataset import DirectoryStore
    texts = shards.translations(DirectoryStore(root), P)
    assert texts["vidC.000000-000030"] == ("Welcome back to the channel", "monitor")
    args = shards.argparse.Namespace(store=f"dir:{root}", prefix=P, out_prefix="out", shard_size=2, source_fps=30.,
                                     min_frames=12, max_seconds=20., min_words=2, min_hands=.5, min_shoulders=.8,
                                     limit=0)
    part = shards.convert_zip((args, "raw_keypoints_1.zip", texts, ["The evaluation sentence itself."]))
    assert part["counts"] == {"kept": 2, "too_short": 1, "evaluation_leak": 1, "hands_not_visible": 1}
    assert [s["clips"] for s in part["shards"]] == [2]
    data = np.load(root / "out/raw_keypoints_1/shard-00000.npz")
    meta = json.loads((root / "out/raw_keypoints_1/shard-00000.json").read_text())
    assert [m["partition"] for m in meta] == ["train", "monitor"] and meta[0]["text"] == "Hello, my name is Ana."
    assert list(data["offsets"]) == [0, 31, 62] and data["points"].shape == (62, 48, 2)
    np.testing.assert_allclose(data["points"][15, 0], [.5, 0.], atol=1e-3)    # the clip's median frame: unit shoulders
    (root / "out/raw_keypoints_1/shard-00000.npz").unlink()
    again = shards.convert_zip((args, "raw_keypoints_1.zip", texts, []))     # part.json present: not redone
    assert again == part
