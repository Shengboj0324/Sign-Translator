import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_asllvd_sign_manifest import clip_key, main  # noqa: E402

HEADER = ("main entry gloss label,entry/variant gloss label,start frame of the sign (relative to full videos),"
          "end frame of the sign (relative to full videos),full video file,sign type,Dominant start handshape\n")


def test_clip_names_parse_to_the_annotation_key():
    assert clip_key("Brady-session-ASL_2011_06_08_Brady-scene-10-10020-10040-camera1") == \
        ("Brady", ("ASL_2011_06_08_Brady", 10, 10020, 10040, 1))
    assert clip_key("not-a-clip") is None


def test_manifest_joins_clips_to_rows_with_signer_partitions(tmp_path):
    root = tmp_path / "store"
    (root / "a").mkdir(parents=True); (root / "r").mkdir()
    (root / "a/signs.csv").write_text(HEADER +
        "CHEAT,(1)CHEAT,1365,1400,ASL_2008_01_11_scene71-camera1.mov,Lexical Signs,1\n"
        "BOOK,BOOK,20,40,ASL_2011_06_08_Brady_scene10-camera1.mov,Lexical Signs,B\n")
    def result(sample):
        return dict(sample_id=sample, status="converted", state_key=f"s/{sample}.npz", state_sha256="0" * 64,
                    available_channels=["body"], seconds=1.0)
    (root / "r/rep.json").write_text(json.dumps({"results": [
        result("Naomi-session-ASL_2008_01_11-scene-71-1365-1400-camera1"),
        result("Brady-session-ASL_2011_06_08_Brady-scene-10-20-40-camera1"),
        result("Brady-session-ASL_2011_06_08_Brady-scene-10-99-120-camera1"),
        dict(sample_id="x", status="refused")]}))
    assert main(["--store", f"dir:{root}", "--annotations-key", "a/signs.csv", "--reports-prefix", "r/",
                 "--manifest-key", "m/manifest.json"]) == 0
    out = json.loads((root / "m/manifest.json").read_text())
    by = {r["sign"]: r for r in out["records"]}
    assert by["Naomi-session-ASL_2008_01_11-scene-71-1365-1400-camera1"]["partition"] == "val"
    assert by["Brady-session-ASL_2011_06_08_Brady-scene-10-20-40-camera1"]["gloss"]["label"] == "BOOK"
    assert out["skipped"] == {"no annotation row": 1, "refused": 1}
