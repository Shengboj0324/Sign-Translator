"""ASLLRP gloss manifest: signer parsing, signer-disjoint partitions, gloss text joined to extracted states."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_asllrp_gloss_manifest as builder  # noqa: E402


def test_signer_from_source_collection():
    assert builder.signer("Cory_2013-6-25_sc0106") == "Cory"
    assert builder.signer("4-Ben-Deaf-Speech") == "Ben"
    assert builder.signer("RIT_P02-s01") == builder.signer("RIT-P02-s02") == "RIT-P02"
    assert builder.signer("RIT-P03_s01") == "RIT-P03"
    assert builder.signer("2488.mp4") is None


def test_manifest_joins_glosses_states_and_partitions(tmp_path):
    def utt(video, source, glosses):
        return dict(collection="asllrp-continuous", source=source, utterance=1, utterance_video=video,
                    signs=[dict(gloss=g, sign_type="Lexical Signs", frames=[i, i + 5]) for i, g in enumerate(glosses)])
    items = [utt("1.mp4", "Cory_2013-6-25_sc0106", ["IX-1p", "LIKE", "COFFEE"]), utt("2.mp4", "Rachel_2012-01-27_sc50", ["BOOK"]),
             utt("3.mp4", "Jonathan_2012-11-27_sc90", ["WHO"]), utt("4.mp4", "4-Ben-Deaf-Speech", ["DEAF"]),
             utt("5.mp4", "2488.mp4", ["X"]), utt("6.mp4", "Cory_2013-6-27_sc107", ["NO-STATE"])]
    (tmp_path / "u.json").write_text(json.dumps(dict(items=items)))
    results = [dict(sample_id=s, status="converted", state_key=f"st/{s}.npz", state_sha256="c" * 64,
                    available_channels=["body"], seconds=2.) for s in ("1", "2", "3", "4", "5")]
    (tmp_path / "rep/x/reports").mkdir(parents=True)
    (tmp_path / "rep/x/reports/r.json").write_text(json.dumps(dict(results=results)))
    assert builder.main(["--store", f"dir:{tmp_path}", "--utterances-key", "u.json", "--reports-prefixes", "rep/x/reports/",
                         "--manifest-key", "out/m.json"]) == 0
    m = json.loads((tmp_path / "out/m.json").read_text())
    records = {r["sign"]: r for r in m["records"]}
    assert records["asllrp-continuous:1"]["text"] == "IX-1p LIKE COFFEE" and records["asllrp-continuous:1"]["partition"] == "train"
    assert records["asllrp-continuous:3"]["partition"] == "val" and records["asllrp-continuous:4"]["partition"] == "test"
    assert m["split"]["counts"] == {"train": 2, "val": 1, "test": 1}
    assert m["skipped"] == {"unknown signer": 1, "no state": 1}
