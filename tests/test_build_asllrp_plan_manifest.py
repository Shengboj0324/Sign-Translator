import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_asllrp_plan_manifest import main  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402


def test_records_get_their_plan_line_and_keep_partitions(tmp_path):
    root = tmp_path / "s"; root.mkdir()
    rec = dict(sign="asllrp-continuous:u1", text="IX BOOK", partition="train", performer_sha256="a" * 64,
               source=dict(collection="asllrp-continuous", source="Cory_2013", utterance=1, utterance_video="1.mp4"),
               continuous=dict(status="converted", state_key="k", state_sha256="b" * 64, available_channels=["body"],
                               seconds=1.))
    gm = canonical_json(build_corpus_manifest(corpus_id="g", inputs={"x": "c" * 64}, records=[rec], split={}))
    (root / "g.json").write_bytes(gm)
    pairs = json.dumps(dict(pairs=[dict(source="Cory_2013", utterance=1, partition="train", english="He has a book?",
                                        surface=["IX-3p:i", "BOOK"],
                                        markers=[dict(tier="y/n question", label="q", token_start=0, token_end=2)])])).encode()
    (root / "p.json").write_bytes(pairs)
    assert main(["--store", f"dir:{root}", "--gloss-manifest-key", "g.json", "--gloss-manifest-sha256",
                 hashlib.sha256(gm).hexdigest(), "--pairs-key", "p.json", "--pairs-sha256", hashlib.sha256(pairs).hexdigest(),
                 "--manifest-key", "out.json"]) == 0
    out = json.loads((root / "out.json").read_text())
    r = out["records"][0]
    assert r["text"] == "[yn IX-3p>1 BOOK ]" and r["english"] == "He has a book?" and r["partition"] == "train"
