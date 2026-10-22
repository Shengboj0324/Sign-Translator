import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_gloss_planner  # noqa: E402
import translate_english  # noqa: E402
from signtranslator.planning.plan_text import plan_from_text  # noqa: E402


def test_plan_only_round_trip_from_a_trained_planner(tmp_path, capsys):
    root = tmp_path / "s"; (root / "p").mkdir(parents=True)
    rows = [dict(english="he gives her the book", surface=["IX-3p:i", "i:GIVE:j", "BOOK"], markers=[], partition=part)
            for part in ("train", "train", "val")]
    payload = json.dumps(dict(pairs=rows)).encode(); (root / "p/pairs.json").write_bytes(payload)
    assert train_gloss_planner.main(["--store", f"dir:{root}", "--pairs-key", "p/pairs.json", "--pairs-sha256",
                                     hashlib.sha256(payload).hexdigest(), "--run", "pl", "--d-model", "32", "--heads", "2",
                                     "--encoder-layers", "1", "--decoder-layers", "1", "--feedforward", "64",
                                     "--min-word-count", "1", "--min-gloss-count", "1", "--val-every", "40",
                                     "--steps", "120", "--lr", "3e-3", "--warmup", "5", "--dropout", "0",
                                     "--label-smoothing", "0", "--device", "cpu", "--batch-size", "2"]) == 0
    capsys.readouterr()
    assert translate_english.main(["--store", f"dir:{root}", "--planner-run", "pl", "--text", "he gives her the book",
                                   "--plan-only", "--beam", "2"]) == 0
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["plan_line"] == "IX-3p>1 1>GIVE>2 BOOK"
    assert plan_from_text(row["plan_line"])[0] == ["IX-3p", "GIVE", "BOOK"]
    assert abs(row["seconds"] - 3 * translate_english.SECONDS_PER_SIGN) < 1e-6
