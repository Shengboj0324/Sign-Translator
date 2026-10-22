import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_gloss_planner  # noqa: E402


def _pairs(root: Path) -> str:
    def pair(english, surface, partition, markers=()):
        return dict(english=english, surface=surface, markers=list(markers), partition=partition)
    rows = [pair("he gives her the book", ["IX-3p:i", "i:GIVE:j", "BOOK"], "train"),
            pair("did you finish", ["IX-2p", "FINISH"], "train",
                 [dict(tier="y/n question", label="q", token_start=0, token_end=2)]),
            pair("i do not like it", ["IX-1p", "NOT", "LIKE"], "val",
                 [dict(tier="negative", label="neg", token_start=1, token_end=3)])]
    payload = json.dumps(dict(pairs=rows)).encode()
    (root / "p").mkdir(parents=True)
    (root / "p/pairs.json").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_trains_validates_checkpoints_and_resumes(tmp_path):
    root = tmp_path / "store"; sha = _pairs(root)
    common = ["--store", f"dir:{root}", "--pairs-key", "p/pairs.json", "--pairs-sha256", sha, "--run", "r",
              "--d-model", "32", "--heads", "2", "--encoder-layers", "1", "--decoder-layers", "1", "--feedforward", "64",
              "--min-word-count", "1", "--min-gloss-count", "1", "--val-every", "5", "--device", "cpu", "--batch-size", "2"]
    assert train_gloss_planner.main(common + ["--steps", "10"]) == 0
    best = json.loads((root / "training-runs/r/best.json").read_text())
    assert {"gloss_wer", "gloss_bleu", "marker_f1", "locus_accuracy"} <= set(best["validation"])
    assert train_gloss_planner.main(common + ["--steps", "15"]) == 0          # resumes from step 10
    assert sorted(p.name for p in (root / "training-runs/r/checkpoints").iterdir())[-1] == "step-00000015.pt"


def test_refuses_a_pairs_file_with_another_hash(tmp_path):
    root = tmp_path / "store"; _pairs(root)
    with pytest.raises(SystemExit, match="does not match"):
        train_gloss_planner.main(["--store", f"dir:{root}", "--pairs-key", "p/pairs.json", "--pairs-sha256", "0" * 64,
                                  "--run", "r", "--device", "cpu"])
