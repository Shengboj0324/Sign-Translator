"""Back-translator training script: vocabulary from training text only, checkpoints, best.json by BLEU, resume."""
import importlib.util
import json
from pathlib import Path

from tests.test_train_text_to_motion_script import store_with_manifest

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("train_m2t", ROOT / "scripts/train_motion_to_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trains_selects_by_bleu_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    common = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
              "--run", "bt", "--batch-size", "2", "--d-model", "32", "--heads", "4", "--encoder-layers", "1",
              "--decoder-layers", "1", "--feedforward", "64", "--warmup", "2", "--val-every", "2", "--checkpoint-every", "2",
              "--workers", "0", "--min-count", "1", "--trained-channels", "body,head_rotation,face"]
    assert script.main(common + ["--steps", "4"]) == 0
    root = tmp_path / "store/training-runs/bt"
    run = json.loads((root / "run.json").read_text())
    assert run["kind"] == "motion_to_text" and run["train_records"] == 3 and run["val_records"] == 1
    assert "rain" not in run["vocabulary"]["words"]              # validation text never enters the vocabulary
    assert set(run["vocabulary"]["words"]) == {"a", "dog", "runs", "cats", "sleep", "birds", "fly"}
    best = json.loads((root / "best.json").read_text())
    assert "val_bleu" in best and best["checkpoint_key"].startswith("training-runs/bt/checkpoints/")
    assert script.main(common + ["--steps", "6"]) == 0          # resumes from step 4 with the stored vocabulary
    log = json.loads((root / "log/step-00000006.json").read_text())
    assert [e["step"] for e in log] == [5, 6] and "val/bleu" in log[-1]


def test_scorer_reads_generated_real_and_blank(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    script = load_script()
    store = tmp_path / "store"
    assert script.main(["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                        "--run", "bt", "--batch-size", "2", "--d-model", "32", "--heads", "4", "--encoder-layers", "1",
                        "--decoder-layers", "1", "--feedforward", "64", "--warmup", "2", "--val-every", "2",
                        "--checkpoint-every", "2", "--workers", "0", "--min-count", "1",
                        "--trained-channels", "body,head_rotation,face", "--steps", "2"]) == 0
    (store / "gen/g1").mkdir(parents=True)
    for i in range(4):                                      # "generated" = the real states: both must score alike
        shutil.copy(store / f"states/s{i}.npz", store / f"gen/g1/s{i}.npz")
    spec = importlib.util.spec_from_file_location("score_bt", ROOT / "scripts/score_back_translation.py")
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    best = json.loads((store / "training-runs/bt/best.json").read_text())["checkpoint_key"]
    assert scorer.main(["--store", f"dir:{store}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
                        "--translator-key", best, "--generated", "gen", "--beam", "2", "--output-key", "out.json"]) == 0
    result = json.loads((store / "out.json").read_text())["results"]["gen"]["g1"]
    assert set(result) == {"generated", "real", "blank", "shuffled"} and "bleu" in result["shuffled"]
    assert result["generated"]["bleu"] == result["real"]["bleu"]
