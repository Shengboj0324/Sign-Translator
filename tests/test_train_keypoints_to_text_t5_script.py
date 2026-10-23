"""2D-keypoint back-translator trainer: trains from shard indexes, selects on val, resumes, warm-starts strictly."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_keypoints_to_text_t5 as trainer  # noqa: E402
from build_uwb_keypoint_shards import _write_shard  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5  # noqa: E402
from tests.test_generation_keypoints_to_text_t5 import tiny  # noqa: E402
from tests.test_generation_motion_to_text_t5 import CharTokenizer  # noqa: E402


def shard_index(store, name, partitions, fps):
    rng = np.random.default_rng(len(name))
    clips = []
    for k, (partition, text) in enumerate(partitions):
        n = 20 + 3 * k
        clips.append(dict(clip_id=f"{name}-{k}", video_id=name, text=text, partition=partition, frames=n,
                          points=rng.normal(size=(n, 48, 2)).astype(np.float32) + k, valid=np.ones((n, 48), bool)))
    shard = _write_shard(store, f"{name}/shard-00000.npz", clips)
    index = dict(source_fps=fps, parts=[dict(shards=[shard])])
    store.put(f"{name}/index.json", json.dumps(index).encode())
    return f"{name}/index.json"


def build_tiny(config, pretrained=True):
    model = tiny()
    return KeypointsToT5(config, model.t5, CharTokenizer())


def args_for(root, run, *extra):
    return ["--store", f"dir:{root}", "--train-index", "uwb/index.json", "--val-index", "h2s/index.json",
            "--monitor-index", "uwb/index.json", "--run", run, "--steps", "40", "--batch-size", "2", "--val-every", "20",
            "--val-clips", "4", "--checkpoint-every", "20", "--workers", "0", "--warmup", "5", "--d-model", "32",
            "--heads", "4", "--encoder-layers", "1", "--feedforward", "64", "--max-tokens", "16", "--t5-name", "tiny",
            *extra]


def test_trains_selects_resumes_and_warm_starts(tmp_path):
    store = DirectoryStore(tmp_path)
    shard_index(store, "uwb", [("train", "dog"), ("train", "cat"), ("monitor", "sun")], 30.)
    shard_index(store, "h2s", [("val", "dog"), ("val", "cat"), ("train", "sun"), ("train", "dog")], 24.)
    assert trainer.main(args_for(tmp_path, "pre"), build_model=build_tiny) == 0
    best = json.loads(store.get("training-runs/pre/best.json"))
    assert best["step"] in (20, 40) and "monitor/bleu" in best["validation"]
    assert trainer.main(args_for(tmp_path, "pre", "--steps", "60"), build_model=build_tiny) == 0   # resumes at 40
    log = json.loads(store.get("training-runs/pre/log/step-00000060.json"))
    assert log[0]["step"] == 41
    key = best["checkpoint_key"]
    import hashlib
    sha = hashlib.sha256(store.get(key)).hexdigest()
    ft = args_for(tmp_path, "ft", "--init-key", key, "--init-sha256", sha)
    ft[ft.index("uwb/index.json")] = "h2s/index.json"
    assert trainer.main(ft, build_model=build_tiny) == 0
    small = args_for(tmp_path, "small", "--batch-size", "8")
    with pytest.raises(SystemExit):      # fewer clips than one batch would never yield a step
        trainer.main(small, build_model=build_tiny)
    with pytest.raises(SystemExit):      # a different config cannot warm-start
        trainer.main(args_for(tmp_path, "bad", "--init-key", key, "--init-sha256", sha, "--d-model", "48",
                              "--heads", "4"), build_model=build_tiny)
