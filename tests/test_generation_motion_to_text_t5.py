"""Back-translator v2: motion encoder under a (tiny, randomly initialised) T5 decoder learns, decodes, round-trips."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.generation.motion_to_text_t5 import MotionToT5, MotionToT5Config  # noqa: E402
from tests.test_generation_text_to_motion import config as motion_config, state  # noqa: E402


class CharTokenizer:
    """A byte tokenizer with T5's interface (pad 0, eos 1, bytes from 3)."""
    pad_token_id, eos_token_id = 0, 1

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        rows = [[3 + b for b in t.encode()[:max_length - 1]] + [1] for t in texts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows])}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [bytes(int(i) - 3 for i in row if int(i) >= 3).decode(errors="replace") for row in ids]


def tiny(states):
    t5 = transformers.T5ForConditionalGeneration(transformers.T5Config(
        vocab_size=260, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    config = MotionToT5Config(motion_config(states), t5_name="tiny", d_model=32, heads=4, encoder_layers=1,
                              feedforward=64, dropout=0., max_tokens=16)
    return MotionToT5(config, t5, CharTokenizer())


def test_learns_to_say_what_each_clip_signs():
    torch.manual_seed(0)
    states = [state(i, 20 + 4 * i) for i in range(3)]
    texts = ["dog", "cat", "sun"]
    model = tiny(states)
    batch = collate_multichannel(states)
    groups = model.param_groups(motion_lr=3e-3, decoder_lr=3e-3, weight_decay=0.)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    optimizer = torch.optim.Adam([{k: v for k, v in g.items() if k != "base_lr"} for g in groups])
    for _ in range(300):
        loss, terms = model.loss(batch, texts)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    assert model.generate_text(batch, beam=1) == texts
    assert MotionToT5Config.from_dict(json.loads(json.dumps(model.config.to_dict()))) == model.config


def test_training_script_trains_selects_and_the_scorer_loads_it(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from tests.test_train_text_to_motion_script import store_with_manifest
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("train_t5", root / "scripts/train_motion_to_text_t5.py")
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    made = []

    def build_model(config, pretrained=True):
        made.append(pretrained)
        return tiny([state(0, 20)]).__class__(config, tiny([state(0, 20)]).t5, CharTokenizer())
    args = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--run", "bt2", "--batch-size", "2", "--d-model", "32", "--heads", "4", "--encoder-layers", "1",
            "--feedforward", "64", "--warmup", "1", "--val-every", "2", "--checkpoint-every", "2", "--workers", "0",
            "--trained-channels", "body,head_rotation,face", "--steps", "2", "--t5-name", "tiny"]
    assert script.main(args, build_model=build_model) == 0
    assert script.main(args[:-4] + ["--steps", "4", "--t5-name", "tiny"], build_model=build_model) == 0   # resumes
    assert made == [True, False]
    best = json.loads((tmp_path / "store/training-runs/bt2/best.json").read_text())
    assert "val/bleu" in best["validation"]
