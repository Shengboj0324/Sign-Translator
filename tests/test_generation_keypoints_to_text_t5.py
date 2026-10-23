"""2D-keypoint back-translator: the shared encoder/decoder learns from (features, valid) batches."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5, KeypointsToT5Config  # noqa: E402
from tests.test_generation_motion_to_text_t5 import CharTokenizer  # noqa: E402


def tiny():
    t5 = transformers.T5ForConditionalGeneration(transformers.T5Config(
        vocab_size=260, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    config = KeypointsToT5Config(t5_name="tiny", d_model=32, heads=4, encoder_layers=1, feedforward=64, dropout=0.,
                                 max_tokens=16)
    return KeypointsToT5(config, t5, CharTokenizer())


def test_learns_to_say_what_each_clip_signs_and_ignores_padding():
    torch.manual_seed(0)
    lengths = [20, 24, 28]
    features = torch.zeros(3, 28, 144)
    valid = torch.zeros(3, 28, dtype=torch.bool)
    for i, n in enumerate(lengths):
        features[i, :n] = torch.randn(1, 144) + torch.linspace(0, 1, n)[:, None]
        valid[i, :n] = True
    texts = ["dog", "cat", "sun"]
    model = tiny()
    groups = model.param_groups(motion_lr=3e-3, decoder_lr=3e-3, weight_decay=0.)
    optimizer = torch.optim.Adam([{k: v for k, v in g.items() if k != "base_lr"} for g in groups])
    for _ in range(300):
        loss, _ = model.loss((features, valid), texts)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    assert model.generate_text((features, valid), beam=1) == texts
    junk = features.clone(); junk[~valid] = 99.                 # padding must not change the output
    assert model.generate_text((junk, valid), beam=1) == texts
    assert KeypointsToT5Config.from_dict(json.loads(json.dumps(model.config.to_dict()))) == model.config


def test_the_scorer_reads_canonical_states_through_the_keypoint_path():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from score_back_translation import translate
    from tests.test_generation_canonical_skeleton import rest_state
    out = translate(tiny().eval(), "keypoints", [rest_state(30), rest_state(12)], beam=1)
    assert len(out) == 2 and all(isinstance(t, str) for t in out)


def test_motion_alone_moves_to_another_device():
    """translate() hands (None, None, motion) to to_device: the missing text side must not break a CUDA move."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_text_to_motion import to_device
    from signtranslator.data.multichannel import collate_multichannel
    from tests.test_generation_canonical_skeleton import rest_state
    motion = collate_multichannel([rest_state(5)])
    _, _, moved = to_device((None, None, motion, None), torch.device("meta"))
    assert moved.timestamps.device.type == "meta"
