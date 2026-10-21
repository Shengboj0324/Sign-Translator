"""Text–motion retrieval evaluator: R-precision maths, padding invariance, duplicate texts, learning."""
import pytest
import torch
import torch.nn.functional as F

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.text_motion_retrieval import (TextMotionRetrieval, TextMotionRetrievalConfig,
                                                             r_precision, retrieval_loss)
from signtranslator.generation.text_to_motion import encode_texts
from tests.test_generation_text_to_motion import config as motion_config, state


def test_r_precision_is_exact_on_known_cases():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(64, 16), dim=-1)
    perfect = r_precision(z, z, group=32)
    assert perfect["r_at_1"] == perfect["r_at_3"] == 1. and perfect["clips"] == 64 and perfect["matching_distance"] < 1e-6
    assert perfect["chance_at_3"] == 3 / 32
    shuffled = r_precision(z, z[torch.randperm(64)], group=32)
    assert shuffled["r_at_3"] < .4                               # unpaired: near chance (3/32)
    # a fixed seed fixes the groups: the same clips always meet the same distractors
    assert r_precision(z, z[torch.randperm(64, generator=torch.Generator().manual_seed(1))], seed=5) == \
        r_precision(z, z[torch.randperm(64, generator=torch.Generator().manual_seed(1))], seed=5)
    with pytest.raises(ValueError):
        r_precision(z[:10], z[:10], group=32)


def test_r_precision_counts_ranks_within_each_group():
    # group of 4: row i's own text is beaten by exactly i other texts
    m = torch.eye(4)
    t = torch.tensor([[1., 0, 0, 0], [.6, .8, 0, 0], [.6, 0, .8, 0], [.6, 0, 0, .8]])
    sim_own = (m * t).sum(1)
    out = r_precision(m, t, group=4, ks=(1, 2), seed=0)
    ranks = [int(((m[i] @ t.t()) > sim_own[i]).sum()) for i in range(4)]
    assert out["r_at_1"] == sum(r < 1 for r in ranks) / 4 and out["r_at_2"] == sum(r < 2 for r in ranks) / 4


def _model(states, **kw):
    return TextMotionRetrieval(TextMotionRetrievalConfig(motion_config(states), d_model=32, heads=4, motion_layers=1,
                                                         text_layers=1, feedforward=64, dropout=0., embedding=16, **kw))


def test_motion_embedding_ignores_padding():
    states = [state(0, 20), state(1, 32)]
    model = _model(states).eval()
    alone = model.encode_motion(collate_multichannel([states[0]]))
    padded = model.encode_motion(collate_multichannel(states))[0:1]      # clip 0 padded to 32 frames
    torch.testing.assert_close(alone, padded, atol=1e-5, rtol=1e-5)
    assert torch.allclose(alone.norm(dim=-1), torch.ones(1))


def test_duplicate_texts_are_not_negatives_and_the_model_learns_to_retrieve():
    torch.manual_seed(0)
    states = [state(i, 20 + 4 * i) for i in range(4)]
    texts = ["a dog runs", "cats sleep", "a dog runs", "birds fly"]
    model = _model(states)
    tokens, mask = encode_texts(texts)
    batch = collate_multichannel(states)
    loss, terms = retrieval_loss(model, tokens, mask, batch)
    assert torch.isfinite(loss)
    unique = ["a dog runs", "cats sleep", "rain falls", "birds fly"]
    tokens, mask = encode_texts(unique)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss, terms = retrieval_loss(model, tokens, mask, batch)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    model.eval()
    with torch.no_grad():
        out = r_precision(model.encode_motion(batch), model.encode_text(tokens, mask), group=4, ks=(1,))
    assert out["r_at_1"] == 1. and float(terms["batch_top1"]) == 1.


def test_config_round_trips():
    states = [state(0, 20)]
    config = TextMotionRetrievalConfig(motion_config(states), d_model=32, heads=4)
    assert TextMotionRetrievalConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        TextMotionRetrievalConfig(motion_config(states), d_model=30, heads=4)


def test_training_script_trains_validates_selects_and_stops(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from tests.test_train_text_to_motion_script import store_with_manifest
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    digest = store_with_manifest(tmp_path / "store")
    spec = importlib.util.spec_from_file_location(
        "train_retrieval", Path(__file__).resolve().parents[1] / "scripts/train_text_motion_retrieval.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = ["--store", f"dir:{tmp_path / 'store'}", "--manifest-key", "manifest.json", "--manifest-sha256", digest,
            "--run", "ret", "--batch-size", "3", "--d-model", "32", "--heads", "4", "--motion-layers", "1",
            "--text-layers", "1", "--feedforward", "64", "--embedding", "16", "--warmup", "1", "--val-every", "2",
            "--checkpoint-every", "100", "--workers", "0", "--trained-channels", "body,head_rotation,face",
            "--patience", "2", "--steps", "40", "--lr", "0"]
    assert script.main(args) == 0                     # lr 0: validation never improves, so it stops after 2 + 1
    run = tmp_path / "store/training-runs/ret"
    best = json.loads((run / "best.json").read_text())
    assert best["step"] == 2 and "val_r_at_3" in best
    log = json.loads(sorted((run / "log").iterdir())[-1].read_text())
    assert [e["step"] for e in log if "val/r_at_1" in e] == [2, 4, 6]
