"""Round 17: plan tokenizer, plan-conditioned detail prior, plan prior and two-stage generation."""
import hashlib
import importlib.util
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.generation.motion_rvq import (
    RVQPrior, RVQPriorConfig, generate_rvq_motion, generate_rvq_motion_batch, rvq_prior_loss,
    rvq_sequence_log_likelihood,
)
from signtranslator.generation.plan_codes import (
    PlanPriorConfig, PlanVQ, PlanVQConfig, check_plan_pair, copy_text_encoder, encode_plan, generate_plan_codes_batch,
    generate_plan_detail_batch, load_plan_tokenizer, plan_checkpoint_bytes, plan_prior_loss, plan_steps,
    plan_tokenizer_loss, pool_windows, relative_latent_error, tokenizer_latents, upsample_windows, window_valid,
)
from signtranslator.generation.text_to_motion import LOCUS_SLOT_WIDTH, diffusion_targets, encode_texts
from tests.test_generation_motion_rvq import prior_setup, small
from tests.test_generation_signing_space import signer_state

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_config(tokenizer, **kw):
    return PlanVQConfig(tokenizer.config, **{**dict(window=2, hidden=16, code_dim=8, codebook=16, stages=2, blocks=1),
                                             **kw})


def planned(prior, plan, **kw):
    """The plan-conditioned version of ``prior``'s config over ``plan``'s layout."""
    return replace(prior.config, plan_window=plan.config.window, plan_codes=plan.config.codes_per_step,
                   plan_codebook=plan.config.codebook, plan_tokenizer=plan.config.sha256(), **kw)


def warm(prior, config):
    """A plan-conditioned prior warm-started from ``prior`` (the training script's rule)."""
    model = RVQPrior(config)
    assert script("train_motion_tokens").plan_warm_start(model, prior.state_dict()) == len(model.plan_parameter_names())
    return model


def wake(model, seed=0):
    """Give the zero-initialised plan modules random values (a trained plan-conditioned prior)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("plan_"):
                p.copy_(torch.randn(p.shape, generator=g) * .3)
    return model


# --------------------------------------------------------------------------------------
# windows


def test_windows_pool_and_upsample_with_ragged_lengths():
    S, W = 8, 3
    valid = torch.arange(S)[None] < torch.tensor([[7], [4], [1], [8]])
    inside, plan_valid = window_valid(valid, W)
    assert plan_steps(S, W) == 3 == inside.shape[1]
    assert plan_valid.tolist() == [[True, True, True], [True, True, False], [True, False, False], [True] * 3]
    assert inside[0, 2].tolist() == [True, False, False] and inside[1, 1].tolist() == [True, False, False]
    x = torch.randn(4, S, 5)
    pooled, pv = pool_windows(x, valid, W)
    assert torch.equal(pv, plan_valid)
    for b, length in enumerate((7, 4, 1, 8)):
        for i in range(3):
            steps = [t for t in range(i * W, min((i + 1) * W, length))]
            want = x[b, steps].mean(0) if steps else torch.zeros(5)
            torch.testing.assert_close(pooled[b, i], want)
    # learned pooling: a softmax over the window's valid steps only (invalid steps never leak in)
    scores = torch.randn(4, S)
    learned, _ = pool_windows(x, valid, W, scores)
    noisy = x.clone(); noisy[~valid] = 1e6
    torch.testing.assert_close(pool_windows(noisy, valid, W, scores)[0], learned)
    w = scores[0, 6:7].softmax(-1)
    torch.testing.assert_close(learned[0, 2], (x[0, 6:7] * w[:, None]).sum(0))
    assert not bool(learned[2, 1:].abs().any())                           # windows without a valid step pool to 0
    # upsampling: step t reads plan step t // W, cut to the steps
    y = torch.arange(3.)[None, :, None].expand(2, 3, 1)
    up = upsample_windows(y, W, 7)
    assert up.shape == (2, 7, 1) and up[0, :, 0].tolist() == [0, 0, 0, 1, 1, 1, 2]
    with pytest.raises(ValueError, match="cannot cover"):
        upsample_windows(y, W, 10)


@pytest.mark.parametrize("blocks", [1, 2])
@pytest.mark.parametrize("pooling", ["mean", "learned"])
def test_plan_codes_of_a_clip_do_not_depend_on_the_batch_or_its_padding(blocks, pooling):
    """Review of c37036b, fix 1: with two residual blocks per stack the padding leaked into valid positions (masking
    only the stack's input); every block is now re-masked. The continuous plan latents, the codes and the
    reconstruction of a clip are the same alone and in a padded batch, whatever the padding holds."""
    torch.manual_seed(0)
    _, tokenizer, _ = prior_setup()
    plan = PlanVQ(plan_config(tokenizer, window=3, pooling=pooling, blocks=blocks))
    wake_all(plan)
    plan = plan.double().eval()                      # float64: rounding far below any leak (the leak was up to 1e5 here)
    D = tokenizer.config.code_dim
    latents = {p: torch.randn(2, 11, D, dtype=torch.float64) for p in plan.parts}
    valid = torch.arange(11)[None] < torch.tensor([[11], [5]])
    codes, plan_valid = plan.encode_latents(latents, valid)
    assert codes.shape == (2, 4, plan.config.codes_per_step) and plan_valid.sum(1).tolist() == [4, 2]
    alone = {p: v[1:, :5] for p, v in latents.items()}
    codes1, valid1 = plan.encode_latents(alone, valid[1:, :5])
    assert torch.equal(codes1[0], codes[1, :2]) and valid1.all()
    z, _ = plan.encode(latents, valid)
    z1, _ = plan.encode(alone, valid[1:, :5])
    garbage = {p: torch.where(valid[..., None], v, torch.full_like(v, 1e3)) for p, v in latents.items()}
    zg, _ = plan.encode(garbage, valid)
    rec = plan.reconstruct(latents, valid)
    rec1 = plan.reconstruct(alone, valid[1:, :5])
    for p in plan.parts:
        torch.testing.assert_close(z1[p][0], z[p][1, :2], rtol=1e-9, atol=1e-7)     # the continuous plan latents
        torch.testing.assert_close(zg[p][plan_valid], z[p][plan_valid], rtol=1e-9, atol=1e-7)
        assert rec[p].shape == (2, 11, D)
        torch.testing.assert_close(rec1[p][0], rec[p][1, :5], rtol=1e-9, atol=1e-7)  # valid steps: the same
    assert torch.equal(plan.encode_latents(garbage, valid)[0], codes)    # padded steps are never read
    if blocks == 2:                                  # the test sees the reviewed defect: masking only a stack's input
        plan._stack = lambda stack, x, mask: PlanVQ._conv(stack, x, mask)
        with torch.no_grad():
            leaked, batched = plan.encode(alone, valid[1:, :5])[0], plan.encode(latents, valid)[0]
        assert max(float((leaked[p][0] - batched[p][1, :2]).abs().max()) for p in plan.parts) > 1e-3


def wake_all(module, scale=.5, seed=0):
    """Random weights of a noticeable size everywhere (a zero-initialised position table hides leaks)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale / max(1, p.shape[-1]) ** .5 * 3)


def test_plan_codes_are_stage_major_and_decode_from_codes_equals_the_forward_path():
    torch.manual_seed(1)
    _, tokenizer, _ = prior_setup()
    plan = PlanVQ(plan_config(tokenizer, stages=3)).eval()
    P, D = len(plan.parts), tokenizer.config.code_dim
    latents = {p: torch.randn(2, 6, D) for p in plan.parts}
    valid = torch.ones(2, 6, dtype=torch.bool)
    out, z, quantized, codes, plan_valid = plan(latents, valid)
    assert codes.shape == (2, 3, 3 * P)
    per_part = plan.split_codes(codes)
    for k, p in enumerate(plan.parts):
        assert torch.equal(per_part[p][..., 0], codes[..., k])            # stage 1 of part k: j = k
        assert torch.equal(per_part[p][..., 2], codes[..., 2 * P + k])    # stage 3: j = 2P + k
    decoded = plan.decode_codes(codes, plan_valid, 6)
    for p in plan.parts:
        torch.testing.assert_close(decoded[p], out[p])                    # eval: straight-through == codes


def test_the_plan_tokenizer_learns_a_toy_signal_and_its_motion_loss_trains():
    torch.manual_seed(0)
    _, tokenizer, _ = prior_setup()
    plan = PlanVQ(plan_config(tokenizer, window=4, hidden=32, code_dim=8, codebook=32, stages=2, blocks=1,
                              motion_weight=0.))
    D, S = tokenizer.config.code_dim, 24
    t = torch.arange(S, dtype=torch.float32)
    # a slow, structured signal per part: what a plan can carry (8 clips, random phases and amplitudes)
    phase, amp = torch.rand(8, 1, 1) * 6.28, torch.rand(8, 1, D) + .5
    latents = {p: amp * torch.sin(t[None, :, None] * .2 * (k + 1) + phase) for k, p in enumerate(plan.parts)}
    valid = torch.ones(8, S, dtype=torch.bool)
    optimizer = torch.optim.Adam(plan.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        rec, z, quantized, codes, plan_valid = plan(latents, valid)
        loss = relative_latent_error(rec, latents, valid)
        commit = sum(((z[p] - quantized[p].detach()) ** 2).mean() for p in plan.parts)
        optimizer.zero_grad(); (loss + .25 * commit).backward(); optimizer.step()
        first = first if first is not None else float(loss.detach())
    plan.eval()
    final = float(relative_latent_error(plan.reconstruct(latents, valid), latents, valid))
    assert final < .3 * first and final < .5
    # on motion: the full loss (latent + decoded motion through the frozen decoder) decreases
    torch.manual_seed(0)
    states = [signer_state(s, frames=16, amplitude=.4) for s in range(2)]
    batch = collate_multichannel(states)
    plan = PlanVQ(plan_config(tokenizer))
    optimizer = torch.optim.Adam(plan.parameters(), lr=3e-3)
    history = []
    tokenizer.requires_grad_(False)                       # as the training script freezes it
    frozen = {k: v.clone() for k, v in tokenizer.state_dict().items()}
    for _ in range(40):
        loss, terms = plan_tokenizer_loss(plan, tokenizer, batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append(float(loss.detach()))
    assert history[-1] < .7 * history[0]
    assert {"latent_relative", "motion", "commitment", "motion_skeleton_velocity", "codes_used_body"} <= set(terms)
    assert all(torch.equal(v, frozen[k]) for k, v in tokenizer.state_dict().items())   # the tokenizer stays frozen


def test_plan_checkpoints_are_deterministic_sha_checked_and_pin_their_tokenizer():
    torch.manual_seed(0)
    states, tokenizer, _ = prior_setup()
    plan = PlanVQ(plan_config(tokenizer)).eval()
    kw = dict(tokenizer_key="tok.pt", tokenizer_sha256="a" * 64, manifest_sha256="m" * 64)
    payload = plan_checkpoint_bytes(plan, tokenizer.state_dict(), **kw)
    assert payload == plan_checkpoint_bytes(plan, tokenizer.state_dict(), **kw)       # same content, same bytes
    sha = hashlib.sha256(payload).hexdigest()
    loaded, loaded_tokenizer, saved = load_plan_tokenizer(payload, sha, tokenizer_sha256="a" * 64)
    assert saved["kind"] == "plan_vq_tokenizer" and loaded.config == plan.config
    batch = collate_multichannel(states)
    assert torch.equal(encode_plan(loaded, loaded_tokenizer, batch)[0], encode_plan(plan, tokenizer, batch)[0])
    with pytest.raises(ValueError, match="SHA-256"):
        load_plan_tokenizer(payload, "0" * 64)
    with pytest.raises(ValueError, match="reads tokenizer"):
        load_plan_tokenizer(payload, sha, tokenizer_sha256="b" * 64)
    assert PlanVQConfig.from_dict(json.loads(json.dumps(plan.config.to_dict()))).sha256() == plan.config.sha256()
    with pytest.raises(ValueError):
        plan_config(tokenizer, pooling="max")


# --------------------------------------------------------------------------------------
# plan-conditioned detail prior


def test_plan_conditioning_is_opt_in_and_leaves_every_earlier_config_and_network_unchanged():
    _, tokenizer, prior = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    earlier = [prior.config, replace(prior.config, history_stages=1, history_dropout=.5, part_valid_loss=True,
                                     locus_conditioning=True),
               replace(prior.config, rollout_blocks=.5, rollout_min=4, rollout_max=16),
               replace(prior.config, guided_attention=.1, input_corruption=.2, scheduled_sampling=.1)]
    for config in earlier:
        data = config.to_dict()
        assert not {"plan_window", "plan_codes", "plan_codebook", "plan_memory", "plan_dropout"} & set(data)
        again = RVQPriorConfig.from_dict(json.loads(json.dumps(data)))
        assert again == config and again.sha256() == config.sha256()
        assert not RVQPrior(config).plan_parameter_names()
    torch.manual_seed(0); a = RVQPrior(prior.config)
    torch.manual_seed(0); b = RVQPrior(planned(prior, plan))
    for k, v in a.state_dict().items():                                 # plan modules are created last
        assert torch.equal(v, b.state_dict()[k]), k
    on = planned(prior, plan, plan_dropout=.3)
    assert RVQPriorConfig.from_dict(json.loads(json.dumps(on.to_dict()))) == on and on.sha256() != prior.config.sha256()
    assert "plan_dropout" not in planned(prior, plan).to_dict()
    with pytest.raises(ValueError, match="plan_window"):
        replace(prior.config, plan_dropout=.5)
    with pytest.raises(ValueError, match="plan conditioning needs"):
        replace(prior.config, plan_window=2, plan_codes=0, plan_codebook=16, plan_tokenizer="x")
    with pytest.raises(ValueError, match="not plan-conditioned"):
        rvq_prior_loss(prior, tokenizer, *encode_texts(["a", "b"]), collate_multichannel(prior_setup()[0]),
                       plan_tokenizer=plan)


def test_the_training_script_allows_exactly_a_plan_less_warm_start():
    tmt = script("train_motion_tokens")
    _, tokenizer, prior = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    on = planned(prior, plan, plan_dropout=.5).to_dict()
    assert tmt.prior_warm_start_compatible(prior.config.to_dict(), on)
    assert tmt.prior_warm_start_compatible(on, planned(prior, plan, plan_dropout=.1).to_dict())   # training-only
    assert not tmt.prior_warm_start_compatible(on, prior.config.to_dict())                      # not the reverse
    assert not tmt.prior_warm_start_compatible(prior.config.to_dict(),
                                               replace(planned(prior, plan), history_stages=1).to_dict())
    assert not tmt.prior_warm_start_compatible(planned(prior, plan).to_dict(),
                                               replace(planned(prior, plan), plan_window=4).to_dict())
    model = RVQPrior(planned(prior, plan))
    with pytest.raises(SystemExit, match="not a plan-less version"):
        tmt.plan_warm_start(model, RVQPrior(replace(prior.config, d_model=32, heads=2)).state_dict() | {"x": torch.ones(1)})


@pytest.mark.parametrize("extra", [{}, dict(history_stages=1, locus_conditioning=True, history_dropout=.5,
                                            part_valid_loss=True)])
@pytest.mark.parametrize("memory", [True, False])
def test_a_zero_initialised_warm_start_equals_the_plan_less_prior_exactly(extra, memory):
    """The declared gate: a plan-conditioned prior warm-started from a plan-less one reproduces it bit for bit at step
    0, in the loss (evaluation and training, plan dropout on), its gradients, the sequence likelihood and sampled
    generation with a fixed seed, whatever the plan."""
    states, tokenizer, base = prior_setup()
    base = RVQPrior(replace(base.config, **extra))
    torch.manual_seed(5)
    plan = PlanVQ(plan_config(tokenizer))
    model = warm(base, planned(base, plan, plan_dropout=.5, plan_memory=memory))
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    loci = (torch.randn(2, LOCUS_SLOT_WIDTH),) if extra else ()
    for train in (False, True):
        base.train(train); model.train(train)
        torch.manual_seed(3)
        a, terms_a = rvq_prior_loss(base, tokenizer, tokens, mask, batch, *loci, train=train)
        torch.manual_seed(3)
        b, terms_b = rvq_prior_loss(model, tokenizer, tokens, mask, batch, *loci, train=train, plan_tokenizer=plan)
        assert torch.equal(a, b) and terms_a == terms_b
        base.zero_grad(); model.zero_grad()
        a.backward(); b.backward()
        own = dict(model.named_parameters())
        for name, p in base.named_parameters():
            assert (p.grad is None and own[name].grad is None) or torch.equal(p.grad, own[name].grad), name
        assert any(p.grad is not None and bool(p.grad.abs().any()) for n, p in own.items() if n.startswith("plan_"))
    base.eval(); model.eval()
    assert torch.equal(rvq_sequence_log_likelihood(base, tokenizer, tokens, mask, batch, *loci),
                       rvq_sequence_log_likelihood(model, tokenizer, tokens, mask, batch, *loci, plan_tokenizer=plan))
    texts, durations = ["ask him", "cut it now", "go"], [.5, .3, .42]
    plans = [torch.randint(0, 16, (plan_steps(-(-max(int(round(d * 24)), 1) // tokenizer.config.step), 2),
                                   plan.config.codes_per_step)) for d in durations]
    one, codes_a = generate_rvq_motion_batch(base, tokenizer, texts, fps=24., durations=durations, guidance=1.5,
                                             temperature={"body": 1.1, "face": .9}, seed=7, return_codes=True)
    two, codes_b = generate_rvq_motion_batch(model, tokenizer, texts, fps=24., durations=durations, guidance=1.5,
                                             temperature={"body": 1.1, "face": .9}, seed=7, return_codes=True, plan=plans)
    assert all(torch.equal(x, y) for x, y in zip(codes_a, codes_b))
    for x, y in zip(one, two):
        for name in x.channels:
            np.testing.assert_array_equal(x.channels[name].values, y.channels[name].values)


def test_a_plan_dropout_row_equals_the_no_plan_row_and_the_plan_matters_once_trained():
    states, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    model = wake(warm(base, planned(base, plan, plan_dropout=.5))).train()
    batch = collate_multichannel(states * 2)
    tokens, mask = encode_texts(["ask him", "cut it", "go now", "give"])
    seen, original = [], model.temporal_states
    model.temporal_states = lambda h, m, prev, *a, **k: seen.append((h, m, prev, a, k)) or original(h, m, prev, *a, **k)
    dropped_rows = set()
    for seed in range(6):
        seen.clear()
        torch.manual_seed(seed)
        rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan)
        h, m, prev, a, k = seen[-1]
        codes, valid = k["plan"]
        latents, step_valid, _ = tokenizer_latents(tokenizer, batch)
        true_codes, true_valid = plan.encode_latents(latents, step_valid)
        assert torch.equal(valid, true_valid)                              # the plan mask is never changed
        empty = model.empty_plan(valid)
        with torch.no_grad():
            got = original(h, m, prev, *a, plan=(codes, valid))
            none = original(h, m, prev, *a, plan=empty)
            real = original(h, m, prev, *a, plan=(true_codes, true_valid))
        for r in range(len(codes)):
            if bool((codes[r] == model.config.plan_codebook).all()):
                dropped_rows.add((seed, r))
                assert torch.equal(got[r], none[r])                         # a dropped row is the no-plan row
            else:
                assert torch.equal(codes[r], true_codes[r]) and torch.equal(got[r], real[r])
    assert 0 < len(dropped_rows) < 24                                       # both kinds of rows occurred
    model.temporal_states = original
    model.eval()
    true, _ = rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=False, plan_tokenizer=plan)
    empty, _ = rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=False, plan_tokenizer=plan, plan_mode="empty")
    assert not torch.equal(true, empty)                                     # the informativeness measure moves
    # all rows dropped == the empty plan everywhere (the same random draws: the plan draw comes last)
    always = RVQPrior(planned(base, plan, plan_dropout=1.)); always.load_state_dict(model.state_dict())
    never = RVQPrior(planned(base, plan)); never.load_state_dict(model.state_dict())
    always.train(); never.train()
    torch.manual_seed(9)
    x, _ = rvq_prior_loss(always, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan)
    torch.manual_seed(9)
    y, _ = rvq_prior_loss(never, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan, plan_mode="empty")
    assert torch.equal(x, y)
    with pytest.raises(ValueError, match="plan_mode"):
        rvq_prior_loss(never, tokenizer, tokens, mask, batch, train=False, plan_tokenizer=plan, plan_mode="half")


def test_the_plan_reaches_every_step_of_its_window_and_the_memory_sees_the_whole_plan():
    states, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer, window=2))
    for memory in (False, True):
        model = wake(warm(base, planned(base, plan, plan_memory=memory))).eval()
        S = 6
        hidden = torch.randn(1, 4, 32); mask = torch.ones(1, 4, dtype=torch.bool)
        prev = torch.randint(0, model.K, (1, S, model.J)); prev[:, 0] = model.K
        seconds = (torch.arange(S) / 24.)[None]; duration = torch.tensor([S / 24.])
        codes = torch.randint(0, 16, (1, 3, plan.config.codes_per_step)); valid = torch.ones(1, 3, dtype=torch.bool)
        changed = codes.clone(); changed[0, 2] = (changed[0, 2] + 1) % 16           # plan step 2: steps 4 and 5
        with torch.no_grad():
            a = model.temporal_states(hidden, mask, prev, seconds, duration, plan=(codes, valid))
            b = model.temporal_states(hidden, mask, prev, seconds, duration, plan=(changed, valid))
        moved = (a - b).abs().amax(-1)[0] > 1e-6
        if memory:
            assert bool(moved.all())                       # the readers see the whole plan, future plan steps included
        else:
            assert moved.tolist() == [False] * 4 + [True] * 2   # the step-rate input: its window, then causal flow
        with pytest.raises(ValueError, match="shorter"):
            model.temporal_states(hidden, mask, prev, seconds, duration, plan=(codes[:, :2], valid[:, :2]))
        with pytest.raises(ValueError, match="needs plan codes"):
            model.temporal_states(hidden, mask, prev, seconds, duration)


# --------------------------------------------------------------------------------------
# plan prior and two-stage generation


def plan_prior_of(tokenizer, plan, **kw):
    return RVQPrior(PlanPriorConfig(plan.config, **{**dict(d_model=32, heads=4, text_layers=1, temporal_layers=1,
                                                           depth_layers=1, feedforward=64, dropout=0.), **kw}))


def test_the_plan_prior_learns_plan_codes_and_its_config_round_trips():
    states, tokenizer, base = prior_setup()
    torch.manual_seed(0)
    plan = PlanVQ(plan_config(tokenizer)).eval()
    prior = plan_prior_of(tokenizer, plan)
    config = prior.config
    again = PlanPriorConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert again == config and again.sha256() == config.sha256() and config.tokenizer.step == 2 * tokenizer.config.step
    assert prior.J == plan.config.codes_per_step and prior.K == plan.config.codebook
    with pytest.raises(ValueError, match="not plan-conditioned"):
        replace(config, plan_window=2, plan_codes=8, plan_codebook=16, plan_tokenizer="x")
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    codes, plan_valid = encode_plan(plan, tokenizer, batch)
    optimizer = torch.optim.Adam(prior.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        loss, terms = plan_prior_loss(prior, tokenizer, plan, tokens, mask, batch, train=False)
        first = first or float(loss.detach())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert float(loss.detach()) < .3 * first and terms["token_accuracy"] > .8
    ll = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False,
                        targets=(codes, plan_valid, None))[1]["tokens"]
    assert abs(ll - plan_prior_loss(prior, tokenizer, plan, tokens, mask, batch, train=False)[1]["tokens"]) < 1e-9
    # part-valid weights pool the per-step rule over each window
    weighted = RVQPrior(replace(config, part_valid_loss=True)); weighted.load_state_dict(prior.state_dict())
    loss_w, _ = plan_prior_loss(weighted, tokenizer, plan, tokens, mask, batch, train=False)
    assert torch.isfinite(loss_w) and not torch.isclose(loss_w, loss)       # the face is untracked in these states
    with pytest.raises(ValueError, match="needs a plan prior"):
        plan_prior_loss(base, tokenizer, plan, tokens, mask, batch)


def test_the_plan_prior_warm_starts_its_text_encoder_from_a_detail_prior():
    _, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    prior = plan_prior_of(tokenizer, plan)
    copied = copy_text_encoder(prior, base.state_dict(), base.config.to_dict())
    assert copied and all(k.startswith(("embed.", "grammar.", "text_encoder.", "duration.")) for k in copied)
    tokens, mask = encode_texts(["ask him"])
    base.eval(); prior.eval()
    with torch.no_grad():
        a, da, _ = base.encode(tokens, mask)
        b, db, _ = prior.encode(tokens, mask)
    assert torch.equal(a, b) and torch.equal(da, db)
    with pytest.raises(ValueError, match="differ in config"):
        copy_text_encoder(plan_prior_of(tokenizer, plan, text_layers=2), base.state_dict(), base.config.to_dict())


def test_two_stage_generation_is_deterministic_with_a_fixed_plan_and_aligned():
    states, tokenizer, base = prior_setup()
    torch.manual_seed(2)
    plan = PlanVQ(plan_config(tokenizer)).eval()
    plan_prior = plan_prior_of(tokenizer, plan)
    detail = wake(warm(base, planned(base, plan))).eval()
    texts, durations = ["ask him", "go"], [.5, .3]
    out, (plans, codes) = generate_plan_detail_batch(plan_prior, detail, tokenizer, texts, fps=24., durations=durations,
                                                     guidance=1.5, temperature=1.1, seed=3, return_codes=True)
    for s, p, c, d in zip(out, plans, codes, durations):
        frames = int(round(d * 24))
        steps = -(-frames // tokenizer.config.step)
        assert len(s.timestamps) == frames and c.shape == (steps, detail.J)
        assert p.shape == (plan_steps(steps, 2), plan.config.codes_per_step)
        s.validate()
    again = generate_plan_detail_batch(plan_prior, detail, tokenizer, texts, fps=24., durations=durations, guidance=1.5,
                                       temperature=1.1, seed=3, return_codes=True)[1]
    assert all(torch.equal(x, y) for x, y in zip(plans + codes, again[0] + again[1]))
    # a fixed plan: deterministic for a seed; the plan prior is not consulted
    fixed = [torch.randint(0, 16, p.shape) for p in plans]
    ran = generate_plan_detail_batch(plan_prior, detail, tokenizer, texts, fps=24.,
                                     durations=durations, guidance=1.5, seed=4, plans=fixed, return_codes=True)[1]
    plan_prior.encode = lambda *a, **k: (_ for _ in ()).throw(AssertionError("the plan prior ran"))
    same = generate_plan_detail_batch(plan_prior, detail, tokenizer, texts, fps=24., durations=durations,
                                      guidance=1.5, seed=4, plans=fixed, return_codes=True)[1]
    assert all(torch.equal(x, y) for x, y in zip(ran[0] + ran[1], same[0] + same[1]))
    other = generate_plan_detail_batch(plan_prior, detail, tokenizer, texts, fps=24., durations=durations,
                                       guidance=1.5, seed=4, plans=[(f + 1) % 16 for f in fixed], return_codes=True)[1]
    assert not all(torch.equal(x, y) for x, y in zip(ran[1], other[1]))    # the plan steers the detail
    with pytest.raises(ValueError, match="plan 0: shape"):
        generate_rvq_motion_batch(detail, tokenizer, texts, fps=24., durations=durations, plan=[f[:1] for f in fixed])
    with pytest.raises(ValueError, match="generate_rvq_motion_batch"):
        generate_rvq_motion(detail, tokenizer, ["go"], fps=24., durations=[.3])
    # predicted durations when none are given; the plan's length follows them
    _, (plans_p, codes_p) = generate_plan_detail_batch(plan_prior_of(tokenizer, plan), detail, tokenizer, ["ask him"],
                                                       fps=24., seed=1, return_codes=True)
    assert plans_p[0].shape[0] == plan_steps(codes_p[0].shape[0], 2)


def test_generation_refuses_a_mismatched_plan_pair():
    _, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    detail = warm(base, planned(base, plan))
    with pytest.raises(ValueError, match="plan layout"):
        check_plan_pair(plan_prior_of(tokenizer, PlanVQ(plan_config(tokenizer, window=3))), detail)
    with pytest.raises(ValueError, match="expected a plan prior"):
        check_plan_pair(detail, detail)
    other = PlanVQ(PlanVQConfig(replace(tokenizer.config, commitment=.5), window=2, hidden=16, code_dim=8, codebook=16,
                                stages=2, blocks=1))
    with pytest.raises(ValueError, match="different tokenizers"):
        check_plan_pair(plan_prior_of(tokenizer, other), detail)
    codes = generate_plan_codes_batch(plan_prior_of(tokenizer, plan), ["ask him", "go"], fps=24., durations=[.5, .3])
    assert [c.shape[0] for c in codes] == [3, 2]          # 12 and 7 frames -> 6 and 4 steps -> 3 and 2 plan steps


def test_the_round_trip_hook_decodes_the_plan_reconstruction():
    """The fidelity instruments' hook (tokenizer_fidelity_check / tokenizer_recognisability_check
    --plan-tokenizer-key): round_trip with a plan decodes the plan-reconstructed latents."""
    rc = script("tokenizer_recognisability_check")
    torch.manual_seed(0)
    state = signer_state(0, frames=16, amplitude=.4)
    tokenizer = prior_setup()[1]
    plan = PlanVQ(plan_config(tokenizer)).eval()
    plain = rc.round_trip(tokenizer, tokenizer.config.motion, state, 24.)
    through = rc.round_trip(tokenizer, tokenizer.config.motion, state, 24., plan=plan)
    through.validate()
    assert len(through.timestamps) == len(plain.timestamps) == 16
    assert not np.array_equal(through.channels["body"].values, plain.channels["body"].values)
    # a plan whose decoder returns the quantised latents reproduces the plain round trip exactly
    unit = diffusion_targets(tokenizer.config.motion, collate_multichannel([state]))
    quantised = tokenizer.quantize(tokenizer.encode(unit))[0]
    plan.reconstruct = lambda latents, valid: quantised
    same = rc.round_trip(tokenizer, tokenizer.config.motion, state, 24., plan=plan)
    for name in plain.channels:
        np.testing.assert_array_equal(same.channels[name].values, plain.channels[name].values)


def test_d2_like_tokenizers_with_one_step_per_frame_align_plan_steps_to_frames():
    torch.manual_seed(0)
    states = [signer_state(s, frames=13, amplitude=.4) for s in range(2)]
    from signtranslator.generation.motion_rvq import MotionRVQ
    tokenizer = MotionRVQ(small(states[0], down_layers=0, base_blocks=1)).eval()
    assert tokenizer.config.step == 1
    plan = PlanVQ(plan_config(tokenizer, window=4)).eval()
    codes, valid = encode_plan(plan, tokenizer, collate_multichannel(states))
    assert codes.shape[:2] == (2, 4) and valid.all() and plan.config.step == 4      # 13 frames -> 4 plan steps


def test_oracle_plans_are_each_clips_own_plan_fitted_to_the_generated_steps():
    from signtranslator.generation.plan_codes import oracle_plans
    states, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer)).eval()
    seconds = [len(s.timestamps) / 24. for s in states]
    plans, adjusted = oracle_plans(plan, tokenizer, states, seconds, 24.)
    assert adjusted == 0
    for p, s in zip(plans, states):
        assert torch.equal(p, encode_plan(plan, tokenizer, collate_multichannel([s]))[0][0])
    short, adjusted = oracle_plans(plan, tokenizer, states, [.2, 1.], 24.)       # 5 and 24 frames
    assert adjusted == 2 and [len(p) for p in short] == [2, 6]
    assert torch.equal(short[0], plans[0][:2]) and torch.equal(short[1][4:], plans[1][-1:].expand(2, -1))
    detail = wake(warm(base, planned(base, plan))).eval()
    out = generate_plan_detail_batch(plan_prior_of(tokenizer, plan), detail, tokenizer, ["ask him", "cut it"], fps=24.,
                                     durations=seconds, plans=plans, seed=0)
    assert [len(o.timestamps) for o in out] == [len(s.timestamps) for s in states]


def test_the_causal_mask_detection_matches_the_decoders_and_its_fallback():
    from signtranslator.generation.motion_rvq import _causal_mask_rule, detect_is_causal_mask
    from torch.nn.modules.transformer import _detect_is_causal_mask
    for size in (1, 4, 7):
        bool_mask = torch.triu(torch.ones(size, size, dtype=torch.bool), 1)
        float_mask = torch.triu(torch.full((size, size), float("-inf")), 1)
        other = bool_mask.clone(); other[0, 0] = True
        for mask in (bool_mask, float_mask, other, torch.zeros(size, size, dtype=torch.bool)):
            assert detect_is_causal_mask(mask, size) == _detect_is_causal_mask(mask, None, size) == \
                _causal_mask_rule(mask, size)
    assert detect_is_causal_mask(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1), 4) is True
    assert _causal_mask_rule(None, 3) is False and _causal_mask_rule(torch.ones(3, 4, dtype=torch.bool), 3) is False


def test_plan_corruption_is_opt_in_training_only_and_replaces_a_fraction_of_plan_steps():
    from signtranslator.generation.motion_rvq import teacher_forced_plan_samples
    states, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer)).eval()
    off = planned(base, plan, plan_dropout=.2)
    assert not {"plan_corruption", "plan_corruption_max", "plan_corruption_source"} & set(off.to_dict())
    on = planned(base, plan, plan_corruption=1., plan_corruption_max=1., plan_corruption_source="random")
    assert RVQPriorConfig.from_dict(json.loads(json.dumps(on.to_dict()))) == on
    with pytest.raises(ValueError, match="plan corruption needs"):
        planned(base, plan, plan_corruption=.5, plan_corruption_source="oracle")
    with pytest.raises(ValueError, match="need plan_window"):
        replace(base.config, plan_corruption=.5)
    # zero-initialised: corruption changes nothing (its draws come after every other draw, before plan dropout)
    batch = collate_multichannel(states * 2)
    tokens, mask = encode_texts(["ask him", "cut it", "go now", "give"])
    model = warm(base, on).train()
    base.train()
    torch.manual_seed(4); a, _ = rvq_prior_loss(base, tokenizer, tokens, mask, batch, train=True)
    torch.manual_seed(4); b, terms = rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan)
    assert torch.equal(a, b) and 0 < terms["plan_corrupted"] <= 1
    # which steps change: only valid plan steps of selected rows; evaluation never corrupts
    wake(model)
    seen, original = [], model.temporal_states
    model.temporal_states = lambda h, m, prev, *x, **k: seen.append(k["plan"]) or original(h, m, prev, *x, **k)
    latents, step_valid, _ = tokenizer_latents(tokenizer, batch)
    true_codes, true_valid = plan.encode_latents(latents, step_valid)
    rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=False, plan_tokenizer=plan)
    assert torch.equal(seen[-1][0], true_codes)
    changed = []
    for seed in range(5):
        torch.manual_seed(seed)
        rvq_prior_loss(model, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan)
        codes, valid = seen[-1]
        assert torch.equal(valid, true_valid)
        changed.append(float((codes != true_codes).any(-1).float().mean()))
    assert 0 < np.mean(changed) < 1
    # the plan prior's source: its teacher-forced samples, at the corrupted steps only
    prior = plan_prior_of(tokenizer, plan).eval()
    with torch.no_grad():
        prior.heads_bias.zero_(); prior.heads_bias[:, 3] = 1e4                 # a plan prior that always says code 3
    drawn = teacher_forced_plan_samples(prior, tokens, mask, None, true_codes, true_valid, batch)
    assert drawn.shape == true_codes.shape and bool((drawn == 3).all()) and not prior.training
    from_prior = RVQPrior(planned(base, plan, plan_corruption=1., plan_corruption_max=1.))
    from_prior.load_state_dict(model.state_dict())
    from_prior.temporal_states = lambda h, m, prev, *x, **k: seen.append(k["plan"]) or original(h, m, prev, *x, **k)
    from_prior.train()
    with pytest.raises(ValueError, match="needs plan_prior"):
        rvq_prior_loss(from_prior, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan)
    torch.manual_seed(1)
    rvq_prior_loss(from_prior, tokenizer, tokens, mask, batch, train=True, plan_tokenizer=plan, plan_prior=prior)
    codes, _ = seen[-1]
    moved = codes != true_codes
    assert bool(moved.any()) and bool((codes[moved] == 3).all())


def test_stage1_tokens_is_the_stage_1_codes_cross_entropy():
    states, tokenizer, base = prior_setup()
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    base.eval()
    _, terms = rvq_prior_loss(base, tokenizer, tokens, mask, batch, train=False)
    seen, original = [], base.depth_logits
    base.depth_logits = lambda h, c: seen.append((original(h, c), c)) or seen[-1][0]
    rvq_prior_loss(base, tokenizer, tokens, mask, batch, train=False)
    logits, prefix = seen[-1]
    P = len(tokenizer.parts)
    unit = __import__("signtranslator.generation.text_to_motion", fromlist=["x"]).diffusion_targets(tokenizer.config.motion, batch)
    codes = tokenizer.quantize(tokenizer.encode(unit))[1][tokenizer.token_valid(batch.frame_valid)]
    expected = torch.nn.functional.cross_entropy(logits[:, :P].reshape(-1, base.K), codes[:, :P].reshape(-1))
    assert abs(terms["stage1_tokens"] - float(expected.detach())) < 1e-6


def test_the_latent_loss_reads_only_observed_parts_and_the_codebook_diagnostics_are_logged():
    from signtranslator.generation.plan_codes import part_step_observed
    states, tokenizer, _ = prior_setup()
    batch = collate_multichannel(states)
    observed = part_step_observed(tokenizer, batch)
    assert not bool(observed["face"].any()) and bool(observed["body"].any())    # the test states track no face
    torch.manual_seed(0)
    plan = PlanVQ(plan_config(tokenizer))
    latents = {p: torch.randn(2, 4, 3) for p in ("a", "b")}
    rec = {p: v + 1. for p, v in latents.items()}
    only_a = relative_latent_error(rec, latents, {"a": torch.ones(2, 4, dtype=torch.bool),
                                                  "b": torch.zeros(2, 4, dtype=torch.bool)})
    z = latents["a"]
    torch.testing.assert_close(only_a, (torch.ones_like(z) ** 2).sum() / ((z - z.mean((0, 1))) ** 2).sum())
    loss, terms = plan_tokenizer_loss(plan, tokenizer, batch)
    assert torch.isfinite(loss)
    for p in plan.parts:
        assert 1. <= float(terms[f"stage1_perplexity_{p}"]) <= plan.config.codebook and f"reseeded_{p}" in terms
    q = plan.quantizers["body"]
    before = q.reseeded
    with torch.no_grad():
        q.code_mass[0, :5] = 0.                                                # five dead codes, never assigned
        q.codes[0, :5] = 1e6
    x = torch.randn(10, plan.config.code_dim)
    _, idx, inputs = q(x)
    q.update(inputs, idx, torch.ones(10, dtype=torch.bool))
    assert q.reseeded >= before + 5 and "reseeded" not in str(q.state_dict().keys())


def test_a_detail_prior_records_its_plan_tokenizer_and_a_pair_must_share_it():
    _, tokenizer, base = prior_setup()
    plan = PlanVQ(plan_config(tokenizer))
    detail = warm(base, planned(base, plan))
    check_plan_pair(plan_prior_of(tokenizer, plan), detail)
    other = PlanVQ(plan_config(tokenizer, hidden=24))                         # same layout, another plan tokenizer
    with pytest.raises(ValueError, match="different plan tokenizers"):
        check_plan_pair(plan_prior_of(tokenizer, other), detail)
    with pytest.raises(ValueError, match="plan tokenizer's config SHA-256"):
        replace(base.config, plan_window=2, plan_codes=8, plan_codebook=16)


def test_plan_only_states_decode_plans_through_the_tokenizers_decoder():
    from signtranslator.generation.plan_codes import decode_plan_states, oracle_plans
    states, tokenizer, _ = prior_setup()
    plan = PlanVQ(plan_config(tokenizer)).eval()
    seconds = [len(s.timestamps) / 24. for s in states]
    plans, _ = oracle_plans(plan, tokenizer, states, seconds, 24.)
    out = decode_plan_states(plan, tokenizer, plans, ["ask him", "cut it"], fps=24., durations=seconds)
    for o, s in zip(out, states):
        o.validate()
        assert len(o.timestamps) == len(s.timestamps)
    with pytest.raises(ValueError, match="plan steps"):
        decode_plan_states(plan, tokenizer, [plans[0][:1]], ["x"], fps=24., durations=seconds[:1])


def test_sentence_retrieval_pairs_and_intervals_resample_unordered_pairs():
    diag = script("round17_diagnostics")
    assert diag.sentence_bleu("the cat sat", "the cat sat") == pytest.approx(1.)
    assert diag.sentence_bleu("", "the cat") == 0. and diag.sentence_bleu("dog", "the cat") == 0.
    assert 0 < diag.sentence_bleu("the cat ran", "the cat sat") < 1
    h = diag.hits(["the cat sat", "a dog ran", "x"], ["the cat sat", "a dog ran", "y"], ["a dog ran", "the cat sat", "z"])
    assert h.tolist() == [1., 1., .5]
    # pairing: consecutive, or neighbours in duration within 10%
    assert diag.make_pairs([1, 2, 3, 4, 5], "consecutive") == [(0, 1), (2, 3)]
    assert diag.make_pairs([1., 5., 1.05, 5.4, 9.], "duration") == [(0, 2), (1, 3)]
    assert diag.make_pairs([1., 1.2, 1.4], "duration") == []
    # a pair's two directions stay together: a pair whose directions are (1, 0) contributes 0.5, never 0 or 1 alone
    split = np.array([[1., 0.]] * 50)
    r = diag.pair_rate(split)
    assert r["rate"] == .5 and r["ci95"] == [.5, .5] and r["pairs"] == 50 and r["directions"] == 100
    a, b = np.array([[1., 1.], [1., 0.], [0., 1.]]), np.array([[0., 1.], [1., 0.], [0., 0.]])
    d = diag.paired_rate_difference(a, b)
    assert d["difference"] == pytest.approx((.5 + 0 + .5) / 3, abs=1e-4) and d["ci95"][0] <= d["difference"] <= d["ci95"][1]
    assert diag.blank_check(1.8594, 1.859)["reproduced"] and not diag.blank_check(1.861, 1.859)["reproduced"]


@pytest.mark.parametrize("variant", ["plain", "part_valid", "detail", "plan_prior"])
def test_per_clip_nats_sum_to_the_losses_teacher_forced_tokens(variant):
    """Text gain per second rests on per-clip summed losses; they must add up to the prior's own loss."""
    diag = script("round17_diagnostics")
    from signtranslator.generation.plan_codes import PLAN_PRIOR_KIND
    states, tokenizer, base = prior_setup()
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["ask him", "cut it"])
    plan = PlanVQ(plan_config(tokenizer)).eval()
    if variant == "plain":
        prior, kind = base, "motion_rvq_prior"
    elif variant == "part_valid":
        prior, kind = RVQPrior(replace(base.config, part_valid_loss=True)), "motion_rvq_prior"
    elif variant == "detail":
        prior, kind = wake(warm(base, planned(base, plan, part_valid_loss=True))), "motion_rvq_prior"
    else:
        prior, kind = plan_prior_of(tokenizer, plan, part_valid_loss=True), PLAN_PRIOR_KIND
    prior.eval()
    nats, nats1, weight = diag.row_nats(prior, tokenizer, plan, kind, tokens, mask, batch)
    if kind == PLAN_PRIOR_KIND:
        _, terms = plan_prior_loss(prior, tokenizer, plan, tokens, mask, batch, train=False)
    else:
        _, terms = rvq_prior_loss(prior, tokenizer, tokens, mask, batch, train=False,
                                  **({"plan_tokenizer": plan} if prior.config.plan_window else {}))
    assert nats.shape == (2,) and bool((weight > 0).all())
    assert float(nats.sum() / weight.sum()) == pytest.approx(terms["tokens"], rel=1e-5)
    assert bool((nats1 <= nats + 1e-6).all())
    batches = [(tokens, mask, batch, ["ask him", "cut it"])]
    out = diag.text_gain(prior, tokenizer, plan, kind, batches, torch.device("cpu"))
    assert out["clips"] == 2 and set(out["text_gain_nats_per_second"]) == {"mean", "ci95", "clips"}
    from signtranslator.generation.text_to_motion import utterance_duration
    shuffled, _, _ = diag.row_nats(prior, tokenizer, plan, kind, tokens.roll(1, 0), mask.roll(1, 0), batch)
    per_second = ((shuffled - nats).double() / utterance_duration(batch).double()).mean()
    assert out["text_gain_nats_per_second"]["mean"] == pytest.approx(float(per_second), abs=1e-4)


def test_the_oracle_precondition_compares_oracle_plans_with_the_controls_generated_bleu():
    cpl = script("compare_pilot_lexicon")
    refs = [f"the cat sat number {i}" for i in range(40)]
    good = [r for r in refs]
    bad = ["nothing here"] * 40
    arm = dict(meaning=dict(references=refs, generated=dict(bleu=1., hypotheses=bad)),
               meaning_oracle_plan=dict(bleu=99., hypotheses=good))
    control = dict(meaning=dict(references=refs, generated=dict(bleu=1., hypotheses=bad)))
    out = cpl.oracle_precondition(arm, control, resamples=200)
    assert out["oracle_precondition"]["passes"] and out["oracle_precondition"]["ci95"][0] > 0
    swapped = dict(arm, meaning_oracle_plan=dict(bleu=0., hypotheses=bad))
    assert not cpl.oracle_precondition(swapped, dict(meaning=dict(references=refs, generated=dict(bleu=99., hypotheses=good))),
                                       resamples=200)["oracle_precondition"]["passes"]
    with pytest.raises(ValueError, match="different references"):
        cpl.oracle_precondition(arm, dict(meaning=dict(references=refs[::-1], generated=dict(bleu=1., hypotheses=bad))))
    with pytest.raises(ValueError, match="no meaning_oracle_plan"):
        cpl.oracle_precondition(control, control)
