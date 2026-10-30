"""Instrument v2 (state-to-text back-translator): featurisation, conform, data guards, controls, gates, a tiny
end-to-end training run, the scorer branch and the unchanged v1 defaults (docs/NEXT_PHASE_PLAN.md, "Instrument v2")."""
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from signtranslator.data import keypoints2d  # noqa: E402
from signtranslator.data.state_features import (feature_width, from_state_3d, native_features, normalise_nd,  # noqa: E402
                                                state_features)
from signtranslator.data_engineering.video_landmarks import BODY_LABELS, finger_joints  # noqa: E402
from signtranslator.generation import bt_corpora as bt  # noqa: E402
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion  # noqa: E402
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d  # noqa: E402

TWO_VIEW = ("two-view-triangulated-camera1-y-up", "sapiens-two-view-swing-ik-y-up-v1")
MONO = ("camera-aligned-y-up-metres", "mediapipe-tasks-swing-ik-y-up-v1")
LABELS = {"body": BODY_LABELS, "left_hand": tuple(j.name for j in finger_joints("Left")),
          "right_hand": tuple(j.name for j in finger_joints("Right")), "face": ("jawOpen", "browUp", "smile"),
          "head_rotation": ("head",), "head_translation": ("head",), "root_translation": ("root",),
          "left_gaze": ("left_eye",), "right_gaze": ("right_eye",), "blink": ("blink",), "contact": ("contact",)}
ABSENT = ("contact",)


def h2s_state(seed=0, frames=30, *, layout=TWO_VIEW, absent=ABSENT, sample="s"):
    """A How2Sign-like state: every channel but ``absent`` valid, smooth random motion."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / 24.
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        labels = LABELS[name]
        n = len(labels)
        if kind == "rotation_6d_columns":
            axis = rng.normal(size=(n, 3)) * .4
            angle = np.sin(2 * t + seed)[:, None, None] + .3
            values = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(axis[None] * angle))).numpy()
        elif kind == "unit_vector":
            v = np.stack([.1 * np.sin(t + seed), .1 * np.cos(t), np.ones_like(t)], -1)[:, None]
            values = v / np.linalg.norm(v, axis=-1, keepdims=True)
        elif kind == "translation":
            values = np.stack([.05 * np.sin(t), .02 * t, 1. + 0 * t], -1)[:, None] + seed
        elif kind == "bounded_scalar":
            values = (.5 + .4 * np.sin(3 * t + seed))[:, None, None] * np.ones((1, n, 1))
        else:
            values = (.3 * np.sin(t[:, None, None] + np.arange(n)[None, :, None] + seed))
        valid = np.full((frames, n), name not in absent)
        frame, convention = (layout if name in ("body", "left_hand", "right_hand") else ("f", "c"))
        channels[name] = MotionChannel(np.where(valid[..., None], values, 0.), valid, valid.copy(),
                                       np.where(valid, 1., 0.), labels, frame, units, "src", "a" * 64, convention)
    return MultichannelMotion(t, "clock", f"{sample}{seed}", channels)


def mono_state(seed=0, frames=30):
    """A monocular (YouTube-like) state: equivalent rotation layout, no translation or gaze, placeholder labels."""
    s = h2s_state(seed, frames, layout=MONO, absent=ABSENT + ("root_translation", "head_translation", "left_gaze",
                                                              "right_gaze"), sample="m")
    ch = dict(s.channels)
    for name in ("root_translation", "head_translation", "left_gaze", "right_gaze"):
        ch[name] = replace(ch[name], labels=("unavailable",), coordinate_frame="none", convention="unavailable")
    return replace(s, channels=ch)


def blank(state):
    from evaluate_text_to_motion import blanked
    return blanked([state])[0]


def config_for(template, **kw):
    from signtranslator.generation.state_to_text_t5 import StateToT5Config, motion_config
    pin = (("manifest_key", "m.json"), ("manifest_sha256", "0" * 64), ("corpus", bt.HOW2SIGN), ("sign", "t0"),
           ("state_sha256", "1" * 64))
    return StateToT5Config(motion_config(template), template=pin, **kw)


# ---- featurisation ---------------------------------------------------------------------------------------------

def test_3d_points_extend_v1s_2d_points_and_normalise_like_v1():
    s = h2s_state(1)
    p3, v3 = from_state_3d(s)
    p2, v2 = keypoints2d.from_state(s)
    assert p3.shape == (30, 48, 3) and v3.shape == (30, 48)
    np.testing.assert_allclose(p3[..., :2], p2, atol=1e-6)
    assert (v3 == v2).all() and v3.all()
    assert np.ptp(p3[..., 2]) > 1e-3                                     # depth is kept
    np.testing.assert_allclose(normalise_nd(p2, v2), keypoints2d.normalise(p2, v2, y_down=False), atol=1e-6)
    n = normalise_nd(p3, v3)
    mid = np.median((n[:, 0] + n[:, 1]) / 2, axis=0)
    width = np.median(np.linalg.norm(n[:, 0] - n[:, 1], axis=-1))
    np.testing.assert_allclose(mid, 0, atol=1e-5)
    np.testing.assert_allclose(width, 1, atol=1e-5)


def test_features_round_trip_every_block_and_have_the_declared_width():
    s = h2s_state(2)
    motion = config_for(s).motion
    f = state_features(s, motion)
    assert f.shape == (30, feature_width(motion)) and f.dtype == np.float32
    points, valid = from_state_3d(s)
    np.testing.assert_allclose(f[:, :144].reshape(30, 48, 3), normalise_nd(points, valid), atol=1e-6)
    assert (f[:, 144:192] == valid).all()
    native, flags = native_features(s, motion)
    np.testing.assert_allclose(f[:, 192:192 + native.shape[1]], native)
    np.testing.assert_allclose(f[:, 192 + native.shape[1]:], flags)
    # the native block decodes back to the state: rotations as stored, face / 0.5, blink 2v - 1, centred translations
    from signtranslator.generation.text_to_motion import diffusion_layout
    at = 0
    for name, kind, elements, width in diffusion_layout(motion):
        block = native[:, at:at + width].reshape(30, elements, -1)
        values = s.channels[name].values
        if kind == "translation":
            np.testing.assert_allclose(block * .1, values - np.median(values, axis=0), atol=1e-6)
            np.testing.assert_allclose(np.median(block, axis=0), 0, atol=1e-6)
        elif kind == "coefficient":
            np.testing.assert_allclose(block * motion.coefficient_norm, values, atol=1e-6)
        elif kind == "bounded_scalar":
            np.testing.assert_allclose((block + 1) / 2, values, atol=1e-6)
        else:
            np.testing.assert_allclose(block, values, atol=1e-6)
        at += width
    assert set(motion.trained_channels) == set(CHANNELS) - {"contact"}


def test_native_values_match_the_diffusion_space_and_a_blank_state_is_all_zero():
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets, diffusion_valid
    s = h2s_state(3)
    motion = replace(config_for(s).motion, trained_channels=("body", "left_hand", "face", "blink", "left_gaze"))
    native, _ = native_features(s, motion)
    batch = collate_multichannel([s])
    expected = (diffusion_targets(motion, batch) * diffusion_valid(motion, batch))[0].numpy()
    np.testing.assert_allclose(native, expected, atol=1e-5)
    b = state_features(blank(s), config_for(s).motion)
    assert b.shape[0] == 30 and not b.any()


# ---- conform ----------------------------------------------------------------------------------------------------

def test_the_validity_mask_survives_conform_and_absent_channels_read_as_absent():
    from signtranslator.generation.state_to_text_t5 import layout_template, prepare
    template = h2s_state(0)
    config = config_for(template)
    layout = layout_template(config)
    s = mono_state(4)
    body = s.channels["body"]
    vb = body.valid.copy(); vb[5:9, BODY_LABELS.index("LeftForearm")] = False       # a gap in tracking
    s.channels["body"] = replace(body, valid=vb, observed=vb.copy(), confidence=np.where(vb, 1., 0.),
                                 values=np.where(vb[..., None], body.values, 0.))
    s.validate()
    f = prepare(s, config, layout)
    from signtranslator.generation.text_to_motion import diffusion_layout
    layout_rows = diffusion_layout(config.motion)
    at, seen = 192 + sum(width for *_, width in layout_rows), {}
    for name, _, elements, _ in layout_rows:
        seen[name] = f[:, at:at + elements]
        at += elements
    assert at == f.shape[1]
    assert (seen["body"] == s.channels["body"].valid).all()
    assert (seen["left_hand"] == 1).all() and (seen["face"] == 1).all()
    for name in ("root_translation", "head_translation", "left_gaze", "right_gaze"):
        assert not seen[name].any()
    point_valid = f[:, 144:192]
    assert not point_valid[5:9, 4].any() and point_valid[:5, 4].all()                # the left wrist follows the forearm
    same = h2s_state(5)
    np.testing.assert_array_equal(prepare(same, config, layout), state_features(same, config.motion))
    other = h2s_state(6, layout=("someone-elses-frame", "unknown-convention"))
    with pytest.raises(ValueError):
        prepare(other, config, layout)


def test_states_off_the_rate_are_resampled_before_featurising():
    from signtranslator.generation.state_to_text_t5 import prepare
    s = h2s_state(7, frames=60)
    s30 = replace(s, timestamps=np.arange(60) / 30.)
    f = prepare(s30, config_for(s))
    assert abs(len(f) - int(59 / 30 * 24) - 1) <= 1


# ---- data guards ------------------------------------------------------------------------------------------------

def record(sign, corpus=bt.HOW2SIGN, partition="train", component="c02", text=None, key=None, sha="0" * 64):
    r = dict(sign=sign, corpus=corpus, partition=partition, text=text or f"the sentence of {sign} goes here now",
             continuous=dict(status="converted", state_key=key or f"states/{sign}.npz", state_sha256=sha,
                             available_channels=sorted(set(CHANNELS) - {"contact"})))
    if corpus == bt.HOW2SIGN:
        r["signer_component"] = component
    return r


def test_val_c00_crossdomain_and_selection_records_are_refused_in_training():
    def role(r):
        return dict(r, _role=bt.role_of(r, r["corpus"]))
    bt.check_loadable_bt(role(record("a")), training=True)
    for bad in (record("v", partition="val", component="c01"), record("s", component="c04"),
                record("f", corpus="2m-flores-asl-v1", partition="crossdomain"),
                record("h", corpus="asl-stem-wiki-v1", partition="holdout"),
                record("y", corpus="youtube-asl-v1", partition="monitor")):
        bt.check_loadable_bt(role(bad), training=False)                  # readable by the gates / selection
        with pytest.raises(RuntimeError):
            bt.check_loadable_bt(role(bad), training=True)
    with pytest.raises(RuntimeError):                                    # signer 9: never, not even its role
        bt.role_of(record("n", partition="train", component="c00"), bt.HOW2SIGN)
    with pytest.raises(RuntimeError):
        bt.check_loadable_bt(dict(record("n", component="c00"), _role=bt.TRAIN), training=False)
    with pytest.raises(RuntimeError):                                    # a forged role is caught
        bt.check_loadable_bt(dict(record("v", partition="val", component="c01"), _role=bt.TRAIN), training=True)
    assert bt.role_of(record("t", partition="test", component="c00"), bt.HOW2SIGN) is None
    assert bt.role_of(dict(record("o", partition="heldout_official"), signer_component=None), bt.HOW2SIGN) is None
    from signtranslator.data.state_text_dataset import StateTextDataset
    config = config_for(h2s_state(0))
    with pytest.raises(RuntimeError):
        StateTextDataset([role(record("v", partition="val", component="c01"))], config, cache=None, fetch=None,
                         training=True)
    with pytest.raises(ValueError):                                      # no control input is ever trained on
        StateTextDataset([role(record("a"))], config, cache=None, fetch=None, training=True, blank=True)


def mixed_manifest(n_monitor=6):
    h = [record(f"t{k}") for k in range(8)] + [record(f"c4-{k}", component="c04") for k in range(3)]
    h += [record(f"v{k}", partition="val", component=c, text=f"val sentence number {k} about cooking pasta")
          for k, c in enumerate(("c01", "c03", "c05"))]
    h += [record("x0", partition="test", component="c00", text="a test sentence about the weather in spain"),
          dict(record("o0", partition="heldout_official", text="an official held out sentence on gardening"),
               signer_component=None)]
    h.append(record("t-dup", text="val sentence number 1 about cooking pasta"))          # an evaluation sentence
    h.append(record("t-near", text="Val sentence number 2 about cooking pasta today"))   # near one
    h.append(record("t-sel", text=h[8]["text"]))                                          # a selection sentence
    y = [record(f"y{k}", corpus="youtube-asl-v1") for k in range(5)]
    y += [record(f"m{k}", corpus="youtube-asl-v1", partition="monitor") for k in range(n_monitor)]
    y.append(record("asl0", corpus="asllvd-text-v1"))
    return dict(records=h + y)


def test_assemble_assigns_roles_drops_evaluation_text_and_samples_the_monitor():
    spec = bt.CorpusSpec("mix", "0" * 64, 2.)
    records, summary = bt.assemble([(mixed_manifest(), spec)], monitor_clips=4)
    roles = {r["sign"]: r["_role"] for r in records}
    assert all(roles[f"t{k}"] == bt.TRAIN for k in range(8))
    assert all(roles[f"c4-{k}"] == bt.SELECT_H2S for k in range(3))
    assert roles["v0"] == bt.VAL_H2S and "x0" not in roles and "o0" not in roles and "asl0" not in roles
    assert sum(r == bt.SELECT_YT for r in roles.values()) == 4
    assert not {"t-dup", "t-near", "t-sel"} & set(roles)
    counts = summary["counts"][bt.HOW2SIGN]
    assert counts["drop:text_in_evaluation"] == 1 and counts["drop:near_evaluation:jaccard"] + \
        counts.get("drop:near_evaluation:containment", 0) == 1 and counts["drop:text_in_selection_or_holdout"] == 1
    assert all(r["_weight"] == 2. for r in records)
    again, _ = bt.assemble([(mixed_manifest(), spec)], monitor_clips=4)
    assert [r["sign"] for r in again] == [r["sign"] for r in records] and bt.data_digest(again) == bt.data_digest(records)
    with pytest.raises(ValueError):
        bt.assemble([(mixed_manifest(n_monitor=3), spec)], monitor_clips=4)
    assert bt.parse_corpus_spec("k@" + "a" * 64 + ":3:youtube-asl-v1") == bt.CorpusSpec("k", "a" * 64, 3.,
                                                                                         ("youtube-asl-v1",))
    for bad in ("k", "k@abc", "k@" + "a" * 64 + ":0", "k@" + "a" * 64 + ":1:asllvd-text-v1"):
        with pytest.raises(ValueError):
            bt.parse_corpus_spec(bad)


# ---- controls -----------------------------------------------------------------------------------------------------

def test_controls_are_built_at_the_reference_duration():
    import validate_state_back_translator as v
    refs = [h2s_state(k, frames=n) for k, n in enumerate((20, 31, 26, 40))]
    channels = set(CHANNELS) - {"contact"}
    forms, info = v.build_controls(refs, seed=0, channels=channels)
    for name in ("blank", "static_valid", "mismatched", "shuffled_chunks"):
        assert [len(s.timestamps) for s in forms[name]] == [len(s.timestamps) for s in refs], name
    assert len(forms["blank_median"]) == 1 and len(forms["blank_median"][0].timestamps) == info["median_frames"] == 28
    assert not any(c.valid.any() for s in forms["blank"] for c in s.channels.values())
    perm = info["derangement"]
    assert all(p != i for i, p in enumerate(perm)) and sorted(perm) == list(range(4))
    for i, s in enumerate(forms["mismatched"]):                           # partner's first and last frames
        np.testing.assert_allclose(s.channels["body"].values[0], refs[perm[i]].channels["body"].values[0], atol=1e-6)
    for s, r in zip(forms["static_valid"], refs):
        for name in channels:
            c = s.channels[name]
            assert c.valid.all() and (c.values == c.values[:1]).all()
        np.testing.assert_allclose(s.channels["body"].values[0], r.channels["body"].values[0])
    # a channel never valid in the clip is filled with its neutral value
    gap = replace(refs[0], channels=dict(refs[0].channels, left_gaze=blank(refs[0]).channels["left_gaze"]))
    sv = v.static_valid(gap, channels)
    assert sv.channels["left_gaze"].valid.all() and np.allclose(sv.channels["left_gaze"].values, [0., 0., 1.])
    shuffled = forms["shuffled_chunks"][3]
    assert sorted(map(tuple, shuffled.channels["face"].values[:, :, 0].round(9))) == \
        sorted(map(tuple, refs[3].channels["face"].values[:, :, 0].round(9)))


def test_paired_gap_kendall_and_gates():
    import validate_state_back_translator as v
    from compare_pilot_lexicon import paired_bleu_difference
    refs = [f"the cat sat on the mat number {k}" for k in range(30)]
    real = [r if k % 3 else "a cat" for k, r in enumerate(refs)]
    control = ["the dog" for _ in refs]
    gap, ci = v.paired_gap(real, {"blank": control}, refs)
    d, ci2 = paired_bleu_difference(real, control, refs)
    assert abs(gap - d) < 1e-9 and np.allclose(ci, ci2)
    gap2, _ = v.paired_gap(real, {"blank": control, "same": real}, refs)
    assert gap2 == 0.
    assert v.kendall_tau([1, 2, 3], [1, 2, 3]) == 1. and v.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.
    assert v.gate_g1(dict(bleu=6.3, ci95=[5., 7.]))["pass"]
    assert not v.gate_g1(dict(bleu=6.3, ci95=[4.1, 7.]))["pass"] and not v.gate_g1(dict(bleu=6.1, ci95=[5., 7.]))["pass"]
    assert v.gate_g2(4.6, (2.3, 6.))["pass"] and not v.gate_g2(4.5, (3., 6.))["pass"]
    assert not v.gate_g2(4.6, (2.2, 6.))["pass"]
    arms = {a: dict(difference=-1., ci95=[-2., -.1]) for a in v.ARMS}
    real_row, trip = dict(bleu=6.), dict(bleu=5.)
    assert v.gate_g3(arms, trip, real_row)["pass"]
    assert not v.gate_g3(dict(arms, R16=dict(difference=-.1, ci95=[-.5, .2])), trip, real_row)["pass"]
    assert not v.gate_g3(arms, dict(bleu=4.7), real_row)["pass"] and v.gate_g3(None, trip, real_row)["pass"] is None
    flores = dict(real=dict(ci95=[3., 5.]), blank=dict(ci95=[1., 2.]), mismatched=dict(ci95=[2., 3.5]),
                  real_minus_blank=dict(difference=2., ci95=[1., 3.]),
                  real_minus_mismatched=dict(difference=1., ci95=[.2, 2.]))
    g4 = v.gate_g4(flores)
    assert g4["pass"] and g4["unpaired_separation"] == dict(blank=True, mismatched=False)
    assert v.gate_g4(None)["pass"] is None
    groups = {g: dict(real_minus_blank=dict(difference=1., ci95=[.1, 2.])) for g in v.GROUPS}
    assert v.gate_g6(groups)["pass"]
    assert not v.gate_g6(dict(groups, c05=dict(real_minus_blank=dict(difference=.1, ci95=[-.2, .4]))))["pass"]


def test_g5_needs_reproduction_and_keeps_every_resolved_sign():
    import validate_state_back_translator as v
    refs = [f"put the pan on the stove and wait {k} minutes" for k in range(40)]
    good = list(refs)
    mid = [r if k % 2 else "put the pan" for k, r in enumerate(refs)]
    bad = ["hello" for _ in refs]
    v1 = dict(real=good, W14b=mid, R16=mid, R16C=bad, composition=mid, blank=bad)
    forms = {f: dict(v1_stored=h, v2=h, reproduced=True) for f, h in v1.items()}
    reread = dict(references=refs, forms=forms)
    g5 = v.gate_g5(reread, refs)
    o = g5["orderings"]
    assert g5["pass"] and o["bleu_v1"] == o["bleu_v2"] and o["kendall_tau"] == round(v.kendall_tau(o["bleu_v1"],
                                                                                                   o["bleu_v1"]), 4)
    flipped = dict(forms, R16C=dict(v1_stored=bad, v2=good, reproduced=True))       # v2 reverses a resolved order
    assert not v.gate_g5(dict(references=refs, forms=flipped), refs)["pass"]
    unrepro = dict(forms, W14b=dict(forms["W14b"], reproduced=False))
    assert v.gate_g5(dict(references=refs, forms=unrepro), refs)["pass"] is False
    assert v.gate_g5(dict(references=refs, forms={k: x for k, x in forms.items() if k != "composition"}),
                     refs)["pass"] is None


# ---- re-reading the arms ------------------------------------------------------------------------------------------

def test_reread_regenerates_with_the_pilots_blocks_and_seeds_and_checks_reproduction():
    import reread_arms as rr
    calls = []

    class Sample:
        @staticmethod
        def batch(texts, fps, seconds, guidance, seed):
            calls.append((len(texts), fps, guidance, seed))
            return [f"{t}|{seed}" for t in texts]
    out = rr.regenerate(Sample, [f"t{k}" for k in range(70)], [1.] * 70, 1.5, lambda o: o + "!")
    assert calls == [(32, 24., 1.5, 1000), (32, 24., 1.5, 1032), (6, 24., 1.5, 1064)]
    assert out[33] == "t33|1032!"
    assert rr.reproduction(["a", "b"], ["a", "b"])["reproduced"]
    r = rr.reproduction(["a", "c"], ["a", "b"])
    assert not r["reproduced"] and r["matches"] == 1 and r["first_mismatches"] == [1]
    texts = ["x", "y"]
    settings = dict(variants=["core_crossfade"], clips=2, translator_key="v1.pt", seed=0)
    stored = dict(settings=settings, references=texts, signs=["a", "b"],
                  forms=dict(core_crossfade=dict(hypotheses=["h1", "h2"])))
    v2 = dict(stored, settings=dict(settings, translator_key="v2.pt"),
              forms=dict(core_crossfade=dict(hypotheses=["g1", "g2"])))
    comp = rr.composition_forms(stored, stored, v2, texts, "v2.pt")
    assert comp["reproduced"] and comp["v2"] == ["g1", "g2"]
    with pytest.raises(SystemExit):                                      # other settings
        rr.composition_forms(stored, stored, dict(v2, settings=dict(v2["settings"], seed=1)), texts, "v2.pt")
    with pytest.raises(SystemExit):                                      # read by another checkpoint
        rr.composition_forms(stored, stored, v2, texts, "other.pt")
    assert rr.parse_arms(None) == rr.ARMS and rr.parse_arms(["A=k"]) == {"A": "k"}
    with pytest.raises(SystemExit):
        rr.pilot_settings(dict(plan={"x": 1}, checkpoint_key="k", part_temperature={}, guidance=1, clips=1))


# ---- tiny end-to-end ----------------------------------------------------------------------------------------------

transformers = pytest.importorskip("transformers")


def tiny_build(config, pretrained=True):
    from signtranslator.generation.state_to_text_t5 import StateToT5
    from tests.test_generation_motion_to_text_t5 import CharTokenizer
    t5 = transformers.T5ForConditionalGeneration(transformers.T5Config(
        vocab_size=260, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    return StateToT5(config, t5, CharTokenizer())


def store_with_states(root):
    """A DirectoryStore holding a mixed manifest and the states of its readable records; returns (store, sha, key)."""
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    Path(root).mkdir(parents=True, exist_ok=True)
    store = DirectoryStore(root)
    manifest = mixed_manifest()
    records = []
    for k, r in enumerate(manifest["records"]):
        s = h2s_state(k, frames=12 + k % 7) if r["corpus"] == bt.HOW2SIGN else mono_state(k, frames=12 + k % 5)
        path = Path(root) / "tmp.npz"
        sha = s.save(path)
        store.put(r["continuous"]["state_key"], path.read_bytes())
        path.unlink()
        records.append(dict(r, continuous=dict(r["continuous"], state_sha256=sha, seconds=1.)))
    m = build_corpus_manifest(corpus_id="mix", records=records, inputs={"x": "0" * 64}, split={})
    payload = canonical_json(m)
    store.put("mix.json", payload)
    return store, hashlib.sha256(payload).hexdigest()


def test_a_tiny_end_to_end_run_trains_selects_on_the_gap_and_the_scorer_reads_it(tmp_path, monkeypatch):
    import train_state_to_text_t5 as trainer
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    store, sha = store_with_states(tmp_path / "store")
    fetched = []
    original = type(store).get

    def spy(self, key):
        fetched.append(key)
        return original(self, key)
    monkeypatch.setattr(type(store), "get", spy)
    argv = ["--store", f"dir:{tmp_path / 'store'}", "--corpus", f"mix.json@{sha}:3:{bt.HOW2SIGN}",
            "--corpus", f"mix.json@{sha}:1:youtube-asl-v1", "--template", f"mix.json@{sha}:{bt.HOW2SIGN}",
            "--run", "bt", "--steps", "4", "--batch-size", "2", "--val-every", "2", "--checkpoint-every", "2",
            "--workers", "0", "--warmup", "1", "--d-model", "32", "--heads", "4", "--encoder-layers", "1",
            "--feedforward", "64", "--max-tokens", "16", "--t5-name", "tiny", "--monitor-clips", "4", "--recon-p", "0",
            "--device", "cpu"]
    assert trainer.main(argv, build_model=tiny_build) == 0
    root = tmp_path / "store"
    best = json.loads((root / "training-runs/bt/best.json").read_text())
    assert best["selected_on"] == "pooled/gap" and best["step"] in (2, 4)
    assert set(best["validation"]) == {bt.SELECT_H2S, bt.SELECT_YT, "pooled"}
    assert best["validation"]["pooled"]["clips"] == 3 + 4
    run = json.loads((root / "training-runs/bt/run.json").read_text())
    assert run["data"]["roles"][bt.TRAIN] == 8 + 5 and run["licence"].startswith("Trained on How2Sign")
    never = {f"states/{s}.npz" for s in ("v0", "v1", "v2", "x0", "o0", "t-dup", "t-near", "t-sel", "asl0")}
    assert not never & set(fetched), never & set(fetched)                # val, test, c00, dropped: never read
    from signtranslator.generation.state_to_text_t5 import load_checkpoint, load_model
    payload = (root / best["checkpoint_key"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == best["checkpoint_sha256"]
    saved = load_checkpoint(payload)                                     # weights_only: no pickled objects
    assert saved["kind"] == "state_to_text_t5" and saved["licence"] == run["licence"]
    model, _ = load_model(payload, expected_sha256=best["checkpoint_sha256"], build_model=tiny_build)
    with pytest.raises(ValueError):
        load_model(payload, expected_sha256="0" * 64, build_model=tiny_build)
    # the scorer: load_translator dispatches on the kind, translate reads states through featurise
    import score_back_translation as sbt
    import signtranslator.generation.state_to_text_t5 as st
    monkeypatch.setattr(st, "build", tiny_build)
    loaded, kind = sbt.load_translator(store, best["checkpoint_key"], "f" * 64)
    assert kind == "states"
    out = sbt.translate(loaded, kind, [h2s_state(1, 15), mono_state(2, 9), blank(h2s_state(3, 11))], beam=1)
    assert len(out) == 3 and all(isinstance(t, str) for t in out)
    assert out == sbt.translate(model, kind, [h2s_state(1, 15), mono_state(2, 9), blank(h2s_state(3, 11))], beam=1)
    # resume refuses other data
    argv2 = [a if a != f"mix.json@{sha}:3:{bt.HOW2SIGN}" else f"mix.json@{sha}:2:{bt.HOW2SIGN}" for a in argv]
    with pytest.raises(SystemExit):
        trainer.main(argv2 + ["--steps", "6"], build_model=tiny_build)


def test_the_dataset_serves_round_trip_copies_under_the_same_text_and_augments(tmp_path):
    from signtranslator.data.state_text_dataset import StateTextDataset
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    store = DirectoryStore(tmp_path)
    real, copy = h2s_state(1, 20), h2s_state(9, 20)
    keys = {}
    for name, s in (("real", real), ("copy", copy)):
        p = tmp_path / f"{name}.npz"
        keys[name] = s.save(p)
    r = dict(record("t0", key="real.npz", sha=keys["real"]), _role=bt.TRAIN, _weight=1.,
             _recon=dict(state_key="copy.npz", state_sha256=keys["copy"]))
    config = config_for(real)
    cache = ContentCache(tmp_path / "c", 2 ** 26)
    always = StateTextDataset([r], config, cache=cache, fetch=store.get, training=True, recon_p=1.)
    never = StateTextDataset([r], config, cache=cache, fetch=store.get, training=True, recon_p=0.)
    f1, t1 = always[0]
    f0, t0 = never[0]
    assert t0 == t1 == r["text"]
    np.testing.assert_allclose(f1.numpy(), state_features(copy, config.motion))
    np.testing.assert_allclose(f0.numpy(), state_features(real, config.motion))
    aug = StateTextDataset([r], config, cache=cache, fetch=store.get, training=True, augment=True,
                           validity_dropout=1., face_dropout=0.)
    lengths = {len(aug[0][0]) for _ in range(8)}
    assert len(lengths) > 1                                              # speed and trim vary the length
    sel = StateTextDataset([dict(r, _role=bt.TRAIN)], config, cache=cache, fetch=store.get, training=False, blank=True)
    assert not sel[0][0].numpy().any() and len(sel[0][0]) == 20


# ---- the v1 paths stay as they were -------------------------------------------------------------------------------

from tests.test_pilot_content_check import DEFAULT_REPORT_SHA256, pilot  # noqa: E402,F401  (the pilot fixture)

def test_translator_run_defaults_to_v1_everywhere():
    import pilot_content_check as pcc
    import round15_compose_eval as ce
    assert pcc.TRANSLATOR_RUN == ce.TRANSLATOR_RUN == "back-translator-kp2d-uwb-ft-v1"
    import tokenizer_fidelity_check as tfc
    tfc.check_translator_kind("keypoints"), tfc.check_translator_kind("states")
    with pytest.raises(SystemExit):
        tfc.check_translator_kind(None)


def test_the_pilot_reads_another_translator_run_only_when_asked(pilot, monkeypatch):  # noqa: F811
    import importlib
    seen = []
    sbt = importlib.import_module("score_back_translation")
    store, run = pilot
    real_load = sbt.load_translator

    def load(store_, key, sha):
        seen.append(key)
        return real_load(store_, key, sha)
    monkeypatch.setattr(sbt, "load_translator", load)
    assert hashlib.sha256(run()).hexdigest() == DEFAULT_REPORT_SHA256
    store.put("training-runs/back-translator-state-t5-v1/best.json", json.dumps(dict(checkpoint_key="v2.pt")).encode())
    report = json.loads(run("--translator-run", "back-translator-state-t5-v1", output="reports/v2.json"))
    assert seen == ["bt.pt", "v2.pt"] and report["translator_run"] == "back-translator-state-t5-v1"
    default = json.loads(store.get("reports/pilot.json"))
    assert set(report) - set(default) == {"translator_run"}


# ---- manifests of extracted video corpora ---------------------------------------------------------------------------

def test_video_corpus_manifest_holds_out_groups_and_marks_crossdomain(tmp_path):
    import build_video_corpus_manifest as b
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    for g in ("P0", "P1", "P2", "P3", "P4"):
        for k in range(3):
            rep = dict(base=f"{g}/v{k}", status="converted", state_key=f"x/states/{g}/v{k}.npz",
                       state_sha256=hashlib.sha256(f"{g}{k}".encode()).hexdigest(), seconds=4., hand_coverage=.9,
                       face_coverage=1., available_channels=["body", "left_hand", "right_hand"],
                       **{"sentence text": f"Rubber is refined in step {k} by {g}.", "participant": g})
            store.put(f"x/videos/{g}/v{k}.json", json.dumps(rep).encode())
    store.put("x/videos/P0/bad.json", json.dumps(dict(base="P0/bad", status="refused")).encode())
    assert b.main(["--store", f"dir:{tmp_path}", "--prefix", "x", "--corpus-id", "asl-stem-wiki-v1",
                   "--manifest-key", "m.json", "--text-field", "sentence text", "--holdout-count", "2",
                   "--licence", "MSR"]) == 0
    m = json.loads(store.get("m.json"))
    held = m["holdout_groups"]
    assert len(held) == 2 and held == sorted(b.group_order("asl-stem-wiki-v1", ["P0", "P1", "P2", "P3", "P4"])[:2])
    assert {r["partition"] for r in m["records"] if r["source"]["group"] in held} == {"holdout"}
    assert sum(r["partition"] == "train" for r in m["records"]) == 9 and m["filters"]["reasons"]["video_refused"] == 1
    roles = {bt.role_of(r, "asl-stem-wiki-v1") for r in m["records"]}
    assert roles == {bt.TRAIN, bt.HOLDOUT_STEM}
    with pytest.raises(SystemExit):                                      # written once
        b.main(["--store", f"dir:{tmp_path}", "--prefix", "x", "--corpus-id", "asl-stem-wiki-v1", "--manifest-key",
                "m.json", "--text-field", "sentence text", "--licence", "MSR"])
    assert b.main(["--store", f"dir:{tmp_path}", "--prefix", "x", "--corpus-id", "2m-flores-asl-v1",
                   "--manifest-key", "f.json", "--text-field", "sentence text", "--crossdomain", "--licence", "CC"]) == 0
    f = json.loads(store.get("f.json"))
    assert {r["partition"] for r in f["records"]} == {"crossdomain"}
    assert {bt.role_of(r, "2m-flores-asl-v1") for r in f["records"]} == {bt.CROSSDOMAIN}


def test_recon_jobs_can_be_limited_to_training_records_of_named_corpora():
    import build_recon_states as brs
    manifest = mixed_manifest()
    jobs, counts = brs.jobs_of([manifest], partitions=("train",), corpora=(bt.HOW2SIGN, "youtube-asl-v1"))
    signs = {j["sign"] for j in jobs}
    assert "t0" in signs and "y0" in signs and "c4-0" in signs
    assert not {"v0", "m0", "asl0", "x0"} & signs
    assert counts["asllvd-text-v1:not_read:corpus"] == 1


# ---- the acceptance and the re-read, end to end on a directory store ------------------------------------------------

def save_state(store, root, key, state):
    path = Path(root) / "tmp-state.npz"
    sha = state.save(path)
    store.put(key, path.read_bytes())
    path.unlink()
    return sha


def acceptance_store(tmp_path, monkeypatch):
    """A store with a pilot-like mix (one c02 train record, two val clips per signer group, a c00 test record), a
    2M-Flores cross-domain manifest, v1's best.json and a tiny instrument v2 checkpoint; returns (store, pins)."""
    import io
    import signtranslator.generation.state_to_text_t5 as st
    from signtranslator.data.text_motion_dataset import DirectoryStore
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json
    root = tmp_path / "store"
    root.mkdir()
    store = DirectoryStore(root)
    sentences = ["you add the flour and mix it well", "then you put it in the oven for an hour",
                 "this is how you hold the brush", "paint the edge of the canvas first",
                 "now we tie the knot around the hook", "pull the line tight and you are done"]
    records = [record("t0", key="s/t0.npz")]
    for k, (text, group) in enumerate(zip(sentences, ("c01", "c01", "c03", "c03", "c05", "c05"))):
        records.append(record(f"v{k}", partition="val", component=group, text=text, key=f"s/v{k}.npz"))
    records.append(record("x0", partition="test", component="c00", text="never read", key="s/x0.npz"))
    for r in records:
        frames = 20 + 3 * int(r["sign"][1:]) if r["sign"] != "t0" else 18
        r["continuous"]["state_sha256"] = save_state(store, root, r["continuous"]["state_key"],
                                                      h2s_state(len(r["sign"]) + frames, frames=frames))
        r["continuous"]["seconds"] = 1.
    payload = canonical_json(build_corpus_manifest(corpus_id="mix", records=records, inputs={"x": "0" * 64}, split={}))
    store.put("mix.json", payload)
    flores = []
    for k in range(4):
        r = record(f"f{k}", corpus="2m-flores-asl-v1", partition="crossdomain", key=f"s/f{k}.npz",
                   text=f"the storm closed the pipeline for {k} days")
        r["continuous"]["state_sha256"] = save_state(store, root, r["continuous"]["state_key"], mono_state(k, 16 + k))
        flores.append(r)
    fpayload = canonical_json(build_corpus_manifest(corpus_id="2m-flores-asl-v1", records=flores, inputs={"x": "0" * 64},
                                                   split={}))
    store.put("flores.json", fpayload)
    store.put("training-runs/back-translator-kp2d-uwb-ft-v1/best.json", json.dumps(dict(checkpoint_key="bt.pt")).encode())
    torch.manual_seed(0)
    config = config_for(h2s_state(0), d_model=32, heads=4, encoder_layers=1, feedforward=64, max_tokens=16,
                        t5_name="tiny")
    model = tiny_build(config)
    buffer = io.BytesIO()
    torch.save(dict(kind=st.KIND, config=config.to_dict(), config_sha256=config.sha256(), model=model.state_dict(),
                    run="tiny", step=1, licence=st.LICENCE), buffer)
    store.put("v2.pt", buffer.getvalue())
    monkeypatch.setattr(st, "build", tiny_build)
    return store, dict(mix=hashlib.sha256(payload).hexdigest(), flores=hashlib.sha256(fpayload).hexdigest(),
                       v2=hashlib.sha256(buffer.getvalue()).hexdigest(), texts=sentences)


def test_the_acceptance_reads_every_form_and_judges_the_gates(tmp_path, monkeypatch):
    import validate_state_back_translator as v
    store, pins = acceptance_store(tmp_path, monkeypatch)
    monkeypatch.setattr(v, "round_trips", lambda store, refs, *a, **k: list(refs))        # identity round trip
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    argv = ["--store", f"dir:{tmp_path / 'store'}", "--translator-key", "v2.pt", "--translator-sha256", pins["v2"],
            "--mix-key", "mix.json", "--mix-sha256", pins["mix"], "--clips", "6", "--beam", "1", "--device", "cpu",
            "--flores-manifest", f"flores.json@{pins['flores']}", "--output-key", "acc.json"]
    code = v.main(argv)
    report = json.loads(store.get("acc.json"))
    assert code == 4 and report["pass"] is False and report["verdicts"]["G1"] is False     # an untrained model
    assert report["verdicts"]["G5"] is None and report["gates"]["G3"]["arms"] is None
    assert sorted(report["references"]) == sorted(pins["texts"])
    assert set(report["forms"]) == {"real", "blank", "blank_median", "static_valid", "mismatched", "shuffled_chunks",
                                    "round_trip"}
    assert set(report["gates"]["G6"]["groups"]) == {"c01", "c03", "c05"}
    assert report["flores"]["clips"] == 4 and set(report["gates"]["G4"]["checks"]) == {"blank", "mismatched"}
    assert report["controls"]["median_frames"] in range(20, 40)
    with pytest.raises(SystemExit):                                       # the checkpoint is pinned
        v.main([a if a != pins["v2"] else "0" * 64 for a in argv[:-1]] + ["other.json"])


def test_reread_reproduces_v1_then_reads_v2_and_feeds_g3_and_g5(tmp_path, monkeypatch):
    import importlib
    import reread_arms as rr
    import validate_state_back_translator as v
    store, pins = acceptance_store(tmp_path, monkeypatch)
    ett = importlib.import_module("evaluate_text_to_motion")
    sbt = importlib.import_module("score_back_translation")
    import pilot_content_check as pcc
    from train_state_to_text_t5 import load_store_manifest
    val = pcc.validation_sentences(load_store_manifest(store, "mix.json", pins["mix"]), 6)
    texts = [r["text"] for r in val]

    class Sample:
        def __init__(self, key):
            self.key = key

        def batch(self, batch_texts, fps, seconds, guidance, seed):
            return [replace(h2s_state(seed + i, frames=int(round(s * fps))), sample_id=f"{self.key}|{t}")
                    for i, (t, s) in enumerate(zip(batch_texts, seconds))]
    monkeypatch.setattr(ett, "load_generator", lambda store, key, sha, device, temperature=None: (None, (), Sample(key)))
    monkeypatch.setattr(sbt, "load_translator", lambda store, key, sha: (torch.nn.Identity(), "kp"))
    original = sbt.translate

    def translate(model, kind, states, **kw):
        if kind == "states":
            return original(model, kind, states, **kw)
        return [s.sample_id if any(c.valid.any() for c in s.channels.values()) else "blank" for s in states]
    monkeypatch.setattr(sbt, "translate", translate)
    for name, ckpt in (("w", "w14b.pt"), ("r", "r16.pt"), ("c", "r16c.pt")):
        gen = [f"{ckpt}|{t}" for t in texts]
        if name == "c":
            gen[2] = "something else"                                    # R16C's stored read is not reproduced
        store.put(f"pilot-{name}.json", json.dumps(dict(
            checkpoint_key=ckpt, part_temperature={"body": 1.1}, guidance=1.5, clips=6,
            meaning=dict(references=texts, generated=dict(hypotheses=gen),
                         real=dict(hypotheses=[f"s{r['sign'][1:]}" for r in val]),
                         blank=dict(hypotheses=["blank"] * 6)))).encode())
    monkeypatch.setattr(ett, "load_state", lambda store_, key, sha: replace(
        h2s_state(int(key.split("/v")[1][0]), frames=20), sample_id=f"s{key.split('/v')[1][0]}"))
    code = rr.main(["--store", f"dir:{tmp_path / 'store'}", "--translator-key", "v2.pt", "--translator-sha256", pins["v2"],
                    "--mix-key", "mix.json", "--mix-sha256", pins["mix"], "--arm", "W14b=pilot-w.json",
                    "--arm", "R16=pilot-r.json", "--arm", "R16C=pilot-c.json", "--composition-stored", "",
                    "--device", "cpu", "--output-key", "reread.json"])
    assert code == 0
    reread = json.loads(store.get("reread.json"))
    assert reread["reproduced"]["W14b"] and reread["reproduced"]["R16"] and not reread["reproduced"]["R16C"]
    assert reread["forms"]["R16C"]["first_mismatches"] == [2] and reread["missing"] == {
        "composition": "not read (--limit or no --composition-stored)"}
    assert reread["forms"]["blank"]["reproduced"] and reread["forms"]["real"]["reproduced"]
    assert all(len(reread["forms"][f]["v2"]) == 6 for f in ("W14b", "R16", "R16C", "real", "blank"))
    g5 = v.gate_g5(reread, texts)
    assert g5["pass"] is False and g5["missing_forms"] == ["composition"]
