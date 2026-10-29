"""Composition knobs: minimum-jerk bridges (boundary values and velocities; the chord gives the linear bridge), space
normalisation (rigid per word: relative hand trajectories kept; targets met), core tightening (never extends), the
How2Sign train statistics script, and the evaluation script with every knob on and off (defaults unchanged)."""
import json

import numpy as np
import pytest

import scripts.round15_compose_eval as ce
import scripts.round15_space_stats as ss
import signtranslator.generation.compositional as comp
import signtranslator.generation.composition_space as cs
from signtranslator.generation import compositional_grammar as cg
from tests.test_compositional_baseline import full_state
from tests.test_generation_signing_space import signer_state
from tests.test_round15_compose_eval import run_args
from tests.test_round15_stage2 import stage2_store  # noqa: F401
from tests.test_round15_compositional import stubbed  # noqa: F401


# ---- minimum-jerk ----------------------------------------------------------------------------------------------------

def test_quintic_meets_the_boundary_values_and_velocities():
    p0, v0, p1, v1 = np.array([0., 1.]), np.array([.3, -.2]), np.array([2., -1.]), np.array([-.1, .4])
    frames = 4000                                                        # dense: finite differences ~ derivatives
    q = cs.quintic(p0, v0, p1, v1, frames)
    assert q.shape == (frames, 2)
    np.testing.assert_allclose(q[0] - p0, v0, atol=2e-3)                 # per-frame velocity at the start
    np.testing.assert_allclose(p1 - q[-1], v1, atol=2e-3)                # and at the end
    np.testing.assert_allclose(q[1] - 2 * q[0] + p0, 0., atol=1e-5)      # zero acceleration at the start
    # the chord's velocity gives the linear bridge exactly (the crossfade's weights)
    T = 6
    lin = cs.quintic(p0, (p1 - p0) / T, p1, (p1 - p0) / T, T - 1)
    w = comp.crossfade_weights(T - 1)[:, None]
    np.testing.assert_allclose(lin, (1 - w) * p0 + w * p1, atol=1e-12)


def test_minimum_jerk_bridge_matches_values_and_velocities():
    a, b = full_state(0, frames=8), full_state(1, frames=8, offset=3.)
    g = cs.minimum_jerk_bridge(a, b, 5)
    assert len(g.timestamps) == 5 and not any(c.observed.any() for c in g.channels.values())
    va, vb = a.channels["root_translation"].values, b.channels["root_translation"].values
    expected = cs.quintic(va[-1], va[-1] - va[-2], vb[0], vb[1] - vb[0], 5)
    np.testing.assert_allclose(g.channels["root_translation"].values, expected, atol=1e-6)
    rot = g.channels["body"].values
    m = cs._matrices(rot)
    np.testing.assert_allclose(np.einsum("...ji,...jk->...ik", m, m), np.broadcast_to(np.eye(3), m.shape), atol=1e-6)
    # with chord velocities it is the linear bridge (translation)
    a2, b2 = full_state(0, frames=2), full_state(0, frames=2)
    lin = comp.bridge(a2, b2, 3).channels["root_translation"].values
    assert lin.shape == (3, 1, 3)


def test_render_minimum_jerk_keeps_the_plan_and_default_is_the_linear_path():
    clips = [full_state(k, frames=12, offset=float(k)) for k in range(3)]
    parts = comp.parts_from_pieces([dict(state=c, lo=2, hi=8) for c in clips])
    plan = cg.plan_frames([.2, .3, .25], .125, 2., "pad", lead_share=.5)
    linear = comp.render(parts, plan)
    explicit = comp.render(parts, plan, bridge_mode="linear")
    jerk = comp.render(parts, plan, bridge_mode="minimum-jerk")
    for name in linear.channels:
        np.testing.assert_array_equal(linear.channels[name].values, explicit.channels[name].values)
    assert len(jerk.timestamps) == len(linear.timestamps) == plan["frames"]
    k = plan["lead"]                                                     # cores are identical, bridges differ
    np.testing.assert_array_equal(jerk.channels["body"].values[k:k + plan["cores"][0]],
                                  linear.channels["body"].values[k:k + plan["cores"][0]])
    gap = slice(k + plan["cores"][0], k + plan["cores"][0] + plan["gaps"][0])
    assert not np.allclose(jerk.channels["root_translation"].values[gap], linear.channels["root_translation"].values[gap])
    with pytest.raises(ValueError):
        comp.render(parts, plan, bridge_mode="cubic")


# ---- space normalisation ---------------------------------------------------------------------------------------------

def pieces(n=3, frames=30):
    return [dict(state=signer_state(10 + k, frames=frames, amplitude=.5), lo=5, hi=24) for k in range(n)]


def test_space_normalise_is_rigid_per_word_and_meets_the_sentence_targets():
    ps = pieces()
    out, info = cs.space_normalise(ps)
    side = info["side"]
    assert side == "Right" and info["scale"] == 1. and set(info["wrist_speeds"]) == {"Left", "Right"}
    dirs = []
    for p, q in zip(ps, out):
        w0, ok0 = cs.wrist_from_shoulder(p["state"], side)
        w1, ok1 = cs.wrist_from_shoulder(q["state"], side)
        assert (ok0 == ok1).all()
        d0 = np.linalg.norm(w0[:, None] - w0[None], axis=-1)            # pairwise distances: the trajectory's shape
        d1 = np.linalg.norm(w1[:, None] - w1[None], axis=-1)
        np.testing.assert_allclose(d0, d1, atol=1e-6)
        dirs.append(cs._unit(cs._unit(w1[q["lo"]:q["hi"] + 1]).mean(0)))
    for d in dirs[1:]:                                                    # every word's centroid on the same direction
        np.testing.assert_allclose(d, dirs[0], atol=1e-6)
    # the torso and the head are untouched (no torso correction: the head channels are in the face camera frame)
    for p, q in zip(ps, out):
        np.testing.assert_array_equal(cs.torso(p["state"])[0], cs.torso(q["state"])[0])
        for name in ("head_rotation", "head_translation"):
            np.testing.assert_array_equal(p["state"].channels[name].values, q["state"].channels[name].values)
    # cores, non-body channels and the other arm are untouched
    other = "Left"
    for p, q in zip(ps, out):
        assert (p["lo"], p["hi"]) == (q["lo"], q["hi"])
        np.testing.assert_array_equal(p["state"].channels["right_hand"].values, q["state"].channels["right_hand"].values)
        np.testing.assert_allclose(cs.wrist_from_shoulder(p["state"], other)[0],
                                   cs.wrist_from_shoulder(q["state"], other)[0], atol=1e-9)


def test_space_normalise_scales_the_excursion_to_the_statistics():
    ps = pieces()
    _, base = cs.space_normalise(ps)
    target = base["current_rms_angle"] if base["current_rms_angle"] else .3
    stats = dict(rms_angle=.8 * cs.space_normalise(ps, dict(rms_angle=1., direction={}))[1]["current_rms_angle"],
                 direction={})
    out, info = cs.space_normalise(ps, stats)
    assert info["scale"] == pytest.approx(.8, abs=1e-6) and target is not None
    side = info["side"]
    dirs = np.concatenate([cs._unit(cs.wrist_from_shoulder(q["state"], side)[0][q["lo"]:q["hi"] + 1]) for q in out])
    m = cs._unit(dirs.mean(0))
    assert cs.rms_angle(dirs, m) == pytest.approx(stats["rms_angle"], rel=.05)
    with pytest.raises(ValueError):
        cs.space_normalise(ps, None, target="how2sign")


def test_space_normalise_how2sign_target():
    ps = pieces()
    target = cs._unit(np.array([.1, -.6, .8]))
    stats = dict(rms_angle=.2, direction={"Right": target.tolist()})
    out, info = cs.space_normalise(ps, stats, target="how2sign")
    for q in out:                                                          # with the excursion scaled about it
        w, _ = cs.wrist_from_shoulder(q["state"], "Right")
        assert cs._unit(cs._unit(w[q["lo"]:q["hi"] + 1]).mean(0)) @ target > .99


def test_minimal_rotation_and_chordal_mean():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = cs._unit(rng.normal(size=3)), cs._unit(rng.normal(size=3))
        np.testing.assert_allclose(cs.minimal_rotation(a, b) @ a, b, atol=1e-9)
    a = np.array([1., 0, 0])
    np.testing.assert_allclose(cs.minimal_rotation(a, -a) @ a, -a, atol=1e-9)
    R = cs.axis_angle(np.array([0., 0, 1]), np.array([.1, -.1]))
    np.testing.assert_allclose(cs.chordal_mean(R), np.eye(3), atol=1e-9)


# ---- tightening ------------------------------------------------------------------------------------------------------

def test_tightening_never_extends_a_core():
    rng = np.random.default_rng(1)
    for k in range(40):
        s = signer_state(k, frames=int(rng.integers(12, 40)), amplitude=.6)
        T = len(s.timestamps)
        lo = int(rng.integers(0, T // 2))
        hi = int(rng.integers(lo, T))
        for reason in ("entry", "all"):
            w = cs.tighten_window(s, dict(lo=lo, hi=hi, reason=reason))
            assert lo <= w["lo"] <= w["hi"] <= hi
            assert w["tightened"] == (hi - lo) - (w["hi"] - w["lo"]) and w["reason"].startswith(reason)
    s = signer_state(3, frames=30)
    assert cs.tighten_window(s, dict(lo=2, hi=20, reason="annotated")) == dict(lo=2, hi=20, reason="annotated",
                                                                               tightened=0)


# ---- the statistics script and the evaluation script --------------------------------------------------------------------

def test_space_stats_script_reads_how2sign_train_only(tmp_path, stage2_store):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    out = tmp_path / "stats.json"
    assert ss.main(["--mix-key", "mix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
                    "--output-file", str(out)]) == 0
    report = json.loads(out.read_text())
    manifest = json.loads(store.get("mix.json"))
    train = {r["sign"] for r in manifest["records"] if r["corpus"] == ss.CORPUS and r["partition"] == "train"}
    assert set(report["signs"]) <= train and report["partition"] == "train"
    assert report["statistics"]["clips"] >= 1 and report["statistics"]["rms_angle"] >= 0
    with pytest.raises(SystemExit, match="refusing"):
        ss.main(["--mix-key", "mix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
                 "--output-file", str(out)])
    assert report["statistics"]["side"] == "Right" and set(report["statistics"]["direction"]) == {"Right"}
    # the signer-9 guard fires on How2Sign's own field (signer_component c00), on an unknown component and on a
    # source signer_id of 9
    first = next(r for r in manifest["records"] if r["corpus"] == ss.CORPUS and r["partition"] == "train")
    for change in (dict(signer_component="c00"), dict(signer_component=None),
                   dict(source=dict(first.get("source") or {}, signer_id=9))):
        bad = dict(manifest, records=manifest["records"] + [dict(first, sign="s9", **change)])
        with pytest.raises(SystemExit, match="signer 9"):
            ss.train_sample(bad, 10, 0)


def test_compose_eval_knobs(tmp_path, stage2_store):  # noqa: F811
    import hashlib
    store, sha, s1, oracle, argv = stage2_store
    assert ss.main(["--mix-key", "mix.json", "--mix-sha256", sha, "--store", f"dir:{tmp_path / 'store'}",
                    "--output-key", "stats.json"]) == 0
    ssha = hashlib.sha256(store.get("stats.json")).hexdigest()
    base = [a for a in run_args(tmp_path, sha, "--clips", "5") if a not in ("--checkpoint-key", "prior.pt")]
    base[base.index("core_crossfade,oracle_stage2")] = "oracle_stage2"
    assert ce.main(base + ["--output-key", "k/default.json"]) == 0
    assert ce.main(base + ["--bridge", "linear", "--space-normalise", "off", "--output-key", "k/explicit.json"]) == 0
    default, explicit = json.loads(store.get("k/default.json")), json.loads(store.get("k/explicit.json"))
    assert {k: v for k, v in default.items() if k != "seconds"} == {k: v for k, v in explicit.items() if k != "seconds"}
    assert "core_counts" not in default and "space_normalise" not in default["settings"]
    stats = ["--space-stats-key", "stats.json", "--space-stats-sha256", ssha]
    runs = {"space": ["--space-normalise", "sentence", *stats], "jerk": ["--bridge", "minimum-jerk"],
            "tight": ["--tighten-cores"],
            "all": ["--space-normalise", "sentence", *stats, "--bridge", "minimum-jerk", "--tighten-cores"]}
    for name, extra in runs.items():
        assert ce.main(base + extra + ["--output-key", f"k/{name}.json"]) == 0
        report = json.loads(store.get(f"k/{name}.json"))
        assert report["core_counts"]["oracle_stage2"]["cores"] > 0
        for row in report["clips"]:
            if "oracle_stage2" in row["plans"]:
                plan = row["plans"]["oracle_stage2"]
                assert plan["frames"] == row["frames_reference"]
                assert (plan["space"] is not None) == (name in ("space", "all"))
                assert plan["bridge"] == ("minimum-jerk" if name in ("jerk", "all") else "linear")
        again = json.loads(store.get(f"k/{name}.json"))
        assert again == report
    tight = json.loads(store.get("k/tight.json"))["core_counts"]["oracle_stage2"]
    assert set(tight["reasons"]) <= {"annotated", "entry_tightened", "entry_untightened", "all_tightened",
                                     "all_untightened"}
    # deterministic: the same knobs twice give the same hypotheses
    assert ce.main(base + runs["all"] + ["--output-key", "k/all2.json"]) == 0
    assert [r["hypotheses"] for r in json.loads(store.get("k/all.json"))["clips"]] == \
           [r["hypotheses"] for r in json.loads(store.get("k/all2.json"))["clips"]]
    for bad in (["--space-normalise", "sentence"], ["--space-normalise", "sentence", "--space-stats-key", "stats.json",
                                                    "--space-stats-sha256", "0" * 64]):
        with pytest.raises(SystemExit):
            ce.main(base + bad + ["--output-file", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        ce.main([a for a in base if a not in ("--core-margin", "0.1")] + ["--timing", "stage1", "--tighten-cores",
                                                                        "--output-file", str(tmp_path / "y.json")])
