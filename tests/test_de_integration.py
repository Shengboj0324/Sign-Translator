"""Doc-10 stage 10h: datasheet + end-to-end pipeline integration + cycle stress."""

import numpy as np
import pytest
import torch

from signtranslator.data_engineering import (
    ConsentState, Sample, validate_sample, gate_download, ProvenanceChain,
    triangulate_dlt, triangulation_confidence, weighted_reprojection_residual,
    average_hash, hamming_distance, per_tier_kappa, grouped_split,
    certify_no_group_leakage, Window, certify_window_split_consistency,
    apply_withdrawal, UsagePolicy, gate_action, infer_sensitive_trait,
    SensitiveInferenceError, Datasheet, PreprocessingManifest,
)
from signtranslator.pose.camera import PerspectiveCamera

USES = ("research",)


# ---- datasheet --------------------------------------------------------------
def _full_datasheet():
    return Datasheet(
        motivation="ASL translation research", composition="continuous + isolated",
        collection="consented, multi-view capture", preprocessing="triangulate+clean",
        uses="SLT training/eval", distribution="not redistributed (licensed)",
        maintenance="versioned; withdrawal honored",
        deaf_annotator_credits=("annotatorA", "annotatorB"),
    )


def test_datasheet_incomplete_until_all_sections_and_credits():
    ds = Datasheet(motivation="x")
    assert "composition" in ds.missing_sections() and not ds.is_complete()
    full = _full_datasheet()
    assert full.missing_sections() == [] and full.is_complete()


def test_datasheet_requires_deaf_credits():
    ds = _full_datasheet()
    ds.deaf_annotator_credits = ()
    assert not ds.is_complete()          # governance: credit is mandatory


def test_manifest_carries_and_verifies_provenance_root():
    chain = ProvenanceChain()
    chain.append("download", {"n": 1})
    chain.append("triangulate", {"j": 25})
    man = PreprocessingManifest.from_chain(chain)
    assert man.verify() and man.step_names() == ["download", "triangulate"]
    man.provenance_root = "deadbeef"     # tamper
    assert not man.verify()


# ---- end-to-end pipeline ----------------------------------------------------
def _cam(eye):
    return PerspectiveCamera.look_at(500., 500., 320., 240., eye, (0., 0., 0.),
                                     dtype=torch.float64)


def test_full_pipeline_gate_to_datasheet():
    # 1) gate before download
    assert gate_download("CC-BY-NC-4.0", ConsentState.GRANTED, "research", USES).allowed

    # 2) provenance chain over steps
    chain = ProvenanceChain()
    chain.append("acquire", {"uri": "rec1"})

    # 3) triangulate a joint from 3 views + confidence
    cams = [_cam(e) for e in [(0, 0, -3), (3, 0, 0), (0, 3, -0.5)]]
    X = torch.tensor([0.1, 0.2, -0.05], dtype=torch.float64)
    obs = torch.stack([_c.project(X)[0] for _c in cams])
    Xhat = triangulate_dlt(cams, obs, torch.tensor([0.9, 0.8, 0.95]))
    assert torch.allclose(Xhat, X, atol=1e-8)
    resid = torch.stack([torch.linalg.norm(_c.project(Xhat)[0] - o)
                         for _c, o in zip(cams, obs)])
    conf3d = triangulation_confidence(resid, torch.tensor([0.9, 0.8, 0.95]))
    assert 0.0 <= float(conf3d) <= 1.0
    chain.append("triangulate", {"conf": round(float(conf3d), 6)})

    # 4) build validated samples across signers/sources
    samples = []
    for k in range(12):
        s = Sample(
            sample_id=f"s{k}", source_id=f"rec{k % 3}",
            signer_id_hash=f"g{k % 4}", target_language="ASL", license="L",
            consent=ConsentState.GRANTED, intended_use="research",
            smplx_version="1.1", provenance=chain.root, split="train",
            confidence_3d=float(conf3d),
        )
        assert validate_sample(s) == []
        samples.append(s)

    # 5) dedup — a duplicated frame is caught
    img = np.random.default_rng(0).random((32, 32))
    assert hamming_distance(average_hash(img), average_hash(img.copy())) == 0

    # 6) per-tier agreement
    k = per_tier_kappa({"gloss": ([0, 1, 0, 1], [0, 1, 0, 1])})
    assert abs(k["gloss"] - 1.0) < 1e-9

    # 7) leakage-certified grouped split + window inheritance
    assign = grouped_split(samples, (0.6, 0.2, 0.2), seed=0)
    assert certify_no_group_leakage(samples, assign).certified
    windows = [Window(i, 0, 16) for i in range(len(samples))]
    assert certify_window_split_consistency(windows, assign)

    # 8) governance: withdrawal + policy + non-inference
    kept = apply_withdrawal(samples, "g0")
    assert all(s.signer_id_hash != "g0" for s in kept)
    assert gate_action(UsagePolicy(), "commercial_use") is False
    with pytest.raises(SensitiveInferenceError):
        infer_sensitive_trait(samples[0], "race")

    # 9) datasheet complete + manifest verifies
    man = PreprocessingManifest.from_chain(chain)
    assert man.verify() and _full_datasheet().is_complete()


def test_cycle_stress_repeated_pipeline_is_deterministic():
    samples = [Sample(f"s{k}", f"rec{k % 2}", f"g{k % 3}", "ASL", "L",
                      ConsentState.GRANTED, "research", "1.1", "p", "train")
               for k in range(30)]
    a = grouped_split(samples, seed=11)
    b = grouped_split(samples, seed=11)
    assert a == b
    assert certify_no_group_leakage(samples, a).certified
