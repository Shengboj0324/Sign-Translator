"""Doc-10 stage 10h: datasheet + end-to-end pipeline integration + cycle stress."""

import numpy as np
import pytest
import torch
import wave
import json
from torch.utils.data import DataLoader

from signtranslator.data_engineering import (
    AuthorizationBasis, ConsentState, DataAuthorization, PersonalityRightsStatus,
    Sample, validate_sample, gate_download, ProvenanceChain,
    triangulate_dlt, triangulation_confidence, weighted_reprojection_residual,
    average_hash, hamming_distance, per_tier_kappa, grouped_split,
    certify_no_group_leakage, Window, certify_window_split_consistency,
    apply_withdrawal, UsagePolicy, gate_action, infer_sensitive_trait,
    SensitiveInferenceError, Datasheet, PreprocessingManifest,
    LandmarkTrack, ExtractedSample, decode_landmark_npz, decode_pcm_wav, decode_video,
    assemble_holistic_track, export_corpus, sha256_file, assess_stage_b_corpus,
)
from signtranslator.data.corpus import (
    SignDataset, collate_corpus, ctc_min_input_length, validate_corpus,
)
from signtranslator.pose.camera import PerspectiveCamera

LANDMARK_PARTS = {"body": [0], "left_hand": [1], "right_hand": [2], "face": [3, 4]}


def _direct_authorization(use="research"):
    return DataAuthorization(
        basis=AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT,
        license_identifier="L", license_url="https://example.test/direct-license",
        licensor="test participant", evidence_uri="direct-consent.txt",
        evidence_sha256="a" * 64, permitted_uses=(use,),
        permitted_actions=("download", "create_derivatives", "model_training"),
        personality_rights=PersonalityRightsStatus.VERIFIED,
    )


def _published_authorization(use="unit-test", evidence_uri="license-evidence.txt",
                             evidence_sha256="b" * 64):
    return DataAuthorization(
        basis=AuthorizationBasis.PUBLISHED_DATASET_LICENSE,
        license_identifier="CC-BY-NC-4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        licensor="test dataset publisher", evidence_uri=evidence_uri,
        evidence_sha256=evidence_sha256, permitted_uses=(use,),
        permitted_actions=("download", "create_derivatives", "model_training"),
        personality_rights=PersonalityRightsStatus.NOT_VERIFIED,
        attribution_notice="Test dataset authors; CC BY-NC 4.0",
        limitations=("No identity, publicity, or privacy permission is asserted.",),
    )


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
    assert gate_download(_direct_authorization(), ConsentState.GRANTED, "research").allowed

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
            sample_id=f"s{k}", source_id=f"g{k % 4}-rec{k % 3}",
            signer_id_hash=f"g{k % 4}", target_language="ASL", license="L",
            consent=ConsentState.GRANTED, intended_use="research",
            smplx_version="1.1", provenance=chain.root, split="train",
            confidence_3d=float(conf3d), authorization=_direct_authorization(),
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
    samples = [Sample(f"s{k}", f"g{k % 3}-rec{k % 2}", f"g{k % 3}", "ASL", "L",
                      ConsentState.GRANTED, "research", "1.1", "p", "train")
               for k in range(30)]
    a = grouped_split(samples, seed=11)
    b = grouped_split(samples, seed=11)
    assert a == b
    assert certify_no_group_leakage(samples, a).certified


# ---- governed-record -> active-loader bridge -------------------------------
def _extracted_records(count=8, *, with_speech=True):
    records = []
    for index in range(count):
        frames = 8 + index % 4
        generator = np.random.default_rng(100 + index)
        values = generator.normal(size=(3, frames, 5)).astype(np.float32)
        validity = np.ones((frames, 5), dtype=np.bool_)
        if index % 3 == 0:
            validity[2, 4] = False
        confidence = generator.uniform(0.5, 1.0, size=(frames, 5)).astype(np.float32)
        confidence[~validity] = 0.0
        track = LandmarkTrack(
            values=values,
            confidence=confidence,
            validity_mask=validity,
            timestamps=np.arange(frames, dtype=np.float64) / 30.0,
        )
        sample = Sample(
            sample_id=f"sample-{index}", source_id=f"source-{index}",
            signer_id_hash=f"signer-{index}", target_language="TEST-SL",
            license="CC-BY-NC-4.0", consent=ConsentState.NOT_DIRECTLY_VERIFIED,
            intended_use="unit-test", smplx_version="not-applicable",
            provenance=f"{10_000 + index:064x}", split="train",
            video_uri=f"video://{index}", authorization=_published_authorization(),
        )
        gloss = ("HELLO", "HELLO") if index % 2 == 0 else ("HELLO", "WORLD")
        source = ("hello", "there", "friend") if index % 2 == 0 else ("goodbye",)
        if with_speech:
            speech = generator.normal(size=(16, 7)).astype(np.float32)
            speech_timestamps = np.arange(16, dtype=np.float64) * 0.01
        else:
            speech = speech_timestamps = None
        records.append(ExtractedSample(
            governance=sample, track=track, gloss_tokens=gloss,
            source_tokens=source, media_sha256=f"{index:064x}",
            extractor_id="test-extractor@sha256:abc", coordinate_system="camera-rh-m",
            speech_features=speech, speech_timestamps=speech_timestamps,
        ))
    return records


def test_exporter_preserves_masks_lengths_labels_and_traceability(tmp_path):
    result = export_corpus(
        _extracted_records(), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS,
        split_ratios=(0.5, 0.25, 0.25), seed=4,
    )
    spec = validate_corpus(result.corpus_dir)
    assert spec.num_concepts == 2
    assert spec.source_token_count == 4
    assert result.spec == spec

    datasets = [SignDataset(result.corpus_dir, split)
                for split in ("train", "val", "test")]
    assert sum(map(len, datasets)) == 8
    for dataset in datasets:
        for item in dataset:
            invalid = ~item["validity_mask"].unsqueeze(0).expand_as(item["pose"])
            assert torch.all(item["pose"][invalid] == 0)
    batch = next(iter(DataLoader(datasets[0], batch_size=len(datasets[0]),
                                 collate_fn=collate_corpus)))
    assert batch["frame_mask"].sum() == batch["motion_lengths"].sum()
    assert batch["validity_mask"].shape == batch["confidence"].shape
    assert torch.all(batch["confidence"][~batch["validity_mask"]] == 0)
    assert batch["src"].shape[1] != batch["gloss_tokens"].shape[1]
    assert batch["speech_input_lengths"].tolist() == [16] * len(datasets[0])
    assert batch["speech_timestamps"].shape[:2] == batch["speech"].shape[:2]
    assert len(batch["sample_ids"]) == len(datasets[0])
    assert "sample-" in (tmp_path / "corpus" / "review.html").read_text()
    manifest = json.loads((tmp_path / "corpus" / "manifest.json").read_text())
    assert manifest["normalization_fit_split"] == "train"
    assert manifest["leakage_certified"] is True
    assert all(record["media_sha256"] for record in manifest["records"])


def test_exporter_and_active_loader_preserve_explicit_2d_motion(tmp_path):
    records = []
    for record in _extracted_records(with_speech=False):
        track = LandmarkTrack(
            values=record.track.values[:2].copy(),
            confidence=record.track.confidence.copy(),
            validity_mask=record.track.validity_mask.copy(),
            timestamps=record.track.timestamps.copy(),
        )
        records.append(ExtractedSample(
            governance=record.governance, track=track,
            gloss_tokens=record.gloss_tokens, source_tokens=record.source_tokens,
            media_sha256=record.media_sha256, extractor_id=record.extractor_id,
            coordinate_system="image-normalized-xy-top-left",
        ))
    result = export_corpus(
        records, tmp_path / "corpus-2d",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS,
        split_ratios=(0.5, 0.25, 0.25), seed=4,
    )
    spec = validate_corpus(result.corpus_dir)
    assert spec.in_channels == 2
    for split in ("train", "val", "test"):
        sample = SignDataset(result.corpus_dir, split)[0]
        assert sample["pose"].shape[0] == 2


def test_exporter_rejects_invalid_confidence_and_overwrite(tmp_path):
    records = _extracted_records(with_speech=False)
    bad = records[0]
    confidence = bad.track.confidence.copy()
    confidence[~bad.track.validity_mask] = 0.5
    records[0] = ExtractedSample(
        governance=bad.governance,
        track=LandmarkTrack(bad.track.values, confidence,
                            bad.track.validity_mask, bad.track.timestamps),
        gloss_tokens=bad.gloss_tokens, source_tokens=bad.source_tokens,
        media_sha256=bad.media_sha256, extractor_id=bad.extractor_id,
        coordinate_system=bad.coordinate_system,
    )
    with pytest.raises(ValueError, match="invalid observations"):
        export_corpus(records, tmp_path / "bad", joint_names=[f"j{i}" for i in range(5)],
                      landmark_parts=LANDMARK_PARTS,
                      split_ratios=(0.5, 0.25, 0.25))

    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "valuable.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="non-empty"):
        export_corpus(_extracted_records(with_speech=False), output,
                      joint_names=[f"j{i}" for i in range(5)],
                      landmark_parts=LANDMARK_PARTS,
                      split_ratios=(0.5, 0.25, 0.25))
    assert (output / "valuable.txt").read_text() == "keep"


def test_exporter_rejects_duplicate_ids_and_inconsistent_source_hashes(tmp_path):
    records = _extracted_records(with_speech=False)
    records[1].governance.sample_id = records[0].governance.sample_id
    with pytest.raises(ValueError, match="globally unique"):
        export_corpus(records, tmp_path / "duplicate", joint_names=[f"j{i}" for i in range(5)],
                      landmark_parts=LANDMARK_PARTS,
                      split_ratios=(0.5, 0.25, 0.25))

    records = _extracted_records(with_speech=False)
    records[1].governance.source_id = records[0].governance.source_id
    with pytest.raises(ValueError, match="inconsistent media"):
        export_corpus(records, tmp_path / "source", joint_names=[f"j{i}" for i in range(5)],
                      landmark_parts=LANDMARK_PARTS,
                      split_ratios=(0.5, 0.25, 0.25))


def test_exact_ctc_feasibility_counts_repeated_labels():
    assert ctc_min_input_length([1, 2, 3]) == 3
    assert ctc_min_input_length([1, 1, 2, 2, 2]) == 8


def test_v2_shard_tampering_is_detected_before_loading(tmp_path):
    result = export_corpus(
        _extracted_records(with_speech=False), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS,
        split_ratios=(0.5, 0.25, 0.25), seed=2,
    )
    shard = tmp_path / "corpus" / "val.npz"
    payload = bytearray(shard.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    shard.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_corpus(result.corpus_dir)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SignDataset(result.corpus_dir, "train")


def test_v2_validation_rejects_semantically_nonzero_padding_even_with_new_hash(tmp_path):
    result = export_corpus(
        _extracted_records(with_speech=False), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS,
        split_ratios=(0.5, 0.25, 0.25), seed=2,
    )
    shard = tmp_path / "corpus" / "train.npz"
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    row = int(np.flatnonzero(arrays["source_lengths"] < arrays["src_concepts"].shape[1])[0])
    arrays["src_concepts"][row, int(arrays["source_lengths"][row])] = 1
    np.savez_compressed(shard, **arrays)
    manifest_path = tmp_path / "corpus" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shard_sha256"]["train.npz"] = sha256_file(shard)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="token padding"):
        validate_corpus(result.corpus_dir)


def _write_test_video(path):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            image = np.full((16, 16, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _records_with_verifiable_local_media(tmp_path):
    template = tmp_path / "template.mp4"
    _write_test_video(template)
    payload = template.read_bytes()
    evidence = tmp_path / "license-evidence.txt"
    evidence.write_text("Immutable test license evidence", encoding="utf-8")
    evidence_digest = sha256_file(evidence)
    records = []
    for index, record in enumerate(_extracted_records(with_speech=False)):
        media = tmp_path / f"source-{index}.mp4"
        media.write_bytes(payload)
        record.governance.video_uri = str(media)
        record.governance.authorization = _published_authorization(
            evidence_uri=str(evidence), evidence_sha256=evidence_digest)
        frames = record.track.values.shape[1]
        track = LandmarkTrack(
            values=record.track.values,
            confidence=record.track.confidence,
            validity_mask=record.track.validity_mask,
            timestamps=np.arange(frames, dtype=np.float64) / 25.0,
        )
        records.append(ExtractedSample(
            governance=record.governance, track=track,
            gloss_tokens=record.gloss_tokens, source_tokens=record.source_tokens,
            media_sha256=sha256_file(media), extractor_id=record.extractor_id,
            coordinate_system=record.coordinate_system,
        ))
    return records


def test_stage_b_gate_fails_closed_without_external_attestations(tmp_path):
    result = export_corpus(
        _records_with_verifiable_local_media(tmp_path), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS, split_ratios=(0.5, 0.25, 0.25), seed=2,
    )
    report = assess_stage_b_corpus(result.corpus_dir)
    assert not report.passed
    failed = {check.name for check in report.checks if not check.passed}
    assert failed == {"dataset_charter", "annotation_agreement", "qualified_visual_review"}
    assert "APPROVED TO PROCEED: NO" in report.summary()


def test_stage_b_gate_accepts_a_complete_machine_verifiable_fixture(tmp_path):
    result = export_corpus(
        _records_with_verifiable_local_media(tmp_path), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS, split_ratios=(0.5, 0.25, 0.25), seed=2,
    )
    root = tmp_path / "corpus"
    (root / "dataset_charter.json").write_text(json.dumps({
        "schema_version": 1,
        "target_language": "TEST-SL",
        "dialect": "test-only",
        "translation_direction": "source tokens to sign motion",
        "task": "continuous sentences",
        "output_representation": "timestamped 3D holistic landmarks",
        "allowed_uses": ["unit-test"],
        "primary_population": "test fixture only",
        "unacceptable_error_definition": "any source or label mismatch",
    }))
    (root / "annotation_agreement.json").write_text(json.dumps({
        "schema_version": 1,
        "tiers": {"gloss": {"kappa": 0.8, "item_count": 8}},
        "uncertainty_and_adjudication": "disagreements retained then adjudicated",
    }))
    manifest = json.loads((root / "manifest.json").read_text())
    (root / "review_attestation.json").write_text(json.dumps({
        "schema_version": 1,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "decision": "approved",
        "reviewer_roles": ["qualified_target_language_signer"],
        "reviewed_stages": ["source_video", "extracted_landmarks", "exported_shard"],
        "reviewed_sample_ids": [record["sample_id"] for record in manifest["records"]],
    }))
    report = assess_stage_b_corpus(result.corpus_dir)
    assert report.passed, report.summary()
    assert "APPROVED TO PROCEED: YES" in report.summary()


def test_stage_b_gate_rejects_tampered_license_evidence(tmp_path):
    result = export_corpus(
        _records_with_verifiable_local_media(tmp_path), tmp_path / "corpus",
        joint_names=[f"joint-{index}" for index in range(5)],
        landmark_parts=LANDMARK_PARTS, split_ratios=(0.5, 0.25, 0.25), seed=2,
    )
    (tmp_path / "license-evidence.txt").write_text("tampered", encoding="utf-8")
    report = assess_stage_b_corpus(result.corpus_dir)
    source_check = next(check for check in report.checks
                        if check.name == "immutable_source_trace")
    assert not source_check.passed
    assert "authorization evidence SHA-256 mismatch" in source_check.detail


def test_timestamped_media_decoders_are_strict(tmp_path):
    wav_path = tmp_path / "tone.wav"
    signal = np.array([0, 1000, -1000, 32767], dtype="<i2")
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(signal.tobytes())
    audio = decode_pcm_wav(wav_path)
    assert audio.samples.shape == (4, 1)
    assert np.allclose(audio.timestamps, np.arange(4) / 8000)

    track_path = tmp_path / "track.npz"
    np.savez(track_path, values=np.zeros((3, 2, 4), dtype=np.float32),
             confidence=np.ones((2, 4), dtype=np.float32),
             validity_mask=np.ones((2, 4), dtype=np.bool_),
             timestamps=np.array([0.0, 0.04]))
    assert decode_landmark_npz(track_path).values.shape == (3, 2, 4)

    video_path = tmp_path / "frames.mp4"
    _write_test_video(video_path)
    video = decode_video(video_path)
    assert video.frames.shape == (12, 16, 16, 3)
    assert np.allclose(video.timestamps, np.arange(12) / 25.0)


def test_holistic_assembly_preserves_dense_parts_and_rejects_clock_drift():
    timestamps = np.array([0.0, 0.04, 0.08])
    def part(joints):
        return LandmarkTrack(
            np.zeros((3, 3, joints), dtype=np.float32),
            np.ones((3, joints), dtype=np.float32),
            np.ones((3, joints), dtype=np.bool_), timestamps.copy())
    combined = assemble_holistic_track({
        "body": part(25), "right_hand": part(21),
        "left_hand": part(21), "face": part(68),
    })
    assert combined.values.shape == (3, 3, 135)
    drifted = part(21)
    drifted.timestamps[1] += 1e-4
    with pytest.raises(ValueError, match="timestamps"):
        assemble_holistic_track({"body": part(25), "right_hand": drifted})
