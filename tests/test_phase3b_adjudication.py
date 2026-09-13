"""Adversarial tests for governed Phase 3B source-to-SIR adjudication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest

from signtranslator.data_engineering.eaf import ingest_eaf_reference
from signtranslator.data_engineering.phase3b_queue import (
    GovernanceFileSpec,
    QueueSelection,
    verify_phase3b_review_queue,
    write_phase3b_review_queue,
)
from signtranslator.grammar.sir import (
    EdgeType,
    EventKind,
    SIREdge,
    SIREvent,
    SIRGraph,
    sir_sha256,
)
from signtranslator.planning.adjudication import (
    AdjudicationOutcome,
    EAFAnnotationBinding,
    EventCorrespondence,
    HumanAdjudication,
    HumanSIRSubmission,
    Phase3BReviewCase,
    SubmissionDecision,
    SubmissionRole,
    WorkflowState,
    audit_phase3b_batch,
    compare_submissions,
    load_eaf_source_catalog,
    load_phase3b_batch_report,
    load_phase3b_case,
    temporal_iou,
    verify_phase3b_batch_report,
)
from signtranslator.planning.supervision import ArtifactKind, GovernedArtifact
from signtranslator.reproducibility import canonical_json_bytes


MEDIA_URL = "file:///publisher/original/sample.mp4"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(kind: ArtifactKind, name: str) -> GovernedArtifact:
    return GovernedArtifact(kind, name, "1.0.0", _digest(name))


def _source() -> EAFAnnotationBinding:
    value = "PUBLISHER-SOURCE-LABEL"
    return EAFAnnotationBinding(
        schema_version=1,
        eaf_manifest_sha256=_digest("manifest"),
        eaf_file_sha256=_digest("eaf"),
        eaf_schema_sha256=_digest("schema"),
        media_descriptor_order=0,
        source_video_sha256=_digest("video"),
        tier_id="ID-gloss",
        source_annotation_id="a1",
        source_annotation_kind="alignable",
        source_order=0,
        source_value=value,
        source_value_sha256=_digest(value),
        begin_time_slot_ref="ts1",
        end_time_slot_ref="ts2",
        begin_ms=0,
        end_ms=2_000,
    )


def _graph(
    *,
    manual_label: int = 10,
    manual_interval: tuple[float, float] = (0.0, 1.0),
    include_nonmanual: bool = True,
) -> SIRGraph:
    events = [
        SIREvent(0, EventKind.MANUAL, manual_label, *manual_interval,
                 referent=1, locus=2),
    ]
    edges: list[SIREdge] = []
    if include_nonmanual:
        events.append(SIREvent(
            1,
            EventKind.NONMANUAL,
            30,
            manual_interval[0],
            manual_interval[1],
        ))
        edges.append(SIREdge(1, 0, EdgeType.SCOPE))
    return SIRGraph(events=events, edges=edges)


def _case(
    *,
    case_id: str = "case-1",
    source: EAFAnnotationBinding | None = None,
    sampling_plan: GovernedArtifact | None = None,
) -> Phase3BReviewCase:
    convention = _artifact(ArtifactKind.ASL_CONVENTION, "asl-convention")
    return Phase3BReviewCase.create_draft(
        case_id=case_id,
        source=_source() if source is None else source,
        convention=convention,
        lexicon=_artifact(ArtifactKind.SIR_LEXICON, "sir-lexicon"),
        lexicon_convention_sha256=convention.sha256,
        annotation_protocol=_artifact(
            ArtifactKind.ANNOTATION_PROTOCOL, "annotation-protocol"),
        review_protocol=_artifact(ArtifactKind.REVIEW_PROTOCOL, "review-protocol"),
        adjudication_protocol=_artifact(
            ArtifactKind.ADJUDICATION_PROTOCOL, "adjudication-protocol"),
        sampling_plan=(
            _artifact(ArtifactKind.SAMPLING_PLAN, "sampling-plan")
            if sampling_plan is None else sampling_plan
        ),
        metrics_preregistration=_artifact(
            ArtifactKind.METRICS_PREREGISTRATION, "metrics-preregistration"),
        creator_pseudonym="case-curator",
        created_at="2026-09-13T09:00:00-07:00",
    )


def _submission(
    case: Phase3BReviewCase,
    role: SubmissionRole,
    *,
    graph: SIRGraph | None = None,
    decision: SubmissionDecision = SubmissionDecision.SIR,
    author: str | None = None,
    submitted_at: str | None = None,
    reason_codes: tuple[str, ...] = (),
) -> HumanSIRSubmission:
    primary = role is SubmissionRole.PRIMARY
    return HumanSIRSubmission.create(
        submission_id="primary-1" if primary else "review-1",
        case_id=case.case_id,
        source=case.source,
        role=role,
        author_pseudonym=author or ("annotator-a" if primary else "reviewer-b"),
        qualification_evidence_sha256=_digest(
            "annotator-qualification" if primary else "reviewer-qualification"),
        independence_evidence_sha256=_digest(
            "annotator-independent-authorship"
            if primary else "reviewer-blind-assignment"),
        attestation_sha256=_digest(
            "annotator-attestation" if primary else "reviewer-attestation"),
        protocol=case.annotation_protocol if primary else case.review_protocol,
        decision=decision,
        submitted_at=submitted_at or (
            "2026-09-13T10:00:00-07:00" if primary
            else "2026-09-13T11:00:00-07:00"),
        graph=graph,
        reason_codes=reason_codes,
    )


def _adjudication(
    case: Phase3BReviewCase,
    primary: HumanSIRSubmission,
    reviewer: HumanSIRSubmission,
    *,
    outcome: AdjudicationOutcome,
    graph: SIRGraph | None,
    adjudicator: str = "adjudicator-c",
    submitted_at: str = "2026-09-13T12:00:00-07:00",
) -> HumanAdjudication:
    return HumanAdjudication.create(
        adjudication_id="adjudication-1",
        case_id=case.case_id,
        source=case.source,
        primary=primary,
        reviewer=reviewer,
        adjudicator_pseudonym=adjudicator,
        qualification_evidence_sha256=_digest("adjudicator-qualification"),
        independence_evidence_sha256=_digest("adjudicator-independence"),
        attestation_sha256=_digest("adjudicator-attestation"),
        protocol=case.adjudication_protocol,
        outcome=outcome,
        submitted_at=submitted_at,
        final_graph=graph,
        reason_codes=("lexical_label_disagreement",),
    )


def _eaf() -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="Corpus Team" DATE="2021-04-01T00:00:00Z"
 VERSION="3.0" FORMAT="3.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
 <LICENSE LICENSE_URL="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</LICENSE>
 <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
  <MEDIA_DESCRIPTOR MEDIA_URL="{MEDIA_URL}" RELATIVE_MEDIA_URL="./sample.mp4"
   MIME_TYPE="video/mp4"/>
 </HEADER>
 <TIME_ORDER>
  <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
  <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="200"/>
 </TIME_ORDER>
 <TIER TIER_ID="ID-gloss" LINGUISTIC_TYPE_REF="source-gloss">
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1"
   TIME_SLOT_REF2="ts2"><ANNOTATION_VALUE>HELLO</ANNOTATION_VALUE>
  </ALIGNABLE_ANNOTATION></ANNOTATION>
 </TIER>
 <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="source-gloss" TIME_ALIGNABLE="true"/>
</ANNOTATION_DOCUMENT>'''.encode("utf-8")


def _write_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _source_bundle(
    tmp_path: Path,
    *,
    eaf_payload: bytes | None = None,
    video_name: str = "sample.mp4",
) -> tuple[Path, bytes]:
    source = tmp_path / "source"
    source.mkdir()
    eaf_path = source / "sample.eaf"
    eaf_path.write_bytes(_eaf() if eaf_payload is None else eaf_payload)
    video = source / video_name
    _write_video(video)
    evidence = source / "license.html"
    evidence.write_text("CC BY-NC-SA evidence", encoding="utf-8")
    result = ingest_eaf_reference(
        eaf_path,
        source,
        license_evidence_path=evidence,
        media_paths={MEDIA_URL: video.name},
    )
    return source, canonical_json_bytes(result.to_manifest())


def _governance_bundle(
    tmp_path: Path,
    source_manifest: bytes,
) -> tuple[Path, tuple[GovernanceFileSpec, ...]]:
    root = tmp_path / "governance"
    root.mkdir()
    specs = []
    for kind in ArtifactKind:
        filename = (
            f"{kind.value}.json"
            if kind is ArtifactKind.SAMPLING_PLAN else f"{kind.value}.md"
        )
        if kind is ArtifactKind.SAMPLING_PLAN:
            (root / filename).write_bytes(canonical_json_bytes({
                "schema_version": 1,
                "kind": "phase3b_exact_source_sampling_plan",
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest).hexdigest(),
                "selections": [{
                    "tier_id": "ID-gloss",
                    "source_annotation_id": "a1",
                    "media_descriptor_order": 0,
                }],
            }))
        else:
            (root / filename).write_text(
                f"# {kind.value}\n\nFrozen test evidence.\n", encoding="utf-8")
        specs.append(GovernanceFileSpec(
            kind=kind,
            artifact_id=f"test-{kind.value}",
            version="1.0.0",
            relative_path=filename,
        ))
    return root, tuple(specs)


def test_eaf_catalog_revalidates_inventory_content_and_source_binding(tmp_path):
    source, manifest = _source_bundle(tmp_path)
    catalog = load_eaf_source_catalog(manifest, source)
    binding = catalog.bind(tier_id="ID-gloss", source_annotation_id="a1")
    assert binding.source_value == "HELLO"
    assert binding.begin_ms == 0
    assert binding.end_ms == 200
    assert binding.source_video_sha256 == hashlib.sha256(
        (source / "sample.mp4").read_bytes()).hexdigest()

    (source / "sample.eaf").write_bytes(_eaf().replace(b"HELLO", b"BYE__"))
    with pytest.raises(RuntimeError, match="differs from the EAF manifest"):
        load_eaf_source_catalog(manifest, source)


def test_eaf_catalog_reports_storage_identity_drift_without_weakening_content_gate(
    tmp_path,
):
    source, manifest = _source_bundle(tmp_path)
    value = json.loads(manifest)
    value["eaf_file"]["device"] += 1
    value["license_evidence_file"]["inode"] += 1
    altered_manifest = canonical_json_bytes(value)
    catalog = load_eaf_source_catalog(altered_manifest, source)
    assert catalog.storage_identity_drift_paths == (
        "sample.eaf", "license.html",
    )
    assert catalog.eaf_file_sha256 == hashlib.sha256(_eaf()).hexdigest()


def test_eaf_catalog_rejects_extra_files_symlinks_and_ambiguous_annotation(tmp_path):
    source, manifest = _source_bundle(tmp_path)
    (source / "unexpected.txt").write_text("unaccounted", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        load_eaf_source_catalog(manifest, source)
    (source / "unexpected.txt").unlink()
    (source / "link").symlink_to(source / "sample.eaf")
    with pytest.raises(ValueError, match="symlinked"):
        load_eaf_source_catalog(manifest, source)
    (source / "link").unlink()
    catalog = load_eaf_source_catalog(manifest, source)
    with pytest.raises(ValueError, match="exactly one"):
        catalog.bind(tier_id="ID-gloss", source_annotation_id="missing")
    with pytest.raises(ValueError, match="descriptor order"):
        catalog.bind(
            tier_id="ID-gloss", source_annotation_id="a1",
            media_descriptor_order=1,
        )


def test_source_binding_preserves_unknown_timing_and_rejects_value_tampering():
    source = replace(
        _source(), begin_ms=None, end_ms=None,
        begin_time_slot_ref="ts-unknown-1", end_time_slot_ref="ts-unknown-2",
    )
    assert EAFAnnotationBinding.from_dict(source.to_dict()) == source
    with pytest.raises(ValueError, match="value hash mismatch"):
        replace(source, source_value="changed")
    with pytest.raises(ValueError, match="positive duration"):
        replace(source, begin_ms=10, end_ms=10)


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ((0.0, 1.0), (0.0, 1.0), 1.0),
        ((0.0, 2.0), (1.0, 3.0), 1.0 / 3.0),
        ((0.0, 1.0), (2.0, 3.0), 0.0),
        ((-2.0, 2.0), (-1.0, 1.0), 0.5),
    ],
)
def test_temporal_iou_exact_cases(first, second, expected):
    assert temporal_iou(first, second) == pytest.approx(expected, abs=1e-15)
    assert temporal_iou(second, first) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    "first,second",
    [
        ((0.0, 0.0), (0.0, 1.0)),
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, float("nan")), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, float("inf"))),
    ],
)
def test_temporal_iou_rejects_invalid_intervals(first, second):
    with pytest.raises(ValueError):
        temporal_iou(first, second)


def test_temporal_iou_has_required_geometric_invariances():
    generator = np.random.default_rng(20260913)
    for _ in range(2_000):
        starts = generator.uniform(-100.0, 100.0, size=2)
        durations = generator.uniform(1e-6, 20.0, size=2)
        first = (float(starts[0]), float(starts[0] + durations[0]))
        second = (float(starts[1]), float(starts[1] + durations[1]))
        result = temporal_iou(first, second)
        assert 0.0 <= result <= 1.0
        assert result == pytest.approx(temporal_iou(second, first), abs=1e-15)

        translation = float(generator.uniform(-1_000.0, 1_000.0))
        translated_first = (first[0] + translation, first[1] + translation)
        translated_second = (second[0] + translation, second[1] + translation)
        assert result == pytest.approx(
            temporal_iou(translated_first, translated_second), abs=1e-12)

        scale = float(generator.uniform(1e-3, 1_000.0))
        scaled_first = (first[0] * scale, first[1] * scale)
        scaled_second = (second[0] * scale, second[1] * scale)
        assert result == pytest.approx(
            temporal_iou(scaled_first, scaled_second), abs=1e-12)


def test_agreement_uses_declared_correspondence_and_reports_each_dimension():
    case = _case()
    primary = _submission(
        case, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    reviewer_graph = SIRGraph(
        events=[
            SIREvent(5, EventKind.MANUAL, 11, 0.2, 1.2, referent=1, locus=3),
            SIREvent(6, EventKind.NONMANUAL, 30, 0.0, 1.2),
        ],
        edges=[SIREdge(6, 5, EdgeType.SCOPE)],
    )
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=reviewer_graph)
    report = compare_submissions(
        primary, reviewer, EventCorrespondence(((0, 5), (1, 6))))
    by_field = {item.field: item for item in report.field_agreement}
    assert by_field["kind"].rate == 1.0
    assert by_field["label"].rate == 0.5
    assert by_field["referent"].rate == 1.0
    assert by_field["locus"].rate == 0.5
    assert report.temporal[0].temporal_iou == pytest.approx(2.0 / 3.0)
    assert report.temporal[0].reviewer_minus_primary_onset == pytest.approx(0.2)
    assert report.temporal[0].reviewer_minus_primary_offset == pytest.approx(0.2)
    assert report.median_temporal_iou == pytest.approx(3.0 / 4.0)
    assert report.comparable_edge_jaccard == 1.0
    assert report.kind_confusion == (("manual", "manual", 1),
                                     ("nonmanual", "nonmanual", 1))
    assert report.label_confusion == ((10, 11, 1), (30, 30, 1))


def test_zero_support_and_zero_edge_union_are_unavailable_not_fake_perfect():
    case = _case()
    primary = _submission(
        case, SubmissionRole.PRIMARY,
        graph=_graph(manual_label=10, include_nonmanual=False))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11, include_nonmanual=False))
    report = compare_submissions(primary, reviewer, EventCorrespondence(()))
    assert report.comparison_available
    assert report.paired_event_count == 0
    assert all(item.rate is None for item in report.field_agreement)
    assert report.median_temporal_iou is None
    assert report.comparable_edge_jaccard is None


def test_abstention_is_not_scored_as_agreement():
    case = _case()
    primary = _submission(
        case, SubmissionRole.PRIMARY, decision=SubmissionDecision.ABSTAIN,
        reason_codes=("insufficient_visual_evidence",))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph())
    report = compare_submissions(primary, reviewer, EventCorrespondence(()))
    assert not report.comparison_available
    assert report.unavailable_reason == "one_submission_abstained"
    assert report.median_temporal_iou is None
    with pytest.raises(ValueError, match="reason"):
        _submission(
            case, SubmissionRole.PRIMARY,
            decision=SubmissionDecision.ABSTAIN,
        )


def test_identical_independent_submissions_finalize_without_adjudication():
    case = _case()
    graph = _graph()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=graph)
    reviewer = _submission(case, SubmissionRole.INDEPENDENT_REVIEWER, graph=graph)
    final = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(())).finalize()
    assert final.state is WorkflowState.ACCEPTED
    assert sir_sha256(final.final_graph()) == sir_sha256(graph)
    assert [event.state for event in final.events] == [
        WorkflowState.DRAFT,
        WorkflowState.PRIMARY_COMPLETE,
        WorkflowState.BLIND_REVIEWED,
        WorkflowState.ACCEPTED,
    ]
    restored = load_phase3b_case(final.canonical_bytes())
    assert restored == final
    assert restored.content_sha256() == final.content_sha256()
    assert not restored.source_training_target_authorized
    assert not restored.commercial_use_authorized
    assert not restored.project_linguistic_validation_complete


def test_disagreement_cannot_finalize_without_third_person_adjudication():
    case = _case()
    primary = _submission(
        case, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11))
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    with pytest.raises(ValueError, match="require adjudication"):
        reviewed.finalize()

    adjudication = _adjudication(
        case, primary, reviewer,
        outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
        graph=primary.graph(),
    )
    final = reviewed.with_adjudication(adjudication).finalize()
    assert final.state is WorkflowState.ACCEPTED
    assert sir_sha256(final.final_graph()) == primary.sir_payload_sha256
    assert [event.sequence for event in final.events] == list(range(5))
    for previous, current in zip(final.events, final.events[1:]):
        assert current.previous_event_sha256 == previous.content_sha256()


@pytest.mark.parametrize(
    "outcome,state",
    [
        (AdjudicationOutcome.REJECTED, WorkflowState.REJECTED),
        (AdjudicationOutcome.ABSTAINED, WorkflowState.ABSTAINED),
    ],
)
def test_nonaccepted_adjudication_has_no_final_sir(outcome, state):
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph(manual_label=11))
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    final = reviewed.with_adjudication(
        _adjudication(case, primary, reviewer, outcome=outcome, graph=None)
    ).finalize()
    assert final.state is state
    with pytest.raises(ValueError, match="only accepted"):
        final.final_graph()


def test_revised_adjudication_must_be_distinct_and_accept_paths_must_match():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph(manual_label=11))
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    with pytest.raises(ValueError, match="distinct SIR"):
        reviewed.with_adjudication(_adjudication(
            case, primary, reviewer,
            outcome=AdjudicationOutcome.REVISED,
            graph=primary.graph(),
        ))
    with pytest.raises(ValueError, match="primary SIR"):
        reviewed.with_adjudication(_adjudication(
            case, primary, reviewer,
            outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
            graph=reviewer.graph(),
        ))
    revised = _graph(manual_label=12)
    final = reviewed.with_adjudication(_adjudication(
        case, primary, reviewer,
        outcome=AdjudicationOutcome.REVISED,
        graph=revised,
    )).finalize()
    assert sir_sha256(final.final_graph()) == sir_sha256(revised)


def test_case_rejects_self_review_self_adjudication_replay_and_time_reversal():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    self_review = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph(),
        author=primary.author_pseudonym)
    with pytest.raises(ValueError, match="must be distinct"):
        case.with_primary(primary).with_reviewer(
            self_review, EventCorrespondence(()))

    early_review = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph(),
        submitted_at="2026-09-13T09:30:00-07:00")
    with pytest.raises(ValueError, match="must follow primary"):
        case.with_primary(primary).with_reviewer(
            early_review, EventCorrespondence(()))

    simultaneous_primary = _submission(
        case, SubmissionRole.PRIMARY, graph=_graph(),
        submitted_at=case.created_at)
    with pytest.raises(ValueError, match="must follow case creation"):
        case.with_primary(simultaneous_primary)

    simultaneous_review = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=_graph(),
        submitted_at=primary.submitted_at)
    with pytest.raises(ValueError, match="must follow primary"):
        case.with_primary(primary).with_reviewer(
            simultaneous_review, EventCorrespondence(()))

    replay = replace(primary, case_id="another-case")
    with pytest.raises(ValueError, match="another case"):
        case.with_primary(replay)

    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11))
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    with pytest.raises(ValueError, match="distinct from both"):
        _adjudication(
            case, primary, reviewer,
            outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
            graph=primary.graph(),
            adjudicator=reviewer.author_pseudonym,
        )

    simultaneous_adjudication = _adjudication(
        case, primary, reviewer,
        outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
        graph=primary.graph(),
        submitted_at=reviewer.submitted_at,
    )
    with pytest.raises(ValueError, match="must follow blind review"):
        reviewed.with_adjudication(simultaneous_adjudication)


def test_hash_chain_and_terminal_claim_tampering_fail_closed():
    case = _case()
    with pytest.raises(ValueError, match="immutable case history"):
        replace(
            case,
            sampling_plan=_artifact(ArtifactKind.SAMPLING_PLAN, "substituted-plan"),
        )
    primary = _submission(
        case, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11))
    identity_correspondence = EventCorrespondence(((0, 0), (1, 1)))
    crossed_correspondence = EventCorrespondence(((0, 1), (1, 0)))
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, identity_correspondence)
    crossed = case.with_primary(primary).with_reviewer(
        reviewer, crossed_correspondence)
    assert reviewed.events[-1].artifact_sha256 != crossed.events[-1].artifact_sha256
    with pytest.raises(ValueError, match="immutable case history"):
        replace(reviewed, correspondence=crossed_correspondence)

    graph = _graph()
    final = case.with_primary(
        _submission(case, SubmissionRole.PRIMARY, graph=graph)
    ).with_reviewer(
        _submission(case, SubmissionRole.INDEPENDENT_REVIEWER, graph=graph),
        EventCorrespondence(()),
    ).finalize()
    manifest = final.to_manifest()
    manifest["events"][2]["previous_event_sha256"] = _digest("forged")
    with pytest.raises(ValueError, match="hash chain"):
        Phase3BReviewCase.from_manifest(manifest)

    manifest = final.to_manifest()
    manifest["events"][1]["case_id"] = "other-case"
    with pytest.raises(ValueError, match="immutable case history"):
        Phase3BReviewCase.from_manifest(manifest)

    for claim in (
        "source_training_target_authorized",
        "commercial_use_authorized",
        "project_linguistic_validation_complete",
    ):
        manifest = final.to_manifest()
        manifest[claim] = True
        with pytest.raises(ValueError, match="cannot claim"):
            Phase3BReviewCase.from_manifest(manifest)


def test_case_loader_rejects_noncanonical_duplicate_nonfinite_and_oversize_json():
    case = _case()
    payload = case.canonical_bytes()
    with pytest.raises(ValueError, match="not canonical"):
        load_phase3b_case(payload + b"\n")
    duplicate = payload.replace(
        b'{"adjudication":', b'{"case_id":"replayed","adjudication":', 1)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_phase3b_case(duplicate)
    nonfinite = payload.replace(b'"schema_version":1', b'"schema_version":NaN', 1)
    with pytest.raises(ValueError, match="non-finite"):
        load_phase3b_case(nonfinite)
    with pytest.raises(ValueError, match="byte limit"):
        load_phase3b_case(payload, max_bytes=10)


def test_correspondence_is_one_to_one_and_cannot_reference_absent_events():
    with pytest.raises(ValueError, match="one-to-one"):
        EventCorrespondence(((0, 1), (0, 2)))
    with pytest.raises(ValueError, match="canonical primary-ID order"):
        EventCorrespondence(((1, 1), (0, 0)))
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11),
    )
    with pytest.raises(ValueError, match="absent event"):
        compare_submissions(primary, reviewer, EventCorrespondence(((99, 0),)))


def test_submission_payload_and_role_protocol_claims_fail_closed():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    with pytest.raises(ValueError, match="canonical|invalid SIR"):
        replace(primary, sir_payload=primary.sir_payload + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(primary, sir_payload_sha256=_digest("wrong"))
    with pytest.raises(ValueError, match="wrong artifact kind"):
        replace(primary, protocol=case.review_protocol)
    with pytest.raises(ValueError, match="qualified"):
        replace(primary, author_qualified_asl=False)
    with pytest.raises(ValueError, match="unsupported SIR timebase"):
        replace(primary, sir_time_unit="seconds")
    with pytest.raises(ValueError, match="outside its bound source interval"):
        _submission(
            case,
            SubmissionRole.INDEPENDENT_REVIEWER,
            graph=_graph(manual_interval=(0.0, 2_001.0)),
        )
    with pytest.raises(ValueError, match="another timing contract"):
        case.with_primary(replace(primary, source_interval_end_ms=3_000))
    manifest = primary.to_dict()
    manifest["gloss_tokens"] = ["ENGLISH", "IS", "NOT", "SIR"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        HumanSIRSubmission.from_dict(manifest)


def test_untimed_source_requires_abstention_instead_of_invented_sir_time():
    untimed = replace(_source(), begin_ms=None, end_ms=None)
    case = _case(source=untimed)
    with pytest.raises(ValueError, match="source timing is unavailable"):
        _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    abstained = _submission(
        case,
        SubmissionRole.PRIMARY,
        decision=SubmissionDecision.ABSTAIN,
        reason_codes=("source_misalignment",),
    )
    assert abstained.source_interval_begin_ms is None
    assert abstained.source_interval_end_ms is None


def test_adjudication_cannot_change_timebase_or_escape_source_interval():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    reviewer = _submission(
        case,
        SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11),
    )
    reviewed = case.with_primary(primary).with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    adjudication = _adjudication(
        reviewed,
        primary,
        reviewer,
        outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
        graph=primary.graph(),
    )
    with pytest.raises(ValueError, match="unsupported SIR timebase"):
        replace(adjudication, sir_time_origin="clip_relative")
    with pytest.raises(ValueError, match="another timing contract"):
        reviewed.with_adjudication(
            replace(adjudication, source_interval_end_ms=3_000))
    with pytest.raises(ValueError, match="outside its bound source interval"):
        _adjudication(
            reviewed,
            primary,
            reviewer,
            outcome=AdjudicationOutcome.REVISED,
            graph=_graph(manual_label=12, manual_interval=(0.0, 2_001.0)),
        )


def test_comparison_rejects_case_replay_and_noncanonical_correspondence():
    case = _case()
    graph = _graph()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=graph)
    reviewer = _submission(
        case, SubmissionRole.INDEPENDENT_REVIEWER, graph=graph)
    with pytest.raises(ValueError, match="implicit complete correspondence"):
        compare_submissions(
            primary, reviewer, EventCorrespondence(((0, 0), (1, 1))))
    with pytest.raises(ValueError, match="different review cases"):
        compare_submissions(
            primary, replace(reviewer, case_id="case-2"), EventCorrespondence(()))
    with pytest.raises(ValueError, match="identifiers must be distinct"):
        compare_submissions(
            primary,
            replace(reviewer, submission_id=primary.submission_id),
            EventCorrespondence(()),
        )
    with pytest.raises(ValueError, match="must follow primary"):
        compare_submissions(
            primary,
            replace(reviewer, submitted_at=primary.submitted_at),
            EventCorrespondence(()),
        )
    with pytest.raises(TypeError, match="event correspondence"):
        compare_submissions(primary, reviewer, object())  # type: ignore[arg-type]

    abstained = _submission(
        case,
        SubmissionRole.INDEPENDENT_REVIEWER,
        decision=SubmissionDecision.ABSTAIN,
        reason_codes=("uncertain_analysis",),
    )
    with pytest.raises(ValueError, match="cannot declare event correspondence"):
        compare_submissions(
            primary, abstained, EventCorrespondence(((0, 0),)))


def test_submission_reason_lists_and_workflow_identifiers_fail_closed():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    with pytest.raises(ValueError, match="cannot contain abstention reason"):
        replace(primary, reason_codes=("uncertain_analysis",))
    malformed = primary.to_dict()
    malformed["limitations"] = [{}]
    with pytest.raises(ValueError, match="bounded, non-empty strings"):
        HumanSIRSubmission.from_dict(malformed)

    reviewer = _submission(
        case,
        SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11),
    )
    primary_case = case.with_primary(primary)
    with pytest.raises(ValueError, match="distinct identifiers"):
        primary_case.with_reviewer(
            replace(reviewer, submission_id=primary.submission_id),
            EventCorrespondence(((0, 0), (1, 1))),
        )

    reviewed = primary_case.with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    adjudication = _adjudication(
        reviewed,
        primary,
        reviewer,
        outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
        graph=_graph(),
    )
    with pytest.raises(ValueError, match="identifiers must be distinct"):
        reviewed.with_adjudication(
            replace(adjudication, adjudication_id=primary.submission_id))


def _second_source() -> EAFAnnotationBinding:
    value = "SECOND-PUBLISHER-SOURCE-LABEL"
    return replace(
        _source(),
        source_annotation_id="a2",
        source_order=1,
        source_value=value,
        source_value_sha256=_digest(value),
        begin_time_slot_ref="ts3",
        end_time_slot_ref="ts4",
        begin_ms=0,
        end_ms=2_000,
    )


def test_public_workflow_boundaries_reject_untyped_and_cross_case_inputs():
    case = _case()
    primary = _submission(case, SubmissionRole.PRIMARY, graph=_graph())
    reviewer = _submission(
        case,
        SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11),
    )
    with pytest.raises(TypeError, match="primary"):
        case.with_primary(object())  # type: ignore[arg-type]
    primary_case = case.with_primary(primary)
    with pytest.raises(TypeError, match="reviewer"):
        primary_case.with_reviewer(  # type: ignore[arg-type]
            object(), EventCorrespondence(()))
    with pytest.raises(TypeError, match="correspondence"):
        primary_case.with_reviewer(reviewer, object())  # type: ignore[arg-type]

    reviewed = primary_case.with_reviewer(
        reviewer, EventCorrespondence(((0, 0), (1, 1))))
    with pytest.raises(TypeError, match="human adjudication"):
        reviewed.with_adjudication(object())  # type: ignore[arg-type]
    other_case = _case(case_id="case-2", source=_second_source())
    with pytest.raises(ValueError, match="another case"):
        _adjudication(
            other_case,
            primary,
            reviewer,
            outcome=AdjudicationOutcome.ACCEPT_PRIMARY,
            graph=primary.graph(),
        )

    with pytest.raises(TypeError, match="source"):
        HumanSIRSubmission.create(  # type: ignore[arg-type]
            submission_id="bad-source",
            case_id=case.case_id,
            source=object(),
            role=SubmissionRole.PRIMARY,
            author_pseudonym="annotator-a",
            qualification_evidence_sha256=_digest("qualification"),
            independence_evidence_sha256=_digest("independence"),
            attestation_sha256=_digest("attestation"),
            protocol=case.annotation_protocol,
            decision=SubmissionDecision.SIR,
            submitted_at="2026-09-13T10:00:00-07:00",
            graph=_graph(),
        )


def test_batch_audit_reconciles_states_and_pools_exact_support():
    first = _case()
    first_graph = _graph()
    first_final = first.with_primary(
        _submission(first, SubmissionRole.PRIMARY, graph=first_graph)
    ).with_reviewer(
        _submission(
            first, SubmissionRole.INDEPENDENT_REVIEWER, graph=first_graph),
        EventCorrespondence(()),
    ).finalize()

    second = _case(case_id="case-2", source=_second_source())
    second_primary = _submission(
        second, SubmissionRole.PRIMARY, graph=_graph(manual_label=10))
    second_reviewer = _submission(
        second, SubmissionRole.INDEPENDENT_REVIEWER,
        graph=_graph(manual_label=11))
    second_reviewed = second.with_primary(second_primary).with_reviewer(
        second_reviewer, EventCorrespondence(((0, 0), (1, 1))))
    second_adjudication = _adjudication(
        second_reviewed,
        second_primary,
        second_reviewer,
        outcome=AdjudicationOutcome.REJECTED,
        graph=None,
    )
    second_final = second_reviewed.with_adjudication(
        second_adjudication).finalize()

    report = audit_phase3b_batch(
        [second_final, first_final], ["case-1", "case-2"])
    repeated = audit_phase3b_batch(
        [first_final, second_final], ["case-2", "case-1"])
    assert report.canonical_bytes() == repeated.canonical_bytes()
    loaded = load_phase3b_batch_report(report.canonical_bytes())
    assert loaded == report
    verify_phase3b_batch_report(loaded, [first_final, second_final])
    with pytest.raises(RuntimeError, match="does not match"):
        verify_phase3b_batch_report(loaded, [first_final])
    with pytest.raises(ValueError, match="not canonical"):
        load_phase3b_batch_report(report.canonical_bytes() + b"\n")
    assert report.software_workflow_complete
    assert report.state_counts == (
        ("draft", 0),
        ("primary_complete", 0),
        ("blind_reviewed", 0),
        ("adjudicated", 0),
        ("accepted", 1),
        ("rejected", 1),
        ("abstained", 0),
    )
    assert report.adjudicated_case_count == 1
    assert report.comparison_available_count == 2
    assert report.exact_sir_match_count == 1
    assert report.paired_event_count == 4
    assert report.primary_event_pair_coverage == 1.0
    assert report.reviewer_event_pair_coverage == 1.0
    fields = {item.field: item for item in report.field_agreement}
    assert (fields["label"].agreements, fields["label"].support,
            fields["label"].rate) == (3, 4, 0.75)
    assert fields["kind"].rate == 1.0
    assert report.temporal_pair_count == 4
    assert report.median_temporal_iou == 1.0
    assert report.comparable_edge_intersection == 2
    assert report.comparable_edge_union == 2
    assert report.comparable_edge_jaccard == 1.0
    assert report.no_acceptance_threshold_selected
    assert not report.approved_for_research_training
    assert not report.commercial_use_authorized
    assert not report.project_linguistic_validation_complete
    report.require_software_workflow_complete()
    with pytest.raises(ValueError, match="prohibited approval claim"):
        replace(report, approved_for_research_training=True)
    tampered = report.to_dict()
    tampered["field_agreement"][1]["rate"] = 0.99
    with pytest.raises(ValueError, match="does not match"):
        load_phase3b_batch_report(canonical_json_bytes(tampered))


def test_batch_audit_exposes_incomplete_mixed_and_duplicate_inputs():
    first = _case()
    report = audit_phase3b_batch([first], ["case-1", "case-2"])
    assert not report.complete_accounting
    assert not report.all_cases_terminal
    assert not report.software_workflow_complete
    assert report.violations == (
        "missing_case:case-2", "nonterminal_case:case-1")
    assert all(item.rate is None for item in report.field_agreement)
    assert report.median_temporal_iou is None
    assert report.comparable_edge_jaccard is None
    with pytest.raises(RuntimeError, match="incomplete"):
        report.require_software_workflow_complete()

    duplicate = _case(case_id="case-2")
    with pytest.raises(ValueError, match="double-counted"):
        audit_phase3b_batch([first, duplicate], ["case-1", "case-2"])

    second_view = replace(
        _source(),
        media_descriptor_order=1,
        source_video_sha256=_digest("second-camera-view"),
    )
    duplicate_across_views = _case(
        case_id="case-2", source=second_view)
    with pytest.raises(ValueError, match="double-counted"):
        audit_phase3b_batch(
            [first, duplicate_across_views], ["case-1", "case-2"])

    reingested_second_view = replace(
        second_view,
        eaf_manifest_sha256=_digest("same-eaf-reingested-manifest"),
    )
    duplicate_after_reingestion = _case(
        case_id="case-2", source=reingested_second_view)
    with pytest.raises(ValueError, match="double-counted"):
        audit_phase3b_batch(
            [first, duplicate_after_reingestion], ["case-1", "case-2"])

    mixed = _case(
        case_id="case-2",
        source=_second_source(),
        sampling_plan=_artifact(ArtifactKind.SAMPLING_PLAN, "other-plan"),
    )
    mixed_report = audit_phase3b_batch(
        [first, mixed], ["case-1", "case-2"])
    assert "mixed_governance:sampling_plan" in mixed_report.violations
    assert not mixed_report.software_workflow_complete


def test_review_queue_is_label_empty_escaped_reproducible_and_reloadable(tmp_path):
    injected = _eaf().replace(
        b"HELLO", b"&lt;script&gt;alert(1)&lt;/script&gt;")
    source, source_manifest = _source_bundle(
        tmp_path, eaf_payload=injected, video_name="sample space.mp4")
    governance, specs = _governance_bundle(tmp_path, source_manifest)
    output = tmp_path / "queue"
    receipt = write_phase3b_review_queue(
        source_manifest_payload=source_manifest,
        source_root=source,
        governance_root=governance,
        governance_specs=tuple(reversed(specs)),
        selections=[QueueSelection("ID-gloss", "a1")],
        output_dir=output,
        creator_pseudonym="curator-1",
        created_at="2026-09-13T13:00:00-07:00",
    )
    assert {path.name for path in output.iterdir()} == {
        "artifact-index.json",
        "phase3b-draft-cases.jsonl",
        "phase3b-review-queue.html",
        "queue-manifest.json",
    }
    cases = verify_phase3b_review_queue(
        output,
        source_manifest_payload=source_manifest,
        source_root=source,
        governance_root=governance,
        expected_queue_manifest_sha256=receipt.queue_manifest_sha256,
    )
    assert receipt.path == output.resolve()
    assert receipt.case_count == 1
    assert len(cases) == 1
    assert cases[0].state is WorkflowState.DRAFT
    assert cases[0].source.source_value == "<script>alert(1)</script>"
    assert cases[0].primary is None
    assert cases[0].reviewer is None
    drafts = (output / "phase3b-draft-cases.jsonl").read_bytes()
    assert b'"primary":null' in drafts
    assert b'"reviewer":null' in drafts
    assert b'"gloss_tokens"' not in drafts
    markup = (output / "phase3b-review-queue.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in markup
    assert "sample%20space.mp4" in markup
    queue_manifest = json.loads(
        (output / "queue-manifest.json").read_text(encoding="utf-8"))
    assert queue_manifest["project_labels_prepopulated"] is False
    assert queue_manifest["source_training_target_authorized"] is False

    repeated_output = tmp_path / "queue-repeated"
    repeated_receipt = write_phase3b_review_queue(
        source_manifest_payload=source_manifest,
        source_root=source,
        governance_root=governance,
        governance_specs=specs,
        selections=[QueueSelection("ID-gloss", "a1")],
        output_dir=repeated_output,
        creator_pseudonym="curator-1",
        created_at="2026-09-13T13:00:00-07:00",
    )
    assert {
        path.name: path.read_bytes() for path in output.iterdir()
    } == {
        path.name: path.read_bytes() for path in repeated_output.iterdir()
    }
    assert repeated_receipt.queue_manifest_sha256 == receipt.queue_manifest_sha256

    with pytest.raises(FileExistsError, match="already exists"):
        write_phase3b_review_queue(
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            governance_specs=specs,
            selections=[QueueSelection("ID-gloss", "a1")],
            output_dir=output,
            creator_pseudonym="curator-1",
            created_at="2026-09-13T13:00:00-07:00",
        )


def test_review_queue_rejects_artifact_governance_and_selection_tampering(tmp_path):
    source, source_manifest = _source_bundle(tmp_path)
    governance, specs = _governance_bundle(tmp_path, source_manifest)
    output = tmp_path / "queue"
    receipt = write_phase3b_review_queue(
        source_manifest_payload=source_manifest,
        source_root=source,
        governance_root=governance,
        governance_specs=specs,
        selections=[QueueSelection("ID-gloss", "a1")],
        output_dir=output,
        creator_pseudonym="curator-1",
        created_at="2026-09-13T13:00:00-07:00",
    )
    with pytest.raises(RuntimeError, match="trusted receipt"):
        verify_phase3b_review_queue(
            output,
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=_digest("untrusted-root"),
        )
    html_path = output / "phase3b-review-queue.html"
    original_html = html_path.read_bytes()
    index_path = output / "artifact-index.json"
    original_index = index_path.read_bytes()
    index_path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="artifact index exceeds the byte limit"):
        verify_phase3b_review_queue(
            output,
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=receipt.queue_manifest_sha256,
        )
    index_path.write_bytes(original_index)

    html_path.write_bytes(original_html + b"tampered")
    with pytest.raises(RuntimeError, match="differs from its hash index"):
        verify_phase3b_review_queue(
            output,
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=receipt.queue_manifest_sha256,
        )

    rewritten_index = json.loads(original_index)
    for record in rewritten_index["files"]:
        if record["name"] == "phase3b-review-queue.html":
            record["sha256"] = hashlib.sha256(html_path.read_bytes()).hexdigest()
            record["size"] = html_path.stat().st_size
    index_path.write_bytes(canonical_json_bytes(rewritten_index))
    with pytest.raises(RuntimeError, match="HTML does not match"):
        verify_phase3b_review_queue(
            output,
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=receipt.queue_manifest_sha256,
        )
    html_path.write_bytes(original_html)
    index_path.write_bytes(original_index)

    governance_path = governance / specs[0].relative_path
    governance_path.write_text("mutated governance", encoding="utf-8")
    with pytest.raises(RuntimeError, match="governance artifacts have changed"):
        verify_phase3b_review_queue(
            output,
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            expected_queue_manifest_sha256=receipt.queue_manifest_sha256,
        )
    with pytest.raises(ValueError, match="escapes"):
        GovernanceFileSpec(
            ArtifactKind.ASL_CONVENTION, "bad", "1.0.0", "../outside.md")

    with pytest.raises(ValueError, match="differ from the frozen sampling plan"):
        write_phase3b_review_queue(
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            governance_specs=specs,
            selections=[QueueSelection("ID-gloss", "another-annotation")],
            output_dir=tmp_path / "sampling-mismatch",
            creator_pseudonym="curator-1",
            created_at="2026-09-13T13:00:00-07:00",
        )

    with pytest.raises(ValueError, match="repeat a source annotation"):
        write_phase3b_review_queue(
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            governance_specs=specs,
            selections=[
                QueueSelection("ID-gloss", "a1"),
                QueueSelection("ID-gloss", "a1"),
            ],
            output_dir=tmp_path / "duplicate-selection",
            creator_pseudonym="curator-1",
            created_at="2026-09-13T13:00:00-07:00",
        )

    with pytest.raises(ValueError, match="across media views"):
        write_phase3b_review_queue(
            source_manifest_payload=source_manifest,
            source_root=source,
            governance_root=governance,
            governance_specs=specs,
            selections=[
                QueueSelection("ID-gloss", "a1", media_descriptor_order=0),
                QueueSelection("ID-gloss", "a1", media_descriptor_order=1),
            ],
            output_dir=tmp_path / "duplicate-selection-across-views",
            creator_pseudonym="curator-1",
            created_at="2026-09-13T13:00:00-07:00",
        )
