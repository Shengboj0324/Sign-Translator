"""Adversarial tests for source-native, fail-closed ELAN ingestion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest

import signtranslator.data_engineering.eaf as eaf_module
from signtranslator.data_engineering.eaf import (
    EAF_SCHEMA_SHA256,
    EAF_SCHEMA_URL,
    EAFIngestionLimits,
    PROJECT_MAPPING_STATUS,
    SOURCE_LABEL_SEMANTICS,
    SourceElement,
    ingest_eaf_reference,
    load_eaf_manifest,
    parse_eaf_bytes,
    write_eaf_inspection_artifacts,
)
from signtranslator.reproducibility import canonical_json_bytes


MEDIA_URL = "file:///publisher/original/sample.mp4"


def _eaf(*, first_value: str = "HELLO", media_url: str = MEDIA_URL) -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="Corpus Team" DATE="2021-04-01T00:00:00Z"
 VERSION="3.0" FORMAT="3.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
 <LICENSE LICENSE_URL="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</LICENSE>
 <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
  <MEDIA_DESCRIPTOR MEDIA_URL="{media_url}" RELATIVE_MEDIA_URL="./sample.mp4"
   MIME_TYPE="video/mp4"/>
  <PROPERTY NAME="corpus">Cokely-shaped test fixture</PROPERTY>
 </HEADER>
 <TIME_ORDER>
  <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
  <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="200"/>
  <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="400"/>
 </TIME_ORDER>
 <TIER TIER_ID="ID-gloss" PARTICIPANT="pseudonymous-signer" ANNOTATOR="source-team"
  LINGUISTIC_TYPE_REF="source-gloss" LANG_REF="asl">
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1"
   TIME_SLOT_REF2="ts2" CVE_REF="cve-hello" LANG_REF="asl">
   <ANNOTATION_VALUE>{first_value}</ANNOTATION_VALUE>
  </ALIGNABLE_ANNOTATION></ANNOTATION>
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts2"
   TIME_SLOT_REF2="ts3" CVE_REF="cve-bye" LANG_REF="asl">
   <ANNOTATION_VALUE>BYE</ANNOTATION_VALUE>
  </ALIGNABLE_ANNOTATION></ANNOTATION>
 </TIER>
 <TIER TIER_ID="source-notes" LINGUISTIC_TYPE_REF="source-note"
  PARENT_REF="ID-gloss" LANG_REF="asl">
  <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="r1" ANNOTATION_REF="a1" LANG_REF="asl">
   <ANNOTATION_VALUE>first note</ANNOTATION_VALUE>
  </REF_ANNOTATION></ANNOTATION>
  <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="r2" ANNOTATION_REF="a2"
   PREVIOUS_ANNOTATION="r1" LANG_REF="asl">
   <ANNOTATION_VALUE>second note</ANNOTATION_VALUE>
  </REF_ANNOTATION></ANNOTATION>
 </TIER>
 <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="source-gloss" TIME_ALIGNABLE="true"
  CONTROLLED_VOCABULARY_REF="source-cv"/>
 <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="source-note" TIME_ALIGNABLE="false"
  CONSTRAINTS="Symbolic_Association"/>
 <LANGUAGE LANG_ID="asl" LANG_DEF="https://glottolog.org/resource/languoid/id/asli1244"
  LANG_LABEL="American Sign Language"/>
 <CONSTRAINT STEREOTYPE="Symbolic_Association" DESCRIPTION="Reference to parent annotation"/>
 <CONTROLLED_VOCABULARY CV_ID="source-cv">
  <DESCRIPTION LANG_REF="asl">Publisher-specific ID labels</DESCRIPTION>
  <CV_ENTRY_ML CVE_ID="cve-hello"><CVE_VALUE LANG_REF="asl">HELLO</CVE_VALUE></CV_ENTRY_ML>
  <CV_ENTRY_ML CVE_ID="cve-bye"><CVE_VALUE LANG_REF="asl">BYE</CVE_VALUE></CV_ENTRY_ML>
 </CONTROLLED_VOCABULARY>
</ANNOTATION_DOCUMENT>'''.encode("utf-8")


def _write_video(path: Path, *, frames: int = 20) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _source_tree(tmp_path: Path, *, value: str = "HELLO",
                 media_name: str = "sample.mp4") -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    eaf = source / "sample.eaf"
    eaf.write_bytes(_eaf(first_value=value))
    video = source / media_name
    _write_video(video)
    evidence = source / "LICENSE-EVIDENCE.html"
    evidence.write_text(
        "Publisher states CC BY-NC-SA 4.0; captured for research evidence.",
        encoding="utf-8",
    )
    return source, eaf, evidence


def _ingest(tmp_path: Path, *, value: str = "HELLO",
            media_name: str = "sample.mp4"):
    source, eaf, evidence = _source_tree(
        tmp_path, value=value, media_name=media_name)
    result = ingest_eaf_reference(
        eaf,
        source,
        license_evidence_path=evidence,
        media_paths={MEDIA_URL: media_name},
    )
    return source, result


def test_source_native_parse_preserves_order_metadata_values_and_reference_chain():
    document = parse_eaf_bytes(_eaf(first_value="HELLO &amp; KEEP"))
    assert [slot.time_slot_id for slot in document.time_slots] == ["ts1", "ts2", "ts3"]
    assert [slot.time_value_ms for slot in document.time_slots] == [0, 200, 400]
    assert [tier.tier_id for tier in document.tiers] == ["ID-gloss", "source-notes"]
    assert document.tiers[0].participant == "pseudonymous-signer"
    assert document.tiers[0].annotator == "source-team"
    assert [item.value for item in document.tiers[0].annotations] == [
        "HELLO & KEEP", "BYE"]
    assert document.tiers[1].annotations[1].annotation_ref == "a2"
    assert document.tiers[1].annotations[1].previous_annotation == "r1"
    assert document.tiers[0].annotations[0].begin_time_slot_ref == "ts1"
    assert document.tiers[0].annotations[0].end_time_slot_ref == "ts2"
    reconstructed = SourceElement.from_dict(document.root.to_dict())
    assert reconstructed == document.root

    shifted = parse_eaf_bytes(
        _eaf().replace(b'MIME_TYPE="video/mp4"',
                       b'MIME_TYPE="video/mp4" TIME_ORIGIN="-250"'))
    assert shifted.media_descriptors[0].time_origin_ms == -250
    overflow = _eaf().replace(
        b'MIME_TYPE="video/mp4"',
        b'MIME_TYPE="video/mp4" TIME_ORIGIN="9223372036854775808"',
    )
    with pytest.raises(ValueError, match="signed 64-bit"):
        parse_eaf_bytes(overflow)
    date_only = _eaf().replace(
        b'DATE="2021-04-01T00:00:00Z"', b'DATE="2021-04-01"')
    with pytest.raises(ValueError, match="ISO-8601 date-time"):
        parse_eaf_bytes(date_only)
    unsupported_version = _eaf().replace(
        b'VERSION="3.0" FORMAT="3.0"', b'VERSION="2.8" FORMAT="2.8"')
    with pytest.raises(ValueError, match="unsupported EAF version: 2.8"):
        parse_eaf_bytes(unsupported_version)


def test_unaligned_elan_time_slots_are_preserved_without_interpolation():
    payload = _eaf().replace(
        b'<TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="200"/>',
        b'<TIME_SLOT TIME_SLOT_ID="ts2"/>',
    )
    document = parse_eaf_bytes(payload)
    first, second = document.tiers[0].annotations
    assert first.begin_time_slot_ref == "ts1"
    assert first.end_time_slot_ref == "ts2"
    assert first.begin_ms == 0
    assert first.end_ms is None
    assert second.begin_time_slot_ref == "ts2"
    assert second.end_time_slot_ref == "ts3"
    assert second.begin_ms is None
    assert second.end_ms == 400


def test_manifest_never_relabels_source_annotations_as_sir_or_training_targets(tmp_path):
    _, result = _ingest(tmp_path)
    manifest = result.to_manifest()
    assert manifest["source_label_semantics"] == SOURCE_LABEL_SEMANTICS
    assert manifest["project_mapping_status"] == PROJECT_MAPPING_STATUS
    assert manifest["training_target_authorized"] is False
    assert manifest["linguistically_validated_by_project"] is False
    assert manifest["eaf_schema_url"] == EAF_SCHEMA_URL
    assert manifest["eaf_schema_sha256"] == EAF_SCHEMA_SHA256
    assert manifest["auxiliary_evidence_files"] == []
    encoded = canonical_json_bytes(manifest)
    assert b'"gloss_tokens"' not in encoded
    assert b'"sir_payload"' not in encoded
    assert result.eaf_file.sha256 == hashlib.sha256(_eaf()).hexdigest()
    assert len(result.media_bindings) == 1
    assert result.media_bindings[0].file.sha256 == hashlib.sha256(
        (tmp_path / "source" / "sample.mp4").read_bytes()).hexdigest()


def test_artifacts_are_hash_indexed_canonical_escaped_and_non_overwriting(tmp_path):
    source, result = _ingest(
        tmp_path, value="=HYPERLINK(&quot;bad&quot;) &lt;script&gt;",
        media_name="sample one&.mp4",
    )
    output = tmp_path / "artifacts" / "pilot"
    output.parent.mkdir()
    write_eaf_inspection_artifacts(result, output, source)
    manifest_bytes = (output / "source-native-manifest.json").read_bytes()
    assert load_eaf_manifest(manifest_bytes) == result.to_manifest()
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256

    with (output / "annotation-inspection.csv").open(
            encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_value"].startswith("'=HYPERLINK")
    assert rows[0]["begin_time_slot_ref"] == "ts1"
    assert rows[0]["end_time_slot_ref"] == "ts2"
    assert rows[0]["time_alignment_status"] == "aligned"
    html_text = (output / "annotation-inspection.html").read_text(encoding="utf-8")
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "sample%20one%26.mp4" in html_text

    index = json.loads((output / "artifact-index.json").read_text(encoding="utf-8"))
    indexed = {item["name"]: item for item in index["files"]}
    assert set(indexed) == {
        "annotation-inspection.csv", "annotation-inspection.html",
        "source-native-manifest.json",
    }
    for name, item in indexed.items():
        payload = (output / name).read_bytes()
        assert item == {
            "name": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    with pytest.raises(FileExistsError, match="overwrite"):
        write_eaf_inspection_artifacts(result, output, source)


@pytest.mark.parametrize("attack", [
    b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
    b'<!-- hidden annotation -->',
    b'<?unsafe instruction?>',
    b'<![CDATA[hidden]]>',
])
def test_dtd_entities_comments_processing_instructions_and_cdata_fail_closed(attack):
    payload = _eaf().replace(b"<ANNOTATION_DOCUMENT", attack + b"<ANNOTATION_DOCUMENT", 1)
    with pytest.raises(ValueError, match="forbidden"):
        parse_eaf_bytes(payload)


def test_xinclude_unknown_elements_attributes_and_malformed_xml_fail_closed():
    xinclude = _eaf().replace(
        b"<TIME_ORDER>",
        b'<xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="secret"/>'
        b"<TIME_ORDER>",
    )
    with pytest.raises(ValueError, match="namespaced|unsupported"):
        parse_eaf_bytes(xinclude)
    unknown = _eaf().replace(b'<TIME_ORDER>', b'<TIME_ORDER UNSUPPORTED="1">')
    with pytest.raises(ValueError, match="attributes invalid"):
        parse_eaf_bytes(unknown)
    with pytest.raises(ValueError, match="malformed"):
        parse_eaf_bytes(_eaf()[:-20])


@pytest.mark.parametrize("replacement, message", [
    (b'TIME_SLOT_REF1="missing"', "dangling time-slot"),
    (b'TIME_SLOT_REF1="ts2"', "non-positive interval"),
    (b'TIME_SLOT_REF2="ts1"', "non-positive interval"),
    (b'CVE_REF="missing"', "invalid CVE_REF"),
    (b'LINGUISTIC_TYPE_REF="missing"', "dangling linguistic-type"),
    (b'LANG_REF="missing"', "dangling language"),
])
def test_dangling_and_invalid_semantic_references_fail_closed(replacement, message):
    if replacement.startswith(b"TIME_SLOT_REF1"):
        payload = _eaf().replace(b'TIME_SLOT_REF1="ts1"', replacement, 1)
    elif replacement.startswith(b"TIME_SLOT_REF2"):
        payload = _eaf().replace(b'TIME_SLOT_REF2="ts2"', replacement, 1)
    elif replacement.startswith(b"CVE_REF"):
        payload = _eaf().replace(b'CVE_REF="cve-hello"', replacement, 1)
    elif replacement.startswith(b"LINGUISTIC"):
        payload = _eaf().replace(
            b'LINGUISTIC_TYPE_REF="source-gloss"', replacement, 1)
    else:
        payload = _eaf().replace(b'LANG_REF="asl"', replacement, 1)
    with pytest.raises(ValueError, match=message):
        parse_eaf_bytes(payload)


def test_duplicate_ids_overlap_and_cycles_fail_closed():
    duplicate = _eaf().replace(b'ANNOTATION_ID="a2"', b'ANNOTATION_ID="a1"')
    with pytest.raises(ValueError, match="duplicate ANNOTATION_ID"):
        parse_eaf_bytes(duplicate)

    overlap = _eaf().replace(
        b'ANNOTATION_ID="a2" TIME_SLOT_REF1="ts2"',
        b'ANNOTATION_ID="a2" TIME_SLOT_REF1="ts1"',
    )
    with pytest.raises(ValueError, match="overlapping"):
        parse_eaf_bytes(overlap)

    previous_cycle = _eaf().replace(
        b'ANNOTATION_ID="r1" ANNOTATION_REF="a1" LANG_REF="asl"',
        b'ANNOTATION_ID="r1" ANNOTATION_REF="a1" PREVIOUS_ANNOTATION="r2" LANG_REF="asl"',
    )
    with pytest.raises(ValueError, match="previous-annotation graph contains a cycle"):
        parse_eaf_bytes(previous_cycle)

    tier_cycle = _eaf().replace(
        b'TIER_ID="ID-gloss" PARTICIPANT=',
        b'TIER_ID="ID-gloss" PARENT_REF="source-notes" PARTICIPANT=',
    )
    with pytest.raises(ValueError, match="tier-parent graph contains a cycle"):
        parse_eaf_bytes(tier_cycle)


def test_unicode_spoofing_and_noncanonical_text_fail_closed():
    bidi = _eaf().replace(b"HELLO</ANNOTATION_VALUE>",
                          "HELLO\u202e</ANNOTATION_VALUE>".encode("utf-8"), 1)
    with pytest.raises(ValueError, match="bidirectional"):
        parse_eaf_bytes(bidi)
    decomposed = _eaf().replace(
        b"HELLO</ANNOTATION_VALUE>", "A\u030a</ANNOTATION_VALUE>".encode("utf-8"), 1)
    with pytest.raises(ValueError, match="NFC"):
        parse_eaf_bytes(decomposed)


def test_resource_limits_are_applied_before_semantic_use():
    payload = _eaf()
    with pytest.raises(ValueError, match="byte limit"):
        parse_eaf_bytes(payload, replace(EAFIngestionLimits(), max_xml_bytes=20))
    with pytest.raises(ValueError, match="nesting-depth"):
        parse_eaf_bytes(payload, replace(EAFIngestionLimits(), max_depth=4))
    with pytest.raises(ValueError, match="element-count"):
        parse_eaf_bytes(payload, replace(EAFIngestionLimits(), max_elements=5))
    with pytest.raises(ValueError, match="annotation-count"):
        parse_eaf_bytes(payload, replace(EAFIngestionLimits(), max_annotations=3))


def test_binding_is_exact_rejects_symlinks_escapes_mutation_and_out_of_range(
    tmp_path, monkeypatch,
):
    source, eaf, evidence = _source_tree(tmp_path)
    with pytest.raises(ValueError, match="binding keys mismatch"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence, media_paths={})
    with pytest.raises(TypeError, match="descriptor strings"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={1: "sample.mp4"})
    with pytest.raises(ValueError, match="binding keys mismatch"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4", "unused": "sample.mp4"})

    audio_eaf = source / "audio.eaf"
    audio_eaf.write_bytes(
        _eaf().replace(b'MIME_TYPE="video/mp4"', b'MIME_TYPE="audio/mp4"'))
    with pytest.raises(ValueError, match="unsupported non-video"):
        ingest_eaf_reference(
            audio_eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"})

    outside = tmp_path / "outside.mp4"
    _write_video(outside)
    with pytest.raises(ValueError, match="escapes dataset root"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: outside})
    link = source / "link.mp4"
    link.symlink_to(source / "sample.mp4")
    with pytest.raises(ValueError, match="symlink"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: link})

    original_decode = eaf_module.decode_video_clock

    def mutate(path):
        result = original_decode(path)
        with Path(path).open("ab") as stream:
            stream.write(b"mutation")
        return result

    monkeypatch.setattr(eaf_module, "decode_video_clock", mutate)
    with pytest.raises(RuntimeError, match="changed after hashing"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"})

    other = tmp_path / "range"
    other.mkdir()
    range_eaf = other / "sample.eaf"
    range_eaf.write_bytes(_eaf().replace(b'TIME_VALUE="400"', b'TIME_VALUE="4000"'))
    range_evidence = other / "license.txt"
    range_evidence.write_text("evidence", encoding="utf-8")
    _write_video(other / "sample.mp4")
    monkeypatch.setattr(eaf_module, "decode_video_clock", original_decode)
    with pytest.raises(ValueError, match="outside bound media range"):
        ingest_eaf_reference(
            range_eaf, other, license_evidence_path=range_evidence,
            media_paths={MEDIA_URL: "sample.mp4"})


def test_same_size_timestamp_preserving_media_mutation_is_detected(
    tmp_path, monkeypatch,
):
    source, eaf, evidence = _source_tree(tmp_path)
    video = source / "sample.mp4"
    original_decode = eaf_module.decode_video_clock

    def mutate_without_identity_change(path):
        clock = original_decode(path)
        target = Path(path)
        before = target.stat()
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 1
        target.write_bytes(payload)
        target_stat = target.stat()
        assert target_stat.st_size == before.st_size
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        return clock

    monkeypatch.setattr(eaf_module, "decode_video_clock", mutate_without_identity_change)
    with pytest.raises(RuntimeError, match="content changed after hashing"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: video})


def test_eaf_and_license_symlinks_are_rejected(tmp_path):
    source, eaf, evidence = _source_tree(tmp_path)
    eaf_link = source / "link.eaf"
    eaf_link.symlink_to(eaf)
    with pytest.raises(ValueError, match="symlink"):
        ingest_eaf_reference(
            eaf_link, source, license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"})
    evidence_link = source / "license-link.txt"
    evidence_link.symlink_to(evidence)
    with pytest.raises(ValueError, match="symlink"):
        ingest_eaf_reference(
            eaf, source, license_evidence_path=evidence_link,
            media_paths={MEDIA_URL: "sample.mp4"})


def test_manifest_loader_rejects_noncanonical_duplicate_nonfinite_and_false_claims(tmp_path):
    _, result = _ingest(tmp_path)
    canonical = canonical_json_bytes(result.to_manifest())
    assert load_eaf_manifest(canonical) == result.to_manifest()
    with pytest.raises(ValueError, match="canonical"):
        load_eaf_manifest(canonical + b"\n")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_eaf_manifest(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="non-finite"):
        load_eaf_manifest(b'{"x":NaN}')
    with pytest.raises(ValueError, match="malformed manifest JSON"):
        load_eaf_manifest(b"\xff")
    false_claim = result.to_manifest()
    false_claim["training_target_authorized"] = True
    with pytest.raises(ValueError, match="unsupported validation or training claim"):
        load_eaf_manifest(canonical_json_bytes(false_claim))
    wrong_schema = result.to_manifest()
    wrong_schema["eaf_schema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema identity was altered"):
        load_eaf_manifest(canonical_json_bytes(wrong_schema))
    with pytest.raises(ValueError, match="byte limit"):
        load_eaf_manifest(canonical, max_bytes=10)

    wrong_binding = result.to_manifest()
    wrong_binding["media_bindings"][0]["descriptor_media_url"] = "file:///wrong.mp4"
    with pytest.raises(ValueError, match="does not match its EAF media descriptor"):
        load_eaf_manifest(canonical_json_bytes(wrong_binding))

    missing_binding = result.to_manifest()
    missing_binding["media_bindings"] = []
    with pytest.raises(ValueError, match="do not match EAF media descriptors"):
        load_eaf_manifest(canonical_json_bytes(missing_binding))

    duplicate_primary = result.to_manifest()
    duplicate_primary["auxiliary_evidence_files"] = [
        duplicate_primary["eaf_file"]]
    with pytest.raises(ValueError, match="duplicates a primary source path"):
        load_eaf_manifest(canonical_json_bytes(duplicate_primary))

    duplicate_media = result.to_manifest()
    duplicate_media["auxiliary_evidence_files"] = [
        duplicate_media["media_bindings"][0]["file"]]
    with pytest.raises(ValueError, match="duplicates bound media"):
        load_eaf_manifest(canonical_json_bytes(duplicate_media))


def test_manifest_tree_reconstruction_enforces_xml_resource_limits():
    leaf: dict[str, object] = {
        "tag": "ANNOTATION_VALUE", "attributes": [], "text": "x", "children": []}
    nested = leaf
    for _ in range(6):
        nested = {
            "tag": "ANNOTATION", "attributes": [], "text": "", "children": [nested]}
    with pytest.raises(ValueError, match="nesting-depth"):
        SourceElement.from_dict(
            nested, replace(EAFIngestionLimits(), max_depth=4))


def test_binding_output_is_deterministic_for_unchanged_source(tmp_path):
    _, first = _ingest(tmp_path / "first")
    _, second = _ingest(tmp_path / "second")
    first_manifest = first.to_manifest()
    second_manifest = second.to_manifest()
    for manifest in (first_manifest, second_manifest):
        for field in ("eaf_file", "license_evidence_file"):
            for machine_field in ("device", "inode", "mtime_ns"):
                manifest[field][machine_field] = 0
        for binding in manifest["media_bindings"]:
            for machine_field in ("device", "inode", "mtime_ns"):
                binding["file"][machine_field] = 0
    assert canonical_json_bytes(first_manifest) == canonical_json_bytes(second_manifest)


def test_binding_configuration_rejects_ambiguity_unsafe_files_and_limits(tmp_path):
    valid = tmp_path / "binding.json"
    valid.write_text(
        json.dumps({"media_paths": {MEDIA_URL: "sample.mp4"}}),
        encoding="utf-8",
    )
    assert eaf_module._load_binding_config(valid) == {MEDIA_URL: "sample.mp4"}

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"media_paths":{},"media_paths":{"x":"y"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate binding configuration key"):
        eaf_module._load_binding_config(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"media_paths":{},"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        eaf_module._load_binding_config(nonfinite)

    empty_path = tmp_path / "empty-path.json"
    empty_path.write_text('{"media_paths":{"x":""}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty strings"):
        eaf_module._load_binding_config(empty_path)

    with pytest.raises(ValueError, match="byte limit"):
        eaf_module._load_binding_config(valid, max_bytes=1)

    link = tmp_path / "binding-link.json"
    link.symlink_to(valid)
    with pytest.raises(ValueError, match="symlink"):
        eaf_module._load_binding_config(link)


def test_cli_creates_one_hash_bound_source_native_bundle(tmp_path, capsys):
    source, eaf, evidence = _source_tree(tmp_path)
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps({"media_paths": {MEDIA_URL: "sample.mp4"}}),
        encoding="utf-8",
    )
    output = tmp_path / "inspection"
    auxiliary = source / "alternate-video.bin"
    auxiliary.write_bytes(b"publisher alternate rendition")
    exit_code = eaf_module.main([
        str(eaf),
        "--source-root", str(source),
        "--license-evidence", str(evidence),
        "--binding-config", str(binding),
        "--auxiliary-evidence", str(auxiliary),
        "--output", str(output),
    ])
    assert exit_code == 0
    printed_hash = capsys.readouterr().out.strip()
    assert len(printed_hash) == 64
    assert all(character in "0123456789abcdef" for character in printed_hash)
    manifest = load_eaf_manifest(
        (output / "source-native-manifest.json").read_bytes())
    assert hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() == printed_hash
    assert manifest["auxiliary_evidence_files"][0]["relative_path"] == auxiliary.name


def test_auxiliary_evidence_is_unique_in_root_and_hash_bound(tmp_path):
    source, eaf, evidence = _source_tree(tmp_path)
    alternate = source / "alternate.mp4"
    alternate.write_bytes(b"alternate rendition")
    second_alternate = source / "second-alternate.mp4"
    second_alternate.write_bytes(b"second alternate rendition")
    result = ingest_eaf_reference(
        eaf,
        source,
        license_evidence_path=evidence,
        media_paths={MEDIA_URL: "sample.mp4"},
        auxiliary_evidence_paths=[alternate, second_alternate],
    )
    assert result.auxiliary_evidence_files[0].sha256 == hashlib.sha256(
        alternate.read_bytes()).hexdigest()
    unexpected = source / "unexpected.txt"
    unexpected.write_text("unaccounted", encoding="utf-8")
    with pytest.raises(ValueError, match="source-root inventory mismatch"):
        ingest_eaf_reference(
            eaf,
            source,
            license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"},
            auxiliary_evidence_paths=[alternate, second_alternate],
        )
    unexpected.unlink()
    with pytest.raises(ValueError, match="unique and non-primary"):
        ingest_eaf_reference(
            eaf,
            source,
            license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"},
            auxiliary_evidence_paths=[alternate, alternate],
        )
    with pytest.raises(ValueError, match="unique and non-primary"):
        ingest_eaf_reference(
            eaf,
            source,
            license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"},
            auxiliary_evidence_paths=[eaf],
        )
    with pytest.raises(ValueError, match="must not duplicate bound media"):
        ingest_eaf_reference(
            eaf,
            source,
            license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"},
            auxiliary_evidence_paths=[source / "sample.mp4"],
        )
    with pytest.raises(ValueError, match="file-count limit"):
        ingest_eaf_reference(
            eaf,
            source,
            license_evidence_path=evidence,
            media_paths={MEDIA_URL: "sample.mp4"},
            auxiliary_evidence_paths=[alternate, second_alternate],
            limits=replace(EAFIngestionLimits(), max_auxiliary_files=1),
        )
