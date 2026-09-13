"""Secure, source-native ingestion for ELAN Annotation Format (EAF) files.

The records in this module are linguistic-reference evidence, not project SIR,
How2Sign labels, or training targets.  The original XML bytes and every bound
media/license file are content-addressed.  Parsing is deliberately fail-closed:
DTD/entity/XInclude/processing-instruction input, unrecognised EAF structure,
ambiguous references, unsafe paths, and invalid temporal structure are rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from ..reproducibility import canonical_json_bytes, implementation_identity
from .exporter import decode_video_clock
from .how2sign_audit import StableFileDigest, assert_file_unchanged, stable_sha256


EAF_INGESTION_SCHEMA_VERSION = 2
SOURCE_LABEL_SEMANTICS = "publisher_defined_source_annotation"
PROJECT_MAPPING_STATUS = "unmapped_not_project_sir"
SUPPORTED_EAF_VERSIONS = frozenset({"3.0"})
EAF_SCHEMA_URL = "https://www.mpi.nl/tools/elan/EAFv3.0.xsd"
EAF_SCHEMA_SHA256 = "59e76f90d5840813314b1e635480d044287b811ebd63a40e25cb4e73c6ad72bb"
MAX_BINDING_CONFIG_BYTES = 1024 * 1024
_XSI_SCHEMA_LOCATION = (
    "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_XML_DECLARATION_RE = re.compile(
    br"\A(?:\xef\xbb\xbf)?\s*<\?xml\s[^?]*\?>", re.DOTALL,
)
_FORBIDDEN_XML_RE = re.compile(
    br"<!DOCTYPE|<!ENTITY|<!\[CDATA\[|<!--|<\?(?!xml\s)", re.IGNORECASE,
)
_BIDI_CONTROLS = frozenset(
    chr(value) for value in (
        0x061C, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
        0x2066, 0x2067, 0x2068, 0x2069,
    )
)


@dataclass(frozen=True)
class EAFIngestionLimits:
    """Hard resource limits applied before semantic interpretation."""

    max_xml_bytes: int = 16 * 1024 * 1024
    max_depth: int = 32
    max_elements: int = 500_000
    max_attributes_per_element: int = 32
    max_attribute_chars: int = 16_384
    max_text_chars: int = 1_000_000
    max_total_text_chars: int = 32_000_000
    max_tiers: int = 2_048
    max_annotations: int = 400_000
    max_media_files: int = 16
    max_auxiliary_files: int = 64
    max_source_files: int = 128

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SourceFileIdentity:
    relative_path: str
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int

    def __post_init__(self) -> None:
        _safe_relative_path(self.relative_path)
        _require_sha256("source file sha256", self.sha256)
        for name in ("size", "device", "inode", "mtime_ns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"source file {name} must be a non-negative integer")

    @classmethod
    def from_digest(
        cls, path: Path, root: Path, digest: StableFileDigest,
    ) -> "SourceFileIdentity":
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return cls(relative.as_posix(), digest.sha256, digest.size, digest.device,
                   digest.inode, digest.mtime_ns)


@dataclass(frozen=True)
class SourceElement:
    """Lossless EAF data-tree node, excluding insignificant indentation."""

    tag: str
    attributes: tuple[tuple[str, str], ...]
    text: str
    children: tuple["SourceElement", ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "attributes": [[key, value] for key, value in self.attributes],
            "text": self.text,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        limits: EAFIngestionLimits | None = None,
    ) -> "SourceElement":
        active_limits = limits or EAFIngestionLimits()
        active_limits.validate()
        element_count = 0
        total_text_chars = 0

        def parse(node: object, depth: int) -> "SourceElement":
            nonlocal element_count, total_text_chars
            element_count += 1
            if depth > active_limits.max_depth:
                raise ValueError("manifest EAF tree exceeds the nesting-depth limit")
            if element_count > active_limits.max_elements:
                raise ValueError("manifest EAF tree exceeds the element-count limit")
            if not isinstance(node, dict) or set(node) != {
                    "tag", "attributes", "text", "children"}:
                raise ValueError("source element fields are invalid")
            attributes = node["attributes"]
            children = node["children"]
            if not isinstance(attributes, list) or not isinstance(children, list):
                raise ValueError("source element attributes and children must be lists")
            if len(attributes) > active_limits.max_attributes_per_element:
                raise ValueError("manifest EAF element exceeds the attribute-count limit")
            parsed_attributes: list[tuple[str, str]] = []
            for item in attributes:
                if (not isinstance(item, list) or len(item) != 2
                        or not all(isinstance(part, str) for part in item)):
                    raise ValueError("source element attributes are malformed")
                if any(len(part) > active_limits.max_attribute_chars for part in item):
                    raise ValueError("manifest EAF attribute exceeds the character limit")
                parsed_attributes.append((item[0], item[1]))
            if len({key for key, _ in parsed_attributes}) != len(parsed_attributes):
                raise ValueError("source element contains duplicate attributes")
            tag = node["tag"]
            text = node["text"]
            if not isinstance(tag, str) or not isinstance(text, str):
                raise ValueError("source element tag and text must be strings")
            if len(text) > active_limits.max_text_chars:
                raise ValueError("manifest EAF text node exceeds the character limit")
            total_text_chars += len(text)
            if total_text_chars > active_limits.max_total_text_chars:
                raise ValueError("manifest EAF tree exceeds the total-text limit")
            return cls(
                tag,
                tuple(parsed_attributes),
                text,
                tuple(parse(child, depth + 1) for child in children),
            )

        return parse(value, 1)

    def attribute(self, name: str) -> str | None:
        for key, value in self.attributes:
            if key == name:
                return value
        return None


@dataclass(frozen=True)
class TimeSlot:
    time_slot_id: str
    time_value_ms: int | None
    source_order: int


@dataclass(frozen=True)
class SourceAnnotation:
    annotation_id: str
    tier_id: str
    kind: str
    value: str
    source_order: int
    begin_time_slot_ref: str | None
    end_time_slot_ref: str | None
    begin_ms: int | None
    end_ms: int | None
    annotation_ref: str | None
    previous_annotation: str | None
    controlled_vocabulary_entry_ref: str | None
    language_ref: str | None
    external_refs: tuple[str, ...]


@dataclass(frozen=True)
class SourceTier:
    tier_id: str
    linguistic_type_ref: str
    participant: str | None
    annotator: str | None
    parent_ref: str | None
    language_ref: str | None
    source_order: int
    annotations: tuple[SourceAnnotation, ...]


@dataclass(frozen=True)
class MediaDescriptor:
    source_order: int
    media_url: str
    relative_media_url: str | None
    mime_type: str
    time_origin_ms: int


@dataclass(frozen=True)
class MediaBinding:
    descriptor_order: int
    descriptor_media_url: str
    file: SourceFileIdentity
    frame_count: int
    width: int
    height: int
    first_pts_ms: float
    last_pts_ms: float
    declared_end_ms: float
    duration_source: str

    def __post_init__(self) -> None:
        if (not isinstance(self.descriptor_order, int)
                or isinstance(self.descriptor_order, bool) or self.descriptor_order < 0):
            raise ValueError("media descriptor order must be a non-negative integer")
        if not isinstance(self.descriptor_media_url, str) or not self.descriptor_media_url:
            raise ValueError("media descriptor URL is required")
        if not isinstance(self.file, SourceFileIdentity):
            raise ValueError("media binding requires a validated source-file identity")
        for name in ("frame_count", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"media {name} must be a positive integer")
        times = (self.first_pts_ms, self.last_pts_ms, self.declared_end_ms)
        if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) for value in times)
                or not self.first_pts_ms <= self.last_pts_ms <= self.declared_end_ms):
            raise ValueError("media timeline bounds are invalid")
        if self.duration_source not in {"stream_duration", "container_duration"}:
            raise ValueError("media duration source is invalid")

    @property
    def observed_time_upper_bound_ms(self) -> float:
        """Declared stream/container end; never inferred from a nominal frame rate."""
        return self.declared_end_ms


@dataclass(frozen=True)
class EAFDocument:
    root: SourceElement
    time_slots: tuple[TimeSlot, ...]
    tiers: tuple[SourceTier, ...]
    media_descriptors: tuple[MediaDescriptor, ...]

    @property
    def annotations(self) -> tuple[SourceAnnotation, ...]:
        return tuple(annotation for tier in self.tiers for annotation in tier.annotations)


@dataclass(frozen=True)
class EAFIngestionResult:
    eaf_file: SourceFileIdentity
    license_evidence_file: SourceFileIdentity
    auxiliary_evidence_files: tuple[SourceFileIdentity, ...]
    document: EAFDocument
    media_bindings: tuple[MediaBinding, ...]
    implementation: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EAF_INGESTION_SCHEMA_VERSION,
            "kind": "source_native_elan_reference",
            "source_label_semantics": SOURCE_LABEL_SEMANTICS,
            "project_mapping_status": PROJECT_MAPPING_STATUS,
            "training_target_authorized": False,
            "linguistically_validated_by_project": False,
            "eaf_schema_url": EAF_SCHEMA_URL,
            "eaf_schema_sha256": EAF_SCHEMA_SHA256,
            "eaf_file": asdict(self.eaf_file),
            "license_evidence_file": asdict(self.license_evidence_file),
            "auxiliary_evidence_files": [
                asdict(item) for item in self.auxiliary_evidence_files],
            "root": self.document.root.to_dict(),
            "media_bindings": [asdict(binding) for binding in self.media_bindings],
            "implementation": dict(self.implementation),
        }

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_manifest())).hexdigest()


_ELEMENT_RULES: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "ANNOTATION_DOCUMENT": (
        frozenset({"AUTHOR", "DATE", "VERSION", "FORMAT", _XSI_SCHEMA_LOCATION}),
        frozenset({"AUTHOR", "DATE", "VERSION"}),
    ),
    "LICENSE": (frozenset({"LICENSE_URL"}), frozenset()),
    "HEADER": (frozenset({"MEDIA_FILE", "TIME_UNITS"}), frozenset()),
    "MEDIA_DESCRIPTOR": (
        frozenset({"MEDIA_URL", "RELATIVE_MEDIA_URL", "MIME_TYPE", "TIME_ORIGIN",
                   "EXTRACTED_FROM"}),
        frozenset({"MEDIA_URL", "MIME_TYPE"}),
    ),
    "LINKED_FILE_DESCRIPTOR": (
        frozenset({"LINK_URL", "RELATIVE_LINK_URL", "MIME_TYPE", "TIME_ORIGIN",
                   "ASSOCIATED_WITH"}),
        frozenset({"LINK_URL", "MIME_TYPE"}),
    ),
    "PROPERTY": (frozenset({"NAME"}), frozenset()),
    "TIME_ORDER": (frozenset(), frozenset()),
    "TIME_SLOT": (frozenset({"TIME_SLOT_ID", "TIME_VALUE"}),
                  frozenset({"TIME_SLOT_ID"})),
    "TIER": (
        frozenset({"TIER_ID", "PARTICIPANT", "ANNOTATOR", "LINGUISTIC_TYPE_REF",
                   "DEFAULT_LOCALE", "PARENT_REF", "EXT_REF", "LANG_REF"}),
        frozenset({"TIER_ID", "LINGUISTIC_TYPE_REF"}),
    ),
    "ANNOTATION": (frozenset(), frozenset()),
    "ALIGNABLE_ANNOTATION": (
        frozenset({"ANNOTATION_ID", "TIME_SLOT_REF1", "TIME_SLOT_REF2", "SVG_REF",
                   "EXT_REF", "LANG_REF", "CVE_REF"}),
        frozenset({"ANNOTATION_ID", "TIME_SLOT_REF1", "TIME_SLOT_REF2"}),
    ),
    "REF_ANNOTATION": (
        frozenset({"ANNOTATION_ID", "ANNOTATION_REF", "PREVIOUS_ANNOTATION",
                   "EXT_REF", "LANG_REF", "CVE_REF"}),
        frozenset({"ANNOTATION_ID", "ANNOTATION_REF"}),
    ),
    "ANNOTATION_VALUE": (frozenset(), frozenset()),
    "LINGUISTIC_TYPE": (
        frozenset({"LINGUISTIC_TYPE_ID", "TIME_ALIGNABLE", "CONSTRAINTS",
                   "GRAPHIC_REFERENCES", "CONTROLLED_VOCABULARY_REF", "EXT_REF",
                   "LEXICON_REF"}),
        frozenset({"LINGUISTIC_TYPE_ID"}),
    ),
    "LOCALE": (frozenset({"LANGUAGE_CODE", "COUNTRY_CODE", "VARIANT"}),
               frozenset({"LANGUAGE_CODE"})),
    "LANGUAGE": (frozenset({"LANG_ID", "LANG_DEF", "LANG_LABEL"}),
                 frozenset({"LANG_ID"})),
    "CONSTRAINT": (frozenset({"STEREOTYPE", "DESCRIPTION"}),
                   frozenset({"STEREOTYPE"})),
    "CONTROLLED_VOCABULARY": (frozenset({"CV_ID", "DESCRIPTION", "EXT_REF"}),
                              frozenset({"CV_ID"})),
    "DESCRIPTION": (frozenset({"LANG_REF"}), frozenset({"LANG_REF"})),
    "CV_ENTRY_ML": (frozenset({"CVE_ID", "EXT_REF"}), frozenset({"CVE_ID"})),
    "CVE_VALUE": (frozenset({"LANG_REF", "DESCRIPTION"}), frozenset({"LANG_REF"})),
    "CV_ENTRY": (frozenset({"CVE_ID", "DESCRIPTION", "EXT_REF"}), frozenset()),
    "LEXICON_REF": (
        frozenset({"LEX_REF_ID", "NAME", "TYPE", "URL", "LEXICON_ID",
                   "LEXICON_NAME", "DATCAT_ID", "DATCAT_NAME"}),
        frozenset({"LEX_REF_ID", "NAME", "TYPE", "URL", "LEXICON_ID",
                   "LEXICON_NAME"}),
    ),
    "REF_LINK_SET": (
        frozenset({"LINK_SET_ID", "LINK_SET_NAME", "EXT_REF", "LANG_REF", "CV_REF"}),
        frozenset({"LINK_SET_ID"}),
    ),
    "CROSS_REF_LINK": (
        frozenset({"REF_LINK_ID", "REF1", "REF2", "DIRECTIONALITY", "EXT_REF",
                   "LANG_REF", "CVE_REF", "REF_TYPE", "REF_LINK_NAME"}),
        frozenset({"REF_LINK_ID", "REF1", "REF2"}),
    ),
    "GROUP_REF_LINK": (
        frozenset({"REF_LINK_ID", "REFS", "EXT_REF", "LANG_REF", "CVE_REF",
                   "REF_TYPE", "REF_LINK_NAME"}),
        frozenset({"REF_LINK_ID", "REFS"}),
    ),
    "EXTERNAL_REF": (frozenset({"EXT_REF_ID", "TYPE", "VALUE"}),
                     frozenset({"EXT_REF_ID", "TYPE", "VALUE"})),
}

_CHILDREN: Mapping[str, frozenset[str]] = {
    "ANNOTATION_DOCUMENT": frozenset({
        "LICENSE", "HEADER", "TIME_ORDER", "TIER", "LINGUISTIC_TYPE", "LOCALE",
        "LANGUAGE", "CONSTRAINT", "CONTROLLED_VOCABULARY", "LEXICON_REF",
        "REF_LINK_SET", "EXTERNAL_REF",
    }),
    "HEADER": frozenset({"MEDIA_DESCRIPTOR", "LINKED_FILE_DESCRIPTOR", "PROPERTY"}),
    "TIME_ORDER": frozenset({"TIME_SLOT"}),
    "TIER": frozenset({"ANNOTATION"}),
    "ANNOTATION": frozenset({"ALIGNABLE_ANNOTATION", "REF_ANNOTATION"}),
    "ALIGNABLE_ANNOTATION": frozenset({"ANNOTATION_VALUE"}),
    "REF_ANNOTATION": frozenset({"ANNOTATION_VALUE"}),
    "CONTROLLED_VOCABULARY": frozenset({"DESCRIPTION", "CV_ENTRY_ML", "CV_ENTRY"}),
    "CV_ENTRY_ML": frozenset({"CVE_VALUE"}),
    "REF_LINK_SET": frozenset({"CROSS_REF_LINK", "GROUP_REF_LINK"}),
}

_EMPTY_ELEMENTS = frozenset({
    "MEDIA_DESCRIPTOR", "LINKED_FILE_DESCRIPTOR", "TIME_SLOT", "LINGUISTIC_TYPE",
    "LOCALE", "LANGUAGE", "CONSTRAINT", "LEXICON_REF", "CROSS_REF_LINK",
    "GROUP_REF_LINK", "EXTERNAL_REF",
})
_TEXT_ELEMENTS = frozenset({"LICENSE", "PROPERTY", "ANNOTATION_VALUE", "DESCRIPTION",
                            "CVE_VALUE", "CV_ENTRY", "CROSS_REF_LINK", "GROUP_REF_LINK"})
_ID_ATTRIBUTES = frozenset({
    "TIME_SLOT_ID", "ANNOTATION_ID", "LINGUISTIC_TYPE_ID", "LANGUAGE_CODE",
    "LANG_ID", "CV_ID", "CVE_ID", "LEX_REF_ID", "LINK_SET_ID", "REF_LINK_ID",
    "EXT_REF_ID",
})
_BOOLEAN_ATTRIBUTES = frozenset({"TIME_ALIGNABLE", "GRAPHIC_REFERENCES"})
_IDREFS_ATTRIBUTES = frozenset({"EXT_REF", "REFS"})


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _safe_relative_path(value: str) -> None:
    if not value or "\\" in value:
        raise ValueError("source path must be a non-empty POSIX relative path")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("source path must not be absolute or traverse directories")


def _validate_unicode(name: str, value: str, *, identifier: bool = False) -> None:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{name} must use NFC-normalized Unicode")
    if any(char in _BIDI_CONTROLS for char in value):
        raise ValueError(f"{name} contains a forbidden bidirectional control")
    if any(
        unicodedata.category(char) in {"Cc", "Cs"} and char not in "\t\n\r"
        for char in value
    ):
        raise ValueError(f"{name} contains a forbidden control or surrogate")
    if identifier and (not value or any(char.isspace() for char in value)):
        raise ValueError(f"{name} must be a non-empty XML identifier without whitespace")


def _split_idrefs(name: str, value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    references = tuple(value.split())
    if not references or len(references) != len(set(references)):
        raise ValueError(f"{name} must contain unique XML references")
    for reference in references:
        _validate_unicode(name, reference, identifier=True)
    return references


def _strict_int(name: str, value: str, *, minimum: int = 0) -> int:
    if not value or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name} must be a base-10 non-negative integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} is below its allowed minimum")
    return parsed


def _strict_signed_int(name: str, value: str) -> int:
    if re.fullmatch(r"[+-]?[0-9]+", value, flags=re.ASCII) is None:
        raise ValueError(f"{name} must be a base-10 integer")
    parsed = int(value)
    if not -(1 << 63) <= parsed < (1 << 63):
        raise ValueError(f"{name} exceeds the signed 64-bit XML Schema range")
    return parsed


def _read_stable_bytes(path: Path, root: Path, limit: int) -> tuple[bytes, StableFileDigest]:
    digest = stable_sha256(path, root)
    if digest.size > limit:
        raise ValueError(f"EAF exceeds the {limit}-byte input limit")
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    if len(payload) != digest.size or hashlib.sha256(payload).hexdigest() != digest.sha256:
        raise RuntimeError("EAF changed between hashing and parsing")
    assert_file_unchanged(resolved, root, digest)
    return payload, digest


def _assert_file_content_unchanged(
    path: Path,
    root: Path,
    expected: StableFileDigest,
) -> None:
    current = stable_sha256(path, root)
    if current != expected:
        raise RuntimeError(f"source artifact content changed after hashing: {path}")


def _validate_complete_source_root(
    root: Path,
    expected_files: set[Path],
    *,
    max_files: int,
) -> None:
    discovered: set[Path] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"symlinked source-root entry is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported source-root entry: {path}")
        discovered.add(path.resolve(strict=True))
        if len(discovered) > max_files:
            raise ValueError("source root exceeds the configured file-count limit")
    if discovered != expected_files:
        unaccounted = sorted(
            path.relative_to(root).as_posix() for path in discovered - expected_files)
        absent = sorted(
            path.relative_to(root).as_posix() for path in expected_files - discovered)
        raise ValueError(
            f"source-root inventory mismatch; unaccounted={unaccounted}, absent={absent}")


def _parse_xml(payload: bytes, limits: EAFIngestionLimits) -> ET.Element:
    if b"\x00" in payload:
        raise ValueError("EAF contains a NUL byte")
    scan_payload = _XML_DECLARATION_RE.sub(b"", payload, count=1)
    forbidden = _FORBIDDEN_XML_RE.search(scan_payload)
    if forbidden is not None:
        raise ValueError("DTD, entity declarations, CDATA, comments, and XML PIs are forbidden")
    parser = ET.XMLPullParser(events=("start", "end"))
    depth = 0
    elements = 0
    total_text = 0
    root: ET.Element | None = None
    try:
        for offset in range(0, len(payload), 64 * 1024):
            parser.feed(payload[offset:offset + 64 * 1024])
            for event, element in parser.read_events():
                if event == "start":
                    depth += 1
                    elements += 1
                    if depth > limits.max_depth:
                        raise ValueError("EAF exceeds the nesting-depth limit")
                    if elements > limits.max_elements:
                        raise ValueError("EAF exceeds the element-count limit")
                    if len(element.attrib) > limits.max_attributes_per_element:
                        raise ValueError("EAF element exceeds the attribute-count limit")
                    for value in element.attrib.values():
                        if len(value) > limits.max_attribute_chars:
                            raise ValueError("EAF attribute exceeds the character limit")
                else:
                    text_length = len(element.text or "")
                    if text_length > limits.max_text_chars:
                        raise ValueError("EAF text node exceeds the character limit")
                    total_text += text_length
                    if total_text > limits.max_total_text_chars:
                        raise ValueError("EAF exceeds the total-text limit")
                    if depth == 1:
                        if root is not None:
                            raise ValueError("EAF contains multiple root elements")
                        root = element
                    depth -= 1
        parser.close()
    except ET.ParseError as error:
        raise ValueError(f"malformed EAF XML: {error}") from error
    if depth != 0:
        raise ValueError("EAF parser ended at a nonzero nesting depth")
    if root is None:
        raise ValueError("EAF contains no root element")
    return root


def _source_element(element: ET.Element) -> SourceElement:
    if not isinstance(element.tag, str) or element.tag.startswith("{"):
        raise ValueError("namespaced or nonstandard EAF elements are forbidden")
    tag = element.tag
    if tag not in _ELEMENT_RULES:
        raise ValueError(f"unsupported EAF element: {tag}")
    allowed, required = _ELEMENT_RULES[tag]
    attributes = tuple(element.attrib.items())
    attribute_names = {name for name, _ in attributes}
    unknown = attribute_names - allowed
    missing = required - attribute_names
    if unknown or missing:
        raise ValueError(
            f"{tag} attributes invalid; unknown={sorted(unknown)}, missing={sorted(missing)}")
    for name, value in attributes:
        _validate_unicode(f"{tag}.{name}", value, identifier=name in _ID_ATTRIBUTES)
        if name in _BOOLEAN_ATTRIBUTES and value not in {"true", "false", "1", "0"}:
            raise ValueError(f"{tag}.{name} must be an XML boolean")
        if name in _IDREFS_ATTRIBUTES:
            _split_idrefs(f"{tag}.{name}", value)
    allowed_children = _CHILDREN.get(tag, frozenset())
    children = tuple(_source_element(child) for child in list(element))
    if any(child.tag not in allowed_children for child in children):
        raise ValueError(f"{tag} contains an unsupported child element")
    if element.tail is not None and element.tail.strip():
        raise ValueError(f"{tag} contains unsupported mixed tail text")
    raw_text = element.text or ""
    if tag in _TEXT_ELEMENTS:
        text = raw_text
        _validate_unicode(f"{tag} text", text)
    else:
        if raw_text.strip():
            raise ValueError(f"{tag} contains unsupported mixed text")
        text = ""
    if tag in _EMPTY_ELEMENTS and children:
        raise ValueError(f"{tag} must not contain child elements")
    return SourceElement(tag, attributes, text, children)


def _validate_source_element_record(element: SourceElement) -> None:
    if element.tag not in _ELEMENT_RULES:
        raise ValueError(f"unsupported EAF element: {element.tag}")
    allowed, required = _ELEMENT_RULES[element.tag]
    names = {name for name, _ in element.attributes}
    if len(names) != len(element.attributes):
        raise ValueError("source element contains duplicate attributes")
    if names - allowed or required - names:
        raise ValueError(f"{element.tag} attributes are invalid")
    for name, value in element.attributes:
        _validate_unicode(
            f"{element.tag}.{name}", value, identifier=name in _ID_ATTRIBUTES)
        if name in _BOOLEAN_ATTRIBUTES and value not in {"true", "false", "1", "0"}:
            raise ValueError(f"{element.tag}.{name} must be an XML boolean")
        if name in _IDREFS_ATTRIBUTES:
            _split_idrefs(f"{element.tag}.{name}", value)
    allowed_children = _CHILDREN.get(element.tag, frozenset())
    if any(child.tag not in allowed_children for child in element.children):
        raise ValueError(f"{element.tag} contains an unsupported child element")
    if element.tag in _TEXT_ELEMENTS:
        _validate_unicode(f"{element.tag} text", element.text)
    elif element.text:
        raise ValueError(f"{element.tag} contains unsupported text")
    for child in element.children:
        _validate_source_element_record(child)


def _children(element: SourceElement, tag: str) -> tuple[SourceElement, ...]:
    return tuple(child for child in element.children if child.tag == tag)


def _required_attribute(element: SourceElement, name: str) -> str:
    value = element.attribute(name)
    if value is None:
        raise ValueError(f"{element.tag} is missing required attribute {name}")
    return value


def _validate_top_level(root: SourceElement) -> None:
    if root.tag != "ANNOTATION_DOCUMENT":
        raise ValueError("EAF root must be ANNOTATION_DOCUMENT")
    version = _required_attribute(root, "VERSION")
    if version not in SUPPORTED_EAF_VERSIONS:
        raise ValueError(f"unsupported EAF version: {version}")
    format_value = root.attribute("FORMAT")
    if format_value is not None and format_value not in SUPPORTED_EAF_VERSIONS:
        raise ValueError(f"unsupported EAF format: {format_value}")
    date_value = _required_attribute(root, "DATE")
    try:
        if "T" not in date_value:
            raise ValueError("date-only values are not XML Schema dateTime values")
        datetime.fromisoformat(date_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("EAF DATE must be an ISO-8601 date-time") from error
    headers = _children(root, "HEADER")
    time_orders = _children(root, "TIME_ORDER")
    if len(headers) != 1 or len(time_orders) != 1:
        raise ValueError("EAF must contain exactly one HEADER and one TIME_ORDER")
    header_units = headers[0].attribute("TIME_UNITS") or "milliseconds"
    if header_units != "milliseconds":
        raise ValueError("only millisecond EAF time units are supported")
    rank = {
        name: index for index, name in enumerate((
            "LICENSE", "HEADER", "TIME_ORDER", "TIER", "LINGUISTIC_TYPE", "LOCALE",
            "LANGUAGE", "CONSTRAINT", "CONTROLLED_VOCABULARY", "LEXICON_REF",
            "REF_LINK_SET", "EXTERNAL_REF",
        ))
    }
    ranks = [rank[child.tag] for child in root.children]
    if ranks != sorted(ranks):
        raise ValueError("EAF top-level elements violate schema order")
    header_rank = {"MEDIA_DESCRIPTOR": 0, "LINKED_FILE_DESCRIPTOR": 1, "PROPERTY": 2}
    header_children = headers[0].children
    header_ranks = [header_rank[child.tag] for child in header_children]
    if header_ranks != sorted(header_ranks):
        raise ValueError("EAF HEADER children violate schema order")
    for vocabulary in _children(root, "CONTROLLED_VOCABULARY"):
        cv_rank = {"DESCRIPTION": 0, "CV_ENTRY_ML": 1, "CV_ENTRY": 1}
        cv_ranks = [cv_rank[child.tag] for child in vocabulary.children]
        if cv_ranks != sorted(cv_ranks):
            raise ValueError("controlled-vocabulary children violate schema order")
        if any(child.tag == "CV_ENTRY_ML" and not _children(child, "CVE_VALUE")
               for child in vocabulary.children):
            raise ValueError("CV_ENTRY_ML must contain at least one CVE_VALUE")


def _unique_elements(root: SourceElement, tag: str, attribute: str) -> dict[str, SourceElement]:
    result: dict[str, SourceElement] = {}
    for element in _children(root, tag):
        key = _required_attribute(element, attribute)
        if key in result:
            raise ValueError(f"duplicate {tag} identifier: {key}")
        result[key] = element
    return result


def _parse_time_slots(root: SourceElement) -> tuple[TimeSlot, ...]:
    time_order = _children(root, "TIME_ORDER")[0]
    slots: list[TimeSlot] = []
    seen: set[str] = set()
    for order, element in enumerate(_children(time_order, "TIME_SLOT")):
        slot_id = _required_attribute(element, "TIME_SLOT_ID")
        if slot_id in seen:
            raise ValueError(f"duplicate TIME_SLOT_ID: {slot_id}")
        seen.add(slot_id)
        raw_value = element.attribute("TIME_VALUE")
        value = None if raw_value is None else _strict_int("TIME_VALUE", raw_value)
        slots.append(TimeSlot(slot_id, value, order))
    return tuple(slots)


def _parse_media_descriptors(root: SourceElement,
                             limits: EAFIngestionLimits) -> tuple[MediaDescriptor, ...]:
    header = _children(root, "HEADER")[0]
    elements = _children(header, "MEDIA_DESCRIPTOR")
    if len(elements) > limits.max_media_files:
        raise ValueError("EAF exceeds the media-descriptor limit")
    descriptors: list[MediaDescriptor] = []
    for order, element in enumerate(elements):
        raw_origin = element.attribute("TIME_ORIGIN")
        origin = (
            0 if raw_origin is None
            else _strict_signed_int("TIME_ORIGIN", raw_origin)
        )
        descriptors.append(MediaDescriptor(
            order,
            _required_attribute(element, "MEDIA_URL"),
            element.attribute("RELATIVE_MEDIA_URL"),
            _required_attribute(element, "MIME_TYPE"),
            origin,
        ))
    return tuple(descriptors)


def _annotation_from_element(
    element: SourceElement,
    tier_id: str,
    order: int,
    slots: Mapping[str, TimeSlot],
) -> SourceAnnotation:
    wrappers = element.children
    if len(wrappers) != 1 or wrappers[0].tag not in {
            "ALIGNABLE_ANNOTATION", "REF_ANNOTATION"}:
        raise ValueError("ANNOTATION must contain exactly one supported annotation")
    item = wrappers[0]
    values = _children(item, "ANNOTATION_VALUE")
    if len(values) != 1:
        raise ValueError("annotation must contain exactly one ANNOTATION_VALUE")
    annotation_id = _required_attribute(item, "ANNOTATION_ID")
    if item.tag == "ALIGNABLE_ANNOTATION":
        first_ref = _required_attribute(item, "TIME_SLOT_REF1")
        second_ref = _required_attribute(item, "TIME_SLOT_REF2")
        if first_ref not in slots or second_ref not in slots:
            raise ValueError(f"annotation {annotation_id} has a dangling time-slot reference")
        begin = slots[first_ref].time_value_ms
        end = slots[second_ref].time_value_ms
        if begin is not None and end is not None and end <= begin:
            raise ValueError(f"annotation {annotation_id} has a non-positive interval")
        annotation_ref = previous = None
        kind = "alignable"
    else:
        first_ref = second_ref = None
        begin = end = None
        annotation_ref = _required_attribute(item, "ANNOTATION_REF")
        previous = item.attribute("PREVIOUS_ANNOTATION")
        kind = "reference"
    return SourceAnnotation(
        annotation_id=annotation_id,
        tier_id=tier_id,
        kind=kind,
        value=values[0].text,
        source_order=order,
        begin_time_slot_ref=first_ref,
        end_time_slot_ref=second_ref,
        begin_ms=begin,
        end_ms=end,
        annotation_ref=annotation_ref,
        previous_annotation=previous,
        controlled_vocabulary_entry_ref=item.attribute("CVE_REF"),
        language_ref=item.attribute("LANG_REF"),
        external_refs=_split_idrefs(
            f"annotation {annotation_id} EXT_REF", item.attribute("EXT_REF")),
    )


def _parse_tiers(root: SourceElement, slots: Sequence[TimeSlot],
                 limits: EAFIngestionLimits) -> tuple[SourceTier, ...]:
    tier_elements = _children(root, "TIER")
    if len(tier_elements) > limits.max_tiers:
        raise ValueError("EAF exceeds the tier-count limit")
    slot_map = {slot.time_slot_id: slot for slot in slots}
    tiers: list[SourceTier] = []
    seen_tiers: set[str] = set()
    seen_annotations: set[str] = set()
    annotation_count = 0
    for tier_order, element in enumerate(tier_elements):
        tier_id = _required_attribute(element, "TIER_ID")
        if tier_id in seen_tiers:
            raise ValueError(f"duplicate TIER_ID: {tier_id}")
        seen_tiers.add(tier_id)
        annotations: list[SourceAnnotation] = []
        previous_end: int | None = None
        for order, wrapper in enumerate(_children(element, "ANNOTATION")):
            annotation_count += 1
            if annotation_count > limits.max_annotations:
                raise ValueError("EAF exceeds the annotation-count limit")
            annotation = _annotation_from_element(wrapper, tier_id, order, slot_map)
            if annotation.annotation_id in seen_annotations:
                raise ValueError(f"duplicate ANNOTATION_ID: {annotation.annotation_id}")
            seen_annotations.add(annotation.annotation_id)
            if annotation.kind == "alignable":
                if (annotation.begin_ms is not None
                        and previous_end is not None
                        and annotation.begin_ms < previous_end):
                    raise ValueError(f"tier {tier_id} contains overlapping alignable annotations")
                if annotation.end_ms is not None:
                    previous_end = annotation.end_ms
            annotations.append(annotation)
        tiers.append(SourceTier(
            tier_id=tier_id,
            linguistic_type_ref=_required_attribute(element, "LINGUISTIC_TYPE_REF"),
            participant=element.attribute("PARTICIPANT"),
            annotator=element.attribute("ANNOTATOR"),
            parent_ref=element.attribute("PARENT_REF"),
            language_ref=element.attribute("LANG_REF"),
            source_order=tier_order,
            annotations=tuple(annotations),
        ))
    return tuple(tiers)


def _assert_acyclic(edges: Mapping[str, str], *, graph_name: str) -> None:
    for start in edges:
        visited: set[str] = set()
        current: str | None = start
        while current in edges:
            if current in visited:
                raise ValueError(f"{graph_name} contains a cycle involving {current}")
            visited.add(current)
            current = edges[current]


def _assert_adjacency_acyclic(
    adjacency: Mapping[str, Sequence[str]], *, graph_name: str,
) -> None:
    state: dict[str, int] = {}

    def visit(node: str) -> None:
        status = state.get(node, 0)
        if status == 1:
            raise ValueError(f"{graph_name} contains a cycle involving {node}")
        if status == 2:
            return
        state[node] = 1
        for target in adjacency.get(node, ()):
            visit(target)
        state[node] = 2

    for node in adjacency:
        visit(node)


def _validate_references(root: SourceElement, tiers: Sequence[SourceTier]) -> None:
    tier_map = {tier.tier_id: tier for tier in tiers}
    linguistic_types = _unique_elements(root, "LINGUISTIC_TYPE", "LINGUISTIC_TYPE_ID")
    languages = _unique_elements(root, "LANGUAGE", "LANG_ID")
    controlled_vocabularies = _unique_elements(root, "CONTROLLED_VOCABULARY", "CV_ID")
    lexicons = _unique_elements(root, "LEXICON_REF", "LEX_REF_ID")
    external_refs = _unique_elements(root, "EXTERNAL_REF", "EXT_REF_ID")
    locales = _unique_elements(root, "LOCALE", "LANGUAGE_CODE")
    constraints = _unique_elements(root, "CONSTRAINT", "STEREOTYPE")
    annotations = {item.annotation_id: item for tier in tiers for item in tier.annotations}

    cv_entries: dict[str, frozenset[str]] = {}
    for cv_id, vocabulary in controlled_vocabularies.items():
        entries: set[str] = set()
        for entry in vocabulary.children:
            if entry.tag not in {"CV_ENTRY", "CV_ENTRY_ML"}:
                continue
            entry_id = entry.attribute("CVE_ID")
            if entry_id is None:
                continue
            if entry_id in entries:
                raise ValueError(f"controlled vocabulary {cv_id} has duplicate CVE_ID")
            entries.add(entry_id)
        cv_entries[cv_id] = frozenset(entries)

    for language_element in (
        child for vocabulary in controlled_vocabularies.values()
        for child in vocabulary.children
        if child.tag == "DESCRIPTION"
    ):
        language_ref = _required_attribute(language_element, "LANG_REF")
        if language_ref not in languages:
            raise ValueError(
                "controlled-vocabulary description has a dangling language reference")
    for vocabulary in controlled_vocabularies.values():
        for entry in _children(vocabulary, "CV_ENTRY_ML"):
            for value in _children(entry, "CVE_VALUE"):
                language_ref = _required_attribute(value, "LANG_REF")
                if language_ref not in languages:
                    raise ValueError(
                        "controlled-vocabulary value has a dangling language reference")

    tier_edges: dict[str, str] = {}
    annotation_edges: dict[str, str] = {}
    previous_edges: dict[str, str] = {}
    for tier in tiers:
        if tier.linguistic_type_ref not in linguistic_types:
            raise ValueError(f"tier {tier.tier_id} has a dangling linguistic-type reference")
        if tier.parent_ref is not None:
            if tier.parent_ref not in tier_map:
                raise ValueError(f"tier {tier.tier_id} has a dangling parent reference")
            tier_edges[tier.tier_id] = tier.parent_ref
        if tier.language_ref is not None and tier.language_ref not in languages:
            raise ValueError(f"tier {tier.tier_id} has a dangling language reference")
        tier_element = next(
            item for item in _children(root, "TIER")
            if _required_attribute(item, "TIER_ID") == tier.tier_id
        )
        locale_ref = tier_element.attribute("DEFAULT_LOCALE")
        if locale_ref is not None and locale_ref not in locales:
            raise ValueError(f"tier {tier.tier_id} has a dangling locale reference")
        for ext_ref in _split_idrefs(
                f"tier {tier.tier_id} EXT_REF", tier_element.attribute("EXT_REF")):
            if ext_ref not in external_refs:
                raise ValueError(f"tier {tier.tier_id} has a dangling external reference")
        linguistic_type = linguistic_types[tier.linguistic_type_ref]
        cv_ref = linguistic_type.attribute("CONTROLLED_VOCABULARY_REF")
        for annotation in tier.annotations:
            if annotation.annotation_ref is not None:
                if annotation.annotation_ref not in annotations:
                    raise ValueError(
                        f"annotation {annotation.annotation_id} has a dangling "
                        "annotation reference")
                target = annotations[annotation.annotation_ref]
                if tier.parent_ref is None or target.tier_id != tier.parent_ref:
                    raise ValueError(
                        f"annotation {annotation.annotation_id} does not reference its parent tier")
                annotation_edges[annotation.annotation_id] = annotation.annotation_ref
            if annotation.previous_annotation is not None:
                if annotation.previous_annotation not in annotations:
                    raise ValueError(
                        f"annotation {annotation.annotation_id} has a dangling previous reference")
                target = annotations[annotation.previous_annotation]
                if target.tier_id != tier.tier_id or target.kind != "reference":
                    raise ValueError(
                        f"annotation {annotation.annotation_id} has an invalid previous reference")
                previous_edges[annotation.annotation_id] = annotation.previous_annotation
            if annotation.language_ref is not None and annotation.language_ref not in languages:
                raise ValueError(
                    f"annotation {annotation.annotation_id} has a dangling language reference")
            if any(reference not in external_refs for reference in annotation.external_refs):
                raise ValueError(
                    f"annotation {annotation.annotation_id} has a dangling external reference")
            cve_ref = annotation.controlled_vocabulary_entry_ref
            if cve_ref is not None:
                if cv_ref is None or cve_ref not in cv_entries.get(cv_ref, frozenset()):
                    raise ValueError(
                        f"annotation {annotation.annotation_id} has an invalid CVE_REF")

    _assert_acyclic(tier_edges, graph_name="tier-parent graph")
    _assert_acyclic(annotation_edges, graph_name="annotation-reference graph")
    _assert_acyclic(previous_edges, graph_name="previous-annotation graph")

    for element in linguistic_types.values():
        cv_ref = element.attribute("CONTROLLED_VOCABULARY_REF")
        if cv_ref is not None and cv_ref not in controlled_vocabularies:
            raise ValueError("LINGUISTIC_TYPE has a dangling controlled-vocabulary reference")
        lex_ref = element.attribute("LEXICON_REF")
        if lex_ref is not None and lex_ref not in lexicons:
            raise ValueError("LINGUISTIC_TYPE has a dangling lexicon reference")
        for ext_ref in _split_idrefs(
                "LINGUISTIC_TYPE.EXT_REF", element.attribute("EXT_REF")):
            if ext_ref not in external_refs:
                raise ValueError("LINGUISTIC_TYPE has a dangling external reference")
        constraint_ref = element.attribute("CONSTRAINTS")
        if constraint_ref is not None and constraint_ref not in constraints:
            raise ValueError("LINGUISTIC_TYPE has a dangling constraint reference")

    for vocabulary in controlled_vocabularies.values():
        ext_ref = vocabulary.attribute("EXT_REF")
        if ext_ref is not None and ext_ref not in external_refs:
            raise ValueError("CONTROLLED_VOCABULARY has a dangling external reference")

    link_sets = _unique_elements(root, "REF_LINK_SET", "LINK_SET_ID")
    link_elements = tuple(
        link for link_set in _children(root, "REF_LINK_SET")
        for link in link_set.children
    )
    links: dict[str, SourceElement] = {}
    for link in link_elements:
        link_id = _required_attribute(link, "REF_LINK_ID")
        if link_id in links or link_id in annotations:
            raise ValueError(f"duplicate annotation/reference-link identifier: {link_id}")
        links[link_id] = link
    link_targets = set(annotations) | set(links)
    adjacency: dict[str, tuple[str, ...]] = {}
    for link_set in link_sets.values():
        language_ref = link_set.attribute("LANG_REF")
        if language_ref is not None and language_ref not in languages:
            raise ValueError("REF_LINK_SET has a dangling language reference")
        for ext_ref in _split_idrefs("REF_LINK_SET.EXT_REF", link_set.attribute("EXT_REF")):
            if ext_ref not in external_refs:
                raise ValueError("REF_LINK_SET has a dangling external reference")
        cv_ref = link_set.attribute("CV_REF")
        if cv_ref is not None and cv_ref not in controlled_vocabularies:
            raise ValueError("REF_LINK_SET has a dangling controlled-vocabulary reference")
        for link in link_set.children:
            link_language = link.attribute("LANG_REF")
            if link_language is not None and link_language not in languages:
                raise ValueError("reference link has a dangling language reference")
            for ext_ref in _split_idrefs(
                    "reference link EXT_REF", link.attribute("EXT_REF")):
                if ext_ref not in external_refs:
                    raise ValueError("reference link has a dangling external reference")
            cve_ref = link.attribute("CVE_REF")
            if cve_ref is not None and (
                    cv_ref is None or cve_ref not in cv_entries.get(cv_ref, frozenset())):
                raise ValueError("reference link has an invalid CVE_REF")
    for link_id, link in links.items():
        if link.tag == "CROSS_REF_LINK":
            targets = (_required_attribute(link, "REF1"), _required_attribute(link, "REF2"))
            directionality = link.attribute("DIRECTIONALITY")
            if directionality is not None and directionality not in {
                    "undirected", "unidirectional", "bidirectional"}:
                raise ValueError("CROSS_REF_LINK has invalid directionality")
        else:
            targets = _split_idrefs("GROUP_REF_LINK.REFS", _required_attribute(link, "REFS"))
            if len(targets) < 2:
                raise ValueError("GROUP_REF_LINK must reference at least two elements")
        if any(target not in link_targets for target in targets):
            raise ValueError(f"reference link {link_id} has a dangling target")
        adjacency[link_id] = tuple(target for target in targets if target in links)
    _assert_adjacency_acyclic(adjacency, graph_name="reference-link graph")


def parse_eaf_bytes(payload: bytes, limits: EAFIngestionLimits | None = None) -> EAFDocument:
    """Parse one immutable EAF byte string without resolving any external resource."""
    active_limits = limits or EAFIngestionLimits()
    active_limits.validate()
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("EAF payload must be non-empty immutable bytes")
    if len(payload) > active_limits.max_xml_bytes:
        raise ValueError("EAF exceeds the configured byte limit")
    raw_root = _parse_xml(payload, active_limits)
    root = _source_element(raw_root)
    _validate_top_level(root)
    slots = _parse_time_slots(root)
    descriptors = _parse_media_descriptors(root, active_limits)
    tiers = _parse_tiers(root, slots, active_limits)
    _validate_references(root, tiers)
    return EAFDocument(root, slots, tiers, descriptors)


def _bind_media(
    document: EAFDocument,
    source_root: Path,
    media_paths: Mapping[str, str | os.PathLike[str]],
) -> tuple[MediaBinding, ...]:
    if not isinstance(media_paths, Mapping):
        raise TypeError("media_paths must map exact descriptor URLs to local paths")
    if any(
        not isinstance(key, str) or not key
        or not isinstance(value, (str, os.PathLike))
        for key, value in media_paths.items()
    ):
        raise TypeError(
            "media_paths must map non-empty descriptor strings to filesystem paths")
    descriptor_urls = [item.media_url for item in document.media_descriptors]
    if len(descriptor_urls) != len(set(descriptor_urls)):
        raise ValueError("duplicate MEDIA_URL values make media binding ambiguous")
    if set(media_paths) != set(descriptor_urls):
        missing = sorted(set(descriptor_urls) - set(media_paths))
        unused = sorted(set(media_paths) - set(descriptor_urls))
        raise ValueError(f"media binding keys mismatch; missing={missing}, unused={unused}")
    bindings: list[MediaBinding] = []
    for descriptor in document.media_descriptors:
        candidate = Path(media_paths[descriptor.media_url])
        if not candidate.is_absolute():
            candidate = source_root / candidate
        if not descriptor.mime_type.casefold().startswith("video/"):
            raise ValueError(
                f"unsupported non-video media descriptor: {descriptor.mime_type}")
        digest = stable_sha256(candidate, source_root)
        clock = decode_video_clock(candidate)
        assert_file_unchanged(candidate, source_root, digest)
        import av
        with av.open(os.fspath(candidate), mode="r") as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                duration_ms = float(stream.duration * stream.time_base) * 1000.0
                duration_source = "stream_duration"
            elif container.duration is not None:
                duration_ms = float(container.duration) / 1000.0
                duration_source = "container_duration"
            else:
                raise ValueError("bound media has no declared stream or container duration")
        assert_file_unchanged(candidate, source_root, digest)
        _assert_file_content_unchanged(candidate, source_root, digest)
        timestamps_ms = clock.timestamps * 1000.0 + descriptor.time_origin_ms
        if timestamps_ms.size < 1 or not all(math.isfinite(value) for value in timestamps_ms):
            raise ValueError("bound media has no finite presentation timestamps")
        identity = SourceFileIdentity.from_digest(candidate, source_root, digest)
        declared_end_ms = float(timestamps_ms[0] + duration_ms)
        if (not math.isfinite(duration_ms) or duration_ms <= 0
                or declared_end_ms < float(timestamps_ms[-1])):
            raise ValueError("bound media has an invalid declared duration")
        bindings.append(MediaBinding(
            descriptor_order=descriptor.source_order,
            descriptor_media_url=descriptor.media_url,
            file=identity,
            frame_count=int(timestamps_ms.size),
            width=clock.width,
            height=clock.height,
            first_pts_ms=float(timestamps_ms[0]),
            last_pts_ms=float(timestamps_ms[-1]),
            declared_end_ms=declared_end_ms,
            duration_source=duration_source,
        ))
    return tuple(bindings)


def _validate_media_range(document: EAFDocument,
                          bindings: Sequence[MediaBinding]) -> None:
    annotations = [item for item in document.annotations if item.kind == "alignable"]
    if not annotations:
        return
    if not bindings:
        raise ValueError("timed EAF annotations require an exact media binding")
    for annotation in annotations:
        for binding in bindings:
            if ((annotation.begin_ms is not None
                 and annotation.begin_ms < binding.first_pts_ms)
                    or (annotation.end_ms is not None
                        and annotation.end_ms > binding.declared_end_ms)):
                raise ValueError(
                    f"annotation {annotation.annotation_id} is outside bound media range")


def _implementation_identity() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    return implementation_identity(
        (Path(__file__), Path(__file__).with_name("how2sign_audit.py"),
         Path(__file__).with_name("exporter.py")),
        repo_root=repo,
    )


def ingest_eaf_reference(
    eaf_path: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
    *,
    license_evidence_path: str | os.PathLike[str],
    media_paths: Mapping[str, str | os.PathLike[str]],
    auxiliary_evidence_paths: Sequence[str | os.PathLike[str]] = (),
    limits: EAFIngestionLimits | None = None,
) -> EAFIngestionResult:
    """Ingest an EAF as an unmapped reference bound to exact local evidence."""
    root = Path(source_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_root must be a directory")
    active_limits = limits or EAFIngestionLimits()
    active_limits.validate()
    eaf = Path(eaf_path)
    if not eaf.is_absolute():
        eaf = root / eaf
    payload, eaf_digest = _read_stable_bytes(eaf, root, active_limits.max_xml_bytes)
    if eaf.suffix.casefold() != ".eaf":
        raise ValueError("linguistic-reference input must use the .eaf extension")
    document = parse_eaf_bytes(payload, active_limits)
    evidence = Path(license_evidence_path)
    if not evidence.is_absolute():
        evidence = root / evidence
    evidence_digest = stable_sha256(evidence, root)
    if evidence_digest.size < 1:
        raise ValueError("license evidence file is empty")
    if isinstance(auxiliary_evidence_paths, (str, bytes, os.PathLike)):
        raise TypeError("auxiliary_evidence_paths must be a sequence of filesystem paths")
    if len(auxiliary_evidence_paths) > active_limits.max_auxiliary_files:
        raise ValueError("auxiliary evidence exceeds the configured file-count limit")
    auxiliary: list[tuple[Path, StableFileDigest]] = []
    seen_auxiliary_paths: set[Path] = set()
    primary_paths = {eaf.resolve(strict=True), evidence.resolve(strict=True)}
    for raw_path in auxiliary_evidence_paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved_candidate = candidate.resolve(strict=True)
        if resolved_candidate in primary_paths or resolved_candidate in seen_auxiliary_paths:
            raise ValueError("auxiliary evidence paths must be unique and non-primary")
        seen_auxiliary_paths.add(resolved_candidate)
        auxiliary.append((candidate, stable_sha256(candidate, root)))
    media = _bind_media(document, root, media_paths)
    auxiliary_relative_paths = {
        SourceFileIdentity.from_digest(path, root, digest).relative_path
        for path, digest in auxiliary
    }
    media_relative_paths = {binding.file.relative_path for binding in media}
    if auxiliary_relative_paths & media_relative_paths:
        raise ValueError("auxiliary evidence must not duplicate bound media")
    expected_files = primary_paths | seen_auxiliary_paths | {
        (root / binding.file.relative_path).resolve(strict=True)
        for binding in media
    }
    _validate_complete_source_root(
        root, expected_files, max_files=active_limits.max_source_files)
    _validate_media_range(document, media)
    assert_file_unchanged(eaf, root, eaf_digest)
    assert_file_unchanged(evidence, root, evidence_digest)
    _assert_file_content_unchanged(eaf, root, eaf_digest)
    _assert_file_content_unchanged(evidence, root, evidence_digest)
    for auxiliary_path, auxiliary_digest in auxiliary:
        _assert_file_content_unchanged(auxiliary_path, root, auxiliary_digest)
    return EAFIngestionResult(
        eaf_file=SourceFileIdentity.from_digest(eaf, root, eaf_digest),
        license_evidence_file=SourceFileIdentity.from_digest(
            evidence, root, evidence_digest),
        auxiliary_evidence_files=tuple(
            SourceFileIdentity.from_digest(path, root, digest)
            for path, digest in auxiliary
        ),
        document=document,
        media_bindings=media,
        implementation=_implementation_identity(),
    )


def _inspection_rows(result: EAFIngestionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in result.document.tiers:
        for annotation in tier.annotations:
            rows.append({
                "tier_order": tier.source_order,
                "tier_id": tier.tier_id,
                "participant": tier.participant or "",
                "annotator": tier.annotator or "",
                "annotation_order": annotation.source_order,
                "annotation_id": annotation.annotation_id,
                "kind": annotation.kind,
                "begin_time_slot_ref": annotation.begin_time_slot_ref or "",
                "end_time_slot_ref": annotation.end_time_slot_ref or "",
                "time_alignment_status": (
                    "aligned" if annotation.begin_ms is not None
                    and annotation.end_ms is not None
                    else "partially_aligned" if annotation.begin_ms is not None
                    or annotation.end_ms is not None
                    else "unaligned"
                ),
                "begin_ms": "" if annotation.begin_ms is None else annotation.begin_ms,
                "end_ms": "" if annotation.end_ms is None else annotation.end_ms,
                "annotation_ref": annotation.annotation_ref or "",
                "previous_annotation": annotation.previous_annotation or "",
                "source_value": annotation.value,
                "project_mapping_status": PROJECT_MAPPING_STATUS,
            })
    return rows


def _write_inspection_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "tier_order", "tier_id", "participant", "annotator", "annotation_order",
        "annotation_id", "kind", "begin_time_slot_ref", "end_time_slot_ref",
        "time_alignment_status", "begin_ms", "end_ms", "annotation_ref",
        "previous_annotation", "source_value", "project_mapping_status",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            safe_row = {
                key: _spreadsheet_safe(value) if isinstance(value, str) else value
                for key, value in row.items()
            }
            writer.writerow(safe_row)


def _spreadsheet_safe(value: str) -> str:
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_inspection_html(path: Path, result: EAFIngestionResult,
                           rows: Sequence[Mapping[str, Any]], source_root: Path) -> None:
    media_links = []
    for binding in result.media_bindings:
        absolute = source_root / binding.file.relative_path
        url = "file://" + quote(absolute.as_posix(), safe="/")
        media_links.append(
            f'<li><a href="{html.escape(url, quote=True)}">'
            f'{html.escape(binding.file.relative_path)}</a> '
            f'({html.escape(binding.file.sha256)})</li>')
    table_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row[field]))}</td>"
            for field in (
                "tier_id", "annotation_id", "kind", "begin_time_slot_ref",
                "end_time_slot_ref", "time_alignment_status", "begin_ms", "end_ms",
                "source_value", "project_mapping_status",
            )
        )
        table_rows.append(f"<tr>{cells}</tr>")
    markup = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Source-native EAF inspection</title>
<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #bbb;padding:.35rem;text-align:left;vertical-align:top}
code{overflow-wrap:anywhere}.warning{background:#fff3cd;padding:1rem}</style></head><body>
<h1>Source-native EAF inspection</h1>
<p class="warning">Publisher-defined annotations only. Unmapped to project SIR and not
authorized here as training targets or project-validated ASL.</p>
<p>Manifest SHA-256: <code>__MANIFEST_HASH__</code></p>
<h2>Bound media</h2><ul>__MEDIA__</ul>
<h2>Annotations</h2><table><thead><tr><th>Tier</th><th>Annotation</th><th>Kind</th>
<th>Begin slot</th><th>End slot</th><th>Alignment</th><th>Begin ms</th><th>End ms</th>
<th>Source value</th><th>Mapping</th></tr></thead>
<tbody>__ROWS__</tbody></table></body></html>
"""
    markup = markup.replace("__MANIFEST_HASH__", result.manifest_sha256)
    markup = markup.replace("__MEDIA__", "".join(media_links))
    markup = markup.replace("__ROWS__", "".join(table_rows))
    path.write_text(markup, encoding="utf-8", newline="\n")


def write_eaf_inspection_artifacts(
    result: EAFIngestionResult,
    output_dir: str | os.PathLike[str],
    source_root: str | os.PathLike[str],
) -> Path:
    """Atomically create a compact, immutable-by-hash inspection bundle."""
    output = Path(output_dir)
    root = Path(source_root).resolve(strict=True)
    if output.exists():
        raise FileExistsError("inspection output already exists; overwrite is forbidden")
    output_parent = output.parent.resolve(strict=True)
    if output_parent == root or root in output_parent.parents:
        raise ValueError("inspection artifacts must not be written inside the source root")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output_parent))
    try:
        manifest_path = temporary / "source-native-manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(result.to_manifest()))
        rows = _inspection_rows(result)
        _write_inspection_csv(temporary / "annotation-inspection.csv", rows)
        _write_inspection_html(
            temporary / "annotation-inspection.html", result, rows, root)
        files = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({"name": path.name, "sha256": digest, "size": path.stat().st_size})
        index = {
            "schema_version": EAF_INGESTION_SCHEMA_VERSION,
            "kind": "source_native_eaf_inspection_bundle",
            "manifest_sha256": result.manifest_sha256,
            "files": files,
        }
        (temporary / "artifact-index.json").write_bytes(canonical_json_bytes(index) + b"\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def load_eaf_manifest(payload: bytes, *, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    """Read only canonical manifests and reject duplicate/non-finite JSON values."""
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("manifest payload must be non-empty immutable bytes")
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1
            or len(payload) > max_bytes):
        raise ValueError("manifest exceeds the configured byte limit")

    def unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    try:
        value = json.loads(payload.decode("utf-8", errors="strict"),
                           object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError(f"malformed manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise ValueError("manifest bytes are not canonical")
    required = {
        "schema_version", "kind", "source_label_semantics", "project_mapping_status",
        "training_target_authorized", "linguistically_validated_by_project",
        "eaf_schema_url", "eaf_schema_sha256", "eaf_file", "license_evidence_file",
        "auxiliary_evidence_files", "root", "media_bindings", "implementation",
    }
    if set(value) != required:
        raise ValueError("manifest fields are invalid")
    if value["schema_version"] != EAF_INGESTION_SCHEMA_VERSION:
        raise ValueError("unsupported EAF ingestion schema")
    if value["kind"] != "source_native_elan_reference":
        raise ValueError("manifest kind is invalid")
    if value["source_label_semantics"] != SOURCE_LABEL_SEMANTICS:
        raise ValueError("source label semantics were altered")
    if value["project_mapping_status"] != PROJECT_MAPPING_STATUS:
        raise ValueError("project mapping status was altered")
    if (value["eaf_schema_url"] != EAF_SCHEMA_URL
            or value["eaf_schema_sha256"] != EAF_SCHEMA_SHA256):
        raise ValueError("EAF schema identity was altered")
    if value["training_target_authorized"] is not False \
            or value["linguistically_validated_by_project"] is not False:
        raise ValueError("manifest makes an unsupported validation or training claim")
    limits = EAFIngestionLimits()
    root = SourceElement.from_dict(value["root"], limits)
    _validate_source_element_record(root)
    _validate_top_level(root)
    slots = _parse_time_slots(root)
    descriptors = _parse_media_descriptors(root, limits)
    tiers = _parse_tiers(root, slots, limits)
    _validate_references(root, tiers)
    primary_relative_paths: set[str] = set()
    for field in ("eaf_file", "license_evidence_file"):
        item = value[field]
        required_file_fields = {"relative_path", "sha256", "size", "device", "inode",
                                "mtime_ns"}
        if not isinstance(item, dict) or set(item) != required_file_fields:
            raise ValueError(f"{field} fields are invalid")
        identity = SourceFileIdentity(**item)
        if identity.relative_path in primary_relative_paths:
            raise ValueError("primary source file paths must be unique")
        primary_relative_paths.add(identity.relative_path)
    auxiliary_evidence = value["auxiliary_evidence_files"]
    if not isinstance(auxiliary_evidence, list):
        raise ValueError("auxiliary_evidence_files must be a list")
    if len(auxiliary_evidence) > limits.max_auxiliary_files:
        raise ValueError("auxiliary evidence exceeds the configured file-count limit")
    auxiliary_paths: set[str] = set()
    for item in auxiliary_evidence:
        required_file_fields = {
            "relative_path", "sha256", "size", "device", "inode", "mtime_ns"}
        if not isinstance(item, dict) or set(item) != required_file_fields:
            raise ValueError("auxiliary evidence file fields are invalid")
        identity = SourceFileIdentity(**item)
        if identity.relative_path in auxiliary_paths:
            raise ValueError("duplicate auxiliary evidence path")
        if identity.relative_path in primary_relative_paths:
            raise ValueError("auxiliary evidence duplicates a primary source path")
        auxiliary_paths.add(identity.relative_path)
    media_bindings = value["media_bindings"]
    if not isinstance(media_bindings, list):
        raise ValueError("media_bindings must be a list")
    if len(media_bindings) != len(descriptors):
        raise ValueError("media bindings do not match EAF media descriptors")
    parsed_bindings: list[MediaBinding] = []
    for descriptor, item in zip(descriptors, media_bindings, strict=True):
        required_binding_fields = {
            "descriptor_order", "descriptor_media_url", "file", "frame_count", "width",
            "height", "first_pts_ms", "last_pts_ms", "declared_end_ms", "duration_source",
        }
        if not isinstance(item, dict) or set(item) != required_binding_fields:
            raise ValueError("media binding fields are invalid")
        file_value = item["file"]
        if not isinstance(file_value, dict):
            raise ValueError("media binding file identity is invalid")
        file_identity = SourceFileIdentity(**file_value)
        binding = MediaBinding(**{**item, "file": file_identity})
        if (binding.descriptor_order != descriptor.source_order
                or binding.descriptor_media_url != descriptor.media_url):
            raise ValueError("media binding does not match its EAF media descriptor")
        parsed_bindings.append(binding)
    if auxiliary_paths & {
            binding.file.relative_path for binding in parsed_bindings}:
        raise ValueError("auxiliary evidence duplicates bound media")
    document = EAFDocument(root, slots, tiers, descriptors)
    _validate_media_range(document, parsed_bindings)
    if not isinstance(value["implementation"], dict):
        raise ValueError("implementation identity must be an object")
    return value


def _load_binding_config(
    path: Path,
    *,
    max_bytes: int = MAX_BINDING_CONFIG_BYTES,
) -> dict[str, str]:
    if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
            or max_bytes < 1):
        raise ValueError("binding configuration byte limit must be a positive integer")
    root = path.parent.resolve(strict=True)
    digest = stable_sha256(path, root)
    if digest.size > max_bytes:
        raise ValueError("binding configuration exceeds the configured byte limit")
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    if len(payload) != digest.size or hashlib.sha256(payload).hexdigest() != digest.sha256:
        raise RuntimeError("binding configuration changed between hashing and parsing")
    assert_file_unchanged(resolved, root, digest)

    def unique_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate binding configuration key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(
            f"non-finite binding configuration value is forbidden: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
        raise ValueError(f"malformed binding configuration: {error}") from error
    if not isinstance(value, dict) or set(value) != {"media_paths"}:
        raise ValueError("binding configuration must contain only media_paths")
    media = value["media_paths"]
    if (not isinstance(media, dict) or any(
            not isinstance(key, str) or not key
            or not isinstance(item, str) or not item
            for key, item in media.items())):
        raise ValueError("media_paths must map non-empty strings to non-empty strings")
    for key, item in media.items():
        _validate_unicode("media descriptor URL", key)
        _validate_unicode("bound media path", item)
    _assert_file_content_unchanged(resolved, root, digest)
    return media


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eaf")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--license-evidence", required=True)
    parser.add_argument("--binding-config", required=True)
    parser.add_argument("--auxiliary-evidence", action="append", default=[])
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    result = ingest_eaf_reference(
        arguments.eaf,
        arguments.source_root,
        license_evidence_path=arguments.license_evidence,
        media_paths=_load_binding_config(Path(arguments.binding_config)),
        auxiliary_evidence_paths=arguments.auxiliary_evidence,
    )
    write_eaf_inspection_artifacts(result, arguments.output, arguments.source_root)
    print(result.manifest_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
