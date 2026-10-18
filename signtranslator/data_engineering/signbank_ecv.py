"""ASL Signbank external controlled vocabulary (ECV) and English-label gloss candidates.

The ECV is ELAN's XML export of the ASL Signbank lexicon: one entry per ID gloss
with its Signbank identifier and English keywords. Parsing is strict and resolves
nothing external: DTDs, entity declarations, CDATA, comments and processing
instructions are refused before the XML parser runs, so only XML's five predefined
entities and numeric character references can expand.

Matching an English sign label to ID glosses is a *candidate* relation, never an
identification. Two rules are kept apart because they carry different evidence:

* **stem** — the ID gloss, stripped of its variant marker (``ABOUTb``,
  ``CAT8`` -> ``ABOUT``, ``CAT``), equals the label in upper case;
* **keyword** — the label equals one of the entry's English keywords.

A label with exactly one stem candidate is ``unique_stem``; several are
``ambiguous_stem`` (lexical variants that only a person viewing the sign can
choose between); none but some keyword candidates is ``keyword_only``; otherwise
``unmatched``. No rule selects among candidates, and every mapping records
``requires_review=True``: two lexicons sharing an English label does not show that
the performed sign is the Signbank sign.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import xml.etree.ElementTree as ET

MAX_ECV_BYTES = 32 * 1024 * 1024
_XML_DECLARATION = re.compile(br"\A(?:\xef\xbb\xbf)?\s*<\?xml\s[^?]*\?>", re.DOTALL)
_FORBIDDEN = re.compile(br"<!DOCTYPE|<!ENTITY|<!\[CDATA\[|<!--|<\?(?!xml\s)", re.IGNORECASE)
# SLAASh ID glosses: upper-case tokens joined by hyphens, then an optional variant
# marker attached to the final letter: lower-case letters, handshape digits, or
# both (ABOUTb, ABOVEarc, CAT8, CHAMPb5). A hyphenated numeric token (TWENTY-1)
# is part of the stem, not a marker.
_TOKENS = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+)*")
_MARKED = re.compile(r"(.*?[A-Z])([a-z]*[0-9]*)")
_LABEL = re.compile(r"[a-z]+(?:[ '-][a-z]+)*")
MATCH_STATUSES = ("unique_stem", "ambiguous_stem", "keyword_only", "unmatched")


@dataclass(frozen=True)
class ECVEntry:
    cve_id: str
    gloss: str               # exactly as published, whitespace included
    keywords: tuple[str, ...]  # case-folded, order and duplicates as published
    description: str

    @property
    def stem(self) -> str | None:
        """Variant-free stem, or None when the gloss is outside the ID-gloss form."""
        marked = _MARKED.fullmatch(self.gloss)
        if marked and _TOKENS.fullmatch(marked.group(1)):
            return marked.group(1)
        return self.gloss if _TOKENS.fullmatch(self.gloss) else None


@dataclass(frozen=True)
class ECVLexicon:
    vocabulary_id: str
    language: str
    date: str
    sha256: str
    entries: tuple[ECVEntry, ...]
    external_refs: dict  # EXT_REF_ID -> resource URL, as declared

    def by_stem(self) -> dict[str, tuple[ECVEntry, ...]]:
        index: dict[str, list[ECVEntry]] = {}
        for entry in self.entries:
            if entry.stem is not None:
                index.setdefault(entry.stem, []).append(entry)
        return {key: tuple(value) for key, value in index.items()}

    def by_keyword(self) -> dict[str, tuple[ECVEntry, ...]]:
        index: dict[str, list[ECVEntry]] = {}
        for entry in self.entries:
            for keyword in dict.fromkeys(entry.keywords):
                index.setdefault(keyword, []).append(entry)
        return {key: tuple(value) for key, value in index.items()}


def _keywords(description: str) -> tuple[str, ...]:
    if not description:
        return ()
    return tuple(part.strip().casefold() for part in description.split(",") if part.strip())


def parse_ecv(payload: bytes) -> ECVLexicon:
    """Parse one ECV snapshot; every structural surprise is an error, not a skip."""
    if not isinstance(payload, bytes) or not payload:
        raise TypeError("ECV payload must be non-empty bytes")
    if len(payload) > MAX_ECV_BYTES:
        raise ValueError("ECV exceeds the byte limit")
    if b"\x00" in payload:
        raise ValueError("ECV contains a NUL byte")
    if _FORBIDDEN.search(_XML_DECLARATION.sub(b"", payload, count=1)):
        raise ValueError("DTD, entity declarations, CDATA, comments and XML PIs are forbidden")
    root = ET.fromstring(payload)
    if root.tag != "CV_RESOURCE":
        raise ValueError("ECV root must be CV_RESOURCE")
    languages = root.findall("LANGUAGE")
    vocabularies = root.findall("CONTROLLED_VOCABULARY")
    references = root.findall("EXTERNAL_REF")
    if len(languages) != 1 or len(vocabularies) != 1 or len(root) != 2 + len(references):
        raise ValueError("ECV must declare exactly one language and one vocabulary")
    external = {}
    for reference in references:
        ref_id, value = reference.get("EXT_REF_ID"), reference.get("VALUE")
        if not ref_id or not value or ref_id in external:
            raise ValueError("external references need unique ids and values")
        external[ref_id] = value
    language = languages[0].get("LANG_ID")
    if not language:
        raise ValueError("ECV language has no LANG_ID")
    vocabulary = vocabularies[0]
    vocabulary_id = vocabulary.get("CV_ID")
    if not vocabulary_id:
        raise ValueError("ECV vocabulary has no CV_ID")
    entries, ids, glosses = [], set(), set()
    for child in vocabulary:
        if child.tag == "DESCRIPTION":
            continue
        if child.tag != "CV_ENTRY_ML":
            raise ValueError(f"unexpected vocabulary element {child.tag}")
        values = list(child)
        if len(values) != 1 or values[0].tag != "CVE_VALUE":
            raise ValueError("each entry must hold exactly one CVE_VALUE")
        value = values[0]
        if value.get("LANG_REF") != language:
            raise ValueError("entry value is not in the declared language")
        ext_ref = child.get("EXT_REF")
        if ext_ref is not None and ext_ref not in external:
            raise ValueError(f"entry refers to an undeclared external reference {ext_ref!r}")
        cve_id, gloss = child.get("CVE_ID"), value.text
        if not cve_id or not gloss or not gloss.strip():
            raise ValueError("entry lacks an identifier or gloss")
        if cve_id in ids or gloss in glosses:
            raise ValueError(f"duplicate ECV identifier or gloss: {cve_id} {gloss!r}")
        ids.add(cve_id)
        glosses.add(gloss)
        description = value.get("DESCRIPTION") or ""
        entries.append(ECVEntry(cve_id, gloss, _keywords(description), description))
    if not entries:
        raise ValueError("ECV holds no entries")
    return ECVLexicon(vocabulary_id, language, root.get("DATE", ""),
                      hashlib.sha256(payload).hexdigest(), tuple(entries), dict(sorted(external.items())))


@dataclass(frozen=True)
class GlossCandidates:
    label: str
    status: str
    stem_candidates: tuple[tuple[str, str], ...]     # (cve_id, gloss), sorted by gloss
    keyword_candidates: tuple[tuple[str, str], ...]
    requires_review: bool = True

    def to_dict(self) -> dict:
        return dict(label=self.label, status=self.status,
                    stem_candidates=[list(item) for item in self.stem_candidates],
                    keyword_candidates=[list(item) for item in self.keyword_candidates],
                    requires_review=self.requires_review)


def gloss_candidates(label: str, lexicon: ECVLexicon, *, _stems=None, _keywords_index=None,
                     ) -> GlossCandidates:
    """Candidate ID glosses for one lower-case English sign label."""
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise ValueError("label must be lower-case English words")
    stems = lexicon.by_stem() if _stems is None else _stems
    keywords = lexicon.by_keyword() if _keywords_index is None else _keywords_index
    stem_key = re.sub(r"[ ']", "-", label).upper()
    ordered = lambda found: tuple(sorted(((e.cve_id, e.gloss) for e in found), key=lambda x: x[1]))
    by_stem = ordered(stems.get(stem_key, ()))
    by_keyword = ordered(keywords.get(label.casefold(), ()))
    if len(by_stem) == 1:
        status = "unique_stem"
    elif by_stem:
        status = "ambiguous_stem"
    elif by_keyword:
        status = "keyword_only"
    else:
        status = "unmatched"
    return GlossCandidates(label, status, by_stem, by_keyword)


def map_labels(labels, lexicon: ECVLexicon) -> dict[str, GlossCandidates]:
    """Candidates for many labels with the lexicon indexes built once."""
    stems, keywords = lexicon.by_stem(), lexicon.by_keyword()
    return {label: gloss_candidates(label, lexicon, _stems=stems, _keywords_index=keywords)
            for label in labels}


__all__ = ["ECVEntry", "ECVLexicon", "GlossCandidates", "MATCH_STATUSES", "parse_ecv",
           "gloss_candidates", "map_labels"]
