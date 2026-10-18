"""Gloss review packet for qualified reviewers, and strict ingestion of their answers.

Label matches against the ASL Signbank ECV are candidates (``signbank_ecv``). Turning
one into a label needs a qualified person to watch the performed sign and the
Signbank entry. This module prepares that work and reads it back:

* :func:`build_review_rows` lists every sign with its candidates and direct Signbank
  links, hardest cases first, with empty decision fields;
* :func:`parse_review_sheet` accepts only well-formed answers: a decision from a
  closed set, a chosen identifier that is one of the offered candidates, and a
  reviewer key that is a SHA-256 digest, never a name;
* :func:`review_agreement` reports raw agreement and Cohen's kappa between two
  reviewers on the signs both answered, so a label can require agreement.

Nothing here decides a gloss. A sign becomes labelled only through reviewed answers,
and the packet inherits Signbank's CC BY-NC-SA 4.0 terms.
"""

from __future__ import annotations

import collections
import csv
import io
import re
from typing import Iterable, Mapping

PRIORITY = ("ambiguous_stem", "keyword_only", "unique_stem", "unmatched")
DECISIONS = ("confirm", "choose", "none_match", "unsure")
MAX_CANDIDATES = 8
_SHA = re.compile(r"[0-9a-f]{64}")
FIELDS = ("sign", "priority", "status", "skeleton_sample", "face_sample",
          *(f"candidate_{i}" for i in range(1, MAX_CANDIDATES + 1)),
          "decision", "chosen_cve_id", "reviewer_sha256", "notes")


def _candidates(gloss: Mapping) -> list[tuple[str, str]]:
    seen, out = set(), []
    for cve_id, label in (*gloss.get("stem_candidates", ()), *gloss.get("keyword_candidates", ())):
        if cve_id not in seen:
            seen.add(cve_id)
            out.append((cve_id, label))
    return out


def build_review_rows(records: Iterable[Mapping], *, gloss_url_base: str) -> list[dict]:
    """One row per sign, hardest first; candidate cells are ``ID|GLOSS|URL``."""
    if not gloss_url_base.startswith("https://"):
        raise ValueError("gloss_url_base must be an HTTPS URL")
    rows = []
    for record in records:
        gloss = record.get("gloss") or {}
        status = gloss.get("status", "unmatched")
        if status not in PRIORITY:
            raise ValueError(f"unknown gloss status {status!r}")
        candidates = _candidates(gloss)
        row = dict(sign=record["sign"], priority=PRIORITY.index(status), status=status,
                   skeleton_sample=bool((record.get("usable") or {}).get("skeleton")),
                   face_sample=bool((record.get("usable") or {}).get("face")),
                   decision="", chosen_cve_id="", reviewer_sha256="", notes="")
        for i in range(1, MAX_CANDIDATES + 1):
            if i <= len(candidates):
                cve_id, label = candidates[i - 1]
                row[f"candidate_{i}"] = f"{cve_id}|{label}|{gloss_url_base.rstrip('/')}/{cve_id}/"
            else:
                row[f"candidate_{i}"] = ""
        row["notes"] = "" if len(candidates) <= MAX_CANDIDATES else f"{len(candidates) - MAX_CANDIDATES} more in evidence"
        rows.append(row)
    rows.sort(key=lambda r: (r["priority"], not r["skeleton_sample"], r["sign"]))
    return rows


def write_review_sheet(rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parse_review_sheet(text: str, offered: list[dict]) -> list[dict]:
    """Validate returned answers against the rows that were offered."""
    by_sign = {r["sign"]: r for r in offered}
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("review sheet columns differ from the issued packet")
    answers = []
    for number, row in enumerate(reader, start=2):
        if not row["decision"]:
            continue  # unanswered rows stay unreviewed
        sign = row["sign"]
        if sign not in by_sign:
            raise ValueError(f"line {number}: {sign!r} was not offered")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"line {number}: decision must be one of {DECISIONS}")
        if not _SHA.fullmatch(row["reviewer_sha256"]):
            raise ValueError(f"line {number}: reviewer must be a SHA-256 key, never a name")
        allowed = {cell.split("|", 1)[0] for key, cell in by_sign[sign].items()
                   if key.startswith("candidate_") and cell}
        chosen = row["chosen_cve_id"]
        if row["decision"] in ("confirm", "choose"):
            if chosen not in allowed:
                raise ValueError(f"line {number}: chosen identifier is not an offered candidate")
        elif chosen:
            raise ValueError(f"line {number}: a {row['decision']} answer chooses nothing")
        answers.append(dict(sign=sign, decision=row["decision"], chosen_cve_id=chosen or None,
                            reviewer_sha256=row["reviewer_sha256"], notes=row["notes"]))
    keys = [(a["sign"], a["reviewer_sha256"]) for a in answers]
    if len(keys) != len(set(keys)):
        raise ValueError("a reviewer answered the same sign twice")
    return answers


def review_agreement(answers: list[dict], first: str, second: str) -> dict:
    """Raw agreement and Cohen's kappa over the outcome each reviewer gave.

    The outcome is the chosen identifier, or the decision itself when nothing was
    chosen. Kappa is undefined (None) when chance agreement is 1.
    """
    outcome = {}
    for answer in answers:
        if answer["reviewer_sha256"] in (first, second):
            outcome[(answer["sign"], answer["reviewer_sha256"])] = answer["chosen_cve_id"] or answer["decision"]
    shared = sorted({s for s, r in outcome if r == first} & {s for s, r in outcome if r == second})
    if not shared:
        return dict(signs=0, agreement=None, kappa=None)
    a = [outcome[(s, first)] for s in shared]
    b = [outcome[(s, second)] for s in shared]
    observed = sum(x == y for x, y in zip(a, b)) / len(shared)
    pa, pb = collections.Counter(a), collections.Counter(b)
    expected = sum(pa[k] * pb[k] for k in pa) / len(shared) ** 2
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    agreed = [s for s, x, y in zip(shared, a, b) if x == y and x not in DECISIONS]
    return dict(signs=len(shared), agreement=observed, kappa=kappa, agreed_labels=len(agreed))


__all__ = ["PRIORITY", "DECISIONS", "FIELDS", "build_review_rows", "write_review_sheet",
           "parse_review_sheet", "review_agreement"]
