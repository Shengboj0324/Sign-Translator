"""Gloss review packet: ordering, links, strict answers, reviewer agreement."""
import csv
import io

import pytest

from signtranslator.data_engineering.gloss_review import (
    agreed_labels, audit_review_sheet, build_review_rows, parse_review_sheet, review_agreement,
    write_review_sheet,
)

A, B = "a" * 64, "b" * 64
URL = "https://aslsignbank.com//dictionary/gloss/"


def rec(sign, status, stems=(), keywords=(), skeleton=True):
    return dict(sign=sign, usable=dict(skeleton=["body"] if skeleton else [], face=[]),
                gloss=dict(status=status, stem_candidates=[list(x) for x in stems],
                           keyword_candidates=[list(x) for x in keywords]))


RECORDS = [rec("hate", "unique_stem", [("4", "HATE")], [("3", "ABHOR"), ("4", "HATE")]),
           rec("about", "ambiguous_stem", [("1", "ABOUTb"), ("2", "ABOUTix")]),
           rec("detest", "keyword_only", keywords=[("4", "HATE")], skeleton=False),
           rec("zebra", "unmatched")]


def answered(rows, edits):
    text = write_review_sheet(rows)
    out = list(csv.DictReader(io.StringIO(text)))
    for row in out:
        row.update(edits.get(row["sign"], {}))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(out)
    return stream.getvalue()


def test_rows_are_hardest_first_with_deduplicated_linked_candidates():
    rows = build_review_rows(RECORDS, gloss_url_base=URL)
    assert [r["sign"] for r in rows] == ["about", "detest", "hate", "zebra"]
    hate = next(r for r in rows if r["sign"] == "hate")
    assert hate["candidate_1"] == "4|HATE|https://aslsignbank.com//dictionary/gloss/4/"
    assert hate["candidate_2"].startswith("3|ABHOR|") and hate["candidate_3"] == ""
    assert all(r["decision"] == "" for r in rows)


def test_valid_answers_are_read_and_unanswered_rows_skipped():
    rows = build_review_rows(RECORDS, gloss_url_base=URL)
    text = answered(rows, {"about": dict(decision="choose", chosen_cve_id="2", reviewer_sha256=A),
                           "zebra": dict(decision="none_match", reviewer_sha256=A)})
    answers = parse_review_sheet(text, rows)
    assert [(a["sign"], a["chosen_cve_id"]) for a in answers] == [("about", "2"), ("zebra", None)]


@pytest.mark.parametrize("edit,message", [
    (dict(decision="choose", chosen_cve_id="99", reviewer_sha256=A), "not an offered candidate"),
    (dict(decision="maybe", reviewer_sha256=A), "decision must be"),
    (dict(decision="confirm", chosen_cve_id="1", reviewer_sha256="Jane Doe"), "never a name"),
    (dict(decision="unsure", chosen_cve_id="1", reviewer_sha256=A), "chooses nothing"),
])
def test_malformed_answers_are_refused(edit, message):
    rows = build_review_rows(RECORDS, gloss_url_base=URL)
    with pytest.raises(ValueError, match=message):
        parse_review_sheet(answered(rows, {"about": edit}), rows)


def test_agreement_and_kappa_are_exact():
    answers = [dict(sign=s, decision="choose", chosen_cve_id=c, reviewer_sha256=r, notes="")
               for s, c, r in (("x", "1", A), ("x", "1", B), ("y", "2", A), ("y", "3", B),
                               ("z", "2", A), ("z", "2", B), ("w", "1", A), ("w", "1", B))]
    result = review_agreement(answers, A, B)
    # observed 3/4; A marginals {1:2, 2:2}, B {1:2, 3:1, 2:1}: expected (2*2 + 2*1)/16 = 3/8
    assert result["agreement"] == pytest.approx(.75) and result["kappa"] == pytest.approx((.75 - .375) / .625)
    assert result["agreed_labels"] == 3
    assert review_agreement(answers, A, "c" * 64)["signs"] == 0


def test_audit_reports_every_problem_and_accepts_decision_case():
    rows = build_review_rows(RECORDS, gloss_url_base=URL)
    text = answered(rows, {"about": dict(decision="Confirm"),  # as received: no choice, no key
                           "hate": dict(decision="CONFIRM", chosen_cve_id="4", reviewer_sha256=A),
                           "zebra": dict(decision="maybe", reviewer_sha256=A)})
    answers, problems = audit_review_sheet(text, rows)
    assert [(a["sign"], a["decision"], a["chosen_cve_id"]) for a in answers] == [("hate", "confirm", "4")]
    issues = {p["sign"]: p["issues"] for p in problems}
    assert any("needs chosen_cve_id" in i for i in issues["about"])
    assert any("never a name" in i for i in issues["about"])
    assert any("decision must be" in i for i in issues["zebra"])


def test_agreed_labels_need_two_unanimous_reviewers():
    answers = [dict(sign="x", decision="choose", chosen_cve_id="1", reviewer_sha256=A, notes=""),
               dict(sign="x", decision="choose", chosen_cve_id="1", reviewer_sha256=B, notes=""),
               dict(sign="y", decision="choose", chosen_cve_id="2", reviewer_sha256=A, notes=""),
               dict(sign="y", decision="none_match", chosen_cve_id=None, reviewer_sha256=B, notes=""),
               dict(sign="z", decision="confirm", chosen_cve_id="3", reviewer_sha256=A, notes="")]
    assert agreed_labels(answers) == {"x": "1"}
    with pytest.raises(ValueError):
        agreed_labels(answers, minimum_reviewers=1)
