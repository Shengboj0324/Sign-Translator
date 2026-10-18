"""ASL Signbank ECV parsing and English-label gloss candidates."""
import pytest

from signtranslator.data_engineering.signbank_ecv import gloss_candidates, map_labels, parse_ecv


def ecv(entries, *, extra=b""):
    body = b"".join(
        b'<CV_ENTRY_ML CVE_ID="%s" EXT_REF="signbank-ecv"><CVE_VALUE DESCRIPTION="%s" '
        b'LANG_REF="eng">%s</CVE_VALUE></CV_ENTRY_ML>' % (i.encode(), d.encode(), g.encode())
        for i, g, d in entries)
    return (b'<?xml version="1.0" ?>\n<CV_RESOURCE DATE="2026-10-18" VERSION="0.2">' + extra +
            b'<LANGUAGE LANG_ID="eng" LANG_LABEL="English (eng)"/>'
            b'<CONTROLLED_VOCABULARY CV_ID="ASL Signbank lexicon"><DESCRIPTION LANG_REF="eng">x'
            b'</DESCRIPTION>' + body + b'</CONTROLLED_VOCABULARY>'
            b'<EXTERNAL_REF EXT_REF_ID="signbank-ecv" TYPE="resource_url" VALUE="https://x.test/g/"/>'
            b'</CV_RESOURCE>')


LEXICON = parse_ecv(ecv([
    ("1", "ABOUTb", "about, regarding"), ("2", "ABOUTix", "about"), ("3", "ABHOR", "abhor, hate"),
    ("4", "HATE", "hate, detest"), ("5", "A-FEW", "few, a few"), ("6", "MUSCLE ", "muscle"),
    ("7", "i(ha)", "ha"), ("8", "DEAF2", "deaf"), ("9", "CHAMPb5", "champion"),
    ("10", "TWENTY-1", "twenty-one"),
]))


def test_parses_entries_exactly_as_published():
    assert len(LEXICON.entries) == 10 and LEXICON.language == "eng"
    assert LEXICON.external_refs == {"signbank-ecv": "https://x.test/g/"}
    entry = {e.cve_id: e for e in LEXICON.entries}
    assert entry["1"].keywords == ("about", "regarding") and entry["1"].stem == "ABOUT"
    assert entry["6"].gloss == "MUSCLE " and entry["6"].stem is None  # trailing space kept, not a stem
    assert entry["7"].stem is None and entry["8"].stem == "DEAF"  # handshape-variant digit
    assert entry["9"].stem == "CHAMP" and entry["10"].stem == "TWENTY-1"


@pytest.mark.parametrize("label,status,stems,keywords", [
    ("about", "ambiguous_stem", ["ABOUTb", "ABOUTix"], ["ABOUTb", "ABOUTix"]),
    ("abhor", "unique_stem", ["ABHOR"], ["ABHOR"]),
    ("hate", "unique_stem", ["HATE"], ["ABHOR", "HATE"]),
    ("detest", "keyword_only", [], ["HATE"]),
    ("deaf", "unique_stem", ["DEAF2"], ["DEAF2"]),
    ("muscle", "keyword_only", [], ["MUSCLE "]),
    ("zebra", "unmatched", [], []),
])
def test_candidate_rules_are_separate_and_never_select(label, status, stems, keywords):
    found = gloss_candidates(label, LEXICON)
    assert found.status == status and found.requires_review
    assert [g for _, g in found.stem_candidates] == stems
    assert [g for _, g in found.keyword_candidates] == keywords


def test_multiword_label_matches_hyphenated_stem_and_bulk_equals_single():
    assert gloss_candidates("a few", LEXICON).stem_candidates == (("5", "A-FEW"),)
    labels = ["about", "hate", "zebra", "a few"]
    assert map_labels(labels, LEXICON) == {label: gloss_candidates(label, LEXICON) for label in labels}


@pytest.mark.parametrize("payload,message", [
    (ecv([("1", "A", "a")], extra=b"<!--x-->"), "forbidden"),
    (b'<!DOCTYPE x [<!ENTITY e "boom">]>' + ecv([("1", "A", "a")]), "forbidden"),
    (ecv([("1", "A", "a"), ("1", "B", "b")]), "duplicate"),
    (ecv([("1", "A", "a"), ("2", "A", "b")]), "duplicate"),
    (ecv([]), "no entries"),
    (ecv([("1", "A", "a")]).replace(b'EXT_REF="signbank-ecv"', b'EXT_REF="other"'), "undeclared"),
    (ecv([("1", "A", "a")]).replace(b'LANG_REF="eng">A', b'LANG_REF="nld">A'), "declared language"),
])
def test_structural_surprises_are_refused(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_ecv(payload)


def test_label_form_is_enforced():
    with pytest.raises(ValueError):
        gloss_candidates("About", LEXICON)
