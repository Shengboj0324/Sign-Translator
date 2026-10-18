"""SAPA phonology: strict per-gloss constancy and exact-gloss join."""
import json

import pytest

from signtranslator.data_engineering.sapa_phonology import PHONOLOGY_FIELDS, join_phonology, parse_sapa_metadata


def entry(gloss, handshape="open_b", n=2, vary=False):
    instances = []
    for i in range(n):
        inst = {f: "x" for f in PHONOLOGY_FIELDS}
        inst["Handshape"] = handshape if not (vary and i) else "other"
        inst["video_id"] = str(i)
        instances.append(inst)
    return dict(gloss=gloss, instances=instances)


def test_parses_constant_features_and_ignores_per_video_fields():
    parsed = parse_sapa_metadata(json.dumps([entry("book"), entry("drink", "c")]).encode())
    assert parsed["drink"]["Handshape"] == "c" and "video_id" not in parsed["book"]


@pytest.mark.parametrize("payload,message", [
    (json.dumps([entry("book", vary=True)]).encode(), "differs between instances"),
    (json.dumps([entry("book"), entry("book")]).encode(), "duplicate"),
    (b"[]", "nonempty"),
])
def test_inconsistent_metadata_is_refused(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_sapa_metadata(payload)


def test_join_is_exact_and_marks_review():
    expert = parse_sapa_metadata(json.dumps([entry("book")]).encode())
    kmeans = parse_sapa_metadata(json.dumps([entry("book", "44")]).encode())
    joined = join_phonology(["book", "zebra"], expert=expert, kmeans=kmeans)
    assert joined["book"]["status"] == "gloss_match" and joined["book"]["handshape_kmeans"] == "44"
    assert joined["book"]["handshape_euclidean"] is None and joined["zebra"]["status"] == "no_gloss_entry"
    assert all(v["requires_review"] for v in joined.values())
