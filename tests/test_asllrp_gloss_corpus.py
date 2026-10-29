import collections
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import asllrp_gloss_corpus as agc  # noqa: E402


def test_category_reads_the_gloss_before_the_sign_type():
    assert agc.category("ns-fs-JOHN") == "name_sign"
    assert agc.category("fs-OF") == "fingerspelled"
    assert agc.category("#IF") == "lexicalised_fs"
    assert agc.category('DCL:5\\gathering information\\""') == "classifier"
    assert agc.category('ICL\\sword fight\\""') == "classifier"
    assert agc.category("CLEVER") == "lexical" and agc.category("CLASS") == "lexical"
    assert agc.category('5\\wave no\\""') == "gesture"
    assert agc.category("IX-1p") == "index" and agc.category("POSS") == "index" and agc.category("part:indef") == "index"
    assert agc.category("SECOND", "Number Signs") == "number"
    assert agc.category("TEACH+AGENT", "Compound Signs") == "compound"
    assert agc.category("ALLOW", "Lexical Signs") == "lexical"
    assert agc.category("SOMETHING", "Classifiers") == "classifier"


def test_index_key_takes_person_and_number_from_the_occurrence():
    assert agc.index_key("IX", "IX-3p:i") == ("IX", "3p", False)
    assert agc.index_key("IX", "(2h)IX-2p") == ("IX", "2p", False)
    assert agc.index_key("IX-pl", "IX-3p-pl-arc") == ("IX", "3p", True)
    assert agc.index_key("IX", "THUMB-IX-3p") == ("IX", "3p", False)
    assert agc.index_key("POSS-1p", "(G/Q)POSS-1p+") == ("POSS", "1p", False)
    assert agc.index_key("IX-honorific", "IX-honorific") == ("IX", None, False)
    assert agc.index_key("IX-1p", "") == ("IX", "1p", False)          # occurrence silent: the main entry
    assert agc.index_key("HAVE", "HAVE") is None
    assert agc.index_key("part:indef", "(1h)part:indef") is None


def test_name_candidates_normalise_decoration_alternatives_and_compounds():
    assert agc.name_candidates("BELOW_2") == [("below",)]
    assert agc.name_candidates("PAST__+NIGHT") == [("past", "night")]
    assert agc.name_candidates("GOOD/THANK-YOU") == [("good",), ("thank you",)]
    assert agc.name_candidates("(1h)HAVE") == [("have",)]
    assert agc.name_candidates("DIFFERENT+") == [("different",)]
    assert agc.name_candidates("GIVE-indef") == [("give",)]
    assert agc.name_candidates("MOTHERwg") == [("mother",)]
    assert agc.name_candidates("IX-1p") == []                  # no pointing stand-in: indices have their own tier
    assert agc.name_candidates("fs-OF") == [] and agc.name_candidates("#IF") == []
    assert agc.name_candidates("ns-fs-JOHN") == [] and agc.name_candidates('5\\wave\\""') == []


def test_spelled_word_and_english_tokens():
    assert agc.spelled_word("ns-fs-JOHN") == "john"
    assert agc.spelled_word("fs-E-MAIL") == "e mail"
    assert agc.spelled_word("#IF") == "if"
    assert agc.spelled_word("HAVE") is None
    assert agc.english_tokens("I don't know; it's Mom's car. Won't, can't, I'm, we'll") == [
        "i", "do", "not", "know", "it", "'s", "mom", "'s", "car", "will", "not", "can", "not", "i", "am", "we", "will"]


def sign(gloss, occurrence=None, sign_type="Lexical Signs", start=None, end=None):
    return dict(gloss=gloss, variant=gloss, occurrence=occurrence or gloss, sign_type=sign_type,
                category=agc.category(gloss, sign_type), start=start, end=end)


def test_alignment_order_and_kendall_tau():
    tokens = agc.english_tokens("My mother gave the book to me yesterday")
    signs = [sign("YESTERDAY"), sign("POSS-1p"), sign("MOTHER"), sign("BOOK"), sign("GIVE"), sign("IX-1p")]
    positions = agc.align(tokens, [agc.english_forms(s) for s in signs])
    assert positions == [7, 0, 1, 4, None, 6]                    # "gave" is irregular: GIVE stays unaligned
    tau, c, d = agc.kendall_tau(positions)
    assert (c, d) == (6, 4) and tau == pytest.approx(.2, abs=1e-4)
    assert agc.kendall_tau([0, 1, 2])[0] == 1 and agc.kendall_tau([2, 1, 0])[0] == -1
    assert agc.kendall_tau([5, None]) is None


def test_alignment_is_one_to_one_by_proportional_position_with_spans_and_inflections():
    forms = [agc.english_forms(sign("GO")), agc.english_forms(sign("GO"))]
    assert agc.align(["go", "home", "go"], forms) == [0, 2]
    assert agc.align(["go", "home", "go"], forms[:1]) == [0]       # a tie takes the earlier token
    assert agc.align_spans(["thank", "you", "very", "much"], [agc.english_forms(sign("GOOD/THANK-YOU"))]) == [(0, 2)]
    assert agc.align(["two", "voices"], [agc.english_forms(sign("VOICE"))]) == [1]
    assert agc.align(["i", "will", "go"], [agc.english_forms(sign("FUTURE"))]) == [1]
    assert agc.align(["i", "will", "go"], [agc.english_forms(sign("FUTURE"), synonyms=False)]) == [None]
    assert agc.align(["you", "know"], [agc.english_forms(sign("IX", "IX-2p")), agc.english_forms(sign("KNOW"))]) == [0, 1]
    assert agc.align(["he", "left"], [agc.english_forms(sign("IX", "IX-2p"))]) == [None]
    assert agc.align(["a", "person", "walking"], [agc.english_forms(sign('SCL:1\\person walking\\""'))]) == [None]


VOCAB = {w: 1.0 for w in ("allow", "win", "below", "past", "night", "me", "you", "if", "of", "john", "i", "ix",
                          "have", "mother")}


def test_map_sign_tiers_in_order():
    signbank = {"ALLOW": collections.Counter(allow=3), "IX": collections.Counter(ix=6)}
    codes, semlex = {"WIN": {"B_02_029"}}, {"B_02_029": collections.Counter(win=2)}

    def tier(gloss, occurrence=None, sign_type="Lexical Signs"):
        m = agc.map_sign(sign(gloss, occurrence, sign_type), VOCAB, signbank, codes, semlex)
        return m["tier"], m["entries"]
    assert tier("ALLOW") == ("signbank_id", ("allow",))
    assert tier("IX", "IX-2p") == ("signbank_id", ("ix",))       # the shared ID wins over the person rule
    assert tier("WIN") == ("asl_lex_code", ("win",))
    assert tier("BELOW_2") == ("name", ("below",))
    assert tier("PAST__+NIGHT") == ("name_compound", ("past", "night"))
    assert tier("IX-1p") == ("index_rule", ("me",))
    assert tier("#IF") == ("lexicalised_fs", ("if",))
    assert tier("fs-OF") == ("fingerspelled_word", ("of",))
    assert tier("fs-I") == (None, ())                              # single letters are never entries (amendment 1)
    assert tier("ns-fs-JOHN") == (None, ())                        # name signs have no tier
    assert tier('DCL:5\\gathering\\""') == (None, ())
    assert tier("DEPART") == (None, ())


def iso_record(corpus, text, label=None, code=None, seconds=1.5, frames=None, partition="train"):
    gloss = dict(label=label or text.upper())
    if code:
        gloss["asl_lex_code"] = code
    r = dict(corpus=corpus, partition=partition, text=text, gloss=gloss, sign=f"{corpus}-{text}-{seconds}",
             continuous=dict(status="converted", state_sha256="0" * 64, available_channels=["body", "right_hand"],
                             seconds=seconds, state_key="k"))
    if frames:
        r["source"] = dict(frames=frames)
    return r


def test_vocabulary_indices_use_train_records_ids_and_exact_citizen_codes_only():
    manifest = dict(records=[
        iso_record("asllvd-text-v1", "allow", "ALLOW", seconds=1.2, frames=[100, 135]),
        iso_record("rit-text-r6", "allow", "ALLOW", seconds=2.0),
        iso_record("asllvd-text-v1", "below", "BELOW", partition="val"),
        iso_record("semlex-text-v1", "win", code="B_02_029"),
        iso_record("semlex-text-v1", "mother", code="C_01_001"),
    ])
    vocab = agc.training_vocabulary(manifest, agc.ISOLATED_CORPORA)
    assert set(vocab) == {"allow", "win", "mother"}
    citizen = dict(mapping={"WIN": dict(method="exact", candidates=["WIN"], asl_lex_code="B_02_029"),
                            "MOTHER1": dict(method="letters_no_number", candidates=["MOTHER"], asl_lex_code="C_01_001")})
    signbank, seconds, codes, semlex = agc.vocabulary_indices(manifest, vocab, citizen)
    assert signbank == {"ALLOW": collections.Counter(allow=2)}
    assert seconds["ALLOW"]["asllvd-text-v1"] == [1.2] and seconds["ALLOW"]["rit-text-r6"] == [2.0]
    assert seconds["ALLOW"][agc.ANNOTATED] == pytest.approx([36 / 60], abs=1e-4)
    assert codes == {"WIN": {"B_02_029"}}                          # the non-exact MOTHER1 candidate is not an ID
    assert set(semlex) == {"B_02_029", "C_01_001"}


def test_semlex_text_v2_records_map_by_asl_lex_code_as_v1_records_do():
    """Round 14b's semlex-text-v2 is read exactly as semlex-text-v1 (its ASL-LEX codes index its entries)."""
    v1 = [iso_record("semlex-text-v1", "win", code="B_02_029"), iso_record("asllvd-text-v1", "allow", "ALLOW")]
    v2 = [dict(r, corpus="semlex-text-v2") if r["corpus"] == "semlex-text-v1" else r for r in v1]
    out = []
    for records in (v1, v2):
        manifest = dict(records=records)
        vocab = agc.training_vocabulary(manifest, agc.isolated_corpora(manifest))
        assert set(vocab) == {"allow", "win"}
        out.append(agc.vocabulary_indices(manifest, vocab, dict(mapping={})))
    assert out[0] == out[1] and out[1][3] == {"B_02_029": collections.Counter(win=1)}
    assert agc.SEMLEX_CORPORA == ("semlex-text-v1", "semlex-text-v2")


def utterance(source, number, glosses_frames, frames, collection="asllrp-continuous"):
    return dict(collection=collection, source=source, utterance=number, utterance_video=f"{number}.mp4", frames=frames,
                signs=[dict(gloss=g, variant=g, occurrence=o, sign_type="Lexical Signs", frames=f)
                       for g, o, f in glosses_frames])


def pair(source, number, english, glosses, partition):
    return dict(id=f"c:{source}-U-{number}", source=source, utterance=number, english=english, glosses=glosses,
                partition=partition, surface=list(glosses), markers=[])


def test_build_corpus_times_signs_and_never_reads_test():
    utts = [utterance("Cory_2013-6-25_sc0106", 1, [("IX-1p", "IX-1p", [30, 35]), ("LIKE", "LIKE", [40, 49]),
                                                    ("", "", [50, 51]), ("CAR", "CAR", [None, None])], [19, 152]),
            utterance("4-Ben-Deaf", 2, [("HELLO", "HELLO", [5, 9])], [1, 20])]
    pairs = [pair("Cory_2013-6-25_sc0106", 1, "I like cars.", ["IX-1p", "LIKE", "CAR"], "train"),
             pair("4-Ben-Deaf", 2, "Hello.", ["HELLO"], "test"),
             pair("Nobody", 9, "x", ["X"], "val")]
    records, counts = agc.build_corpus(utts, pairs)
    assert [r["signer"] for r in records] == ["Cory"]
    assert counts == dict(no_utterance_record=1, pairs_test=1, pairs_train=1, pairs_val=1, signs_untimed=1,
                          signs_without_gloss=1)
    fps = 30000 / 1001
    s = records[0]["signs"]
    assert [x["gloss"] for x in s] == ["IX-1p", "LIKE", "CAR"]
    assert s[0]["start"] == round(11 / fps, 4) and s[0]["end"] == round(17 / fps, 4)
    assert s[2]["start"] is None and records[0]["seconds"] == round(134 / fps, 4)
    assert s[0]["category"] == "index"


def test_timing_gaps_slots_and_compression():
    rec = dict(seconds=3.0, signs=[sign("ALLOW", start=.5, end=.8), sign("BELOW", start=1.0, end=1.4),
                                   sign("DCL:x", sign_type="Classifiers", start=1.4, end=1.9)])
    table = {"ALLOW": dict(tier="signbank_id", entries=("allow",)), "BELOW": dict(tier="name", entries=("below",)),
             "DCL:x": dict(tier=None, entries=())}
    vocab = {"allow": 1.5, "below": 2.0}
    seconds = {"ALLOW": {"asllvd-text-v1": [1.5], agc.ANNOTATED: [.6]}}
    t = agc.timing_statistics([rec], table, vocab, seconds, {"crossfade": dict(fitted_word_seconds_median=.74)})
    assert t["sign_seconds"]["median"] == pytest.approx(.4, abs=1e-4)
    assert t["gap_seconds"]["n"] == 2 and t["gap_seconds"]["median"] == pytest.approx(.1, abs=1e-4)
    assert t["gaps"]["overlapping_share"] == 0
    assert t["leading_seconds"]["median"] == pytest.approx(.5, abs=1e-4)
    assert t["trailing_seconds"]["median"] == pytest.approx(1.1, abs=1e-4)
    assert t["compression"]["vs_vocabulary_entry_typical"]["median"] == pytest.approx((.3 / 1.5 + .4 / 2) / 2, abs=1e-4)
    by = t["compression"]["vs_same_signbank_gloss_clips_by_corpus"]
    assert by[agc.ANNOTATED]["median"] == pytest.approx(.5, abs=1e-4)
    assert by["asllvd-text-v1"]["median"] == pytest.approx(.2, abs=1e-4)
    assert t["compression"]["slot_vs_asllvd_annotated_sign"]["median"] == pytest.approx(.5 / .6, abs=1e-4)
    assert t["compression"]["asllvd_trimmed_clip_over_annotated_sign_per_gloss"]["median"] == pytest.approx(2.5, abs=1e-4)
    assert t["compression"]["sentence_span_vs_sum_of_isolated_typical"] is None   # the classifier is unmapped
    assert t["round15_comparison"]["fitted_over_asllrp_sign"] == {"crossfade": round(.74 / .4, 3)}


def test_rule_check_levenshtein_and_overlap():
    assert agc.levenshtein(["a", "b", "c"], ["a", "c"]) == 1 and agc.levenshtein([], ["x"]) == 1
    rec = dict(english="The mother will allow it", signs=[sign("ALLOW"), sign("MOTHER"), sign("DCL:x")])
    table = {"ALLOW": dict(tier="signbank_id", entries=("allow",)), "MOTHER": dict(tier="name", entries=("mother",)),
             "DCL:x": dict(tier=None, entries=())}
    out = agc.rule_check([rec], table, {"mother": 1., "allow": 1., "it": 1.})
    assert (out["rule_entries"], out["gold_entries"]) == (3, 2)        # rule: mother allow it; gold: allow mother
    assert out["bag_precision"] == pytest.approx(2 / 3, abs=1e-4) and out["bag_recall"] == 1
    assert out["entry_wer"] == 1.0 and out["shared_entry_order_kendall_tau"]["median"] == -1
    ov = agc.vocabulary_overlap(["the cat sat"], ["the cat ran", "a dog"], aligned_types={"cat"})
    assert ov["all_words"]["type_overlap"] == pytest.approx(2 / 5, abs=1e-4)
    assert ov["all_words"]["token_overlap"] == pytest.approx(2 / 5, abs=1e-4)
    assert ov["content_words"]["type_overlap"] == pytest.approx(1 / 3, abs=1e-4)
    assert ov["content_words"]["token_overlap_aligned_types"] == pytest.approx(1 / 3, abs=1e-4)


def write_store(root: Path):
    utts = [utterance("Cory_2013-6-25_sc0106", 1, [("IX-1p", "IX-1p", [30, 35]), ("ALLOW", "ALLOW", [40, 49]),
                                                    ("fs-OF", "fs-OF", [55, 60])], [19, 152]),
            utterance("Jonathan_2013-7-1_sc0001", 2, [("MOTHER", "MOTHER", [10, 20]), ("WIN", "WIN", [25, 33])],
                      [1, 60]),
            utterance("4-Ben-Deaf", 3, [("HELLO", "HELLO", [5, 9])], [1, 20])]
    pairs = [pair("Cory_2013-6-25_sc0106", 1, "I allow it.", ["IX-1p", "ALLOW", "fs-OF"], "train"),
             pair("Jonathan_2013-7-1_sc0001", 2, "Mother won.", ["MOTHER", "WIN"], "val"),
             pair("4-Ben-Deaf", 3, "Hello.", ["HELLO"], "test")]
    manifest = dict(records=[
        iso_record("asllvd-text-v1", "allow", "ALLOW", frames=[1, 30]), iso_record("semlex-text-v1", "win", code="B1"),
        iso_record("wlasl-text-r6", "mother"), iso_record("wlasl-text-r6", "me"),
        dict(corpus="how2sign-two-view-text-v1", partition="val", signer_component="c01", text="Mother allows it.")])
    files = {"u.json": dict(items=utts), "p.json": dict(pairs=pairs),
             "c.json": dict(mapping={"WIN": dict(method="exact", candidates=["WIN"], asl_lex_code="B1")}),
             "m.json": manifest, "s.json": dict(timing={"crossfade": dict(fitted_word_seconds_median=.74)})}
    for name, body in files.items():
        (root / name).write_bytes(json.dumps(body).encode())
    return hashlib.sha256((root / "m.json").read_bytes()).hexdigest()


def test_main_end_to_end_write_once_and_hash_checked(tmp_path, capsys):
    root = tmp_path / "store"; root.mkdir()
    sha = write_store(root)
    argv = ["--store", f"dir:{root}", "--utterances-key", "u.json", "--pairs-key", "p.json", "--citizen-map-key",
            "c.json", "--mix-key", "m.json", "--mix-sha256", sha, "--stage1-report-key", "s.json",
            "--output-file", str(tmp_path / "out/report.json"), "--corpus-file", str(tmp_path / "out/corpus.json")]
    assert agc.main(argv) == 0
    capsys.readouterr()
    report = json.loads((tmp_path / "out/report.json").read_text())
    corpus = json.loads((tmp_path / "out/corpus.json").read_text())
    assert report["utterances"]["train"]["utterances"] == 1 and report["utterances"]["val"]["utterances"] == 1
    assert report["test_partition"].startswith("counted, never read: 1")
    assert "HELLO" not in corpus["gloss_map"] and len(corpus["records"]) == 2
    assert {g: m["tier"] for g, m in corpus["gloss_map"].items()} == {
        "IX-1p": "index_rule", "ALLOW": "signbank_id", "fs-OF": None, "MOTHER": "name", "WIN": "asl_lex_code"}
    cov = report["mapping"]["train+val"]["coverage"]
    assert cov["strict"]["by_token"] == pytest.approx(3 / 5, abs=1e-4)
    assert cov["with_index"]["by_token"] == pytest.approx(4 / 5, abs=1e-4)
    assert report["alignment"]["train"]["order"]["kendall_tau"]["median"] == 1
    assert report["generalisation"]["asllrp_train_vs_how2sign_val"]["target_sentences"] == 1
    assert report["inputs"]["m.json"] == sha
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        agc.main(argv)
    with pytest.raises(SystemExit, match="does not match"):
        agc.main(["0" * 64 if x == sha else str(tmp_path / "x.json") if x.endswith("report.json") else x
                  for x in argv])
