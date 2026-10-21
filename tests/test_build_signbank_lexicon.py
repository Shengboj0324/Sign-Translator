"""Sign Bank lexicon, tiered ASL Citizen gloss map and utterance gloss sequences from DAI 2 annotation CSVs."""
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_signbank_lexicon as builder  # noqa: E402

CITATION_COLUMNS = ["Video ID number", "main entry gloss label", "entry/variant gloss label", "occurrence label",
                    "Dominant start handshape", "Non-dominant start handshape", "Dominant end handshape",
                    "Non-dominant end handshape", "sign type", "Class Label"]
CONTINUOUS_COLUMNS = ["Video ID number", "Main entry gloss label", "Entry/variant gloss label", "Occurrence label",
                      "Start frame of the sign video", "End frame of the sign video",
                      "Start frame of the containing utterance", "End frame of the containing utterance",
                      "Dominant start handshape", "Non-dominant start handshape", "Dominant end handshape",
                      "Non-dominant end handshape", "Sign video filename", "Utterance video filename",
                      "Source collection", "Utterance number", "Master video filename", "Sign type", "Class label", "Hidden"]


def write_csv(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer); writer.writerow(columns); writer.writerows(rows)
    path.write_text(buffer.getvalue())


def corpus(tmp_path):
    p = tmp_path / builder.PREFIX
    write_csv(p / builder.CITATION["asllvd-citation"], CITATION_COLUMNS, [
        ["1", "THANK-YOU", "THANK-YOU", "THANK-YOU", "B", "", "B", "", "Lexical Signs", "THANK-YOU"],
        ["2", "(1)CHEAT", "(elbow)CHEAT", "(elbow)CHEAT", "F", "B", "F", "B", "Lexical Signs", "(1)CHEAT"],
        ["3", "CANDY", "CANDY", "CANDY", "1", "", "1", "", "Lexical Signs", "CANDY"]])
    write_csv(p / builder.CITATION["rit-citation"], CITATION_COLUMNS, [
        ["4", "LIBRARY", "LIBRARY", "LIBRARY", "L", "", "L", "", "Lexical Signs", "LIBRARY"]])
    write_csv(p / builder.CONTINUOUS["asllrp-continuous"], CONTINUOUS_COLUMNS, [
        ["5", "LIBRARY", "LIBRARY", "LIBRARY", "40", "52", "10", "90", "L", "", "L", "", "s2.mp4", "u1.mp4", "Cory_sc1", "1", "m.mp4", "Lexical Signs", "LIBRARY", "F"],
        ["6", "IX-1p", "IX-1p", "IX-1p", "12", "20", "10", "90", "1", "", "1", "", "s1.mp4", "u1.mp4", "Cory_sc1", "1", "m.mp4", "Lexical Signs", "IX-1p", "F"],
        ["7", "SECRET", "SECRET", "SECRET", "60", "70", "10", "90", "A", "", "A", "", "s3.mp4", "u1.mp4", "Cory_sc1", "1", "m.mp4", "Lexical Signs", "SECRET", "T"]])
    write_csv(p / builder.CONTINUOUS["rit-sentences"], CONTINUOUS_COLUMNS, [])
    manifest = dict(records=[dict(gloss=dict(label=g, asl_lex_code=c)) for g, c in
                             (("LIBRARY", "A_01"), ("THANKYOU", "B_02"), ("CANDY1", None), ("CHEAT", None), ("CELERY", None))])
    (tmp_path / "citizen.json").write_text(json.dumps(manifest))


def run(tmp_path):
    return builder.main(["--store", f"dir:{tmp_path}", "--citizen-manifest-key", "citizen.json", "--output-prefix", "out"])


def test_lexicon_map_and_utterances(tmp_path):
    corpus(tmp_path)
    assert run(tmp_path) == 0
    lexicon = json.loads((tmp_path / "out/lexicon.json").read_text())["lexicon"]
    assert lexicon["LIBRARY"]["tokens"] == {"rit-citation": 1, "asllrp-continuous": 1}
    assert lexicon["(1)CHEAT"]["handshapes"]["Non-dominant start handshape"] == {"B": 1}
    assert "SECRET" not in lexicon                          # hidden rows are left out
    gloss_map = json.loads((tmp_path / "out/asl-citizen-gloss-map.json").read_text())
    m = gloss_map["mapping"]
    assert m["LIBRARY"] == dict(method="exact", candidates=["LIBRARY"], asl_lex_code="A_01", requires_review=False)
    assert m["THANKYOU"]["method"] == "letters" and m["THANKYOU"]["candidates"] == ["THANK-YOU"]
    assert m["CANDY1"]["method"] == "letters_no_number" and m["CANDY1"]["requires_review"]
    assert m["CHEAT"]["method"] == "letters" and m["CHEAT"]["candidates"] == ["(1)CHEAT"]
    assert gloss_map["unmatched"] == ["CELERY"] and gloss_map["counts"]["unmatched"] == 1
    utterances = json.loads((tmp_path / "out/utterances.json").read_text())["items"]
    assert len(utterances) == 1 and [s["gloss"] for s in utterances[0]["signs"]] == ["IX-1p", "LIBRARY"]
    assert utterances[0]["frames"] == [10, 90] and utterances[0]["utterance_video"] == "u1.mp4"


def test_outputs_are_written_once(tmp_path):
    corpus(tmp_path)
    assert run(tmp_path) == 0
    try:
        run(tmp_path)
        raise AssertionError("a second run must not overwrite")
    except SystemExit as error:
        assert "refusing to overwrite" in str(error)
