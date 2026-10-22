import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_asllrp_english_gloss import main, markers, tiers  # noqa: E402

ROUGH = ("\n\t    \t\n\t    \t\t   cond/when:  _______________when \n"
         "          dh:  SAME__ VOICE_ WHEN PERSON_ (2h)TALK \n"
         "    dh shape:   \n"
         " wh question:" + " " * 21 + "____q/wh\n"
         "  engl trans: Like when a person is speaking.\n")


def test_markers_cover_the_tokens_under_their_underline():
    t = tiers(ROUGH)
    assert t["engl trans"].strip() == "Like when a person is speaking."
    found = {m["tier"]: m for m in markers(t)}
    assert (found["cond/when"]["token_start"], found["cond/when"]["token_end"]) == (0, 3)
    assert found["cond/when"]["label"] == "when"
    assert (found["wh question"]["token_start"], found["wh question"]["token_end"]) == (3, 4)


def test_pairs_join_search_rows_to_utterances_by_source_and_number(tmp_path):
    root = tmp_path / "store"
    (root / "s").mkdir(parents=True); (root / "u").mkdir()
    (root / "s/search.json").write_text(json.dumps({"utterances": [
        {"name": "7-Ben-Social-Msg-U-1", "resource_id": "1", "rough_gloss_text": ROUGH},
        {"name": "Cory_2013-6-25_sc0106-U-2", "resource_id": "2", "rough_gloss_text": "  engl trans: Hi.\n          dh: HELLO\n"},
        {"name": "Nobody-U-1", "resource_id": "3", "rough_gloss_text": ROUGH}]}))
    sign = lambda g: {"gloss": g}  # noqa: E731
    (root / "u/utt.json").write_text(json.dumps({"items": [
        {"collection": "asllrp-continuous", "source": "7-Ben-Social-Msg", "utterance": 1, "utterance_video": "1.mp4",
         "signs": [sign("SAME"), sign("VOICE"), sign("WHEN"), sign("PERSON"), sign("TALK")]},
        {"collection": "asllrp-continuous", "source": "Cory_2013-6-25_sc0106", "utterance": 2, "utterance_video": "2.mp4",
         "signs": [sign("HELLO")]}]}))
    assert main(["--store", f"dir:{root}", "--search-key", "s/search.json", "--utterances-key", "u/utt.json",
                 "--output-key", "o/pairs.json"]) == 0
    out = json.loads((root / "o/pairs.json").read_text())
    assert out["partitions"] == {"train": 1, "val": 0, "test": 1} and out["skipped"] == {"no utterance record": 1}
    ben = next(p for p in out["pairs"] if p["source"] == "7-Ben-Social-Msg")
    assert ben["english"] == "Like when a person is speaking." and ben["glosses"][0] == "SAME"
