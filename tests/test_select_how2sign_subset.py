import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_how2sign_subset import clip_regex, select, sessions  # noqa: E402


def test_whole_sessions_small_components_kept_take5_excluded_and_regex_exact():
    rows = []
    for video, take, comp, n in (("aa", "1", "c00", 2), ("bb", "5", "c02", 3), ("cc", "2", "c02", 2), ("dd", "3", "c02", 2),
                                 ("e-e", "1", "c03", 1)):
        for i in range(n):
            rows.append(dict(VIDEO_NAME=f"{video}-{take}-rgb_front", SENTENCE_NAME=f"{video}_{i}-{take}-rgb_front",
                             START_REALIGNED="0", END_REALIGNED="1800"))
    comps = {r["SENTENCE_NAME"]: c for r, c in zip(rows, ["c00"] * 2 + ["c02"] * 7 + ["c03"])}
    table = sessions(rows, comps)
    assert table["aa-1"]["hours"] == 1. and table["bb-5"]["component"] == "c02"
    chosen = select(table, c02_hours=.5)
    assert "aa-1" in chosen and "e-e-1" in chosen and "bb-5" not in chosen
    assert sum(n in chosen for n in ("cc-2", "dd-3")) == 1            # stops once 0.5 h of c02 is reached
    pattern = re.compile(clip_regex(chosen))
    assert pattern.match("aa_12-1-rgb_side") and pattern.match("e-e_0-1-rgb_front")
    assert not pattern.match("aa_12-2-rgb_side") and not pattern.match("bb_0-5-rgb_front")


def test_validation_component_cap():
    rows = [dict(VIDEO_NAME=f"v{k}-1-rgb_front", SENTENCE_NAME=f"v{k}_0-1-rgb_front", START_REALIGNED="0",
                 END_REALIGNED="3600") for k in range(4)]
    table = sessions(rows, {r["SENTENCE_NAME"]: "c03" for r in rows})
    assert len(select(table, 0., c03_hours=1.5)) == 2 and len(select(table, 0.)) == 4
