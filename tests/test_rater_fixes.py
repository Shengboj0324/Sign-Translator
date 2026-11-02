"""The signer study tool's fixes, one block per finding of the final check (docs/NEXT_PHASE_PLAN.md, "Rater read 1: tool fixes and the
official arm's export"), and the three checks earlier review left to mutants."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import build_human_eval_packet as bh  # noqa: E402
import rater_word_format as wf  # noqa: E402
import round20_lib as lib  # noqa: E402
import score_human_eval as sh  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval_framework import human_study as hs  # noqa: E402
from signtranslator.eval_framework.human_study import (BAR, NEGATION_COLUMNS, BarRefused, SheetError, bar_read, check_roster,  # noqa: E402
                                                       decode_sheet, design_study, load_roster, negation_words, parse_negation_sheet, parse_roster,
                                                       parse_sheet, parse_stage1, parse_stage2, read_stages, read_two_stage, study_coverage,
                                                       word_format_read, write_negation_sheet)
from tests.test_eval_human_study import bar_key, rater_of, roster_of, stage1_text, stage2_text  # noqa: E402
from tests.test_rater_word_format import PACKET, SIGNS, built, eight, pilot_setup, study  # noqa: E402
from tests.test_round15_compose_eval import ce  # noqa: E402
from tests.test_round15_compositional import stubbed  # noqa: E402,F401
from tests.test_round15_stage2 import stage2_store  # noqa: E402,F401
from tests.test_round20_eval import put_set  # noqa: E402

OWNER_NOTE = ("Owner, 2026-11-02 20:30 UTC, asked whether ASLLVD train clips may be shown to raters under the DAI terms, answered: Yes. This covers ASLLVD train "
              "clips only, shown to consenting reviewers, research use only and non-commercial, under the DAI terms. RIT and WLASL aren't covered.")


def read(name):
    return (ROOT / name).read_text()


def csv_of(columns, rows):
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def draw_args(root, sha, set_sha, out, *extra, main="3", pilot="1"):
    return ["how2sign-draw", "--store", root, "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key", "sets/d20.json", "--sentences-sha256", set_sha,
            "--main", main, "--pilot", pilot, "--min-seconds", ".5", "--output-dir", str(out), *extra]


def how2sign_csvs(store, rows):
    """How2Sign's published metadata, tab-separated as the stored files are. ``rows``: (sign, split)."""
    for split in {s for _, s in rows}:
        cut = [dict(VIDEO_ID="v", VIDEO_NAME=f"{s}-session", SENTENCE_ID="v_0", SENTENCE_NAME=s, START="10.5", END="12.25", SENTENCE="x") for s, sp in rows if sp == split]
        aligned = [dict(VIDEO_ID="v", VIDEO_NAME=f"{s}-session", SENTENCE_ID="v_0", SENTENCE_NAME=s, START_REALIGNED="11.0", END_REALIGNED="13.0", SENTENCE="x") for s, sp in rows if sp == split]
        for name, table in ((f"how2sign_{split}.csv", cut), (f"how2sign_realigned_{split}.csv", aligned)):
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=list(table[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(table)
            store.put(f"{bh.H2S_PUBLIC}/{name}", out.getvalue().encode())


# ---- D1: the checker's sheet ------------------------------------------------------------------------------------------------------

def checker_rows(candidates, **change):
    rows = list(csv.DictReader(io.StringIO(write_negation_sheet(candidates))))
    for r in rows:
        r.update({**dict(only_real_recordings_seen="yes", watched_before_reading="yes", confirmed="yes", checker_sha256="c" * 64), **change})
    return csv_of(NEGATION_COLUMNS, rows)


CANDIDATES = [dict(sign="a", english="I do not", negation_words=["not"]), dict(sign="b", english="never mind", negation_words=["never"])]


def test_the_attestation_column_says_what_it_means_and_the_code_accepts_what_it_says():
    assert "only_real_recordings_seen" in NEGATION_COLUMNS and "no_generated_clip_seen" not in NEGATION_COLUMNS and "watched_before_reading" in NEGATION_COLUMNS
    assert parse_negation_sheet(checker_rows(CANDIDATES), CANDIDATES) == (dict(a="yes", b="yes"), [])
    for bad in ("no", "", "maybe"):
        problems = parse_negation_sheet(checker_rows(CANDIDATES, only_real_recordings_seen=bad), CANDIDATES)[1]
        assert any("only_real_recordings_seen must be yes" in p["issue"] for p in problems)


def test_a_sheet_with_the_old_column_says_how_to_get_the_new_one():
    old = ",".join(c if c != "only_real_recordings_seen" else "no_generated_clip_seen" for c in NEGATION_COLUMNS) + "\n"
    truth, problems = parse_negation_sheet(old, CANDIDATES)
    assert truth == {} and "how2sign-checker-sheet" in problems[0]["issue"]


def test_the_checkers_list_comes_from_how2signs_csvs_and_says_what_it_could_not_find(tmp_path):
    store = DirectoryStore(tmp_path)
    how2sign_csvs(store, [("in-train", "train"), ("in-val", "val")])
    candidates = [dict(sign=s, english="x", negation_words=["not"]) for s in ("in-train", "in-val", "nowhere")]
    fetched = []
    real = store.get
    store.get = lambda key: (fetched.append(key), real(key))[1]
    rows, missing = bh.checker_clip_table(store, candidates, lambda sign: 7.75)
    assert missing == 1 and [r["found"] for r in rows] == ["yes", "yes", "no"] and [r["row"] for r in rows] == [1, 2, 3]
    assert rows[0]["clip_zip"].endswith("/train_rgb_front_clips.zip") and rows[1]["clip_zip"].endswith("/val_rgb_front_clips.zip")
    assert rows[0]["clip_file"] == "raw_videos/in-train.mp4" and rows[0]["session_video"] == "in-train-session" and rows[0]["seconds"] == 7.75
    assert (rows[0]["start_seconds"], rows[0]["end_seconds"], rows[0]["realigned_start_seconds"], rows[0]["realigned_end_seconds"]) == ("10.5", "12.25", "11.0", "13.0")
    assert rows[2]["clip_zip"] == "" and rows[2]["session_video"] == "" and rows[2]["clip_file"] == "raw_videos/nowhere.mp4"
    assert not any("test" in k for k in fetched)                       # the held-out CSV has no business here
    text = bh.write_checker_clips(rows)
    assert text.splitlines()[0].split(",") == list(bh.CHECKER_CLIP_COLUMNS) and text.splitlines()[1].split(",")[4] == "raw_videos/in-train.mp4"
    assert bh.CHECKER_CLIP_COLUMNS[4] == "clip_file"                   # the guide's cut -f5


def test_the_checker_guide_is_one_page_owner_led_and_watching_comes_first():
    guide = read("reviewer_package/CHECKER_GUIDE.md")
    assert len(guide.split()) < 500 and "leads" in guide
    assert guide.index("**Watch.**") < guide.index("**Read.**") < guide.index("**Final answer.**")
    assert "no words on the screen" in guide and "only_real_recordings_seen" in guide and "watched_before_reading" in guide
    lowered = guide.lower()
    assert "terminal" not in lowered and "you do not need a computer" in lowered
    steps = read("reviewer_package/OPERATOR_GUIDE.md")
    assert "## 11. The checker's session" in steps and "`found`" in steps and "how2sign_realigned_train.csv" in steps


def test_the_checker_has_a_consent_draft_that_fits_and_the_board_is_told():
    form = read("reviewer_package/CHECKER_CONSENT_FORM.md")
    body = form[form.index("**Project:**"):]
    assert "real signers" in body and "will not be shown any computer-made signing" in body and "watch short videos of computer-made signing" not in body
    assert "CHECKER_CONSENT_FORM.md" in read("reviewer_package/README.md")
    guide = read("reviewer_package/OPERATOR_GUIDE.md")
    board = guide[guide.index("**Questions for the review board.**"):guide.index("**What changed for raters:**")]
    assert "CHECKER_CONSENT_FORM.md" in board and "CHECKER_CONSENT_FORM.md" in read("reviewer_package/CONSENT_FORM.md")


def test_a_row_where_the_checker_read_first_reaches_the_key_and_the_report(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    monkeypatch.setattr(bh, "negation_words", lambda t: ["off"] if "off" in t else [])
    root = f"dir:{tmp_path / 'store'}"
    assert bh.main(draw_args(root, sha, set_sha, tmp_path / "d", "--seed", "11", "--name", "t")) == 0
    draw = json.loads((tmp_path / "d" / "draw.json").read_text())
    rows = list(csv.DictReader(io.StringIO((tmp_path / "d" / "negation_check.csv").read_text())))
    rows[0].update(only_real_recordings_seen="yes", watched_before_reading="no", confirmed="yes", checker_sha256="c" * 64)
    text = csv_of(NEGATION_COLUMNS, rows)
    assert parse_negation_sheet(text, draw["candidates"])[1] == [] and hs.negation_read_first(text) == [rows[0]["sign"]]
    sheet = tmp_path / "checked.csv"
    sheet.write_text(text)
    pin = hashlib.sha256((tmp_path / "d" / "draw.json").read_bytes()).hexdigest()
    assert bh.main(["how2sign-packet", "--store", root, "--draw", str(tmp_path / "d" / "draw.json"), "--draw-sha256", pin, "--negation-sheet", str(sheet),
                    "--negation-sheet-sha256", hashlib.sha256(sheet.read_bytes()).hexdigest(), "--set", "main", "--renderer", "skeleton", "--panel", "96",
                    "--system", "truth=reference", "--output-dir", str(tmp_path / "p")]) == 0
    key = json.loads((tmp_path / "p" / "key" / "key.json").read_text())
    assert key["negation_read_first"] == [rows[0]["sign"]] and key["excluded_no_selected_word"] == [] and key["distractors"]["without_negation_words"] is False


# ---- D2: the draw ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["--seed", "--name"])
def test_the_draw_needs_a_declared_seed_and_a_name(tmp_path, missing, capsys):
    args = ["how2sign-draw", "--sentences-key", "k", "--sentences-sha256", "s", "--output-dir", str(tmp_path / "o"), "--seed", "1", "--name", "n"]
    k = args.index(missing)
    with pytest.raises(SystemExit) as stop:
        bh.main(args[:k] + args[k + 2:])
    assert stop.value.code == 2 and missing in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_the_draw_prints_its_seed_records_its_commit_and_is_stored_once(tmp_path, stage2_store, monkeypatch, capsys):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    monkeypatch.setattr(bh, "negation_words", lambda t: ["off"] if "off" in t else [])
    root = f"dir:{tmp_path / 'store'}"
    assert bh.main(draw_args(root, sha, set_sha, tmp_path / "a", "--seed", "7", "--name", "read-1")) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["seed"] == 7 and printed["name"] == "read-1" and printed["store_prefix"] == "human-eval/draws/read-1"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    local = (tmp_path / "a" / "draw.json").read_bytes()
    draw = json.loads(local)
    assert draw["seed"] == 7 and draw["code"]["commit"] == head and isinstance(draw["code"]["dirty"], bool) and printed["code"] == draw["code"]
    for file in ("draw.json", "negation_check.csv", "checker_clips.csv"):
        assert store.get(f"human-eval/draws/read-1/{file}") == (tmp_path / "a" / file).read_bytes()
    assert printed["recordings_not_found"] == printed["candidates"]    # the fixture's store holds no How2Sign CSV: each row says so
    # written once: refused before anything is read or written, the first copy untouched
    with monkeypatch.context() as m:
        m.setattr(bh, "load_d20", lambda *a: pytest.fail("the set was read"))
        with pytest.raises(SystemExit, match="written once"):
            bh.main(draw_args(root, sha, set_sha, tmp_path / "b", "--seed", "8", "--name", "read-1"))
    assert not (tmp_path / "b").exists() and store.get("human-eval/draws/read-1/draw.json") == local
    assert bh.main(draw_args(root, sha, set_sha, tmp_path / "c", "--seed", "7", "--name", "read-2")) == 0
    other = json.loads((tmp_path / "c" / "draw.json").read_text())
    assert (other["main"], other["pilot"]) == (draw["main"], draw["pilot"])     # the seed decides the draw
    with pytest.raises(SystemExit, match="--name"):
        bh.main(draw_args(root, sha, set_sha, tmp_path / "d", "--seed", "7", "--name", "../out"))


def test_a_stored_draw_blocks_the_files_before_any_is_written(tmp_path):
    store = DirectoryStore(tmp_path)
    store.put("human-eval/draws/n/checker_clips.csv", b"old")
    with pytest.raises(SystemExit, match="written once"):
        bh.store_draw_files(store, "n", {"draw.json": b"new", "checker_clips.csv": b"other"}, same_bytes_ok=False)
    assert not store.exists("human-eval/draws/n/draw.json") and store.get("human-eval/draws/n/checker_clips.csv") == b"old"
    assert bh.store_draw_files(store, "n", {"draw.json": b"new", "checker_clips.csv": b"old"}, same_bytes_ok=True) == "human-eval/draws/n"
    assert store.get("human-eval/draws/n/draw.json") == b"new"


def test_a_draw_from_before_the_rename_gets_its_sheet_without_drawing_again(tmp_path, capsys):
    root = f"dir:{tmp_path / 'store'}"
    store = DirectoryStore(tmp_path / "store")
    how2sign_csvs(store, [("s1", "train")])
    candidates = [dict(sign="s1", english="I do not", negation_words=["not"]), dict(sign="s2", english="no way", negation_words=["no"])]
    draw = dict(kind="human-eval-how2sign-draw", name="read-1", seed=5, candidates=candidates, rows=dict(s1=dict(seconds=2.), s2=dict(seconds=3.)))
    path = tmp_path / "draw.json"
    path.write_text(json.dumps(draw))
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="does not match"):
        bh.main(["how2sign-checker-sheet", "--store", root, "--draw", str(path), "--draw-sha256", "0" * 64, "--output-dir", str(tmp_path / "o")])
    args = ["how2sign-checker-sheet", "--store", root, "--draw", str(path), "--draw-sha256", pin]
    assert bh.main(args + ["--output-dir", str(tmp_path / "o")]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed == dict(name="read-1", seed=5, rows=2, recordings_not_found=1, store_prefix="human-eval/draws/read-1")
    assert (tmp_path / "o" / "negation_check.csv").read_text().splitlines()[0].split(",") == list(NEGATION_COLUMNS)
    clips = list(csv.DictReader(io.StringIO((tmp_path / "o" / "checker_clips.csv").read_text())))
    assert [c["found"] for c in clips] == ["yes", "no"] and clips[1]["seconds"] == "3.0"
    assert store.get("human-eval/draws/read-1/draw.json") == path.read_bytes()
    assert bh.main(args + ["--output-dir", str(tmp_path / "o2")]) == 0     # the same draw again is fine
    with pytest.raises(SystemExit, match="not empty"):
        bh.main(args + ["--output-dir", str(tmp_path / "o")])
    other = tmp_path / "other.json"
    other.write_text(json.dumps(dict(draw, seed=6)))
    with pytest.raises(SystemExit, match="written once"):                  # another draw under the name
        bh.main(["how2sign-checker-sheet", "--store", root, "--draw", str(other), "--draw-sha256", hashlib.sha256(other.read_bytes()).hexdigest(), "--output-dir", str(tmp_path / "o3")])
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    with pytest.raises(SystemExit, match="not a draw"):
        bh.main(["how2sign-checker-sheet", "--store", root, "--draw", str(bad), "--draw-sha256", hashlib.sha256(bad.read_bytes()).hexdigest(), "--output-dir", str(tmp_path / "o4")])


# ---- P1: the roster, coverage, every row with no choice, and stage 1 kept ---------------------------------------------------------

def written_study(tmp_path, key, *, stage2_for=None, per_group=2, choice="a"):
    """A two-stage study on disk: the key, the roster, receipts, stage 1 for everyone, stage 2 for ``stage2_for`` (default everyone)."""
    (tmp_path / "key.json").write_text(json.dumps(dict(key, key_sha256=hs.key_sha256(key))))
    roster = roster_of(key, per_group)
    (tmp_path / "roster.json").write_text(json.dumps(roster))
    firsts, seconds, receipts = [], [], {}
    for reviewer, g in roster.items():
        one = stage1_text(key, g, reviewer)
        receipts[reviewer] = hashlib.sha256(one).hexdigest()
        path = tmp_path / f"s1-{reviewer[:2]}.csv"
        path.write_bytes(one)
        firsts += ["--sheet", str(path)]
        if stage2_for is None or reviewer in stage2_for:
            two = tmp_path / f"s2-{reviewer[:2]}.csv"
            two.write_bytes(stage2_text(key, one, choice=choice))
            seconds += ["--stage2-sheet", str(two)]
    (tmp_path / "receipts.json").write_text(json.dumps(receipts))
    return ["--key", str(tmp_path / "key.json"), *firsts, *seconds, "--stage1-receipts", str(tmp_path / "receipts.json"), "--reference", "reference"], roster


def test_a_two_stage_key_needs_the_roster_and_reports_coverage_even_when_all_came_back(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    with pytest.raises(SystemExit, match="roster"):
        sh.main(args + ["--output", str(tmp_path / "r.json")])
    assert sh.main(args + ["--roster", str(tmp_path / "roster.json"), "--output", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    coverage = report["coverage"]
    assert coverage["short"] == {} and coverage["threshold"] == .95 and coverage["allowed_below_threshold"] is False
    assert {v["share"] for v in coverage["systems"].values()} == {1.} and {v["share"] for v in coverage["groups"].values()} == {1.}
    assert report["stage1"]["rows"] == 72 and report["problems"] == []


def test_under_95_percent_the_read_lists_every_row_writes_its_report_and_exits_nonzero(tmp_path, capsys):
    key = bar_key(12, 3)
    keep = {rater_of(g, k) for g in range(3) for k in range(2)} - {rater_of(0, 0)}
    args, roster = written_study(tmp_path, key, stage2_for=keep)
    args += ["--roster", str(tmp_path / "roster.json"), "--output", str(tmp_path / "r.json")]
    assert sh.main(args) == 2
    report = json.loads((tmp_path / "r.json").read_text())       # the report is written either way
    assert "--allow-low-coverage" in capsys.readouterr().err and set(report["coverage"]["short"]) >= {"group 0"}
    listed = [p for p in report["problems"] if p.get("no_stage2_choice")]
    assert len(listed) == 12 and {p["reviewer_sha256"] for p in listed} == {rater_of(0, 0)}
    assert {p["item_id"] for p in listed} == {i["item_id"] for i in key["items"] if i["group"] == 0}
    assert sh.main(args + ["--allow-low-coverage"]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["coverage"]["allowed_below_threshold"] is True
    # the flag never touches the bar's own gate
    assert sh.main(args + ["--allow-low-coverage", "--bar", "--bootstrap", "5"]) == 2
    assert "coverage" in json.loads((tmp_path / "r.json").read_text())["bar"]["refused"]


def test_the_list_of_rows_with_no_choice_leaves_the_bar_read_exactly_as_it_was():
    key = bar_key(30, 10)
    items = {i["item_id"]: i for i in key["items"]}
    first, second, receipts = [], [], {}
    for g in range(3):
        for k in range(2):
            r = rater_of(g, k)
            one = stage1_text(key, g, r)
            receipts[r] = hashlib.sha256(one).hexdigest()
            first.append(one)
            blank = {i["item_id"] for i in key["items"] if i["group"] == g and i["position"] == 29} if (g, k) == (0, 0) else set()
            second.append(stage2_text(key, one, edit=lambda row, blank=blank: row.update(choice="" if row["item_id"] in blank else items[row["item_id"]]["correct"])))
    answers, problems = read_two_stage(first, second, key, receipts)
    assert [p["item_id"] for p in problems if p.get("no_stage2_choice")] and all(p.get("no_stage2_choice") for p in problems)
    roster = roster_of(key, 2)
    with_list = bar_read(answers, key, reference="reference", roster=roster, problems=problems, boot=0)
    without = bar_read(answers, key, reference="reference", roster=roster, problems=[], boot=0)
    assert with_list == without and with_list["dropped_rows"] == []
    assert study_coverage(answers, key, roster, problems)["systems"] == with_list["coverage"]["systems"]


def test_stage_1_answers_count_for_stage_1_contrasts_when_stage_2_is_missing():
    key = study(20)
    for i in key["items"]:
        i["english"] = "cut onion"
    roster = {f"{g}{k}".ljust(64, "e"): g for g in range(4) for k in ("a", "b")}
    first, second, receipts = [], [], {}
    lost = "0a".ljust(64, "e")
    for reviewer, g in roster.items():
        one = stage1_text(key, g, reviewer)
        receipts[reviewer] = hashlib.sha256(one).hexdigest()
        first.append(one)
        if reviewer != lost:
            second.append(stage2_text(key, one))
    read_back = read_stages(first, second, key, receipts)
    assert {a["reviewer_sha256"] for a in read_back["answers"]} == set(roster) - {lost}      # no choice, no answer: as before
    assert {a["reviewer_sha256"] for a in read_back["stage1"]} == set(roster) and len(read_back["stage1"]) == 160
    assert [a for a in read_back["stage1"] if a["reviewer_sha256"] == lost and a["choice"]] == []
    plain = word_format_read(read_back["answers"], key)
    kept = word_format_read(read_back["answers"], key, stage1=read_back["stage1"])
    assert kept["raters"] == plain["raters"] == 7 and kept["stage1_raters"] == 8 and kept["stage1_ratings"] == 160 and plain["stage1_ratings"] == 140
    row = "shuffled_minus_composition"
    assert kept["contrasts"][row]["all"]["comprehension_choice"] == plain["contrasts"][row]["all"]["comprehension_choice"]
    # group 0's lost rater is the second rating of its cells: naturalness still has both, read from stage 1
    items = {i["item_id"]: i for i in key["items"]}
    counted = [a for a in read_back["stage1"] if items[a["item_id"]]["group"] == 0]
    assert len(counted) == 40
    summary = hs.stage1_summary(read_back["stage1"], key)
    assert summary["raters"] == 8 and summary["rows"] == 160 and sum(v["without_a_choice"] for v in summary["systems"].values()) == 20


def test_a_stage_1_with_no_receipt_is_kept_for_stage_1_and_cannot_join_or_reach_the_bar():
    key = bar_key(12, 3)
    r0, r1 = rater_of(0, 0), rater_of(1, 0)
    one, other = stage1_text(key, 0, r0), stage1_text(key, 1, r1, unviewable=[next(i["item_id"] for i in key["items"] if i["group"] == 1)])
    receipts = {r0: hashlib.sha256(one).hexdigest()}
    out = read_stages([one, other], [stage2_text(key, one)], key, receipts)
    assert {a["reviewer_sha256"] for a in out["answers"]} == {r0}              # not even its unviewable rows
    assert {a["reviewer_sha256"] for a in out["stage1"]} == {r0, r1} and any("no receipt" in p["issue"] for p in out["problems"])
    # a stage 1 that does not match its receipt reaches neither
    edited = read_stages([one.replace(b",no,no,3,", b",yes,no,3,", 1)], [], key, receipts)
    assert edited["stage1"] == [] and edited["answers"] == [] and any("receipt" in p["issue"] for p in edited["problems"])


def test_a_stranger_or_a_checker_is_not_counted_and_the_plain_read_exits_nonzero(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    stranger = "f" * 64
    one = stage1_text(key, 0, stranger)
    path = tmp_path / "stranger.csv"
    path.write_bytes(one)
    receipts = json.loads((tmp_path / "receipts.json").read_text())
    (tmp_path / "receipts.json").write_text(json.dumps(dict(receipts, **{stranger: hashlib.sha256(one).hexdigest()})))
    two = tmp_path / "stranger2.csv"
    two.write_bytes(stage2_text(key, one))
    full = [*args, "--sheet", str(path), "--stage2-sheet", str(two), "--roster", str(tmp_path / "roster.json"), "--output", str(tmp_path / "r.json")]
    assert sh.main(full) == 2
    report = json.loads((tmp_path / "r.json").read_text())
    assert any("not on the roster" in p["issue"] for p in report["problems"]) and report["reviewers"] == 6
    assert sh.main(full + ["--bar", "--bootstrap", "5"]) == 2                  # the bar read refuses it, as it always did
    assert "not on the roster" in json.loads((tmp_path / "r.json").read_text())["bar"]["refused"]
    kept, problems = hs.people_problems([dict(reviewer_sha256="c" * 64, item_id=key["items"][0]["item_id"])], dict(key, negation_checkers=["c" * 64]), {"c" * 64: 0})
    assert kept == [] and "checker" in problems[0]["issue"]
    kept, problems = hs.people_problems([dict(reviewer_sha256=A_KEY, item_id=key["items"][0]["item_id"])], dict(key, excluded_raters=[A_KEY]), {A_KEY: 0})
    assert kept == [] and "excluded" in problems[0]["issue"]


A_KEY = "a" * 64


# ---- P2: a repeated item_id --------------------------------------------------------------------------------------------------------

def test_a_repeated_item_is_a_problem_on_both_rows_in_both_parsers():
    key = bar_key(12, 3)
    r0 = rater_of(0, 0)
    one = stage1_text(key, 0, r0)
    lines = one.decode().splitlines(True)
    repeated = "".join(lines + lines[1:2])
    answers, problems = parse_stage1(repeated, key)
    rid = lines[1].split(",")[0]
    assert len(answers) == 11 and rid not in {a["item_id"] for a in answers}
    assert [p["row"] for p in problems] == [2, 14] and all("repeated" in p["issue"] and p["item_id"] == rid and p["reviewer_sha256"] == r0 for p in problems)
    two = stage2_text(key, one).decode().splitlines(True)
    rid2 = two[1].split(",")[0]
    state = {r0: dict(sha256=hashlib.sha256(one).hexdigest(), answers={a["item_id"]: a for a in parse_stage1(one.decode(), key)[0]})}
    choices, problems = parse_stage2("".join(two + two[1:2]), key, state)
    assert (r0, rid2) not in choices and len(choices) == 11 and len(problems) == 2 and all("repeated" in p["issue"] for p in problems)


def test_a_repeated_item_stops_the_release_and_counts_as_lost_in_a_read(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    r0 = rater_of(0, 0)
    one = stage1_text(key, 0, r0)
    lines = one.decode().splitlines(True)
    path = tmp_path / "sheet.csv"
    path.write_text("".join(lines + lines[1:2]))
    release = ["how2sign-stage2", "--key", str(tmp_path / "key.json"), "--roster", str(tmp_path / "roster.json"), "--stage1-sheet", str(path),
               "--receipts", str(tmp_path / "new-receipts.json"), "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="repeated"):
        bh.main(release)
    assert not (tmp_path / "new-receipts.json").exists() and not (tmp_path / "out").exists()
    # in a read: neither row counts, and the item counts as lost, by the rule the bar read already has
    receipts = {r0: hashlib.sha256(path.read_bytes()).hexdigest()}
    answers, problems = read_two_stage([path.read_bytes()], [], key, receipts)
    rid = lines[1].split(",")[0]
    assert rid not in {a["item_id"] for a in answers} and any(p.get("item_id") == rid and "repeated" in p["issue"] for p in problems)
    system = next(i["system"] for i in key["items"] if i["item_id"] == rid)
    assert study_coverage(answers, key, roster, problems)["systems"][system]["dropped"] == 1


# ---- P3: a partial stage 1 --------------------------------------------------------------------------------------------------------

def test_a_partial_stage_1_needs_the_flag_and_loses_nothing(tmp_path, capsys):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    r0 = rater_of(0, 0)
    mine = sorted((i for i in key["items"] if i["group"] == 0), key=lambda i: i["position"])
    skip = [i["item_id"] for i in mine[9:]]
    sheet = stage1_text(key, 0, r0, skip=skip)
    path = tmp_path / "partial.csv"
    path.write_bytes(sheet)
    release = ["how2sign-stage2", "--key", str(tmp_path / "key.json"), "--roster", str(tmp_path / "roster.json"), "--stage1-sheet", str(path),
               "--receipts", str(tmp_path / "new-receipts.json"), "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match=r"3 of its 12 rows unanswered.*--partial"):
        bh.main(release)
    assert not (tmp_path / "new-receipts.json").exists() and not (tmp_path / "out").exists() and path.read_bytes() == sheet
    assert bh.main(release + ["--partial"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["unanswered"] == 3
    assert json.loads((tmp_path / "new-receipts.json").read_text())[r0] == hashlib.sha256(sheet).hexdigest()
    sent = list(csv.DictReader(io.StringIO((tmp_path / "out" / f"stage2-{r0[:12]}.csv").read_text())))
    assert {r["item_id"] for r in sent} == {i["item_id"] for i in mine[:9]}
    # a complete sheet needs no flag; a row marked unviewable is an answer, not a gap
    full = tmp_path / "full.csv"
    full.write_bytes(stage1_text(key, 0, rater_of(0, 1), unviewable=[mine[0]["item_id"]]))
    assert bh.main([*release[:5], "--stage1-sheet", str(full), *release[7:]]) == 0


# ---- P4: the owner's flow ---------------------------------------------------------------------------------------------------------

RATER_PAGES = ["docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md", "docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md", "reviewer_package/FOR_REVIEWERS.md",
               "reviewer_package/RATING_SHEET.md", "reviewer_package/RATING_SHEET_TWO_STAGE.md", "reviewer_package/CHECKER_GUIDE.md"]


@pytest.mark.parametrize("name", RATER_PAGES)
def test_nothing_a_rater_reads_asks_for_a_terminal_a_hash_or_a_csv(name):
    text = read(name)
    if name.endswith("CHECKER_GUIDE.md"):                               # its last paragraph is addressed to the owner
        text = text[:text.index("**For [owner's name], on the sheet")]
    lowered = text.lower()
    for word in ("terminal", "shasum", "sha-256", "sha256", "passphrase", ".csv", "`viewable`", "reviewer key"):
        assert word not in lowered, word


@pytest.mark.parametrize("name", ["docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md", "reviewer_package/RATING_SHEET_TWO_STAGE.md"])
def test_the_two_stage_pages_reveal_nothing_of_the_arms(name):
    text = read(name).lower()
    for word in ("shuffl", "scrambl", "permut", "word order", "word-order", "english order", "order of the words", "oracle", "composition", "stitch",
                 "word-for-word", "word by word", "real clip", "generated", "control"):
        assert word not in text, word


def test_the_rater_guide_gives_the_time_the_breaks_and_takes_gloss():
    guide = read("docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md")
    assert "about 2.5 hours" in guide and "about every 15 minutes" in guide and "in English or in gloss" in guide
    assert "printed form" in guide and "in ASL" in guide
    assert "Gloss answers accepted equally with English" in read("reviewer_package/OPERATOR_GUIDE.md")
    assert hs._content_share("CUT ONION", "cut the onion") == 1.       # a gloss answer scores like English on the order-free measure


def test_every_doc_agrees_on_the_owners_flow():
    protocol, guide = read("docs/HUMAN_EVALUATION_PROTOCOL.md"), read("reviewer_package/OPERATOR_GUIDE.md")
    assert "SHA-256 key of a private passphrase" not in protocol and "owner generates" in protocol.lower()
    assert "assumes a rater edits" not in guide and "They never use a terminal or compute a hash" in guide and "openssl rand -hex 32" in guide
    assert "Too technical" not in read("reviewer_package/SOURCES.md")
    assert "Raters never use a terminal or compute a hash" in read("reports/rater_package_plan.md")
    assert "RATING_SHEET_TWO_STAGE.md" in read("reviewer_package/README.md") and "The owner's flow" in read("docs/NEXT_PHASE_PLAN.md")


# ---- P5: the avatar's acceptance --------------------------------------------------------------------------------------------------

def fake_renderer(monkeypatch, *, mirror=True, passed=None):
    import render_avatar
    calls = []

    def main(argv):
        out = Path(argv[argv.index("--out-dir") + 1])
        out.mkdir()
        (out / "avatar.mp4").write_bytes(b"\0\0\0\x18ftypmp42")
        calls.append(argv)
        ok = len(calls) % 2 == 0 if passed is None else passed
        record = dict(frames=3, acceptance=dict(groups=dict(body=dict(threshold_median_deg=5., median_deg=1.5 + len(calls), n=9, passed=ok)), mirror=mirror, passed=ok and mirror))
        record["render_settings_sha256"] = render_avatar.settings_sha256(768)
        (out / "verification.json").write_text(json.dumps(record))
        return 0
    monkeypatch.setattr(render_avatar, "main", main)
    return calls


def small_state():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class State:
        channels: dict = field(default_factory=dict)
        timestamps: tuple = (0., 1., 2.)

        def save(self, path):
            Path(path).write_bytes(b"x")
    return State()


def test_the_render_keeps_its_verification_and_a_mirror_failure_stops_it(tmp_path, monkeypatch):
    fake_renderer(monkeypatch, passed=True)
    made = bh.render_avatar_video(small_state(), tmp_path / "a.mp4", verification=tmp_path / "k" / "a.json")
    assert made["frames"] == 3 and made["acceptance"]["mirror"] is True and (tmp_path / "a.mp4").exists()
    assert json.loads((tmp_path / "k" / "a.json").read_text())["acceptance"] == made["acceptance"]
    fake_renderer(monkeypatch, mirror=False, passed=True)
    with pytest.raises(SystemExit, match="mirror"):
        bh.render_avatar_video(small_state(), tmp_path / "b.mp4", verification=tmp_path / "k" / "b.json")
    assert not (tmp_path / "b.mp4").exists() and not (tmp_path / "k" / "b.json").exists()


def test_acceptance_is_summarised_by_arm():
    def item(system, passed, mirror=True, angle=2.):
        return dict(system=system, acceptance=dict(passed=passed, mirror=mirror, groups=dict(body=dict(passed=passed, median_deg=angle))))
    got = bh.avatar_acceptance([item("a", True), item("a", False, angle=9.), item("b", True), dict(system="c")])
    assert got == dict(a=dict(clips=2, passed=1, mirror_passed=2, groups=dict(body=dict(passed=1, worst_median_deg=9.))),
                       b=dict(clips=1, passed=1, mirror_passed=1, groups=dict(body=dict(passed=1, worst_median_deg=2.))))
    assert bh.avatar_acceptance([dict(system="c")]) == {}


def avatar_packet(tmp_path, store, sha, monkeypatch, *extra):
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    assert ce.main(args) == 0
    pin = lambda key: f"{key}@{hashlib.sha256(store.get(key)).hexdigest()}"   # noqa: E731
    sheet = tmp_path / "checked.csv"
    sheet.write_text(",".join(NEGATION_COLUMNS) + "\n")
    return ["how2sign-packet", "--store", info["root"], "--draw", str(info["draw_path"]), "--draw-sha256", info["draw_sha"], "--negation-sheet", str(sheet),
            "--negation-sheet-sha256", hashlib.sha256(sheet.read_bytes()).hexdigest(), "--seed", "5", "--set", "pilot",
            "--system", f"mixed=composition:{pin('x/arm/word-format/composition_shuffled/index.json')}",
            "--system", f"real=composition:{pin('x/arm/word-format/oracle/index.json')}",
            "--system", f"arm=composition:{pin('x/arm/index.json')}", "--system", "truth=reference", *extra], info


def test_a_packet_keeps_acceptance_on_every_item_by_arm_and_out_of_the_zips(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    packet, info = avatar_packet(tmp_path, store, sha, monkeypatch)
    calls = fake_renderer(monkeypatch)
    out = tmp_path / "packet"
    assert bh.main(packet + ["--renderer", "avatar", "--output-dir", str(out)]) == 0
    key = json.loads((out / "key" / "key.json").read_text())
    assert len(calls) == 16
    for i in key["items"]:
        assert i["renderer"] == "avatar" and i["acceptance"]["mirror"] is True and "body" in i["acceptance"]["groups"]
        kept = json.loads((out / "key" / "avatar-verification" / f"{i['item_id']}.json").read_text())
        assert kept["acceptance"] == i["acceptance"]
    summary = key["avatar_acceptance"]
    assert set(summary) == set(key["systems"]) and sum(v["clips"] for v in summary.values()) == 16 and sum(v["passed"] for v in summary.values()) == 8
    assert all(v["mirror_passed"] == v["clips"] for v in summary.values())
    for g in range(4):
        with zipfile.ZipFile(out / "packet" / f"group-{g}.zip") as z:
            assert not any("verification" in n or "acceptance" in z.read(n).decode(errors="ignore") for n in z.namelist())
    assert not (out / "packet" / "group-0" / "verification.json").exists()
    # a mirror failure stops the build before any key exists
    fake_renderer(monkeypatch, mirror=False)
    with pytest.raises(SystemExit, match="mirror"):
        bh.main(packet + ["--renderer", "avatar", "--output-dir", str(tmp_path / "packet2")])
    assert not (tmp_path / "packet2" / "key" / "key.json").exists()


def test_the_pages_claim_only_what_the_renderer_shows():
    source = read("scripts/render_avatar.py")
    assert "No lip geometry" in source and '"hair", "brows", "lashes"' in source              # if this changes, the pages change with it
    for name in ("docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md", "docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md"):
        guide = read(name)
        assert "short dark hair" in guide and "no separate lips" in guide and "painted on" not in guide and "no hair" not in guide and "shows eyebrow movements" not in guide and "on purpose" not in guide.split("## What you will see")[1].split("##")[0]


# ---- P6 and P7: the pages, the debrief and the board --------------------------------------------------------------------------------

def test_the_page_for_this_read_says_what_the_protocol_relies_on_and_the_old_one_is_aside():
    page = read("reviewer_package/FOR_REVIEWERS.md")
    assert "hard to follow or poor" in page and "**on purpose**" in page and "About **2.5 hours in all**" in page and "one sign" not in page
    assert "Second part" in page and "on paper" in page
    assert "some videos may be hard or poor, some of that on purpose" in read("docs/HUMAN_EVALUATION_PROTOCOL.md")
    old = read("reviewer_package/FOR_REVIEWERS_ORIGINAL_SESSION.md")
    assert "**30 to 45 minutes**" in old and "FOR_REVIEWERS_ORIGINAL_SESSION.md" in read("reviewer_package/README.md")
    assert "FOR_REVIEWERS_ORIGINAL_SESSION.md" in read("reviewer_package/OPERATOR_GUIDE.md")


def test_the_debrief_says_what_was_actually_made():
    debrief = read("reviewer_package/DEBRIEF.md")
    assert "Small words" in debrief and "were not signed" in debrief and "that word was left out" in debrief and "fewer signs" in debrief
    assert "everyone in your part of the study" in debrief and "one sign for each English word" not in debrief and "a sign for each English word" not in debrief
    assert "For the first part of the study" not in debrief


@pytest.mark.parametrize("flag", ["Computer-made signing", "Cloud storage and retention", "Public release", "Independent contact", "Per-rater rows", "Training use", "Attribution"])
def test_the_board_is_asked_and_no_consent_wording_was_drafted_for_it(flag):
    guide = read("reviewer_package/OPERATOR_GUIDE.md")
    assert flag in guide[guide.index("**Questions for the review board.**"):guide.index("**What changed for raters:**")]
    assert "**`report.json` is not aggregate:**" in guide


def test_the_rater_consent_wording_is_left_for_the_board():
    assert "I will watch short videos of computer-made signing and answer questions about them." in read("reviewer_package/CONSENT_FORM.md")


def test_the_attribution_the_plan_asks_for_is_on_a_page_raters_see():
    assert "attribute and state changes" in read("reports/rater_package_plan.md")
    page = read("reviewer_package/FOR_REVIEWERS.md")
    assert "How2Sign" in page and "CC BY-NC 4.0" in page and "We changed that material" in page and "ASLLVD" in page


# ---- P8: the owner's note names each corpus --------------------------------------------------------------------------------------------

def test_a_corpus_the_note_does_not_name_is_refused():
    wf.check_oracle_corpora(("asllvd-text-v1",), OWNER_NOTE)
    for corpus in ("rit-text-r6", "wlasl-text-r6"):
        with pytest.raises(SystemExit, match="does not name"):
            wf.check_oracle_corpora((corpus,), OWNER_NOTE)              # the note names them only to say they are not covered
        with pytest.raises(SystemExit, match="does not name"):
            wf.check_oracle_corpora(("asllvd-text-v1", corpus), OWNER_NOTE)
    with pytest.raises(SystemExit, match="does not name"):
        wf.check_oracle_corpora(("asllvd-text-v1",), "the owner agreed to everything")
    with pytest.raises(SystemExit, match="does not name"):
        wf.check_oracle_corpora(("rit-text-r6",), "The owner has not cleared RIT clips.")
    wf.check_oracle_corpora(("rit-text-r6",), "The owner cleared RIT clips for raters, 2027-01-01.")
    assert not wf.note_names("merit and spirit", "rit-text-r6")         # a whole word


def test_the_packet_refuses_an_index_whose_note_names_another_corpus():
    arms = built()
    arms["real"]["word_format"]["licence_note"] = "owner cleared WLASL clips"
    with pytest.raises(SystemExit, match="does not name"):
        bh.check_word_format(PACKET, arms, which="pilot", signs=SIGNS)


def test_the_docs_keep_rit_and_wlasl_uncovered():
    guide = read("reviewer_package/OPERATOR_GUIDE.md")
    assert "RIT and WLASL aren't covered" in guide and "Please decide whether train clips" not in guide
    assert "may be shown under the DAI terms (§10)" not in guide
    assert "| ASLLVD train |" in guide


# ---- P9: the official arm's exact export ----------------------------------------------------------------------------------------------

def plan_command():
    plan = read("docs/NEXT_PHASE_PLAN.md")
    section = plan[plan.index("### Rater read 1: tool fixes and the official arm's export"):]
    block = section[section.index("```\n") + 4:]
    return shlex.split(block[:block.index("```")].replace("\\\n", " ").replace("<", "'<").replace(">", ">'")), section


def test_the_plan_holds_the_official_arms_command():
    command, section = plan_command()
    value = lambda flag: command[command.index(flag) + 1]    # noqa: E731
    assert command[:3] == ["python", "scripts/round15_compose_eval.py", "run"]
    assert value("--core-margin") == "0.1" and value("--word-cache-n") == "16" and value("--export-forms") == "core_crossfade@predicted"
    assert "--predicted-duration" in command and "--export-only" in command and "--export-word-format" in command
    assert "no-decoder-gains" not in " ".join(command) and "--no-gains" not in command
    assert value("--word-cache-key").endswith("round20/phase-a/cache-w14b-r10/index.json") and value("--mix-sha256") == bh.EVAL_MIX_SHA and value("--sentences-sha256") == bh.D20_SHA256
    assert "--expect-duration-model=-0.083,0.3691,18289" in command
    wf.check_oracle_corpora(("asllvd-text-v1",), value("--word-format-licence-note"))
    assert value("--word-format-licence-note") == OWNER_NOTE
    assert "`--core-margin 0.1`, verified from stored records" in section and "34f85ecf" in section and "85388696" in section


def test_every_flag_of_the_plans_command_is_one_the_export_takes():
    source = read("scripts/round15_compose_eval.py")
    command, _ = plan_command()
    flags = {c.partition("=")[0] for c in command if c.startswith("--")}
    assert flags and all(f'"{f}"' in source for f in flags), sorted(f for f in flags if f'"{f}"' not in source)


def test_the_arms_refuse_any_form_but_predicted_at_the_export_and_at_the_packet():
    with pytest.raises(SystemExit, match="@predicted"):
        wf.check_predicted_form("core_crossfade")
    wf.check_predicted_form("core_crossfade@predicted")
    with pytest.raises(SystemExit, match="@predicted"):
        bh.check_word_format(PACKET, built(form="core_crossfade"), which="pilot", signs=SIGNS)
    assert bh.check_word_format(PACKET, built(), which="pilot", signs=SIGNS)["signs"] == SIGNS


def test_the_export_refuses_word_format_arms_at_reference_length(tmp_path, stage2_store, monkeypatch, capsys):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    k = args.index("--export-forms")
    with pytest.raises(SystemExit):
        ce.main(args[:k] + ["--export-forms", "core_crossfade"] + args[k + 2:])
    assert "@predicted" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        ce.main(args[:k] + args[k + 2:])
    assert not store.exists("x/arm/index.json")


def test_the_how2sign_packet_needs_a_named_renderer(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        bh.main(["how2sign-packet", "--draw", "d", "--draw-sha256", "0", "--negation-sheet", "n", "--negation-sheet-sha256", "0", "--set", "pilot", "--system", "r=reference", "--output-dir", str(tmp_path / "o")])
    assert stop.value.code == 2 and "--renderer" in capsys.readouterr().err


# ---- P10: the pilot's roster, the roster's shape, what a CSV can throw -----------------------------------------------------------------

def test_a_main_packet_refuses_once_a_pilot_packet_exists(tmp_path, stage2_store, monkeypatch, capsys):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    packet, info = avatar_packet(tmp_path, store, sha, monkeypatch)
    packet = packet + ["--renderer", "skeleton", "--panel", "96"]
    base = [a for n, a in enumerate(packet) if a != "--system" and (n == 0 or packet[n - 1] != "--system")]
    base[base.index("--set") + 1] = "main"
    main_arm = ["--system", "truth=reference"]                        # the fixture's main set is one sentence, so one system
    # before the pilot: the main set warns, and builds
    assert bh.main(base + main_arm + ["--output-dir", str(tmp_path / "m0")]) == 0
    assert "WARNING: a main packet without --excluded-raters-roster" in capsys.readouterr().err
    marker = "human-eval/draws/test-draw/pilot-packet.json"
    assert not store.exists(marker)
    # the pilot's packet marks the draw
    assert bh.main(packet + ["--output-dir", str(tmp_path / "p0")]) == 0
    assert json.loads(store.get(marker))["key_sha256"]
    with pytest.raises(SystemExit, match="--excluded-raters-roster"):
        bh.main(base + main_arm + ["--output-dir", str(tmp_path / "m1")])
    assert not (tmp_path / "m1").exists()
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    with pytest.raises(SystemExit, match="--excluded-raters-roster"):
        bh.main(base + main_arm + ["--excluded-raters-roster", str(empty), "--output-dir", str(tmp_path / "m2")])
    roster = tmp_path / "pilot-roster.json"
    roster.write_text(json.dumps({A_KEY: 0}))
    assert bh.main(base + main_arm + ["--excluded-raters-roster", str(roster), "--output-dir", str(tmp_path / "m3")]) == 0
    assert json.loads((tmp_path / "m3" / "key" / "key.json").read_text())["excluded_raters"] == [A_KEY]
    assert "WARNING" not in capsys.readouterr().err


def test_a_roster_with_a_repeated_key_or_a_thin_group_is_refused(tmp_path):
    repeated = '{"%s": 0, "%s": 1}' % (A_KEY, A_KEY)
    with pytest.raises(ValueError, match="repeats"):
        parse_roster(repeated)
    path = tmp_path / "r.json"
    path.write_text(repeated)
    with pytest.raises(SystemExit, match="repeats"):
        load_roster(path)
    path.write_text("not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        load_roster(path)
    with pytest.raises(SystemExit):
        load_roster(tmp_path / "absent.json")
    assert parse_roster('{"%s": 0, "%s": 1}' % (A_KEY, "b" * 64)) == {A_KEY: 0, "b" * 64: 1}
    key = dict(groups=2)
    assert check_roster({"a": 0, "b": 0, "c": 1, "d": 1}, key) == []
    assert any("group 1 has 1" in i for i in check_roster({"a": 0, "b": 0, "c": 1}, key))
    assert any("group 1 has 0" in i for i in check_roster({"a": 0, "b": 0}, key))
    assert any("group 7" in i for i in check_roster({"a": 0, "b": 0, "c": 1, "d": 1, "e": 7}, key))


def test_a_thin_roster_stops_the_release_and_the_read(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    thin = tmp_path / "thin.json"
    r0 = rater_of(0, 0)
    thin.write_text(json.dumps({r0: 0, rater_of(0, 1): 0, rater_of(1, 0): 1, rater_of(1, 1): 1, rater_of(2, 0): 2}))
    path = tmp_path / "one.csv"
    path.write_bytes(stage1_text(key, 0, r0))
    with pytest.raises(SystemExit, match="group 2 has 1"):
        bh.main(["how2sign-stage2", "--key", str(tmp_path / "key.json"), "--roster", str(thin), "--stage1-sheet", str(path), "--receipts", str(tmp_path / "x.json"), "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "x.json").exists()
    with pytest.raises(SystemExit, match="group 2 has 1"):
        sh.main(args + ["--roster", str(thin), "--output", str(tmp_path / "r.json")])


def test_excel_encodings_are_read_and_the_hash_stays_on_the_files_own_bytes():
    text = "item_id,notes\nx,café ’\n"
    assert decode_sheet(text.encode("utf-8")) == text
    assert decode_sheet(b"\xef\xbb\xbf" + text.encode("utf-8")) == text
    assert decode_sheet(text.encode("cp1252")) == text
    assert decode_sheet(text.encode("utf-16")) == text


@pytest.mark.parametrize("payload,message", [(b"PK\x03\x04rest", "Excel workbook"), (b"a,b\n\x81\x8d\xff\n", "encoding"), (b"a,b\n1\x002\n", "NUL")])
def test_a_file_that_cannot_be_read_gets_a_plain_message(payload, message):
    with pytest.raises(SheetError, match=message):
        decode_sheet(payload)


def test_the_release_gives_a_plain_message_for_a_bad_file_and_takes_excels_encoding(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    r0 = rater_of(0, 0)
    path = tmp_path / "sheet.csv"
    release = ["how2sign-stage2", "--key", str(tmp_path / "key.json"), "--roster", str(tmp_path / "roster.json"), "--stage1-sheet", str(path),
               "--receipts", str(tmp_path / "new.json"), "--output-dir", str(tmp_path / "out")]
    path.write_bytes(b"PK\x03\x04 workbook")
    with pytest.raises(SystemExit, match="Excel workbook"):
        bh.main(release)
    text = stage1_text(key, 0, r0).decode().replace(f"{r0[:2]} says", "café says", 1)
    path.write_bytes(text.encode("cp1252"))
    assert bh.main(release) == 0
    assert json.loads((tmp_path / "new.json").read_text())[r0] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_read_sets_an_unreadable_sheet_aside_and_keeps_the_rest():
    key = bar_key(12, 3)
    r0 = rater_of(0, 0)
    one = stage1_text(key, 0, r0)
    out = read_stages([b"PK\x03\x04 workbook", one], [b"\xff\xfe\x00", stage2_text(key, one)], key, {r0: hashlib.sha256(one).hexdigest()})
    assert len(out["answers"]) == 12 and any("Excel workbook" in p["issue"] for p in out["problems"])


def test_a_row_with_too_many_cells_is_a_problem_not_a_crash():
    key = bar_key(12, 3)
    r0 = rater_of(0, 0)
    lines = stage1_text(key, 0, r0).decode().splitlines(True)
    rid = lines[1].split(",")[0]
    lines[1] = lines[1].rstrip("\n") + ",an,unquoted,comma\n"
    answers, problems = parse_stage1("".join(lines), key)
    assert len(answers) == 11 and problems[0]["item_id"] == rid and "more cells" in problems[0]["issue"] and problems[0]["reviewer_sha256"] == r0
    # the other parsers too
    two = stage2_text(key, stage1_text(key, 0, r0)).decode().splitlines(True)
    two[1] = two[1].rstrip("\n") + ",extra\n"
    state = {r0: dict(sha256="x", answers={})}
    assert any("more cells" in p["issue"] for p in parse_stage2("".join(two), key, state)[1])
    sheet = checker_rows(CANDIDATES).splitlines(True)
    sheet[1] = sheet[1].rstrip("\n") + ",extra\n"
    assert any("more cells" in p["issue"] for p in parse_negation_sheet("".join(sheet), CANDIDATES)[1])
    from signtranslator.eval_framework.human_study import write_sheet
    single = design_study([dict(sign=f"s{k}", english=f"plain sentence number {k}", truth=dict(negated="no")) for k in range(8)], ["a", "b"], seed=1)
    rows = write_sheet(single, 0).splitlines(True)
    rows[1] = rows[1].rstrip("\n") + ",extra\n"
    assert any("more cells" in p["issue"] for p in parse_sheet("".join(rows), single)[1])


def test_empty_rows_from_excel_are_no_rows_and_broken_csv_is_plain():
    key = bar_key(12, 3)
    text = stage1_text(key, 0, rater_of(0, 0)).decode()
    assert parse_stage1(text + ",,,,,,,,,\n,,,,,,,,,\n", key) == parse_stage1(text, key)
    assert parse_stage1("a,b\n1,2\n", key)[1][0]["row"] == 0
    assert parse_negation_sheet("a,b\n", CANDIDATES)[1][0]["row"] == 0


def test_a_one_stage_read_gives_the_same_plain_message_and_takes_excels_encoding(tmp_path):
    from signtranslator.eval_framework.human_study import write_sheet
    single = design_study([dict(sign=f"s{k}", english=f"plain sentence number {k}", truth=dict(negated="no")) for k in range(8)], ["a", "b"], seed=1)
    (tmp_path / "key.json").write_text(json.dumps(single))
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"PK\x03\x04")
    with pytest.raises(SystemExit, match="Excel workbook"):
        sh.main(["--key", str(tmp_path / "key.json"), "--sheet", str(bad), "--output", str(tmp_path / "r.json")])
    good = tmp_path / "good.csv"
    good.write_bytes(write_sheet(single, 0).encode("cp1252"))
    assert sh.main(["--key", str(tmp_path / "key.json"), "--sheet", str(good), "--output", str(tmp_path / "r.json")]) == 0
    assert "coverage" not in json.loads((tmp_path / "r.json").read_text())


def test_a_stage_1_from_off_the_roster_never_counts_even_with_no_stage_2(tmp_path):
    key = bar_key(12, 3)
    args, roster = written_study(tmp_path, key)
    path = tmp_path / "stranger.csv"
    path.write_bytes(stage1_text(key, 0, "f" * 64))
    assert sh.main([*args, "--sheet", str(path), "--roster", str(tmp_path / "roster.json"), "--output", str(tmp_path / "r.json")]) == 2
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["stage1"]["raters"] == 6 and any("not on the roster" in p["issue"] for p in report["problems"])


def test_an_unreadable_negation_sheet_stops_the_packet_plainly(tmp_path):
    draw = dict(candidates=[], counts=dict(eligible=1))
    path = tmp_path / "draw.json"
    path.write_text(json.dumps(draw))
    sheet = tmp_path / "n.csv"
    sheet.write_bytes(b"PK\x03\x04")
    with pytest.raises(SystemExit, match="Excel workbook"):
        bh.main(["how2sign-packet", "--draw", str(path), "--draw-sha256", hashlib.sha256(path.read_bytes()).hexdigest(), "--negation-sheet", str(sheet),
                 "--negation-sheet-sha256", hashlib.sha256(b"PK\x03\x04").hexdigest(), "--set", "pilot", "--renderer", "skeleton", "--system", "r=reference", "--output-dir", str(tmp_path / "o")])


# ---- P11: a pilot's options ------------------------------------------------------------------------------------------------------------

def test_a_pilots_options_hold_no_negation_word_and_the_main_sets_pool_stays():
    texts = [f"plain sentence number {k}" for k in range(20)] + [f"plain sentence not {k}" for k in range(20)]
    chosen = [dict(sign=f"s{k}", english=texts[k]) for k in range(6)]
    assert bh.without_negation(texts) == texts[:20]
    pilot = bh.how2sign_key(chosen, ["a", "b"], seed=1, d20_texts=texts, confirmed={}, eligible=40, pool=bh.without_negation(texts))
    assert not any(negation_words(t) for i in pilot["items"] for t in i["options"])
    main = bh.how2sign_key(chosen, ["a", "b"], seed=1, d20_texts=texts, confirmed={}, eligible=40)
    assert any(negation_words(t) for i in main["items"] for t in i["options"])     # 20 of 40 here: the whole pool


def test_the_whole_pilot_packet_filters_and_says_so(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    store, sha, s1, oracle, argv = stage2_store
    packet, info = avatar_packet(tmp_path, store, sha, monkeypatch, "--renderer", "skeleton", "--panel", "96")
    texts = [r["text"].strip() for r in json.loads(store.get("sets/d20.json"))["records"]]
    flagged = next(t for t in texts if t not in {info["draw"]["rows"][s]["english"] for s in info["draw"]["pilot"]})
    monkeypatch.setattr(bh, "negation_words", lambda t: ["off"] if t == flagged else [])
    assert bh.main(packet + ["--output-dir", str(tmp_path / "p")]) == 0
    key = json.loads((tmp_path / "p" / "key" / "key.json").read_text())
    assert key["distractors"]["without_negation_words"] is True
    assert flagged not in {t for i in key["items"] for t in i["options"]}
    # a pilot sentence that holds one is refused: the draw says the pilot holds none
    pilot_text = info["draw"]["rows"][info["draw"]["pilot"][0]]["english"]
    monkeypatch.setattr(bh, "negation_words", lambda t: ["off"] if t == pilot_text else [])
    with pytest.raises(SystemExit, match="pilot sentence holds a negation word"):
        bh.main(packet + ["--output-dir", str(tmp_path / "p2")])


# ---- P12: the doc errors ---------------------------------------------------------------------------------------------------------------

def test_the_protocol_no_longer_says_every_clip_runs_at_the_reference_length():
    protocol = read("docs/HUMAN_EVALUATION_PROTOCOL.md")
    assert "\nAll systems sign on the recorded utterance's clock, so duration is no cue." not in protocol
    assert "In the first How2Sign read they do not" in protocol


def test_the_package_plan_names_the_arm_actually_used():
    plan = read("reports/rater_package_plan.md")
    assert "First arm, no-gains stack" not in plan and "First arm: W14b, gains on, best of 16" in plan and "not the no-gains stack" in plan
    assert "`@predicted` waits for the next read" not in plan


def test_the_power_scripts_docstring_is_the_command_that_wrote_the_stored_report():
    import rater_pilot_power as power
    stored = json.loads(read("reports/rater_pilot_power.json"))
    assert f"--sims {stored['sims']} --seed {stored['seed']}" in power.__doc__ and "--sims 2000" not in power.__doc__


def test_left_out_sentences_reach_the_key_and_the_report(tmp_path):
    key = bar_key(12, 3)
    key["excluded_no_selected_word"] = ["left-1", "left-2"]
    key["negation_read_first"] = ["s0"]
    key["avatar_acceptance"] = dict(reference=dict(clips=1))
    args, roster = written_study(tmp_path, key)
    assert sh.main(args + ["--roster", str(tmp_path / "roster.json"), "--output", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["excluded_no_selected_word"] == ["left-1", "left-2"] and report["negation_read_first"] == ["s0"] and report["avatar_acceptance"] == dict(reference=dict(clips=1))


def test_the_packet_copies_the_draws_left_out_sentences_to_the_key(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    import scripts.round15_compose_eval as sce
    store, sha, s1, oracle, argv = stage2_store
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    monkeypatch.setattr(bh, "negation_words", lambda t: [])
    stored = json.loads(store.get("sets/d20.json"))
    wordless = stored["records"][2]["text"]
    real = ce.stage1_selection

    def selection(texts, vocab):
        details, words = real(texts, vocab)
        return details, [[] if t == wordless else w for t, w in zip(texts, words)]
    for module in (ce, sce, sys.modules["round15_compose_eval"]):
        monkeypatch.setattr(module, "stage1_selection", selection)
    root = f"dir:{tmp_path / 'store'}"
    assert bh.main(draw_args(root, sha, set_sha, tmp_path / "d", "--seed", "1", "--name", "t", main="2", pilot="2")) == 0
    sheet = tmp_path / "checked.csv"
    sheet.write_text(",".join(NEGATION_COLUMNS) + "\n")
    draw_path = tmp_path / "d" / "draw.json"
    assert bh.main(["how2sign-packet", "--store", root, "--draw", str(draw_path), "--draw-sha256", hashlib.sha256(draw_path.read_bytes()).hexdigest(), "--negation-sheet", str(sheet),
                    "--negation-sheet-sha256", hashlib.sha256(sheet.read_bytes()).hexdigest(), "--set", "main", "--renderer", "skeleton", "--panel", "96",
                    "--system", "truth=reference", "--output-dir", str(tmp_path / "p")]) == 0
    key = json.loads((tmp_path / "p" / "key" / "key.json").read_text())
    left_out = stored["records"][2]["sign"]
    assert key["excluded_no_selected_word"] == [left_out] and left_out not in {i["sign"] for i in key["items"]}


def test_the_operator_guide_has_no_stale_decision():
    guide = read("reviewer_package/OPERATOR_GUIDE.md")
    line = next(x for x in guide.splitlines() if x.startswith("- **ASLLVD train clips: confirmed by you"))
    assert "Please decide" not in line and "can be named under the same note" not in line
    assert "because timing, signers and wrist speed already fail it" not in guide


def test_no_ai_disclosure_or_ci_in_the_new_docs():
    for name in ("reviewer_package/CHECKER_GUIDE.md", "reviewer_package/CHECKER_CONSENT_FORM.md", "reviewer_package/RATING_SHEET_TWO_STAGE.md", "reviewer_package/FOR_REVIEWERS.md"):
        text = read(name)
        assert not re.search(r"\b(claude|anthropic|generated by|continuous integration)\b", text, re.IGNORECASE), name


# ---- the three that earlier review left to mutants ---------------------------------------------------------------------------------------

def test_the_validation_pool_never_holds_c00_or_another_signer():
    def record(k, component, **change):
        return dict(sign=f"s{k}", corpus=lib.H2S, partition="val", signer_component=component, text=f"text {k}", **change)
    keep = [record(k, c) for k, c in enumerate(("c01", "c03", "c05"))]
    planted = [record(10, "c00"), record(11, "c11"), record(12, None), dict(record(13, "c01"), partition="test"), dict(record(14, "c01"), corpus="asllvd-text-v1"), dict(record(15, "c01"), text="")]
    assert lib.validation_pool(dict(records=keep + planted)) == keep


@pytest.mark.parametrize("bad", [dict(signer_component="c00"), dict(partition="test"), dict(partition="train"), dict(corpus="asl-citizen")])
def test_the_set_loader_refuses_c00_and_test(bad):
    base = dict(sign="x", text="hello there", corpus=lib.H2S, partition="val", signer_component="c01", continuous=dict(state_sha256="a"))
    stored = dict(kind="round20-sentence-set", manifest_key="m", manifest_sha256="s", records=[dict(sign="x", text="hello there", state_sha256="a")])
    assert lib.load_sentence_set(stored, dict(records=[base]), "m", "s")[0]["sign"] == "x"
    with pytest.raises(SystemExit, match="not this manifest's validation record"):
        lib.load_sentence_set(stored, dict(records=[dict(base, **bad)]), "m", "s")


def test_the_join_log_reads_the_moved_order_and_the_compositions_own_plan(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    """Each row's ``joins`` measure the parts in the moved order against the plan the moved clip used; its ``composition_joins`` measure each
    word's own part against the composition's own plan: the same gaps, each word's own core frames."""
    import round15_stage2 as r2
    store, sha, s1, oracle, argv = stage2_store
    args, info = pilot_setup(tmp_path, store, sha, monkeypatch)
    plan_words = r2.plan_words

    def unequal(words, model, vocab, **options):                      # the fixture's words all get one length: moving them would show nothing
        plan = plan_words(words, model, vocab, **options)
        return dict(plan, weights=[w * (1 + k) for k, w in enumerate(plan["weights"])])
    monkeypatch.setattr(r2, "plan_words", unequal)
    calls, plans = [], {}
    real = wf.join_excess
    monkeypatch.setattr(wf, "join_excess", lambda parts, plan, *, min_speed: (calls.append((parts, plan)), real(parts, plan, min_speed=min_speed))[1])
    for module in {ce, sys.modules.get("round15_compose_eval"), sys.modules.get("scripts.round15_compose_eval")} - {None}:
        original = module.render_round20

        def spy(a, material, job, secs, model, vocab, *, original=original, **options):
            out = original(a, material, job, secs, model, vocab, **options)
            plans.setdefault(options.get("sample_id") or "", out[1])
            return out
        monkeypatch.setattr(module, "render_round20", spy)
    assert ce.main(args) == 0
    moved_rows = json.loads(store.get("x/arm/word-format/composition_shuffled/index.json"))["states"]
    assert len(calls) == 2 * len(moved_rows) > 0
    compared = told_apart = 0
    for k, row in enumerate(moved_rows):
        moved, own = calls[2 * k], calls[2 * k + 1]
        order = row["order"]
        for at, word in enumerate(order):
            one, two = moved[0]["cores"][at], own[0]["cores"][word]
            for name in one.channels:
                assert np.array_equal(one.channels[name].values, two.channels[name].values, equal_nan=True), (at, word, name)
        assert [own[1]["cores"][w] for w in order] == list(moved[1]["cores"]) and list(own[1]["gaps"]) == list(moved[1]["gaps"])
        composition = plans[f"round15-eval-core_crossfade@predicted:{row['clip']}"]       # the plan the exported composition was rendered on
        assert list(composition["cores"]) == list(own[1]["cores"]) and list(composition["gaps"]) == list(own[1]["gaps"])
        compared += 1
        told_apart += len(set(composition["cores"])) > 1 and list(composition["cores"]) != list(moved[1]["cores"])
    assert compared == len(moved_rows) and told_apart > 0


def test_the_bar_is_untouched():
    assert BAR == dict(comprehension_lower=.70, within_points=.10, recall_lower=.70, false_alarm_upper=.20, unviewable_share=.05, coverage=.95, comprehension_alpha=.04, gap_alpha=.05)
    protocol = read("docs/HUMAN_EVALUATION_PROTOCOL.md")
    bar = protocol[protocol.index("**Release bar (owner's rule"):protocol.index("The instrument, realism, serving")]
    assert hashlib.sha256(bar.encode()).hexdigest() == "4aaa1ebbd19ae49f638388976c4aeaa6e96aff6f4cabfe3def6874efa304599b"
    with pytest.raises(BarRefused, match="report-only"):
        key = study(20)
        bar_read(eight(key), key, reference="reference", roster={}, boot=0)
