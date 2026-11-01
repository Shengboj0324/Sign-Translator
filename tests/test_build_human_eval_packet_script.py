"""Review packet builder: system specs, display-only gap holding, the test-partition firewall."""
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_human_eval_packet import choose_utterances, held_for_display, main, parse_system  # noqa: E402
from tests.test_round15_stage2 import stage2_store  # noqa: E402,F401  (fixture)
from tests.test_round15_compositional import stubbed  # noqa: E402,F401


def test_system_specs():
    assert parse_system("ref=reference") == dict(label="ref", kind="reference")
    s = parse_system("gen=plan:training-runs/x/step-1.pt@" + "a" * 64)
    assert s == dict(label="gen", kind="plan", key="training-runs/x/step-1.pt", manifest_sha256="a" * 64)
    for bad in ("ref", "g=plan:key", "g=other:k@" + "a" * 64, "=reference"):
        with pytest.raises(ValueError):
            parse_system(bad)


@dataclass(frozen=True)
class C:
    values: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class S:
    channels: dict


def test_gaps_are_held_from_the_nearest_valid_frame_for_display():
    v = np.arange(5.)[:, None, None] * np.ones((5, 2, 1))
    ok = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], bool)
    shown = held_for_display(S(dict(face=C(v, ok))))
    np.testing.assert_array_equal(shown.channels["face"].values[:, 0, 0], [1, 1, 1, 3, 3])
    assert shown.channels["face"].valid[:, 0].all()
    np.testing.assert_array_equal(shown.channels["face"].values[:, 1, 0], v[:, 1, 0])   # never valid: unchanged
    assert not shown.channels["face"].valid[:, 1].any()


def test_utterances_are_distinct_usable_sentences():
    import random
    rec = lambda i, e, p="val", s=3.: dict(sign=f"s{i}", english=e, partition=p, plan=dict(markers=[]), text="X",
                                           continuous=dict(seconds=s))
    records = [rec(0, "A b"), rec(1, "a B"), rec(2, "c"), rec(3, "d", "test"), rec(4, "e", s=.5), rec(5, "f")]
    chosen = choose_utterances(records, "val", 3, 1., random.Random(0))
    assert len({r["english"].lower() for r in chosen}) == 3 and all(r["partition"] == "val" for r in chosen)
    with pytest.raises(SystemExit):
        choose_utterances(records, "val", 4, 1., random.Random(0))


def test_test_partition_needs_the_final_study_flag(tmp_path):
    with pytest.raises(SystemExit, match="final study"):
        main(["--manifest-key", "k", "--manifest-sha256", "0" * 64, "--system", "r=reference", "--partition", "test",
              "--output-dir", str(tmp_path)])


# ---- How2Sign mode ----------------------------------------------------------------------------------------------------

def test_composition_systems_belong_to_the_how2sign_packet(tmp_path):
    s = parse_system("gen=composition:exports/a/index.json@" + "b" * 64)
    assert s == dict(label="gen", kind="composition", key="exports/a/index.json", index_sha256="b" * 64)
    with pytest.raises(SystemExit, match="how2sign-packet"):
        main(["--manifest-key", "k", "--manifest-sha256", "0" * 64, "--system", "gen=composition:k@" + "b" * 64,
              "--output-dir", str(tmp_path)])


def record(k, text, seconds=3.):
    return dict(sign=f"s{k}", text=text, continuous=dict(seconds=seconds, state_key=f"k{k}", state_sha256="0" * 64))


def hands(share):
    ok = np.zeros((100, 2), bool)
    ok[:int(100 * share)] = True
    return S(dict(left_hand=C(None, np.zeros((100, 2), bool)), right_hand=C(None, ok)))


def test_eligible_sentences_are_long_enough_tracked_and_distinct():
    from build_human_eval_packet import eligible_sentences
    records = [record(0, "  Cut it now.  "), record(1, "short one", seconds=1.4), record(2, "lost hands"),
               record(3, "cut IT, now"), record(4, "I do not know"), record(5, "exactly there", seconds=1.5),
               record(6, "lost hands")]
    coverage = dict(s0=.9, s2=.69, s3=.9, s4=.7, s5=1., s6=.8)
    asked = []

    def state_of(r):
        asked.append(r["sign"])
        return hands(coverage[r["sign"]])
    rows, counts = eligible_sentences(records, state_of, min_seconds=1.5, min_hand_coverage=.7)
    assert [r["sign"] for r in rows] == ["s0", "s4", "s5", "s6"]     # s6: its text's first record lost its hands
    assert rows[0]["english"] == "Cut it now."                       # options are compared stripped on the sheets
    assert rows[1]["negation_words"] == ["not"] and rows[0]["negation_words"] == []
    assert "s1" not in asked                                         # no state is loaded for a short clip
    assert counts == dict(sentences=7, too_short=1, hands_lost=1, repeated_text=1, eligible=4, negation_candidates=1)


D20_TEXTS = [f"development sentence number {k}" for k in range(12)]
CHOSEN = [dict(sign=f"s{k}", english=D20_TEXTS[k]) for k in range(6)]


def test_distractors_come_only_from_d20():
    from build_human_eval_packet import check_key_within_d20, d20_distractor_pool, how2sign_key
    assert d20_distractor_pool(D20_TEXTS) == D20_TEXTS
    assert d20_distractor_pool(D20_TEXTS, D20_TEXTS[:5]) == D20_TEXTS[:5]
    beyond = D20_TEXTS + ["a confirmation sentence the read must never show"]
    with pytest.raises(SystemExit, match="not in D20"):
        d20_distractor_pool(D20_TEXTS, beyond)
    with pytest.raises(SystemExit, match="not in D20"):
        how2sign_key(CHOSEN, ["ref", "gen"], seed=1, d20_texts=D20_TEXTS, confirmed={}, eligible=12, pool=beyond)
    key = how2sign_key(CHOSEN, ["ref", "gen"], seed=1, d20_texts=D20_TEXTS, confirmed={}, eligible=12)
    assert {t for i in key["items"] for t in i["options"]} <= set(D20_TEXTS)
    assert key["bar"]["strata"] == dict(confirmed=0., rejected=0., non_candidate=1.) and key["stages"] == 2
    assert all(len(i["item_id"]) == 32 for i in key["items"])         # 128 bits, not from the design's seed
    again = how2sign_key(CHOSEN, ["ref", "gen"], seed=1, d20_texts=D20_TEXTS, confirmed={}, eligible=12)
    assert {i["item_id"] for i in key["items"]}.isdisjoint(i["item_id"] for i in again["items"])
    assert [i["options"] for i in key["items"]] == [i["options"] for i in again["items"]]
    # a study sentence outside D20 enters design_study's pool by itself: the check on the key catches it
    with pytest.raises(SystemExit, match="not in D20"):
        how2sign_key([dict(sign="x", english="fresh text")] + CHOSEN, ["ref", "gen"], seed=1, d20_texts=D20_TEXTS,
                     confirmed={}, eligible=12)
    tampered = dict(items=[dict(options=["fresh text"])])
    with pytest.raises(SystemExit, match="not in D20"):
        check_key_within_d20(tampered, D20_TEXTS)


def test_the_key_counts_only_confirmed_sentences_as_negated():
    from build_human_eval_packet import how2sign_key
    key = how2sign_key(CHOSEN, ["ref", "gen"], seed=1, d20_texts=D20_TEXTS, confirmed=dict(s0="yes", s1="no"),
                       eligible=12)
    truth = {i["sign"]: i["truth"] for i in key["items"]}
    assert truth["s0"]["negated"] == "yes" and truth["s1"]["negated"] == "no" and truth["s2"]["negated"] == "no"
    assert truth["s0"]["sentence_type"] is None and truth["s0"]["topic"] is None
    assert (truth["s0"]["stratum"], truth["s1"]["stratum"], truth["s2"]["stratum"]) == ("confirmed", "rejected", "non_candidate")
    assert key["bar"]["strata"] == dict(confirmed=1 / 12, rejected=1 / 12, non_candidate=10 / 12)
    assert key["bar"]["confirmed"] == 1 and key["bar"]["rejected"] == 1


def test_near_identical_sentences_are_not_offered_together():
    from build_human_eval_packet import how2sign_key
    texts = ["Cut it now.", "cut it, now", "CUT IT NOW!"] + D20_TEXTS[:6]
    key = how2sign_key([dict(sign=f"s{k}", english=texts[k]) for k in (0, 3, 4, 5)], ["ref", "gen"], seed=2,
                       d20_texts=texts, confirmed={}, eligible=12)
    for i in key["items"]:
        if i["english"] == "Cut it now.":
            assert not {"cut it, now", "CUT IT NOW!"} & set(i["options"])


def test_the_draw_refuses_any_set_but_d20(tmp_path, monkeypatch):
    import build_human_eval_packet as bh
    from build_human_eval_packet import how2sign_draw
    from signtranslator.data.text_motion_dataset import DirectoryStore
    import hashlib
    import json
    store = DirectoryStore(tmp_path / "store")
    payload = json.dumps(dict(kind="round20-sentence-set", set="C20", records=[])).encode()
    store.put("sets/c20.json", payload)
    with pytest.raises(SystemExit, match="D20 only"):
        how2sign_draw(["--store", f"dir:{tmp_path / 'store'}", "--sentences-key", "sets/c20.json", "--sentences-sha256",
                       hashlib.sha256(payload).hexdigest(), "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit, match="does not match"):
        how2sign_draw(["--store", f"dir:{tmp_path / 'store'}", "--sentences-key", "sets/c20.json", "--sentences-sha256",
                       "0" * 64, "--output-dir", str(tmp_path / "out")])
    # labelled D20 with another hash: refused by the pin; the right hash with a C20 label: refused by the label
    other = json.dumps(dict(kind="round20-sentence-set", set="D20", records=[])).encode()
    store.put("sets/other.json", other)
    with pytest.raises(SystemExit, match="is not D20's"):
        how2sign_draw(["--store", f"dir:{tmp_path / 'store'}", "--sentences-key", "sets/other.json", "--sentences-sha256",
                       hashlib.sha256(other).hexdigest(), "--output-dir", str(tmp_path / "out")])
    assert bh.D20_SHA256 == "d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe"
    monkeypatch.setattr(bh, "D20_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(SystemExit, match="D20 only, not 'C20'"):
        how2sign_draw(["--store", f"dir:{tmp_path / 'store'}", "--sentences-key", "sets/c20.json", "--sentences-sha256",
                       hashlib.sha256(payload).hexdigest(), "--output-dir", str(tmp_path / "out")])


def composition_index(store, key, *, list_sha="a" * 64, set_sha="d" * 64, rows=()):
    import hashlib
    import json
    payload = json.dumps(dict(kind="round15-composed-states", arm="arm", sentences=dict(sha256=list_sha),
                              sentence_set=dict(sha256=set_sha), word_cache=None, states=list(rows))).encode()
    store.put(key, payload)
    return dict(label="gen", kind="composition", key=key, index_sha256=hashlib.sha256(payload).hexdigest())


def test_a_composition_index_is_pinned_and_complete(tmp_path):
    from build_human_eval_packet import load_composition
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path)
    row = lambda sign, empty=False, form="core_crossfade", frames=48, reference_frames=48: dict(   # noqa: E731
        sign=sign, empty=empty, form=form, frames=frames, reference_frames=reference_frames)
    pins = dict(list_sha256="a" * 64, sentences_sha256="d" * 64)
    good = composition_index(store, "good.json", rows=[row("a"), row("b"), row("c")])
    assert set(load_composition(store, good, signs=["a", "b"], **pins)["rows"]) == {"a", "b"}
    with pytest.raises(SystemExit, match="does not match"):
        load_composition(store, dict(good, index_sha256="0" * 64), signs=["a"], **pins)
    with pytest.raises(SystemExit, match="another sentence list"):
        load_composition(store, good, signs=["a"], list_sha256="b" * 64, sentences_sha256="d" * 64)
    with pytest.raises(SystemExit, match="another sentence list"):
        load_composition(store, good, signs=["a"], list_sha256="a" * 64, sentences_sha256="e" * 64)
    with pytest.raises(SystemExit, match="lacks 1"):
        load_composition(store, good, signs=["a", "z"], **pins)
    assert load_composition(store, good, signs=["a"], **pins)["timing"] == "reference-length"
    # a main form that does not fill its reference would make duration a cue
    long = composition_index(store, "long.json", rows=[row("a", frames=60)])
    with pytest.raises(SystemExit, match="duration would be a cue"):
        load_composition(store, long, signs=["a"], **pins)
    # the label comes from the data: a predicted form may differ from its reference
    predicted = composition_index(store, "p.json", rows=[row("a", frames=60, form="core_crossfade@predicted")])
    assert load_composition(store, predicted, signs=["a"], **pins)["timing"] == "predicted"
    mixed = composition_index(store, "m.json", rows=[row("a"), row("b", form="core_crossfade@lowpass")])
    with pytest.raises(SystemExit, match="several forms"):
        load_composition(store, mixed, signs=["a", "b"], **pins)
    bare = composition_index(store, "b.json", rows=[dict(sign="a", empty=False, form="core_crossfade")])
    with pytest.raises(SystemExit, match="no frames"):
        load_composition(store, bare, signs=["a"], **pins)
    two = composition_index(store, "two.json", rows=[row("a"), row("a", form="core_crossfade@lowpass")])
    with pytest.raises(SystemExit, match="several forms"):
        load_composition(store, two, signs=["a"], **pins)
    empty = composition_index(store, "empty.json", rows=[row("a", empty=True), row("b")])
    with pytest.raises(SystemExit, match="no selected word"):
        load_composition(store, empty, signs=["a", "b"], **pins)
    with pytest.raises(SystemExit, match=r"\['a'\]"):                 # the sentences are listed
        load_composition(store, empty, signs=["a", "b"], **pins)
    assert load_composition(store, empty, signs=["b"], **pins)["rows"]["b"]["empty"] is False


def test_every_clock_is_rebuilt_and_zip_times_are_fixed(tmp_path):
    from dataclasses import dataclass as dc
    import time
    import zipfile
    from build_human_eval_packet import bytes_per_frame, uniform_clock, write_group_files, write_zip

    @dc(frozen=True)
    class Clocked:
        timestamps: np.ndarray
    jitter = np.array([5., 5.0401, 5.0845, 5.1249])
    got = uniform_clock(Clocked(jitter), 24.).timestamps
    np.testing.assert_allclose(got, 5. + np.arange(4) / 24.)
    a, b = tmp_path / "b.txt", tmp_path / "a.txt"
    a.write_text("x"); b.write_text("y")
    os_time = time.time() - 12345
    import os
    os.utime(a, (os_time, os_time))
    write_zip(tmp_path / "z.zip", "group-0", [a, b])
    first = (tmp_path / "z.zip").read_bytes()
    os.utime(a, (os_time + 999, os_time + 999))
    write_zip(tmp_path / "z2.zip", "group-0", [a, b])
    assert first == (tmp_path / "z2.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "z.zip") as z:
        assert [i.date_time for i in z.infolist()] == [(1980, 1, 1, 0, 0, 0)] * 2
        assert z.namelist() == ["group-0/a.txt", "group-0/b.txt"] and z.read("group-0/b.txt") == b"x"
    rows = [dict(system="r", video_bytes=1000, frames=10), dict(system="r", video_bytes=3000, frames=10),
            dict(system="g", video_bytes=500, frames=10)]
    assert bytes_per_frame(rows) == dict(g=dict(clips=1, mean=50., low=50., high=50.),
                                         r=dict(clips=2, mean=200., low=100., high=300.))
    # --output-prefix is write-once: nothing is copied over an existing key
    from signtranslator.data.text_motion_dataset import DirectoryStore
    store = DirectoryStore(tmp_path / "store")
    key = dict(groups=1, items=[])
    out = tmp_path / "out"
    (out / "packet" / "group-0").mkdir(parents=True)
    (out / "key").mkdir()
    write_group_files(dict(key), out, store, "p/x")
    assert store.exists("p/x/key/key.json") and store.exists("p/x/packet/group-0.zip")
    store.put("p/y/packet/group-0/sheet.csv", b"old")
    with pytest.raises(SystemExit, match="written once"):
        write_group_files(dict(key), out, store, "p/y")
    assert store.get("p/y/packet/group-0/sheet.csv") == b"old" and not store.exists("p/y/key/key.json")


def test_the_how2sign_packet_end_to_end_with_the_real_renderer(tmp_path, stage2_store, monkeypatch):  # noqa: F811
    """Synthetic smoke: draw, checker's sheet, composed exports, packet (reference and compositions through the same
    skeleton renderer), reviewers' sheets, the bar read."""
    import csv
    import hashlib
    import io
    import json
    import zipfile
    import build_human_eval_packet as bh
    import scripts.round15_compose_eval as ce
    import score_human_eval as sh
    from evaluate_text_to_motion import load_state
    from signtranslator.eval_framework.human_study import STAGE1_COLUMNS, STAGE2_COLUMNS, write_stage1_sheet
    from tests.test_round15_compose_eval import run_args
    from tests.test_round20_eval import put_set
    store, sha, s1, oracle, argv = stage2_store
    root = f"dir:{tmp_path / 'store'}"
    set_sha, _ = put_set(store, sha, order=(0, 1, 2, 3, 4))
    monkeypatch.setattr(bh, "D20_SHA256", set_sha)
    TWO_STAGE = bh.TWO_STAGE_GUIDE
    # the fixture's sentences hold no negation word: a stand-in rule makes one candidate (the rule has its own tests)
    monkeypatch.setattr(bh, "negation_words", lambda t: ["off"] if "off" in t else [])
    draw_dir = tmp_path / "draw"
    assert main(["how2sign-draw", "--store", root, "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key",
                 "sets/d20.json", "--sentences-sha256", set_sha, "--main", "3", "--pilot", "1", "--min-seconds", ".5",
                 "--seed", "11", "--output-dir", str(draw_dir)]) == 0
    draw_path = draw_dir / "draw.json"
    draw = json.loads(draw_path.read_text())
    draw_sha = hashlib.sha256(draw_path.read_bytes()).hexdigest()
    assert draw["counts"]["eligible"] == 5 and draw["counts"]["negation_candidates"] == 1
    assert len(draw["main"]) == 3 and len(draw["pilot"]) == 1 and draw["signs"] == draw["main"] + draw["pilot"]
    candidate = draw["candidates"][0]["sign"]
    assert candidate in draw["main"]
    with pytest.raises(SystemExit, match="not empty"):
        main(["how2sign-draw", "--store", root, "--mix-key", "mix.json", "--mix-sha256", sha, "--sentences-key",
              "sets/d20.json", "--sentences-sha256", set_sha, "--output-dir", str(draw_dir)])
    # the checker confirms it
    rows = list(csv.DictReader(io.StringIO((draw_dir / "negation_check.csv").read_text())))
    assert [r["sign"] for r in rows] == [candidate] and rows[0]["confirmed"] == ""
    sheet_path = tmp_path / "checked.csv"

    def write_checked(confirmed):
        r = dict(rows[0], no_generated_clip_seen="yes", confirmed=confirmed, checker_sha256="c" * 64)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=list(r), lineterminator="\n")
        w.writeheader(); w.writerow(r)
        sheet_path.write_text(out.getvalue())
        return hashlib.sha256(sheet_path.read_bytes()).hexdigest()
    # the composed exports of two arms, on the draw's own sentence list
    systems = ["real=reference"]
    for arm, guidance in (("arm-a", "1.5"), ("arm-b", "2.0")):
        args = [a for a in run_args(tmp_path, sha) if a != "core_crossfade,oracle_stage2"]
        args.insert(args.index("--variants") + 1, "core_crossfade")
        assert ce.main(args + ["--clips", "5", "--guidance", guidance, "--sentences-key", "sets/d20.json",
                               "--sentences-sha256", set_sha, "--export-prefix", f"x/{arm}", "--export-arm", arm,
                               "--export-sentences-file", str(draw_path), "--export-sentences-sha256", draw_sha,
                               "--export-only", "--output-file", str(tmp_path / f"{arm}.json")]) == 0
        index_sha = hashlib.sha256(store.get(f"x/{arm}/index.json")).hexdigest()
        systems.append(f"{arm}=composition:x/{arm}/index.json@{index_sha}")
    packet = ["how2sign-packet", "--store", root, "--draw", str(draw_path), "--draw-sha256", draw_sha,
              "--negation-sheet", str(sheet_path), "--set", "main", "--panel", "96", "--seed", "5"]
    no_reference = list(packet)
    for s in systems:
        packet += ["--system", s]
    for s in systems[1:]:
        no_reference += ["--system", s]
    # an unanswered candidate: the truth is not fixed, nothing is built
    unanswered = write_checked("")
    with pytest.raises(SystemExit, match="not fixed"):
        main(packet + ["--negation-sheet-sha256", unanswered, "--output-dir", str(tmp_path / "p0")])
    assert not (tmp_path / "p0").exists()
    checked = write_checked("yes")
    with pytest.raises(SystemExit, match="negation-sheet-sha256"):
        main(packet + ["--negation-sheet-sha256", "0" * 64, "--output-dir", str(tmp_path / "p0")])
    with pytest.raises(SystemExit, match="one reference"):
        main(no_reference + ["--negation-sheet-sha256", checked, "--output-dir", str(tmp_path / "p0")])
    out = tmp_path / "packet"
    assert main(packet + ["--negation-sheet-sha256", checked, "--output-dir", str(out)]) == 0
    key = json.loads((out / "key" / "key.json").read_text())
    assert key.pop("key_sha256") and key["systems"] == ["real", "arm-a", "arm-b"] and key["groups"] == 3
    assert key["stages"] == 2 and key["negation_checkers"] == ["c" * 64]
    assert key["bar"]["strata"] == dict(confirmed=.2, rejected=0., non_candidate=.8)
    assert key["bar"]["confirmed"] == 1 and key["bar"]["eligible"] == 5
    assert key["draw_sha256"] == draw_sha and key["negation_sheet_sha256"] == checked
    assert key["arms"]["arm-a"]["arm"] == "arm-a" and key["timing"] == "reference-length"
    assert key["arms"]["arm-a"]["form"] == "core_crossfade" and "reference-length" in key["diagnostic"]
    assert set(key["video_bytes_per_frame"]) == {"real", "arm-a", "arm-b"}
    stored = json.loads(store.get("sets/d20.json"))
    d20_texts = {r["text"].strip() for r in stored["records"]}
    assert {t for i in key["items"] for t in i["options"]} <= d20_texts
    for i in key["items"]:
        assert i["truth"]["negated"] == ("yes" if i["sign"] == candidate else "no")
        assert i["truth"]["stratum"] == ("confirmed" if i["sign"] == candidate else "non_candidate")
        assert (out / "packet" / f"group-{i['group']}" / f"{i['item_id']}.mp4").stat().st_size > 0 and i["frames"] > 0
        assert i["renderer"] == "skeleton" and i["seconds"] == i["reference_seconds"] and len(i["item_id"]) == 32
        if i["system"] == "real":
            assert i["state"]["sha256"] in {r["state_sha256"] for r in stored["records"]}
        else:
            load_state(store, i["state"]["key"], i["state"]["sha256"])
            assert i["state"]["form"] == "core_crossfade" and i["state"]["words"]
    options = {t for i in key["items"] for t in i["options"]}
    for g in range(3):
        sheet = (out / "packet" / f"group-{g}" / "sheet.csv").read_text()
        for word in ("arm-a", "arm-b", "real", "correct", "core_crossfade", "option", "choice", *options):
            assert word not in sheet                                   # stage 1: no options anywhere
        assert sheet == write_stage1_sheet(key, g)
        assert (out / "packet" / f"group-{g}" / TWO_STAGE.name).exists()
        with zipfile.ZipFile(out / "packet" / f"group-{g}.zip") as z:
            names = z.namelist()
            assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)}
            for name in names:
                if name.endswith((".csv", ".md")):
                    text = z.read(name).decode()
                    assert not any(t in text for t in options)
        assert f"group-{g}/sheet.csv" in names and sum(n.endswith(".mp4") for n in names) == 3
        assert not any("key" in n for n in names)
    # stage 1 comes back, stage 2 goes out
    items = {i["item_id"]: i for i in key["items"]}
    roster = {f"{g}{k}".ljust(64, "a"): g for g in range(3) for k in range(2)}
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(roster))
    receipts = tmp_path / "receipts.json"
    key_path = out / "key" / "key.json"
    firsts, seconds = [], []
    for reviewer, g in roster.items():
        filled = list(csv.DictReader(io.StringIO(write_stage1_sheet(key, g))))
        for r in filled:
            item = items[r["item_id"]]
            r.update(viewable="yes", translation=f"{reviewer[:2]} {item['position']}", sentence_type="statement",
                     negated=item["truth"]["negated"], topic="no", naturalness="3", reviewer_sha256=reviewer)
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=STAGE1_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(filled)
        path = tmp_path / f"stage1-{reviewer[:2]}.csv"
        path.write_text(buffer.getvalue())
        release = ["how2sign-stage2", "--key", str(key_path), "--roster", str(roster_path), "--stage1-sheet", str(path),
                   "--receipts", str(receipts), "--output-dir", str(tmp_path / "stage2")]
        assert main(release) == 0
        firsts += ["--sheet", str(path)]
        sent = tmp_path / "stage2" / f"stage2-{reviewer[:12]}.csv"
        back = list(csv.DictReader(io.StringIO(sent.read_text())))
        assert back[0]["stage1_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert back[0]["reviewer_sha256"] == reviewer and set(items[back[0]["item_id"]]["options"]) <= set(back[0].values())
        for r in back:
            r["choice"] = items[r["item_id"]]["correct"]
        returned = tmp_path / f"stage2-back-{reviewer[:2]}.csv"
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=STAGE2_COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(back)
        returned.write_text(buffer.getvalue())
        seconds += ["--stage2-sheet", str(returned)]
    assert json.loads(receipts.read_text())[next(iter(roster))]
    # stage 1 is final once released; the checker and strangers are refused
    last = path.read_text()
    path.write_text(last.replace(",3,", ",4,", 1))
    with pytest.raises(SystemExit, match="stage 1 is final"):
        main(release)
    path.write_text(last)
    stranger = tmp_path / "stranger.json"
    stranger.write_text(json.dumps({"f" * 64: 0}))
    with pytest.raises(SystemExit, match="not on the roster"):
        main([*release[:3], "--roster", str(stranger), *release[5:]])
    checker_roster = tmp_path / "checker.json"
    checker_roster.write_text(json.dumps({"c" * 64: 0}))
    checker_sheet = tmp_path / "checker-stage1.csv"
    checker_sheet.write_text(last.replace(next(iter(reversed(list(roster)))), "c" * 64))
    with pytest.raises(SystemExit, match="checker"):
        main(["how2sign-stage2", "--key", str(key_path), "--roster", str(checker_roster), "--stage1-sheet",
              str(checker_sheet), "--receipts", str(receipts), "--output-dir", str(tmp_path / "stage2")])
    # scored
    common = ["--key", str(key_path), *firsts, "--reference", "real", "--stage1-receipts", str(receipts)]
    assert sh.main(common + seconds + ["--output", str(tmp_path / "plain.json")]) == 0
    assert "bar" not in json.loads((tmp_path / "plain.json").read_text())
    with pytest.raises(SystemExit, match="roster"):
        sh.main(common + seconds + ["--bar", "--output", str(tmp_path / "x.json")])
    bar = ["--bar", "--roster", str(roster_path), "--bootstrap", "50"]
    assert sh.main(common + seconds + bar + ["--output", str(tmp_path / "bar.json")]) == 0
    report = json.loads((tmp_path / "bar.json").read_text())
    read = report["bar"]["systems"]["arm-a"]
    assert report["bar"]["may_release"] is False and report["systems"]["arm-a"]["sentence_type"]["n"] == 0
    assert read["conditions"]["comprehension_wilson_lower"]["result"] == "fail"      # 6 ratings cannot clear 70%
    assert read["comprehension"]["unweighted"]["rate"] == 1 and read["negation"]["recall"]["rate"] == 1
    # no read: without stage 2 the CLI says why and exits nonzero
    assert sh.main(common + bar + ["--output", str(tmp_path / "refused.json")]) == 2
    refused = json.loads((tmp_path / "refused.json").read_text())["bar"]
    assert "coverage" in refused["refused"] and refused["short"]
    # write-once on the store
    (tmp_path / "again").mkdir()
    prefix = ["--output-prefix", "published/x"]
    assert main(packet + ["--negation-sheet-sha256", checked, "--output-dir", str(tmp_path / "again" / "a"), *prefix]) == 0
    with pytest.raises(SystemExit, match="written once"):
        main(packet + ["--negation-sheet-sha256", checked, "--output-dir", str(tmp_path / "again" / "b"), *prefix])
    # the states are checked against their index when loaded
    index_key = systems[1].split(":")[1].split("@")[0]
    index = json.loads(store.get(index_key))
    for row in index["states"]:
        row["frames"] += 1
        row["reference_frames"] += 1
    store.put("x/lying/index.json", json.dumps(index).encode())
    lying = f"arm-a=composition:x/lying/index.json@{hashlib.sha256(store.get('x/lying/index.json')).hexdigest()}"
    shifted = [lying if a == systems[1] else a for a in packet]
    with pytest.raises(SystemExit, match="the index says"):
        main(shifted + ["--negation-sheet-sha256", checked, "--output-dir", str(tmp_path / "again" / "c")])
