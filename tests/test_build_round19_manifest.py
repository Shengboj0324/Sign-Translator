"""Round 19 mix builder on a synthetic store: round14b's YouTube component rebuilt from the current reports with the
snapshot's own filters, round14b's val/test texts added to the leak filter, every other record byte-identical,
validation prefix and pilot sample unchanged, write-once, and the refusals."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import build_round19_manifest as b19  # noqa: E402
import build_youtube_asl_manifest as yt  # noqa: E402

from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest  # noqa: E402

REPORTS = "yt/videos"
FILTERS = dict(allow_auto=False, channel_cap=1., corpus_id="youtube-asl-v1", max_abs_yaw=30., max_chars=300,
               max_other_person=.1, max_seconds=20., max_words_per_second=4., min_face_coverage=.9,
               min_hand_coverage=.5, min_seconds=1., min_words=2, min_words_per_second=.5, monitor_share=.01)
H2S_VAL = ["I like apples very much", "She reads every night", "The bus was late again"]


def segment(video, cue, text, **q):
    sample = f"{video}-{cue:04d}"
    return dict(dict(sample_id=sample, cue=cue, text=text, start=3. * cue, end=3. * cue + 2., hand_coverage=1.,
                     face_coverage=1., median_abs_yaw_deg=1., other_person_fraction=0., seconds=2.,
                     state_key=f"yt/states/{sample}.npz", state_sha256=hashlib.sha256(sample.encode()).hexdigest(),
                     available_channels=["body", "left_hand", "right_hand"]), **q)


def report(video, channel, segments):
    return dict(video_id=video, status="converted", caption_kind="manual", caption_seconds=None,
                meta=dict(channel_id=channel, duration=600, height=720, fps=30, upload_date="20200101"),
                segments=segments)


OLD = [report("vidA", "chanA", [segment("vidA", 0, "Hello there my good friend"),
                                segment("vidA", 1, "We went to the market today"),
                                segment("vidA", 2, "Hidden hands here", hand_coverage=.1)]),
       report("vidB", "chanB", [segment("vidB", 0, "A second video speaks"),
                                segment("vidB", 1, "She reads every night")])]          # a round14b val text
NEW = [report("vidC", "chanC", [segment("vidC", 0, "A video extracted after the snapshot"),
                                segment("vidC", 1, "With a second good sentence"),
                                segment("vidC", 2, "I like apples very much")])]         # a round14b val text


def record(sign, corpus, partition, text, **extra):
    return dict(sign=f"{corpus}:{sign}", corpus=corpus, partition=partition, text=text, performer_sha256=None,
                continuous=dict(status="converted", state_key=f"states/{sign}.npz", seconds=2.,
                                state_sha256=hashlib.sha256(sign.encode()).hexdigest(),
                                available_channels=["body", "left_hand", "right_hand"]), **extra)


def put(root: Path, key: str, manifest: dict) -> str:
    data = canonical_json(manifest)
    (root / key).parent.mkdir(parents=True, exist_ok=True)
    (root / key).write_bytes(data)
    return f"{key}@{hashlib.sha256(data).hexdigest()}"


def write_reports(root: Path, reports):
    (root / REPORTS).mkdir(parents=True, exist_ok=True)
    for r in reports:
        (root / REPORTS / f"{r['video_id']}.json").write_text(json.dumps(r))


@pytest.fixture()
def world(tmp_path):
    root = tmp_path
    write_reports(root, OLD)
    # the snapshot: built before vidC existed, without round14b's val texts in its leak filter (as manifest-r3b2)
    records, why = yt.build(OLD, args=SimpleNamespace(**FILTERS), corpus_id="youtube-asl-v1")
    snapshot = build_corpus_manifest(corpus_id="youtube-asl-v1", records=records, inputs={"reports": "b" * 64},
                                     split=dict(kind="continuous", counts={}))
    snapshot.update(license="YouTube terms", filters=dict(FILTERS, eval_manifest=[], reasons=why, prefix="yt",
                                                          manifest_key="yt/manifest-r3b2.json", store="r2",
                                                          licence="YouTube terms"))
    ours = put(root, "yt/manifest-r3b2.json", snapshot)
    base_records = [dict(r, corpus="youtube-asl-v1") for r in snapshot["records"]]
    base_records += [record(f"v-{k}", "how2sign-two-view-text-v1", "val", t, signer_component=("c01", "c03", "c05")[k])
                     for k, t in enumerate(H2S_VAL)]
    base_records += [record("t-0", "how2sign-two-view-text-v1", "test", "A test sentence of ours"),
                     record("tr-0", "how2sign-two-view-text-v1", "train", "A training sentence"),
                     record("s-0", "semlex-text-v2", "val", "apple"), record("s-1", "semlex-text-v2", "train", "dog")]
    base = build_corpus_manifest(corpus_id="round14b-mixed-v1", records=base_records, inputs={"x": "a" * 64},
                                 split=dict(kind="continuous", counts={"test": 0, "train": 0, "val": 0}))
    base.update(license="all terms", trim=dict(rule="one-word trims"), extends=dict(base="round10-mixed-v1"))
    spec = put(root, "mix/round14b.json", base)
    write_reports(root, NEW)                                   # extraction went on after the snapshot
    return dict(root=root, base=spec, ours=ours)


def args(w, *extra):
    return ["--store", f"dir:{w['root']}", "--base", w["base"], "--our-manifest", w["ours"], "--reports-prefix",
            REPORTS, "--youtube-key", "yt/manifest-r19.json", "--manifest-key", "mix/round19.json", "--workers", "2",
            *extra]


def summary_of(out: str) -> dict:
    return json.loads(out[:out.index("YOUTUBE_SHA=")])


def load(root: Path, key: str, sha: str) -> dict:
    path = root / (key + ".nl")
    path.write_bytes((root / key).read_bytes() + b"\n")
    return load_manifest(path, expected_sha256=sha)


def test_the_youtube_component_is_rebuilt_and_everything_else_is_unchanged(world, capsys):
    root = world["root"]
    assert b19.main(args(world)) == 0
    out = capsys.readouterr().out
    s = summary_of(out)
    assert s["manifest_sha256"] == out.split("MANIFEST_SHA=")[1].split()[0]
    mix = load(root, "mix/round19.json", s["manifest_sha256"])
    base = load(root, "mix/round14b.json", world["base"].split("@")[1])
    ytn = load(root, "yt/manifest-r19.json", s["youtube_sha256"])
    signs = {r["sign"] for r in mix["records"] if r["corpus"] == "youtube-asl-v1"}
    assert signs == {"youtube-asl-v1:vidA-0000", "youtube-asl-v1:vidA-0001", "youtube-asl-v1:vidB-0000",
                     "youtube-asl-v1:vidC-0000", "youtube-asl-v1:vidC-0001"}
    assert s["youtube_delta"] == dict(kept=3, kept_but_changed=0, dropped=1, dropped_hours=round(2 / 3600, 2),
                                      dropped_by_reason={"evaluation_leak": 1}, new=2, new_hours=round(4 / 3600, 2),
                                      new_partitions={"train": 2})
    assert s["youtube_reasons"]["evaluation_leak"] == 2 and s["youtube_reasons"]["hands_not_visible"] == 1
    assert ytn["filters"]["channel_cap"] == 1. and ytn["filters"]["reasons"] == s["youtube_reasons"]
    by = {r["sign"]: r for r in mix["records"]}
    assert all(canonical_json(by[r["sign"]]) == canonical_json(r) for r in base["records"] if r["corpus"] != "youtube-asl-v1")
    assert mix["trim"] == base["trim"] and mix["extends"] == base["extends"] and mix["license"] == base["license"]
    assert mix["replaces"]["new_sha256"] == s["youtube_sha256"] and mix["corpus_id"] == "round19-mixed-v1"
    assert s["val_prefix_unchanged"] == {"1536": True, "2048": True} and s["pilot_sample_unchanged"]
    assert s["delta"] == {"youtube-asl-v1/train": dict(records=1, hours=round(2 / 3600, 2))}
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        b19.main(args(world))


def test_a_dry_run_writes_nothing_and_a_tampered_base_is_refused(world):
    assert b19.main(args(world, "--dry-run")) == 0
    assert not (world["root"] / "mix/round19.json").exists() and not (world["root"] / "yt/manifest-r19.json").exists()
    m = json.loads((world["root"] / "mix/round14b.json").read_text())
    for r in m["records"]:
        if r["sign"] == "youtube-asl-v1:vidA-0000":
            r["text"] = "Hello there my good friends"
    spec = put(world["root"], "mix/tampered.json", m)
    with pytest.raises(SystemExit, match="not manifest-r3b2's"):
        b19.main(args(world, "--dry-run")[:2] + ["--base", spec] + args(world, "--dry-run")[4:])


def test_replace_refuses_a_changed_non_youtube_record_and_youtube_outside_train_or_monitor():
    base = dict(records=[record("a", "youtube-asl-v1", "train", "x y"), record("h", "how2sign-two-view-text-v1", "val", "v w")],
                split=dict(counts={"train": 1, "val": 1}))
    for r in base["records"]:
        r["usable"] = {"continuous": ["body", "left_hand", "right_hand"], "face": [], "skeleton": []}
    yt_ok = dict(records=[strip_usable(record("b", "youtube-asl-v1", "train", "p q"))])
    mix = b19.replace_youtube("a" * 64, base, "b" * 64, yt_ok, "c" * 64)
    assert {r["sign"] for r in mix["records"]} == {"youtube-asl-v1:b", "how2sign-two-view-text-v1:h"}
    with pytest.raises(SystemExit, match="outside train/monitor"):
        b19.replace_youtube("a" * 64, base, "b" * 64, dict(records=[record("c", "youtube-asl-v1", "val", "p q")]), "c" * 64)


def strip_usable(r):
    return {k: v for k, v in r.items() if k != "usable"}
