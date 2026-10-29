"""Recogniser v3 end to end on a directory store: decoded copies (``build_recon_states.py``: the ``recon_full`` form of
``tokenizer_domain_read.py``, sharded, resumable, write-once, merged), training with several corpora, held-out and
excluded signers, decoded copies and validity dropout (``train_sign_recognizer.py``), and the G1-G7 gate report
(``validate_sign_recognizer.py --gates``); the v2 command line keeps v2's run.json."""
from dataclasses import replace
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import build_recon_states as brs  # noqa: E402
import train_sign_recognizer as trainer  # noqa: E402
import validate_sign_recognizer as vsr  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig  # noqa: E402
from tests.test_generation_motion_rvq import motion_config  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402

V2_ARG_KEYS = {"augment", "batch_size", "cache_bytes", "checkpoint_every", "d_model", "device", "downsample", "dropout",
               "ema_decay", "extra_manifest_key", "extra_manifest_sha256", "feedforward", "fps", "gloss_map_key", "heads",
               "label_smoothing", "layers", "lr", "manifest_key", "manifest_sha256", "max_seconds", "min_hand_fraction",
               "patience", "run", "seed", "steps", "store", "trained_channels", "val_clips", "val_every", "warmup",
               "weight_decay", "workers"}                     # sign-recognizer-v2's run.json args, exactly
TINY = ["--batch-size", "4", "--d-model", "32", "--heads", "4", "--layers", "1", "--feedforward", "64", "--warmup", "2",
        "--val-every", "2", "--checkpoint-every", "2", "--workers", "0", "--fps", "24",
        "--trained-channels", "body,left_hand,right_hand"]
GLOSSES = ("BOOK", "CAT", "DOG")
TWO_VIEW = ("two-view-triangulated-camera1-y-up", "sapiens-two-view-swing-ik-y-up-v1")      # the generator's layout
MONOCULAR = ("camera-aligned-y-up-metres", "mediapipe-tasks-swing-ik-y-up-v1")            # the isolated corpora's


def laid_out(state, layout=None):
    """``state`` with body and hands in ``layout`` (frame, convention); monocular by default, as the corpora."""
    frame, convention = layout or MONOCULAR
    channels = dict(state.channels)
    for name in ("body", "left_hand", "right_hand"):
        channels[name] = replace(channels[name], coordinate_frame=frame, convention=convention)
    return replace(state, channels=channels)


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.seed = 0

    def state(self, key, frames=24, amplitude=.6, layout=None):
        self.seed += 1
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return laid_out(signer_state(self.seed, frames=frames, amplitude=amplitude), layout).save(path)

    def manifest(self, name, corpus_id, records):
        payload = canonical_json(build_corpus_manifest(corpus_id=corpus_id, inputs={"x": "b" * 64}, records=records,
                                                       split={}))
        (self.root / name).write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def put(self, name, payload: bytes):
        (self.root / name).parent.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()


def record(store, sign, label, partition, *, performer, corpus=None, code=None, signer=None, quality=None, window=None,
           frames=24, layout=None, **extra):
    key = f"states/{sign}.npz"
    sha = store.state(key, frames, layout=layout)
    r = dict(sign=sign, text=None, partition=partition, performer_sha256=performer,
             gloss=dict(status="x", label=label, **({"asl_lex_code": code} if code else {})),
             quality=quality or dict(frames=frames, left_hand_frames=0, right_hand_frames=frames),
             continuous=dict(status="converted", state_key=key, state_sha256=sha, seconds=frames / 24.,
                             available_channels=["body", "left_hand", "right_hand"],
                             **({"window_seconds": window} if window else {})), **extra)
    if corpus:
        r["corpus"] = corpus
    if signer is not None:
        r["source"] = dict(signer_id=signer)
    return r


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGN_CACHE_DIR", str(tmp_path / "cache"))
    s = Store(tmp_path / "store")
    w = dict(store=s, root=s.root, codes=dict(BOOK="A_1", CAT="A_2", DOG="A_3"))
    cit = []
    for p, signer in enumerate(("1", "2", "3", "4")):
        for g in GLOSSES:
            cit.append(record(s, f"c{signer}{g}", g, "train", performer=signer * 64, code=w["codes"][g]))
    for g in GLOSSES:
        cit.append(record(s, f"cv{g}", g, "val", performer="5" * 64, code=w["codes"][g]))
        cit.append(record(s, f"ct{g}", g, "test", performer="6" * 64, code=w["codes"][g]))
    w["cit"] = s.manifest("cit.json", "asl-citizen-video-estimate-v1", cit)
    sem = []
    for signer in ("11", "12", "13"):
        for g in GLOSSES:
            q = dict(frames=24, hand_fraction=.8)
            sem.append(record(s, f"s{signer}{g}", g.lower(), "train", performer="7" * 64, corpus="semlex-text-v2",
                              code=w["codes"][g], signer=signer, quality=q, window=[0., .8]))
            sem.append(record(s, f"sv{signer}{g}", g.lower(), "val", performer="7" * 64, corpus="semlex-text-v2",
                              code=w["codes"][g], signer=signer, quality=q))
    w["sem"] = s.manifest("sem.json", "semlex-text-v2", sem)
    lvd = [record(s, f"l{g}", g + "-SB", "train", performer="8" * 64) for g in GLOSSES]
    lvd += [record(s, f"lv{g}", g + "-SB", "val", performer="9" * 64) for g in GLOSSES]
    lvd += [record(s, f"lt{g}", g + "-SB", "test", performer="a" * 64) for g in GLOSSES]
    w["lvd"] = s.manifest("lvd.json", "asllvd-dai2-signs-v1", lvd)
    s.put("map.json", json.dumps(dict(mapping={g: dict(method="exact", candidates=[g + "-SB"]) for g in GLOSSES}))
          .encode())
    w["mix"] = s.manifest("mix.json", "mix", [
        record(s, "h2s-test", "x", "test", performer="c" * 64, corpus="how2sign-two-view-text-v1", signer_component="c00"),
        record(s, "h2s-1", "x", "train", performer="c" * 64, corpus="how2sign-two-view-text-v1", signer_component="c01",
               layout=TWO_VIEW)])
    torch.manual_seed(0)
    tokenizer = MotionRVQ(MotionRVQConfig(motion_config(laid_out(signer_state(0), TWO_VIEW)), hidden=48, code_dim=16, codebook=32, stages=3))
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_tokenizer", config=tokenizer.config.to_dict(), model=tokenizer.state_dict(),
                    manifest_sha256="d" * 64), buffer)
    w["tok_key"] = "training-runs/tok/checkpoints/step-00000001.pt"
    w["tok_sha"] = s.put(w["tok_key"], buffer.getvalue())
    w["tokenizer"] = tokenizer.eval()
    w["template"] = f"mix.json@{w['mix']}:how2sign-two-view-text-v1"
    return w


def build(world, *extra, shard=0, shards=1):
    return brs.main(["--store", f"dir:{world['root']}", "--prefix", "recon", "--tokenizer-key", world["tok_key"],
                     "--tokenizer-sha256", world["tok_sha"], "--template", world["template"],
                     "--cache-dir", str(world["root"].parent / "rcache"),
                     "--manifest", f"cit.json@{world['cit']}", "--manifest", f"sem.json@{world['sem']}",
                     "--manifest", f"lvd.json@{world['lvd']}", "--shard", str(shard), "--shards", str(shards), *extra])


def test_the_decoded_copy_is_tokenizer_domain_reads_recon_full(world):
    import tokenizer_domain_read as tdr
    from exposure_diagnostic import make_post
    state = signer_state(5, frames=30)
    post = make_post(None)
    ours = brs.decode_state(state, world["tokenizer"], world["tokenizer"].config.motion, post, text="book")
    forms, _ = tdr.read_forms(world["tokenizer"], world["tokenizer"].config.motion, post, [("book", state)])
    theirs = forms["recon_full"][0]
    for name in state.channels:
        np.testing.assert_array_equal(ours.channels[name].values, theirs.channels[name].values)
        np.testing.assert_array_equal(ours.channels[name].valid, theirs.channels[name].valid)
    trained = world["tokenizer"].config.motion.trained_channels
    assert all(ours.channels[n].valid.all() for n in trained)       # the generated form: every trained channel valid


def test_recon_states_are_sharded_write_once_resumable_and_never_read_test(world, capsys):
    root = world["root"]
    assert build(world, shard=0, shards=2) == 0 and build(world, shard=1, shards=2) == 0
    assert brs.main(["--store", f"dir:{root}", "--prefix", "recon", "--merge", "--shards", "2"]) == 0
    index = json.loads((root / "recon/index.json").read_text())
    signs = {e["sign"] for e in index["entries"]}
    assert not any(s.startswith(("ct", "lt")) for s in signs)       # test partitions never decoded
    assert len(signs) == 12 + 3 + 18 + 6 and index["status_counts"] == {"converted": 39}
    state_files = sorted((root / "recon/states").glob("*.npz"))
    assert len(state_files) == 39
    before = {p.name: p.stat().st_mtime_ns for p in state_files}
    assert build(world, shard=0, shards=2) == 0                      # the shard has its index: skipped
    assert "shard done" in capsys.readouterr().out
    (root / brs.shard_key("recon", 0, 2)).unlink()                   # a crashed shard rerun: states re-hashed only
    assert build(world, shard=0, shards=2) == 0
    part = json.loads((root / brs.shard_key("recon", 0, 2)).read_text())
    assert set(part["status_counts"]) == {"already_present"}
    assert {p.name: p.stat().st_mtime_ns for p in sorted((root / "recon/states").glob("*.npz"))} == before
    with pytest.raises(SystemExit, match="written once"):           # the index is never rewritten
        brs.main(["--store", f"dir:{root}", "--prefix", "recon", "--merge", "--shards", "2"])
    assert json.loads((root / "recon/index.json").read_text()) == index
    with pytest.raises(SystemExit, match="another configuration"):
        _other_config(world)
    with pytest.raises(SystemExit, match="missing"):
        brs.main(["--store", f"dir:{root}", "--prefix", "recon", "--merge", "--shards", "3"])
    entry = next(e for e in index["entries"] if e["sign"] == "s11BOOK")
    sem = json.loads((root / "sem.json").read_text())["records"]
    rec = next(r for r in sem if r["sign"] == "s11BOOK")
    from signtranslator.generation.recognizer_corpora import load_digest
    assert entry["load_digest"] == load_digest(dict(rec, corpus="semlex-text-v2"))   # the window is part of it


def _other_config(world):
    """A shard of the same prefix with another tokenizer: refused (one configuration per prefix)."""
    tok = MotionRVQ(MotionRVQConfig(motion_config(laid_out(signer_state(0), TWO_VIEW)), hidden=48, code_dim=16, codebook=32, stages=2))
    buffer = io.BytesIO()
    torch.save(dict(kind="motion_rvq_tokenizer", config=tok.config.to_dict(), model=tok.state_dict()), buffer)
    sha = world["store"].put("training-runs/tok2/checkpoints/step-00000001.pt", buffer.getvalue())
    return brs.main(["--store", f"dir:{world['root']}", "--prefix", "recon", "--tokenizer-key",
                     "training-runs/tok2/checkpoints/step-00000001.pt", "--tokenizer-sha256", sha,
                     "--template", world["template"], "--cache-dir", str(world["root"].parent / "rcache"),
                     "--manifest", f"cit.json@{world['cit']}"])


def test_worker_processes_write_the_same_copies(world):
    root = world["root"]
    assert build(world) == 0
    assert brs.main(["--store", f"dir:{root}", "--prefix", "recon", "--merge", "--shards", "1"]) == 0
    one = {e["sign"]: e["recon_sha256"] for e in json.loads((root / "recon/index.json").read_text())["entries"]}
    args = ["--store", f"dir:{root}", "--prefix", "recon2", "--tokenizer-key", world["tok_key"], "--tokenizer-sha256",
            world["tok_sha"], "--template", world["template"], "--cache-dir", str(root.parent / "rcache2"),
            "--manifest", f"cit.json@{world['cit']}", "--manifest", f"sem.json@{world['sem']}",
            "--manifest", f"lvd.json@{world['lvd']}", "--workers", "2"]
    assert brs.main(args) == 0
    assert brs.main(["--store", f"dir:{root}", "--prefix", "recon2", "--merge", "--shards", "1"]) == 0
    two = {e["sign"]: e["recon_sha256"] for e in json.loads((root / "recon2/index.json").read_text())["entries"]}
    assert one == two


def test_recon_builder_refuses_test_partitions_and_how2sign_signer_nine(world):
    with pytest.raises(SystemExit):
        brs.jobs_of([], partitions=("train", "test"))
    bad = dict(corpus_id="how2sign-x", records=[dict(sign="h", partition="train", corpus="how2sign-x",
                                                     signer_component="c00", continuous=dict(state_sha256="a" * 64))])
    with pytest.raises(RuntimeError):
        brs.jobs_of([bad])


def v2_run(world):
    return trainer.main(["--store", f"dir:{world['root']}", "--manifest-key", "cit.json", "--manifest-sha256", world["cit"],
                         "--run", "v2", "--steps", "4", "--augment", "--extra-manifest-key", "lvd.json",
                         "--extra-manifest-sha256", world["lvd"], "--gloss-map-key", "map.json", *TINY])


def test_the_v2_command_line_keeps_v2s_run_json(world):
    assert v2_run(world) == 0
    run = json.loads((world["root"] / "training-runs/v2/run.json").read_text())
    assert set(run["args"]) == V2_ARG_KEYS
    assert "recognizer_v3" not in run and "data_sha256" not in run and run["labels"] == list(GLOSSES)
    best = json.loads((world["root"] / "training-runs/v2/best.json").read_text())
    assert "val_top1" in best and "cross/top1" in best["validation"]
    ck = torch.load(io.BytesIO((world["root"] / best["checkpoint_key"]).read_bytes()), weights_only=False)
    assert "data_sha256" not in ck


def v3_args(world, index_sha, run="v3", *extra):
    return ["--store", f"dir:{world['root']}", "--manifest-key", "cit.json", "--manifest-sha256", world["cit"],
            "--run", run, "--steps", "4", "--augment", "--gloss-map-key", "map.json",
            "--corpus", f"sem.json@{world['sem']}:asllex", "--corpus", f"lvd.json@{world['lvd']}:signbank:2",
            "--holdout-signers", "semlex-text-v2:11", "--exclude-signers", "asl-citizen-video-estimate-v1:" + "4" * 16,
            "--labels-from-run", "v2", "--conform-template", world["template"], "--recon-prefix", "recon",
            "--recon-index-sha256", index_sha, "--recon-p", ".5", "--validity-dropout", ".5", *TINY, *extra]


def test_v3_trains_on_every_corpus_selects_on_real_and_decoded_and_validates_g1_to_g7(world, tmp_path):
    root = world["root"]
    assert v2_run(world) == 0
    assert build(world) == 0
    assert brs.main(["--store", f"dir:{root}", "--prefix", "recon", "--merge", "--shards", "1"]) == 0
    index_sha = hashlib.sha256((root / "recon/index.json").read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="recon-index-sha256"):
        trainer.main(v3_args(world, "0" * 64, "bad"))
    with pytest.raises(SystemExit, match="--extra-manifest-key"):
        trainer.main(v3_args(world, index_sha, "bad2", "--extra-manifest-key", "lvd.json"))
    assert trainer.main(v3_args(world, index_sha)) == 0
    run = json.loads((root / "training-runs/v3/run.json").read_text())
    assert run["recognizer_v3"] and run["labels"] == list(GLOSSES)
    parts = run["data"]["partitions"]
    assert parts["holdout:semlex-text-v2"] == 6 and parts["val:semlex-text-v2"] == 6
    assert parts["train"] == 9 + 6 + 3                                # citizen minus signer 4, Sem-Lex 12 and 13, ASLLVD
    assert set(run["val_records"]) == {"val:asl-citizen-video-estimate-v1", "val:semlex-text-v2",
                                       "val:asllvd-dai2-signs-v1", "holdout:semlex-text-v2"}
    assert run["data"]["recon"]["train_coverage"] == 1.
    best = json.loads((root / "training-runs/v3/best.json").read_text())
    v = best["validation"]
    expected = np.mean([v[f"val:{c}/{f}/top1"] for c in ("asl-citizen-video-estimate-v1", "semlex-text-v2",
                                                          "asllvd-dai2-signs-v1") for f in ("real", "decoded")])
    assert best["select_score"] == pytest.approx(expected) and "holdout:semlex-text-v2/real/top1" in v
    ck = torch.load(io.BytesIO((root / best["checkpoint_key"]).read_bytes()), weights_only=False)
    assert ck["data_sha256"] == run["data_sha256"]
    assert trainer.main(v3_args(world, index_sha)) == 0             # resumes: same config, data and checkpoints
    # the gate report
    s = world["store"]
    s.put("split.json", json.dumps(dict(covered=["BOOK", "CAT"], uncovered=["DOG"])).encode())
    split_sha = hashlib.sha256((root / "split.json").read_bytes()).hexdigest()
    utterance = record(s, "u1", "x", "train", performer="e" * 64, frames=40,
                       source=dict(utterance_video="utt1.mov"))
    cm_sha = s.manifest("cm.json", "asllrp-gloss", [utterance])
    rows = [{"Utterance video filename": "utt1.mov", "Main entry gloss label": g + "-SB", "Hidden": "F",
             "Start frame of the sign video": str(a), "End frame of the sign video": str(a + 9),
             "Start frame of the containing utterance": "100", "End frame of the containing utterance": "139"}
            for g, a in (("BOOK", 100), ("CAT", 115), ("DOG", 128))]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader(), writer.writerows(rows)
    s.put("signs.csv", text.getvalue().encode())
    out = tmp_path / "gates.json"
    common = ["--store", f"dir:{root}", "--run", "v3", "--gates", "--baseline-run", "v2", "--word-split-key",
              "split.json", "--word-split-sha256", split_sha, "--continuous-manifest-key", "cm.json",
              "--continuous-manifest-sha256", cm_sha, "--continuous-csv-key", "signs.csv", "--device", "cpu",
              "--cache-dir", str(tmp_path / "vcache")]
    assert vsr.main(common + ["--output", str(out), "--output-key", "training-runs/v3/gates.json"]) == 0
    report = json.loads(out.read_text())
    assert [g["gate"] for g in report["gates"]] == [f"G{i}" for i in range(1, 8)]
    assert [g["plan"] for g in report["gates"]] == [vsr.GATE_TEXT[f"G{i}"] for i in range(1, 8)]
    assert report["verdict"] in ("v3 usable as the instrument", "v2 stays")
    assert report["all_gates_pass"] == all(g["passed"] for g in report["gates"])
    sets = report["per_set"]
    assert set(sets) == set(run["val_records"])
    for row in sets.values():
        for form in ("real", "v2_real", "mirrored", "blank", "no_hands", "random_code", "decoded", "v2_decoded"):
            assert row[f"{form}_top5"] is not None
        assert row["decoded_clips"] == row["clips"]
    assert report["in_context_asllrp"]["clips"] == 3
    assert report["reliability"]["r5_words"] == 3 and report["reliability"]["uncovered_words"] == 1
    assert report["controls"]["clips"] == sum(r["clips"] for r in sets.values())
    with pytest.raises(SystemExit, match="written once"):
        vsr.main(common + ["--output-key", "training-runs/v3/gates.json"])
    with pytest.raises(SystemExit, match="not a recogniser v3 run"):
        vsr.main(common[:3] + ["v2"] + common[4:])


def quantities(**over):
    q = dict(controls=dict(blank_top1=.0, random_code_top1=.001, chance_top1=.001),
             per_set={"val:a": dict(real_top1=.6, v2_real_top1=.5, real_top5=.8, mirrored_top5=.4),
                      "holdout:semlex-text-v2": dict(real_top1=.5, v2_real_top1=.5, real_top5=.7, mirrored_top5=.3)},
             decoded_words=dict(covered=dict(top5=.75, clips=10), uncovered=dict(top5=.65, clips=5)),
             reliability=dict(r5=120, r5_words=150, uncovered=30, uncovered_words=42), in_context_top1=.08)
    for k, v in over.items():
        q[k] = v
    return q


def test_each_gate_reads_its_declared_quantity_and_threshold():
    t = dict(vsr.GATE_DEFAULTS)
    assert all(g["passed"] for g in vsr.evaluate_gates(quantities(), t))
    fails = lambda q: [g["gate"] for g in vsr.evaluate_gates(q, t) if not g["passed"]]  # noqa: E731
    assert fails(quantities(controls=dict(blank_top1=.03, random_code_top1=0., chance_top1=.001))) == ["G1"]
    assert fails(quantities(controls=dict(blank_top1=0., random_code_top1=.021, chance_top1=.001))) == ["G1"]
    worse = quantities()
    worse["per_set"]["val:a"]["real_top1"] = .49
    assert fails(worse) == ["G2"]
    low = quantities()
    low["per_set"]["holdout:semlex-text-v2"]["real_top5"] = .59
    assert fails(low) == ["G3"]
    missing = quantities()
    del missing["per_set"]["holdout:semlex-text-v2"]
    assert fails(missing) == ["G3"]
    assert fails(quantities(decoded_words=dict(covered=dict(top5=.69, clips=1), uncovered=dict(top5=.9, clips=1)))) == ["G4"]
    assert fails(quantities(decoded_words=dict(covered=dict(top5=.9, clips=1), uncovered=dict(top5=.59, clips=1)))) == ["G4"]
    tie = quantities()
    tie["per_set"]["val:a"]["mirrored_top5"] = .8                     # stored must be strictly above mirrored
    assert fails(tie) == ["G5"]
    assert fails(quantities(reliability=dict(r5=99, r5_words=150, uncovered=30, uncovered_words=42))) == ["G6"]
    assert fails(quantities(reliability=dict(r5=100, r5_words=150, uncovered=24, uncovered_words=42))) == ["G6"]
    assert fails(quantities(in_context_top1=.069)) == ["G7"] and fails(quantities(in_context_top1=None)) == ["G7"]
    edge = quantities(reliability=dict(r5=100, r5_words=150, uncovered=25, uncovered_words=42), in_context_top1=.07)
    assert fails(edge) == []                                           # the declared bounds are inclusive
    assert "| G1 |" in vsr.gate_table(vsr.evaluate_gates(quantities(), t)) and "PASS" in vsr.gate_table(
        vsr.evaluate_gates(quantities(), t))
    assert vsr.GATE_DEFAULTS["g3_min"] == .6 and vsr.GATE_DEFAULTS["g4_covered_min"] == .7
    assert vsr.GATE_DEFAULTS["g4_uncovered_min"] == .6 and vsr.GATE_DEFAULTS["g6_min_r5"] == 100
    assert vsr.GATE_DEFAULTS["g6_min_uncovered"] == 25 and vsr.GATE_DEFAULTS["g7_min_top1"] == .07


def test_ranks_and_reliable_classes():
    logits = torch.tensor([[3., 1, 2], [1., 3, 2], [3., 3, 1]])
    assert vsr.ranks_from_logits(logits, [0, 2, 1]).tolist() == [0, 1, 0]   # a tie counts for the clip
    labels = [0, 0, 0, 1, 1, 1, 1, 2, 2]
    ranks = [0, 0, 0, 0, 0, 0, 3, 0, 0]
    assert vsr.reliable_classes(labels, ranks, threshold=.8, min_clips=3) == {0}   # 1: 3/4 < 0.8; 2: two clips
    assert vsr.reliable_classes(labels, ranks, threshold=.75, min_clips=2) == {0, 1, 2}
