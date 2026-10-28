"""Mixed text -> motion manifests: records kept and tagged by corpus, collisions refused, licences carried."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_mixed_text_manifest import mix, parse_input  # noqa: E402

SHA = "a" * 64


def manifest(corpus, signs, licence=None):
    return dict(corpus_id=corpus, license=licence,
                records=[dict(sign=s, text=f"text {s}", partition=p,
                              continuous=dict(state_key=f"{s}.npz", state_sha256=SHA, available_channels=["body"]))
                         for s, p in signs])


def test_records_are_kept_tagged_and_partitioned_as_given():
    m = mix([("1" * 64, manifest("en", [("a", "train"), ("b", "val")])),
             ("2" * 64, manifest("plan", [("c", "train"), ("d", "test")], "research only"))], corpus_id="mixed")
    assert {r["sign"]: (r["corpus"], r["partition"]) for r in m["records"]} == {
        "a": ("en", "train"), "b": ("en", "val"), "c": ("plan", "train"), "d": ("plan", "test")}
    assert m["inputs"] == {"en": "1" * 64, "plan": "2" * 64}
    assert m["license"] == "plan: research only" and m["split"]["counts"]["plan/test"] == 1


def test_collisions_are_refused():
    with pytest.raises(ValueError, match="occurs in two inputs"):
        mix([("1" * 64, manifest("en", [("a", "train")])), ("2" * 64, manifest("plan", [("a", "train")]))],
            corpus_id="m")
    with pytest.raises(ValueError, match="share corpus_id"):
        mix([("1" * 64, manifest("en", [("a", "train")])), ("2" * 64, manifest("en", [("b", "train")]))],
            corpus_id="m")
    with pytest.raises(ValueError):
        parse_input("key@short")


# ---- extending a mix (round 14: round10-mixed-v1 + semlex-text-v1) ------------------------------------------------

def _built(corpus_id, records, **extra):
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest
    m = build_corpus_manifest(corpus_id=corpus_id, inputs={"x": "0" * 64}, records=records, split=dict(
        kind="continuous", source="each input's own signer-disjoint partitions, unchanged",
        counts={"train": 0, "val": 0, "test": 0}))
    m.update(extra)
    return m


def _rec(sign, corpus, partition, text, **extra):
    return dict(sign=sign, corpus=corpus, partition=partition, text=text, performer_sha256=None,
                continuous=dict(status="converted", state_key=f"{sign}.npz", state_sha256=SHA,
                                available_channels=["body", "right_hand"], **extra))


def _base():
    return _built("round10-mixed-v1", [
        _rec("--h2s", "how2sign-two-view-text-v1", "train", "A sentence."),
        _rec("Aval", "how2sign-two-view-text-v1", "val", "Validation."),
        _rec("rit-1", "rit-text-r6", "train", "steep", window_seconds=[0., 1.5], seconds=1.5417),
        _rec("zz-hold", "how2sign-two-view-text-v1", "heldout_official", "Held out."),
        _rec("zzval", "how2sign-two-view-text-v1", "val", "Last.")],
        license="how2sign: CC BY-NC 4.0", trim=dict(rule="x", corpora=["rit-text-r6"]))


def _semlex():
    m = _built("semlex-text-v1", [_rec("semlex-a", "semlex-text-v1", "train", "cat", window_seconds=[.1, .9]),
                                  _rec("semlex-b", "semlex-text-v1", "val", "ice", window_seconds=[0., .8])],
               license="semlex-text-v1: CC BY-NC-SA 4.0", provenance=dict(source="video_extraction"))
    m["split"] = dict(m["split"], signer_disjoint=False)
    return m


def test_extend_keeps_every_base_record_byte_identical():
    from build_mixed_text_manifest import extend, val_prefix_unchanged
    from signtranslator.data_engineering.corpus_manifest import canonical_json, manifest_sha256
    base = _base()
    out = extend(manifest_sha256(base), base, [("3" * 64, _semlex())], corpus_id="round14-mixed-v1")
    by = {r["sign"]: r for r in out["records"]}
    for r in base["records"]:
        assert canonical_json(by[r["sign"]]) == canonical_json(r)
    assert by["semlex-a"]["corpus"] == "semlex-text-v1" and by["semlex-b"]["partition"] == "val"
    assert len(out["records"]) == len(base["records"]) + 2
    assert out["inputs"] == {"round10-mixed-v1": manifest_sha256(base), "semlex-text-v1": "3" * 64}
    assert out["split"]["counts"] == {"train": 3, "val": 3, "test": 0}          # the base's partitions, recounted
    assert out["split"]["signer_disjoint_by_corpus"] == {"semlex-text-v1": False}
    assert "signer_disjoint false" in out["split"]["source"]
    assert out["license"] == "how2sign: CC BY-NC 4.0; semlex-text-v1: CC BY-NC-SA 4.0"   # no doubled prefix
    assert out["trim"] == base["trim"] and out["extends"]["base_sha256"] == manifest_sha256(base)
    assert out["extends"]["added"]["semlex-text-v1"]["partitions"] == {"train": 1, "val": 1}
    assert out["corpus_id"] == "round14-mixed-v1" and out["records"] == sorted(out["records"], key=lambda r: r["sign"])
    # the trainer's first validation records ("Aval" sorts before "semlex-b"; "zzval" after it)
    assert val_prefix_unchanged(base, out, 1) and not val_prefix_unchanged(base, out, 2)
    # the trainer's max_seconds filter: an over-long base val record is not read, so it must not count
    long = _built("round10-mixed-v1", [_rec("Aval", "how2sign-two-view-text-v1", "val", "x", seconds=30.),
                                       _rec("zzval", "how2sign-two-view-text-v1", "val", "y", seconds=3.)])
    longer = extend("1" * 64, long, [("3" * 64, _semlex())], corpus_id="m")
    assert not val_prefix_unchanged(long, longer, 1) and val_prefix_unchanged(long, longer, 1, None)


def test_extend_refuses_collisions():
    from build_mixed_text_manifest import extend
    base = _base()
    clash = _semlex()
    clash["records"][0]["sign"] = "rit-1"
    with pytest.raises(ValueError, match="occurs in two inputs"):
        extend("1" * 64, base, [("3" * 64, clash)], corpus_id="m")
    same = _semlex()
    same["corpus_id"] = "rit-text-r6"
    with pytest.raises(ValueError, match="already in the mix"):
        extend("1" * 64, base, [("3" * 64, same)], corpus_id="m")
    tagged = _semlex()
    tagged["records"][0]["corpus"] = "other"
    with pytest.raises(ValueError, match="tagged"):
        extend("1" * 64, base, [("3" * 64, tagged)], corpus_id="m")


def test_extend_cli_writes_once(tmp_path, capsys):
    import json
    from build_mixed_text_manifest import main
    from signtranslator.data_engineering.corpus_manifest import canonical_json, load_manifest, manifest_sha256
    base, semlex = _base(), _semlex()
    for name, m in (("base.json", base), ("semlex.json", semlex)):
        (tmp_path / name).write_bytes(canonical_json(m))                     # store form: no trailing newline
    argv = ["--store", f"dir:{tmp_path}", "--base", f"base.json@{manifest_sha256(base)}",
            "--input", f"semlex.json@{manifest_sha256(semlex)}", "--corpus-id", "round14-mixed-v1",
            "--manifest-key", "out/manifest.json", "--check-val-prefix", "1", "--check-val-prefix", "2"]
    with pytest.raises(SystemExit, match=r"first \['2'\] validation records differ"):     # nothing is written
        main(argv)
    assert not (tmp_path / "out/manifest.json").exists()
    argv = argv[:-2]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["base_records_identical"] and printed["val_prefix_unchanged"] == {"1": True}
    assert printed["by_corpus"]["semlex-text-v1/train"] == 1 and printed["base_records"] == 5
    stored = (tmp_path / "out/manifest.json").read_bytes()
    (tmp_path / "check.json").write_bytes(stored + b"\n")
    assert load_manifest(tmp_path / "check.json", expected_sha256=printed["manifest_sha256"])["corpus_id"] == \
        "round14-mixed-v1"
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(argv)
    with pytest.raises(ValueError, match="hash mismatch"):
        main(argv[:3] + [f"base.json@{'0' * 64}"] + argv[4:-4] + ["--manifest-key", "out/other.json"])
