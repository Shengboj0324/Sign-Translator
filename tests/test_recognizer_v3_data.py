"""Recogniser v3 data: corpus specs, the four label mappings, partitions with held-out and excluded signers, the
test-partition and How2Sign signer-9 guard, decoded-copy attachment, validity dropout, and that augmentation without
the new option is recogniser v2's augmentation draw for draw."""
from dataclasses import replace
import random

import numpy as np
import pytest

from signtranslator.data_engineering.corpus_manifest import ContentCache
from signtranslator.generation import recognizer_corpora as rc
from signtranslator.generation.sign_recognizer import (GlossMotionDataset, VALIDITY_GROUPS, VALIDITY_GROUPS_JOINTS,
                                                       augment_sign, invalidate_joints)
from tests.test_generation_signing_space import signer_state

SHA = "a" * 64


def rec(sign, label, partition="train", signer=None, performer=None, code=None, corpus=None, window=None, **extra):
    r = dict(sign=sign, partition=partition, performer_sha256=performer or ("b" * 64),
             gloss=dict(label=label, **({"asl_lex_code": code} if code else {})),
             continuous=dict(status="converted", state_key=f"states/{sign}.npz", state_sha256=SHA,
                             available_channels=["body", "left_hand", "right_hand"], seconds=1.,
                             **({"window_seconds": window} if window else {})), **extra)
    if signer is not None:
        r["source"] = dict(signer_id=signer)
    if corpus:
        r["corpus"] = corpus
    return r


def test_corpus_specs_parse_strictly():
    s = rc.parse_corpus_spec(f"a/b/manifest.json@{SHA}:signbank:3")
    assert (s.key, s.sha256, s.mapping, s.weight) == ("a/b/manifest.json", SHA, "signbank", 3.)
    assert rc.parse_corpus_spec(f"k@{SHA}:upper").weight == 1.
    for bad in (f"k{SHA}:label", f"k@{SHA}", f"k@{SHA}:nope", f"k@abc:label", f"k@{SHA}:label:0",
                f"k@{SHA}:label:-1", f"k@{SHA}:label:1:2", f"@{SHA}:label", f"k@{SHA}:label:inf"):
        with pytest.raises(ValueError):
            rc.parse_corpus_spec(bad)
    assert rc.parse_signer_specs(["semlex:11,23", "semlex:4", "cit:abc"]) == {"cit": ("abc",),
                                                                             "semlex": ("11", "23", "4")}
    for bad in (["semlex"], ["semlex:"], ["semlex:1,,2"], [":1"], ["a:b:c"]):
        with pytest.raises(ValueError):
            rc.parse_signer_specs(bad)


def test_the_four_mappings_are_exact_and_refuse_ambiguity():
    labels = {"BOOK", "CAT", "DEAF1", "DEAF2", "HOUSE"}
    primary = dict(records=[rec("p1", "BOOK", code="A_01"), rec("p2", "CAT", code="A_02"), rec("p3", "DEAF1", code="A_03"),
                            rec("p4", "DEAF2", code="A_03"), rec("p5", "HOUSE", code="NA"),
                            rec("p6", "ZEBRA", code="A_09")])           # ZEBRA is not a label: its code maps nothing
    codes = rc.code_labels(primary, labels)
    asllex = rc.label_mapper("asllex", labels, codes=codes)
    assert asllex(rec("s", "book", code="A_01")) == "BOOK"
    assert asllex(rec("s", "deaf", code="A_03")) is None                # two labels share the code
    assert asllex(rec("s", "house", code="NA")) is None and asllex(rec("s", "x")) is None
    assert asllex(rec("s", "zebra", code="A_09")) is None
    gmap = dict(mapping={"BOOK": dict(method="exact", candidates=["BOOK", "BOOK-2"]),
                         "CAT": dict(method="letters", candidates=["CAT"]),
                         "DEAF1": dict(method="exact", candidates=["DEAF"]), "DEAF2": dict(method="exact", candidates=["DEAF"])})
    sb = rc.label_mapper("signbank", labels, signbank=rc.signbank_targets(gmap, labels))
    assert sb(rec("a", "BOOK-2")) == "BOOK" and sb(rec("a", "BOOK")) == "BOOK"
    assert sb(rec("a", "CAT")) is None and sb(rec("a", "DEAF")) is None   # not exact; ambiguous
    up = rc.label_mapper("upper", labels)
    assert up(rec("w", "book")) == "BOOK" and up(rec("w", "deaf")) is None and up(rec("w", "thank you")) is None
    lab = rc.label_mapper("label", labels)
    assert lab(rec("c", "BOOK")) == "BOOK" and lab(rec("c", "book")) is None
    with pytest.raises(ValueError):
        rc.label_mapper("asllex", labels)
    with pytest.raises(ValueError):
        rc.label_mapper("signbank", labels)


def corpora_for_assembly():
    labels = {"BOOK", "CAT"}
    cit = dict(corpus_id="cit", records=[rec("c1", "BOOK", performer="1" * 64), rec("c2", "CAT", "val", performer="2" * 64),
                                         rec("c3", "BOOK", "test", performer="3" * 64),
                                         rec("c4", "CAT", performer="4" * 64)])
    sem = dict(corpus_id="sem", records=[rec("s1", "book", signer="11", code="A", corpus="sem", window=[.1, .5]),
                                         rec("s2", "cat", "val", signer="11", code="B", corpus="sem"),
                                         rec("s3", "cat", signer="12", code="B", corpus="sem"),
                                         rec("s4", "cat", "val", signer="13", code="B", corpus="sem"),
                                         rec("s5", "dog", signer="12", code="Z", corpus="sem"),
                                         rec("s6", "cat", "heldout_official", signer="12", code="B", corpus="sem")])
    primary_codes = dict(records=[rec("x", "BOOK", code="A"), rec("y", "CAT", code="B")])
    codes = rc.code_labels(primary_codes, labels)
    return labels, [(cit, rc.CorpusSpec("cit.json", SHA, "label", 1.), rc.label_mapper("label", labels)),
                    (sem, rc.CorpusSpec("sem.json", SHA, "asllex", 2.), rc.label_mapper("asllex", labels, codes=codes))]


def test_assembly_partitions_holds_out_and_excludes_signers_and_never_keeps_test():
    labels, corpora = corpora_for_assembly()
    records, summary = rc.assemble(corpora, labels, holdout={"sem": ("11",)}, exclude={"cit": ("4" * 16,)},
                                   whole_clip=["sem"])
    by = {r["sign"]: r for r in records}
    assert set(by) == {"c1", "c2", "s1", "s2", "s3", "s4"}          # c3 test, c4 excluded, s5 unmapped, s6 heldout
    assert by["c1"]["partition"] == "train" and by["c2"]["partition"] == "val:cit"
    assert by["s1"]["partition"] == by["s2"]["partition"] == "holdout:sem"     # train and val clips of signer 11
    assert by["s1"]["source_partition"] == "train" and by["s2"]["source_partition"] == "val"
    assert by["s3"]["partition"] == "train" and by["s4"]["partition"] == "val:sem"
    assert by["s1"]["gloss"]["label"] == "BOOK" and by["s1"]["gloss"]["source_label"] == "book"
    assert "window_seconds" not in by["s1"]["continuous"]           # --whole-clip
    assert by["s3"]["_weight"] == 2. and by["c1"]["_weight"] == 1.
    counts = summary["corpora"]["cit"]["counts"]
    assert counts["drop:test_partition"] == 1 and counts["drop:excluded_signer"] == 1
    assert summary["corpora"]["sem"]["counts"]["drop:heldout_official_partition"] == 1
    assert summary["corpora"]["sem"]["counts"]["drop:unmapped_label"] == 1
    for r in records:
        rc.check_loadable(r)


def test_assembly_refuses_unknown_corpora_signers_and_conflicts():
    labels, corpora = corpora_for_assembly()
    with pytest.raises(ValueError, match="unknown corpora"):
        rc.assemble(corpora, labels, holdout={"nope": ("1",)})
    with pytest.raises(ValueError, match="no record of signer"):
        rc.assemble(corpora, labels, holdout={"sem": ("99",)})
    with pytest.raises(ValueError, match="no record of signer"):
        rc.assemble(corpora, labels, exclude={"cit": ("3" * 64,)})  # only in the test partition: never matched
    with pytest.raises(ValueError, match="both held out and excluded"):
        rc.assemble(corpora, labels, holdout={"sem": ("12",)}, exclude={"sem": ("12",)})
    with pytest.raises(ValueError, match="unknown corpora"):
        rc.assemble(corpora, labels, whole_clip=["nope"])
    with pytest.raises(ValueError, match="share a name"):
        rc.assemble([corpora[0], corpora[0]], labels)


def test_signer_ids_match_own_id_full_sha_or_a_long_prefix_only():
    r = rec("a", "BOOK", performer="abcdef0123456789" + "0" * 48, signer="7")
    assert rc.signer_matches(r, ("7",)) == "7"
    assert rc.signer_matches(r, ("abcdef012345",)) == "abcdef012345"
    assert rc.signer_matches(r, ("abcdef01",)) is None              # too short to be a pseudonym prefix
    assert rc.signer_matches(rec("b", "BOOK"), ("7",)) is None
    assert rc.signer_of(r) == "7" and rc.signer_of(rec("b", "BOOK")) == "b" * 64


def test_the_guard_refuses_test_records_and_how2sign_signer_nine():
    rc.check_loadable(rec("ok", "BOOK", "val"))
    for bad in (rec("t", "BOOK", "test"), rec("h", "BOOK", "heldout_official"),
                dict(rec("v", "BOOK", "val:x"), source_partition="test"),
                rec("h9", "BOOK", "train", corpus="how2sign-two-view-text-v1", signer_component="c00"),
                rec("h?", "BOOK", "train", corpus="how2sign-two-view-text-v1"),
                rec("h9b", "BOOK", "train", corpus="how2sign-text-v1", signer_component="c01", signer="9")):
        with pytest.raises(RuntimeError):
            rc.check_loadable(bad)
    rc.check_loadable(rec("h1", "BOOK", "train", corpus="how2sign-two-view-text-v1", signer_component="c01"))
    labels = {"BOOK"}
    h2s = dict(corpus_id="how2sign-x", records=[rec("h", "BOOK", corpus="how2sign-x", signer_component="c00")])
    with pytest.raises(RuntimeError):
        rc.assemble([(h2s, rc.CorpusSpec("h.json", SHA, "label"), rc.label_mapper("label", labels))], labels)


def store_with_states(tmp_path, n=3):
    root = tmp_path / "store"
    shas = []
    for i in range(n):
        path = root / f"states/s{i}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        shas.append(signer_state(i, frames=12).save(path))
    return root, shas


def test_the_dataset_guards_on_construction_and_on_every_load(tmp_path):
    root, shas = store_with_states(tmp_path)
    r = dict(rec("s0", "BOOK", "val:x"), continuous=dict(rec("s0", "BOOK")["continuous"], state_sha256=shas[0]))
    cache = ContentCache(tmp_path / "cache", 2 ** 26)
    fetch = lambda k: (root / k).read_bytes()  # noqa: E731
    ds = GlossMotionDataset(dict(records=[r]), partition="val:x", labels={"BOOK": 0}, cache=cache, fetch=fetch)
    assert ds[0][0] == 0
    r["source_partition"] = "test"                                  # a record changed after construction
    with pytest.raises(RuntimeError):
        ds[0]
    with pytest.raises(RuntimeError):
        GlossMotionDataset(dict(records=[r]), partition="val:x", labels={"BOOK": 0}, cache=cache, fetch=fetch)


def test_decoded_copies_attach_only_to_the_record_they_were_made_from_and_are_served(tmp_path):
    root, shas = store_with_states(tmp_path)
    records = [dict(rec(f"s{i}", "BOOK", "train"), corpus="c",
                    continuous=dict(rec("x", "BOOK")["continuous"], state_key=f"states/s{i}.npz", state_sha256=shas[i]))
               for i in range(2)]
    entries = [dict(corpus="c", sign="s0", status="converted", load_digest=rc.load_digest(records[0]),
                    recon_key="states/s2.npz", recon_sha256=shas[2]),
               dict(corpus="c", sign="s1", status="converted", load_digest="0" * 64,      # made from another load
                    recon_key="states/s2.npz", recon_sha256=shas[2])]
    counts = rc.attach_recon(records, rc.recon_lookup(dict(entries=entries)))
    assert counts == {"train:with": 1, "train:without": 1}
    assert records[0]["_recon"] == dict(state_key="states/s2.npz", state_sha256=shas[2]) and "_recon" not in records[1]
    windowed = dict(records[0], continuous=dict(records[0]["continuous"], window_seconds=[0., .2]))
    assert rc.load_digest(windowed) != rc.load_digest(records[0])
    assert rc.load_digest(dict(records[0], mirror=True)) != rc.load_digest(records[0])
    cache = ContentCache(tmp_path / "cache", 2 ** 26)
    fetch = lambda k: (root / k).read_bytes()  # noqa: E731
    kw = dict(partition="train", labels={"BOOK": 0}, cache=cache, fetch=fetch)
    always = GlossMotionDataset(dict(records=records), recon_p=1., **kw)
    never = GlossMotionDataset(dict(records=records), recon_p=0., **kw)
    decoded = GlossMotionDataset(dict(records=records), decoded=True, **kw)
    assert always[0][1].sample_id == "s2" and always[1][1].sample_id == "s1"   # s1 has no copy: real
    assert never[0][1].sample_id == "s0"
    assert len(decoded) == 1 and decoded[0][1].sample_id == "s2"
    with pytest.raises(ValueError):
        rc.recon_lookup(dict(entries=[entries[0], dict(entries[0], recon_sha256="f" * 64)]))
    with pytest.raises(ValueError):
        GlossMotionDataset(dict(records=records), recon_p=1.5, **kw)


def test_hand_fraction_fallback_reads_semlex_quality_only_when_asked(tmp_path):
    root, shas = store_with_states(tmp_path, 1)
    base = dict(rec("s0", "BOOK"), continuous=dict(rec("s0", "BOOK")["continuous"], state_sha256=shas[0]))
    semlex = dict(base, quality=dict(frames=30, hand_fraction=.76))
    low = dict(base, sign="s1", quality=dict(frames=30, hand_fraction=.1))
    citizen = dict(base, sign="s2", quality=dict(frames=30, left_hand_frames=3, right_hand_frames=0))
    kw = dict(partition="train", labels={"BOOK": 0}, cache=ContentCache(tmp_path / "c", 2 ** 20),
              fetch=lambda k: b"", min_hand_fraction=.2)
    v2 = GlossMotionDataset(dict(records=[semlex, low, citizen]), **kw)
    v3 = GlossMotionDataset(dict(records=[semlex, low, citizen]), hand_fraction_fallback=True, **kw)
    assert [r["sign"] for r in v2.records] == []                    # v2's rule: no hand counts, no hand seen
    assert [r["sign"] for r in v3.records] == ["s0"]


def augment_v2(state, rng, *, speed=(.8, 1.25), trim=.1, face_dropout=.2):
    """Recogniser v2's ``augment_sign``, as it was before recogniser v3 (copied unchanged)."""
    from signtranslator.data_engineering.corpus_manifest import mask_channels
    t = state.timestamps
    n = len(t)
    if n > 8 and trim > 0:
        a = int(rng.random() * trim * n); b = n - int(rng.random() * trim * n)
        if b - a >= 4:
            state = replace(state, timestamps=t[a:b], channels={k: replace(c, values=c.values[a:b], valid=c.valid[a:b],
                            observed=c.observed[a:b], confidence=c.confidence[a:b]) for k, c in state.channels.items()})
    factor = speed[0] + rng.random() * (speed[1] - speed[0])
    state = replace(state, timestamps=state.timestamps[0] + (state.timestamps - state.timestamps[0]) / factor)
    if rng.random() < face_dropout:
        state = mask_channels(state, [k for k in state.channels if k not in ("face", "blink")])
    return state


def test_augmentation_without_validity_dropout_is_v2s_draw_for_draw():
    for seed in range(20):
        s = signer_state(seed, frames=30)
        a, b = random.Random(seed), random.Random(seed)
        x, y = augment_sign(s, a), augment_v2(s, b)
        assert a.random() == b.random()                             # the same number of draws
        np.testing.assert_array_equal(x.timestamps, y.timestamps)
        for name in s.channels:
            np.testing.assert_array_equal(x.channels[name].values, y.channels[name].values)
            np.testing.assert_array_equal(x.channels[name].valid, y.channels[name].valid)


def test_validity_dropout_invalidates_whole_groups_of_body_joints_only():
    s = signer_state(0, frames=40)
    seen = set()
    for seed in range(200):
        out = augment_sign(s, random.Random(seed), trim=0., face_dropout=0., validity_dropout=1.)
        out.validate()
        body = out.channels["body"]
        dropped = {label for j, label in enumerate(body.labels) if not body.valid[:, j].all()}
        seen.add(frozenset(dropped))
        for name in ("left_hand", "right_hand"):
            np.testing.assert_array_equal(out.channels[name].valid, s.channels[name].valid)
        assert np.all(body.values[~body.valid] == 0) and np.all(body.confidence[~body.valid] == 0)
        assert not (body.observed & ~body.valid).any()
    groups = {frozenset(v) for v in VALIDITY_GROUPS_JOINTS.values()}
    assert seen == groups and len(VALIDITY_GROUPS) == 5             # every group drawn, nothing else ever dropped
    unchanged = augment_sign(s, random.Random(3), trim=0., face_dropout=0., validity_dropout=0.)
    np.testing.assert_array_equal(unchanged.channels["body"].valid, s.channels["body"].valid)


def test_invalidate_joints_partial_frames_and_missing_labels():
    s = signer_state(1, frames=6)
    rows = np.array([True, False, True, False, False, False])
    out = invalidate_joints(s, ("LeftForearm", "NotAJoint"), rows)
    j = s.channels["body"].labels.index("LeftForearm")
    assert out.channels["body"].valid[:, j].tolist() == [False, True, False, True, True, True]
    others = [i for i in range(len(s.channels["body"].labels)) if i != j]
    np.testing.assert_array_equal(out.channels["body"].valid[:, others], s.channels["body"].valid[:, others])
    assert invalidate_joints(s, ("NotAJoint",), rows) is s and invalidate_joints(s, ("LeftForearm",), ~np.ones(6, bool)) is s
    out.validate()
