"""Round 14 pilot instrument: the mirrored read, the covered / newly covered word split, and the default report
byte for byte (end-to-end runs of ``pilot_content_check.main`` on a directory store, a stub generator whose states
carry the two-view layout, a stub back-translator and a stub recogniser)."""
from dataclasses import replace
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import scripts.pilot_content_check as pcc  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.pose.mirror import mirror_state  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402

TWO_VIEW = ("two-view-triangulated-camera1-y-up", "sapiens-two-view-swing-ik-y-up-v1")
MONOCULAR = ("camera-aligned-y-up-metres", "mediapipe-tasks-swing-ik-y-up-v1")
WORDS = ["ASK", "BIG", "CAT", "DOG", "EAT", "FLY", "GO", "HAT", "ICE", "JOB", "KEY", "LOW"]
LABELS = ["Z0", "Z1", "Z2", "Z3"] + WORDS          # four non-words fill the stub's background top-5 ranks
COVERED, UNCOVERED = WORDS[::3] + WORDS[1::3], WORDS[2::3]
DRAWN = []                                          # what the stub generator has drawn (words, sentence batches)
TEXTS = ["cut the onion in half now", "add some salt to the pan now", "turn the heat off now"]


def generated_state(sample_id, frames=20, layout=TWO_VIEW, drop=None, amplitude=0.):
    """A state in the generator's layout (body and hands in the two-view frame and convention); ``drop`` =
    (hand channel, frames) makes that hand unavailable on its first frames."""
    s = signer_state(0, frames=frames, amplitude=amplitude)
    channels = dict(s.channels)
    for name in ("body", "left_hand", "right_hand"):
        channels[name] = replace(channels[name], coordinate_frame=layout[0], convention=layout[1])
    if drop:
        name, n = drop
        c = channels[name]
        valid = c.valid.copy(); valid[:n] = False
        channels[name] = replace(c, values=np.where(valid[..., None], c.values, 0.), valid=valid,
                                 observed=c.observed & valid, confidence=np.where(valid, c.confidence, 0.))
    return replace(s, sample_id=sample_id, channels=channels)


def stub_sample(text, fps, seconds, guidance, steps, seed):
    """Word k, sample s: mode (k + s) % 3 -- 0 drops left-hand frames (read as generated), 1 drops right-hand frames
    (read when mirrored), 2 drops none (read in neither)."""
    DRAWN.append(text)
    k, s = WORDS.index(text.upper()), seed % 10
    mode = (k + s) % 3
    drop = ("left_hand", 3 + k % 4) if mode == 0 else ("right_hand", 3 + k % 4) if mode == 1 else None
    return generated_state(f"{text}|{seed}", drop=drop)


def stub_batch(texts, fps, seconds, guidance, seed):
    DRAWN.append("sentences")
    return [generated_state(f"{t}|{seed + i}", frames=int(round(d * fps))) for i, (t, d) in enumerate(zip(texts, seconds))]


stub_sample.batch = stub_batch


def stub_logits(model, states, device):
    """The target class (from the sample id) wins when the right hand is available on more frames than the left;
    otherwise a fixed wrong class. A fixed descending background makes every top-5 deterministic."""
    out = []
    for s in states:
        target = LABELS.index(s.sample_id.split("|")[0].upper())
        right, left = int(s.channels["right_hand"].valid.sum()), int(s.channels["left_hand"].valid.sum())
        row = -1e-3 * torch.arange(len(LABELS), dtype=torch.float64)
        row[target if right > left else (target + 7) % len(LABELS)] += 10.
        out.append(row)
    return torch.stack(out)


def stub_translate(model, kind, states):
    said = {18: TEXTS[0], 20: "add the salt to the pan now", 22: "turn the heat off"}
    return [said.get(len(s.timestamps), "now") if s.channels["body"].valid.any() else "blank" for s in states]


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    """main() of the pilot against a directory store with every heavy component stubbed; returns (store, run)."""
    import signtranslator.data.text_motion_dataset as tmd
    import signtranslator.data_engineering.corpus_manifest as cm
    store = DirectoryStore(tmp_path / "store")
    records = [dict(sign=f"t{k}", corpus="how2sign-two-view-text-v1", partition="train", text=" ".join(WORDS[k::4]).lower())
               for k in range(4)]
    records += [dict(sign=f"v{k}", corpus="how2sign-two-view-text-v1", partition="val", signer_component="c01",
                     text=t, continuous=dict(state_key=f"states/{k}.npz", state_sha256="0" * 64))
                for k, t in enumerate(TEXTS)]
    manifest = dict(records=records)
    store.put("mix.json", b"{}")
    store.put("training-runs/back-translator-kp2d-uwb-ft-v1/best.json", json.dumps(dict(checkpoint_key="bt.pt")).encode())
    store.put("training-runs/sign-recognizer-v2/validation.json",
              json.dumps(dict(per_gloss=dict(reliable_glosses=list(reversed(WORDS))))).encode())
    store.put("split.json", json.dumps(dict(covered=COVERED, uncovered=UNCOVERED)).encode())
    monkeypatch.setattr(tmd, "R2Store", lambda env: store)
    monkeypatch.setattr(cm, "load_manifest", lambda path, expected_sha256=None: manifest)
    modules = {n: importlib.import_module(n) for n in ("evaluate_text_to_motion", "score_back_translation",
                                                        "validate_sign_recognizer")}
    monkeypatch.setattr(modules["evaluate_text_to_motion"], "load_generator",
                        lambda store, key, sha, device, temperature=None: ("stub", (), stub_sample))
    monkeypatch.setattr(modules["evaluate_text_to_motion"], "load_state",
                        lambda store, key, sha: generated_state(key, frames=18 + 2 * int(key[7])))
    monkeypatch.setattr(modules["score_back_translation"], "load_translator", lambda store, key, sha: (torch.nn.Identity(), "kp"))
    monkeypatch.setattr(modules["score_back_translation"], "translate", stub_translate)
    monkeypatch.setattr(modules["validate_sign_recognizer"], "load_model", lambda store, run, device: (None, LABELS, None))
    monkeypatch.setattr(modules["validate_sign_recognizer"], "logits_of", stub_logits)

    def run(*extra, output="reports/pilot.json"):
        assert pcc.main(["--checkpoint-key", "prior.pt", "--mix-key", "mix.json", "--clips", "3", "--words", "12",
                         "--device", "cpu", "--output-key", output, *extra]) == 0
        return store.get(output)
    return store, run


# The report main() wrote before round 14 (scripts/pilot_content_check.py at 0051d3f0) on this fixture, byte for byte.
DEFAULT_REPORT_SHA256 = "08d1d25d4e36c514da014571998e02b5885cfa6bf2e8182e5818a72449c5ba26"


def test_the_default_report_is_byte_identical_to_the_round_13_instrument(pilot):
    store, run = pilot
    payload = run()
    assert hashlib.sha256(payload).hexdigest() == DEFAULT_REPORT_SHA256, hashlib.sha256(payload).hexdigest()


def test_both_flags_add_blocks_and_leave_the_rest_of_the_report_byte_identical(pilot):
    store, run = pilot
    default = run()
    payload = run("--mirrored-read", "--word-split-key", "split.json", output="reports/both.json")
    report = json.loads(payload)
    assert set(report) - set(json.loads(default)) == {"lexicon_mirrored", "lexicon_split"}
    trimmed = {k: v for k, v in report.items() if k not in ("lexicon_mirrored", "lexicon_split")}
    assert json.dumps(trimmed, indent=2, sort_keys=True, default=float).encode() == default
    only_mirror = json.loads(run("--mirrored-read", output="reports/mirror.json"))
    assert "lexicon_split" not in only_mirror and only_mirror["lexicon_mirrored"] == report["lexicon_mirrored"]


def test_the_mirrored_read_and_the_split_report_what_the_stub_signs(pilot):
    """As generated, a word is read on its mode-0 sample (k % 3 in {0, 2}); mirrored, on its mode-1 sample
    (k % 3 in {0, 1}); blanks never."""
    store, run = pilot
    split_sha = hashlib.sha256(store.get("split.json")).hexdigest()
    report = json.loads(run("--mirrored-read", "--word-split-key", "split.json", "--word-split-sha256", split_sha))
    lex, mir = report["lexicon"], report["lexicon_mirrored"]
    assert lex["per_word"] == {w: int(k % 3 != 1) for k, w in enumerate(WORDS)}
    assert mir["per_word"] == {w: int(k % 3 != 2) for k, w in enumerate(WORDS)}
    assert {f: mir[f] for f in ("words", "word_set", "samples", "recognizer_run")} == \
        {f: lex[f] for f in ("words", "word_set", "samples", "recognizer_run")}
    assert mir["generated_top5"] == lex["generated_top5"] == round(8 / 24, 4) and mir["blank_top5"] == 0.
    assert mir["available_channels"] == ["body", "left_hand", "right_hand"]
    assert mir["mirror"] == "signtranslator.pose.mirror.mirror_state"
    split = report["lexicon_split"]
    assert split["key"] == "split.json" and split["sha256"] == split_sha
    assert split["words"] == {"covered": [w for w in WORDS if w in COVERED], "uncovered": UNCOVERED}
    assert split["lexicon"] == {"covered": dict(words=8, samples=16, generated_top5=.25),
                                "uncovered": dict(words=4, samples=8, generated_top5=.5)}
    assert split["lexicon_mirrored"] == {"covered": dict(words=8, samples=16, generated_top5=.5),
                                         "uncovered": dict(words=4, samples=8, generated_top5=0.)}
    with pytest.raises(SystemExit, match="SHA-256"):
        run("--word-split-key", "split.json", "--word-split-sha256", "0" * 64, output="reports/bad.json")
    store.put("short.json", json.dumps(dict(covered=COVERED, uncovered=UNCOVERED[1:])).encode())
    DRAWN.clear()
    with pytest.raises(ValueError, match="partition"):           # refused before anything is generated
        run("--word-split-key", "short.json", output="reports/bad.json")
    assert DRAWN == [] and not store.exists("reports/bad.json")
    with pytest.raises(SystemExit, match="needs --word-split-key"):
        run("--word-split-sha256", split_sha, output="reports/bad.json")


# ---- mirroring the generator's layout ------------------------------------------------------------------------------

def test_mirroring_the_two_view_layout_is_an_involution_and_equals_the_monocular_rule():
    state = generated_state("s", frames=9, amplitude=.7, drop=("left_hand", 4))
    twice = mirror_state(mirror_state(state))
    for name, c in state.channels.items():
        np.testing.assert_allclose(twice.channels[name].values, c.values, atol=1e-9, err_msg=name)
        np.testing.assert_array_equal(twice.channels[name].valid, c.valid, err_msg=name)
        assert (twice.channels[name].labels, twice.channels[name].convention) == (c.labels, c.convention), name
    mono = generated_state("s", frames=9, amplitude=.7, drop=("left_hand", 4), layout=MONOCULAR)
    a, b = mirror_state(state), mirror_state(mono)
    for name in state.channels:
        np.testing.assert_array_equal(a.channels[name].values, b.channels[name].values, err_msg=name)
        np.testing.assert_array_equal(a.channels[name].valid, b.channels[name].valid, err_msg=name)
    assert a.channels["body"].convention == TWO_VIEW[1] and a.channels["left_hand"].coordinate_frame == TWO_VIEW[0]
    with pytest.raises(ValueError, match="mirroring is defined for"):
        mirror_state(generated_state("s", layout=("f", "some-other-ik")))
    contact = state.channels["contact"]
    touching = replace(state, channels={**state.channels, "contact": replace(
        contact, valid=np.ones_like(contact.valid), confidence=np.ones_like(contact.confidence),
        inference_method="stub")})
    with pytest.raises(ValueError, match="contact"):                  # no mirror rule: refused, not passed through
        mirror_state(touching)


def test_availability_moves_with_the_mirrored_values():
    state = generated_state("s", frames=8, amplitude=.5, drop=("left_hand", 3))
    body = state.channels["body"]
    valid = body.valid.copy(); valid[:2, list(body.labels).index("LeftForearm")] = False
    state = replace(state, channels={**state.channels, "body": replace(
        body, valid=valid, values=np.where(valid[..., None], body.values, 0.), confidence=np.where(valid, 1., 0.))})
    mirrored = pcc.mirrored_read(state)
    assert mirrored.sample_id == "s"
    np.testing.assert_array_equal(mirrored.channels["right_hand"].valid, state.channels["left_hand"].valid)
    np.testing.assert_array_equal(mirrored.channels["left_hand"].valid, state.channels["right_hand"].valid)
    labels = list(body.labels)
    assert not mirrored.channels["body"].valid[:2, labels.index("RightForearm")].any()
    assert mirrored.channels["body"].valid[:, labels.index("LeftForearm")].all()
    # a one-handed state: the available hand changes sides
    left = state.channels["left_hand"]
    right_only = replace(state, channels={**state.channels, "left_hand": replace(
        left, values=np.zeros_like(left.values), valid=np.zeros_like(left.valid), observed=np.zeros_like(left.valid),
        confidence=np.zeros_like(left.confidence))})
    assert pcc.available_channels(right_only) == ["body", "right_hand"]
    assert pcc.available_channels(pcc.mirrored_read(right_only)) == ["body", "left_hand"]
    # the ASL Citizen defect (values mirrored, availability not) is refused
    good = mirror_state(state)
    keep = {n: state.channels[n].valid for n in good.channels}
    rest = lambda c: np.where(c.valid[..., None], c.values, [1., 0., 0., 0., 1., 0.][:c.values.shape[-1]])
    stale = replace(good, channels={n: replace(c, valid=keep[n], observed=c.observed & keep[n],
                                               values=np.where(keep[n][..., None], rest(c), 0.),
                                               confidence=np.where(keep[n], 1., 0.))
                                    for n, c in good.channels.items()})
    with pytest.raises(ValueError, match="availability"):
        pcc.check_mirrored_availability(state, stale)
    pcc.check_mirrored_availability(state, good)


# ---- the word split --------------------------------------------------------------------------------------------------

def test_the_split_must_partition_the_scored_words_and_its_subsets_average_to_the_whole():
    words = WORDS
    ok = pcc.word_split(json.dumps(dict(covered=COVERED, uncovered=UNCOVERED, note="x")).encode(), words)
    assert ok == {"covered": [w for w in words if w in COVERED], "uncovered": UNCOVERED}
    for bad in (dict(covered=COVERED, uncovered=UNCOVERED[:-1]),                      # a scored word unassigned
                dict(covered=COVERED, uncovered=UNCOVERED + ["MOON"]),                # a word not scored
                dict(covered=COVERED + UNCOVERED[:1], uncovered=UNCOVERED),           # in both subsets
                dict(covered=COVERED + COVERED[:1], uncovered=UNCOVERED),             # repeated
                dict(covered=",".join(COVERED), uncovered=UNCOVERED), dict(covered=COVERED), [COVERED, UNCOVERED]):
        with pytest.raises(ValueError, match="word split"):
            pcc.word_split(json.dumps(bad).encode(), words)
    rng = np.random.default_rng(0)
    hits = rng.integers(0, 2, 2 * len(words)).astype(float)
    parts = pcc.split_lexicon(words, hits, ok)
    assert sum(p["samples"] for p in parts.values()) == len(hits) and sum(p["words"] for p in parts.values()) == 12
    rows = {k: [2 * words.index(w) + s for w in v for s in (0, 1)] for k, v in ok.items()}
    for k, p in parts.items():
        assert p["generated_top5"] == round(float(hits[rows[k]].mean()), 4)
    weighted = sum(hits[rows[k]].sum() for k in ok) / len(hits)
    assert weighted == pytest.approx(hits.mean(), abs=1e-12)
    # the R5 shape: 108 + 42 of 150
    r5 = [f"W{i:03d}" for i in range(150)]
    split = pcc.word_split(json.dumps(dict(covered=r5[:108], uncovered=r5[108:])).encode(), r5)
    parts = pcc.split_lexicon(r5, np.r_[np.ones(216), np.zeros(84)], split)
    assert parts == {"covered": dict(words=108, samples=216, generated_top5=1.),
                     "uncovered": dict(words=42, samples=84, generated_top5=0.)}


def test_a_pinned_word_list_scores_exactly_its_words_and_adds_only_its_record(pilot):
    """Recogniser v3: --word-list-key pins the words; the R5 split read as a list gives the default words, so the
    report is the default one plus ``word_list``; the reliable-gloss list is then not read at all."""
    store, run = pilot
    default = json.loads(run())
    split_sha = hashlib.sha256(store.get("split.json")).hexdigest()
    (store.root / "training-runs/sign-recognizer-v2/validation.json").unlink()   # a new instrument has no v2 list
    pinned = json.loads(run("--word-list-key", "split.json", "--word-list-sha256", split_sha, output="reports/pin.json"))
    assert pinned["word_list"] == dict(key="split.json", sha256=split_sha, words=12)
    assert {k: v for k, v in pinned.items() if k != "word_list"} == default
    reordered = list(reversed(WORDS))
    store.put("list.json", json.dumps(dict(words=reordered)).encode())
    other = json.loads(run("--word-list-key", "list.json", "--word-list-sha256",
                           hashlib.sha256(store.get("list.json")).hexdigest(), output="reports/rev.json"))
    assert list(other["lexicon"]["per_word"]) == sorted(reordered) and other["lexicon"]["words"] == 12


def test_a_pinned_word_list_is_checked_before_anything_runs(pilot):
    store, run = pilot
    store.put("list.json", json.dumps(WORDS).encode())
    sha = hashlib.sha256(store.get("list.json")).hexdigest()
    with pytest.raises(SystemExit, match="SHA-256"):
        run("--word-list-key", "list.json", "--word-list-sha256", "0" * 64)
    with pytest.raises(SystemExit, match="go together"):
        run("--word-list-key", "list.json")
    with pytest.raises(SystemExit, match="dev"):
        run("--word-list-key", "list.json", "--word-list-sha256", sha, "--word-set", "dev")
    index = {w: i for i, w in enumerate(LABELS)}
    assert pcc.pinned_words(json.dumps(WORDS).encode(), index, 12) == WORDS
    with pytest.raises(ValueError, match="not labels"):
        pcc.pinned_words(json.dumps(WORDS[:11] + ["ZEBRA"]).encode(), index, 12)
    with pytest.raises(ValueError, match="--words"):
        pcc.pinned_words(json.dumps(WORDS[:11]).encode(), index, 12)
    with pytest.raises(ValueError, match="distinct"):
        pcc.pinned_words(json.dumps(WORDS[:11] + ["ASK"]).encode(), index, 12)
    with pytest.raises(ValueError, match="expected"):
        pcc.pinned_words(json.dumps(dict(x=1)).encode(), {}, 12)
