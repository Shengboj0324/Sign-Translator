"""Stage S2's probes on synthetic data (docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design received"). P1: the reveal order is a
chain rule and leaks nothing; planted text information is recovered at the right size; unrelated text gains nothing;
units match the stored AR figures. Every probe's guards, damage forms, verdict bars and write-once outputs, each also
shown failing on a deliberately broken input."""
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s0_ceilings as s0  # noqa: E402
import s1_text_kl as s1  # noqa: E402
import s2_amplitude_probe as p4  # noqa: E402
import s2_length_probe as p3  # noqa: E402
import s2_lib as L  # noqa: E402
import s2_noise_curve as p2  # noqa: E402
import s2_text_info_probe as p1  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402
from signtranslator.eval import s2_text_info as T  # noqa: E402
from signtranslator.generation.motion_rvq import MotionRVQ, MotionRVQConfig, RVQPrior, RVQPriorConfig  # noqa: E402
from tests.test_generation_motion_rvq import motion_config  # noqa: E402
from tests.test_generation_signing_space import signer_state  # noqa: E402

H2S = L.H2S


def tiny_tokenizer(stages=4, codebook=32, seed=0):
    torch.manual_seed(seed)
    state = signer_state(0, frames=16, amplitude=.4)
    return MotionRVQ(MotionRVQConfig(motion_config(state), hidden=48, code_dim=16, codebook=codebook, stages=stages,
                                     down_layers=0)).eval()


def tiny_head(codebook=16, parts=2, text_dim=8, d=48, layers=2, dropout=0., seed=0):
    torch.manual_seed(seed)
    return T.TextInfoHead(T.HeadConfig(codebook=codebook, parts=parts, d_model=d, heads=4, layers=layers, feedforward=96,
                                       dropout=dropout, text_dim=text_dim))


def random_batch(seed=0, K=16, P=2, lengths=(9, 16, 13), text_dim=8):
    g = torch.Generator().manual_seed(seed)
    S = max(lengths)
    codes = torch.randint(0, K, (len(lengths), S, P), generator=g)
    return codes, torch.tensor(lengths), torch.randn(len(lengths), 5, text_dim, generator=g), torch.ones(len(lengths), 5, dtype=torch.bool)


# ---- the reveal order ------------------------------------------------------------------------------------------------

def test_valuations_read_step_zero_first_then_coarse_to_fine():
    assert T.valuations(9).tolist() == [T.TOP_VALUATION, 0, 1, 0, 2, 0, 1, 0, 3]
    groups = T.chain_groups(9, 2)
    assert groups[:2] == [(T.TOP_VALUATION, 0), (T.TOP_VALUATION, 1)] and groups[2:4] == [(3, 0), (3, 1)]
    assert groups[-1] == (0, 1) and len(groups) == 5 * 2


def test_reveal_is_a_chain_rule_each_code_predicted_once_and_context_is_exactly_what_came_before():
    codes, lengths, _, _ = random_batch(lengths=(37, 21), P=4)
    seen_so_far = torch.zeros_like(codes, dtype=torch.bool)
    total = torch.zeros_like(codes, dtype=torch.long)
    for group in T.chain_groups(37, 4):
        inputs, target = T.reveal(codes, lengths, group, 16)
        revealed = inputs != 16
        assert torch.equal(revealed, seen_so_far), group       # the context is the earlier groups' targets, nothing else
        assert not (revealed & target).any()                   # a target never sees itself
        assert torch.equal(inputs[revealed], codes[revealed])
        total += target.long()
        seen_so_far |= target
    valid = (torch.arange(37)[None] < lengths[:, None])[..., None].expand_as(codes)
    assert torch.equal(total, valid.long())                    # every valid code exactly once, padding never


def test_the_leak_check_passes_on_the_head_and_catches_a_target_or_a_future_code_in_the_input():
    head = tiny_head().eval()
    codes, lengths, text, mask = random_batch()
    assert T.check_no_leak(head, codes, lengths, text, mask) == len(T.chain_groups(16, 2))

    def target_leak(c, n, group, m):
        inputs, target = T.reveal(c, n, group, m)
        return torch.where(target, c, inputs), target

    def future_leak(c, n, group, m):
        inputs, target = T.reveal(c, n, group, m)
        out = inputs.clone()
        out[:, 15, 0] = c[:, 15, 0]                            # the last step (read last) is visible to every group
        return out, target

    for broken in (target_leak, future_leak):
        with pytest.raises(ValueError, match="not revealed"):
            T.check_no_leak(head, codes, lengths, text, mask, reveal_fn=broken)


def test_the_leak_check_is_not_vacuous_when_the_revealed_codes_are_ignored():
    head = tiny_head().eval()
    codes, lengths, text, mask = random_batch()

    def blind(c, n, group, m):
        _, target = T.reveal(c, n, group, m)
        return torch.full_like(c, m), target

    with pytest.raises(ValueError, match="vacuous"):
        T.check_no_leak(head, codes, lengths, text, mask, reveal_fn=blind)


def test_a_clips_chain_likelihood_does_not_depend_on_its_batch():
    head = tiny_head().eval()
    codes, lengths, text, mask = random_batch()
    together = T.chain_log_probs(head, codes, lengths, text, mask)[0]
    for i, n in enumerate(lengths.tolist()):
        alone = T.chain_log_probs(head, codes[i:i + 1, :n], lengths[i:i + 1], text[i:i + 1], mask[i:i + 1])[0]
        assert float(alone[0]) == pytest.approx(float(together[i]), abs=2e-3)
    other = codes.clone()
    other[0, 9:] = 3                                            # beyond clip 0's length
    assert float(T.chain_log_probs(head, other, lengths, text, mask)[0][0]) == pytest.approx(float(together[0]), abs=1e-6)


def test_the_chain_refuses_a_head_in_train_mode():
    head = tiny_head().train()
    codes, lengths, text, mask = random_batch()
    with pytest.raises(ValueError, match="eval mode"):
        T.chain_log_probs(head, codes, lengths, text, mask)
    with pytest.raises(ValueError, match="eval mode"):
        T.sample_chain(head, lengths, text, mask)


def test_sampling_reads_no_reference_and_fills_every_valid_code():
    head = tiny_head().eval()
    _, lengths, text, mask = random_batch()
    a = T.sample_chain(head, lengths, text, mask, seed=1)
    b = T.sample_chain(head, lengths, text, mask, seed=1)
    assert torch.equal(a, b) and a[0, :9].max() < 16 and a[0, 9:].min() == 16


# ---- recovery: planted information, and none ----------------------------------------------------------------------

CLASSES = 8


def planted_rows(n, seed, *, persistent=True, unrelated=False, K=16):
    """Part 0 holds a class for the whole clip, part 1 is noise. The text is the class as a one-hot token, so
    I(text; codes) = ln 8 exactly, however long the clip. ``unrelated``: the text is another, independent class."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        y = int(rng.integers(CLASSES))
        S = int(rng.integers(16, 33))
        codes = np.stack([np.full(S, y), rng.integers(0, K, S)], 1)
        t = int(rng.integers(CLASSES)) if unrelated else y
        token = np.eye(CLASSES, dtype=np.float32)[[t]]
        rows.append((codes, token))
    return rows


def fit_planted(rows, *, reveal_fn=T.reveal, steps=900, seed=0):
    head = tiny_head(seed=seed)
    empty = torch.zeros(1, 1, CLASSES)
    T.fit_head(head, rows, empty_text=empty, steps=steps, batch=32, lr=2e-3, seed=seed, reveal_fn=reveal_fn)
    return head.eval(), empty


def gains(head, rows, empty, *, reveal_fn=T.reveal, tokens_of=None):
    out, seconds = [], []
    for start in range(0, len(rows), 32):
        chunk = rows[start:start + 32]
        codes, lengths = T.padded_codes([c for c, _ in chunk])
        text, mask = T.padded_text([(tokens_of(i + start) if tokens_of else t) for i, (_, t) in enumerate(chunk)])
        none = (empty.expand(len(chunk), -1, -1), torch.ones(len(chunk), 1, dtype=torch.bool))
        own = T.chain_log_probs(head, codes, lengths, text, mask, reveal_fn=reveal_fn)[0]
        base = T.chain_log_probs(head, codes, lengths, *none, reveal_fn=reveal_fn)[0]
        out += (own - base).tolist()
        seconds += [len(c) / L.FPS for c, _ in chunk]
    return np.array(out), np.array(seconds)


@pytest.fixture(scope="module")
def planted():
    rows, held = planted_rows(800, 0), planted_rows(200, 1)
    head, empty = fit_planted(rows, steps=4000)
    return head, empty, rows, held


def test_planted_information_is_recovered_at_its_true_size_and_not_counted_per_step(planted):
    head, empty, _, held = planted
    gain, seconds = gains(head, held, empty)
    assert math.log(CLASSES) * .85 < gain.mean() < math.log(CLASSES) * 1.1   # ln 8 once, not 24 × ln 8
    rate = L.pooled_rate(gain, seconds)
    assert rate["rate"] == pytest.approx(gain.sum() / seconds.sum(), abs=1e-4) and rate["ci95"][0] > 0


def test_an_underfitted_empty_branch_overstates_the_gain_and_more_fitting_removes_it(planted):
    """The estimator's known bias, which the verdict's readers must keep in mind: the empty branch must infer from revealed
    codes what the text branch is told, and any shortfall counts as gain. It falls as the fit converges."""
    head, empty, rows, held = planted
    early, empty_early = fit_planted(rows, steps=900)
    assert gains(early, held, empty_early)[0].mean() > gains(head, held, empty)[0].mean() > math.log(CLASSES) * .85


def test_the_deranged_text_gains_nothing_on_a_head_that_reads_its_text(planted):
    head, empty, _, held = planted
    deranged = [np.eye(CLASSES, dtype=np.float32)[[(int(c[0, 0]) + 1) % CLASSES]] for c, _ in held]
    gain, seconds = gains(head, held, empty, tokens_of=lambda i: deranged[i])
    assert L.pooled_rate(gain, seconds)["rate"] <= .2         # P1's shuffled bar, on the planted case: well below 0


def test_text_unrelated_to_the_codes_gains_about_zero_on_held_out_clips():
    head, empty = fit_planted(planted_rows(800, 0, unrelated=True))
    gain, _ = gains(head, planted_rows(200, 1, unrelated=True), empty)
    assert abs(gain.mean()) < .15 * math.log(CLASSES)


def test_breaking_the_chain_counts_the_same_information_at_every_step():
    def nothing_revealed(c, n, group, m):
        _, target = T.reveal(c, n, group, m)
        return torch.full_like(c, m), target
    rows, held = planted_rows(800, 0), planted_rows(200, 1)
    head, empty = fit_planted(rows, reveal_fn=nothing_revealed)
    gain, _ = gains(head, held, empty, reveal_fn=nothing_revealed)
    assert gain.mean() > 5 * math.log(CLASSES)                # what a marginal read would report: the chain prevents it


def test_a_target_leak_in_the_fit_and_read_is_caught_before_it_could_report_a_gain():
    def leak(c, n, group, m):
        inputs, target = T.reveal(c, n, group, m)
        return torch.where(target, c, inputs), target
    head = tiny_head().eval()
    codes, lengths, text, mask = random_batch()
    with pytest.raises(ValueError):
        T.check_no_leak(head, codes, lengths, text, mask, reveal_fn=leak)


def test_the_empty_branch_is_trained_on_half_of_the_rows_so_it_is_not_worse_fitted():
    head = tiny_head()
    codes, lengths, text, mask = random_batch(lengths=(16, 16, 16, 16, 16, 16, 16, 16))
    seen = []
    original = T.TextInfoHead.forward

    def spy(self, inputs, n, tokens, token_mask):
        seen.append(token_mask.sum(1).tolist())
        return original(self, inputs, n, tokens, token_mask)
    T.TextInfoHead.forward = spy
    try:
        counts = []
        for s in range(50):
            T.training_batch_loss(head, codes, lengths, text[:1].expand(8, -1, -1), mask[:1].expand(8, -1),
                                  torch.zeros(1, 1, 8), torch.Generator().manual_seed(s))
            counts.append(sum(x == 1 for x in seen[-1]))
    finally:
        T.TextInfoHead.forward = original
    assert 3 < np.mean(counts) < 5                             # 8 rows, half with the 1-token empty text


def test_early_stopping_keeps_the_best_read_and_stops_after_the_declared_patience():
    head = tiny_head(dropout=.1)
    rows = planted_rows(16, 0)
    values = iter([3., 2., 2.5, 2.6, 9.])
    kept = {}

    def stop():
        v = next(values)
        if v == 2.:
            kept["state"] = {k: x.clone() for k, x in head.state_dict().items()}
        return v
    out = T.fit_head(head, rows, empty_text=torch.zeros(1, 1, CLASSES), steps=100, batch=4, every=5, patience=2, stop_read=stop)
    assert out["best_step"] == 10 and out["steps_run"] == 20 and out["best_stop_nll"] == 2.
    assert not head.training
    for k, x in head.state_dict().items():
        assert torch.equal(x, kept["state"][k]), k


# ---- units -----------------------------------------------------------------------------------------------------------

def test_the_rate_unit_reproduces_the_stored_ar_figures():
    assert round(L.STORED_AR_GAIN_PER_CODE["flant5"] * L.STAGE1_CODES_PER_SECOND, 2) == 1.42
    assert round(L.STORED_AR_GAIN_PER_CODE["w14b"] * L.STAGE1_CODES_PER_SECOND, 2) == -.51
    per_code = np.array([.01, .02, -.005])
    steps = np.array([120, 48, 240])
    nats = per_code * steps * 4
    pooled = L.pooled_rate(nats, steps / 24.)
    assert pooled["rate"] == pytest.approx((per_code * steps).sum() / steps.sum() * 96, abs=1e-4)


def test_a_wrong_duration_or_unpaired_input_is_refused():
    with pytest.raises(ValueError):
        L.pooled_rate([1., 2.], [1.])
    with pytest.raises(ValueError):
        L.pooled_rate([1.], [0.])


def s1_tiny(parts_codes=True):
    tokenizer = tiny_tokenizer(stages=3)
    torch.manual_seed(3)
    prior = RVQPrior(RVQPriorConfig(tokenizer.config, d_model=32, heads=4, text_layers=1, temporal_layers=1, depth_layers=1,
                                    feedforward=64, dropout=0., locus_conditioning=True)).eval()
    return tokenizer, prior


def test_s1_rates_match_its_per_code_gain_times_codes_per_second_and_list_clips_left_out():
    tokenizer, prior = s1_tiny()
    states = [signer_state(s, frames=n, amplitude=.4) for s, n in ((1, 16), (2, 11), (3, 12))]
    texts = ["dog", "a cat runs", "sun"]
    reads = s1.clip_reads(prior, tokenizer, texts, states, loci_mode="plan", device=torch.device("cpu"))
    seconds = [(len(s.timestamps)) / 24. for s in states]
    summary = s1.summarise(reads, [p for p, _ in tokenizer.parts], seconds)
    s1.add_rates(summary, reads, seconds)
    P = len(tokenizer.parts)
    assert summary["gain_nats_per_second"]["rate"] == pytest.approx(float(reads["gain"].mean()) * P * 24., abs=1e-4)
    assert summary["per_clip"]["seconds"] == pytest.approx(seconds, abs=1e-4)
    keep = reads["row"] != 1                                    # clip 1 left out, as a clip over max_seconds is
    partial = {k: v[keep] for k, v in reads.items()}
    summary2 = s1.summarise(partial, [p for p, _ in tokenizer.parts], seconds)
    s1.add_rates(summary2, partial, seconds)
    assert summary2["per_clip"]["gain_nats"][1] is None and summary2["gain_nats_per_second"]["n"] == 2


# ---- lists and guards ------------------------------------------------------------------------------------------------

class SpyStore:
    def __init__(self, files=None):
        self.files, self.read = dict(files or {}), []

    def get(self, key):
        self.read.append(key)
        return self.files[key]

    def exists(self, key):
        return key in self.files


def val_manifest(n=3096):
    from build_round20_sets import PREFIX  # noqa: F401  (the sets' own module loads)
    groups = L.VAL_GROUPS
    recs = [dict(corpus=H2S, partition="val", signer_component=groups[i % 3], text=f"sentence number {i} here now",
                 sign=f"s{i}", continuous=dict(state_key=f"k{i}", state_sha256="0" * 64, seconds=3.)) for i in range(n)]
    return dict(records=recs)


def d20_files(manifest, *, name="D20", take=None):
    """A D20-like list outside the 1,000 and the 300 (or the given records), and its files for a SpyStore."""
    from pilot_content_check import validation_sentences
    from round20_lib import set_payload
    forbidden = {r["sign"] for k in (1000, 300) for r in validation_sentences(manifest, k)}
    records = take if take is not None else [r for r in manifest["records"] if r["sign"] not in forbidden][:50]
    raw = json.dumps(set_payload(name, records, manifest_key=L.EVAL_MIX, manifest_sha256=L.EVAL_MIX_SHA, rule="t", counts={})).encode()
    return raw, records


def pin(monkeypatch, raw):
    import hashlib
    monkeypatch.setattr(L, "D20_SHA", hashlib.sha256(raw).hexdigest())


def test_d20_loads_when_pinned_and_c20_is_refused_without_being_read(monkeypatch):
    manifest = val_manifest()
    raw, records = d20_files(manifest)
    pin(monkeypatch, raw)
    store = SpyStore({L.D20_KEY: raw})
    assert [r["sign"] for r in L.load_read_list(store, manifest)] == [r["sign"] for r in records]
    c20 = L.D20_KEY.replace("d20", "c20")
    store = SpyStore({c20: raw})
    with pytest.raises(SystemExit, match="D20 only"):
        L.load_read_list(store, manifest, c20)
    assert store.read == []                                     # never fetched


def test_a_list_with_another_hash_or_name_is_refused(monkeypatch):
    manifest = val_manifest()
    raw, _ = d20_files(manifest)
    with pytest.raises(SystemExit, match="SHA-256"):
        L.load_read_list(SpyStore({L.D20_KEY: raw}), manifest)                 # the real pin, a different file
    named, _ = d20_files(manifest, name="C20")
    pin(monkeypatch, named)
    with pytest.raises(SystemExit, match="not D20"):
        L.load_read_list(SpyStore({L.D20_KEY: named}), manifest)


@pytest.mark.parametrize("k", [1000, 300])
def test_a_list_holding_a_clip_of_the_1000_or_the_300_is_refused(monkeypatch, k):
    from pilot_content_check import validation_sentences
    manifest = val_manifest()
    raw, _ = d20_files(manifest, take=validation_sentences(manifest, k)[:5])
    pin(monkeypatch, raw)
    with pytest.raises(SystemExit, match="1,000 or the 300"):
        L.load_read_list(SpyStore({L.D20_KEY: raw}), manifest)


@pytest.mark.parametrize("change", [dict(partition="test"), dict(partition="heldout_official"), dict(signer_component="c00"),
                                    dict(partition="train"), dict(corpus="asl-citizen")])
def test_a_read_refuses_test_held_out_c00_and_train_records(change):
    record = dict(corpus=H2S, partition="val", signer_component="c01", sign="x", text="t")
    L.check_read_records([record])
    with pytest.raises(SystemExit):
        L.check_read_records([dict(record, **change)])


def train_manifest():
    recs = []
    for i in range(30):
        recs.append(dict(corpus=H2S, partition="train", signer_component="c02", sign=f"t{i}", text=f"train words number{i} alpha{i} beta{i}",
                         continuous=dict(state_key=f"k{i}", state_sha256="0" * 64, seconds=2. + i / 10)))
    for i in range(8):
        recs.append(dict(corpus=H2S, partition="train", signer_component="c04", sign=f"f{i}", text=f"other sentence{i} gamma{i} delta{i} zeta{i}",
                         continuous=dict(state_key=f"f{i}", state_sha256="0" * 64, seconds=2.)))
    recs += [dict(corpus=H2S, partition="train", signer_component="c00", sign="n", text="signer nine", continuous=dict(state_key="n", state_sha256="0"*64)),
             dict(corpus=H2S, partition="test", signer_component="c02", sign="q", text="test words", continuous=dict(state_key="q", state_sha256="0"*64)),
             dict(corpus=H2S, partition="val", signer_component="c02", sign="v", text="val words", continuous=dict(state_key="v", state_sha256="0"*64))]
    return dict(records=recs)


def test_train_records_never_return_c00_test_or_val_and_refuse_being_asked():
    manifest = train_manifest()
    signs = {r["sign"] for r in L.train_records(manifest, ("c02", "c04"))}
    assert not signs & {"n", "q", "v"} and len(signs) == 38
    with pytest.raises(SystemExit, match="signer 9"):
        L.train_records(manifest, ("c00",))
    with pytest.raises(SystemExit):
        L.check_train_records([dict(sign="v", partition="val", signer_component="c02")], ("c02",))


def test_the_fit_is_c02_train_only_and_drops_read_stop_and_near_texts():
    manifest = train_manifest()
    read = [dict(sign="r0", text="train words number3 alpha3 beta3", corpus=H2S, partition="val", signer_component="c01"),   # same text as t3
            dict(sign="r1", text="train words number4 alpha4 beta4 extra", corpus=H2S, partition="val", signer_component="c01"),  # near t4
            dict(sign="t5", text="something else entirely here", corpus=H2S, partition="train", signer_component="c02")]     # a smoke's read clip
    manifest["records"][6]["text"] = manifest["records"][30]["text"]                # t6 shares a text with stop clip f0
    fit, stop, counts = p1.fit_lists(manifest, read, fit_clips=100, stop_clips=4)
    signs = {r["sign"] for r in fit}
    assert not signs & {"t3", "t4", "t5", "t6", "n", "q", "v"}
    assert {r["signer_component"] for r in fit} == {"c02"} and {r["signer_component"] for r in stop} == {"c04"}
    assert len(stop) == 4 and counts["fit_dropped_read_text"] == 2 and counts["fit_dropped_read_clip"] == 1
    assert counts["fit_dropped_stop_text"] == 1
    again = p1.fit_lists(manifest, read, fit_clips=100, stop_clips=4)
    assert [r["sign"] for r in again[0]] == [r["sign"] for r in fit]                # fixed order, fixed before any read
    assert [r["sign"] for r in again[1]] == [r["sign"] for r in stop]


def test_the_stopping_split_does_not_move_with_the_fit_size():
    manifest = train_manifest()
    a = p1.fit_lists(manifest, [], fit_clips=5, stop_clips=4)
    b = p1.fit_lists(manifest, [], fit_clips=20, stop_clips=4)
    assert [r["sign"] for r in a[1]] == [r["sign"] for r in b[1]]
    assert [r["sign"] for r in a[0]] == [r["sign"] for r in b[0]][:5]


def test_shuffled_texts_always_differ_from_the_clips_own():
    texts = [f"word{i}" for i in range(8)] + ["word0"]
    from round20_lib import text_key
    for seed in range(30):
        p = p1.derangement(texts, seed)
        assert sorted(p) == list(range(9)) and all(text_key(texts[i]) != text_key(texts[p[i]]) for i in range(9))
    with pytest.raises(ValueError):
        p1.derangement(["same", "same"])


def control_manifest():
    recs = []
    for corpus in p1.WORD_CORPORA + ("asl-citizen-text",):
        for performer in range(40):
            for word in range(6):
                recs.append(dict(corpus=corpus, partition="train", performer_sha256=f"p{performer}", text=f"word{word}", sign=f"{corpus}-{performer}-{word}",
                                 continuous=dict(state_key="k", state_sha256="0" * 64)))
    recs.append(dict(corpus=p1.WORD_CORPORA[0], partition="test", performer_sha256="p1", text="word0", sign="test", continuous=dict(state_key="k", state_sha256="0"*64)))
    recs.append(dict(corpus=p1.WORD_CORPORA[0], partition="train", performer_sha256="p1", text="two words", sign="long", continuous=dict(state_key="k", state_sha256="0"*64)))
    return dict(records=recs)


def test_the_word_control_holds_out_performers_and_reads_no_licence_barred_or_test_clip():
    fit, read, counts = p1.control_lists(control_manifest(), fit_clips=50, words=4)
    assert len(read) == 4 and len({r["text"] for r in read}) == 4 and counts["fit"] == 50
    assert not {r["performer_sha256"] for r in fit} & {r["performer_sha256"] for r in read}
    assert {r["text"] for r in read} <= {r["text"] for r in fit}
    for r in fit + read:
        assert r["corpus"] in p1.WORD_CORPORA and r["partition"] == "train" and r["sign"] not in ("test", "long")


# ---- P1's verdict ----------------------------------------------------------------------------------------------------

def p1_inputs(rate=3., ar=1., shuffled=-.5, top5=.2, control=.5, groups=(1., 1., 1.), n=500, short=86):
    signs = [f"s{i}" for i in range(n)]
    seconds = np.full(n, 5.)
    gain = np.full(n, rate * 5.)
    labels = [L.VAL_GROUPS[i % 3] for i in range(n)]
    read = dict(signs=signs, groups=labels, seconds=seconds.tolist(), gain_nats=gain.tolist(),
                shuffled_gain_nats=[shuffled * 5.] * n, gain=L.pooled_rate(gain, seconds),
                shuffled_gain=L.pooled_rate(np.full(n, shuffled * 5.), seconds),
                by_signer_group={g: dict(rate=v) for g, v in zip(L.VAL_GROUPS, groups)},
                breakdown_nats_per_clip={})
    rank = dict(clips=short, summary=dict(top5=top5))
    ctrl = dict(words=64, summary=dict(top1=control))
    prior = lambda r: dict(gain=r / 96., gain_nats_per_second=dict(rate=r), per_clip=dict(gain_nats=[r * 5.] * n, seconds=[5.] * n))
    s1_report = dict(sentence_set=dict(sha256=L.D20_SHA), signs=signs, priors=dict(flant5=prior(ar), w14b=prior(-.5)))
    return read, rank, ctrl, s1_report


def test_p1_passes_only_with_every_bar_and_each_broken_bar_fails_it():
    v = p1.verdict(*p1_inputs(), declared_read=True, scale="pod")
    assert v["outcome"] == "pass" and all(b["passed"] for b in v["bars"]) and v["control"]["passed"]
    for broken in (dict(rate=2.8), dict(rate=3., ar=1.6), dict(shuffled=.3), dict(groups=(1., 1., -.1)), dict(top5=.14)):
        v = p1.verdict(*p1_inputs(**broken), declared_read=True, scale="pod")
        assert v["outcome"] == "fail", broken
        assert not v["confirm_on_pod"]


def test_p1_bars_sit_exactly_where_they_were_declared():
    assert (p1.GAIN_BAR, p1.RATIO_BAR, p1.SHUFFLED_BAR, p1.SHORT_TOP5_BAR, p1.CONTROL_BAR) == (2.84, 2., .2, .15, .3)
    assert p1.verdict(*p1_inputs(rate=2.84, ar=1.42), declared_read=True, scale="pod")["outcome"] == "pass"
    assert p1.verdict(*p1_inputs(ar=-.5), declared_read=True, scale="pod")["outcome"] == "fail"    # an AR below 0: no ratio to pass


def test_p1_is_void_when_the_control_fails_whatever_the_gain_and_a_mac_fail_asks_for_the_pod():
    assert p1.verdict(*p1_inputs(control=.29), declared_read=True, scale="mac")["outcome"] == "void"
    v = p1.verdict(*p1_inputs(rate=1.), declared_read=True, scale="mac")
    assert v["outcome"] == "fail" and v["confirm_on_pod"]


def test_p1_refuses_mismatched_units_other_clips_and_short_declared_reads():
    read, rank, ctrl, report = p1_inputs()
    report["priors"]["flant5"]["gain"] = 1.                                          # not nats/s ÷ 96
    with pytest.raises(SystemExit, match="units differ"):
        p1.verdict(read, rank, ctrl, report, declared_read=True, scale="pod")
    read, rank, ctrl, report = p1_inputs()
    report["sentence_set"]["sha256"] = "other"
    with pytest.raises(SystemExit, match="not a D20 read"):
        p1.verdict(read, rank, ctrl, report, declared_read=True, scale="pod")
    with pytest.raises(SystemExit, match="short clips"):
        p1.verdict(*p1_inputs(short=20), declared_read=True, scale="pod")
    read, rank, ctrl, report = p1_inputs()
    report["signs"] = [f"x{i}" for i in range(500)]
    with pytest.raises(SystemExit, match="shares no clip"):
        p1.verdict(read, rank, ctrl, report, declared_read=True, scale="pod")
    read, rank, ctrl, report = p1_inputs()
    report["priors"]["flant5"]["per_clip"]["seconds"] = [9.] * 500
    with pytest.raises(SystemExit, match="durations"):
        p1.verdict(read, rank, ctrl, report, declared_read=True, scale="pod")


def test_p1_compares_on_the_clips_the_ar_read_and_says_a_smoke_is_a_smoke():
    read, rank, ctrl, report = p1_inputs()
    report["priors"]["flant5"]["per_clip"]["gain_nats"][0] = None                    # left out by the AR prior
    report["priors"]["w14b"]["per_clip"]["gain_nats"][0] = None
    v = p1.verdict(read, rank, ctrl, report, declared_read=True, scale="pod")
    assert v["bars"][1]["clips"] == 499
    v = p1.verdict(*p1_inputs(short=3, n=30), declared_read=False, scale="mac")
    assert v["declared_read"] is False and "SMOKE" in v["note"]


def test_p1_runs_end_to_end_on_a_tiny_tokenizer_with_write_once_lists(tmp_path):
    tokenizer = tiny_tokenizer(codebook=16)
    def records(prefix, n, comp, partition="train"):
        return [dict(sign=f"{prefix}{i}", text=f"{prefix} text {i}", signer_component=comp, partition=partition) for i in range(n)]
    states = {}
    def fetch(batch):
        out = []
        for r in batch:
            states[r["sign"]] = signer_state(abs(hash(r["sign"])) % 1000, frames=20, amplitude=.4)
            out.append(states[r["sign"]])
        return out
    fit, stop, read = records("f", 12, "c02"), records("p", 4, "c04"), records("r", 6, "c01", "val")
    for name, recs in (("fit", fit), ("stop", stop), ("read", read)):
        p1.write_code_list(tmp_path / f"codes-{name}.npz", name, recs, tokenizer, fetch)
    with pytest.raises(SystemExit, match="write-once"):
        p1.write_code_list(tmp_path / "codes-fit.npz", "fit", fit, tokenizer, fetch)
    lists = {n: p1.read_code_list(tmp_path / f"codes-{n}.npz") for n in ("fit", "stop", "read")}
    assert lists["fit"]["rows"][0].shape == (20, len(tokenizer.parts))

    rng = np.random.default_rng(0)
    p1.embed_all(tmp_path, [t for x in lists.values() for t in x["texts"]], tokens_of=lambda g: [rng.normal(size=(3, 768)).astype(np.float32) for _ in g])
    tokens = p1.Tokens(tmp_path)
    assert tokens("").shape == (3, 768)
    import argparse
    args = argparse.Namespace(seed=0, codebook=16, steps=6, batch=4, every=3, patience=2)
    p1.run_fit(tmp_path, "head", lists["fit"], lists["stop"], tokens, args)
    with pytest.raises(SystemExit, match="write-once"):
        p1.run_fit(tmp_path, "head", lists["fit"], lists["stop"], tokens, args)
    bad = dict(lists["fit"], partitions=["val"] * 12)
    with pytest.raises(SystemExit, match="not train"):
        p1.run_fit(tmp_path, "head2", bad, lists["stop"], tokens, args)
    head = p1.load_head(tmp_path, "head", 16)
    assert not head.training and json.loads((tmp_path / "head.json").read_text())["parameters"] > 0
    report = p1.read_report(head, lists["read"], tokens, p1.PARTS, batch=4)
    assert report["gain"]["n"] == 6 and set(report["by_signer_group"]) == {"c01"}
    gain = np.array(report["gain_nats"])
    assert sum(report["breakdown_nats_per_clip"]["by_part"].values()) == pytest.approx(gain.mean(), abs=.01)
    ranked = p1.rank_report(head, lists["read"], tokens, indices=[0, 1], candidates=6, batch=4)
    assert ranked["summary"]["clips"] == 2 and 1 <= min(ranked["ranks"]) and max(ranked["ranks"]) <= 6
    sampled = p1.sample_report(head, tokenizer, lists["read"], tokens, [states[s] for s in lists["read"]["signs"]], batch=4)
    assert len(sampled["per_clip_rho_mean"]) == 6 and 0 <= sampled["code_match"] <= 1 and sampled["latent_distance"]["mean"] > 0
    stop_read = p1.stop_reader(head, lists["stop"], tokens)
    assert stop_read() > 0


# ---- P2's damage -----------------------------------------------------------------------------------------------------

def grid(S=48, Q=4, P=4, K=32, seed=0):
    return np.random.default_rng(seed).integers(0, K, (S, Q, P))


def test_iid_noise_replaces_about_its_share_from_the_marginal_only():
    g = grid(S=2000)
    marg = np.zeros((4, 4, 32))
    marg[..., 5] = 1.                                            # every replacement is code 5
    out = p2.iid_noise(g, .25, marg, np.random.default_rng(0))
    changed = out != g
    assert .22 < changed.mean() < .27 and (out[changed] == 5).all() and out.shape == g.shape


def test_block_noise_replaces_exactly_its_share_of_blocks_per_part_from_another_clip():
    grids = [np.full((48, 4, 4), 0), np.full((48, 4, 4), 1), np.full((6, 4, 4), 2)]
    out = p2.block_noise(grids[0], .25, 0, grids, np.random.default_rng(0))
    for p in range(4):
        blocks = out[:, 0, p].reshape(8, 6)
        whole = (blocks != 0).all(1)
        assert whole.sum() == 2 and ((blocks == 0).all(1) | whole).all()   # 25% of 8 blocks, whole blocks only
    assert 0 not in np.unique(out[out != 0])                       # never the clip itself
    tiny = p2.block_noise(np.zeros((3, 4, 4), np.int64), .5, 0, [np.zeros((3, 4, 4), np.int64), np.ones((1, 4, 4), np.int64)], np.random.default_rng(0))
    assert tiny.shape == (3, 4, 4)                                  # a donor shorter than the block and a clip under one block


def test_a_lone_clip_cannot_borrow_from_itself():
    with pytest.raises(ValueError):
        p2.block_noise(grid(), .5, 0, [grid()], np.random.default_rng(0))


def test_jitter_is_monotone_within_the_declared_offset_with_fixed_ends():
    for seed in range(5):
        idx = p2.jitter_index(200, np.random.default_rng(seed))
        assert idx[0] == 0 and idx[-1] == 199 and (np.diff(idx) >= 0).all()
        assert np.abs(idx - np.arange(200)).max() <= p2.JITTER_STEPS and (idx != np.arange(200)).any()
    assert p2.jitter_index(2, np.random.default_rng(0)).tolist() == [0, 1]
    for seed in range(5):                                       # offsets wider than the knot spacing stay monotone
        assert (np.diff(p2.jitter_index(200, np.random.default_rng(seed), offset=40, every=10)) >= 0).all()


def test_keyframes_keep_key_steps_and_interpolate_between():
    z = {"a": torch.arange(9.).pow(2)[None, :, None]}
    out = p2.key_latents(z, 4)["a"][0, :, 0]
    assert out[[0, 4, 8]].tolist() == [0., 16., 64.] and out[2].item() == pytest.approx(8.)
    end = p2.key_latents({"a": torch.arange(10.)[None, :, None]}, 4)["a"][0, :, 0]
    assert end[-1].item() == 9.                                     # the last step is always a key


def test_damage_seeds_depend_only_on_clip_and_form():
    a = p2.rng_for(3, "iid25").integers(0, 10 ** 9)
    assert a == p2.rng_for(3, "iid25").integers(0, 10 ** 9)
    assert a != p2.rng_for(4, "iid25").integers(0, 10 ** 9) and a != p2.rng_for(3, "iid50").integers(0, 10 ** 9)


def test_decoding_the_clean_grid_is_the_stages_1_to_4_round_trip():
    tokenizer = tiny_tokenizer(stages=8)
    state = signer_state(1, frames=16, amplitude=.4)
    g = p2.code_grid(tokenizer, state)
    assert g.shape == (16, 4, len(tokenizer.parts))
    ours = p2.decode_grid(tokenizer, state, g)
    theirs = s0.truncated_round_trip(tokenizer, state, 4)
    for name in ("body", "left_hand", "right_hand"):
        np.testing.assert_allclose(ours.channels[name].values, theirs.channels[name].values, atol=1e-6)
    changed = p2.decode_grid(tokenizer, state, p2.iid_noise(g, .5, p2.marginals_of([g], 512), np.random.default_rng(0)))
    assert not np.allclose(changed.channels["body"].values, theirs.channels["body"].values)


def test_every_declared_form_builds_on_a_tiny_list():
    tokenizer = tiny_tokenizer(stages=8)
    refs = [signer_state(s, frames=20, amplitude=.4) for s in range(3)]
    context = dict(refs=refs, tokenizer=tokenizer)
    for name in p2.FORMS:
        states = p2.build_form(name, context)
        assert len(states) == 3 and all(len(s.timestamps) == 20 for s in states)
    with pytest.raises(SystemExit, match="unknown"):
        p2.build_form("iidx", context)


# ---- verdicts on form reads ------------------------------------------------------------------------------------------

REFS = [f"the quick brown fox number {w} jumps over the lazy dog" for w in range(60)]


def form_read(name, match, *, nll=1., distance=.1, declared=True, sha="i", speed=1.):
    """A form's read where a fraction ``match`` of hypotheses equal their references; the rest are one common word."""
    k = int(round(match * len(REFS)))
    hyps = REFS[:k] + ["hello"] * (len(REFS) - k)
    return dict(form=name, signs=[f"s{i}" for i in range(len(REFS))], instrument=dict(sha256=sha), declared_read=declared, list=dict(list="x"),
                references=REFS, hypotheses=hyps, nll=[[nll + .001 * (i % 7)] for i in range(len(REFS))], latent_distance=[distance] * len(REFS),
                seconds_over_reference=[1.] * len(REFS), motion=dict(wrist_speed=[speed] * len(REFS), wrist_spread=[speed] * len(REFS)))


def p2_reads(**over):
    table = dict(real=(1., .5), blank=(0., 2.), clean=(.8, 1.2), iid10=(.6, 1.4), iid25=(.5, 1.5), iid50=(.2, 1.7), block25=(.6, 1.4),
                 block50=(.3, 1.6), jitter=(.8, 1.25), key12=(.76, 1.3), key6=(.5, 1.5))
    table.update(over)
    return {n: form_read(n, m, nll=l, distance=.1 + .05 * i) for i, (n, (m, l)) in enumerate(table.items())}


def test_p2_passes_and_each_broken_bar_fails_it():
    v = p2.make_verdict(p2_reads())
    assert v["passed"] and v["option2_kept"] and v["declared_read"]
    v = p2.make_verdict(p2_reads(block25=(.01, 1.4)))
    assert not v["passed"] and not v["bars"][0]["passed"]
    assert not p2.make_verdict(p2_reads(block25=(0., 2.)))["bars"][1]["passed"]         # at blank: not above it
    v = p2.make_verdict(p2_reads(jitter=(.2, 1.25)))
    assert not v["passed"] and not v["bars"][2]["passed"]
    v = p2.make_verdict(p2_reads(key12=(.2, 1.3)))
    assert v["passed"] and not v["option2_kept"]                                        # option 2 drops; not a stop


def test_p2s_bars_are_the_declared_numbers():
    assert (p2.COMPOSITION_D20_BLEU, p2.JITTER_BAR, p2.KEY_BAR, p2.EVIDENCE_BLEU, p2.EVIDENCE_DNLL) == (2.328, .3, .9, .2, .1)
    assert set(p2.FORMS) == {"real", "blank", "clean", "iid10", "iid25", "iid50", "block25", "block50", "jitter", "key12", "key6"}
    assert p2.BLOCK_STEPS == 6 and p2.JITTER_STEPS == 6            # 0.25 s at 24 steps/s


def test_the_curve_interpolates_and_says_when_it_is_read_beyond_its_ends():
    points = [dict(latent_distance=.2, bleu=3.), dict(latent_distance=.4, bleu=2.), dict(latent_distance=.3, bleu=2.5)]
    assert p2.curve_at(points, .25, "bleu") == dict(value=2.75, beyond_curve=False, distance_range=[.2, .4])
    out = p2.curve_at(points, .9, "bleu")
    assert out["value"] == 2. and out["beyond_curve"]


def test_the_evidence_test_needs_all_three_numbers():
    curve = [dict(latent_distance=.2, bleu=3., delta_nll=.02), dict(latent_distance=.4, bleu=2., delta_nll=-.2)]
    signs = ["a", "b", "c"]
    b0 = dict(signs=signs, forms=dict(w14b_nogains_ref=dict(bleu=dict(bleu=2.), delta_nll=dict(mean=-.3), rho_faith=dict(mean=dict(mean=.98)),
                                                             per_clip=dict(rho_mean=[.98, .97, .99]))))
    p = dict(signs=signs, latent_distance=dict(mean=.3), per_clip_rho_mean=[.5, .6, .55], rho_mean=.55)
    assert p2.evidence(curve, p, b0)["passed"]
    assert not p2.evidence(curve, dict(p, latent_distance=dict(mean=.4)), b0)["bars"][0]["passed"]     # BLEU only equal
    assert not p2.evidence(curve, dict(p, per_clip_rho_mean=[.99, .98, 1.]), b0)["bars"][2]["passed"]
    worse = json.loads(json.dumps(b0)); worse["forms"]["w14b_nogains_ref"]["delta_nll"]["mean"] = -.1
    assert not p2.evidence(curve, p, worse)["bars"][1]["passed"]
    with pytest.raises(SystemExit, match="same clips"):
        p2.evidence(curve, dict(p, signs=["x"]), b0)


def test_form_reads_refuse_mixing_lists_instruments_or_a_smoke(tmp_path):
    (tmp_path / "form-real.json").write_text(json.dumps(form_read("real", 1.)))
    (tmp_path / "form-blank.json").write_text(json.dumps(form_read("blank", 0., sha="other")))
    with pytest.raises(SystemExit, match="another instrument"):
        L.load_form_reads(tmp_path, ("real", "blank"))
    (tmp_path / "form-blank.json").write_text(json.dumps(form_read("blank", 0., declared=False)))
    with pytest.raises(SystemExit, match="smoke"):
        L.load_form_reads(tmp_path, ("real", "blank"))
    (tmp_path / "form-blank.json").write_text(json.dumps(form_read("real", 0.)))
    with pytest.raises(SystemExit, match="holds form"):
        L.load_form_reads(tmp_path, ("real", "blank"))
    with pytest.raises(SystemExit, match="missing"):
        L.load_form_reads(tmp_path, ("real", "absent"))


def test_p3_bars_and_its_lengths():
    def reads(pred_match=.9, pred_nll=1.01):
        return dict(real=form_read("real", 1., nll=1.), blank=form_read("blank", 0., nll=2.), predicted=form_read("predicted", pred_match, nll=pred_nll))
    signs = [f"s{i}" for i in range(len(REFS))]
    lengths = dict(signs=signs, pooled_ratio=1.08, median_ratio=1.1, median_abs_error=.25, model={})
    v = p3.make_verdict(reads(), lengths)
    assert v["passed"] and v["blind"] is False
    assert not p3.make_verdict(reads(), dict(lengths, pooled_ratio=1.21))["passed"]
    assert not p3.make_verdict(reads(), dict(lengths, pooled_ratio=.94))["passed"]
    v = p3.make_verdict(reads(pred_nll=1.05), lengths)                   # ΔNLL 0.05 below real
    assert not v["passed"] and not v["bars"][1]["passed"]
    with pytest.raises(SystemExit, match="not this read"):
        p3.make_verdict(reads(), dict(lengths, signs=["x"]))


def test_the_balanced_length_fit_gives_each_signer_equal_weight_and_the_default_is_unchanged():
    from signtranslator.generation.composition_v2 import fit_duration_model
    recs = []
    for i in range(900):                                                  # a slow signer, many clips
        n = 2 + i % 20
        recs.append(dict(corpus=H2S, partition="train", signer_component="c02", text=" ".join(f"w{j}" for j in range(n)), continuous=dict(seconds=.4 * n)))
    for i in range(100):                                                  # a fast signer, few clips
        n = 2 + i % 20
        recs.append(dict(corpus=H2S, partition="train", signer_component="c04", text=" ".join(f"w{j}" for j in range(n)), continuous=dict(seconds=.2 * n)))
    recs.append(dict(corpus=H2S, partition="val", signer_component="c01", text="a b", continuous=dict(seconds=99.)))
    manifest = dict(records=recs)
    plain = fit_duration_model(manifest)
    x = np.array([2 + i % 20 for i in range(1000)], float); y = np.r_[.4 * x[:900], .2 * x[900:]]
    slope, intercept = np.polyfit(x, y, 1)
    assert plain["slope"] == pytest.approx(slope, abs=1e-3) and plain["intercept"] == pytest.approx(intercept, abs=1e-3)
    assert "signer_balanced" not in plain
    balanced = fit_duration_model(manifest, components=("c02", "c04"), signer_balanced=True)
    assert balanced["slope"] == pytest.approx(.3, abs=1e-3) and balanced["components"] == dict(c02=900, c04=100)
    only = fit_duration_model(manifest, components=("c02",))
    assert only["slope"] == pytest.approx(.4, abs=1e-3) and only["records"] == 900
    with pytest.raises(ValueError, match="two signer"):
        fit_duration_model(manifest, components=("c02",), signer_balanced=True)


def test_a_resampled_clip_lasts_its_predicted_length():
    state = signer_state(0, frames=48, amplitude=.4)
    longer = p3.resampled(state, 3.)
    assert len(longer.timestamps) / 24. == pytest.approx(3., abs=.05)
    shorter = p3.resampled(state, 1.)
    assert len(shorter.timestamps) / 24. == pytest.approx(1., abs=.05)


def test_amplitude_scales_rotation_excursions_and_keeps_timing_and_validity():
    state = signer_state(0, frames=24, amplitude=.4)
    small_, big = p4.scaled(state, .8), p4.scaled(state, 1.27)
    assert np.array_equal(small_.timestamps, state.timestamps)
    for name in p4.CHANNELS:
        assert np.array_equal(small_.channels[name].valid, state.channels[name].valid)
        def spread(s):
            return float(np.std(s.channels[name].values[state.channels[name].valid], axis=0).mean())
        assert spread(small_) < spread(state) < spread(big)
    assert p4.CHANNELS == ("body", "left_hand", "right_hand")
    np.testing.assert_array_equal(p4.scaled(state, 1.).channels["body"].values, state.channels["body"].values)


def test_p4_binds_only_on_a_resolved_preference_for_the_smaller_motion():
    def reads(low_nll):
        out = {n: form_read(n, .9, nll=1.) for n in p4.FORMS}
        out["blank"] = form_read("blank", 0., nll=2.)
        out["amp0.8"] = form_read("amp0.8", .9, nll=low_nll)
        return out
    v = p4.make_verdict(reads(.9))
    assert v["motion_bands_bind"] and v["stop"] is False and "bind" in v["consequence"]
    v = p4.make_verdict(reads(1.))
    assert not v["motion_bands_bind"] and v["stop"] is False
    assert not p4.make_verdict(reads(1.1))["motion_bands_bind"]
    assert p4.AMPLITUDES == (.8, .9, 1.1, 1.27)
    noisy = reads(1.)
    spread = np.random.default_rng(0).normal(0, .5, len(REFS))
    noisy["amp0.8"]["nll"] = [[noisy["real"]["nll"][i][0] - .01 + spread[i]] for i in range(len(REFS))]
    v = p4.make_verdict(noisy)                                  # a small unresolved preference does not bind
    assert v["bars"][0]["value"] < 0 and not v["motion_bands_bind"]


# ---- the read harness, end to end --------------------------------------------------------------------------------

class FakeReader:
    info = dict(run="fake", checkpoint_key="k", sha256="f" * 64, kind="fake")

    def read(self, states, texts, candidates, group=2):
        size = [float(np.abs(s.channels["body"].values).mean()) for s in states]
        return dict(hypotheses=[t if a > 0 else "you" for t, a in zip(texts, size)],
                    nll=np.array([[2. - a] * candidates.shape[1] for a in size]))


def harness(monkeypatch, *, declared=True):
    tokenizer = tiny_tokenizer(stages=8)
    records = [dict(sign=f"s{i}", text=f"word{i} goes here", signer_component="c01", partition="val", continuous=dict(seconds=1.)) for i in range(4)]
    refs = [signer_state(i, frames=24, amplitude=.4) for i in range(4)]
    monkeypatch.setattr(L, "open_read_list", lambda store, smoke: (records, declared and not smoke, dict(list="t", clips=4)))
    monkeypatch.setattr(L, "load_states", lambda store, recs: refs)
    monkeypatch.setattr(L, "load_tokenizer", lambda store: tokenizer)
    monkeypatch.setattr(L, "make_reader", lambda store, **k: FakeReader())
    return records, refs


def test_the_amplitude_probe_runs_read_then_verdict_and_is_write_once(tmp_path, monkeypatch):
    harness(monkeypatch)
    w = tmp_path / "w"
    store = ["--store", f"dir:{tmp_path / 's'}"]
    assert p4.main(["read", *store, "--out-dir", str(w), "--forms", "real,blank"]) == 0
    assert p4.main(["read", *store, "--out-dir", str(w), "--forms", "amp0.8,amp0.9,amp1.1,amp1.27"]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        p4.main(["read", *store, "--out-dir", str(w), "--forms", "real"])
    with pytest.raises(SystemExit, match="missing"):
        p3.main(["verdict", *store, "--out-dir", str(w), "--output-file", str(w / "x.json")])  # P3's forms are not here
    out = w / "verdict.json"
    assert p4.main(["verdict", *store, "--out-dir", str(w), "--output-file", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["kind"] == "s2-p4-verdict" and report["declared_read"] and "motion_bands_bind" in report
    assert report["forms"]["amp0.8"]["wrist_speed_over_real"] < 1 < report["forms"]["amp1.27"]["wrist_speed_over_real"]
    with pytest.raises(SystemExit, match="write-once"):
        p4.main(["verdict", *store, "--out-dir", str(w), "--output-file", str(out)])


def test_the_noise_curve_runs_every_form_and_a_form_read_alone_equals_it_read_with_others(tmp_path, monkeypatch):
    harness(monkeypatch)
    store = ["--store", f"dir:{tmp_path / 's'}"]
    w = tmp_path / "w"
    assert p2.main(["read", *store, "--out-dir", str(w), "--forms", ",".join(p2.FORMS)]) == 0
    out = w / "v.json"
    assert p2.main(["verdict", *store, "--out-dir", str(w), "--output-file", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["forms"]) == set(p2.FORMS) and len(report["curve"]) == 9
    assert report["forms"]["clean"]["latent_distance"]["mean"] < report["forms"]["iid50"]["latent_distance"]["mean"]
    w2 = tmp_path / "w2"
    assert p2.main(["read", *store, "--out-dir", str(w2), "--forms", "block25"]) == 0
    a, b = json.loads((w / "form-block25.json").read_text()), json.loads((w2 / "form-block25.json").read_text())
    assert a["latent_distance"] == b["latent_distance"] and a["nll"] == b["nll"]
    with pytest.raises(SystemExit, match="unknown forms"):
        p2.main(["read", *store, "--out-dir", str(w2), "--forms", "iid99"])


def test_the_length_probe_reads_its_lengths_and_verdict(tmp_path, monkeypatch):
    records, _ = harness(monkeypatch)
    w = tmp_path / "w"
    w.mkdir()
    (w / "lengths.json").write_text(json.dumps(dict(signs=[r["sign"] for r in records], predicted_seconds=[1.5, 1.5, 2., 1.2], real_seconds=[1.] * 4,
                                                    pooled_ratio=1.05, median_ratio=1., median_abs_error=.1, model=dict())))
    store = ["--store", f"dir:{tmp_path / 's'}"]
    assert p3.main(["read", *store, "--out-dir", str(w), "--forms", "real,blank,predicted"]) == 0
    report = json.loads((w / "form-predicted.json").read_text())
    assert report["seconds_over_reference"][0] == pytest.approx(1.5, abs=.05)
    assert p3.main(["verdict", *store, "--out-dir", str(w), "--output-file", str(w / "v.json")]) == 0
    assert json.loads((w / "v.json").read_text())["kind"] == "s2-p3-verdict"


def test_the_reader_reads_as_s0_does():
    from tests.test_s0s1_scripts import tiny_translator
    torch.manual_seed(0)
    model = tiny_translator().eval()
    states = [signer_state(i, frames=16, amplitude=.4) for i in range(2)]
    from score_back_translation import translate
    reader = L.Reader(model, "keypoints", dict(), "cpu")
    cands = np.array([[0], [1]])
    raw = reader.read(states, ["dog", "cat"], cands)
    assert raw["hypotheses"] == translate(model, "keypoints", states)
    np.testing.assert_allclose(raw["nll"], s0.candidate_nll(model, s0.translator_items(model, "keypoints", states), ["dog", "cat"], cands, device="cpu"))


def test_motion_statistics_compare_the_frames_real_signing_was_tracked_on():
    ref = signer_state(0, frames=24, amplitude=.4)
    seen = ref.channels["right_hand"].valid.copy()
    seen[8:16] = False
    gappy = replace(ref, channels={**ref.channels, "right_hand": replace(ref.channels["right_hand"], valid=seen, values=np.where(seen[..., None], ref.channels["right_hand"].values, 0.),
                                                                          observed=ref.channels["right_hand"].observed & seen, confidence=ref.channels["right_hand"].confidence * seen)})
    masked = L.observed_like(ref, gappy)
    assert not masked.channels["right_hand"].valid[8:16].any() and masked.channels["right_hand"].valid[:8].all()
    np.testing.assert_array_equal(masked.channels["body"].values, np.where(gappy.channels["body"].valid[..., None], ref.channels["body"].values, 0.))
    other = signer_state(1, frames=20, amplitude=.4)
    assert L.observed_like(other, gappy) is other                # another length keeps its own validity
    stats = L.motion_statistics([ref], [gappy])
    assert stats["wrist_speed"][0] is not None


def test_a_round_trips_neutral_fill_does_not_count_as_wrist_speed():
    ref = signer_state(0, frames=48, amplitude=.1)
    body = ref.channels["body"]
    body = replace(body, values=np.broadcast_to(body.values[:1], body.values.shape).copy())    # still, so only the gap moves
    ref = replace(ref, channels={**ref.channels, "body": body})
    seen = body.valid.copy()
    seen[10:14] = False
    gappy = replace(ref, channels={**ref.channels, "body": replace(body, valid=seen, values=np.where(seen[..., None], body.values, 0.),
                                                                   observed=body.observed & seen, confidence=body.confidence * seen)})
    values = body.values.copy()
    values[10:14] = values[10:14][..., [3, 4, 5, 0, 1, 2]]      # a form that jumps elsewhere where real was not tracked
    form = replace(ref, channels={**ref.channels, "body": replace(body, values=values)})
    filled = L.motion_statistics([form], [form])["wrist_speed"][0]
    masked = L.motion_statistics([form], [gappy])["wrist_speed"][0]
    assert masked < .2 * filled


def test_the_slim_instrument_is_fetched_once_checked_by_the_originals_hash_and_never_read_for_another(tmp_path):
    import hashlib
    import io
    def payload(tag):
        buffer = io.BytesIO()
        torch.save(dict(kind="keypoints_to_text_t5", config=dict(tag=tag), ema=dict(w=torch.ones(2)), model=dict(w=torch.zeros(2)), optimizer=dict(big=torch.zeros(1000))), buffer)
        return buffer.getvalue()
    raw = payload("a")
    sha = hashlib.sha256(raw).hexdigest()
    store = SpyStore({"k": raw})
    first, digest = L.instrument_payload(store, "k", sha, str(tmp_path))
    assert digest == sha and store.read == ["k"] and len(first) < len(raw)
    saved = torch.load(io.BytesIO(first), weights_only=False)
    assert "optimizer" not in saved and "model" not in saved and saved["ema"]["w"].tolist() == [1., 1.]
    second, _ = L.instrument_payload(store, "k", sha, str(tmp_path))
    assert store.read == ["k"] and second == first               # the second process does not fetch
    with pytest.raises(SystemExit, match="SHA-256"):
        L.instrument_payload(SpyStore({"k": payload("b")}), "k", "0" * 64, str(tmp_path))
    plain, digest = L.instrument_payload(SpyStore({"k": raw}), "k", sha, None)
    assert plain == raw and digest == sha


def test_smoke_reads_are_train_clips_and_say_so(monkeypatch):
    manifest = train_manifest()
    monkeypatch.setattr(L, "load_manifest_at", lambda store, key, sha: manifest)
    records, declared, described = L.open_read_list(None, 5)
    assert not declared and len(records) == 5 and {r["signer_component"] for r in records} == {"c02"}
    assert {r["partition"] for r in records} == {"train"} and described["list"].startswith("smoke")
    assert [r["sign"] for r in L.smoke_records(manifest, 5)] == [r["sign"] for r in records]


# ---- the S1 and S0 additions ----------------------------------------------------------------------------------------

def test_s1_accepts_d20_only_and_keeps_its_default_clip_draw(tmp_path):
    with pytest.raises(SystemExit):
        s1.main(["--store", f"dir:{tmp_path}", "--sentence-set", "c20", "--output-file", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        s0.main(["--store", f"dir:{tmp_path}", "--sentence-set", "c20", "--output-file", str(tmp_path / "x")])


@pytest.mark.parametrize("extra", [["--stored", "a"], ["--regenerate", "a"], ["--merge-from", "a"]])
def test_b0_refuses_anything_that_reads_the_1000(tmp_path, extra):
    with pytest.raises(SystemExit, match="read the 1,000"):
        s0.main(["--store", f"dir:{tmp_path}", "--sentence-set", "d20", "--forms", "real,blank", *extra, "--output-file", str(tmp_path / "x")])


def test_b0_pilot_and_lengths_go_together_on_a_d20_read(tmp_path):
    for extra in (["--sentence-set", "d20", "--b0-pilot", "p"], ["--sentence-set", "d20", "--b0-lengths", "l"], ["--b0-pilot", "p", "--b0-lengths", "l"]):
        with pytest.raises(SystemExit, match="go together"):
            s0.main(["--store", f"dir:{tmp_path}", "--forms", "real,blank", *extra, "--output-file", str(tmp_path / "x")])


def test_b0_forms_regenerate_with_the_pilots_settings_at_the_given_lengths(monkeypatch):
    import evaluate_text_to_motion as etm
    calls, loads = [], []

    def loader(store, key, sha, device, temperature=1., decoder_gains=True, **k):
        loads.append(dict(key=key, sha=sha, temperature=temperature, gains=decoder_gains))

        def sample():
            pass
        sample.batch = lambda texts, fps, seconds, guidance, seed: (calls.append((len(texts), list(seconds), guidance, seed)),
                                                                     [signer_state(0, frames=12) for _ in texts])[1]
        return "w", ("body",), sample
    monkeypatch.setattr(etm, "load_generator", loader)
    pilot = dict(checkpoint_key="c.pt", part_temperature=dict(body=1.1), guidance=1.5)
    texts = [f"t{i}" for i in range(40)]
    seconds = [1. + i for i in range(40)]
    out = s0.b0_form(None, pilot, "sha", texts, seconds, False, "cpu")
    assert len(out) == 40 and loads == [dict(key="c.pt", sha="sha", temperature=dict(body=1.1), gains=False)]
    assert [(c[0], c[2], c[3]) for c in calls] == [(32, 1.5, 1000), (8, 1.5, 1032)] and calls[1][1] == seconds[32:]
    assert set(s0.B0_FORMS) == {"w14b_gains_ref", "w14b_nogains_ref", "w14b_gains_pred", "w14b_nogains_pred"}


def test_b0_runs_end_to_end_with_every_form_and_keeps_reading_through_the_instrument(tmp_path, monkeypatch):
    """The whole main: real, blank and the four W14b forms, each read by the same instrument (the first smoke of this code
    lost the translator's kind to a loop variable, which only an end-to-end read shows)."""
    import hashlib
    import diagnose_tokenizer as dt
    import evaluate_text_to_motion as etm
    import score_back_translation as sbt
    from signtranslator.data_engineering import corpus_manifest as cm
    store = DirectoryStore(tmp_path / "store")
    store.put(L.EVAL_MIX, b"{}")
    pilot = dict(checkpoint_key="training-runs/w/checkpoints/s.pt", part_temperature=dict(body=1.), guidance=1.5)
    store.put("p.json", json.dumps(pilot).encode())
    store.put("training-runs/w/run.json", json.dumps(dict(manifest_sha256="m", manifest_key="k")).encode())
    store.put(f"training-runs/{s0.TRANSLATOR_RUN}/best.json", json.dumps(dict(checkpoint_key="t.pt")).encode())
    store.put("t.pt", b"translator")
    store.put(s0.TOKENIZER, b"tokenizer")
    tokenizer = tiny_tokenizer(stages=8)
    records = [dict(sign=f"s{i}", text=f"word{i} goes here", signer_component="c02", partition="train",
                    continuous=dict(state_key=f"k{i}", state_sha256="0" * 64)) for i in range(4)]
    monkeypatch.setattr(cm, "load_manifest", lambda path, expected_sha256=None: dict(records=[]))
    monkeypatch.setattr(L, "smoke_records", lambda manifest, n: records[:n])
    monkeypatch.setattr(etm, "load_state", lambda s, k, sha: signer_state(int(k[1:]), frames=20, amplitude=.4))
    monkeypatch.setattr(etm, "accepted_decoder_gains", lambda s, k: dict(body=1.17))
    monkeypatch.setattr(dt, "load_tokenizer", lambda s, k: (tiny_tokenizer(stages=8), {}))
    monkeypatch.setattr(sbt, "load_translator", lambda s, k, sha: (tiny_translator_stub(), "keypoints"))
    kinds = []
    monkeypatch.setattr(sbt, "translate", lambda model, kind, states, **k: (kinds.append(kind), ["word goes here"] * len(states))[1])
    monkeypatch.setattr(s0, "translator_items", lambda model, kind, states: (kinds.append(kind), [None] * len(states))[1])
    monkeypatch.setattr(s0, "candidate_nll", lambda model, items, texts, cands, device, group: np.arange(cands.size, dtype=float).reshape(cands.shape) / 10)
    seen = []

    def loader(s, key, sha, device, temperature=1., decoder_gains=True, **k):
        def sample():
            pass

        def batch(texts, fps, seconds, guidance, seed):
            seen.append((decoder_gains, tuple(seconds), sha))
            return [signer_state(i, frames=int(round(t * 24)), amplitude=.4) for i, t in enumerate(seconds)]
        sample.batch = batch
        return "w", ("body",), sample
    monkeypatch.setattr(etm, "load_generator", loader)
    lengths = tmp_path / "lengths.json"
    lengths.write_text(json.dumps(dict(signs=[r["sign"] for r in records], predicted_seconds=[1., 1.5, 1., .5], pooled_ratio=1.05)))
    out = tmp_path / "b0.json"
    argv = ["--store", f"dir:{tmp_path / 'store'}", "--smoke-train", "4", "--forms", "real,blank", "--device", "cpu", "--candidates", "3",
            "--rho-references", "1", "--no-rho-parts", "--tokenizer-sha256", hashlib.sha256(b"tokenizer").hexdigest(), "--b0-pilot", "p.json",
            "--b0-lengths", str(lengths), "--output-file", str(out)]
    assert s0.main(argv) == 0
    report = json.loads(out.read_text())
    assert set(report["forms"]) == {"real", "blank"} | set(s0.B0_FORMS) and set(kinds) == {"keypoints"}
    assert report["b0"]["declared_read"] is False and report["sample"]["rule"].startswith("c02")
    rounded = {(g, tuple(round(x, 6) for x in s)) for g, s, _ in seen}
    ref = tuple([round(20 / 24., 6)] * 4)
    assert rounded == {(True, (1., 1.5, 1., .5)), (False, (1., 1.5, 1., .5)), (True, ref), (False, ref)}
    assert {x for _, _, x in seen} == {"m"}
    assert len(report["forms"]["w14b_nogains_pred"]["per_clip"]["rho_mean"]) == 4
    with pytest.raises(SystemExit, match="write-once"):
        s0.main(argv)
    other = json.loads(lengths.read_text())
    other["signs"] = ["x"] * 4
    lengths.write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="another list"):
        s0.main(argv[:-1] + [str(tmp_path / "b1.json")])


def tiny_translator_stub():
    class Stub:
        def to(self, device):
            return self

        def eval(self):
            return self
    return Stub()


def test_the_mac_runner_stays_within_the_limits_and_reads_no_other_list():
    text = (ROOT / "scripts/s2_probes_mac.sh").read_text()
    assert "OMP_NUM_THREADS=1" in text and "nice" in text and "--sentence-set d20" in text
    for word in ("c20", "C20", "val1000", "eval1000", "--output-key"):
        assert word not in text
    for line in text.splitlines():
        if line.strip().startswith("forms scripts/"):
            groups = [x for x in line.split()[4:] if x != ";;"]
            assert 1 <= len(groups) <= 5                        # at most 5 background processes after the first form
    import subprocess
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts/s2_probes_mac.sh")]).returncode == 0


def test_the_b0_job_reads_d20_smokes_on_train_first_and_names_no_other_list():
    text = (ROOT / "runpod/fleet/s2_b0.sh").read_text()
    assert "--sentence-set d20" in text and "--smoke-train 8" in text
    assert text.index("--smoke-train 8") < text.index("--sentence-set d20")
    for word in ("c20", "C20", "val1000", "eval1000"):
        assert word not in text
    import subprocess
    assert subprocess.run(["bash", "-n", str(ROOT / "runpod/fleet/s2_b0.sh")]).returncode == 0
