"""Stage 5: the evaluation harness running the document's acceptance criteria.

Produces, for a trained recogniser: WER/CER with an S/D/I breakdown, timestamp
error, ECE, revision rate, explicit streaming configuration with median/p95
emission latency, and a transcript-only / acoustic-only / fused ablation across
clean, noisy, accented, code-switched and long-form speech.

Every perturbation level here was **measured** before use (see the sweep recorded
in ``evaluation.STANDARD_CONDITIONS``); levels picked by intuition produced a
vacuous evaluation.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech import (
    LogMelSpectrogram, ctc_greedy_path, collapse, ctc_prefix_beam_search,
    ctc_forced_alignment, token_timings, FrameTimeMapper, Lattice,
    StreamingDecoder, LatencyModel, measure_emission_latency,
    expected_calibration_error, N_MELS, HOP_LENGTH, N_FFT,
)
from signtranslator.speech.evaluation import (
    Condition, STANDARD_CONDITIONS, characterise_condition, word_error_rate,
    character_error_rate, timestamp_error, ArmResult, EvaluationReport,
)
from signtranslator.models import SpeechRecognizer

SR = 16000
WORD_S, GAP_S = 0.20, 0.08
PRIMARY = {120.0: 1, 210.0: 2, 320.0: 3}
SECONDARY = {160.0: 4, 260.0: 5, 400.0: 6}
ALL = {**PRIMARY, **SECONDARY}
SPELLING = {1: "cat", 2: "dog", 3: "bird", 4: "fish", 5: "horse", 6: "mouse"}


def _utterance(f0s, noise=0.0, pitch=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts, spans, cursor = [], [], 0.0
    for f0 in f0s:
        n = int(WORD_S * SR)
        t = torch.arange(n, dtype=torch.float32) / SR
        x = sum(torch.sin(2 * math.pi * (f0 * pitch) * h * t) / h for h in (1, 2, 3))
        parts.append(x * torch.hann_window(n, periodic=False) * 0.5)
        spans.append((cursor, cursor + WORD_S))
        cursor += WORD_S
        parts.append(torch.zeros(int(GAP_S * SR)))
        cursor += GAP_S
    wav = torch.cat(parts)
    if noise:
        wav = wav + noise * torch.randn(wav.numel(), generator=g)
    return wav, spans


def _make(cond: Condition, fe, seed=0, n_utt=6):
    prim, sec = list(PRIMARY), list(SECONDARY)
    g = torch.Generator().manual_seed(seed)
    feats, targets, spans = [], [], []
    for i in range(n_utt):
        seq = []
        for j in range(cond.words):
            if cond.vocabulary == "mixed":
                src = prim if j % 2 == 0 else sec
            elif cond.vocabulary == "secondary":
                src = sec
            else:
                src = prim
            seq.append(src[int(torch.randint(len(src), (1,), generator=g))])
        wav, sp = _utterance(seq, cond.noise, cond.pitch_scale, seed * 100 + i)
        feats.append(fe(wav).t().unsqueeze(0))
        targets.append([ALL[f] for f in seq])
        spans.append(sp)
    n = min(f.shape[1] for f in feats)
    return torch.cat([f[:, :n] for f in feats], dim=0), targets, spans


@pytest.fixture(scope="module")
def system():
    """Train the reference recogniser on clean primary-vocabulary speech."""
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    feats, targets, _ = _make(Condition("clean", is_baseline=True), fe, seed=0)
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=6, hidden_dim=96,
                           num_layers=2, num_heads=4, subsample=2)
    tgt = torch.tensor(targets)
    lengths = torch.full((feats.shape[0],), tgt.shape[1], dtype=torch.long)
    opt = torch.optim.Adam(rec.parameters(), lr=3e-3)
    for _ in range(320):
        loss = rec.loss(feats, tgt, lengths)
        opt.zero_grad(); loss.backward(); opt.step()
    rec.eval()
    return rec, fe


def _decode(rec, feats):
    with torch.no_grad():
        log_probs = rec(feats)
    return log_probs, [collapse(ctc_greedy_path(log_probs[i]), blank=0)
                       for i in range(feats.shape[0])]


# ---------------------------------------------------------------------------
# Condition characterisation (must precede any conclusion)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def profiles(system):
    rec, fe = system
    out = {}
    for cond in STANDARD_CONDITIONS:
        feats, targets, _ = _make(cond, fe, seed=7)
        _, hyps = _decode(rec, feats)
        out[cond.name] = characterise_condition(cond.name, hyps, targets,
                                                is_baseline=cond.is_baseline)
    return out


def test_baseline_is_at_ceiling_and_is_not_flagged(profiles):
    """Clean speech SHOULD be at ceiling; that is not a defect."""
    clean = profiles["clean"]
    assert clean.accuracy > 0.95
    assert clean.is_informative          # exempt as a baseline


def test_every_perturbed_condition_is_informative(profiles):
    """No condition may be vacuous -- neither ceiling nor total collapse.

    This is the guard the Stage 3 lesson demands: intuition-chosen severities
    gave zero decoded tokens (noise) or zero degradation (pitch).
    """
    degenerate = [name for name, p in profiles.items() if not p.is_informative]
    assert degenerate == [], f"degenerate conditions: {degenerate}"


def test_perturbed_conditions_actually_degrade(profiles):
    """Each perturbation must move accuracy below the clean baseline."""
    clean = profiles["clean"].accuracy
    for name in ("noisy", "accented", "code_switched", "long_form"):
        assert profiles[name].accuracy < clean, f"{name} did not degrade"


# ---------------------------------------------------------------------------
# Core metrics per condition
# ---------------------------------------------------------------------------
def test_wer_breakdown_identifies_the_failure_mode(system, profiles):
    """The S/D/I split must reveal *how* each condition breaks the model."""
    rec, fe = system
    modes = {}
    for cond in STANDARD_CONDITIONS:
        feats, targets, _ = _make(cond, fe, seed=7)
        _, hyps = _decode(rec, feats)
        modes[cond.name] = word_error_rate(hyps, targets)
    assert modes["clean"].error_rate < 0.1
    # Noise removes tokens; the breakdown should show that, not substitutions.
    noisy = modes["noisy"]
    assert noisy.error_rate > modes["clean"].error_rate
    assert noisy.deletions >= noisy.substitutions, noisy.summary()


def test_cer_is_computed_over_a_real_orthography(system):
    rec, fe = system
    feats, targets, _ = _make(Condition("noisy", noise=0.010), fe, seed=7)
    _, hyps = _decode(rec, feats)
    cer = character_error_rate(hyps, targets, SPELLING)
    wer = word_error_rate(hyps, targets)
    assert cer.reference_length > wer.reference_length     # chars > words
    assert cer.error_rate >= 0.0


def test_timestamp_error_is_measured_against_physical_spans(system):
    """Timestamps are scored against where the tones actually sounded."""
    rec, fe = system
    feats, targets, spans = _make(Condition("clean", is_baseline=True), fe, seed=7)
    log_probs, _ = _decode(rec, feats)
    mapper = FrameTimeMapper(hop_length=HOP_LENGTH, sample_rate=SR,
                             n_fft=N_FFT, subsample=rec.subsample)
    errors = []
    for i in range(feats.shape[0]):
        al = ctc_forced_alignment(log_probs[i], targets[i], blank=0)
        timings = token_timings(al, mapper=mapper)
        pred = [(t.start_s, t.end_s) for t in timings]
        errors.append(timestamp_error(pred, spans[i]))
    mean_start = sum(e.mean_start_error_s for e in errors) / len(errors)
    assert mean_start < 0.25, f"timestamps off by {mean_start * 1000:.0f} ms"


def test_ece_and_revision_rate_are_reported(system):
    rec, fe = system
    feats, targets, _ = _make(Condition("noisy", noise=0.010), fe, seed=7)
    log_probs, _ = _decode(rec, feats)

    conf, corr = [], []
    for i in range(feats.shape[0]):
        lat = Lattice.from_nbest(ctc_prefix_beam_search(log_probs[i], beam_width=10))
        for pos, (tok, c) in enumerate(zip(lat.best, lat.confidence())):
            conf.append(float(c))
            corr.append(pos < len(targets[i]) and targets[i][pos] == tok)
    ece = expected_calibration_error(conf, corr, n_bins=10) if conf else 0.0

    dec = StreamingDecoder(beam_width=10, agreement_k=2, stability=2)
    for i in range(0, log_probs.shape[1], 8):
        dec.update(log_probs[0, i:i + 8])
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= dec.stats.revision_rate <= 1.0


# ---------------------------------------------------------------------------
# Input-arm ablation
# ---------------------------------------------------------------------------
def _arm_features(rec, feats, arm: str) -> torch.Tensor:
    """Pooled representation for one input arm.

    transcript_only: the lexical posteriors (what a text-only planner sees).
    acoustic_only:   the encoder's acoustic states.
    fused:           both, concatenated.
    """
    with torch.no_grad():
        hidden = rec.encode(feats)
        posteriors = F.softmax(rec.classifier(hidden), dim=-1)
    lexical = posteriors.mean(dim=1)
    acoustic = hidden.mean(dim=1)
    if arm == "transcript_only":
        return lexical
    if arm == "acoustic_only":
        return acoustic
    if arm == "fused":
        return torch.cat([lexical, acoustic], dim=-1)
    raise ValueError(f"unknown arm {arm}")


def _train_probe(train_x, train_y, n_classes, seed=0):
    """Tiny probe standing in for the sign planner.

    Trained ONCE per arm on clean speech and then evaluated unchanged on every
    condition -- that is the ablation. Retraining per condition would confound
    "how well does this arm survive perturbation" with "how well can a fresh
    probe fit the perturbed data".
    """
    torch.manual_seed(seed)
    probe = torch.nn.Sequential(torch.nn.Linear(train_x.shape[-1], 32),
                                torch.nn.GELU(),
                                torch.nn.Linear(32, n_classes))
    opt = torch.optim.Adam(probe.parameters(), lr=0.02)
    for _ in range(400):
        loss = F.cross_entropy(probe(train_x), train_y)
        opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    return probe


def _probe_accuracy(probe, x, y):
    with torch.no_grad():
        return float((probe(x).argmax(-1) == y).double().mean())


@pytest.fixture(scope="module")
def ablation(system):
    """Downstream sign-plan accuracy per arm, per condition.

    Uses many utterances: an earlier version trained on 6 samples for a 6-way
    task and scored 0.0 on clean -- below chance. With too little data the
    ablation measures probe noise rather than representation quality.
    """
    rec, fe = system
    n_classes = max(ALL.values()) + 1
    train_feats, train_tgt, _ = _make(Condition("clean", is_baseline=True),
                                      fe, seed=0, n_utt=48)
    train_y = torch.tensor([t[0] for t in train_tgt])

    results = {}
    for arm in ("transcript_only", "acoustic_only", "fused"):
        probe = _train_probe(_arm_features(rec, train_feats, arm),
                             train_y, n_classes)
        results[arm] = {}
        for cond in STANDARD_CONDITIONS:
            ef, et, _ = _make(cond, fe, seed=21, n_utt=24)
            ex = _arm_features(rec, ef, arm)
            ey = torch.tensor([t[0] for t in et])
            results[arm][cond.name] = _probe_accuracy(probe, ex, ey)
    return results


def test_probe_learns_the_clean_task_before_it_is_used_for_ablation(ablation):
    """A probe near chance would make every ablation number meaningless."""
    chance = 1.0 / (max(ALL.values()) + 1)
    for arm, per_cond in ablation.items():
        assert per_cond["clean"] > 3 * chance, (
            f"{arm}: probe scored {per_cond['clean']:.3f} on clean, near chance "
            f"({chance:.3f}) -- the ablation would be measuring noise")


def test_all_three_arms_are_evaluated(ablation):
    assert set(ablation) == {"transcript_only", "acoustic_only", "fused"}
    for arm, per_cond in ablation.items():
        assert set(per_cond) == {c.name for c in STANDARD_CONDITIONS}
        for name, acc in per_cond.items():
            assert 0.0 <= acc <= 1.0, (arm, name)


def test_downstream_degrades_under_perturbation(ablation):
    """Sign-plan degradation must be visible, not just recogniser WER."""
    for arm, per_cond in ablation.items():
        clean = per_cond["clean"]
        worst = min(per_cond[n] for n in ("noisy", "accented", "code_switched"))
        assert worst <= clean, f"{arm}: perturbation did not degrade downstream"


def test_arms_are_genuinely_different_representations(system):
    """Guard against an ablation whose arms are secretly identical."""
    rec, fe = system
    feats, _, _ = _make(Condition("clean", is_baseline=True), fe, seed=3)
    lex = _arm_features(rec, feats, "transcript_only")
    ac = _arm_features(rec, feats, "acoustic_only")
    fused = _arm_features(rec, feats, "fused")
    assert lex.shape[-1] != ac.shape[-1] or not torch.allclose(lex, ac)
    assert fused.shape[-1] == lex.shape[-1] + ac.shape[-1]


def test_unknown_arm_is_rejected(system):
    rec, fe = system
    feats, _, _ = _make(Condition("clean"), fe, seed=1)
    with pytest.raises(ValueError):
        _arm_features(rec, feats, "telepathy")


# ---------------------------------------------------------------------------
# The assembled report
# ---------------------------------------------------------------------------
def test_full_report_renders_with_all_required_fields(system, profiles, ablation):
    rec, fe = system
    latency = LatencyModel(chunk_frames=8, right_context=4)
    meas = measure_emission_latency(SR, latency)
    report = EvaluationReport(
        streaming_config=latency.describe(),
        latency_median_s=meas.median_s, latency_p95_s=meas.p95_s)
    report.profiles.extend(profiles.values())

    for cond in STANDARD_CONDITIONS:
        feats, targets, _ = _make(cond, fe, seed=7)
        _, hyps = _decode(rec, feats)
        wer = word_error_rate(hyps, targets)
        cer = character_error_rate(hyps, targets, SPELLING)
        for arm in ("transcript_only", "acoustic_only", "fused"):
            report.add(ArmResult(arm=arm, condition=cond.name, wer=wer, cer=cer,
                                 downstream_accuracy=ablation[arm][cond.name]))

    text = report.summary()
    assert "chunk=8" in text and "right_context=4" in text
    assert "median" in text and "p95" in text
    for cond in STANDARD_CONDITIONS:
        assert cond.name in text
    for arm in ("transcript_only", "acoustic_only", "fused"):
        assert arm in text
    assert "S=" in text and "D=" in text and "I=" in text     # breakdown present
    assert report.degenerate_conditions() == []
    assert len(report.results) == 3 * len(STANDARD_CONDITIONS)
