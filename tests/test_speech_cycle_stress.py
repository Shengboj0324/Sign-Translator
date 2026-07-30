"""Cycle-level stress and integration test for the whole speech foundation layer.

Every stage is exercised in one continuous chain, on real synthesised audio:

    waveform -> log-Mel -> prosody -> streaming -> CTC -> beam/N-best ->
    forced alignment -> timestamps -> calibration -> fail-closed policy ->
    composite objective -> freeze-first schedule

and then stressed along four axes that unit tests do not cover:

* **Adversarial inputs** -- silence, DC, extreme gain, single-sample, NaN/Inf.
* **Determinism** -- identical seeds must give identical outputs; stateful
  components must not leak between cycles.
* **Dtype** -- float32 and float64 paths must both work and agree.
* **Repetition** -- many cycles back-to-back, checking nothing drifts.

The point is not to re-test the mathematics (that is proved stage by stage) but
to catch the failures that only appear when components are *composed*: shape
drift across a seam, state carried between runs, and dtype promotion.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech import (
    LogMelSpectrogram, ProsodyExtractor, StreamingFeatureExtractor,
    LatencyModel, measure_emission_latency, SpeechProjector,
    ctc_prefix_beam_search, ctc_exact_posteriors, ctc_greedy_path, collapse,
    ctc_forced_alignment, token_timings, FrameTimeMapper, Lattice,
    StreamingDecoder, TemperatureScaler, expected_calibration_error,
    brier_decomposition, FailClosedPolicy, Action,
    BoundaryHead, SpeechTrainingObjective, ObjectiveWeights,
    FreezeFirstSchedule, FreezeFirstConfig, Phase,
    inject_lora, N_MELS, HOP_LENGTH, N_FFT, SAMPLE_RATE,
)
from signtranslator.speech.evaluation import (
    word_error_rate, character_error_rate, timestamp_error,
)
from signtranslator.models import SpeechRecognizer
from signtranslator.models.alignment import ContrastiveAligner

SR = SAMPLE_RATE
VOCAB = {120.0: 1, 210.0: 2, 320.0: 3}
SPELLING = {1: "cat", 2: "dog", 3: "bird"}


def _utterance(f0s, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts, spans, cursor = [], [], 0.0
    for f0 in f0s:
        n = int(0.20 * SR)
        t = torch.arange(n, dtype=torch.float32) / SR
        x = sum(torch.sin(2 * math.pi * f0 * h * t) / h for h in (1, 2, 3))
        parts.append(x * torch.hann_window(n, periodic=False) * 0.5)
        spans.append((cursor, cursor + 0.20)); cursor += 0.20
        parts.append(torch.zeros(int(0.08 * SR))); cursor += 0.08
    wav = torch.cat(parts)
    if noise:
        wav = wav + noise * torch.randn(wav.numel(), generator=g)
    return wav, spans


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    seqs = [list(p) for p in itertools.permutations([120.0, 210.0, 320.0])]
    feats = []
    for i, s in enumerate(seqs):
        wav, _ = _utterance(s, seed=i)
        feats.append(fe(wav).t().unsqueeze(0))
    n = min(f.shape[1] for f in feats)
    feats = torch.cat([f[:, :n] for f in feats], dim=0)
    targets = torch.tensor([[VOCAB[f] for f in s] for s in seqs])
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=3, hidden_dim=96,
                           num_layers=2, num_heads=4, subsample=2)
    lengths = torch.full((feats.shape[0],), 3, dtype=torch.long)
    opt = torch.optim.Adam(rec.parameters(), lr=3e-3)
    for _ in range(280):
        loss = rec.loss(feats, targets, lengths)
        opt.zero_grad(); loss.backward(); opt.step()
    rec.eval()
    return rec, fe


# ---------------------------------------------------------------------------
# The full cycle
# ---------------------------------------------------------------------------
def _run_cycle(rec, fe, sequence, noise=0.0, seed=0):
    """One end-to-end pass, returning every intermediate for inspection."""
    wav, spans = _utterance(sequence, noise=noise, seed=seed)
    out = {"wav": wav, "spans": spans}

    # 1. features
    mel = fe(wav)
    out["mel"] = mel
    assert mel.shape[0] == N_MELS and torch.isfinite(mel).all()

    # 2. prosody
    prosody = ProsodyExtractor(hop_length=HOP_LENGTH)(wav)
    out["prosody"] = prosody
    assert torch.isfinite(prosody).all()

    # 3. streaming features must reproduce the offline ones.
    #
    # Compared only where the mel channel carries signal. Speech with silence
    # gaps drives many channels to the 1e-10 clamp floor, and log10 amplifies
    # float32 rounding there without bound: a 1.2e-10 vs 1.4e-10 difference is
    # numerically irrelevant but shifts the log by ~0.014. Exactness of the
    # streaming logic itself is proved separately in float64
    # (test_streaming_equals_offline_exactly_on_tonal_audio).
    stream_fe = LogMelSpectrogram(floor_mode="none")
    stream = StreamingFeatureExtractor(stream_fe)
    pieces = [stream.push(wav[i:i + 1600]) for i in range(0, wav.numel(), 1600)]
    streamed = torch.cat([p for p in pieces if p.shape[1]], dim=1)
    offline = stream_fe(wav)
    out["streamed"] = streamed
    live = stream_fe.mel_energies(wav) > 1e-8
    assert torch.allclose(streamed[live], offline[live], atol=2e-3)

    # 4. projection into planner width
    proj = SpeechProjector(encoder_dim=N_MELS, planner_dim=64)
    paths = proj(mel.t().unsqueeze(0), prosody=prosody.unsqueeze(0),
                 target_length=mel.shape[1])
    out["pathways"] = paths
    assert paths.acoustic.shape[1] == mel.shape[1]

    # 5. recognition
    feats = mel.t().unsqueeze(0)
    with torch.no_grad():
        log_probs = rec(feats)[0]
    out["log_probs"] = log_probs

    # 6. decoding
    nbest = ctc_prefix_beam_search(log_probs, beam_width=12, blank=0)
    out["nbest"] = nbest
    out["hyp"] = nbest.best.tokens
    assert abs(sum(nbest.posteriors) - 1.0) < 1e-9

    # 7. alignment + timestamps (against the TRUE transcript)
    reference = tuple(VOCAB[f] for f in sequence)
    out["reference"] = reference
    al = ctc_forced_alignment(log_probs, list(reference), blank=0)
    mapper = FrameTimeMapper(hop_length=HOP_LENGTH, sample_rate=SR,
                             n_fft=N_FFT, subsample=rec.subsample)
    timings = token_timings(al, mapper=mapper, log_probs=log_probs)
    out["timings"] = timings
    assert len(timings) == len(reference)

    # 8. confidence -> policy
    lat = Lattice.from_nbest(nbest)
    policy = FailClosedPolicy(emit_threshold=0.9, fingerspell_threshold=0.4,
                              sign_lexicon=[1, 2, 3])
    out["decisions"] = policy.decide_sequence(list(lat.best), lat.confidence())
    return out


def test_full_cycle_on_clean_audio(trained):
    rec, fe = trained
    cyc = _run_cycle(rec, fe, [120.0, 210.0, 320.0])
    assert cyc["hyp"] == cyc["reference"]
    # timestamps must land inside the intervals where the tones sounded
    for timing, (lo, hi) in zip(cyc["timings"], cyc["spans"]):
        assert min(timing.end_s, hi) - max(timing.start_s, lo) > 0
    # a confident, correct decode should be assertable
    assert any(d.action is Action.EMIT for d in cyc["decisions"].decisions)


def test_full_cycle_under_noise_degrades_safely(trained):
    """Under noise the chain must stay finite and fail closed, not crash."""
    rec, fe = trained
    cyc = _run_cycle(rec, fe, [120.0, 210.0, 320.0], noise=0.02, seed=5)
    assert torch.isfinite(cyc["log_probs"]).all()
    for d in cyc["decisions"].decisions:
        if d.action is Action.EMIT:
            assert d.confidence >= 0.9          # never assert below threshold


@pytest.mark.parametrize("sequence", [
    [120.0], [320.0, 120.0], [210.0, 210.0, 210.0],
    [120.0, 210.0, 320.0, 120.0, 210.0],
])
def test_cycle_across_utterance_shapes(trained, sequence):
    """Includes a repeated-token utterance, which exercises the CTC blank rule."""
    rec, fe = trained
    cyc = _run_cycle(rec, fe, sequence)
    assert len(cyc["timings"]) == len(sequence)
    for a, b in zip(cyc["timings"], cyc["timings"][1:]):
        assert a.start_frame < b.start_frame


def test_repeated_cycles_do_not_drift(trained):
    """Stateful components must not leak between runs."""
    rec, fe = trained
    first = _run_cycle(rec, fe, [120.0, 210.0, 320.0])
    for _ in range(5):
        again = _run_cycle(rec, fe, [120.0, 210.0, 320.0])
        assert again["hyp"] == first["hyp"]
        assert torch.allclose(again["log_probs"], first["log_probs"], atol=1e-6)
        assert torch.allclose(again["mel"], first["mel"], atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism and statefulness
# ---------------------------------------------------------------------------
def test_streaming_extractor_reset_restores_a_clean_slate():
    fe = LogMelSpectrogram(floor_mode="none")
    stream = StreamingFeatureExtractor(fe)
    wav, _ = _utterance([120.0, 210.0])
    a = torch.cat([p for p in (stream.push(wav[i:i + 1600])
                               for i in range(0, wav.numel(), 1600)) if p.shape[1]], 1)
    stream.reset()
    b = torch.cat([p for p in (stream.push(wav[i:i + 1600])
                               for i in range(0, wav.numel(), 1600)) if p.shape[1]], 1)
    assert torch.equal(a, b)


def test_streaming_decoder_reset_restores_a_clean_slate(trained):
    rec, fe = trained
    wav, _ = _utterance([120.0, 210.0, 320.0])
    with torch.no_grad():
        lp = rec(fe(wav).t().unsqueeze(0))[0]
    dec = StreamingDecoder(beam_width=8, stability=2)

    def run():
        for i in range(0, lp.shape[0], 8):
            dec.update(lp[i:i + 8])
        return dec.finalize().committed

    a = run()
    dec.reset()
    b = run()
    assert a == b
    assert dec.stats.updates == len(range(0, lp.shape[0], 8))


def test_decoding_is_deterministic(trained):
    rec, fe = trained
    wav, _ = _utterance([210.0, 120.0, 320.0])
    with torch.no_grad():
        lp = rec(fe(wav).t().unsqueeze(0))[0]
    runs = [ctc_prefix_beam_search(lp, beam_width=10).best.tokens for _ in range(5)]
    assert len(set(runs)) == 1


# ---------------------------------------------------------------------------
# Adversarial / degenerate inputs
# ---------------------------------------------------------------------------
def test_silence_survives_the_whole_chain():
    """Digital silence must not produce NaN, nor a confident pitch or sign."""
    fe = LogMelSpectrogram()
    silence = torch.zeros(SR // 2)
    mel = fe(silence)
    assert torch.isfinite(mel).all()
    prosody = ProsodyExtractor(hop_length=320)(silence)
    assert torch.isfinite(prosody).all()
    assert float(prosody[:, 1].max()) == 0.0        # nothing voiced


def test_dc_and_extreme_gain_inputs():
    fe = LogMelSpectrogram()
    for wav in (torch.full((4000,), 0.7),          # DC
                torch.randn(4000) * 1e4,           # very loud
                torch.randn(4000) * 1e-8):         # near-silent
        out = fe(wav)
        assert torch.isfinite(out).all(), "front-end produced non-finite output"


def test_single_sample_and_empty_inputs_are_handled():
    fe = LogMelSpectrogram()
    assert fe.num_frames(1) == 0
    stream = StreamingFeatureExtractor(LogMelSpectrogram(floor_mode="none"))
    assert stream.push(torch.zeros(1)).shape[1] == 0
    assert ProsodyExtractor()(torch.zeros(10)).shape[0] == 0


def test_non_finite_audio_is_not_silently_accepted():
    """NaN in must not become plausible-looking features out."""
    fe = LogMelSpectrogram()
    wav = torch.randn(4000)
    wav[100] = float("nan")
    out = fe(wav)
    assert torch.isnan(out).any(), "NaN was silently swallowed by the front-end"


def test_policy_and_decoder_handle_empty_hypotheses():
    policy = FailClosedPolicy(emit_threshold=0.8, sign_lexicon=[1])
    assert len(policy.decide_sequence([], [])) == 0
    assert StreamingDecoder().finalize().full == ()


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------
def test_front_end_agrees_across_dtypes():
    wav = torch.randn(8000)
    f32 = LogMelSpectrogram()(wav)
    f64 = LogMelSpectrogram().double()(wav.double())
    assert torch.allclose(f32.double(), f64, atol=2e-3)


def test_decoder_accepts_both_dtypes(trained):
    rec, fe = trained
    wav, _ = _utterance([120.0, 320.0])
    with torch.no_grad():
        lp = rec(fe(wav).t().unsqueeze(0))[0]
    a = ctc_prefix_beam_search(lp.float(), beam_width=10).best.tokens
    b = ctc_prefix_beam_search(lp.double(), beam_width=10).best.tokens
    assert a == b


# ---------------------------------------------------------------------------
# Cross-stage mathematical consistency, re-checked in composition
# ---------------------------------------------------------------------------
def test_beam_still_matches_exhaustive_enumeration_on_model_output(trained):
    """The proof is re-run on REAL posteriors, not synthetic ones."""
    rec, fe = trained
    wav, _ = _utterance([120.0])
    with torch.no_grad():
        lp = rec(fe(wav).t().unsqueeze(0))[0].double()
    short = F.log_softmax(lp[:6], dim=-1)           # keep enumeration tractable
    exact = ctc_exact_posteriors(short, blank=0)
    beam = {h.tokens: h.log_prob
            for h in ctc_prefix_beam_search(short, beam_width=10_000).hypotheses}
    for label, ref in exact.items():
        if ref > -float("inf"):
            assert abs(beam[label] - ref) < 1e-9


def test_greedy_is_never_more_probable_than_the_beam(trained):
    rec, fe = trained
    wav, _ = _utterance([210.0, 320.0])
    with torch.no_grad():
        lp = rec(fe(wav).t().unsqueeze(0))[0].double()
    nbest = ctc_prefix_beam_search(lp, beam_width=20)
    greedy = collapse(ctc_greedy_path(lp), blank=0)
    scores = {h.tokens: h.log_prob for h in nbest.hypotheses}
    if greedy in scores:
        assert nbest.best.log_prob >= scores[greedy] - 1e-9


# ---------------------------------------------------------------------------
# Training-side integration
# ---------------------------------------------------------------------------
def test_objective_and_schedule_compose_over_many_steps():
    """Train through a phase transition and confirm nothing degenerates."""
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    seqs = [list(p) for p in itertools.permutations([120.0, 210.0, 320.0])]
    feats = []
    for i, s in enumerate(seqs):
        wav, _ = _utterance(s, seed=i)
        feats.append(fe(wav).t().unsqueeze(0))
    n = min(f.shape[1] for f in feats)
    feats = torch.cat([f[:, :n] for f in feats], dim=0)
    targets = torch.tensor([[VOCAB[f] for f in s] for s in seqs])
    lengths = torch.full((feats.shape[0],), 3, dtype=torch.long)

    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=3, hidden_dim=64,
                           num_layers=3, num_heads=4, subsample=2)
    inject_lora(rec, target_suffixes=("linear1", "linear2"), r=4)
    head = BoundaryHead(64)
    aligner = ContrastiveAligner(motion_dim=64, language_dim=64, latent_dim=16)
    obj = SpeechTrainingObjective(rec, ObjectiveWeights(contrastive=0.1,
                                                        boundary=0.5, brier=0.1),
                                  boundary_head=head, aligner=aligner)
    sched = FreezeFirstSchedule(rec, list(rec.encoder.layers),
                                FreezeFirstConfig(adapt_steps=8, base_lr=1e-3,
                                                  unfreeze_blocks=1),
                                extra_trainable=[head, rec.classifier, aligner])
    signs = torch.randn(feats.shape[0], 64)
    losses = []
    for _ in range(20):
        out = obj(feats, targets, lengths, sign_embeddings=signs)
        assert torch.isfinite(out.total)
        losses.append(out.total.detach().item())
        sched.step(out.total)
    assert sched.phase == Phase.REFINE
    assert losses[-1] < losses[0]
    for p in rec.parameters():
        assert torch.isfinite(p).all(), "training produced non-finite parameters"


def test_latency_budget_is_reported_and_respected():
    model = LatencyModel(chunk_frames=8, right_context=4)
    meas = measure_emission_latency(SR, model)
    assert meas.median_s <= meas.p95_s <= model.max_latency_s + 1e-9
    assert "chunk=8" in model.describe() and "right_context=4" in model.describe()


def test_end_to_end_metrics_are_computable_from_a_cycle(trained):
    """The harness metrics must be derivable from the cycle's own outputs."""
    rec, fe = trained
    cycles = [_run_cycle(rec, fe, list(s), noise=0.008, seed=i)
              for i, s in enumerate(itertools.permutations([120.0, 210.0, 320.0]))]
    hyps = [list(c["hyp"]) for c in cycles]
    refs = [list(c["reference"]) for c in cycles]
    wer = word_error_rate(hyps, refs)
    cer = character_error_rate(hyps, refs, SPELLING)
    ts = timestamp_error([(t.start_s, t.end_s) for t in cycles[0]["timings"]],
                         cycles[0]["spans"])
    assert wer.reference_length == 18
    assert cer.reference_length > wer.reference_length
    assert ts.count == 3
    assert 0.0 <= wer.error_rate <= 2.0
