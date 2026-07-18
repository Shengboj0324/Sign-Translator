"""Stage-1 integration: real waveforms through the front-end into the model.

Everything upstream is tested in isolation; this file checks the seams. It
synthesises actual audio (formant-like tones with silences), runs it through the
log-Mel front-end and the prosody pathway, projects it to planner width, and
feeds it to the existing CTC ``SpeechRecognizer`` -- confirming the stage-1
output is directly consumable by the model that was previously fed synthetic
feature tensors.
"""

import math

import torch

from signtranslator.speech import (
    LogMelSpectrogram, ProsodyExtractor, SpeechProjector, StreamingFeatureExtractor,
    LatencyModel, measure_emission_latency, inject_lora, mark_only_lora_trainable,
    SAMPLE_RATE, HOP_LENGTH, N_MELS,
)
from signtranslator.models import SpeechRecognizer

SR = SAMPLE_RATE


def _utterance(word_f0s, word_s=0.25, gap_s=0.08, sr=SR):
    """Synthesise a crude 'utterance': tone bursts separated by silences.

    Each 'word' is a harmonic stack at its own fundamental, so the acoustic
    features genuinely differ per word and CTC has something learnable.
    """
    parts = []
    for f0 in word_f0s:
        t = torch.arange(int(word_s * sr), dtype=torch.float32) / sr
        x = sum(torch.sin(2 * math.pi * f0 * h * t) / h for h in (1, 2, 3, 4))
        env = torch.hann_window(x.numel(), periodic=False)      # avoid clicks
        parts.append(x * env * 0.5)
        parts.append(torch.zeros(int(gap_s * sr)))
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# Front-end -> features
# ---------------------------------------------------------------------------
def test_real_waveform_produces_sane_logmel():
    fe = LogMelSpectrogram()
    x = _utterance([120.0, 200.0, 300.0])
    feats = fe(x)
    assert feats.shape[0] == N_MELS
    assert feats.shape[1] == fe.num_frames(x.numel())
    assert torch.isfinite(feats).all()
    # Voiced bursts must carry more energy than the silent gaps.
    frame_energy = feats.mean(dim=0)
    assert float(frame_energy.max() - frame_energy.min()) > 0.1


def test_prosody_tracks_the_synthesised_pitch_contour():
    """Three words at 120/200/300 Hz must be recovered in the right order."""
    ex = ProsodyExtractor(hop_length=HOP_LENGTH)
    feats = ex(_utterance([120.0, 200.0, 300.0], word_s=0.4, gap_s=0.15))
    voiced = feats[:, 1] > 0.5
    f0 = torch.exp(feats[:, 0])[voiced]
    assert f0.numel() > 10
    # Split the voiced frames into thirds; medians must be increasing.
    n = f0.numel() // 3
    m1, m2, m3 = (float(f0[:n].median()), float(f0[n:2 * n].median()),
                  float(f0[2 * n:].median()))
    assert m1 < m2 < m3, (m1, m2, m3)
    assert abs(m1 - 120.0) / 120.0 < 0.15
    assert abs(m3 - 300.0) / 300.0 < 0.15


def test_pauses_align_with_the_injected_gaps():
    from signtranslator.speech import rms_energy, detect_pauses
    x = _utterance([150.0, 150.0, 150.0], word_s=0.3, gap_s=0.2)
    pauses = detect_pauses(rms_energy(x, 400, HOP_LENGTH), min_frames=8)
    assert len(pauses) >= 2          # gaps after the first two words at least


# ---------------------------------------------------------------------------
# Front-end -> projection -> planner width
# ---------------------------------------------------------------------------
def test_front_end_to_planner_width_pipeline():
    fe = LogMelSpectrogram()
    ex = ProsodyExtractor(hop_length=HOP_LENGTH)
    x = _utterance([140.0, 240.0])

    mel = fe(x).t().unsqueeze(0)                       # (1, T, n_mels)
    prosody = ex(x).unsqueeze(0)                       # (1, T', 4)
    projector = SpeechProjector(encoder_dim=N_MELS, planner_dim=64)
    out = projector(mel, prosody=prosody, target_length=mel.shape[1])

    assert out.acoustic.shape == (1, mel.shape[1], 64)
    assert out.prosody.shape == (1, mel.shape[1], ProsodyExtractor.N_FEATURES)
    assert out.as_fused().shape == (1, mel.shape[1], 68)
    assert torch.isfinite(out.acoustic).all()


# ---------------------------------------------------------------------------
# Front-end -> existing CTC recogniser
# ---------------------------------------------------------------------------
def test_logmel_features_drive_the_existing_recognizer():
    """The stage-1 front-end output must be directly consumable by the model."""
    fe = LogMelSpectrogram()
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=8, hidden_dim=64,
                           num_layers=2, num_heads=2)
    x = _utterance([120.0, 200.0, 300.0])
    feats = fe(x).t().unsqueeze(0)                     # (1, T, n_mels)
    log_probs = rec(feats)
    assert log_probs.shape[0] == 1
    assert log_probs.shape[2] == 9                     # 8 tokens + blank
    assert torch.allclose(log_probs.exp().sum(-1),
                          torch.ones_like(log_probs[..., 0]), atol=1e-5)


def test_ctc_learns_from_real_logmel_features():
    """A real end-to-end learning check on genuine DSP output, not fake tensors.

    Three distinct pitch patterns are mapped to three distinct token sequences;
    the CTC loss must fall substantially.
    """
    torch.manual_seed(0)
    fe = LogMelSpectrogram()
    patterns = [[120.0, 300.0], [300.0, 120.0], [200.0, 200.0]]
    feats = torch.cat([fe(_utterance(p)).t().unsqueeze(0) for p in patterns], dim=0)
    targets = torch.tensor([[1, 3], [3, 1], [2, 2]])
    lengths = torch.full((3,), 2, dtype=torch.long)

    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=4, hidden_dim=64,
                           num_layers=2, num_heads=2)
    opt = torch.optim.Adam(rec.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = rec.loss(feats, targets, lengths)
        assert torch.isfinite(loss)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first * 0.5, f"{first} -> {float(loss)}"


def test_lora_adapts_the_recognizer_without_touching_base_weights():
    """Freeze-first protocol applied to the real recogniser."""
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=6, hidden_dim=64,
                           num_layers=2, num_heads=2)
    adapted = inject_lora(rec, target_suffixes=("out_proj",), r=4)
    assert len(adapted) > 0, "no attention projections were adapted"
    n_trainable = mark_only_lora_trainable(rec)
    assert n_trainable > 0
    for name, p in rec.named_parameters():
        if p.requires_grad:
            assert "lora_" in name


# ---------------------------------------------------------------------------
# Streaming integration
# ---------------------------------------------------------------------------
def test_streaming_front_end_feeds_recognizer_incrementally():
    """Push real audio in chunks and decode from the accumulated features."""
    fe = LogMelSpectrogram(floor_mode="none")
    stream = StreamingFeatureExtractor(fe)
    rec = SpeechRecognizer(input_dim=N_MELS, num_tokens=5, hidden_dim=32,
                           num_layers=1, num_heads=2)
    x = _utterance([150.0, 250.0])

    model = LatencyModel(chunk_frames=8, right_context=4)
    chunk_samples = model.chunk_frames * HOP_LENGTH
    collected = []
    for i in range(0, x.numel(), chunk_samples):
        out = stream.push(x[i:i + chunk_samples])
        if out.shape[1]:
            collected.append(out)
    feats = torch.cat(collected, dim=1).t().unsqueeze(0)

    decoded = rec.decode(feats)
    assert len(decoded) == 1 and isinstance(decoded[0], list)

    meas = measure_emission_latency(x.numel(), model)
    assert meas.p95_s <= model.max_latency_s + 1e-9
    assert model.max_latency_s < 0.2                   # documented budget


def test_reported_streaming_configuration_is_explicit():
    """The spec requires chunk size and right context to be stated, not implied."""
    text = LatencyModel(chunk_frames=8, right_context=4).describe()
    for token in ("chunk=8", "right_context=4", "latency"):
        assert token in text
