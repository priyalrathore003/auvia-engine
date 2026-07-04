"""
dsp_pipeline.py — Auvia Engine
Accepts WAV or MP3 bytes, returns enhanced WAV bytes.
MP3 support requires ffmpeg (included in Dockerfile).
"""

import io
import os
import tempfile

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


def apply_timbre_enhancement(input_bytes: bytes, input_format: str = "wav") -> bytes:
    """
    Two-stage DSP pipeline:
      1. Noise reduction    — noisereduce spectral gating
      2. Spectral shaping   — librosa STFT EQ
         · Sub-bass roll-off : < 80 Hz  × 0.3
         · Low-mid warmth    : 150–500 Hz × 1.4
         · Presence lift     : 2–5 kHz  × 1.15
         · Peak normalize    : −1 dBFS, 16-bit PCM WAV

    Input:  raw bytes (WAV or MP3)
    Output: enhanced WAV bytes (always WAV out — safe, lossless)
    """
    # ── Load audio ────────────────────────────────────────────────────────────
    # librosa handles WAV and MP3 (MP3 requires ffmpeg in PATH)
    suffix = f".{input_format.lower().replace('audio/', '')}"
    if "mpeg" in suffix or "mp3" in suffix:
        suffix = ".mp3"
    else:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(input_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=False)
    finally:
        os.unlink(tmp_path)

    # Ensure 2D (channels, samples)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    enhanced_channels = []

    for channel in y:
        # ── Stage 1: Noise reduction ──────────────────────────────────────────
        reduced = nr.reduce_noise(
            y=channel,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            prop_decrease=0.75,
        )

        # ── Stage 2: Spectral shaping ─────────────────────────────────────────
        D    = librosa.stft(reduced, n_fft=2048, hop_length=512)
        mag  = np.abs(D)
        phase = np.angle(D)

        freqs = np.fft.rfftfreq(2048, d=1.0 / sr)   # shape: (1025,)
        gain  = np.ones_like(freqs)

        # Sub-bass roll-off
        gain[freqs < 80] = 0.3
        # Low-mid warmth
        gain[(freqs >= 150) & (freqs <= 500)] = 1.4
        # Presence lift
        gain[(freqs >= 2000) & (freqs <= 5000)] = 1.15

        gain_col   = gain[:, np.newaxis]              # broadcast over time
        mag_shaped = mag * gain_col
        D_shaped   = mag_shaped * np.exp(1j * phase)

        shaped = librosa.istft(D_shaped, hop_length=512, length=len(reduced))

        # ── Peak normalize to −1 dBFS ─────────────────────────────────────────
        peak = np.max(np.abs(shaped))
        if peak > 0:
            target = 10 ** (-1.0 / 20)           # −1 dBFS
            shaped = shaped * (target / peak)

        enhanced_channels.append(shaped)

    # ── Reconstruct + output as WAV ───────────────────────────────────────────
    enhanced = np.stack(enhanced_channels, axis=0)  # (channels, samples)
    if enhanced.shape[0] == 1:
        enhanced = enhanced[0]                       # mono → 1D
    else:
        enhanced = enhanced.T                        # stereo → (samples, channels)

    buf = io.BytesIO()
    sf.write(buf, enhanced, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()