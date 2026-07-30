"""
orchestration_pipeline.py — Auvia Engine "Sing & Orchestrate"
Pure DSP/analysis functions: vocal analysis (tempo/key/duration) and
vocal+backing mixdown. No LLM or network calls live here — mirrors the
separation of concerns in dsp_pipeline.py.
"""

import io
import logging
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_BACKING_GAIN_DB = -6.0
_TARGET_PEAK_DBFS = -1.0


def _load_from_bytes(audio_bytes: bytes, ext: str, sr=None, duration=None, mono=True):
    suffix = f".{ext.lower().replace('audio/', '')}" or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, actual_sr = librosa.load(tmp_path, sr=sr, mono=mono, duration=duration)
    finally:
        os.unlink(tmp_path)
    return y, actual_sr


def _to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _peak_normalize(y: np.ndarray, target_dbfs: float = _TARGET_PEAK_DBFS) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        target = 10 ** (target_dbfs / 20)
        y = y * (target / peak)
    return y


def _estimate_key(y: np.ndarray, sr: int) -> str:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "C major"

    for i in range(12):
        major_rot = np.roll(_MAJOR_PROFILE, i)
        minor_rot = np.roll(_MINOR_PROFILE, i)

        major_score = np.corrcoef(chroma_mean, major_rot)[0, 1]
        minor_score = np.corrcoef(chroma_mean, minor_rot)[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = f"{_PITCH_CLASSES[i]} major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{_PITCH_CLASSES[i]} minor"

    return best_key


def analyze_vocal(audio_bytes: bytes, ext: str) -> dict:
    """
    Analyzes a vocal recording for tempo, musical key, and duration.
    Returns {"tempo_bpm": float, "key": str, "duration_sec": float}.
    """
    y, sr = _load_from_bytes(audio_bytes, ext, sr=None, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    key = _estimate_key(y, sr)

    logger.info(
        "[ANALYZE] tempo=%.1f bpm | key=%s | duration=%.1fs",
        tempo_bpm, key, duration_sec,
    )
    return {"tempo_bpm": round(tempo_bpm, 1), "key": key, "duration_sec": round(duration_sec, 2)}


def trim_audio_bytes(audio_bytes: bytes, ext: str, max_seconds: float) -> bytes:
    """Trims audio to at most max_seconds and re-encodes as WAV (used before Sarvam STT)."""
    y, sr = _load_from_bytes(audio_bytes, ext, sr=None, duration=max_seconds, mono=True)
    return _to_wav_bytes(y, sr)


def mix_vocal_with_backing(
    vocal_bytes: bytes, vocal_ext: str, backing_bytes: bytes, backing_ext: str
) -> bytes:
    """
    Mixes an (already DSP-enhanced) vocal with an AI-generated backing track.
    Backing is resampled to the vocal's sample rate, tiled/trimmed to match
    the vocal's duration, gain-staged under the vocal, and peak-normalized.
    Returns mixed WAV bytes.
    """
    vocal, sr = _load_from_bytes(vocal_bytes, vocal_ext, sr=None, mono=True)
    backing, backing_sr = _load_from_bytes(backing_bytes, backing_ext, sr=None, mono=True)

    if backing_sr != sr:
        backing = librosa.resample(backing, orig_sr=backing_sr, target_sr=sr)

    target_len = len(vocal)
    if len(backing) < target_len:
        repeats = int(np.ceil(target_len / max(len(backing), 1)))
        backing = np.tile(backing, repeats)
    backing = backing[:target_len]

    backing_gain = 10 ** (_BACKING_GAIN_DB / 20)
    mixed = vocal + backing * backing_gain
    mixed = _peak_normalize(mixed)

    logger.info("[MIX] vocal=%d samples backing=%d samples @ %d Hz", len(vocal), len(backing), sr)
    return _to_wav_bytes(mixed, sr)
