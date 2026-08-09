"""
intelligence_pipeline.py — Auvia Engine Acoustic Intelligence
Structured audio analysis: pitch, note, key, tempo, and quality metrics.
Pure librosa/numpy — no LLM, no network calls. Reuses the audio-loading
and key-estimation utilities already built for Sing & Orchestrate.
"""

import logging

import librosa
import numpy as np

from orchestration_pipeline import estimate_key, load_audio_from_bytes

logger = logging.getLogger(__name__)

CLIPPING_THRESHOLD = 0.99


def _detect_pitch(y: np.ndarray, sr: int) -> tuple[float, str | None]:
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    if len(voiced_f0) == 0:
        return 0.0, None

    pitch_hz = float(np.median(voiced_f0))
    note = librosa.hz_to_note(pitch_hz)
    return pitch_hz, note


def _measure_levels(y: np.ndarray) -> tuple[float, float, float]:
    rms = librosa.feature.rms(y=y)[0]
    noise_floor_db = float(20 * np.log10(np.percentile(rms, 10) + 1e-9))
    peak_db = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    dynamic_range_db = round(peak_db - noise_floor_db, 1)
    clipping_ratio = float(np.mean(np.abs(y) > CLIPPING_THRESHOLD))
    return noise_floor_db, dynamic_range_db, clipping_ratio


def _compute_quality_score(dynamic_range_db: float, noise_floor_db: float, clipping_ratio: float) -> int:
    score = 100.0
    if dynamic_range_db < 30:
        score -= (30 - dynamic_range_db) * 1.5
    if noise_floor_db > -50:
        score -= (noise_floor_db + 50) * 1.2
    score -= clipping_ratio * 500
    return int(round(max(0, min(100, score))))


def _recommend_processing(noise_floor_db: float, dynamic_range_db: float, clipping_ratio: float) -> list[str]:
    recs = []
    if noise_floor_db > -45:
        recs.append("noise_reduction")
    if dynamic_range_db < 20:
        recs.append("dynamic_range_expansion")
    if clipping_ratio > 0.001:
        recs.append("declipping")
    recs.append("eq_warmth")
    return recs


def analyze_intelligence(audio_bytes: bytes, ext: str) -> dict:
    """
    Full acoustic-intelligence report for a single audio file.
    Returns pitch/note/key/tempo plus a 0-100 quality score, noise floor,
    dynamic range, and a short list of recommended processing steps.
    """
    y, sr = load_audio_from_bytes(audio_bytes, ext, sr=None, mono=True)

    pitch_hz, note = _detect_pitch(y, sr)
    key = estimate_key(y, sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    noise_floor_db, dynamic_range_db, clipping_ratio = _measure_levels(y)
    quality_score = _compute_quality_score(dynamic_range_db, noise_floor_db, clipping_ratio)
    recommended_processing = _recommend_processing(noise_floor_db, dynamic_range_db, clipping_ratio)

    logger.info(
        "[INTELLIGENCE] pitch=%.1fHz note=%s key=%s tempo=%.1f quality=%d",
        pitch_hz, note, key, tempo_bpm, quality_score,
    )

    return {
        "pitch_hz": round(pitch_hz, 2),
        "note": note,
        "key": key,
        "tempo_bpm": round(tempo_bpm, 1),
        "quality_score": quality_score,
        "noise_floor_db": round(noise_floor_db, 1),
        "dynamic_range_db": dynamic_range_db,
        "recommended_processing": recommended_processing,
    }
