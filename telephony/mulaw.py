"""
telephony/mulaw.py — G.711 μ-law codec + sample-rate conversion for the
Twilio Media Streams bridge.

Twilio Media Streams is fixed at 8kHz mono μ-law (confirmed against Twilio's
docs, not assumed — see telephony/twilio_bridge.py header). The rest of the
voice agent pipeline (webrtcvad, Sarvam STT) expects 16kHz mono PCM16. This
module is the only place that boundary is crossed.

Library choice: numpy, not audioop. audioop is deprecated as of Python 3.11
and removed entirely in 3.13 (PEP 594) — since this Dockerfile already pins
python:3.11-slim, audioop would work today, but shipping a codec on a
module that's already gone in the next Python major version is a bad trade
for ~40 lines of numpy. Resampling reuses librosa.resample, already a
project dependency (dsp_pipeline.py, orchestration_pipeline.py) — no new
dependency for either direction.

The μ-law encode/decode below is a numpy port of CPython's own audioop.c
(st_14linear2ulaw / the ulaw2lin table walk, from Modules/audioop.c) — not
a from-memory reconstruction of the general G.711 spec. That distinction
matters: an early version of this file used the textbook exponent/mantissa
formula from memory and was numerically wrong at every negative-sample
exponent boundary (381/65536 possible int16 values) versus what audioop
actually produces, traced to encode using an arithmetic right-shift into a
14-bit domain (sign-preserving, i.e. floor toward -inf for negative values)
before biasing — not `abs()` then shift. Verified bit-exact against
audioop as an oracle across all 65,536 possible int16 values on both
directions before being trusted here; audioop itself is never imported at
runtime, only used during development to check this file.
"""
from __future__ import annotations

import numpy as np

TWILIO_SAMPLE_RATE = 8000
PIPELINE_SAMPLE_RATE = 16000

# -- decode (ulaw -> pcm16): standard G.711 expansion, bit-exact vs audioop.ulaw2lin --
_BIAS = 0x84

# -- encode (pcm16 -> ulaw): ported from CPython's st_14linear2ulaw, which works in a
# 14-bit domain reached via (sample << 16) >> 18 == an arithmetic (sign-preserving) >> 2.
_BIAS_14 = _BIAS >> 2  # 0x21 = 33
_SEG_UEND_14 = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def ulaw_to_pcm16(ulaw_bytes: bytes) -> np.ndarray:
    """Decode raw μ-law bytes to int16 PCM samples."""
    u = np.frombuffer(ulaw_bytes, dtype=np.uint8).astype(np.int32)
    u = (~u) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa << 3) + _BIAS) << exponent
    magnitude -= _BIAS
    samples = np.where(sign != 0, -magnitude, magnitude)
    return np.clip(samples, -32768, 32767).astype(np.int16)


def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM samples to raw μ-law bytes."""
    # arithmetic (sign-preserving) right shift by 2 — numpy's >> on a signed
    # int32 array does this natively, matching C's (int32_t)(s<<16) >> 18.
    pcm_val = pcm.astype(np.int32) >> 2

    negative = pcm_val < 0
    magnitude = np.where(negative, -pcm_val, pcm_val) + _BIAS_14

    seg = np.searchsorted(_SEG_UEND_14, magnitude, side="left").astype(np.int32)
    seg_clamped = np.clip(seg, 0, 7)  # only used where seg < 8; seg==8 branch below ignores it
    mantissa = (magnitude >> (seg_clamped + 1)) & 0x0F
    uval = np.where(seg >= 8, 0x7F, (seg_clamped << 4) | mantissa)

    mask = np.where(negative, 0x7F, 0xFF)
    ulaw = uval ^ mask
    return (ulaw & 0xFF).astype(np.uint8).tobytes()


def resample_pcm16(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample int16 PCM between sample rates via librosa (float32 internally)."""
    if orig_sr == target_sr:
        return pcm
    import librosa

    float_samples = pcm.astype(np.float32) / 32768.0
    resampled = librosa.resample(float_samples, orig_sr=orig_sr, target_sr=target_sr)
    resampled = np.clip(resampled * 32768.0, -32768, 32767)
    return resampled.astype(np.int16)


def twilio_ulaw_to_pipeline_pcm16(ulaw_bytes: bytes) -> np.ndarray:
    """Inbound: Twilio 8kHz μ-law -> 16kHz PCM16 for TurnTaker/VAD/STT."""
    pcm_8k = ulaw_to_pcm16(ulaw_bytes)
    return resample_pcm16(pcm_8k, TWILIO_SAMPLE_RATE, PIPELINE_SAMPLE_RATE)


def pipeline_pcm16_to_twilio_ulaw(pcm: np.ndarray, source_sr: int) -> bytes:
    """Outbound: PCM16 at `source_sr` -> Twilio 8kHz μ-law."""
    pcm_8k = resample_pcm16(pcm, source_sr, TWILIO_SAMPLE_RATE)
    return pcm16_to_ulaw(pcm_8k)
