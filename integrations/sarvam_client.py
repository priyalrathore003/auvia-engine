"""
integrations/sarvam_client.py
Thin wrapper around the Sarvam Speech-to-Text API (POST /speech-to-text).
Best-effort: transcription enriches the music prompt but must never break
the orchestration pipeline, so every failure is swallowed and logged.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

_EXT_TO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
}


def transcribe_vocal(audio_bytes: bytes, ext: str) -> dict:
    """
    Transcribes a short (<30s) vocal clip via Sarvam STT.
    Returns {"transcript": str, "language_code": str} — empty strings on any failure.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        logger.warning("[SARVAM] SARVAM_API_KEY not set — skipping transcription")
        return {"transcript": "", "language_code": ""}

    mime = _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")

    try:
        response = httpx.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": api_key},
            files={"file": (f"vocal.{ext}", audio_bytes, mime)},
            data={
                "model": "saarika:v2.5",
                "language_code": "unknown",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        transcript = data.get("transcript", "") or ""
        language_code = data.get("language_code", "") or ""
        logger.info("[SARVAM] transcribed %d chars, language=%s", len(transcript), language_code)
        return {"transcript": transcript, "language_code": language_code}

    except Exception as e:
        logger.warning("[SARVAM] transcription failed (non-fatal): %s", e)
        return {"transcript": "", "language_code": ""}
