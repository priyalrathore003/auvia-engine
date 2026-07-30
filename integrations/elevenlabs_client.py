"""
integrations/elevenlabs_client.py
Thin wrappers around ElevenLabs APIs:
  - Music (POST /v1/music) — generates an instrumental backing track.
  - TTS streaming (POST /v1/text-to-speech/{voice_id}/stream) — low-latency
    speech synthesis for the real-time voice agent.
"""

import logging
import os
from collections.abc import Iterator

import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_MUSIC_URL = "https://api.elevenlabs.io/v1/music"
MIN_DURATION_MS = 3_000
MAX_DURATION_MS = 600_000

# "Rachel" — ElevenLabs' standard premade voice, used as a default so the
# voice agent works out of the box without requiring a custom voice_id.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
ELEVENLABS_TTS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


def compose_music(prompt: str, duration_ms: int) -> bytes:
    """
    Calls ElevenLabs Music to generate an instrumental backing track.
    Returns raw audio bytes (mp3). Raises RuntimeError on failure.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    clamped_ms = max(MIN_DURATION_MS, min(MAX_DURATION_MS, int(duration_ms)))

    try:
        response = httpx.post(
            ELEVENLABS_MUSIC_URL,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "music_length_ms": clamped_ms,
                "force_instrumental": True,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        logger.info(
            "[ELEVENLABS] composed %s ms of audio (%s bytes)",
            clamped_ms, len(response.content),
        )
        return response.content

    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300]
        raise RuntimeError(f"ElevenLabs Music API error {e.response.status_code}: {detail}") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"ElevenLabs Music API request failed: {e}") from e


def stream_speech(
    text: str,
    voice_id: str | None = None,
    model_id: str = "eleven_flash_v2_5",
) -> Iterator[bytes]:
    """
    Streams synthesized speech from ElevenLabs as mp3 chunks arrive
    (chunked HTTP transfer — first chunk typically arrives well before
    the full utterance finishes generating). Raises RuntimeError on failure.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    url = ELEVENLABS_TTS_STREAM_URL.format(voice_id=voice_id or DEFAULT_VOICE_ID)

    try:
        with httpx.stream(
            "POST",
            url,
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=60.0,
        ) as response:
            if response.status_code >= 400:
                detail = response.read().decode(errors="replace")[:300]
                raise RuntimeError(f"ElevenLabs TTS API error {response.status_code}: {detail}")
            for chunk in response.iter_bytes():
                if chunk:
                    yield chunk

    except httpx.HTTPError as e:
        raise RuntimeError(f"ElevenLabs TTS API request failed: {e}") from e
