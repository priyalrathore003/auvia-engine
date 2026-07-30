"""
integrations/elevenlabs_client.py
Thin wrapper around the ElevenLabs Music API (POST /v1/music).
Generates an instrumental backing track from a text prompt.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_MUSIC_URL = "https://api.elevenlabs.io/v1/music"
MIN_DURATION_MS = 3_000
MAX_DURATION_MS = 600_000


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
