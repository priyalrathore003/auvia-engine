"""
telephony/twilio_bridge.py — Twilio Media Streams bridge for the existing
real-time voice agent. Adds a telephony transport alongside voice_agent_ws.py;
does not modify it. Reuses TurnTaker, transcribe_vocal, get_llm (via
voice_agent_ws._generate_reply_sync), and stream_speech (via
voice_agent_ws._synthesize_speech_sync) exactly as they already exist.

Protocol details below are verified against Twilio's docs (Media Streams
WebSocket Messages + TwiML Stream reference), not memory:
  - <Connect><Stream url="wss://..."/></Connect> is correct for bidirectional
    audio as-is; the `track` attribute only applies to unidirectional
    <Start><Stream>, not this.
  - Inbound events: connected, start, media, stop, mark, dtmf — all JSON text
    frames (media.payload is base64 μ-law inside the JSON, not a raw binary
    WS frame).
  - start.mediaFormat is always {encoding: "audio/x-mulaw", sampleRate: 8000,
    channels: 1} — Twilio does not negotiate this.
  - Outbound "media"/"mark"/"clear" messages all require streamSid.
  - Twilio buffers and plays outbound media in the order received —
    confirmed via docs ("media messages are buffered and played in the
    order received"), so this does NOT self-pace sends with sleeps. A
    "mark" echoes back only once all audio queued ahead of it has actually
    finished playing (not on receipt) — useful for a future playback-
    complete timestamp, not used yet in this file.
  - The only way to end a bidirectional stream is ending the call — there
    is no graceful "unsubscribe" message.

Deliberately NOT done in this file (flagged, not silently skipped):
  - Twilio webhook signature validation (X-Twilio-Signature) on the TwiML
    endpoint. Twilio's own docs warn this is easy to get subtly wrong
    (exact-URL reconstruction, no re-encoding) and recommend their SDK
    over hand-rolling it — adding that SDK is a new dependency not cleared
    for tonight, and this wasn't in tonight's scope list. The TwiML
    endpoint is unauthenticated for now.
  - True mid-stream TTS cancellation. stream_speech() (integrations/
    elevenlabs_client.py) stays a sync generator consumed via to_thread,
    per instruction not to touch it tonight. Barge-in here is the "naive"
    version: a flag checked between outbound frames that stops forwarding
    audio and sends `clear`, not a request that actually stops ElevenLabs
    generation mid-flight. See _stream_pcm_to_twilio.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from telephony.latency_events import emit_event
from telephony.mulaw import TWILIO_SAMPLE_RATE, pcm16_to_ulaw, resample_pcm16, twilio_ulaw_to_pipeline_pcm16
from voice_agent_pipeline import TurnTaker, pcm16_to_wav_bytes

logger = logging.getLogger(__name__)

TWILIO_FRAME_MS = 20
TWILIO_FRAME_SAMPLES = int(TWILIO_SAMPLE_RATE * TWILIO_FRAME_MS / 1000)  # 160


def build_twiml_response() -> str:
    """TwiML for the inbound-call webhook: opens a bidirectional Media Stream."""
    stream_url = os.getenv("PUBLIC_WS_URL")
    if not stream_url:
        raise RuntimeError("PUBLIC_WS_URL is not set — needed to build the <Stream> URL")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}" />'
        "</Connect>"
        "</Response>"
    )


@dataclass
class TwilioCallSession:
    websocket: WebSocket
    call_sid: str
    stream_sid: str
    turn_taker: TurnTaker = field(default_factory=TurnTaker)
    is_playing: bool = False
    barge_in_requested: bool = False
    turn_task: asyncio.Task | None = None


async def handle_twilio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session: TwilioCallSession | None = None

    try:
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "connected":
                logger.info("[TWILIO] stream connected")

            elif event == "start":
                start = message["start"]
                session = TwilioCallSession(
                    websocket=websocket,
                    call_sid=start["callSid"],
                    stream_sid=start["streamSid"],
                )
                logger.info(
                    "[TWILIO] call started call_sid=%s stream_sid=%s",
                    session.call_sid, session.stream_sid,
                )

            elif event == "media":
                if session is None:
                    continue  # media before start — shouldn't happen per Twilio's ordering, but don't crash
                await _handle_inbound_media(session, message["media"]["payload"])

            elif event == "mark":
                logger.debug("[TWILIO] mark ack: %s", message.get("mark", {}).get("name"))
                if session is not None:
                    emit_event(session.call_id, "playback_mark_received")

            elif event == "stop":
                logger.info("[TWILIO] call stopped")
                break

    except WebSocketDisconnect:
        logger.info("[TWILIO] websocket disconnected")
    except Exception:
        logger.exception("[TWILIO] session error")
    finally:
        if session is not None and session.turn_task is not None and not session.turn_task.done():
            session.turn_task.cancel()
        logger.info("[TWILIO] session closed")


async def _handle_inbound_media(session: TwilioCallSession, payload_b64: str) -> None:
    ulaw_bytes = base64.b64decode(payload_b64)
    pcm16_16k = twilio_ulaw_to_pipeline_pcm16(ulaw_bytes)

    was_speaking = session.turn_taker.is_speaking

    # Naive barge-in: TurnTaker already sees every inbound frame regardless of
    # playback state, so its own state IS "is the caller making sound right
    # now" — no second VAD instance needed. Trigger only while we're actually
    # sending audio (is_playing); the interrupted TTS-forwarding loop is what
    # notices this flag and sends `clear` (see _stream_pcm_to_twilio).
    utterance_pcm = session.turn_taker.push_audio(pcm16_16k.tobytes())

    if not was_speaking and session.turn_taker.is_speaking:
        emit_event(session.call_id, "caller_speech_start")

    if session.is_playing and session.turn_taker.is_speaking:
        session.barge_in_requested = True

    if utterance_pcm is None:
        return

    # user_speech_end uses the real last-voiced-frame timestamp (an honest
    # stand-in for "the caller stopped talking"); endpoint_decision is "now"
    # — push_audio() just returned synchronously, no I/O in between, so this
    # is an accurate instant for when the 500ms trailing-silence threshold
    # was actually met.
    emit_event(
        session.call_id, "user_speech_end",
        ts=session.turn_taker.last_speech_frame_time or datetime.now(timezone.utc),
    )
    emit_event(session.call_id, "endpoint_decision")

    if session.turn_task is not None and not session.turn_task.done():
        # A turn is already being generated/played. Naive scope: drop this
        # utterance rather than overlap two replies. (If it arrived because
        # of a barge-in, the caller will simply re-speak once the current
        # turn's audio stops — barge-in already interrupts playback above.)
        logger.info("[TWILIO] dropping utterance — a turn is already in flight")
        return

    # Run off the receive loop so inbound media (and barge-in detection)
    # keeps flowing while STT/LLM/TTS run.
    session.turn_task = asyncio.create_task(_handle_turn(session, utterance_pcm))


async def _handle_turn(session: TwilioCallSession, utterance_pcm_16k: bytes) -> None:
    # Reused, unmodified: voice_agent_ws.py's own STT/LLM/TTS helpers.
    from integrations.sarvam_client import transcribe_vocal
    from voice_agent_ws import _generate_reply_sync, _synthesize_speech_sync
    from voice_agent_pipeline import LatencyTracker

    lt = LatencyTracker()

    wav_bytes = pcm16_to_wav_bytes(utterance_pcm_16k)
    emit_event(session.call_id, "stt_request_sent")
    stt_result = await asyncio.to_thread(transcribe_vocal, wav_bytes, "wav")
    emit_event(session.call_id, "transcript_final")
    transcript = (stt_result.get("transcript") or "").strip()
    if not transcript:
        logger.info("[TWILIO] no speech detected in this turn")
        return

    emit_event(session.call_id, "llm_request_sent")
    reply_text = await asyncio.to_thread(_generate_reply_sync, transcript)
    emit_event(session.call_id, "llm_generation_complete")

    # _synthesize_speech_sync fully assembles the ElevenLabs mp3 response
    # (sync generator, consumed via to_thread — unmodified from the WS path)
    # before we decode+transcode it. That means the caller hears nothing
    # until the whole reply has been generated, not truly incremental
    # playback — a known, deliberate limitation for tonight (see module
    # docstring: no async TTS client yet).
    emit_event(session.call_id, "tts_request_sent")
    mp3_bytes = await asyncio.to_thread(_synthesize_speech_sync, reply_text, lt)
    # tts_first_byte_ms/tts_complete_ms fire inside a worker thread with no
    # running event loop, so their real wall-clock instants are recovered
    # from LatencyTracker.marks_wall_clock (see voice_agent_pipeline.py) and
    # emitted here, after the fact, in the async caller — not approximated
    # as "now" (which would be measurably late for a many-hundred-ms TTS call).
    if "tts_first_byte_ms" in lt.marks_wall_clock:
        emit_event(session.call_id, "tts_first_audio_byte", ts=lt.marks_wall_clock["tts_first_byte_ms"])
    if "tts_complete_ms" in lt.marks_wall_clock:
        emit_event(session.call_id, "tts_last_chunk_received", ts=lt.marks_wall_clock["tts_complete_ms"])
    if not mp3_bytes:
        logger.warning("[TWILIO] TTS produced no audio for this turn")
        return

    pcm16, source_sr = await asyncio.to_thread(_decode_mp3_to_pcm16, mp3_bytes)
    await _stream_pcm_to_twilio(session, pcm16, source_sr)


def _decode_mp3_to_pcm16(mp3_bytes: bytes) -> tuple[np.ndarray, int]:
    import librosa

    float_samples, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    pcm16 = np.clip(float_samples * 32768.0, -32768, 32767).astype(np.int16)
    return pcm16, sr


async def _stream_pcm_to_twilio(session: TwilioCallSession, pcm16: np.ndarray, source_sr: int) -> None:
    pcm_8k = resample_pcm16(pcm16, source_sr, TWILIO_SAMPLE_RATE)

    session.is_playing = True
    session.barge_in_requested = False
    first_frame_sent = False
    try:
        for start in range(0, len(pcm_8k), TWILIO_FRAME_SAMPLES):
            if session.barge_in_requested:
                logger.info("[TWILIO] barge-in — stopping playback, sending clear")
                await session.websocket.send_json({
                    "event": "clear",
                    "streamSid": session.stream_sid,
                })
                return

            frame = pcm_8k[start:start + TWILIO_FRAME_SAMPLES]
            ulaw_bytes = pcm16_to_ulaw(frame)
            await session.websocket.send_json({
                "event": "media",
                "streamSid": session.stream_sid,
                "media": {"payload": base64.b64encode(ulaw_bytes).decode()},
            })
            if not first_frame_sent:
                # This IS the honest phone-path equivalent of "audio begins
                # reaching the caller" — the WS path has no matching signal
                # (see scripts/compare_transports.py).
                emit_event(session.call_id, "playback_start")
                first_frame_sent = True

        await session.websocket.send_json({
            "event": "mark",
            "streamSid": session.stream_sid,
            "mark": {"name": "reply-complete"},
        })
    finally:
        session.is_playing = False
