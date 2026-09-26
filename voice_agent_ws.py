"""
voice_agent_ws.py — Auvia Engine Real-Time Voice Agent
WebSocket session handler: continuous mic PCM16 in → VAD turn-taking →
Sarvam STT → LLM reply → ElevenLabs streaming TTS → assembled reply out,
with per-turn latency instrumentation (LatencyTracker).

All blocking network/CPU calls (STT, LLM, TTS) run via asyncio.to_thread
so one voice session never stalls the event loop for other connections —
that's the difference between a demo and something that could actually
hold concurrent real-time sessions.
"""

import asyncio
import base64
import logging
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from langgraph_orchestrator import get_llm
from voice_agent_pipeline import LatencyTracker, TurnTaker, pcm16_to_wav_bytes

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Auvia's voice assistant. Keep replies short (1-2 sentences), "
    "conversational, and spoken-friendly — no markdown, no lists, no emoji."
)


async def handle_voice_session(websocket: WebSocket) -> None:
    await websocket.accept()
    turn_taker = TurnTaker()
    logger.info("[VOICE-AGENT] session connected")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk is None:
                continue  # ignore non-binary control frames for now

            utterance_pcm = turn_taker.push_audio(chunk)
            if utterance_pcm is None:
                continue

            # Captured immediately so the phone-vs-WS comparison (see
            # scripts/compare_transports.py) has a real, comparable
            # "endpointing" duration for this path too, not just the phone
            # bridge — same TurnTaker class, same silence threshold.
            endpoint_decision_time = datetime.now(timezone.utc)
            await _process_turn(websocket, utterance_pcm, turn_taker.last_speech_frame_time, endpoint_decision_time)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("[VOICE-AGENT] session error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info("[VOICE-AGENT] session closed")


async def _process_turn(
    websocket: WebSocket,
    utterance_pcm: bytes,
    speech_end_time: "datetime | None" = None,
    endpoint_decision_time: "datetime | None" = None,
) -> None:
    lt = LatencyTracker()

    from integrations.sarvam_client import transcribe_vocal

    wav_bytes = pcm16_to_wav_bytes(utterance_pcm)
    stt_result = await asyncio.to_thread(transcribe_vocal, wav_bytes, "wav")
    transcript = (stt_result.get("transcript") or "").strip()
    lt.mark("stt_ms")

    if not transcript:
        await websocket.send_json({
            "type": "turn",
            "transcript": "",
            "reply_text": "",
            "audio_b64": None,
            "latency_ms": lt.report(),
            "note": "No speech detected in this turn.",
        })
        return

    reply_text = await asyncio.to_thread(_generate_reply_sync, transcript)
    lt.mark("llm_ms")

    audio_bytes = await asyncio.to_thread(_synthesize_speech_sync, reply_text, lt)
    audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None

    latency = lt.report()
    if speech_end_time is not None and endpoint_decision_time is not None:
        # Additive field, not one of LatencyTracker's own stages (that timer
        # only starts after push_audio() already returned the utterance, so
        # it can't see this duration itself). Real measured value: the same
        # TurnTaker/500ms-silence-threshold logic as the phone bridge.
        latency["endpointing_ms"] = round(
            (endpoint_decision_time - speech_end_time).total_seconds() * 1000, 1
        )
    logger.info("[VOICE-AGENT] turn latency: %s", latency)

    await websocket.send_json({
        "type": "turn",
        "transcript": transcript,
        "reply_text": reply_text,
        "audio_b64": audio_b64,
        "latency_ms": latency,
    })


def _generate_reply_sync(transcript: str) -> str:
    try:
        llm = get_llm()
        prompt = f"{SYSTEM_PROMPT}\n\nUser said: {transcript}\n\nYour reply:"
        response = llm.invoke(prompt)
        reply_text = (response.text or "").strip()
        return reply_text or "Sorry, could you say that again?"
    except Exception as e:
        logger.warning("[VOICE-AGENT] LLM failed: %s", e)
        return "Sorry, I couldn't process that."


def _synthesize_speech_sync(text: str, lt: LatencyTracker) -> bytes:
    try:
        from integrations.elevenlabs_client import stream_speech

        chunks = []
        first = True
        for chunk in stream_speech(text):
            if first:
                lt.mark("tts_first_byte_ms")
                first = False
            chunks.append(chunk)
        if first:
            lt.mark("tts_first_byte_ms")  # nothing streamed at all

        audio_bytes = b"".join(chunks)
        lt.mark("tts_complete_ms")
        return audio_bytes

    except Exception as e:
        logger.warning("[VOICE-AGENT] TTS failed: %s", e)
        lt.mark("tts_first_byte_ms")
        lt.mark("tts_complete_ms")
        return b""
