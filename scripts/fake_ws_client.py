#!/usr/bin/env python3
"""
scripts/fake_ws_client.py — drives the browser-facing WebSocket voice agent
(/ws/voice-agent, see voice_agent_ws.py) the same way scripts/fake_twilio_client.py
drives the phone bridge, so scripts/compare_transports.py has real,
independently-measured numbers for both transports.

Sends raw PCM16 16kHz binary frames (matching what a real browser mic
capture would stream — see the "Voice Agent" tab in static/index.html) and
reads back the single JSON `{transcript, reply_text, audio_b64, latency_ms}`
response the existing handler already sends. No new instrumentation is
needed on the WS path for this: latency_ms already carries real,
server-measured numbers for every stage that path can see (stt_ms, llm_ms,
tts_first_byte_ms, tts_complete_ms, total_ms, and — as of this session —
endpointing_ms). This script only exercises that existing contract.

Usage:
    python scripts/fake_ws_client.py --say "Hey, what can you help me with?"
    python scripts/fake_ws_client.py --url ws://localhost:8000/ws/voice-agent --say "hello"
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from voice_agent_pipeline import SAMPLE_RATE  # noqa: E402 — 16000, required by TurnTaker/Sarvam

FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320
TRAILING_SILENCE_MS = 900  # > TurnTaker's 500ms silence threshold, with margin


def _load_test_audio_pcm16(args: argparse.Namespace) -> np.ndarray:
    """Returns int16 PCM at 16kHz — the caller's simulated speech."""
    import librosa

    from telephony.mulaw import resample_pcm16

    if args.audio:
        float_samples, sr = librosa.load(args.audio, sr=None, mono=True)
        pcm16 = np.clip(float_samples * 32768.0, -32768, 32767).astype(np.int16)
        return resample_pcm16(pcm16, sr, SAMPLE_RATE)

    print(f"[setup] synthesizing test utterance via ElevenLabs: {args.say!r}")
    from integrations.elevenlabs_client import stream_speech

    mp3_bytes = b"".join(stream_speech(args.say))
    float_samples, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    pcm16 = np.clip(float_samples * 32768.0, -32768, 32767).astype(np.int16)
    return resample_pcm16(pcm16, sr, SAMPLE_RATE)


def _build_frames(pcm16: np.ndarray) -> list[bytes]:
    silence_samples = int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000)
    padded = np.concatenate([pcm16, np.zeros(silence_samples, dtype=np.int16)])

    frames = []
    for start in range(0, len(padded), FRAME_SAMPLES):
        chunk = padded[start:start + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        frames.append(chunk.tobytes())
    return frames


async def run(args: argparse.Namespace) -> dict:
    pcm16 = _load_test_audio_pcm16(args)
    duration_sec = len(pcm16) / SAMPLE_RATE
    print(f"[setup] test utterance: {duration_sec:.2f}s @ {SAMPLE_RATE}Hz "
          f"+ {TRAILING_SILENCE_MS}ms trailing silence")

    frames = _build_frames(pcm16)
    print(f"[setup] {len(frames)} outbound 20ms binary frames to send")

    async with websockets.connect(args.url) as ws:
        print(f"[ws] connected to {args.url}")

        async def sender():
            for frame in frames:
                await ws.send(frame)
                await asyncio.sleep(FRAME_MS / 1000)

        async def receiver() -> dict:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "turn":
                    return msg
            raise RuntimeError("connection closed before a turn response arrived")

        _, result = await asyncio.gather(sender(), asyncio.wait_for(receiver(), timeout=60))

    print(f"[ws] turn received — transcript={result.get('transcript')!r}")
    print(f"[ws] latency_ms={result.get('latency_ms')}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8000/ws/voice-agent", help="Voice agent WS URL")
    parser.add_argument("--say", default="Hey, what can you help me with today?",
                         help="Text to synthesize as the simulated caller's speech (via ElevenLabs)")
    parser.add_argument("--audio", default=None, help="Path to a real audio file to use instead of --say")
    parser.add_argument("--out-jsonl", default=None,
                         help="Append this turn's latency_ms result as one JSON line to this file")
    args = parser.parse_args()

    result = asyncio.run(run(args))

    if args.out_jsonl:
        with open(args.out_jsonl, "a") as f:
            f.write(json.dumps(result.get("latency_ms", {})) + "\n")
        print(f"[out] appended to {args.out_jsonl}")


if __name__ == "__main__":
    main()
