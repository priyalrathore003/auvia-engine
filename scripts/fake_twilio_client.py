#!/usr/bin/env python3
"""
scripts/fake_twilio_client.py — exercises the full Twilio bridge locally
without a real phone call or Twilio account, by speaking Twilio's actual
Media Streams WebSocket protocol from the client side.

Message shapes here match what's verified in telephony/twilio_bridge.py's
module docstring against Twilio's own docs: connected/start/media/stop
inbound, base64 μ-law 8kHz mono 20ms frames, streamSid required on every
outbound-from-Twilio (== inbound-to-us) message.

Usage:
    # synthesize the test utterance via ElevenLabs (needs ELEVENLABS_API_KEY)
    python scripts/fake_twilio_client.py --say "Hey, what can you help me with?"

    # or drive it from a real recording instead
    python scripts/fake_twilio_client.py --audio path/to/vocal.wav

    # against a non-default bridge URL/port
    python scripts/fake_twilio_client.py --url ws://localhost:8000/twilio-stream --say "hello"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from telephony.mulaw import TWILIO_SAMPLE_RATE, pcm16_to_ulaw, resample_pcm16, ulaw_to_pcm16  # noqa: E402

FRAME_MS = 20
FRAME_SAMPLES = int(TWILIO_SAMPLE_RATE * FRAME_MS / 1000)  # 160
TRAILING_SILENCE_MS = 900  # > TurnTaker's 500ms silence threshold, with margin


def _load_test_audio_pcm16_8k(args: argparse.Namespace) -> np.ndarray:
    """Returns int16 PCM at 8kHz — the caller's simulated speech."""
    if args.audio:
        import librosa

        float_samples, sr = librosa.load(args.audio, sr=None, mono=True)
        pcm16 = np.clip(float_samples * 32768.0, -32768, 32767).astype(np.int16)
        return resample_pcm16(pcm16, sr, TWILIO_SAMPLE_RATE)

    print(f"[setup] synthesizing test utterance via ElevenLabs: {args.say!r}")
    from integrations.elevenlabs_client import stream_speech

    mp3_bytes = b"".join(stream_speech(args.say))
    import librosa

    float_samples, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    pcm16 = np.clip(float_samples * 32768.0, -32768, 32767).astype(np.int16)
    return resample_pcm16(pcm16, sr, TWILIO_SAMPLE_RATE)


def _build_frames(pcm16_8k: np.ndarray) -> list[bytes]:
    silence_samples = int(TWILIO_SAMPLE_RATE * TRAILING_SILENCE_MS / 1000)
    padded = np.concatenate([pcm16_8k, np.zeros(silence_samples, dtype=np.int16)])

    frames = []
    for start in range(0, len(padded), FRAME_SAMPLES):
        chunk = padded[start:start + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        frames.append(pcm16_to_ulaw(chunk))
    return frames


async def run(args: argparse.Namespace) -> None:
    pcm16_8k = _load_test_audio_pcm16_8k(args)
    duration_sec = len(pcm16_8k) / TWILIO_SAMPLE_RATE
    print(f"[setup] test utterance: {duration_sec:.2f}s @ {TWILIO_SAMPLE_RATE}Hz "
          f"+ {TRAILING_SILENCE_MS}ms trailing silence")

    frames = _build_frames(pcm16_8k)
    print(f"[setup] {len(frames)} outbound 20ms frames to send")

    call_sid = f"CAfake{uuid.uuid4().hex[:24]}"
    stream_sid = f"MZfake{uuid.uuid4().hex[:24]}"

    received_ulaw_frames: list[bytes] = []
    marks_received: list[str] = []

    async with websockets.connect(args.url) as ws:
        print(f"[ws] connected to {args.url}")

        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "accountSid": "ACfake0000000000000000000000000",
                "streamSid": stream_sid,
                "callSid": call_sid,
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": TWILIO_SAMPLE_RATE, "channels": 1},
                "customParameters": {},
            },
            "streamSid": stream_sid,
        }))
        print(f"[ws] sent connected + start (call_sid={call_sid})")

        async def sender():
            for i, frame in enumerate(frames):
                await ws.send(json.dumps({
                    "event": "media",
                    "sequenceNumber": str(i + 2),
                    "media": {
                        "track": "inbound",
                        "chunk": str(i + 1),
                        "timestamp": str(i * FRAME_MS),
                        "payload": base64.b64encode(frame).decode(),
                    },
                    "streamSid": stream_sid,
                }))
                await asyncio.sleep(FRAME_MS / 1000)
            print("[ws] all inbound frames sent, waiting for a reply...")

        async def receiver():
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    event = msg.get("event")
                    if event == "media":
                        payload = base64.b64decode(msg["media"]["payload"])
                        received_ulaw_frames.append(payload)
                    elif event == "mark":
                        name = msg.get("mark", {}).get("name")
                        marks_received.append(name)
                        print(f"[ws] mark received: {name!r} — reply playback complete")
                        return  # done, one turn is enough for this smoke test
                    elif event == "clear":
                        print("[ws] clear received (barge-in flush)")
            except websockets.exceptions.ConnectionClosed:
                print("[ws] connection closed by server")

        await asyncio.gather(sender(), asyncio.wait_for(receiver(), timeout=60))

        await ws.send(json.dumps({
            "event": "stop",
            "sequenceNumber": str(len(frames) + 3),
            "stop": {"accountSid": "ACfake0000000000000000000000000", "callSid": call_sid},
            "streamSid": stream_sid,
        }))

    print(f"\n=== RESULT ===")
    print(f"Received {len(received_ulaw_frames)} outbound media frames, {len(marks_received)} mark(s)")

    if received_ulaw_frames:
        combined_ulaw = b"".join(received_ulaw_frames)
        pcm_8k = ulaw_to_pcm16(combined_ulaw)
        out_path = Path(args.out)
        import soundfile as sf
        sf.write(out_path, pcm_8k, TWILIO_SAMPLE_RATE, subtype="PCM_16")
        print(f"Reply audio decoded and saved to {out_path} ({len(pcm_8k) / TWILIO_SAMPLE_RATE:.2f}s)")
    else:
        print("No reply audio received — check server logs (STT may have found no speech, or a stage failed).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8000/twilio-stream", help="Twilio bridge WS URL")
    parser.add_argument("--say", default="Hey, what can you help me with today?",
                         help="Text to synthesize as the simulated caller's speech (via ElevenLabs)")
    parser.add_argument("--audio", default=None, help="Path to a real audio file to use instead of --say")
    parser.add_argument("--out", default="/tmp/fake_twilio_reply.wav", help="Where to save the decoded reply audio")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
