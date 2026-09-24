"""
voice_agent_pipeline.py — Auvia Engine Real-Time Voice Agent
VAD-driven turn-taking state machine + latency instrumentation.
No LLM/network calls here — mirrors the separation of concerns in
dsp_pipeline.py and orchestration_pipeline.py.
"""

import enum
import io
import time
import wave

import webrtcvad

SAMPLE_RATE = 16000          # required by both webrtcvad and Sarvam PCM input
FRAME_MS = 20                # webrtcvad accepts only 10/20/30ms frames
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2   # 16-bit mono PCM
SILENCE_TRAILING_MS = 500    # trailing silence to call end-of-turn
MIN_SPEECH_MS = 200          # ignore blips shorter than this


class TurnState(str, enum.Enum):
    IDLE = "idle"
    SPEAKING = "speaking"


class TurnTaker:
    """
    Feed raw PCM16 mono 16kHz bytes in arbitrary chunk sizes via push_audio().
    Returns the assembled utterance (bytes) the moment enough trailing
    silence follows real speech — that's the turn-taking decision.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        silence_trailing_ms: int = SILENCE_TRAILING_MS,
        min_speech_ms: int = MIN_SPEECH_MS,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames_threshold = max(1, silence_trailing_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._buffer = bytearray()
        self._speech_frames: list[bytes] = []
        self._state = TurnState.IDLE
        self._silence_run = 0
        self._speech_run = 0

    def push_audio(self, chunk: bytes) -> bytes | None:
        self._buffer.extend(chunk)
        result = None

        while len(self._buffer) >= FRAME_BYTES:
            frame = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]

            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                self._speech_frames.append(frame)
                self._speech_run += 1
                self._silence_run = 0
                self._state = TurnState.SPEAKING
            elif self._state == TurnState.SPEAKING:
                self._speech_frames.append(frame)  # keep a little trailing silence — natural cutoff
                self._silence_run += 1
                if (
                    self._silence_run >= self.silence_frames_threshold
                    and self._speech_run >= self.min_speech_frames
                ):
                    result = b"".join(self._speech_frames)
                    self._reset()

        return result

    def _reset(self):
        self._speech_frames = []
        self._state = TurnState.IDLE
        self._silence_run = 0
        self._speech_run = 0


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wraps raw PCM16 mono bytes in a WAV container (no re-encoding needed)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class LatencyTracker:
    """
    Purpose-built for the voice agent's turn latency budget: call .mark(stage)
    right after each pipeline stage completes, in order. .report() returns
    per-stage deltas in ms plus a total — exactly what's shown in the demo
    UI's latency table and logged for the build log.
    """

    def __init__(self):
        self._t0 = time.monotonic()
        self._last = self._t0
        self.stages: dict[str, float] = {}

    def mark(self, stage: str) -> None:
        now = time.monotonic()
        self.stages[stage] = round((now - self._last) * 1000, 1)
        self._last = now

    def report(self) -> dict:
        total_ms = round((self._last - self._t0) * 1000, 1)
        return {**self.stages, "total_ms": total_ms}
