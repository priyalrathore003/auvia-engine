"""
orchestration_graph.py — Auvia Engine "Sing & Orchestrate"
LangGraph state machine that turns a recorded vocal into a downloadable,
AI-orchestrated mix.

Graph topology:
    START → analyze_node → transcribe_node → prompt_synthesis_node → compose_node
                                                                          → [route]
                                                        mix_node ↙            ↘ synthesis_node (compose failed)
                                                   synthesis_node ← ──────────┘

Airlock contract (same discipline as langgraph_orchestrator.py):
    - vocal_path is always a local path string; raw bytes never enter graph state
    - Only analyze_node, transcribe_node, and mix_node read the audio file
"""

import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from langgraph_orchestrator import get_llm

load_dotenv()
logger = logging.getLogger(__name__)

SARVAM_MAX_CLIP_SECONDS = 30.0
MAX_BACKING_SECONDS = 30.0


class OrchestrationState(TypedDict):
    # Inputs — set once at entry, never mutated
    session_id:      str
    vocal_path:      str    # AIRLOCK: path string only, never bytes
    vocal_ext:       str
    mood_hint:       str

    # Analysis outputs
    tempo_bpm:       Optional[float]
    key:             Optional[str]
    duration_sec:    Optional[float]

    # Transcription outputs (best-effort)
    transcript:      str
    language_code:   str

    # Composition
    music_prompt:    str
    backing_path:    Optional[str]
    mixed_path:      Optional[str]

    # Final
    agent_response:  str
    error:           Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Analyze — tempo/key/duration from the raw vocal
# ─────────────────────────────────────────────────────────────────────────────

def analyze_node(state: OrchestrationState) -> dict:
    logger.info(f"[ANALYZE] session={state['session_id']}")
    try:
        from orchestration_pipeline import analyze_vocal
        with open(state["vocal_path"], "rb") as f:
            vocal_bytes = f.read()
        analysis = analyze_vocal(vocal_bytes, state["vocal_ext"])
        return {
            "tempo_bpm":    analysis["tempo_bpm"],
            "key":          analysis["key"],
            "duration_sec": analysis["duration_sec"],
            "error":        None,
        }
    except Exception as e:
        logger.error(f"[ANALYZE] failed: {e}")
        return {
            "tempo_bpm": None, "key": None, "duration_sec": None,
            "error": f"Vocal analysis failed: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Transcribe — best-effort Sarvam STT on a trimmed clip
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_node(state: OrchestrationState) -> dict:
    logger.info(f"[TRANSCRIBE] session={state['session_id']}")
    try:
        from orchestration_pipeline import trim_audio_bytes
        from integrations.sarvam_client import transcribe_vocal

        with open(state["vocal_path"], "rb") as f:
            vocal_bytes = f.read()
        trimmed = trim_audio_bytes(vocal_bytes, state["vocal_ext"], SARVAM_MAX_CLIP_SECONDS)
        result = transcribe_vocal(trimmed, "wav")
        return {"transcript": result["transcript"], "language_code": result["language_code"]}
    except Exception as e:
        logger.warning(f"[TRANSCRIBE] non-fatal failure: {e}")
        return {"transcript": "", "language_code": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Prompt synthesis — LLM turns analysis + transcript into a music prompt
# ─────────────────────────────────────────────────────────────────────────────

def prompt_synthesis_node(state: OrchestrationState) -> dict:
    logger.info(f"[PROMPT] session={state['session_id']}")

    tempo = state.get("tempo_bpm") or 100
    key = state.get("key") or "C major"
    mood_hint = state.get("mood_hint") or ""
    transcript = (state.get("transcript") or "")[:300]
    language = state.get("language_code") or ""

    fallback_prompt = (
        f"EXACTLY {tempo:.0f} BPM in {key}. An instrumental backing track"
        + (f", {mood_hint} style" if mood_hint else "")
        + ", tasteful arrangement that leaves space for lead vocals. Strict tempo, no rubato."
    )

    try:
        llm = get_llm()
        prompt = f"""
You are a music producer writing a text prompt for an AI music generator
(ElevenLabs Music). It will generate an INSTRUMENTAL backing track to sit
underneath a vocal recording someone just sang. Tempo drift is the most
common failure mode for this kind of generation, so the prompt MUST open
by stating the BPM and key as hard constraints, not stylistic suggestions.

Detected tempo: {tempo:.0f} BPM
Detected key: {key}
User's mood/genre hint (may be empty): {mood_hint or "none given"}
Vocal transcript (may be empty/partial): {transcript or "none"}
Detected language: {language or "unknown"}

Write ONE concise music-generation prompt (max 40 words). It MUST start
with "EXACTLY {tempo:.0f} BPM in {key}." verbatim, then describe genre,
mood, and instrumentation. It must produce a steady, click-track-accurate
instrumental that complements the vocal, not competes with it. Respond
with ONLY the prompt text, no quotes, no markdown.
""".strip()

        response = llm.invoke(prompt)
        music_prompt = response.text.strip().strip('"')
        if not music_prompt:
            music_prompt = fallback_prompt

    except Exception as e:
        logger.warning(f"[PROMPT] LLM failed ({e}) — using fallback template")
        music_prompt = fallback_prompt

    logger.info(f"[PROMPT] {music_prompt}")
    return {"music_prompt": music_prompt}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: Compose — ElevenLabs Music generates the backing track
# ─────────────────────────────────────────────────────────────────────────────

def compose_node(state: OrchestrationState) -> dict:
    logger.info(f"[COMPOSE] session={state['session_id']}")
    try:
        from integrations.elevenlabs_client import compose_music
        from orchestration_pipeline import tempo_lock_backing

        # Cap the generated loop at 30s regardless of vocal length — a
        # shorter loop stays in tighter sync and is cheaper to generate;
        # mix_node tiles it to cover the full vocal duration.
        requested_sec = min(state.get("duration_sec") or 20.0, MAX_BACKING_SECONDS)
        duration_ms = int(requested_sec * 1000)
        backing_bytes = compose_music(state["music_prompt"], duration_ms)

        backing_bytes = tempo_lock_backing(
            state.get("tempo_bpm"), backing_bytes, "mp3"
        )

        base, _ = os.path.splitext(state["vocal_path"])
        backing_path = f"{base}_backing.wav"
        with open(backing_path, "wb") as f:
            f.write(backing_bytes)

        logger.info(f"[COMPOSE] backing track → {backing_path}")
        return {"backing_path": backing_path, "error": None}

    except Exception as e:
        logger.error(f"[COMPOSE] failed: {e}")
        return {"backing_path": None, "error": f"Music generation failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Mix — enhance vocal + blend with backing track
# ─────────────────────────────────────────────────────────────────────────────

def mix_node(state: OrchestrationState) -> dict:
    logger.info(f"[MIX] session={state['session_id']}")
    try:
        from dsp_pipeline import apply_timbre_enhancement
        from orchestration_pipeline import mix_vocal_with_backing

        with open(state["vocal_path"], "rb") as f:
            vocal_bytes = f.read()
        enhanced_vocal = apply_timbre_enhancement(vocal_bytes, state["vocal_ext"])

        with open(state["backing_path"], "rb") as f:
            backing_bytes = f.read()

        mixed_bytes = mix_vocal_with_backing(enhanced_vocal, "wav", backing_bytes, "wav")

        base, _ = os.path.splitext(state["vocal_path"])
        mixed_path = f"{base}_mixed.wav"
        with open(mixed_path, "wb") as f:
            f.write(mixed_bytes)

        logger.info(f"[MIX] mixed track → {mixed_path}")
        return {"mixed_path": mixed_path, "error": None}

    except Exception as e:
        logger.error(f"[MIX] failed: {e}")
        return {"mixed_path": None, "error": f"Mixing failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Synthesis — user-facing summary
# ─────────────────────────────────────────────────────────────────────────────

def synthesis_node(state: OrchestrationState) -> dict:
    logger.info(f"[SYNTHESIS] session={state['session_id']}")

    if state.get("mixed_path"):
        summary = (
            f"Generated a {state.get('key', 'unknown key')} backing track at "
            f"{state.get('tempo_bpm', '?')} BPM and mixed it under your vocal."
        )
    elif state.get("error"):
        summary = f"Couldn't generate a backing track: {state['error']}. Your enhanced vocal is still available."
    else:
        summary = "Processing incomplete."

    try:
        llm = get_llm()
        prompt = f"""
You are Auvia, an AI assistant for independent musicians.
Explain this "Sing & Orchestrate" result in clear, encouraging, practical terms.

What happened: {summary}
Music prompt used: {state.get('music_prompt', '')}
Detected tempo/key: {state.get('tempo_bpm', '?')} BPM, {state.get('key', '?')}
Transcript (if any): {(state.get('transcript') or '')[:200]}

Give a concise, friendly 2-3 sentence response.
""".strip()
        response = llm.invoke(prompt)
        agent_response = response.text
    except Exception as e:
        logger.error(f"[SYNTHESIS] LLM failed: {e}")
        agent_response = summary

    return {"agent_response": agent_response}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_compose(state: OrchestrationState) -> Literal["mix_node", "synthesis_node"]:
    if state.get("error") or not state.get("backing_path"):
        return "synthesis_node"
    return "mix_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder — called once at app startup
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestration_graph():
    g = StateGraph(OrchestrationState)

    g.add_node("analyze_node",           analyze_node)
    g.add_node("transcribe_node",        transcribe_node)
    g.add_node("prompt_synthesis_node",  prompt_synthesis_node)
    g.add_node("compose_node",           compose_node)
    g.add_node("mix_node",               mix_node)
    g.add_node("synthesis_node",         synthesis_node)

    g.add_edge(START, "analyze_node")
    g.add_edge("analyze_node", "transcribe_node")
    g.add_edge("transcribe_node", "prompt_synthesis_node")
    g.add_edge("prompt_synthesis_node", "compose_node")
    g.add_conditional_edges(
        "compose_node",
        route_after_compose,
        {
            "mix_node":       "mix_node",
            "synthesis_node": "synthesis_node",
        }
    )
    g.add_edge("mix_node", "synthesis_node")
    g.add_edge("synthesis_node", END)

    return g.compile()


# Graph is compiled lazily via main.get_orchestration_graph()
