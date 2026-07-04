"""
langgraph_orchestrator.py
Replaces create_agent() in main.py with a deterministic LangGraph state machine.

Graph topology:
    START → diagnostic_node → [route] → execution_node → synthesis_node → END
                                      ↘ synthesis_node (no DSP needed)

Airlock contract (never broken):
    - audio_file_path is always a local path string
    - Raw base64 / bytes never enter graph state
    - Only dsp_pipeline.py reads the file

LLM provider: Groq (free) by default. Swap via LLM_PROVIDER env var.
    LLM_PROVIDER=groq      → Llama-3 on Groq (free, recommended)
    LLM_PROVIDER=openai    → GPT-4o-mini
    LLM_PROVIDER=anthropic → Claude (when credits restored)
"""

import json
import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory — swap provider by setting LLM_PROVIDER env var
# ─────────────────────────────────────────────────────────────────────────────

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama3-8b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use groq/openai/anthropic")


# ─────────────────────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────────────────────

class AudioState(TypedDict):
    # Inputs — set once at entry, never mutated
    session_id:       str
    audio_file_path:  str    # AIRLOCK: path string only, never bytes
    user_query:       str

    # Diagnostic outputs
    rag_context:      str
    requires_dsp:     bool
    reasoning:        str

    # Execution outputs
    dsp_success:      bool
    enhanced_path:    Optional[str]
    dsp_error:        Optional[str]

    # Final
    agent_response:   str
    error:            Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Diagnostic — RAG retrieval + DSP routing decision
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_node(state: AudioState) -> dict:
    """
    1. Queries ChromaDB for relevant context (via rag_storage)
    2. Asks LLM: does this request need DSP?
    Returns routing decision + RAG context.
    """
    logger.info(f"[DIAGNOSTIC] session={state['session_id']}")

    # RAG retrieval
    rag_context = ""
    try:
        from rag_storage import query_audio_transcripts
        rag_context = query_audio_transcripts(state["user_query"])
        logger.info(f"[DIAGNOSTIC] RAG context retrieved: {len(rag_context)} chars")
    except Exception as e:
        logger.warning(f"[DIAGNOSTIC] RAG unavailable: {e}")
        rag_context = "No prior context available."

    # LLM routing decision
    requires_dsp = bool(state["audio_file_path"])  # safe default
    reasoning    = "Audio file present — defaulting to DSP."

    try:
        llm = get_llm()
        prompt = f"""
You are an audio processing router. Decide if DSP enhancement is needed.

User request: {state['user_query']}
Audio file present: {bool(state['audio_file_path'])}
Retrieved context: {rag_context[:500]}

Respond ONLY as valid JSON (no markdown):
{{"requires_dsp": true/false, "reasoning": "<one line>"}}
""".strip()

        response = llm.invoke(prompt)
        raw      = response.content.strip().strip("```json").strip("```").strip()
        parsed   = json.loads(raw)
        requires_dsp = bool(parsed.get("requires_dsp", requires_dsp))
        reasoning    = parsed.get("reasoning", reasoning)

    except Exception as e:
        logger.warning(f"[DIAGNOSTIC] LLM routing failed ({e}) — using defaults")

    logger.info(f"[DIAGNOSTIC] requires_dsp={requires_dsp} | {reasoning}")
    return {
        "rag_context":   rag_context,
        "requires_dsp":  requires_dsp,
        "reasoning":     reasoning,
        "error":         None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Execution — runs DSP pipeline (ONLY node that touches audio file)
# ─────────────────────────────────────────────────────────────────────────────

def execution_node(state: AudioState) -> dict:
    """
    Calls dsp_pipeline.apply_timbre_enhancement() with the audio file path.
    This is the ONLY place audio bytes are read — Airlock enforced.
    """
    logger.info(f"[EXECUTION] path={state['audio_file_path']}")

    try:
        from dsp_pipeline import apply_timbre_enhancement
        with open(state["audio_file_path"], "rb") as f:
            audio_bytes = f.read()

        enhanced_bytes = apply_timbre_enhancement(audio_bytes)

        # Write enhanced audio to temp path
        enhanced_path = state["audio_file_path"].replace(".wav", "_enhanced.wav")
        with open(enhanced_path, "wb") as f:
            f.write(enhanced_bytes)

        logger.info(f"[EXECUTION] DSP complete → {enhanced_path}")
        return {
            "dsp_success":  True,
            "enhanced_path": enhanced_path,
            "dsp_error":    None,
        }

    except Exception as e:
        logger.error(f"[EXECUTION] DSP failed: {e}")
        return {
            "dsp_success":   False,
            "enhanced_path": None,
            "dsp_error":     str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: Synthesis — generates user-facing response
# ─────────────────────────────────────────────────────────────────────────────

def synthesis_node(state: AudioState) -> dict:
    """Combines DSP results + RAG context into a musician-friendly response."""
    logger.info(f"[SYNTHESIS] session={state['session_id']}")

    dsp_summary = ""
    if state.get("dsp_success"):
        dsp_summary = "Audio enhanced: noise reduced, spectral EQ applied (sub-bass roll-off, low-mid warmth, presence lift), peak normalized to -1 dBFS."
    elif state.get("dsp_error"):
        dsp_summary = f"DSP failed: {state['dsp_error']}"
    else:
        dsp_summary = "No DSP processing performed."

    try:
        llm = get_llm()
        prompt = f"""
You are Auvia, an AI assistant for independent musicians.
Explain the audio processing results in clear, practical terms.

User asked: {state['user_query']}
Processing result: {dsp_summary}
Relevant context: {state.get('rag_context', '')[:400]}
Routing reasoning: {state.get('reasoning', '')}

Give a concise, helpful response. Be specific about what was done and why.
""".strip()

        response = llm.invoke(prompt)
        agent_response = response.content

    except Exception as e:
        logger.error(f"[SYNTHESIS] LLM failed: {e}")
        agent_response = f"Processing complete. {dsp_summary}"

    return {"agent_response": agent_response}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_diagnostic(
    state: AudioState
) -> Literal["execution_node", "synthesis_node"]:
    if state.get("error"):
        return "synthesis_node"
    if state.get("requires_dsp") and state.get("audio_file_path"):
        return "execution_node"
    return "synthesis_node"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder — called once at app startup
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AudioState)

    g.add_node("diagnostic_node", diagnostic_node)
    g.add_node("execution_node",  execution_node)
    g.add_node("synthesis_node",  synthesis_node)

    g.add_edge(START, "diagnostic_node")
    g.add_conditional_edges(
        "diagnostic_node",
        route_after_diagnostic,
        {
            "execution_node": "execution_node",
            "synthesis_node": "synthesis_node",
        }
    )
    g.add_edge("execution_node", "synthesis_node")
    g.add_edge("synthesis_node", END)

    return g.compile()


# Singleton — reused across all requests
audio_graph = build_graph()