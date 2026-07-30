"""
main.py — Auvia Engine (dev/langgraph)
UploadFile endpoints (WAV + MP3), serves static frontend, LangGraph orchestration.
"""

import asyncio
import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/auvia")
os.makedirs(TEMP_DIR, exist_ok=True)

# Lazy graph import — avoids loading LangGraph before uvicorn binds
_audio_graph = None
_orchestration_graph = None


def get_audio_graph():
    global _audio_graph
    if _audio_graph is None:
        from langgraph_orchestrator import build_graph
        _audio_graph = build_graph()
        logger.info("LangGraph compiled")
    return _audio_graph


def get_orchestration_graph():
    global _orchestration_graph
    if _orchestration_graph is None:
        from orchestration_graph import build_orchestration_graph
        _orchestration_graph = build_orchestration_graph()
        logger.info("Orchestration graph compiled")
    return _orchestration_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Auvia Engine starting | LLM=%s | temp=%s",
        os.getenv("LLM_PROVIDER", "groq"),
        TEMP_DIR,
    )
    yield
    logger.info("Auvia Engine shutting down.")


app = FastAPI(
    title="Auvia Engine API",
    version="2.0.0",
    description="Agentic audio processing — LangGraph + RAG + DSP pipeline",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def write_temp(audio_bytes: bytes, ext: str, session_id: str) -> str:
    path = os.path.join(TEMP_DIR, f"{session_id}_input.{ext}")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    logger.info("[AIRLOCK] %s bytes → %s", f"{len(audio_bytes):,}", path)
    return path


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError as e:
            logger.warning("[CLEANUP] %s", e)


def get_ext(file: UploadFile) -> str:
    ct = (file.content_type or "").lower()
    name = (file.filename or "").lower()
    if "webm" in ct or name.endswith(".webm"):
        return "webm"
    if "ogg" in ct or name.endswith(".ogg"):
        return "ogg"
    if "mpeg" in ct or "mp3" in ct or name.endswith(".mp3"):
        return "mp3"
    return "wav"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
        "graph": "langgraph",
        "formats": ["wav", "mp3", "webm", "ogg"],
        "orchestrate": bool(os.getenv("ELEVENLABS_API_KEY")),
    }


@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(..., description="WAV or MP3 audio file"),
    query: str = Form(default="Enhance this audio recording."),
):
    """Full agentic path: file + query → LangGraph (RAG + DSP) → enhanced audio."""
    session_id = str(uuid.uuid4())
    ext = get_ext(file)
    input_path = None

    try:
        audio_bytes = await file.read()
        input_path = write_temp(audio_bytes, ext, session_id)

        initial_state = {
            "session_id": session_id,
            "audio_file_path": input_path,
            "user_query": query,
            "rag_context": "",
            "requires_dsp": False,
            "reasoning": "",
            "dsp_success": False,
            "enhanced_path": None,
            "dsp_error": None,
            "agent_response": "",
            "error": None,
        }

        result = await asyncio.to_thread(get_audio_graph().invoke, initial_state)

        enhanced_bytes = None
        enhanced_path = result.get("enhanced_path")
        if enhanced_path and os.path.exists(enhanced_path):
            with open(enhanced_path, "rb") as f:
                enhanced_bytes = f.read()
            cleanup(enhanced_path)
        else:
            from dsp_pipeline import apply_timbre_enhancement
            enhanced_bytes = apply_timbre_enhancement(audio_bytes, ext)

        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "result": result.get("agent_response", ""),
            "dsp_performed": result.get("dsp_success", False),
            "reasoning": result.get("reasoning", ""),
            "audio_b64": base64.b64encode(enhanced_bytes).decode(),
            "filename": f"auvia_enhanced_{session_id[:8]}.wav",
        })

    except Exception as e:
        logger.exception("process-audio error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        cleanup(input_path)


@app.post("/enhance-audio")
async def enhance_audio(
    file: UploadFile = File(..., description="WAV or MP3 audio file"),
):
    """DSP-only bypass — no LLM. Always works regardless of API credits."""
    session_id = str(uuid.uuid4())
    ext = get_ext(file)

    try:
        audio_bytes = await file.read()
        from dsp_pipeline import apply_timbre_enhancement

        enhanced_bytes = await asyncio.to_thread(
            apply_timbre_enhancement, audio_bytes, ext
        )

        return JSONResponse({
            "status": "success",
            "result": "Audio enhanced: noise reduction + spectral EQ applied.",
            "dsp_performed": True,
            "audio_b64": base64.b64encode(enhanced_bytes).decode(),
            "filename": f"auvia_enhanced_{session_id[:8]}.wav",
        })

    except Exception as e:
        logger.exception("enhance-audio error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/orchestrate")
async def orchestrate(
    file: UploadFile = File(..., description="Recorded vocal (wav/mp3/webm/ogg)"),
    mood: str = Form(default=""),
):
    """Sing & Orchestrate: vocal recording → LangGraph (analyze + Sarvam STT +
    ElevenLabs Music) → mixed, downloadable track."""
    session_id = str(uuid.uuid4())
    ext = get_ext(file)
    vocal_path = None

    try:
        audio_bytes = await file.read()
        vocal_path = write_temp(audio_bytes, ext, session_id)

        initial_state = {
            "session_id":     session_id,
            "vocal_path":     vocal_path,
            "vocal_ext":      ext,
            "mood_hint":      mood,
            "tempo_bpm":      None,
            "key":            None,
            "duration_sec":   None,
            "transcript":     "",
            "language_code":  "",
            "music_prompt":   "",
            "backing_path":   None,
            "mixed_path":     None,
            "agent_response": "",
            "error":          None,
        }

        result = await asyncio.to_thread(get_orchestration_graph().invoke, initial_state)

        mixed_path = result.get("mixed_path")
        backing_path = result.get("backing_path")

        audio_b64 = None
        filename = f"auvia_orchestrated_{session_id[:8]}.wav"
        if mixed_path and os.path.exists(mixed_path):
            with open(mixed_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            cleanup(mixed_path)
        else:
            # Fall back to the DSP-enhanced vocal alone if composition/mixing failed
            from dsp_pipeline import apply_timbre_enhancement
            enhanced = apply_timbre_enhancement(audio_bytes, ext)
            audio_b64 = base64.b64encode(enhanced).decode()
            filename = f"auvia_vocal_enhanced_{session_id[:8]}.wav"

        instrumental_b64 = None
        if backing_path and os.path.exists(backing_path):
            with open(backing_path, "rb") as f:
                instrumental_b64 = base64.b64encode(f.read()).decode()
            cleanup(backing_path)

        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "music_prompt": result.get("music_prompt", ""),
            "tempo_bpm": result.get("tempo_bpm"),
            "key": result.get("key"),
            "transcript": result.get("transcript", ""),
            "language_code": result.get("language_code", ""),
            "agent_response": result.get("agent_response", ""),
            "audio_b64": audio_b64,
            "instrumental_b64": instrumental_b64,
            "filename": filename,
        })

    except Exception as e:
        logger.exception("orchestrate error")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        cleanup(vocal_path)


@app.get("/")
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
