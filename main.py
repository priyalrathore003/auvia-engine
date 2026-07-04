"""
main.py — Auvia Engine (dev/langgraph)
UploadFile endpoints (WAV + MP3), serves static frontend, LangGraph orchestration.
"""

import io
import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

from langgraph_orchestrator import audio_graph
from dsp_pipeline import apply_timbre_enhancement

TEMP_DIR = "/tmp/auvia"
os.makedirs(TEMP_DIR, exist_ok=True)
ALLOWED_TYPES = {"audio/wav", "audio/wave", "audio/mpeg", "audio/mp3", "audio/x-wav"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Auvia Engine starting | LLM={os.getenv('LLM_PROVIDER', 'groq')}")
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


# ── Airlock helper ────────────────────────────────────────────────────────────

def write_temp(audio_bytes: bytes, ext: str, session_id: str) -> str:
    path = os.path.join(TEMP_DIR, f"{session_id}_input.{ext}")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"[AIRLOCK] {len(audio_bytes):,} bytes → {path}")
    return path


def cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning(f"[CLEANUP] {e}")


def get_ext(file: UploadFile) -> str:
    ct = (file.content_type or "").lower()
    name = (file.filename or "").lower()
    if "mpeg" in ct or "mp3" in ct or name.endswith(".mp3"):
        return "mp3"
    return "wav"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
        "graph": "langgraph",
        "formats": ["wav", "mp3"],
    }


@app.post("/process-audio")
async def process_audio(
    file:  UploadFile = File(..., description="WAV or MP3 audio file"),
    query: str        = Form(default="Enhance this audio recording."),
):
    """Full agentic path: file + query → LangGraph (RAG + DSP) → enhanced audio download."""
    session_id = str(uuid.uuid4())
    ext        = get_ext(file)
    input_path = None

    try:
        audio_bytes = await file.read()
        input_path  = write_temp(audio_bytes, ext, session_id)

        initial_state = {
            "session_id":      session_id,
            "audio_file_path": input_path,
            "user_query":      query,
            "rag_context":     "",
            "requires_dsp":    False,
            "reasoning":       "",
            "dsp_success":     False,
            "enhanced_path":   None,
            "dsp_error":       None,
            "agent_response":  "",
            "error":           None,
        }

        result = audio_graph.invoke(initial_state)

        # Read enhanced audio
        enhanced_bytes = None
        if result.get("enhanced_path") and os.path.exists(result["enhanced_path"]):
            with open(result["enhanced_path"], "rb") as f:
                enhanced_bytes = f.read()
            cleanup(result["enhanced_path"])
        else:
            # DSP didn't run — return original processed through DSP directly
            enhanced_bytes = apply_timbre_enhancement(audio_bytes, ext)

        return JSONResponse({
            "status":       "success",
            "session_id":   session_id,
            "result":       result.get("agent_response", ""),
            "dsp_performed": result.get("dsp_success", False),
            "reasoning":    result.get("reasoning", ""),
            "audio_b64":    __import__("base64").b64encode(enhanced_bytes).decode(),
            "filename":     f"auvia_enhanced_{session_id[:8]}.wav",
        })

    except Exception as e:
        logger.exception("process-audio error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup(input_path)


@app.post("/enhance-audio")
async def enhance_audio(
    file: UploadFile = File(..., description="WAV or MP3 audio file"),
):
    """DSP-only bypass — no LLM. Always works regardless of API credits."""
    session_id = str(uuid.uuid4())
    ext        = get_ext(file)

    try:
        audio_bytes    = await file.read()
        enhanced_bytes = apply_timbre_enhancement(audio_bytes, ext)

        return JSONResponse({
            "status":       "success",
            "result":       "Audio enhanced: noise reduction + spectral EQ applied.",
            "dsp_performed": True,
            "audio_b64":    __import__("base64").b64encode(enhanced_bytes).decode(),
            "filename":     f"auvia_enhanced_{session_id[:8]}.wav",
        })

    except Exception as e:
        logger.exception("enhance-audio error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve frontend ────────────────────────────────────────────────────────────
# Must be AFTER API routes — catches everything else

from fastapi.responses import FileResponse

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def serve_frontend():
    path = os.path.join(BASE_DIR, "static", "index.html")
    return FileResponse(path)