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
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
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
        "intelligence": True,
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


@app.post("/intelligence")
async def intelligence(
    file: UploadFile = File(..., description="WAV or MP3 audio file"),
):
    """Acoustic intelligence: pitch/note/key/tempo + quality metrics as JSON. No LLM."""
    ext = get_ext(file)

    try:
        audio_bytes = await file.read()
        from intelligence_pipeline import analyze_intelligence

        report = await asyncio.to_thread(analyze_intelligence, audio_bytes, ext)

        return JSONResponse({
            "status": "success",
            **report,
        })

    except Exception as e:
        logger.exception("intelligence error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/orchestrate")
async def orchestrate(
    file: UploadFile = File(..., description="Recorded vocal (wav/mp3/webm/ogg)"),
    mood: str = Form(default=""),
    x_device_id: str = Header(default=None, alias="X-Device-Id"),
):
    """Sing & Orchestrate: vocal recording → LangGraph (analyze + Sarvam STT +
    ElevenLabs Music) → mixed, downloadable track.

    Gated on the free tier (3 uses per anonymous device) if the caller sends
    an X-Device-Id header — omitted entirely for callers that don't send one
    (e.g. direct API usage), so this only gates the web UI's casual-use path.
    """
    if x_device_id:
        from usage_tracking import check_and_increment

        usage = check_and_increment(x_device_id)
        if not usage["allowed"]:
            return JSONResponse(
                status_code=402,
                content={
                    "status": "upgrade_required",
                    "message": "You've used your 3 free orchestrations. Upgrade to Pro for unlimited use.",
                },
            )

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


@app.post("/api/create-subscription")
async def create_subscription(
    plan: str = Form(..., description="'pro' or 'studio'"),
    billing: str = Form(default="monthly", description="'monthly' or 'annual'"),
    x_device_id: str = Header(..., alias="X-Device-Id"),
):
    """Creates a Razorpay subscription for the given plan and returns a
    hosted checkout URL. The subscription is tagged with the caller's
    anonymous device ID so the webhook can upgrade the right device once
    payment is confirmed."""
    if plan not in ("pro", "studio"):
        raise HTTPException(status_code=400, detail="plan must be 'pro' or 'studio'")
    if billing not in ("monthly", "annual"):
        raise HTTPException(status_code=400, detail="billing must be 'monthly' or 'annual'")

    try:
        from integrations.razorpay_client import create_subscription as rzp_create_subscription

        result = await asyncio.to_thread(rzp_create_subscription, plan, x_device_id, billing)
        return JSONResponse({
            "status": "success",
            "checkout_url": result["checkout_url"],
            "subscription_id": result["subscription_id"],
            "billing": result["billing"],
        })

    except Exception as e:
        logger.exception("create-subscription error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/razorpay-webhook")
async def razorpay_webhook(request: Request):
    """Razorpay calls this on subscription lifecycle events. Verifies the
    HMAC signature, then on activation/charge marks the tagged device as
    Pro so /orchestrate stops gating it."""
    from integrations.razorpay_client import verify_webhook_signature
    from usage_tracking import mark_pro

    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not verify_webhook_signature(raw_body, signature):
        raise HTTPException(status_code=400, detail="invalid webhook signature")

    payload = await request.json()
    event = payload.get("event", "")
    logger.info("[RAZORPAY] webhook event=%s", event)

    if event in ("subscription.activated", "subscription.charged"):
        subscription = payload.get("payload", {}).get("subscription", {}).get("entity", {})
        notes = subscription.get("notes", {}) or {}
        device_id = notes.get("device_id")
        plan = notes.get("plan", "pro")

        if device_id:
            mark_pro(device_id, plan, subscription.get("id", ""))
        else:
            logger.warning("[RAZORPAY] webhook %s had no device_id in notes", event)

    return JSONResponse({"status": "ok"})


@app.websocket("/ws/voice-agent")
async def voice_agent_ws(websocket: WebSocket):
    """Real-time voice conversation: mic PCM16 in, VAD-driven turn-taking,
    Sarvam STT → LLM → ElevenLabs TTS, spoken reply + latency breakdown out."""
    from voice_agent_ws import handle_voice_session
    await handle_voice_session(websocket)


def _serve_static_page(filename: str):
    page = STATIC_DIR / filename
    if not page.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(page)


@app.get("/")
def serve_frontend():
    return _serve_static_page("index.html")


@app.get("/api")
def serve_api_docs():
    return _serve_static_page("api.html")


@app.get("/studio")
def serve_studio():
    return _serve_static_page("studio.html")


@app.get("/pricing")
def serve_pricing():
    return _serve_static_page("pricing.html")


@app.get("/privacy")
def serve_privacy():
    return _serve_static_page("privacy.html")


@app.get("/terms")
def serve_terms():
    return _serve_static_page("terms.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
