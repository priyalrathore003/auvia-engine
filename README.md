# Auvia Engine 🎵

[![Live Demo](https://img.shields.io/badge/Live_Demo-Auvia_Engine-7C3AED?style=flat-square)](https://auvia-engine-754304552652.asia-south1.run.app)

**[→ Try the live demo](https://auvia-engine-754304552652.asia-south1.run.app)**

**Agentic audio intelligence backend for independent musicians.**

An AI agent that accepts a natural-language instruction + WAV audio, reasons about audio issues using a RAG knowledge base, runs a DSP enhancement pipeline, and returns professionally processed audio — all without raw audio bytes ever crossing the LLM boundary.

---

## Architecture

```
Client (Streamlit)
       │
       │  audio_b64 + query
       ▼
FastAPI Gateway  ──── decodes base64 ──▶  /tmp/auvia_input_*.wav
       │                                          │
       │  file path (string only)                 │
       ▼                                          │
LangChain Agent (Claude)                          │
       │                                          │
       ├── query_audio_transcripts ──▶  ChromaDB RAG  (HuggingFace embeddings)
       │                                               known issue patterns
       └── enhance_audio ────────────────────────────▶ DSP Pipeline
                                                        │
                                          ┌─────────────┴──────────────┐
                                          │  1. Noise reduction        │
                                          │     (noisereduce / spectral │
                                          │      gating, n_fft=2048)   │
                                          │  2. Spectral shaping       │
                                          │     (librosa STFT EQ)      │
                                          │     · Sub-bass roll-off    │
                                          │     · Low-mid warmth       │
                                          │     · Presence lift        │
                                          │     · Peak normalize -1dBFS│
                                          └─────────────┬──────────────┘
                                                        │
       ◀─────── processed_audio_b64 ───────────────────┘
```

**Airlock pattern:** Raw audio bytes are intercepted at the FastAPI gateway and written to a local temp path. The LangChain agent only ever receives and passes file path strings — never raw base64. This prevents token explosion from binary payloads inside LLM prompts.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangChain (≥1.3), Anthropic Claude |
| RAG / Vector store | ChromaDB + HuggingFace `all-MiniLM-L6-v2` |
| DSP pipeline | `noisereduce`, `librosa`, `soundfile`, `numpy` |
| Backend API | FastAPI + Uvicorn (async) |
| Frontend | Streamlit |
| Containerisation | Docker (Python 3.11-slim) |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/process-audio` | Full agent path: query + audio → agent → RAG → DSP → enhanced audio |
| `POST` | `/enhance-audio` | DSP-only bypass: audio → pipeline → enhanced audio (no LLM) |

**Request (multipart or JSON):**
```json
{
  "query": "Remove the background hum and warm up the vocals",
  "audio_b64": "<base64 encoded WAV>"
}
```

**Response:**
```json
{
  "status": "success",
  "result": "<agent reasoning text>",
  "processed_audio_b64": "<base64 encoded WAV>"
}
```

---

## DSP Pipeline

Two-stage audio enhancement in `dsp_pipeline.py`:

**Stage 1 — Noise Reduction**
Spectral gating via `noisereduce` (`n_fft=2048`, `hop_length=512`)

**Stage 2 — Spectral Shaping (librosa STFT EQ)**
```
Sub-bass roll-off  : < 80 Hz    × 0.3   (remove mud)
Low-mid warmth     : 150–500 Hz × 1.4   (body + warmth)
Presence lift      : 2–5 kHz    × 1.15  (vocal clarity)
Output             : Peak normalize to −1 dBFS, 16-bit PCM WAV
```

---

## RAG Layer

`rag_storage.py` — ChromaDB persisted at `./chroma_db`

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)
- Seeded with audio issue patterns: hum, sibilance, clipping
- Tool exposed to agent: `query_audio_transcripts(query)` → relevant context

---

## Run Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-handle>/auvia-engine.git
cd auvia-engine
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 3. Terminal 1 — start API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Terminal 2 — start UI
streamlit run app.py
```

- API: http://localhost:8000 (interactive docs at `/docs`)
- UI: http://localhost:8501

> **No Anthropic credits?** Use `POST /enhance-audio` for the DSP-only path — no LLM required.

---

## Project Structure

```
auvia-engine/
├── main.py              # FastAPI app, LangChain agent, API endpoints
├── app.py               # Streamlit UI (upload, play, download)
├── agent_gateway.py     # LangChain @tool wrappers for DSP + RAG
├── dsp_pipeline.py      # Core DSP: noise reduction + spectral shaping
├── rag_storage.py       # ChromaDB + HuggingFace embeddings
├── requirements.txt
├── Dockerfile
└── .env                 # Local only — never committed
```

---

## What's Next

- [ ] Migrate agent orchestration to **LangGraph** state machine (deterministic routing)
- [ ] Deploy to **GCP Cloud Run** (scale-to-zero, replaces Render)
- [ ] Add format support beyond WAV (MP3, FLAC)
- [ ] Productionise RAG with real session transcripts
- [ ] Add observability layer (Langfuse / structured logging)

---

## Built by

**Priyal Rathore** — AI Engineer & Founder, Auvia  
Trained vocalist building AI tools for musicians who can't afford a studio.

[LinkedIn](https://www.linkedin.com/in/priyalrathore17) · auviaengine.com
