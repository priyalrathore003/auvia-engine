# Auvia Engine — GCP Cloud Run
FROM python:3.11-slim

WORKDIR /app

# System deps: librosa (soundfile), MP3 decode (ffmpeg), gcc (webrtcvad C extension build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# librosa imports pkg_resources at load time
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time into /tmp (avoids cold-start HF fetch)
# /tmp is the only writable directory on Cloud Run
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface
RUN mkdir -p /tmp/huggingface && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || echo "WARN: HF model pre-download failed (will lazy-load at runtime)"

COPY main.py dsp_pipeline.py langgraph_orchestrator.py rag_storage.py ./
COPY orchestration_pipeline.py orchestration_graph.py ./
COPY voice_agent_pipeline.py voice_agent_ws.py ./
COPY intelligence_pipeline.py usage_tracking.py ./
COPY integrations/ integrations/
COPY telephony/ telephony/
COPY static/ static/

# Cloud Run writable temp + model cache
ENV PORT=8080
ENV TEMP_DIR=/tmp/auvia
ENV CHROMA_PERSIST_DIR=/tmp/chroma_db
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Shell form so Cloud Run's injected $PORT is expanded
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 120