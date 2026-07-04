# Auvia Engine — GCP Cloud Run
# Port MUST be 8080 for Cloud Run (Render used 10000 — changed here)

FROM python:3.11-slim

WORKDIR /app

# System deps for librosa + noisereduce
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY . .

# Cloud Run requires PORT env var respected
# Default 8080 — Cloud Run injects $PORT automatically
ENV PORT=8080

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}