#!/usr/bin/env bash
# Deploy Auvia Engine to GCP Cloud Run
# Usage: ./scripts/deploy-cloudrun.sh [PROJECT_ID] [REGION]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

PROJECT_ID="${1:-${GCP_PROJECT_ID:-}}"
REGION="${2:-asia-south1}"
SERVICE="auvia-engine"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Usage: $0 PROJECT_ID [REGION]"
  echo "  or set GCP_PROJECT_ID env var"
  exit 1
fi

# Pull API keys from .env — Secret Manager isn't enabled on this project,
# so this matches how the service is actually run today (plain env vars).
if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_DIR}/.env"
  set +a
fi

echo "==> Project: ${PROJECT_ID} | Region: ${REGION}"

gcloud config set project "${PROJECT_ID}"

echo "==> Enabling APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com

echo "==> Building image..."
gcloud builds submit --tag "${IMAGE}" "${REPO_DIR}"

echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "LLM_PROVIDER=${LLM_PROVIDER:-gemini},CHROMA_PERSIST_DIR=/tmp/chroma_db,TEMP_DIR=/tmp/auvia,GEMINI_API_KEY=${GEMINI_API_KEY:-},GROQ_API_KEY=${GROQ_API_KEY:-},ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-},SARVAM_API_KEY=${SARVAM_API_KEY:-},RAZORPAY_KEY_ID=${RAZORPAY_KEY_ID:-},RAZORPAY_KEY_SECRET=${RAZORPAY_KEY_SECRET:-},RAZORPAY_WEBHOOK_SECRET=${RAZORPAY_WEBHOOK_SECRET:-},RAZORPAY_PRO_PLAN_ID=${RAZORPAY_PRO_PLAN_ID:-},RAZORPAY_STUDIO_PLAN_ID=${RAZORPAY_STUDIO_PLAN_ID:-},RAZORPAY_PRO_ANNUAL_PLAN_ID=${RAZORPAY_PRO_ANNUAL_PLAN_ID:-},RAZORPAY_STUDIO_ANNUAL_PLAN_ID=${RAZORPAY_STUDIO_ANNUAL_PLAN_ID:-}"

echo "==> Done. Service URL:"
gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)'
