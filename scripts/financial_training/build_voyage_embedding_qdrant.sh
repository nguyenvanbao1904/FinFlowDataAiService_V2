#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${VOYAGE_EMBED_MODEL:-voyage-3.5-lite}"
COLLECTION="${VOYAGE_QDRANT_COLLECTION:-${CHAT_QDRANT_COLLECTION:-annual_report_chunks_voyage_3_5_lite}}"
BATCH_SIZE="${VOYAGE_EMBED_BATCH_SIZE:-64}"
MAX_INPUT_CHARS="${VOYAGE_EMBED_MAX_INPUT_CHARS:-12000}"

if [[ -z "${VOYAGE_API_KEY:-}" ]]; then
  echo "[VOYAGE][ERR] VOYAGE_API_KEY is missing. Set it in data_ai_service/.env"
  exit 2
fi

venv/bin/python scripts/financial_training/embed_annual_reports_chunks_voyage.py \
  --embed-base-url https://api.voyageai.com/v1 \
  --embed-api-key "$VOYAGE_API_KEY" \
  --embed-model "$MODEL" \
  --embed-input-type document \
  --batch-size "$BATCH_SIZE" \
  --max-input-chars "$MAX_INPUT_CHARS" \
  --timeout-seconds 180 \
  --max-retries 5 \
  --retry-sleep-seconds 2 \
  --qdrant-upsert \
  --qdrant-url "${CHAT_QDRANT_URL:-http://127.0.0.1:6333}" \
  --qdrant-api-key "${CHAT_QDRANT_API_KEY:-}" \
  --qdrant-collection "$COLLECTION" \
  "$@"
