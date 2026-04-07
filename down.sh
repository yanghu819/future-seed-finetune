#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p artifacts cache runs logs

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B-Base}"
DOWNLOAD_MODE="${DOWNLOAD_MODE:-none}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/artifacts/models/qwen3_5_9b_base_probe}"

if [[ "${DOWNLOAD_MODE}" == "none" ]]; then
  echo "No external assets required for smoke."
  exit 0
fi

CMD=(uv run python scripts/download_qwen35_probe_assets.py --model-id "${MODEL_ID}" --output-dir "${MODEL_DIR}")
if [[ "${DOWNLOAD_MODE}" == "full" ]]; then
  CMD+=(--full-weights)
fi

"${CMD[@]}"
