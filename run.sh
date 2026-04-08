#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
. "${ROOT_DIR}/repo_env.sh"

RUN_MODE="${RUN_MODE:-smoke}"
FS_MODE="${FS_MODE:-enabled}"
EXPERIMENT_NAME_DEFAULT="qwen35-scalar-fs-smoke"
if [[ "${RUN_MODE}" == "validate-config" ]]; then
  EXPERIMENT_NAME_DEFAULT="qwen35-validate-config"
elif [[ "${RUN_MODE}" == "validate-pretrained" ]]; then
  EXPERIMENT_NAME_DEFAULT="qwen35-validate-pretrained"
elif [[ "${RUN_MODE}" == "train-smoke" ]]; then
  if [[ "${FS_MODE}" == "disabled" ]]; then
    EXPERIMENT_NAME_DEFAULT="qwen35-train-smoke-baseline"
  else
    EXPERIMENT_NAME_DEFAULT="qwen35-train-smoke"
  fi
elif [[ "${RUN_MODE}" == "train-pretrained" ]]; then
  if [[ "${FS_MODE}" == "disabled" ]]; then
    EXPERIMENT_NAME_DEFAULT="qwen35-train-pretrained-baseline"
  else
    EXPERIMENT_NAME_DEFAULT="qwen35-train-pretrained"
  fi
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_NAME_DEFAULT}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${ROOT_DIR}/runs/${EXPERIMENT_NAME}-${RUN_TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
ENTRYPOINT="scripts/smoke_qwen35_scalar_fs.py"
RESULT_FILE="${RESULT_DIR}/smoke_metrics.json"
ERROR_LOG="${LOG_DIR}/smoke.stderr.log"

case "${RUN_MODE}" in
  smoke)
    ;;
  validate-config|validate-pretrained)
    ENTRYPOINT="scripts/validate_qwen35_prefill.py"
    RESULT_FILE="${RESULT_DIR}/validate_metrics.json"
    ERROR_LOG="${LOG_DIR}/validate.stderr.log"
    ;;
  train-smoke|train-pretrained)
    ENTRYPOINT="scripts/train_awkward_scalar_fs.py"
    RESULT_FILE="${RESULT_DIR}/train_metrics.json"
    ERROR_LOG="${LOG_DIR}/train.stderr.log"
    ;;
  *)
    echo "Unknown RUN_MODE=${RUN_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo untracked)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export ROOT_DIR RUN_MODE EXPERIMENT_NAME RUN_DIR COMMIT_SHA STARTED_AT ENTRYPOINT

python3 - <<'PY' > "${RUN_DIR}/metadata.json"
import json
import os
from datetime import datetime, timezone

root = os.environ["ROOT_DIR"]
run_mode = os.environ["RUN_MODE"]
experiment_name = os.environ["EXPERIMENT_NAME"]
commit_sha = os.environ["COMMIT_SHA"]
started_at = os.environ["STARTED_AT"]

meta = {
    "commit_sha": commit_sha,
    "run_mode": run_mode,
    "config": {
        "entrypoint": os.environ.get("ENTRYPOINT", "scripts/smoke_qwen35_scalar_fs.py"),
        "experiment_name": experiment_name,
    },
    "started_at": started_at,
    "finished_at": None,
    "exit_status": None,
    "cost_estimate": "0",
    "gpu_type": "cpu-smoke",
}
print(json.dumps(meta, indent=2))
PY

set +e
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/artifacts/models/qwen3_5_9b_base_probe}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/artifacts/datasets/awkward_kv}"
LOAD_DTYPE="${LOAD_DTYPE:-float32}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"
ALPHA_INIT="${ALPHA_INIT:-0.25}"
START_LAYER="${START_LAYER:--1}"
SEED_CLIP_VALUE="${SEED_CLIP_VALUE:-1.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
FS_ALPHA_CLAMP="${FS_ALPHA_CLAMP:-0.0}"
SKIP_NONFINITE_LOSS="${SKIP_NONFINITE_LOSS:-0}"
OPTIMIZE_IN_EVAL_MODE="${OPTIMIZE_IN_EVAL_MODE:-0}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
  export ENTRYPOINT
  uv run python "${ENTRYPOINT}" > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
elif [[ "${RUN_MODE}" == "validate-config" ]]; then
  export ENTRYPOINT MODEL_DIR
  uv run python "${ENTRYPOINT}" --model-dir "${MODEL_DIR}" > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
elif [[ "${RUN_MODE}" == "validate-pretrained" ]]; then
  export ENTRYPOINT MODEL_DIR
  VALIDATE_CMD=(uv run python "${ENTRYPOINT}" --model-dir "${MODEL_DIR}" --from-pretrained --load-dtype "${LOAD_DTYPE}")
  if [[ "${LOW_CPU_MEM_USAGE}" == "1" ]]; then
    VALIDATE_CMD+=(--low-cpu-mem-usage)
  fi
  "${VALIDATE_CMD[@]}" > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
elif [[ "${RUN_MODE}" == "train-smoke" ]]; then
  export ENTRYPOINT MODEL_DIR DATASET_DIR
  TRAIN_CMD=(uv run python "${ENTRYPOINT}" \
    --dataset-dir "${DATASET_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --model-backend tiny \
    --output-dir "${RUN_DIR}/outputs" \
    --max-steps "${MAX_STEPS:-20}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --lr "${LR:-5e-4}" \
    --alpha-init "${ALPHA_INIT}" \
    --start-layer "${START_LAYER}" \
    --seed-clip-value "${SEED_CLIP_VALUE}" \
    --grad-clip-norm "${GRAD_CLIP_NORM}" \
    --fs-alpha-clamp "${FS_ALPHA_CLAMP}" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}")
  if [[ "${UNFREEZE_BACKBONE:-1}" == "1" ]]; then
    TRAIN_CMD+=(--unfreeze-backbone)
  fi
  if [[ "${FS_MODE}" == "disabled" ]]; then
    TRAIN_CMD+=(--disable-future-seed)
  fi
  if [[ "${SKIP_NONFINITE_LOSS}" == "1" ]]; then
    TRAIN_CMD+=(--skip-nonfinite-loss)
  fi
  if [[ "${OPTIMIZE_IN_EVAL_MODE}" == "1" ]]; then
    TRAIN_CMD+=(--optimize-in-eval-mode)
  fi
  "${TRAIN_CMD[@]}" > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
elif [[ "${RUN_MODE}" == "train-pretrained" ]]; then
  export ENTRYPOINT MODEL_DIR DATASET_DIR
  TRAIN_CMD=(uv run python "${ENTRYPOINT}" \
    --dataset-dir "${DATASET_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --model-backend pretrained \
    --output-dir "${RUN_DIR}/outputs" \
    --max-steps "${MAX_STEPS:-100}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --lr "${LR:-1e-4}" \
    --alpha-init "${ALPHA_INIT}" \
    --start-layer "${START_LAYER}" \
    --seed-clip-value "${SEED_CLIP_VALUE}" \
    --grad-clip-norm "${GRAD_CLIP_NORM}" \
    --fs-alpha-clamp "${FS_ALPHA_CLAMP}" \
    --load-dtype "${LOAD_DTYPE}" \
    --eval-limit "${EVAL_LIMIT}" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}")
  if [[ "${UNFREEZE_BACKBONE:-0}" == "1" ]]; then
    TRAIN_CMD+=(--unfreeze-backbone)
  fi
  if [[ "${LOW_CPU_MEM_USAGE}" == "1" ]]; then
    TRAIN_CMD+=(--low-cpu-mem-usage)
  fi
  if [[ "${FS_MODE}" == "disabled" ]]; then
    TRAIN_CMD+=(--disable-future-seed)
  fi
  if [[ "${SKIP_NONFINITE_LOSS}" == "1" ]]; then
    TRAIN_CMD+=(--skip-nonfinite-loss)
  fi
  if [[ "${OPTIMIZE_IN_EVAL_MODE}" == "1" ]]; then
    TRAIN_CMD+=(--optimize-in-eval-mode)
  fi
  "${TRAIN_CMD[@]}" > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
else
  export ENTRYPOINT MODEL_DIR
  uv run python "${ENTRYPOINT}" --model-dir "${MODEL_DIR}" --from-pretrained > "${RESULT_FILE}" 2> "${ERROR_LOG}"
  EXIT_CODE=$?
fi
set -e

FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export FINISHED_AT EXIT_CODE
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_DIR"]) / "metadata.json"
data = json.loads(path.read_text())
data["finished_at"] = os.environ["FINISHED_AT"]
data["exit_status"] = int(os.environ["EXIT_CODE"])
path.write_text(json.dumps(data, indent=2) + "\n")
PY

exit "${EXIT_CODE}"
