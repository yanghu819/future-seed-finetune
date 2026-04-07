#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_MODE="${RUN_MODE:-smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-scalar-fs-smoke}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${ROOT_DIR}/runs/${EXPERIMENT_NAME}-${RUN_TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo untracked)"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export ROOT_DIR RUN_MODE EXPERIMENT_NAME RUN_DIR COMMIT_SHA STARTED_AT

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
        "entrypoint": "scripts/smoke_qwen35_scalar_fs.py",
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
uv run python scripts/smoke_qwen35_scalar_fs.py > "${RESULT_DIR}/smoke_metrics.json" 2> "${LOG_DIR}/smoke.stderr.log"
EXIT_CODE=$?
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
