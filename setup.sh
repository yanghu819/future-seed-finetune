#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
. "${ROOT_DIR}/repo_env.sh"

if [[ ! -x "${UV_INSTALL_DIR}/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
fi

uv sync --locked
