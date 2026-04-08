#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
. "${ROOT_DIR}/repo_env.sh"

REQUIRED_UV_VERSION="${REQUIRED_UV_VERSION:-0.9.27}"
CURRENT_UV_VERSION=""
if [[ -x "${UV_INSTALL_DIR}/uv" ]]; then
  CURRENT_UV_VERSION="$("${UV_INSTALL_DIR}/uv" --version | awk '{print $2}')"
fi

if [[ ! -x "${UV_INSTALL_DIR}/uv" || "${CURRENT_UV_VERSION}" != "${REQUIRED_UV_VERSION}" ]]; then
  rm -f "${UV_INSTALL_DIR}/uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh -s -- --version "${REQUIRED_UV_VERSION}"
fi

uv sync --locked
