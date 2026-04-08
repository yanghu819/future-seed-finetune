#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROOT_DIR
export REPO_CACHE_ROOT="${ROOT_DIR}/artifacts/cache"
export UV_CACHE_DIR="${REPO_CACHE_ROOT}/uv"
export XDG_CACHE_HOME="${REPO_CACHE_ROOT}/xdg"
export HF_HOME="${REPO_CACHE_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_CACHE="${HF_HOME}/tokenizers"
export TORCH_HOME="${REPO_CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${REPO_CACHE_ROOT}/pip"
export UV_INSTALL_DIR="${ROOT_DIR}/artifacts/tools/uv"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export PATH="${UV_INSTALL_DIR}:${PATH}"

mkdir -p \
  "${REPO_CACHE_ROOT}" \
  "${UV_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TOKENIZERS_CACHE}" \
  "${TORCH_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${ROOT_DIR}/artifacts/tools"
