#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. ./repo_env.sh
rm -rf artifacts/wheelhouse-linux-x86_64
mkdir -p artifacts/wheelhouse-linux-x86_64
python3 -m pip download --dest artifacts/wheelhouse-linux-x86_64 --platform manylinux_2_28_x86_64 --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
  'torch==2.7.1' 'triton==3.3.1'
python3 -m pip download --dest artifacts/wheelhouse-linux-x86_64 --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
  'nvidia-cublas-cu12==12.6.4.1' \
  'nvidia-cuda-cupti-cu12==12.6.80' \
  'nvidia-cuda-nvrtc-cu12==12.6.77' \
  'nvidia-cuda-runtime-cu12==12.6.77' \
  'nvidia-cudnn-cu12==9.5.1.17' \
  'nvidia-cufft-cu12==11.3.0.4' \
  'nvidia-cufile-cu12==1.11.1.6' \
  'nvidia-curand-cu12==10.3.7.77' \
  'nvidia-cusolver-cu12==11.7.1.2' \
  'nvidia-cusparse-cu12==12.5.4.2' \
  'nvidia-cusparselt-cu12==0.6.3' \
  'nvidia-nccl-cu12==2.26.2' \
  'nvidia-nvjitlink-cu12==12.6.85' \
  'nvidia-nvtx-cu12==12.6.77'
python3 -m pip download --dest artifacts/wheelhouse-linux-x86_64 --no-deps 'accelerate==1.10.1' 'peft==0.17.1'
python3 -m pip wheel --no-deps -w artifacts/wheelhouse-linux-x86_64 'https://github.com/huggingface/transformers/archive/b9f0fbf532c124ff836466d896a716e26dbe4722.tar.gz'
tar -C artifacts -czf artifacts/wheelhouse-linux-x86_64.tar.gz wheelhouse-linux-x86_64
