#!/usr/bin/env bash
set -euo pipefail
workdir=$(cd -- "$(dirname -- "$0")" && pwd)
docker run --rm --gpus all --ipc=host \
 -e TORCHINDUCTOR_CACHE_DIR=/work/torch-compile-cache \
 -e TRITON_CACHE_DIR=/work/triton-cache \
 -v "$workdir:/work" -w /work \
 nvcr.io/nvidia/pytorch@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
 python benchmark_hmc_solvers.py "$@"
