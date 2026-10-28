#!/usr/bin/env bash
set -euo pipefail
workdir=$(cd -- "$(dirname -- "$0")" && pwd)
team=/home/vjordan/.local/state/experiment-playground/teams/research-lab
test -f "$workdir/quda-cache/tunecache.tsv"
docker run --rm --gpus all --ipc=host \
  -e QUDA_RESOURCE_PATH=/work/quda-cache \
  -e TORCHINDUCTOR_CACHE_DIR=/work/torch-compile-cache \
  -e TRITON_CACHE_DIR=/work/triton-cache \
  -e LD_LIBRARY_PATH=/team/experiments/quda-two-flavour-ensemble-20261025/quda/build/lib_v2ref:/usr/local/cuda/lib64:/usr/local/nvidia/lib64 \
  -v "$workdir:/work" -v "$team:/team:ro" -w /work \
  nvcr.io/nvidia/pytorch@sha256:6e8ccc607fc2a51e3741667b86316a0889418ba8b78ed0ba96e8d282b648a7a6 \
  python benchmark_solvers.py \
  --bridge /team/experiments/fermion-force-research/ckpt_replay_pytorch/libquda_replay_op.so \
  --gauge /team/experiments/quda-max-volume-20261026/emmy_max_vol/ens_56x96_tiled/conf.00002.bin "$@"
