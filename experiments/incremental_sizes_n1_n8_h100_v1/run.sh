#!/usr/bin/env bash
set -euo pipefail

cd /output/Profiliing

python3 script/run_profiling_suite.py \
  --matrix-list artifacts/online_safe_v2/complete_24_matrices.txt \
  --matrix-cache matrix/MM \
  --binary ./test/PanguLU \
  --block-sizes 6144 10240 12288 14336 \
  --repeats 1 \
  --warmups 0 \
  --np 1 \
  --timeout 1800 \
  --seed 20260803 \
  --hardware-id H100_SXM_80GB \
  --gpu-model "NVIDIA H100 80GB HBM3" \
  --output artifacts/platforms/H100_SXM_80GB/incremental_sizes_n1_n8_v1/campaign \
  --resume
