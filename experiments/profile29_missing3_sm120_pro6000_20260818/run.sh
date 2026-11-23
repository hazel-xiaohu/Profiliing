#!/usr/bin/env bash
set -euo pipefail

repo=/output/Profiliing
out="$repo/artifacts/platforms/RTX_PRO_6000_BLACKWELL/profile29_missing3_sm120_dense_n1_n8_20260818/campaign"
binary="$repo/artifacts/binaries/PanguLU_sm120_pro6000_20260814"
cuda=/output/toolchains/cuda-12.8

export PATH="$cuda/bin:/openbayes/home/mpich-install/bin:$PATH"
export LD_LIBRARY_PATH="$cuda/lib64:$cuda/targets/x86_64-linux/lib:/openbayes/home/mpich-install/lib:${LD_LIBRARY_PATH:-}"

cd "$repo"
python3 script/run_profiling_suite.py \
  /output/Profiliing/artifacts/platforms/H100_SXM_80GB/unseen_timeout11_v1/normalized_real/atmosmodd.mtx \
  /output/Profiliing/artifacts/platforms/H100_SXM_80GB/unseen_timeout11_v1/normalized_real/kkt_power.mtx \
  /output/Profiliing/artifacts/platforms/H100_SXM_80GB/unseen_timeout11_v1/normalized_real/language.mtx \
  --binary "$binary" \
  --block-sizes 2048 4096 6144 8192 10240 12288 14336 16384 \
  --variants dense \
  --repeats 1 \
  --warmups 0 \
  --np 1 \
  --mpirun /openbayes/home/mpich-install/bin/mpirun \
  --timeout 1800 \
  --seed 20260818 \
  --hardware-id RTX_PRO_6000_BLACKWELL \
  --gpu-model "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  --output "$out" \
  --resume
