#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${repo_dir}/build/feature_replay}"
mpicxx="${MPICXX:-/openbayes/home/mpich-install/bin/mpic++}"
metis_lib="${METIS_LIB:-/openbayes/home/PanguLU-main-bak/metis/build/Linux-x86_64/libmetis/libmetis.a}"

mkdir -p "${build_dir}"
common_flags=(-O3 -Wall -std=c++11 -fopenmp -DPANGULU_FEATURE_REPLAY -I"${repo_dir}/include")
"${mpicxx}" "${common_flags[@]}" -c "${repo_dir}/src/pangulu.cpp" -o "${build_dir}/pangulu_feature_replay.o"
"${mpicxx}" "${common_flags[@]}" -c "${repo_dir}/test/test.cpp" -o "${build_dir}/test_feature_replay.o"
"${mpicxx}" "${build_dir}/test_feature_replay.o" "${build_dir}/pangulu_feature_replay.o" \
  "${metis_lib}" -fopenmp -lpthread -lm -o "${build_dir}/PanguLUFeatureReplay"
printf '%s\n' "${build_dir}/PanguLUFeatureReplay"
