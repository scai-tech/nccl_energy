#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GPUS="${GPUS:-0,1}"
SIZES="${SIZES:-16K,256K,4M,64M}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-20}"
REPEATS="${REPEATS:-5}"
SLEEP_NS_LIST="${SLEEP_NS_LIST:-64 256}"
SLEEP_EVERY="${SLEEP_EVERY:-8}"
COLLECTIVES="${COLLECTIVES:-allreduce allgather}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90}"

mkdir -p "${OUT_ROOT}"

build_nccl() {
  local label="$1"
  local sleep_ns="$2"
  local build_dir="$3"
  echo "==> Building NCCL variant '${label}' in ${build_dir}"
  if [[ "${sleep_ns}" == "0" ]]; then
    make -C "${REPO_ROOT}" src.build \
      BUILDDIR="${build_dir}" \
      NVCC_GENCODE="${NVCC_GENCODE}"
  else
    make -C "${REPO_ROOT}" src.build \
      BUILDDIR="${build_dir}" \
      NVCC_GENCODE="${NVCC_GENCODE}" \
      NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS="${sleep_ns}" \
      NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY="${SLEEP_EVERY}"
  fi
}

build_bench() {
  local label="$1"
  local nccl_home="$2"
  local bench_build="$3"
  echo "==> Building collective energy benchmark for '${label}'"
  make -C "${REPO_ROOT}/experiments/nccl_allreduce_energy" \
    NCCL_HOME="${nccl_home}" \
    BUILDDIR="${bench_build}"
}

run_variant() {
  local label="$1"
  local nccl_home="$2"
  local bench_bin="$3"
  local csv_file="${OUT_ROOT}/phase4_results.csv"
  for collective in ${COLLECTIVES}; do
    local log_file="${OUT_ROOT}/${collective}_${label}.log"
    echo "==> Running '${label}' collective='${collective}'"
    LD_LIBRARY_PATH="${nccl_home}/lib:${LD_LIBRARY_PATH:-}" \
      "${bench_bin}" \
        --collective="${collective}" \
        --gpus="${GPUS}" \
        --bytes="${SIZES}" \
        --iters="${ITERS}" \
        --warmup="${WARMUP}" \
        --repeats="${REPEATS}" \
        --mode-label="${label}" \
        --csv="${csv_file}" \
      | tee "${log_file}"
  done
}

echo "Phase 4 fixed-backoff experiment"
echo "repo=${REPO_ROOT}"
echo "out=${OUT_ROOT}"
echo "gpus=${GPUS}"
echo "sizes=${SIZES}"
echo "iters=${ITERS} warmup=${WARMUP} repeats=${REPEATS}"
echo "sleep_ns_list=${SLEEP_NS_LIST} sleep_every=${SLEEP_EVERY}"
echo "collectives=${COLLECTIVES}"
echo "nvcc_gencode=${NVCC_GENCODE}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
echo "NCCL_PROTO=${NCCL_PROTO}"

baseline_build="${OUT_ROOT}/build_baseline"
baseline_bench_build="${OUT_ROOT}/bench_baseline"
baseline_bench="${baseline_bench_build}/allreduce_energy_bench"
build_nccl baseline 0 "${baseline_build}"
build_bench baseline "${baseline_build}" "${baseline_bench_build}"
run_variant baseline "${baseline_build}" "${baseline_bench}"

for sleep_ns in ${SLEEP_NS_LIST}; do
  label="sleep_${sleep_ns}ns_every${SLEEP_EVERY}"
  nccl_build="${OUT_ROOT}/build_${label}"
  bench_build="${OUT_ROOT}/bench_${label}"
  bench_bin="${bench_build}/allreduce_energy_bench"
  build_nccl "${label}" "${sleep_ns}" "${nccl_build}"
  build_bench "${label}" "${nccl_build}" "${bench_build}"
  run_variant "${label}" "${nccl_build}" "${bench_bin}"
done

echo "==> Done"
echo "Logs and CSV are in ${OUT_ROOT}"
echo "CSV: ${OUT_ROOT}/phase4_results.csv"
