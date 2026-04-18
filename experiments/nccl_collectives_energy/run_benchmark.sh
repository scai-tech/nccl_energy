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
SLEEP_EVERY_LIST="${SLEEP_EVERY_LIST:-${SLEEP_EVERY}}"
BACKOFF_OP_LIST="${BACKOFF_OP_LIST:-all}"
COLLECTIVES="${COLLECTIVES:-allreduce allgather}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90}"
JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-8}}"
RESULT_PREFIX="${RESULT_PREFIX:-benchmark}"

mkdir -p "${OUT_ROOT}"

backoff_op_code() {
  case "$1" in
    all|0) echo 0 ;;
    recv|1) echo 1 ;;
    send|2) echo 2 ;;
    recvsend|3) echo 3 ;;
    *)
      echo "Unknown BACKOFF_OP '$1'. Use: all, recv, send, recvsend." >&2
      return 1
      ;;
  esac
}

build_nccl() {
  local label="$1"
  local sleep_ns="$2"
  local sleep_every="$3"
  local backoff_op="$4"
  local build_dir="$5"
  echo "==> Building NCCL variant '${label}' in ${build_dir}"
  if [[ "${sleep_ns}" == "0" ]]; then
    make -C "${REPO_ROOT}" -j "${JOBS}" src.build \
      BUILDDIR="${build_dir}" \
      NVCC_GENCODE="${NVCC_GENCODE}"
  else
    make -C "${REPO_ROOT}" -j "${JOBS}" src.build \
      BUILDDIR="${build_dir}" \
      NVCC_GENCODE="${NVCC_GENCODE}" \
      NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS="${sleep_ns}" \
      NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY="${sleep_every}" \
      NCCL_EXPERIMENT_WAIT_BACKOFF_OP="${backoff_op}"
  fi
}

build_bench() {
  local label="$1"
  local nccl_home="$2"
  local bench_build="$3"
  echo "==> Building collective energy benchmark for '${label}'"
  make -C "${REPO_ROOT}/experiments/nccl_collectives_energy" -j "${JOBS}" \
    NCCL_HOME="${nccl_home}" \
    BUILDDIR="${bench_build}"
}

run_variant() {
  local label="$1"
  local nccl_home="$2"
  local bench_bin="$3"
  local csv_file="${OUT_ROOT}/${RESULT_PREFIX}_results.csv"
  for collective in ${COLLECTIVES}; do
    local log_file="${OUT_ROOT}/${RESULT_PREFIX}_${collective}_${label}.log"
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

echo "NCCL collective energy benchmark"
echo "repo=${REPO_ROOT}"
echo "out=${OUT_ROOT}"
echo "gpus=${GPUS}"
echo "sizes=${SIZES}"
echo "iters=${ITERS} warmup=${WARMUP} repeats=${REPEATS}"
echo "sleep_ns_list=${SLEEP_NS_LIST} sleep_every_list=${SLEEP_EVERY_LIST}"
echo "backoff_op_list=${BACKOFF_OP_LIST}"
echo "collectives=${COLLECTIVES}"
echo "nvcc_gencode=${NVCC_GENCODE}"
echo "jobs=${JOBS}"
echo "result_prefix=${RESULT_PREFIX}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
echo "NCCL_PROTO=${NCCL_PROTO}"

baseline_build="${OUT_ROOT}/build_baseline"
baseline_bench_build="${OUT_ROOT}/bench_baseline"
baseline_bench="${baseline_bench_build}/collectives_energy_bench"
build_nccl baseline 0 0 0 "${baseline_build}"
build_bench baseline "${baseline_build}" "${baseline_bench_build}"
run_variant baseline "${baseline_build}" "${baseline_bench}"

for sleep_ns in ${SLEEP_NS_LIST}; do
  for sleep_every in ${SLEEP_EVERY_LIST}; do
    for backoff_op_name in ${BACKOFF_OP_LIST}; do
      backoff_op="$(backoff_op_code "${backoff_op_name}")"
      label="sleep_${sleep_ns}ns_every${sleep_every}"
      if [[ "${backoff_op_name}" != "all" ]]; then
        label="${label}_${backoff_op_name}"
      fi
      nccl_build="${OUT_ROOT}/build_${label}"
      bench_build="${OUT_ROOT}/bench_${label}"
      bench_bin="${bench_build}/collectives_energy_bench"
      build_nccl "${label}" "${sleep_ns}" "${sleep_every}" "${backoff_op}" "${nccl_build}"
      build_bench "${label}" "${nccl_build}" "${bench_build}"
      run_variant "${label}" "${nccl_build}" "${bench_bin}"
    done
  done
done

echo "==> Done"
echo "Logs and CSV are in ${OUT_ROOT}"
echo "CSV: ${OUT_ROOT}/${RESULT_PREFIX}_results.csv"
