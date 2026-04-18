#!/bin/bash
# Submit from the repository root:
#   sbatch experiments/nccl_collectives_energy/run_benchmark_h100.sbatch
#
# 8 hours is conservative for building NCCL variants plus running the benchmark.
# For a short queue-friendly smoke test, override with:
#   sbatch -t 2:00:00 experiments/nccl_collectives_energy/run_benchmark_h100.sbatch

#SBATCH -J nccl-h100-energy
#SBATCH -A gts-dmahajan7
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-gpu=32G
#SBATCH -t 8:00:00
#SBATCH -o slurm-%x-%j.out

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

set +u
source "${HOME}/.bashrc"
conda activate ccoverlap
set -u

nvidia-smi

export JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-8}}"
export RESULT_PREFIX="${RESULT_PREFIX:-h100}"
export OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR}/experiments/nccl_collectives_energy/outputs/${RESULT_PREFIX}_${SLURM_JOB_ID}}"
export NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_90,code=sm_90}"
export GPUS="${GPUS:-0,1,2,3}"
export SIZES="${SIZES:-64M,128M}"
export ITERS="${ITERS:-200}"
export WARMUP="${WARMUP:-20}"
export REPEATS="${REPEATS:-5}"
export SLEEP_NS_LIST="${SLEEP_NS_LIST:-64 256}"
export SLEEP_EVERY="${SLEEP_EVERY:-8}"
export COLLECTIVES="${COLLECTIVES:-allreduce allgather}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"

echo "jobs=${JOBS}"
echo "result_prefix=${RESULT_PREFIX}"
echo "out_root=${OUT_ROOT}"
echo "nvcc_gencode=${NVCC_GENCODE}"
echo "gpus=${GPUS}"
echo "sizes=${SIZES}"
echo "iters=${ITERS} warmup=${WARMUP} repeats=${REPEATS}"
echo "sleep_ns_list=${SLEEP_NS_LIST} sleep_every=${SLEEP_EVERY}"
echo "collectives=${COLLECTIVES}"
echo "NCCL_PROTO=${NCCL_PROTO}"

srun --ntasks=1 ./experiments/nccl_collectives_energy/run_benchmark.sh
