#!/bin/bash
# Submit from the repository root:
#   sbatch experiments/nccl_collectives_energy/run_benchmark_a100.sbatch

#SBATCH -J nccl-a100-energy
#SBATCH -A gts-dmahajan7
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:4
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
export RESULT_PREFIX="${RESULT_PREFIX:-a100}"
export OUT_ROOT="${OUT_ROOT:-${SLURM_SUBMIT_DIR}/experiments/nccl_collectives_energy/outputs/${RESULT_PREFIX}_${SLURM_JOB_ID}}"
export NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_80,code=sm_80}"
export GPUS="${GPUS:-0,1,2,3}"
export SIZES="${SIZES:-64M,128M}"
export ITERS="${ITERS:-200}"
export WARMUP="${WARMUP:-20}"
export REPEATS="${REPEATS:-5}"
export SLEEP_NS_LIST="${SLEEP_NS_LIST:-64 256}"
export SLEEP_EVERY="${SLEEP_EVERY:-8}"
export SLEEP_EVERY_LIST="${SLEEP_EVERY_LIST:-${SLEEP_EVERY}}"
export BACKOFF_OP_LIST="${BACKOFF_OP_LIST:-all}"
export BACKOFF_POLICY_LIST="${BACKOFF_POLICY_LIST:-fixed}"
export ADAPTIVE_START_SPINS_LIST="${ADAPTIVE_START_SPINS_LIST:-64 128}"
export ADAPTIVE_MEDIUM_SPINS_LIST="${ADAPTIVE_MEDIUM_SPINS_LIST:-512}"
export ADAPTIVE_SMALL_NS_LIST="${ADAPTIVE_SMALL_NS_LIST:-32}"
export ADAPTIVE_LARGE_NS_LIST="${ADAPTIVE_LARGE_NS_LIST:-128}"
export COLLECTIVES="${COLLECTIVES:-allreduce allgather}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"

echo "jobs=${JOBS}"
echo "result_prefix=${RESULT_PREFIX}"
echo "out_root=${OUT_ROOT}"
echo "nvcc_gencode=${NVCC_GENCODE}"
echo "gpus=${GPUS}"
echo "sizes=${SIZES}"
echo "iters=${ITERS} warmup=${WARMUP} repeats=${REPEATS}"
echo "sleep_ns_list=${SLEEP_NS_LIST} sleep_every_list=${SLEEP_EVERY_LIST}"
echo "backoff_op_list=${BACKOFF_OP_LIST}"
echo "backoff_policy_list=${BACKOFF_POLICY_LIST}"
echo "adaptive_start_spins_list=${ADAPTIVE_START_SPINS_LIST}"
echo "adaptive_medium_spins_list=${ADAPTIVE_MEDIUM_SPINS_LIST}"
echo "adaptive_small_ns_list=${ADAPTIVE_SMALL_NS_LIST}"
echo "adaptive_large_ns_list=${ADAPTIVE_LARGE_NS_LIST}"
echo "collectives=${COLLECTIVES}"
echo "NCCL_PROTO=${NCCL_PROTO}"

srun --ntasks=1 ./experiments/nccl_collectives_energy/run_benchmark.sh
