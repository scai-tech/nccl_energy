#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/build/poll_energy_bench"
OUT_DIR="${SCRIPT_DIR}/outputs/$(date +%Y%m%d_%H%M%S)"

make -C "${SCRIPT_DIR}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --gpu-index=0
  --blocks=0
  --threads=256
  --iters=200000
  --launches=50
  --repeats=5
  --warmup=2
  --measure-idle=1
)

"${BIN}" "${COMMON_ARGS[@]}" --mode=spin \
  | tee "${OUT_DIR}/spin.txt"

"${BIN}" "${COMMON_ARGS[@]}" --mode=sleep --sleep-ns=64 --sleep-every=8 \
  | tee "${OUT_DIR}/sleep_64ns_every8.txt"

"${BIN}" "${COMMON_ARGS[@]}" --mode=sleep --sleep-ns=256 --sleep-every=8 \
  | tee "${OUT_DIR}/sleep_256ns_every8.txt"

echo "Wrote outputs to ${OUT_DIR}"
