# Phase 4: Fixed Wait-Backoff Collective Energy Experiment

This directory runs the Phase 4 A/B experiment:

- baseline NCCL
- NCCL with fixed `__nanosleep()` inserted into Simple-protocol `waitPeer` polling
- AllReduce and AllGather whole-GPU energy measured outside NCCL with NVML

The NCCL change is compile-time guarded. Baseline builds are unchanged because the default sleep value is zero.

## What Was Modified

The experimental hook is in `src/device/prims_simple.h`.

For this first test we only target Simple protocol. The hook is inside the Simple primitive `waitPeer()` loop that waits for peer step/head progress. It calls `__nanosleep()` every configurable number of failed poll iterations when NCCL is built with:

```bash
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=<ns>
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=<polls>
```

Default values do not enable sleep:

```text
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=0
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=8
```

This is intentionally a fixed-sleep experiment, not the final adaptive algorithm. The script exports `NCCL_PROTO=Simple` before running the benchmark so the measured collective path matches the modified primitive.

## One-Command Run

From the repository root:

```bash
./experiments/phase4_wait_backoff/run_phase4.sh
```

Default settings:

```text
GPUS=0,1
SIZES=16K,256K,4M,64M
ITERS=100
WARMUP=20
REPEATS=5
SLEEP_NS_LIST="64 256"
SLEEP_EVERY=8
NCCL_PROTO=Simple
COLLECTIVES="allreduce allgather"
```

The script builds separate NCCL trees under:

```text
experiments/phase4_wait_backoff/outputs/<timestamp>/build_baseline
experiments/phase4_wait_backoff/outputs/<timestamp>/build_sleep_64ns_every8
experiments/phase4_wait_backoff/outputs/<timestamp>/build_sleep_256ns_every8
```

It writes:

```text
allreduce_baseline.log
allgather_baseline.log
allreduce_sleep_64ns_every8.log
allgather_sleep_64ns_every8.log
phase4_results.csv
```

## Recommended First Server Run

For a quick 2-GPU smoke test:

```bash
GPUS=0,1 \
SIZES=256K,4M \
ITERS=20 \
WARMUP=5 \
REPEATS=2 \
SLEEP_NS_LIST="64" \
COLLECTIVES="allreduce allgather" \
./experiments/phase4_wait_backoff/run_phase4.sh
```

If that works, run the fuller first experiment:

```bash
GPUS=0,1 \
SIZES=16K,256K,4M,64M \
ITERS=100 \
WARMUP=20 \
REPEATS=5 \
SLEEP_NS_LIST="64 256" \
SLEEP_EVERY=8 \
COLLECTIVES="allreduce allgather" \
./experiments/phase4_wait_backoff/run_phase4.sh
```

For 4 GPUs:

```bash
GPUS=0,1,2,3 \
SIZES=16K,256K,4M,64M \
ITERS=100 \
WARMUP=20 \
REPEATS=5 \
SLEEP_NS_LIST="64 256" \
SLEEP_EVERY=8 \
COLLECTIVES="allreduce allgather" \
./experiments/phase4_wait_backoff/run_phase4.sh
```

## Ampere/Hopper Build Notes

By default the script passes:

```text
-gencode=arch=compute_80,code=sm_80
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_90,code=compute_90
```

If your CUDA toolkit cannot compile Hopper `sm_90`, override it.

Ampere-only:

```bash
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80" \
./experiments/phase4_wait_backoff/run_phase4.sh
```

Hopper-only:

```bash
NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90" \
./experiments/phase4_wait_backoff/run_phase4.sh
```

## How To Interpret

Compare rows in `phase4_results.csv`.

Important columns:

```text
mode
collective
bytes
elapsed_ms
energy_mj
avg_power_w
alg_bw_GBps
energy_per_GB_mJ
```

For each collective and message size:

```text
energy reduction = (baseline energy_mj - sleep energy_mj) / baseline energy_mj
runtime overhead = (sleep elapsed_ms - baseline elapsed_ms) / baseline elapsed_ms
```

A promising fixed-sleep result looks like:

```text
energy_mj decreases meaningfully
elapsed_ms increases only slightly
energy_per_GB_mJ decreases
```

If energy falls but runtime grows too much, that is still useful: it motivates the adaptive policy in Phase 5/6.

## Caveats

- NVML total energy is whole-GPU energy, summed across participating GPUs.
- It is not per-wait-loop or per-kernel-component energy.
- Run on an otherwise idle GPU node.
- Small messages may need larger `ITERS` because the NVML energy counter can be coarse.
- The fixed backoff is deliberately simple. It is a mechanism test before designing adaptive sleep.
