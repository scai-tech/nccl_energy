# NCCL Collective Energy Benchmark

This benchmark measures whole-GPU NVML energy around repeated `ncclAllReduce` or `ncclAllGather` calls. It is intended for comparing a baseline NCCL build against NCCL builds with fixed `__nanosleep()` wait-polling backoff.

The measurement is host-side:

1. synchronize all participating streams
2. read NVML total energy for every participating GPU
3. record CUDA start events
4. launch `iters` collective operations
5. record CUDA stop events and synchronize
6. read NVML total energy again
7. report elapsed time, total energy, average power, algorithm bandwidth, and energy per GB

Energy is summed across all participating GPUs. By default, the benchmark exits with an error if any participating GPU does not support `nvmlDeviceGetTotalEnergyConsumption`, because energy is the primary result. Use `--allow-missing-energy=1` only for runtime-only debugging; in that mode energy fields are reported as `NA`.

## Build Manually

Build NCCL first:

```bash
make -j32 src.build \
  BUILDDIR=$PWD/build_baseline \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

Build the benchmark against that NCCL build:

```bash
make -C experiments/nccl_collectives_energy \
  NCCL_HOME=$PWD/build_baseline
```

This benchmark directly includes `nvml.h` and links with `-lnvidia-ml`. If NVML headers or libraries are outside the CUDA default paths, add `NVML_INC=/path/to/nvml/include` and `NVML_LIB=/path/to/nvml/lib` to the benchmark `make` command.

## Run Manually

AllReduce:

```bash
LD_LIBRARY_PATH=$PWD/build_baseline/lib:$LD_LIBRARY_PATH \
NCCL_PROTO=Simple \
./experiments/nccl_collectives_energy/build/collectives_energy_bench \
  --collective=allreduce \
  --gpus=0,1,2,3 \
  --bytes=64M,128M \
  --iters=200 \
  --warmup=20 \
  --repeats=5 \
  --mode-label=baseline \
  --csv=benchmark_results.csv
```

AllGather uses the same command shape:

```bash
LD_LIBRARY_PATH=$PWD/build_baseline/lib:$LD_LIBRARY_PATH \
NCCL_PROTO=Simple \
./experiments/nccl_collectives_energy/build/collectives_energy_bench \
  --collective=allgather \
  --gpus=0,1,2,3 \
  --bytes=64M,128M \
  --iters=200 \
  --warmup=20 \
  --repeats=5 \
  --mode-label=baseline \
  --csv=benchmark_results.csv
```

For AllGather, `--bytes` is the per-rank send size. The receive buffer is allocated as `bytes * nranks`.

## Automated Run

Use `run_benchmark.sh` to build and run baseline plus fixed sleep-backoff NCCL variants:

```bash
MAKEFLAGS="-j32" \
NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" \
GPUS=0,1,2,3 \
SIZES=64M,128M \
ITERS=200 \
WARMUP=20 \
REPEATS=5 \
SLEEP_NS_LIST="64 256" \
SLEEP_EVERY=8 \
COLLECTIVES="allreduce allgather" \
./experiments/nccl_collectives_energy/run_benchmark.sh
```

The script writes outputs under:

```text
experiments/nccl_collectives_energy/outputs/<timestamp>/
```

Expected files include:

```text
allreduce_baseline.log
allgather_baseline.log
allreduce_sleep_64ns_every8.log
allgather_sleep_64ns_every8.log
benchmark_results.csv
```

The script exports `NCCL_PROTO=Simple` by default so the measured collective path matches the Simple-protocol wait backoff hook.

## Backoff Build Flags

The fixed backoff hook is compile-time guarded. Baseline builds are unchanged because the default sleep value is zero.

```bash
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=<ns>
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=<polls>
```

For example, the automated script builds `sleep_64ns_every8` with:

```bash
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=64
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=8
```

The hook is in `src/device/prims_simple.h`, inside the Simple primitive `waitPeer()` polling loop.

## Options Summary

```bash
## --gpus
## CUDA GPU index list. Can be omitted; default is all GPUs visible from cudaGetDeviceCount().
--gpus=0,1,2,3

## --bytes
## Communication message size. Suffixes: K, M, G.
## AllReduce: bytes per GPU send/recv buffer.
## AllGather: bytes per-rank send buffer; recv buffer per GPU is bytes * nranks.
--bytes=64M,128M

## --collective
## allreduce or allgather.
--collective=allreduce

## --iters
## Collective calls inside one measured repeat.
## elapsed_ms and energy_mj in one CSV row cover this many collective calls.
--iters=100

## --warmup
## Warmup collective calls per message size before measurement.
--warmup=5

## --repeats
## Measured repeats per message size. CSV gets one row per repeat.
--repeats=5

## --csv
## CSV output path. New rows are appended.
--csv=benchmark_results.csv

## --mode-label
## Label written to logs and CSV, such as baseline or sleep_64ns_every8.
--mode-label=baseline

## --allow-missing-energy
## 0 = fail if NVML energy cannot be measured. Recommended.
## 1 = continue with energy fields as NA. Runtime/debug only.
--allow-missing-energy=0
```

## CSV Columns

```text
mode,nranks,gpus,bytes,iters,repeat,elapsed_ms,energy_mj,avg_power_w,alg_bw_GBps,energy_per_GB_mJ,collective
```

One CSV row is one measured repeat.

```text
elapsed_ms:
  runtime for iters collective calls

energy_mj:
  total NVML energy delta across all participating GPUs for iters collective calls

avg_power_w:
  energy_mj / elapsed_ms

alg_bw_GBps:
  logicalGB / (elapsed_ms / 1000)

energy_per_GB_mJ:
  energy_mj / logicalGB
```

For per-collective averages:

```text
per_iter_elapsed_ms = elapsed_ms / iters
per_iter_energy_mj = energy_mj / iters
```

## Caveats

- NVML total energy is whole-GPU energy, summed across participating GPUs.
- It is not per-wait-loop or per-kernel-component energy.
- Run on an otherwise idle GPU node.
- Small messages may need larger `ITERS` because the NVML energy counter can be coarse.
- The fixed backoff is deliberately simple. It is a mechanism test before designing adaptive sleep.
