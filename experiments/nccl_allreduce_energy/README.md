# NCCL Collective Energy Benchmark

This benchmark measures whole-GPU NVML energy around repeated `ncclAllReduce` or `ncclAllGather` calls. It is intended for comparing a baseline NCCL build against experimental wait-polling backoff builds.

The measurement is host-side:

1. synchronize all participating streams
2. read NVML total energy for every participating GPU
3. record CUDA start events
4. launch `iters` collective operations
5. record CUDA stop events and synchronize
6. read NVML total energy again
7. report elapsed time, total energy, average power, algorithm bandwidth, and energy per GB

Energy is summed across all participating GPUs. If any GPU does not support `nvmlDeviceGetTotalEnergyConsumption`, energy fields are reported as `NA`.

## Build

Build NCCL first:

```bash
make src.build BUILDDIR=$PWD/build_phase4_baseline
```

Build the benchmark against that NCCL build:

```bash
make -C experiments/nccl_allreduce_energy \
  NCCL_HOME=$PWD/build_phase4_baseline
```

## Run

```bash
LD_LIBRARY_PATH=$PWD/build_phase4_baseline/lib:$LD_LIBRARY_PATH \
NCCL_PROTO=Simple \
./experiments/nccl_allreduce_energy/build/allreduce_energy_bench \
  --collective=allreduce \
  --gpus=0,1 \
  --bytes=16K,256K,4M,64M \
  --iters=100 \
  --warmup=20 \
  --repeats=5 \
  --mode-label=baseline \
  --csv=allreduce_energy.csv
```

Use longer `--iters` for small messages if energy deltas are noisy.

AllGather uses the same command shape:

```bash
LD_LIBRARY_PATH=$PWD/build_phase4_baseline/lib:$LD_LIBRARY_PATH \
NCCL_PROTO=Simple \
./experiments/nccl_allreduce_energy/build/allreduce_energy_bench \
  --collective=allgather \
  --gpus=0,1 \
  --bytes=16K,256K,4M,64M \
  --iters=100 \
  --warmup=20 \
  --repeats=5 \
  --mode-label=baseline \
  --csv=collective_energy.csv
```

For AllGather, `--bytes` is the per-rank send size. The receive buffer is allocated as `bytes * nranks`.

## Output

Each repeat prints one line:

```text
run mode=baseline collective=allreduce bytes=4194304 repeat=0 elapsed_ms=... energy_mj=... avg_power_w=... energy_per_GB_mJ=... alg_bw_GBps=...
```

Each size also prints summary lines:

```text
summary_begin mode=baseline bytes=4194304
elapsed_ms: mean=... ms stddev=... ms min=... ms max=... ms
energy_mj: mean=... mJ stddev=... mJ min=... mJ max=... mJ
avg_power_w: mean=... W stddev=... W min=... W max=... W
alg_bw_GBps: mean=... GB/s stddev=... GB/s min=... GB/s max=... GB/s
energy_per_GB_mJ: mean=... mJ/GB stddev=... mJ/GB min=... mJ/GB max=... mJ/GB
summary_end mode=baseline bytes=4194304
```
