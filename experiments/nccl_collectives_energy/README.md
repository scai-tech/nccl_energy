# NCCL Collective Energy Benchmark

This benchmark measures whole-GPU NVML energy around repeated `ncclAllReduce` or `ncclAllGather` calls. It is intended for comparing a baseline NCCL build against NCCL builds with fixed or adaptive `__nanosleep()` wait-polling backoff.

The measurement is host-side:

1. synchronize all participating streams
2. read NVML total energy for every participating GPU
3. record CUDA start events
4. launch `iters` collective operations
5. record CUDA stop events and synchronize
6. read NVML total energy again
7. optionally read local NCCL wait counters
8. report elapsed time, total energy, average power, algorithm bandwidth, energy per GB, and wait-poll counts

Energy is summed across all participating GPUs. By default, the benchmark exits with an error if any participating GPU does not support `nvmlDeviceGetTotalEnergyConsumption`, because energy is the primary result. Use `--allow-missing-energy=1` only for runtime-only debugging; in that mode energy fields are reported as `NA`.

When NCCL is built with `NCCL_EXPERIMENT_WAIT_STATS=1`, the Simple-protocol `waitPeer()` loop also counts local polling behavior. Each waiting thread accumulates `poll_loads` and `sleep_calls` in local registers, then adds the totals to device-global counters once when that wait loop exits. This avoids one global atomic per poll load. The benchmark resets and reads these counters around every measured repeat.

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

Use `run_benchmark.sh` to build and run baseline plus fixed or adaptive sleep-backoff NCCL variants:

```bash
JOBS=32 \
NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" \
GPUS=0,1,2,3 \
SIZES=64M,128M \
ITERS=200 \
WARMUP=20 \
REPEATS=5 \
SLEEP_NS_LIST="64 256" \
SLEEP_EVERY_LIST="4 8 16" \
BACKOFF_OP_LIST="all" \
BACKOFF_POLICY_LIST="fixed" \
COLLECTIVES="allreduce allgather" \
./experiments/nccl_collectives_energy/run_benchmark.sh
```

`run_benchmark.sh` sweeps all combinations requested by `BACKOFF_POLICY_LIST`. For `fixed`, it sweeps `SLEEP_NS_LIST`, `SLEEP_EVERY_LIST`, and `BACKOFF_OP_LIST`. For `adaptive`, it sweeps `ADAPTIVE_START_SPINS_LIST`, `ADAPTIVE_MEDIUM_SPINS_LIST`, `ADAPTIVE_SMALL_NS_LIST`, `ADAPTIVE_LARGE_NS_LIST`, `SLEEP_EVERY_LIST`, and `BACKOFF_OP_LIST`.

The H100 sbatch defaults run a recvsend-focused adaptive comparison:

```bash
SIZES=64M,128M,192M
ITERS=3000
WARMUP=100
REPEATS=10
SLEEP_NS_LIST="128"
SLEEP_EVERY_LIST="8"
BACKOFF_OP_LIST="recvsend"
BACKOFF_POLICY_LIST="fixed adaptive"
ADAPTIVE_START_SPINS_LIST="64 128 256"
ADAPTIVE_MEDIUM_SPINS_LIST="512 1024"
ADAPTIVE_SMALL_NS_LIST="32"
ADAPTIVE_LARGE_NS_LIST="128 256"
```

To run a narrower role-targeted follow-up, override the defaults at submit time:

```bash
SIZES="64M,128M,256M" \
ITERS=1000 \
WARMUP=50 \
REPEATS=10 \
SLEEP_NS_LIST="64" \
SLEEP_EVERY_LIST="8" \
BACKOFF_OP_LIST="all recv send recvsend" \
BACKOFF_POLICY_LIST="fixed" \
sbatch experiments/nccl_collectives_energy/run_benchmark_h100.sbatch
```

The script writes outputs under:

```text
experiments/nccl_collectives_energy/outputs/<timestamp>/
```

GPU-specific Slurm launchers are provided for repeated queued runs:

```bash
sbatch experiments/nccl_collectives_energy/run_benchmark_h100.sbatch
sbatch experiments/nccl_collectives_energy/run_benchmark_a100.sbatch
sbatch experiments/nccl_collectives_energy/run_benchmark_l40s.sbatch
```

Each launcher sets the matching CUDA architecture and result prefix:

```text
H100: sm_90, result prefix h100
A100: sm_80, result prefix a100
L40S: sm_89, result prefix l40s
```

When launched through these sbatch files, the default output directory includes both GPU type and Slurm job id:

```text
experiments/nccl_collectives_energy/outputs/<gpu_type>_<SLURM_JOB_ID>/
```

Expected files include the GPU type in their names:

```text
h100_allreduce_baseline.log
h100_allgather_baseline.log
h100_allreduce_sleep_64ns_every8.log
h100_allgather_sleep_64ns_every8.log
h100_results.csv
```

The script exports `NCCL_PROTO=Simple` by default so the measured collective path matches the Simple-protocol wait backoff hook.

## Backoff Build Flags

The backoff hook is compile-time guarded. Baseline builds use policy `fixed` with sleep value zero.

```bash
NCCL_EXPERIMENT_WAIT_BACKOFF_POLICY=<0|1>
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=<ns>
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=<polls>
NCCL_EXPERIMENT_WAIT_BACKOFF_OP=<0|1|2|3>
  NCCL_EXPERIMENT_WAIT_BACKOFF_START_SPINS=<spins>
  NCCL_EXPERIMENT_WAIT_BACKOFF_MEDIUM_SPINS=<spins>
  NCCL_EXPERIMENT_WAIT_BACKOFF_SMALL_NS=<ns>
  NCCL_EXPERIMENT_WAIT_BACKOFF_LARGE_NS=<ns>
  NCCL_EXPERIMENT_WAIT_STATS=<0|1>
```

For example, the automated script builds `sleep_64ns_every8` with:

```bash
NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS=64
NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY=8
NCCL_EXPERIMENT_WAIT_BACKOFF_OP=0
NCCL_EXPERIMENT_WAIT_BACKOFF_POLICY=0
```

Policy values:

```text
0: fixed, sleep NCCL_EXPERIMENT_WAIT_BACKOFF_SLEEP_NS every NCCL_EXPERIMENT_WAIT_BACKOFF_EVERY polls
1: adaptive, busy-spin until START_SPINS, then use SMALL_NS until MEDIUM_SPINS, then LARGE_NS
```

`NCCL_EXPERIMENT_WAIT_BACKOFF_OP` selects which Simple `waitPeer()` template shape receives backoff. `BACKOFF_OP_LIST` accepts either the names below or the numeric values:

```text
0: all waitPeer polling paths
1: recv-only waitPeer calls, where Recv=1 and Send=0
2: send-only waitPeer calls, where Send=1 and Recv=0
3: recvsend waitPeer calls, where Recv=1 and Send=1
```

The hook is in `src/device/prims_simple.h`, inside the Simple primitive `waitPeer()` polling loop.

`NCCL_EXPERIMENT_WAIT_STATS=1` enables the local counters used to check whether a backoff policy reduces polling:

```text
wait_entries: waitPeer loops entered by waiting threads
poll_loads: loadStepValue(connStepPtr) calls inside those loops
sleep_calls: __nanosleep() calls made by the backoff hook
```

Counters are grouped by Simple `waitPeer()` template shape:

```text
other: neither recv nor send template shape, normally zero
recv: Recv=1, Send=0
send: Recv=0, Send=1
recvsend: Recv=1, Send=1
```

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

## --collect-wait-stats
## 1 = reset/read NCCL wait counters around each measured repeat.
## Requires NCCL_EXPERIMENT_WAIT_STATS=1 in the NCCL build for nonzero values.
--collect-wait-stats=1
```

## CSV Columns

```text
mode,nranks,gpus,collective,bytes,iters,repeat,elapsed_ms,per_iter_elapsed_ms,energy_mj,per_iter_energy_mj,avg_power_w,alg_bw_GBps,energy_per_GB_mJ,wait_stats_enabled,wait_entries_total,poll_loads_total,sleep_calls_total,poll_loads_per_iter,sleep_calls_per_iter,wait_entries_other,poll_loads_other,sleep_calls_other,wait_entries_recv,poll_loads_recv,sleep_calls_recv,wait_entries_send,poll_loads_send,sleep_calls_send,wait_entries_recvsend,poll_loads_recvsend,sleep_calls_recvsend
```

One CSV row is one measured repeat.

```text
elapsed_ms:
  runtime for iters collective calls

per_iter_elapsed_ms:
  elapsed_ms / iters

energy_mj:
  total NVML energy delta across all participating GPUs for iters collective calls

per_iter_energy_mj:
  energy_mj / iters

avg_power_w:
  energy_mj / elapsed_ms

alg_bw_GBps:
  logicalGB / (elapsed_ms / 1000)

energy_per_GB_mJ:
  energy_mj / logicalGB

wait_stats_enabled:
  1 if the linked NCCL library was built with NCCL_EXPERIMENT_WAIT_STATS=1

wait_entries_total:
  total waitPeer loop entries across all participating GPUs for this measured repeat

poll_loads_total:
  total loadStepValue(connStepPtr) calls across all participating GPUs for this measured repeat

sleep_calls_total:
  total __nanosleep() calls across all participating GPUs for this measured repeat

poll_loads_per_iter:
  poll_loads_total / iters

sleep_calls_per_iter:
  sleep_calls_total / iters

wait_entries_recvsend, poll_loads_recvsend, sleep_calls_recvsend:
  same counters, restricted to Recv=1 and Send=1 waitPeer calls
```

## Caveats

- NVML total energy is whole-GPU energy, summed across participating GPUs.
- It is not per-wait-loop or per-kernel-component energy.
- Run on an otherwise idle GPU node.
- Small messages may need larger `ITERS` because the NVML energy counter can be coarse.
- The fixed backoff is deliberately simple. It is a mechanism test before designing adaptive sleep.
- Wait counters are instrumentation counters, not hardware memory-transaction counters. They count executed `loadStepValue()` calls in the instrumented Simple wait loop.
- `NCCL_EXPERIMENT_WAIT_STATS=1` adds a few global atomics per completed wait loop. Use it to compare polling counts across variants; for final energy numbers, rerun with `WAIT_STATS=0` after choosing promising policies.
