# GPU Energy Polling Experiment

This directory contains a standalone CUDA/NVML benchmark for validating whether GPU energy can be measured reliably around a kernel execution window, and whether a simple `__nanosleep()` backoff changes energy use for a synthetic polling-like kernel.

This is not an NCCL `RecvSend` modification. It is a first validation benchmark that keeps all changes out of NCCL communication logic.

## What It Measures

The benchmark launches a deterministic polling-like kernel many times inside each measured window.

- `spin` mode repeatedly loads a device flag and does minimal compare/branch/arithmetic work.
- `sleep` mode uses the same polling structure, but calls `__nanosleep()` every configurable number of failed polls.

For each measured repeat it reports:

- elapsed kernel-window time from CUDA events
- total GPU energy delta from `nvmlDeviceGetTotalEnergyConsumption`, in mJ, when supported
- average power, computed as `energy_mj / elapsed_ms`
- optional idle-window energy for a host sleep of roughly the same duration

Because `1 mJ / 1 ms = 1 W`, the average power calculation does not need an additional scale factor.

## Build

From the repository root:

```bash
make -C experiments/gpu_energy_polling
```

To build for a specific GPU architecture and reduce compile time:

```bash
make -C experiments/gpu_energy_polling SM=90
```

By default the Makefile targets Ampere and Hopper with `sm_80`, `sm_90`, and `compute_90` PTX. Override `SM` for a single target machine, for example `SM=80` on Ampere or `SM=90` on Hopper. If your CUDA toolkit is too old to recognize `sm_90`, build Ampere-only with `SM=80`.

If CUDA is not installed under `/usr/local/cuda`, set `CUDA_HOME`:

```bash
make -C experiments/gpu_energy_polling CUDA_HOME=/path/to/cuda
```

## Run

Baseline spin:

```bash
./experiments/gpu_energy_polling/build/poll_energy_bench \
  --mode=spin \
  --gpu-index=0 \
  --blocks=0 \
  --threads=256 \
  --iters=200000 \
  --launches=50 \
  --repeats=5 \
  --warmup=2 \
  --measure-idle=1
```

Sleep backoff:

```bash
./experiments/gpu_energy_polling/build/poll_energy_bench \
  --mode=sleep \
  --gpu-index=0 \
  --blocks=0 \
  --threads=256 \
  --iters=200000 \
  --launches=50 \
  --repeats=5 \
  --warmup=2 \
  --sleep-ns=128 \
  --sleep-every=8 \
  --measure-idle=1
```

`--blocks=0` means `4 * SM count`.

## Scripted Experiment

Run three representative cases and save logs:

```bash
./experiments/gpu_energy_polling/run_experiment.sh
```

The script runs:

- spin baseline
- sleep with `--sleep-ns=64 --sleep-every=8`
- sleep with `--sleep-ns=256 --sleep-every=8`

Output files are written under `experiments/gpu_energy_polling/outputs/<timestamp>/`.

## Useful Flags

```text
--blocks=N          CUDA blocks; 0 means 4 * SM count
--threads=N         CUDA threads per block
--iters=N           polling-loop iterations per kernel launch
--launches=N        kernel launches per measured repeat
--repeats=N         measured windows
--warmup=N          warmup windows before measurement
--sleep-ns=N        __nanosleep argument in sleep mode
--sleep-every=N     sleep every N failed polls in sleep mode
--gpu-index=N       CUDA and NVML GPU index
--measure-idle=0|1  measure an idle host sleep window after each run
```

Increase `--iters` or `--launches` if energy deltas are zero or noisy. The total measured work per repeat is roughly `iters * launches` loop iterations per thread.

## Example Output Shape

```text
config mode=spin gpu_index=0 gpu_name="NVIDIA ..." sm_count=120 blocks=480 threads=256 iters=200000 launches=50 repeats=5 warmup=2 sleep_ns=100 sleep_every=1 measure_idle=1
nvml_status NVML total energy counter is available.
warmup_begin count=2
warmup run=0 elapsed_ms=...
warmup run=1 elapsed_ms=...
warmup_end
measure_begin
run=0 elapsed_ms=123.456 energy_mj=45678.000 avg_power_w=370.123 idle_elapsed_ms=123.560 idle_energy_mj=...
run=1 elapsed_ms=...
measure_end
summary_begin
elapsed_ms: mean=... ms stddev=... ms min=... ms max=... ms
energy_mj: mean=... mJ stddev=... mJ min=... mJ max=... mJ
avg_power_w: mean=... W stddev=... W min=... W max=... W
idle_energy_mj: mean=... mJ stddev=... mJ min=... mJ max=... mJ
idle_avg_power_w: mean=... W stddev=... W min=... W max=... W
summary_end
```

If NVML or the total-energy API is unavailable, the benchmark prints a `nvml_status` message and reports energy fields as `NA` while still reporting CUDA-event runtime.

The benchmark prefers the CUDA device PCI bus ID when selecting the NVML handle, then falls back to NVML index lookup if PCI lookup is unavailable. This helps avoid measuring a different physical GPU when `CUDA_VISIBLE_DEVICES` remaps CUDA indices.

## Interpretation Notes

Compare `spin` and `sleep` using the summary lines:

- Lower `energy_mj` means less total GPU energy for the measured kernel windows.
- Lower `avg_power_w` means the GPU consumed less power during the window.
- Higher `elapsed_ms` is the runtime cost of backoff.
- Idle-window measurements help show the baseline/noise floor over a similar wall-clock duration.

For a useful result, run on an otherwise idle GPU, keep clocks and persistence settings consistent between runs if your environment allows it, and repeat the experiment. If energy deltas are zero or the benchmark warns about instability, make each measured window longer with `--iters` or `--launches`.

## Limitations

- NVML total energy is whole-GPU energy over the measured interval, not per-kernel or per-SM energy.
- Some GPUs or NVML versions do not support `nvmlDeviceGetTotalEnergyConsumption`.
- The counter can have coarse update granularity, so short windows may report zero or noisy deltas.
- The synthetic polling kernel is only a controlled proxy for NCCL-style waiting; it does not prove the same effect inside NCCL communication paths.
