#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nvml_energy.h"

namespace {

enum class Mode {
  kSpin,
  kSleep,
};

struct Options {
  int blocks = 0;
  int threads = 256;
  long long iters = 100000;
  int launches = 50;
  int repeats = 5;
  int warmup = 2;
  unsigned int sleepNs = 100;
  unsigned int sleepEvery = 1;
  int gpuIndex = 0;
  bool measureIdle = false;
  Mode mode = Mode::kSpin;
};

struct RunMetrics {
  double elapsedMs = 0.0;
  bool hasEnergy = false;
  double energyMj = 0.0;
  double avgPowerW = 0.0;
  bool hasIdleEnergy = false;
  double idleElapsedMs = 0.0;
  double idleEnergyMj = 0.0;
  double idleAvgPowerW = 0.0;
};

struct Summary {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
};

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << cudaGetErrorString(status) << std::endl;                  \
      return false;                                                          \
    }                                                                        \
  } while (0)

const char* ModeName(Mode mode) {
  return mode == Mode::kSpin ? "spin" : "sleep";
}

bool ParseLongLong(const char* text, long long* value) {
  char* end = nullptr;
  long long parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool ParseInt(const char* text, int* value) {
  long long parsed = 0;
  if (!ParseLongLong(text, &parsed)) return false;
  if (parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseUInt(const char* text, unsigned int* value) {
  long long parsed = 0;
  if (!ParseLongLong(text, &parsed) || parsed < 0 ||
      static_cast<unsigned long long>(parsed) >
          std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  *value = static_cast<unsigned int>(parsed);
  return true;
}

bool ParseBool01(const char* text, bool* value) {
  if (std::strcmp(text, "0") == 0) {
    *value = false;
    return true;
  }
  if (std::strcmp(text, "1") == 0) {
    *value = true;
    return true;
  }
  return false;
}

bool GetArgValue(int argc, char** argv, int* index, const char* key,
                 const char** value) {
  const size_t keyLen = std::strlen(key);
  const char* arg = argv[*index];
  if (std::strncmp(arg, key, keyLen) != 0) return false;
  if (arg[keyLen] == '=') {
    *value = arg + keyLen + 1;
    return true;
  }
  if (arg[keyLen] == '\0' && *index + 1 < argc) {
    ++(*index);
    *value = argv[*index];
    return true;
  }
  return false;
}

void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --mode=[spin|sleep]        Kernel variant to run (default: spin)\n"
      << "  --blocks=N                 CUDA blocks (default: 4 * SM count)\n"
      << "  --threads=N                CUDA threads per block (default: 256)\n"
      << "  --iters=N                  Poll-loop iterations per kernel launch (default: 100000)\n"
      << "  --launches=N               Kernel launches per measured repeat (default: 50)\n"
      << "  --repeats=N                Measured repeat windows (default: 5)\n"
      << "  --warmup=N                 Warmup windows before measurement (default: 2)\n"
      << "  --sleep-ns=N               __nanosleep argument for sleep mode (default: 100)\n"
      << "  --sleep-every=N            Sleep every N failed polls in sleep mode (default: 1)\n"
      << "  --gpu-index=N              CUDA and NVML GPU index (default: 0)\n"
      << "  --measure-idle=[0|1]       Measure host idle window after each run (default: 0)\n"
      << "  --help                     Print this message\n";
}

bool ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const char* value = nullptr;
    if (std::strcmp(argv[i], "--help") == 0) {
      PrintHelp(argv[0]);
      std::exit(0);
    } else if (GetArgValue(argc, argv, &i, "--mode", &value)) {
      if (std::strcmp(value, "spin") == 0) {
        options->mode = Mode::kSpin;
      } else if (std::strcmp(value, "sleep") == 0) {
        options->mode = Mode::kSleep;
      } else {
        std::cerr << "Invalid --mode value: " << value << std::endl;
        return false;
      }
    } else if (GetArgValue(argc, argv, &i, "--blocks", &value)) {
      if (!ParseInt(value, &options->blocks)) return false;
    } else if (GetArgValue(argc, argv, &i, "--threads", &value)) {
      if (!ParseInt(value, &options->threads)) return false;
    } else if (GetArgValue(argc, argv, &i, "--iters", &value)) {
      if (!ParseLongLong(value, &options->iters)) return false;
    } else if (GetArgValue(argc, argv, &i, "--launches", &value)) {
      if (!ParseInt(value, &options->launches)) return false;
    } else if (GetArgValue(argc, argv, &i, "--repeats", &value)) {
      if (!ParseInt(value, &options->repeats)) return false;
    } else if (GetArgValue(argc, argv, &i, "--warmup", &value)) {
      if (!ParseInt(value, &options->warmup)) return false;
    } else if (GetArgValue(argc, argv, &i, "--sleep-ns", &value)) {
      if (!ParseUInt(value, &options->sleepNs)) return false;
    } else if (GetArgValue(argc, argv, &i, "--sleep-every", &value)) {
      if (!ParseUInt(value, &options->sleepEvery)) return false;
    } else if (GetArgValue(argc, argv, &i, "--gpu-index", &value)) {
      if (!ParseInt(value, &options->gpuIndex)) return false;
    } else if (GetArgValue(argc, argv, &i, "--measure-idle", &value)) {
      if (!ParseBool01(value, &options->measureIdle)) return false;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      return false;
    }
  }

  if (options->threads <= 0 || options->blocks < 0 || options->iters <= 0 ||
      options->launches <= 0 || options->repeats <= 0 || options->warmup < 0 ||
      options->gpuIndex < 0 || options->sleepEvery == 0) {
    std::cerr << "Invalid non-positive numeric option. Use --help for defaults."
              << std::endl;
    return false;
  }
  return true;
}

template <bool UseSleep>
__global__ void PollKernel(const int* flag, uint64_t* sink, long long iters,
                           unsigned int sleepNs, unsigned int sleepEvery) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  volatile const int* observedFlag = flag;
  uint64_t accum = static_cast<uint64_t>(tid) + 1;
  unsigned int sleepCountdown = 0;

#pragma unroll 1
  for (long long i = 0; i < iters; ++i) {
    const int value = *observedFlag;
    if (value != 0) {
      accum ^= static_cast<uint64_t>(value) + static_cast<uint64_t>(i);
    } else {
      accum += (static_cast<uint64_t>(i) ^ static_cast<uint64_t>(tid)) & 1ull;
      if (UseSleep && sleepNs > 0) {
        ++sleepCountdown;
        if (sleepCountdown >= sleepEvery) {
          __nanosleep(sleepNs);
          sleepCountdown = 0;
        }
      }
    }
  }

  sink[tid] = accum;
}

bool LaunchOnce(const Options& options, const int* flag, uint64_t* sink) {
  if (options.mode == Mode::kSpin) {
    PollKernel<false><<<options.blocks, options.threads>>>(
        flag, sink, options.iters, 0, options.sleepEvery);
  } else {
    PollKernel<true><<<options.blocks, options.threads>>>(
        flag, sink, options.iters, options.sleepNs, options.sleepEvery);
  }
  CUDA_CHECK(cudaGetLastError());
  return true;
}

bool RunWindow(const Options& options, const int* flag, uint64_t* sink,
               NvmlEnergyReader* energyReader, bool recordEnergy,
               RunMetrics* metrics) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t energyStart = 0;
  uint64_t energyStop = 0;
  std::string energyError;
  const bool canReadEnergy =
      recordEnergy && energyReader != nullptr && energyReader->energy_supported();
  bool energyStartOk = false;
  if (canReadEnergy) {
    energyStartOk = energyReader->ReadMilliJoules(&energyStart, &energyError);
    if (!energyStartOk) {
      std::cerr << "Warning: failed to read starting energy: " << energyError
                << std::endl;
    }
  }

  CUDA_CHECK(cudaEventRecord(start));
  for (int launch = 0; launch < options.launches; ++launch) {
    if (!LaunchOnce(options, flag, sink)) return false;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  float elapsedMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

  if (canReadEnergy && energyStartOk &&
      energyReader->ReadMilliJoules(&energyStop, &energyError)) {
    metrics->hasEnergy = true;
    metrics->energyMj = static_cast<double>(energyStop - energyStart);
    metrics->avgPowerW =
        elapsedMs > 0.0f ? metrics->energyMj / static_cast<double>(elapsedMs) : 0.0;
  } else if (canReadEnergy && energyStartOk) {
    std::cerr << "Warning: failed to read ending energy: " << energyError
              << std::endl;
  }

  metrics->elapsedMs = static_cast<double>(elapsedMs);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return true;
}

bool MeasureIdleWindow(double requestedMs, NvmlEnergyReader* energyReader,
                       RunMetrics* metrics) {
  if (energyReader == nullptr || !energyReader->energy_supported()) return true;
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t energyStart = 0;
  uint64_t energyStop = 0;
  std::string error;
  if (!energyReader->ReadMilliJoules(&energyStart, &error)) {
    std::cerr << "Warning: failed to read idle starting energy: " << error
              << std::endl;
    return true;
  }

  const auto start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(
      std::chrono::duration<double, std::milli>(requestedMs));
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto stop = std::chrono::steady_clock::now();

  if (!energyReader->ReadMilliJoules(&energyStop, &error)) {
    std::cerr << "Warning: failed to read idle ending energy: " << error
              << std::endl;
    return true;
  }

  metrics->hasIdleEnergy = true;
  metrics->idleElapsedMs =
      std::chrono::duration<double, std::milli>(stop - start).count();
  metrics->idleEnergyMj = static_cast<double>(energyStop - energyStart);
  metrics->idleAvgPowerW =
      metrics->idleElapsedMs > 0.0 ? metrics->idleEnergyMj / metrics->idleElapsedMs
                                   : 0.0;
  return true;
}

Summary ComputeSummary(const std::vector<double>& values) {
  Summary summary;
  if (values.empty()) return summary;

  double sum = 0.0;
  summary.min = values[0];
  summary.max = values[0];
  for (double value : values) {
    sum += value;
    summary.min = std::min(summary.min, value);
    summary.max = std::max(summary.max, value);
  }
  summary.mean = sum / static_cast<double>(values.size());

  double sq = 0.0;
  for (double value : values) {
    const double diff = value - summary.mean;
    sq += diff * diff;
  }
  summary.stddev = std::sqrt(sq / static_cast<double>(values.size()));
  return summary;
}

void PrintSummaryLine(const char* label, const char* unit,
                      const std::vector<double>& values) {
  if (values.empty()) {
    std::cout << label << ": NA\n";
    return;
  }
  const Summary summary = ComputeSummary(values);
  std::cout << label << ": mean=" << summary.mean << " " << unit
            << " stddev=" << summary.stddev << " " << unit
            << " min=" << summary.min << " " << unit
            << " max=" << summary.max << " " << unit << "\n";
}

bool HasZero(const std::vector<double>& values) {
  for (double value : values) {
    if (value == 0.0) return true;
  }
  return false;
}

bool IsUnstable(const std::vector<double>& values) {
  if (values.size() < 2) return false;
  const Summary summary = ComputeSummary(values);
  if (summary.mean <= 0.0) return false;
  return summary.stddev / summary.mean > 0.15;
}

void PrintConfig(const Options& options, const cudaDeviceProp& prop,
                 const char* pciBusId) {
  std::cout << "config"
            << " mode=" << ModeName(options.mode)
            << " gpu_index=" << options.gpuIndex
            << " gpu_name=\"" << prop.name << "\""
            << " pci_bus_id=" << (pciBusId != nullptr ? pciBusId : "unknown")
            << " sm_count=" << prop.multiProcessorCount
            << " blocks=" << options.blocks
            << " threads=" << options.threads
            << " iters=" << options.iters
            << " launches=" << options.launches
            << " repeats=" << options.repeats
            << " warmup=" << options.warmup
            << " sleep_ns=" << options.sleepNs
            << " sleep_every=" << options.sleepEvery
            << " measure_idle=" << (options.measureIdle ? 1 : 0) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintHelp(argv[0]);
    return 1;
  }

  if (cudaSetDevice(options.gpuIndex) != cudaSuccess) {
    std::cerr << "Failed to set CUDA device " << options.gpuIndex << ": "
              << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, options.gpuIndex) != cudaSuccess) {
    std::cerr << "Failed to query CUDA device properties: "
              << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }
  if (options.blocks == 0) {
    options.blocks = std::max(1, prop.multiProcessorCount * 4);
  }

  char pciBusId[32] = {0};
  cudaError_t pciStatus =
      cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), options.gpuIndex);
  if (pciStatus != cudaSuccess) {
    std::cerr << "Warning: failed to query CUDA PCI bus ID: "
              << cudaGetErrorString(pciStatus)
              << "; NVML will fall back to index lookup." << std::endl;
    pciBusId[0] = '\0';
  }

  PrintConfig(options, prop, pciBusId[0] != '\0' ? pciBusId : nullptr);

  NvmlEnergyReader energyReader;
  energyReader.Init(static_cast<unsigned int>(options.gpuIndex),
                    pciBusId[0] != '\0' ? pciBusId : nullptr);
  std::cout << "nvml_status " << energyReader.status_message() << "\n";
  if (!energyReader.energy_supported()) {
    std::cout << "warning energy_mj and avg_power_w will be NA because the NVML "
                 "total energy counter is unavailable.\n";
  }

  int* deviceFlag = nullptr;
  uint64_t* deviceSink = nullptr;
  const size_t sinkCount =
      static_cast<size_t>(options.blocks) * static_cast<size_t>(options.threads);
  if (cudaMalloc(&deviceFlag, sizeof(int)) != cudaSuccess ||
      cudaMalloc(&deviceSink, sinkCount * sizeof(uint64_t)) != cudaSuccess) {
    std::cerr << "Failed to allocate device buffers: "
              << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaFree(deviceFlag);
    cudaFree(deviceSink);
    return 1;
  }
  int zero = 0;
  if (cudaMemcpy(deviceFlag, &zero, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cerr << "Failed to initialize device flag: "
              << cudaGetErrorString(cudaGetLastError()) << std::endl;
    cudaFree(deviceFlag);
    cudaFree(deviceSink);
    return 1;
  }

  std::cout << "warmup_begin count=" << options.warmup << "\n";
  for (int i = 0; i < options.warmup; ++i) {
    RunMetrics metrics;
    if (!RunWindow(options, deviceFlag, deviceSink, &energyReader, false,
                   &metrics)) {
      cudaFree(deviceFlag);
      cudaFree(deviceSink);
      return 1;
    }
    std::cout << "warmup run=" << i << " elapsed_ms=" << metrics.elapsedMs
              << "\n";
  }
  std::cout << "warmup_end\n";

  std::vector<RunMetrics> runs;
  runs.reserve(static_cast<size_t>(options.repeats));
  std::cout << "measure_begin\n";
  std::cout << std::fixed << std::setprecision(3);
  for (int i = 0; i < options.repeats; ++i) {
    RunMetrics metrics;
    if (!RunWindow(options, deviceFlag, deviceSink, &energyReader, true,
                   &metrics)) {
      cudaFree(deviceFlag);
      cudaFree(deviceSink);
      return 1;
    }
    if (options.measureIdle &&
        !MeasureIdleWindow(metrics.elapsedMs, &energyReader, &metrics)) {
      cudaFree(deviceFlag);
      cudaFree(deviceSink);
      return 1;
    }

    std::cout << "run=" << i << " elapsed_ms=" << metrics.elapsedMs;
    if (metrics.hasEnergy) {
      std::cout << " energy_mj=" << metrics.energyMj
                << " avg_power_w=" << metrics.avgPowerW;
    } else {
      std::cout << " energy_mj=NA avg_power_w=NA";
    }
    if (options.measureIdle) {
      if (metrics.hasIdleEnergy) {
        std::cout << " idle_elapsed_ms=" << metrics.idleElapsedMs
                  << " idle_energy_mj=" << metrics.idleEnergyMj
                  << " idle_avg_power_w=" << metrics.idleAvgPowerW;
      } else {
        std::cout << " idle_elapsed_ms=NA idle_energy_mj=NA idle_avg_power_w=NA";
      }
    }
    std::cout << "\n";
    runs.push_back(metrics);
  }
  std::cout << "measure_end\n";

  std::vector<double> elapsedMs;
  std::vector<double> energyMj;
  std::vector<double> avgPowerW;
  std::vector<double> idleEnergyMj;
  std::vector<double> idleAvgPowerW;
  for (const RunMetrics& run : runs) {
    elapsedMs.push_back(run.elapsedMs);
    if (run.hasEnergy) {
      energyMj.push_back(run.energyMj);
      avgPowerW.push_back(run.avgPowerW);
    }
    if (run.hasIdleEnergy) {
      idleEnergyMj.push_back(run.idleEnergyMj);
      idleAvgPowerW.push_back(run.idleAvgPowerW);
    }
  }

  std::cout << "summary_begin\n";
  PrintSummaryLine("elapsed_ms", "ms", elapsedMs);
  PrintSummaryLine("energy_mj", "mJ", energyMj);
  PrintSummaryLine("avg_power_w", "W", avgPowerW);
  if (options.measureIdle) {
    PrintSummaryLine("idle_energy_mj", "mJ", idleEnergyMj);
    PrintSummaryLine("idle_avg_power_w", "W", idleAvgPowerW);
  }
  if (!energyMj.empty()) {
    if (HasZero(energyMj)) {
      std::cout << "warning at least one measured energy delta was zero; increase "
                   "--iters or --launches to make the measurement window longer.\n";
    }
    if (IsUnstable(energyMj)) {
      std::cout << "warning energy_mj varied by more than 15% coefficient of "
                   "variation; run on an idle GPU or increase --launches.\n";
    }
  }
  std::cout << "summary_end\n";

  cudaFree(deviceFlag);
  cudaFree(deviceSink);
  return 0;
}
