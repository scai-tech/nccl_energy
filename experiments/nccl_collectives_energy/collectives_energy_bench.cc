#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nvml_energy.h"

namespace {

struct Options {
  std::vector<int> gpus;
  std::vector<size_t> sizes;
  int iters = 100;
  int warmup = 20;
  int repeats = 5;
  std::string csvPath;
  std::string modeLabel = "baseline";
  std::string collective = "allreduce";
  bool allowMissingEnergy = false;
  bool collectWaitStats = true;
};

struct RunMetrics {
  double elapsedMs = 0.0;
  bool hasEnergy = false;
  double energyMj = 0.0;
  double avgPowerW = 0.0;
  double algBwGBps = 0.0;
  double energyPerGB = 0.0;
  bool waitStatsEnabled = false;
  bool hasWaitStats = false;
  ncclExperimentWaitStats_t waitStats{};
};

struct Summary {
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
};

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t status = (call);                                              \
    if (status != ncclSuccess) {                                               \
      std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << ncclGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

bool ParseInt(const char* text, int* value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

bool ParseSize(const std::string& text, size_t* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (end == text.c_str()) return false;

  unsigned long long scale = 1;
  if (*end != '\0') {
    char suffix = static_cast<char>(std::tolower(*end));
    if (*(end + 1) != '\0') return false;
    if (suffix == 'k') scale = 1024ull;
    else if (suffix == 'm') scale = 1024ull * 1024ull;
    else if (suffix == 'g') scale = 1024ull * 1024ull * 1024ull;
    else return false;
  }
  *value = static_cast<size_t>(parsed * scale);
  return true;
}

std::vector<std::string> Split(const std::string& text, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, delim)) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
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
      << "  --gpus=0,1,2,3        CUDA device list for one-process NCCL run\n"
      << "  --bytes=16K,4M,64M    Message sizes per rank (default: 16K,256K,4M,64M)\n"
      << "  --collective=NAME     allreduce or allgather (default: allreduce)\n"
      << "  --iters=N             Collective calls per measured repeat (default: 100)\n"
      << "  --warmup=N            Warmup collective calls per size (default: 20)\n"
      << "  --repeats=N           Measured repeats per size (default: 5)\n"
      << "  --csv=PATH            Append CSV results to PATH\n"
      << "  --mode-label=NAME     Label stored in logs and CSV (default: baseline)\n"
      << "  --allow-missing-energy=0|1 Continue with energy=NA if NVML energy is unavailable (default: 0)\n"
      << "  --collect-wait-stats=0|1 Reset/read NCCL wait counters around each repeat (default: 1)\n"
      << "  --help                Print this message\n";
}

bool ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const char* value = nullptr;
    if (std::strcmp(argv[i], "--help") == 0) {
      PrintHelp(argv[0]);
      std::exit(0);
    } else if (GetArgValue(argc, argv, &i, "--gpus", &value)) {
      for (const std::string& part : Split(value, ',')) {
        int gpu = -1;
        if (!ParseInt(part.c_str(), &gpu) || gpu < 0) return false;
        options->gpus.push_back(gpu);
      }
    } else if (GetArgValue(argc, argv, &i, "--bytes", &value)) {
      for (const std::string& part : Split(value, ',')) {
        size_t bytes = 0;
        if (!ParseSize(part, &bytes) || bytes == 0) return false;
        options->sizes.push_back(bytes);
      }
    } else if (GetArgValue(argc, argv, &i, "--collective", &value)) {
      options->collective = value;
      if (options->collective != "allreduce" &&
          options->collective != "allgather") {
        std::cerr << "Invalid --collective value: " << value << std::endl;
        return false;
      }
    } else if (GetArgValue(argc, argv, &i, "--iters", &value)) {
      if (!ParseInt(value, &options->iters)) return false;
    } else if (GetArgValue(argc, argv, &i, "--warmup", &value)) {
      if (!ParseInt(value, &options->warmup)) return false;
    } else if (GetArgValue(argc, argv, &i, "--repeats", &value)) {
      if (!ParseInt(value, &options->repeats)) return false;
    } else if (GetArgValue(argc, argv, &i, "--csv", &value)) {
      options->csvPath = value;
    } else if (GetArgValue(argc, argv, &i, "--mode-label", &value)) {
      options->modeLabel = value;
    } else if (GetArgValue(argc, argv, &i, "--allow-missing-energy", &value)) {
      int parsed = 0;
      if (!ParseInt(value, &parsed) || (parsed != 0 && parsed != 1)) return false;
      options->allowMissingEnergy = parsed != 0;
    } else if (GetArgValue(argc, argv, &i, "--collect-wait-stats", &value)) {
      int parsed = 0;
      if (!ParseInt(value, &parsed) || (parsed != 0 && parsed != 1)) return false;
      options->collectWaitStats = parsed != 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      return false;
    }
  }

  if (options->gpus.empty()) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
      std::cerr << "No CUDA GPUs found and --gpus was not provided." << std::endl;
      return false;
    }
    for (int i = 0; i < count; ++i) options->gpus.push_back(i);
  }
  if (options->sizes.empty()) {
    options->sizes = {16ull * 1024, 256ull * 1024, 4ull * 1024 * 1024,
                      64ull * 1024 * 1024};
  }
  if (options->gpus.size() < 2) {
    std::cerr << "Use at least two GPUs for this collective energy experiment."
              << std::endl;
    return false;
  }
  if (options->iters <= 0 || options->warmup < 0 || options->repeats <= 0) {
    std::cerr << "Invalid --iters/--warmup/--repeats value." << std::endl;
    return false;
  }
  return true;
}

Summary ComputeSummary(const std::vector<double>& values) {
  Summary s;
  if (values.empty()) return s;
  s.min = values[0];
  s.max = values[0];
  double sum = 0.0;
  for (double v : values) {
    sum += v;
    s.min = std::min(s.min, v);
    s.max = std::max(s.max, v);
  }
  s.mean = sum / static_cast<double>(values.size());
  double sq = 0.0;
  for (double v : values) {
    const double d = v - s.mean;
    sq += d * d;
  }
  s.stddev = std::sqrt(sq / static_cast<double>(values.size()));
  return s;
}

std::string JoinInts(const std::vector<int>& values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) oss << ",";
    oss << values[i];
  }
  return oss.str();
}

bool ReadTotalEnergy(const std::vector<NvmlEnergyReader*>& readers,
                     uint64_t* total, std::string* error) {
  *total = 0;
  for (size_t i = 0; i < readers.size(); ++i) {
    uint64_t value = 0;
    std::string localError;
    if (!readers[i]->ReadMilliJoules(&value, &localError)) {
      if (error != nullptr) {
        std::ostringstream oss;
        oss << "GPU reader " << i << ": " << localError;
        *error = oss.str();
      }
      return false;
    }
    *total += value;
  }
  return true;
}

void ResetWaitStatsAll(const std::vector<int>& gpus) {
  for (int gpu : gpus) {
    CUDA_CHECK(cudaSetDevice(gpu));
    NCCL_CHECK(ncclExperimentWaitStatsReset());
  }
}

void AddWaitStats(ncclExperimentWaitStats_t* dst,
                  const ncclExperimentWaitStats_t& src) {
  for (int op = 0; op < NCCL_EXPERIMENT_WAIT_STATS_OP_COUNT; ++op) {
    for (int counter = 0; counter < NCCL_EXPERIMENT_WAIT_STATS_COUNTER_COUNT;
         ++counter) {
      dst->counters[op][counter] += src.counters[op][counter];
    }
  }
}

ncclExperimentWaitStats_t ReadWaitStatsAll(const std::vector<int>& gpus) {
  ncclExperimentWaitStats_t total{};
  for (int gpu : gpus) {
    ncclExperimentWaitStats_t local{};
    CUDA_CHECK(cudaSetDevice(gpu));
    NCCL_CHECK(ncclExperimentWaitStatsRead(&local));
    AddWaitStats(&total, local);
  }
  return total;
}

uint64_t WaitStatCounter(const ncclExperimentWaitStats_t& stats, int op,
                         int counter) {
  return stats.counters[op][counter];
}

uint64_t WaitStatTotal(const ncclExperimentWaitStats_t& stats, int counter) {
  uint64_t total = 0;
  for (int op = 0; op < NCCL_EXPERIMENT_WAIT_STATS_OP_COUNT; ++op) {
    total += stats.counters[op][counter];
  }
  return total;
}

bool SynchronizeAll(const std::vector<int>& gpus,
                    const std::vector<cudaStream_t>& streams) {
  for (size_t i = 0; i < gpus.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
  return true;
}

bool RunCollectiveLoop(const Options& options,
                       const std::vector<ncclComm_t>& comms,
                       const std::vector<cudaStream_t>& streams,
                       const std::vector<float*>& sendbuff,
                       const std::vector<float*>& recvbuff, size_t count,
                       int iters) {
  for (int iter = 0; iter < iters; ++iter) {
    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < options.gpus.size(); ++i) {
      CUDA_CHECK(cudaSetDevice(options.gpus[i]));
      if (options.collective == "allreduce") {
        NCCL_CHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count, ncclFloat,
                                 ncclSum, comms[i], streams[i]));
      } else {
        NCCL_CHECK(ncclAllGather(sendbuff[i], recvbuff[i], count, ncclFloat,
                                 comms[i], streams[i]));
      }
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  return true;
}

bool MeasureRepeat(const Options& options, const std::vector<ncclComm_t>& comms,
                   const std::vector<cudaStream_t>& streams,
                   const std::vector<cudaEvent_t>& starts,
                   const std::vector<cudaEvent_t>& stops,
                   const std::vector<float*>& sendbuff,
                   const std::vector<float*>& recvbuff,
                   const std::vector<NvmlEnergyReader*>& energyReaders,
                   size_t bytes, bool waitStatsEnabled, RunMetrics* metrics) {
  const size_t count = bytes / sizeof(float);
  if (!SynchronizeAll(options.gpus, streams)) return false;
  metrics->waitStatsEnabled = waitStatsEnabled;
  if (options.collectWaitStats) ResetWaitStatsAll(options.gpus);

  uint64_t energyStart = 0;
  uint64_t energyStop = 0;
  std::string energyError;
  const bool canReadEnergy =
      !energyReaders.empty() &&
      ReadTotalEnergy(energyReaders, &energyStart, &energyError);
  if (!canReadEnergy && !energyReaders.empty()) {
    std::cerr << "Failed to read starting energy: " << energyError << std::endl;
    if (!options.allowMissingEnergy) return false;
  }

  for (size_t i = 0; i < options.gpus.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(options.gpus[i]));
    CUDA_CHECK(cudaEventRecord(starts[i], streams[i]));
  }
  if (!RunCollectiveLoop(options, comms, streams, sendbuff, recvbuff, count,
                         options.iters)) {
    return false;
  }
  for (size_t i = 0; i < options.gpus.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(options.gpus[i]));
    CUDA_CHECK(cudaEventRecord(stops[i], streams[i]));
  }
  for (size_t i = 0; i < options.gpus.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(options.gpus[i]));
    CUDA_CHECK(cudaEventSynchronize(stops[i]));
  }
  if (!SynchronizeAll(options.gpus, streams)) return false;
  if (options.collectWaitStats) {
    metrics->hasWaitStats = true;
    metrics->waitStats = ReadWaitStatsAll(options.gpus);
  }

  float elapsedMaxMs = 0.0f;
  for (size_t i = 0; i < options.gpus.size(); ++i) {
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaSetDevice(options.gpus[i]));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, starts[i], stops[i]));
    elapsedMaxMs = std::max(elapsedMaxMs, elapsedMs);
  }

  metrics->elapsedMs = elapsedMaxMs;
  const double elapsedSec = static_cast<double>(elapsedMaxMs) / 1000.0;
  const double bytesPerRank =
      options.collective == "allgather"
          ? static_cast<double>(bytes) * static_cast<double>(options.gpus.size())
          : static_cast<double>(bytes);
  const double logicalGB = bytesPerRank * static_cast<double>(options.iters) / 1.0e9;
  metrics->algBwGBps = elapsedSec > 0.0 ? logicalGB / elapsedSec : 0.0;

  if (canReadEnergy &&
      ReadTotalEnergy(energyReaders, &energyStop, &energyError)) {
    metrics->hasEnergy = true;
    metrics->energyMj = static_cast<double>(energyStop - energyStart);
    metrics->avgPowerW =
        elapsedMaxMs > 0.0f ? metrics->energyMj / metrics->elapsedMs : 0.0;
    metrics->energyPerGB =
        logicalGB > 0.0 ? metrics->energyMj / logicalGB : 0.0;
  } else if (canReadEnergy) {
    std::cerr << "Failed to read ending energy: " << energyError << std::endl;
    if (!options.allowMissingEnergy) return false;
  }
  return true;
}

void AppendCsvHeaderIfNeeded(const std::string& path) {
  if (path.empty()) return;
  std::ifstream in(path.c_str());
  if (in.good() && in.peek() != std::ifstream::traits_type::eof()) return;
  std::ofstream out(path.c_str(), std::ios::app);
  out << "mode,nranks,gpus,collective,bytes,iters,repeat,elapsed_ms,"
         "per_iter_elapsed_ms,energy_mj,per_iter_energy_mj,avg_power_w,"
         "alg_bw_GBps,energy_per_GB_mJ,wait_stats_enabled,"
         "wait_entries_total,poll_loads_total,sleep_calls_total,"
         "poll_loads_per_iter,sleep_calls_per_iter,"
         "wait_entries_other,poll_loads_other,sleep_calls_other,"
         "wait_entries_recv,poll_loads_recv,sleep_calls_recv,"
         "wait_entries_send,poll_loads_send,sleep_calls_send,"
         "wait_entries_recvsend,poll_loads_recvsend,sleep_calls_recvsend\n";
}

void AppendWaitStatsCsvFields(std::ofstream* out, const Options& options,
                              const RunMetrics& metrics) {
  const uint64_t entriesTotal =
      WaitStatTotal(metrics.waitStats,
                    NCCL_EXPERIMENT_WAIT_STATS_COUNTER_ENTRIES);
  const uint64_t pollLoadsTotal =
      WaitStatTotal(metrics.waitStats,
                    NCCL_EXPERIMENT_WAIT_STATS_COUNTER_POLL_LOADS);
  const uint64_t sleepCallsTotal =
      WaitStatTotal(metrics.waitStats,
                    NCCL_EXPERIMENT_WAIT_STATS_COUNTER_SLEEP_CALLS);
  const double pollLoadsPerIter =
      options.iters > 0
          ? static_cast<double>(pollLoadsTotal) / static_cast<double>(options.iters)
          : 0.0;
  const double sleepCallsPerIter =
      options.iters > 0
          ? static_cast<double>(sleepCallsTotal) / static_cast<double>(options.iters)
          : 0.0;

  *out << "," << (metrics.waitStatsEnabled ? 1 : 0) << "," << entriesTotal
       << "," << pollLoadsTotal << "," << sleepCallsTotal << ","
       << pollLoadsPerIter << "," << sleepCallsPerIter;

  const int ops[] = {NCCL_EXPERIMENT_WAIT_STATS_OP_OTHER,
                     NCCL_EXPERIMENT_WAIT_STATS_OP_RECV,
                     NCCL_EXPERIMENT_WAIT_STATS_OP_SEND,
                     NCCL_EXPERIMENT_WAIT_STATS_OP_RECVSEND};
  for (int op : ops) {
    *out << ","
         << WaitStatCounter(metrics.waitStats, op,
                            NCCL_EXPERIMENT_WAIT_STATS_COUNTER_ENTRIES)
         << ","
         << WaitStatCounter(metrics.waitStats, op,
                            NCCL_EXPERIMENT_WAIT_STATS_COUNTER_POLL_LOADS)
         << ","
         << WaitStatCounter(metrics.waitStats, op,
                            NCCL_EXPERIMENT_WAIT_STATS_COUNTER_SLEEP_CALLS);
  }
}

void AppendCsvRow(const Options& options, size_t bytes, int repeat,
                  const RunMetrics& metrics) {
  if (options.csvPath.empty()) return;
  std::ofstream out(options.csvPath.c_str(), std::ios::app);
  const double perIterElapsed =
      options.iters > 0 ? metrics.elapsedMs / static_cast<double>(options.iters)
                        : 0.0;
  out << options.modeLabel << "," << options.gpus.size() << ",\""
      << JoinInts(options.gpus) << "\"," << options.collective << "," << bytes
      << "," << options.iters << "," << repeat << "," << metrics.elapsedMs
      << "," << perIterElapsed << ",";
  if (metrics.hasEnergy) {
    const double perIterEnergy =
        options.iters > 0 ? metrics.energyMj / static_cast<double>(options.iters)
                          : 0.0;
    out << metrics.energyMj << "," << perIterEnergy << ","
        << metrics.avgPowerW << "," << metrics.algBwGBps << ","
        << metrics.energyPerGB;
  } else {
    out << "NA,NA,NA," << metrics.algBwGBps << ",NA";
  }
  AppendWaitStatsCsvFields(&out, options, metrics);
  out << "\n";
}

void PrintSummary(const std::string& label, const std::vector<double>& values,
                  const char* unit) {
  if (values.empty()) {
    std::cout << label << ": NA\n";
    return;
  }
  Summary s = ComputeSummary(values);
  std::cout << label << ": mean=" << s.mean << " " << unit
            << " stddev=" << s.stddev << " " << unit
            << " min=" << s.min << " " << unit
            << " max=" << s.max << " " << unit << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintHelp(argv[0]);
    return 1;
  }

  const int nranks = static_cast<int>(options.gpus.size());
  std::cout << "config mode=" << options.modeLabel
            << " collective=" << options.collective
            << " nranks=" << nranks
            << " gpus=" << JoinInts(options.gpus)
            << " iters=" << options.iters
            << " warmup=" << options.warmup
            << " repeats=" << options.repeats
            << " allow_missing_energy="
            << (options.allowMissingEnergy ? 1 : 0)
            << " collect_wait_stats=" << (options.collectWaitStats ? 1 : 0)
            << "\n";

  int waitStatsEnabled = 0;
  NCCL_CHECK(ncclExperimentWaitStatsGetEnabled(&waitStatsEnabled));
  std::cout << "wait_stats_status compiled_enabled=" << waitStatsEnabled
            << " collected=" << (options.collectWaitStats ? 1 : 0) << "\n";
  if (options.collectWaitStats && waitStatsEnabled == 0) {
    std::cout << "warning wait stats counters are zero because NCCL was not "
                 "built with NCCL_EXPERIMENT_WAIT_STATS=1\n";
  }

  size_t maxBytes = 0;
  for (size_t bytes : options.sizes) {
    if (bytes % sizeof(float) != 0) {
      std::cerr << "Message size must be a multiple of sizeof(float): "
                << bytes << std::endl;
      return 1;
    }
    const size_t recvBytes =
        options.collective == "allgather" ? bytes * options.gpus.size() : bytes;
    maxBytes = std::max(maxBytes, recvBytes);
  }

  std::vector<cudaStream_t> streams(nranks);
  std::vector<cudaEvent_t> starts(nranks);
  std::vector<cudaEvent_t> stops(nranks);
  std::vector<float*> sendbuff(nranks, nullptr);
  std::vector<float*> recvbuff(nranks, nullptr);

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(options.gpus[r]));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, options.gpus[r]));
    char pciBusId[32] = {0};
    cudaError_t pciStatus =
        cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), options.gpus[r]);
    if (pciStatus != cudaSuccess) std::strcpy(pciBusId, "unknown");
    std::cout << "gpu rank=" << r << " cuda_index=" << options.gpus[r]
              << " name=\"" << prop.name << "\" pci_bus_id=" << pciBusId
              << "\n";
    CUDA_CHECK(cudaStreamCreate(&streams[r]));
    CUDA_CHECK(cudaEventCreate(&starts[r]));
    CUDA_CHECK(cudaEventCreate(&stops[r]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sendbuff[r]), maxBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&recvbuff[r]), maxBytes));
    CUDA_CHECK(cudaMemset(sendbuff[r], 0, maxBytes));
    CUDA_CHECK(cudaMemset(recvbuff[r], 0, maxBytes));
  }

  std::vector<int> ncclDevices = options.gpus;
  std::vector<ncclComm_t> comms(nranks);
  NCCL_CHECK(ncclCommInitAll(comms.data(), nranks, ncclDevices.data()));

  std::vector<NvmlEnergyReader> energyReadersStorage(nranks);
  std::vector<NvmlEnergyReader*> energyReaders;
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(options.gpus[r]));
    char pciBusId[32] = {0};
    cudaError_t pciStatus =
        cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), options.gpus[r]);
    energyReadersStorage[r].Init(
        static_cast<unsigned int>(options.gpus[r]),
        pciStatus == cudaSuccess ? pciBusId : nullptr);
    std::cout << "nvml_status rank=" << r << " "
              << energyReadersStorage[r].status_message() << "\n";
    if (energyReadersStorage[r].energy_supported()) {
      energyReaders.push_back(&energyReadersStorage[r]);
    }
  }
  if (static_cast<int>(energyReaders.size()) != nranks) {
    if (!options.allowMissingEnergy) {
      std::cerr << "Fatal: NVML total energy counter is required on every "
                   "participating GPU for this benchmark. Use "
                   "--allow-missing-energy=1 only when you intentionally want "
                   "runtime-only output.\n";
      return 1;
    }
    std::cout << "warning energy fields will be NA because "
                 "--allow-missing-energy=1 was set and at least one GPU does "
                 "not expose NVML total energy.\n";
    energyReaders.clear();
  }

  AppendCsvHeaderIfNeeded(options.csvPath);
  std::cout << std::fixed << std::setprecision(3);

  for (size_t bytes : options.sizes) {
    const size_t count = bytes / sizeof(float);
    std::cout << "size_begin bytes=" << bytes << " count_float=" << count
              << "\n";

    if (options.warmup > 0) {
      if (!RunCollectiveLoop(options, comms, streams, sendbuff, recvbuff, count,
                             options.warmup)) {
        return 1;
      }
      if (!SynchronizeAll(options.gpus, streams)) return 1;
    }
    std::cout << "warmup_done bytes=" << bytes
              << " warmup_iters=" << options.warmup << "\n";

    std::vector<double> elapsed;
    std::vector<double> energy;
    std::vector<double> avgPower;
    std::vector<double> algBw;
    std::vector<double> energyPerGB;
    std::vector<double> pollLoadsPerIter;
    std::vector<double> sleepCallsPerIter;

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      RunMetrics metrics;
      if (!MeasureRepeat(options, comms, streams, starts, stops, sendbuff,
                         recvbuff, energyReaders, bytes,
                         waitStatsEnabled != 0, &metrics)) {
        return 1;
      }
      elapsed.push_back(metrics.elapsedMs);
      algBw.push_back(metrics.algBwGBps);
      if (metrics.hasEnergy) {
        energy.push_back(metrics.energyMj);
        avgPower.push_back(metrics.avgPowerW);
        energyPerGB.push_back(metrics.energyPerGB);
      }
      const uint64_t pollLoadsTotal =
          WaitStatTotal(metrics.waitStats,
                        NCCL_EXPERIMENT_WAIT_STATS_COUNTER_POLL_LOADS);
      const uint64_t sleepCallsTotal =
          WaitStatTotal(metrics.waitStats,
                        NCCL_EXPERIMENT_WAIT_STATS_COUNTER_SLEEP_CALLS);
      pollLoadsPerIter.push_back(
          static_cast<double>(pollLoadsTotal) / static_cast<double>(options.iters));
      sleepCallsPerIter.push_back(
          static_cast<double>(sleepCallsTotal) / static_cast<double>(options.iters));

      std::cout << "run mode=" << options.modeLabel
                << " collective=" << options.collective
                << " bytes=" << bytes
                << " repeat=" << repeat
                << " elapsed_ms=" << metrics.elapsedMs;
      if (metrics.hasEnergy) {
        std::cout << " energy_mj=" << metrics.energyMj
                  << " avg_power_w=" << metrics.avgPowerW
                  << " energy_per_GB_mJ=" << metrics.energyPerGB;
      } else {
        std::cout << " energy_mj=NA avg_power_w=NA energy_per_GB_mJ=NA";
      }
      std::cout << " alg_bw_GBps=" << metrics.algBwGBps
                << " wait_stats_enabled=" << (metrics.waitStatsEnabled ? 1 : 0)
                << " poll_loads_total=" << pollLoadsTotal
                << " sleep_calls_total=" << sleepCallsTotal
                << " poll_loads_recvsend="
                << WaitStatCounter(metrics.waitStats,
                                   NCCL_EXPERIMENT_WAIT_STATS_OP_RECVSEND,
                                   NCCL_EXPERIMENT_WAIT_STATS_COUNTER_POLL_LOADS)
                << " sleep_calls_recvsend="
                << WaitStatCounter(metrics.waitStats,
                                   NCCL_EXPERIMENT_WAIT_STATS_OP_RECVSEND,
                                   NCCL_EXPERIMENT_WAIT_STATS_COUNTER_SLEEP_CALLS)
                << "\n";
      AppendCsvRow(options, bytes, repeat, metrics);
    }

    std::cout << "summary_begin mode=" << options.modeLabel
              << " collective=" << options.collective
              << " bytes=" << bytes << "\n";
    PrintSummary("elapsed_ms", elapsed, "ms");
    PrintSummary("energy_mj", energy, "mJ");
    PrintSummary("avg_power_w", avgPower, "W");
    PrintSummary("alg_bw_GBps", algBw, "GB/s");
    PrintSummary("energy_per_GB_mJ", energyPerGB, "mJ/GB");
    PrintSummary("poll_loads_per_iter", pollLoadsPerIter, "loads/iter");
    PrintSummary("sleep_calls_per_iter", sleepCallsPerIter, "calls/iter");
    std::cout << "summary_end mode=" << options.modeLabel
              << " collective=" << options.collective
              << " bytes=" << bytes << "\n";
    std::cout << "size_end bytes=" << bytes << "\n";
  }

  for (int r = 0; r < nranks; ++r) {
    ncclCommDestroy(comms[r]);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(options.gpus[r]));
    cudaFree(sendbuff[r]);
    cudaFree(recvbuff[r]);
    cudaEventDestroy(starts[r]);
    cudaEventDestroy(stops[r]);
    cudaStreamDestroy(streams[r]);
  }
  return 0;
}
