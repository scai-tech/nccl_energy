#include "nvml_energy.h"

#include <dlfcn.h>

#include <sstream>

namespace {

constexpr int kNvmlSuccess = 0;
constexpr int kNvmlErrorNotSupported = 3;
constexpr int kNvmlErrorFunctionNotFound = 13;

void* LoadSymbol(void* library, const char* name) {
  return dlsym(library, name);
}

}  // namespace

NvmlEnergyReader::NvmlEnergyReader()
    : library_(nullptr),
      device_(nullptr),
      initialized_(false),
      available_(false),
      energySupported_(false),
      nvmlShutdown_(nullptr),
      nvmlDeviceGetHandleByIndex_(nullptr),
      nvmlDeviceGetHandleByPciBusId_(nullptr),
      nvmlDeviceGetTotalEnergyConsumption_(nullptr),
      nvmlErrorString_(nullptr) {}

NvmlEnergyReader::~NvmlEnergyReader() {
  if (initialized_ && nvmlShutdown_ != nullptr) {
    nvmlShutdown_();
  }
  if (library_ != nullptr) {
    dlclose(library_);
  }
}

bool NvmlEnergyReader::Init(unsigned int gpuIndex, const char* pciBusId) {
  library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    library_ = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
  }
  if (library_ == nullptr) {
    statusMessage_ = "NVML library not found; energy metrics will be reported as NA.";
    return false;
  }

  NvmlInitFn nvmlInit =
      reinterpret_cast<NvmlInitFn>(LoadSymbol(library_, "nvmlInit_v2"));
  if (nvmlInit == nullptr) {
    nvmlInit = reinterpret_cast<NvmlInitFn>(LoadSymbol(library_, "nvmlInit"));
  }
  nvmlShutdown_ =
      reinterpret_cast<NvmlShutdownFn>(LoadSymbol(library_, "nvmlShutdown"));
  nvmlDeviceGetHandleByIndex_ =
      reinterpret_cast<NvmlDeviceGetHandleByIndexFn>(
          LoadSymbol(library_, "nvmlDeviceGetHandleByIndex_v2"));
  if (nvmlDeviceGetHandleByIndex_ == nullptr) {
    nvmlDeviceGetHandleByIndex_ =
        reinterpret_cast<NvmlDeviceGetHandleByIndexFn>(
            LoadSymbol(library_, "nvmlDeviceGetHandleByIndex"));
  }
  nvmlDeviceGetHandleByPciBusId_ =
      reinterpret_cast<NvmlDeviceGetHandleByPciBusIdFn>(
          LoadSymbol(library_, "nvmlDeviceGetHandleByPciBusId_v2"));
  if (nvmlDeviceGetHandleByPciBusId_ == nullptr) {
    nvmlDeviceGetHandleByPciBusId_ =
        reinterpret_cast<NvmlDeviceGetHandleByPciBusIdFn>(
            LoadSymbol(library_, "nvmlDeviceGetHandleByPciBusId"));
  }
  nvmlDeviceGetTotalEnergyConsumption_ =
      reinterpret_cast<NvmlDeviceGetTotalEnergyConsumptionFn>(
          LoadSymbol(library_, "nvmlDeviceGetTotalEnergyConsumption"));
  nvmlErrorString_ =
      reinterpret_cast<NvmlErrorStringFn>(LoadSymbol(library_, "nvmlErrorString"));

  if (nvmlInit == nullptr || nvmlDeviceGetHandleByIndex_ == nullptr) {
    statusMessage_ =
        "Required NVML entry points are missing; energy metrics will be reported as NA.";
    return false;
  }

  NvmlReturn result = nvmlInit();
  if (result != kNvmlSuccess) {
    std::ostringstream oss;
    oss << "nvmlInit failed: " << ErrorString(result)
        << "; energy metrics will be reported as NA.";
    statusMessage_ = oss.str();
    return false;
  }
  initialized_ = true;

  if (pciBusId != nullptr && pciBusId[0] != '\0' &&
      nvmlDeviceGetHandleByPciBusId_ != nullptr) {
    result = nvmlDeviceGetHandleByPciBusId_(pciBusId, &device_);
  } else {
    result = nvmlDeviceGetHandleByIndex_(gpuIndex, &device_);
  }
  if (result != kNvmlSuccess) {
    std::ostringstream oss;
    if (pciBusId != nullptr && pciBusId[0] != '\0' &&
        nvmlDeviceGetHandleByPciBusId_ != nullptr) {
      oss << "nvmlDeviceGetHandleByPciBusId(" << pciBusId << ") failed: ";
    } else {
      oss << "nvmlDeviceGetHandleByIndex(" << gpuIndex << ") failed: ";
    }
    oss << ErrorString(result) << "; energy metrics will be reported as NA.";
    statusMessage_ = oss.str();
    return false;
  }

  available_ = true;
  if (nvmlDeviceGetTotalEnergyConsumption_ == nullptr) {
    statusMessage_ =
        "nvmlDeviceGetTotalEnergyConsumption is not present in this NVML library; "
        "energy metrics will be reported as NA.";
    return true;
  }

  uint64_t ignored = 0;
  std::string error;
  if (!ReadMilliJoules(&ignored, &error)) {
    statusMessage_ = error + "; energy metrics will be reported as NA.";
    return true;
  }

  energySupported_ = true;
  statusMessage_ = "NVML total energy counter is available.";
  return true;
}

bool NvmlEnergyReader::ReadMilliJoules(uint64_t* milliJoules, std::string* error) {
  if (milliJoules == nullptr) {
    if (error != nullptr) *error = "ReadMilliJoules called with null output pointer";
    return false;
  }
  if (!available_ || nvmlDeviceGetTotalEnergyConsumption_ == nullptr) {
    if (error != nullptr) *error = "NVML total energy counter is unavailable";
    return false;
  }

  unsigned long long value = 0;
  NvmlReturn result = nvmlDeviceGetTotalEnergyConsumption_(device_, &value);
  if (result != kNvmlSuccess) {
    std::ostringstream oss;
    if (result == kNvmlErrorNotSupported) {
      oss << "nvmlDeviceGetTotalEnergyConsumption is unsupported on this GPU";
    } else if (result == kNvmlErrorFunctionNotFound) {
      oss << "nvmlDeviceGetTotalEnergyConsumption is missing from this NVML library";
    } else {
      oss << "nvmlDeviceGetTotalEnergyConsumption failed: " << ErrorString(result);
    }
    if (error != nullptr) *error = oss.str();
    return false;
  }

  *milliJoules = static_cast<uint64_t>(value);
  return true;
}

const char* NvmlEnergyReader::ErrorString(NvmlReturn result) const {
  if (nvmlErrorString_ != nullptr) {
    return nvmlErrorString_(result);
  }
  switch (result) {
    case kNvmlSuccess:
      return "NVML_SUCCESS";
    case kNvmlErrorNotSupported:
      return "NVML_ERROR_NOT_SUPPORTED";
    case kNvmlErrorFunctionNotFound:
      return "NVML_ERROR_FUNCTION_NOT_FOUND";
    default:
      return "unknown NVML error";
  }
}
