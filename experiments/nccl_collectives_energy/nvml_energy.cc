#include "nvml_energy.h"

#include <sstream>

NvmlEnergyReader::NvmlEnergyReader()
    : device_(nullptr),
      initialized_(false),
      available_(false),
      energySupported_(false) {}

NvmlEnergyReader::~NvmlEnergyReader() {
  if (initialized_) {
    nvmlShutdown();
  }
}

bool NvmlEnergyReader::Init(unsigned int gpuIndex, const char* pciBusId) {
  nvmlReturn_t result = nvmlInit_v2();
  if (result != NVML_SUCCESS) {
    statusMessage_ = FormatError("nvmlInit_v2", result);
    return false;
  }
  initialized_ = true;

  if (pciBusId != nullptr && pciBusId[0] != '\0') {
    result = nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device_);
    if (result != NVML_SUCCESS) {
      std::ostringstream oss;
      oss << FormatError("nvmlDeviceGetHandleByPciBusId_v2", result)
          << " pci_bus_id=" << pciBusId;
      statusMessage_ = oss.str();
      return false;
    }
  } else {
    result = nvmlDeviceGetHandleByIndex_v2(gpuIndex, &device_);
    if (result != NVML_SUCCESS) {
      std::ostringstream oss;
      oss << FormatError("nvmlDeviceGetHandleByIndex_v2", result)
          << " gpu_index=" << gpuIndex;
      statusMessage_ = oss.str();
      return false;
    }
  }

  available_ = true;

  uint64_t ignored = 0;
  std::string error;
  if (!ReadMilliJoules(&ignored, &error)) {
    statusMessage_ = error;
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
  if (!available_) {
    if (error != nullptr) *error = "NVML total energy counter is unavailable";
    return false;
  }

  unsigned long long value = 0;
  nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(device_, &value);
  if (result != NVML_SUCCESS) {
    if (error != nullptr) {
      *error = FormatError("nvmlDeviceGetTotalEnergyConsumption", result);
    }
    return false;
  }

  *milliJoules = static_cast<uint64_t>(value);
  return true;
}

std::string NvmlEnergyReader::FormatError(const char* call,
                                          nvmlReturn_t result) const {
  std::ostringstream oss;
  oss << call << " failed: " << nvmlErrorString(result);
  return oss.str();
}
