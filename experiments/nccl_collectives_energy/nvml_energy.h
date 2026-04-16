#ifndef NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_
#define NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_

#include <stdint.h>

#include <string>

#include <nvml.h>

class NvmlEnergyReader {
 public:
  NvmlEnergyReader();
  ~NvmlEnergyReader();

  NvmlEnergyReader(const NvmlEnergyReader&) = delete;
  NvmlEnergyReader& operator=(const NvmlEnergyReader&) = delete;

  bool Init(unsigned int gpuIndex, const char* pciBusId = nullptr);
  bool ReadMilliJoules(uint64_t* milliJoules, std::string* error);

  bool available() const { return available_; }
  bool energy_supported() const { return energySupported_; }
  const std::string& status_message() const { return statusMessage_; }

 private:
  nvmlDevice_t device_;
  bool initialized_;
  bool available_;
  bool energySupported_;
  std::string statusMessage_;

  std::string FormatError(const char* call, nvmlReturn_t result) const;
};

#endif  // NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_
