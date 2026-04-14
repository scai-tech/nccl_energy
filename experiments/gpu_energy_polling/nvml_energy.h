#ifndef NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_
#define NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_

#include <stdint.h>

#include <string>

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
  struct nvmlDevice_st;
  typedef nvmlDevice_st* NvmlDevice;

  void* library_;
  NvmlDevice device_;
  bool initialized_;
  bool available_;
  bool energySupported_;
  std::string statusMessage_;

  typedef int NvmlReturn;
  typedef NvmlReturn (*NvmlInitFn)();
  typedef NvmlReturn (*NvmlShutdownFn)();
  typedef NvmlReturn (*NvmlDeviceGetHandleByIndexFn)(unsigned int, NvmlDevice*);
  typedef NvmlReturn (*NvmlDeviceGetHandleByPciBusIdFn)(const char*, NvmlDevice*);
  typedef NvmlReturn (*NvmlDeviceGetTotalEnergyConsumptionFn)(NvmlDevice, unsigned long long*);
  typedef const char* (*NvmlErrorStringFn)(NvmlReturn);

  NvmlShutdownFn nvmlShutdown_;
  NvmlDeviceGetHandleByIndexFn nvmlDeviceGetHandleByIndex_;
  NvmlDeviceGetHandleByPciBusIdFn nvmlDeviceGetHandleByPciBusId_;
  NvmlDeviceGetTotalEnergyConsumptionFn nvmlDeviceGetTotalEnergyConsumption_;
  NvmlErrorStringFn nvmlErrorString_;

  const char* ErrorString(NvmlReturn result) const;
};

#endif  // NCCL_EXPERIMENTS_GPU_ENERGY_POLLING_NVML_ENERGY_H_
