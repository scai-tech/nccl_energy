CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= ../../build
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
BUILDDIR ?= build
TARGET ?= allreduce_energy_bench

CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
CXXFLAGS += -pthread
CPPFLAGS += -I$(CUDA_INC) -I$(NCCL_HOME)/include -I../gpu_energy_polling
LDLIBS += -L$(CUDA_LIB) -L$(NCCL_HOME)/lib -lcudart -lnccl -ldl -pthread
LDFLAGS += -Wl,-rpath,$(abspath $(NCCL_HOME))/lib

OBJECTS := $(BUILDDIR)/allreduce_energy_bench.o $(BUILDDIR)/nvml_energy.o

.PHONY: all clean run

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/allreduce_energy_bench.o: allreduce_energy_bench.cc ../gpu_energy_polling/nvml_energy.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/nvml_energy.o: ../gpu_energy_polling/nvml_energy.cc ../gpu_energy_polling/nvml_energy.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) $(LDFLAGS) -o $@

run: $(BUILDDIR)/$(TARGET)
	LD_LIBRARY_PATH=$(abspath $(NCCL_HOME))/lib:$${LD_LIBRARY_PATH:-} ./$(BUILDDIR)/$(TARGET)

clean:
	rm -rf $(BUILDDIR)
