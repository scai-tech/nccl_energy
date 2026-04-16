CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= ../../build
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
NVML_INC ?=
NVML_LIB ?=
BUILDDIR ?= build
TARGET ?= collectives_energy_bench

CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
CXXFLAGS += -pthread
CPPFLAGS += -I$(CUDA_INC) $(if $(NVML_INC),-I$(NVML_INC),) -I$(NCCL_HOME)/include
LDLIBS += -L$(CUDA_LIB) $(if $(NVML_LIB),-L$(NVML_LIB),) -L$(NCCL_HOME)/lib -lcudart -lnccl -lnvidia-ml -pthread
LDFLAGS += -Wl,-rpath,$(abspath $(NCCL_HOME))/lib

OBJECTS := $(BUILDDIR)/collectives_energy_bench.o $(BUILDDIR)/nvml_energy.o

.PHONY: all clean run

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/collectives_energy_bench.o: collectives_energy_bench.cc nvml_energy.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/nvml_energy.o: nvml_energy.cc nvml_energy.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) $(LDFLAGS) -o $@

run: $(BUILDDIR)/$(TARGET)
	LD_LIBRARY_PATH=$(abspath $(NCCL_HOME))/lib:$${LD_LIBRARY_PATH:-} ./$(BUILDDIR)/$(TARGET)

clean:
	rm -rf $(BUILDDIR)
