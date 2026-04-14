CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_LIB ?= $(CUDA_HOME)/lib64
BUILDDIR ?= build
TARGET ?= poll_energy_bench

# Default to Ampere + Hopper. Override with SM=80, SM=90, etc. for one machine.
ifneq ($(SM),)
  NVCC_ARCH := -gencode=arch=compute_$(SM),code=sm_$(SM)
else
  NVCC_ARCH ?= -gencode=arch=compute_80,code=sm_80 \
               -gencode=arch=compute_90,code=sm_90 \
               -gencode=arch=compute_90,code=compute_90
endif

CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
NVCUFLAGS ?= -O3 -std=c++14
CXXFLAGS += -pthread
NVCUFLAGS += -Xcompiler -pthread
CPPFLAGS += -I$(CUDA_INC)
LDLIBS += -L$(CUDA_LIB) -lcudart -ldl -pthread

SOURCES := poll_energy_bench.cu nvml_energy.cc
OBJECTS := $(BUILDDIR)/poll_energy_bench.o $(BUILDDIR)/nvml_energy.o

.PHONY: all clean run

all: $(BUILDDIR)/$(TARGET)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/poll_energy_bench.o: poll_energy_bench.cu nvml_energy.h | $(BUILDDIR)
	$(NVCC) $(NVCC_ARCH) $(NVCUFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/nvml_energy.o: nvml_energy.cc nvml_energy.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_ARCH) $(NVCUFLAGS) $^ $(LDLIBS) -o $@

run: $(BUILDDIR)/$(TARGET)
	./$(BUILDDIR)/$(TARGET)

clean:
	rm -rf $(BUILDDIR)
