FC     ?= gfortran
CFLAGS ?= $(shell gpufort --print-gfortran-config)
LDFLAGS ?= $(shell gpufort --print-ldflags)

HIPCC ?= hipcc 
HIPCC += -DGPUFORT_PRINT_KERNEL_ARGS_ALL 
#HIPCC += -DGPUFORT_PRINT_INPUT_ARRAY_NORMS_ALL
#HIPCC += -DGPUFORT_PRINT_OUTPUT_ARRAY_NORMS_ALL
HIPCC += -DGPUFORT_PRINT_INPUT_ARRAYS_ALL
HIPCC += -DGPUFORT_PRINT_OUTPUT_ARRAYS_ALL
HIPFC ?= hipfc

HIPCC_CFLAGS = $(shell gpufort --print-cpp-config) -std=c++14

OMPFC        ?= /opt/rocm/llvm/bin/flang
OMPFC_CFLAGS ?= $(CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
