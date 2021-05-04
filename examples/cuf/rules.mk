FC     ?= gfortran
CFLAGS ?= -std=f2008 -ffree-line-length-none -D__GPUFORT

HIPCC ?= hipcc -fPIC 
#HIPCC += -DGPUFORT_PRINT_KERNEL_ARGS_ALL 
#HIPCC += -DGPUFORT_PRINT_INPUT_ARRAY_NORMS_ALL
#HIPCC += -DGPUFORT_PRINT_OUTPUT_ARRAY_NORMS_ALL
HIPFC ?= hipfc

OMPFC        ?= /opt/rocm/llvm/bin/flang
OMPFC_CFLAGS ?= $(CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908
