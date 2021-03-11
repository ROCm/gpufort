FC     ?= gfortran
CFLAGS ?= -std=f2008 -ffree-line-length-none -D__GPUFORT

HIPCC ?= hipcc -fPIC 
HIPCC += -DGPUFORT_PRINT_INPUT_ARRAYS_ALL
HIPFC ?= hipfc

OMPFC        ?= /opt/rocm/llvm/bin/flang
OMPFC_CFLAGS ?= $(CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

GPUFORT_ACC_DIR ?= /scratch/dominic/openacc-fortran-interfaces/gpufort_acc_runtime
ACC_INC = -I/$(GPUFORT_ACC_DIR)/include
ACC_LIB = -L/$(GPUFORT_ACC_DIR)/lib -lgpufort_acc        

