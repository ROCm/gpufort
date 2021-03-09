FC    ?= gfortran
FLAGS ?= -std=f2008 -ffree-line-length-none -D__GPUFORT

HIPCC ?= hipcc
HIPFC ?= hipfc

OMPFC        ?= /opt/rocm/llvm/bin/flang
OMPFC_CFLAGS ?= -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

GPUFORT_ACC_DIR ?= $(HOME)/gpufort/tools/acc_runtime
ACC_INC = -I/$(GPUFORT_ACC_DIR)/include
ACC_LIB = -L/$(GPUFORT_ACC_DIR)/lib -lgpufort_acc        

