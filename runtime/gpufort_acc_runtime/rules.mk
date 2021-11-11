FC ?= hipfc
HIPFORT_PATH ?= /opt/rocm/hipfort

FCFLAGS ?= -std=f2008 -ffree-line-length-none -cpp
ifeq ($(HIP_PLATFORM),nvcc)
  FCFLAGS += -I/$(HIPFORT_PATH)/include/nvptx
else
  FCFLAGS += -I/$(HIPFORT_PATH)/include/amdgcn
endif

SUFFIX        ?= $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT_ACC = libgpufort_acc_$(SUFFIX).a

FCFLAGS += -DNULLPTR_MEANS_NOOP -DDELETE_NORECORD_MEANS_NOOP
#FCFLAGS += -DBLOCKING_COPIES
#FCFLAGS += -DEXIT_REGION_SYNCHRONIZE_STREAMS
#FCFLAGS += -g -ggdb -O0 -fbacktrace -fmax-errors=5 # -DDEBUG=3

CXX ?= g++
CXXFLAGS ?= 
