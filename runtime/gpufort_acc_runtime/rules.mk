FC ?= hipfc

HIP_PLATFORM ?= amd
ifeq ($(HIP_PLATFORM),nvcc)
  HIPFORT_INC ?= /opt/rocm/include/hipfort/nvptx
else
  HIPFORT_INC ?= /opt/rocm/include/hipfort/amdgcn
endif
FCFLAGS ?= -std=f2008 -ffree-line-length-none -cpp \
           -I$(HIPFORT_INC)

SUFFIX        ?= $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT_ACC = libgpufort_acc_$(SUFFIX).a

FCFLAGS += -DNULLPTR_MEANS_NOOP -DDELETE_NORECORD_MEANS_NOOP
#FCFLAGS += -DBLOCKING_COPIES
#FCFLAGS += -DEXIT_REGION_SYNCHRONIZE_STREAMS
#FCFLAGS += -g -ggdb -O0 -fbacktrace -fmax-errors=5 # -DDEBUG=3

CXX ?= g++
CXXFLAGS ?= 
