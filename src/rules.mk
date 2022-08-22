FC    = hipfc
HIPCC = hipcc

GPUFORT_INC = -I$(shell gpufort --print-path)/include

HIP_PLATFORM ?= amd
HIPFORT_PATH ?= /opt/rocm
HIPFORT_BACKEND = amdgcn
ifeq ($(HIP_PLATFORM), amd)
  HIPFORT_BACKEND = amdgcn
else 
  HIPFORT_BACKEND = nvptx
endif
HIPFORT_INC ?= $(HIPFORT_PATH)/include/hipfort/$(HIPFORT_BACKEND)
#HIPFORT_LIB ?= $(HIPFORT_PATH)/lib

FC_CFLAGS    ?= -std=f2008 -ffree-line-length-none -cpp -fmax-errors=5 \
                -Wimplicit-procedure -Wimplicit-interface \
                -I$(HIPFORT_INC)
HIPCC_CFLAGS ?= -fPIC -DGPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
#HIPCC_CFLAGS  += -g -ggdb
