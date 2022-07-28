# compiler options
FC    ?= hipfc
HIPCC ?= hipcc

GPUFORT_INC = -I$(shell gpufort --print-path)/include

HIP_PLATFORM ?= amd
HIPFORT_PATH ?= /opt/rocm
HIPFORT_BACKEND = "amdgcn"
ifeq ($(HIP_PLATFORM), amd)
  HIPFORT_BACKEND = "amdgcn"
else 
  HIPFORT_BACKEND = "nvptx"
endif
HIPFORT_INC ?= $(HIPFORT_PATH)/include/hipfort/$(HIPFORT_BACKEND)
#HIPFORT_LIB ?= $(HIPFORT_PATH)/lib

FC_CFLAGS    ?= -std=f2008 -ffree-line-length-none -cpp -fmax-errors=5 \
                -Wimplicit-procedure -Wimplicit-interface \
                -I$(HIPFORT_INC)
HIPCC_CFLAGS ?= -fPIC -DGPUFORT_ARRAYS_ALWAYS_PIN_HOST_DATA
#HIPCC_CFLAGS  += -g -ggdb
LIBGPUFORT    = libgpufort_$(HIP_PLATFORM).a

# files
HIP_SRC  = $(shell find . -name "*.cpp")
FORT_SRC = gpufort.f03 gpufort_array.f03 

HIP_OBJ  = $(HIP_SRC:%.cpp=%.cpp.o)
FORT_OBJ = $(FORT_SRC:%.f03=%.o)

# targets
.PHONY: all clean clean_all gpufort_sources $(LIBGPUFORT)

all: gpufort_sources $(LIBGPUFORT) 

gpufort_sources:
	gpufort --create-gpufort-sources

$(HIP_OBJ): %.cpp.o: %.cpp
	$(HIPCC) $(GPUFORT_INC) $(HIPCC_CFLAGS) -c $^ -o $@

$(FORT_OBJ): %.o: %.f03
	$(FC) $(FC_CFLAGS) -c $^ -o $@

$(LIBGPUFORT): $(FORT_OBJ) $(HIP_OBJ)
	ar -crs $@ $^

clean:
	rm -f $(FORT_OBJ) $(HIP_OBJ) *.mod $(LIBGPUFORT)

clean_all: clean
	rm -f $(FORT_SRC) $(HIP_SRC)
