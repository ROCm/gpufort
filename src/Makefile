# compiler options
FC    ?= gfortran
HIPCC ?= hipcc

GPUFORT_INC = -I$(shell gpufort --path)/include

FC_CFLAGS    ?= -std=f2008 -ffree-line-length-none -cpp
HIPCC_CFLAGS ?= -fPIC
SUFFIX       ?= $(if $(HIP_PLATFORM),$(HIP_PLATFORM),amd)
LIBGPUFORT    = libgpufort_$(SUFFIX).a

# files
HIP_SRC  = $(shell find . -name "*.hip.cpp")
FORT_SRC = $(shell find . -name "*.f03")

HIP_OBJ  = $(HIP_SRC:%.hip.cpp=%.hip.o)
FORT_OBJ = $(FORT_SRC:%.f03=%.o)

# targets
.PHONY: all clean clean_all generate_gpufort_sources $(LIBGPUFORT)

all: generate_gpufort_sources $(LIBGPUFORT) 

generate_gpufort_sources:
	gpufort --create-gpufort-sources

$(HIP_OBJ): %.hip.o: %.hip.cpp
	$(HIPCC) $(GPUFORT_INC) $(HIPCC_CFLAGS) -c $^ -o $@

$(FORT_OBJ): %.o: %.f03
	$(FC) $(FC_CFLAGS) -c $^ -o $@

$(LIBGPUFORT): $(FORT_OBJ) $(HIP_OBJ)
	ar -crs $@ $^

clean:
	rm -f $(FORT_OBJ) $(HIP_OBJ) *.mod $(LIBGPUFORT)

clean_all: clean
	rm -f $(FORT_SRC) $(HIP_SRC)
