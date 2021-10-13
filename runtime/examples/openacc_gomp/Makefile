FC            = gfortran
HIPFORT_PATH ?= /opt/rocm/hipfort

FOPTS  ?= 
FOPTS  += -std=f2008 -ffree-line-length-none -foffload=amdgcn-amdhsa="-march=gfx906"
FOPTS  += -I/$(HIPFORT_PATH)/include/amdgcn
FOPTS  += -fbacktrace -g -ggdb -O0
LDOPTS ?=
LDOPTS += -L$(HIPFORT_PATH)/lib -lhipfort-amdgcn
LDOPTS += -fopenacc -foffload=amdgcn-amdhsa="-march=gfx906" -L/opt/rocm/lib -lhipblas -lrocblas -lrocsolver -lamdhip64 -lstdc++ -lgfortran

GPUFORT_ACC_DIR = $(shell gpufort --path)/runtime/openacc_gomp
ACC_INC = -I$(GPUFORT_ACC_DIR)/include 
ACC_LIB = $(GPUFORT_ACC_DIR)/lib/libopenacc_gomp.a 

TESTS = $(shell find . -name "*.f90")
TESTS := $(TESTS:.f90=)

.PHONY: clean all run $(TESTS)

$(TESTS): %: %.f90
	$(FC) $< -o $@ $(FOPTS) $(ACC_INC) $(ACC_LIB) $(LDOPTS)
	GOMP_DEBUG=1 ./$@

clean:
	rm -f $(TESTS)
