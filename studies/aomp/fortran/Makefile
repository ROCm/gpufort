TARGETS = $(shell find . -name "*.f95" | sed "s,\./,,g")
TARGETS := $(TARGETS:.f95=) 
INSPECT_TARGETS := $(addprefix inspect.,$(TARGETS)) 

FC      = /opt/rocm-4.0.1/llvm/bin/flang

CFLAGS  = -ffree-form 

$(TARGETS): %: %.f95
	$(FC) $(CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $^ -o $@

$(INSPECT_TARGETS): inspect.%: %.f95
	$(FC) $(CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $^ -emit-llvm -S
clean:
	rm -f $(TARGETS) *.mod *.o 
