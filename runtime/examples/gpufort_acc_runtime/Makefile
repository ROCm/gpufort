FOPTS ?= -std=f2008 -ffree-line-length-none

GPUFORT_ACC_DIR = $(shell gpufort --path)/runtime/gpufort_acc_runtime
ACC_INC = -I$(GPUFORT_ACC_DIR)/include 
ACC_LIB = -L$(GPUFORT_ACC_DIR)/lib -lgpufort_acc 

.PHONY: nested-scope multi-module-test clean all

all: nested-scope multi-module-test

nested-scope: clean 
	cd $@ && hipfc test_acc.f90 -o test_acc $(FOPTS) $(ACC_INC) $(ACC_LIB)
	cd $@ && HIP_TRACE_API=1 ./test_acc

multi-module-test: clean
	cd $@ && hipfc test_acc.f90 -o test_acc $(FOPTS) $(ACC_INC) $(ACC_LIB)
	cd $@ && ./test_acc

clean:
	cd nested-scope && rm -f *.o *.mod test_acc
	cd multi-module-test && rm -f *.o *.mod test_acc
