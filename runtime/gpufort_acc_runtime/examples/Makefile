FC       = hipfc
FCFLAGS ?= -std=f2008 -ffree-line-length-none
FCFLAGS += -g -ggdb

GPUFORT_ACC_DIR = $(shell gpufort --print-path)/runtime/gpufort_acc_runtime
ACC_INC         = -I$(GPUFORT_ACC_DIR)/include 
ACC_LIB         = -L$(GPUFORT_ACC_DIR)/lib -lgpufort_acc 

TEST_SRC = $(shell find . -name "test-*.f90")
TEST_APP = $(TEST_SRC:.f90=.x)
#TEST_ENV = AMD_LOG_LEVEL=3 
TEST_ENV = GPUFORT_LOG_LEVEL=3

.PHONY: $(TEST_APP)

all: $(TEST_APP)

$(TEST_APP): %.x: %.f90
	hipfc $^ -o $@ $(FCFLAGS) $(ACC_INC) $(ACC_LIB)
	$(TEST_ENV) ./$@

clean:
	rm -rf *.o *.mod $(TEST_APP)