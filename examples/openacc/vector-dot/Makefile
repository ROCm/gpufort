include ../rules.mk

TEST_SRC = vector-dot.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build codegen clean

build.hip:
	hipcc -c $(TEST_NAME).kernels.hip.cpp
	hipfc -c $(TEST_NAME).kernels.f08 $(FOPTS)
	hipfc $(TEST_NAME).hipified.f90 -o $(TEST_NAME) $(TEST_NAME).kernels.o $(TEST_NAME).kernels.hip.o $(FOPTS) $(ACC_INC) $(ACC_LIB)

codegen.hip:
	gpufort -w $(TEST_SRC) -E hip-gpufort-rt

build.omp:
	/opt/rocm/llvm/bin/flang -ffree-form -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

codegen.omp:
	gpufort -w $(TEST_SRC) -E omp

clean:
	rm -f *.hipified.f90 *kernels.f08 *kernels.hip.cpp *.o *.mod $(TEST_NAME)
