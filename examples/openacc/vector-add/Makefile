include ../rules.mk

TEST_SRC = vector-add.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build.hip build.omp codegen.hip codegen.omp clean

codegen.hip:
	gpufort -w $(TEST_SRC) -E hip-gpufort-rt
codegen.omp:
	gpufort -w $(TEST_SRC) -E omp
build.hip: codegen.hip
	$(HIPCC) -c $(TEST_NAME).kernels.hip.cpp
	$(HIPFC) -c $(TEST_NAME).kernels.f08 $(CFLAGS)
	$(HIPFC) $(TEST_NAME).hipified.f90 -o $(TEST_NAME) $(TEST_NAME).kernels.o $(TEST_NAME).kernels.hip.o $(CFLAGS) $(ACC_INC) $(ACC_LIB)
build.omp: codegen.omp
	$(OMPFC) -ffree-form $(OMPFC_CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

clean:
	rm -f *.hipified.f90 *kernels.f08 *kernels.hip.cpp *.o *.mod gpufort*.h $(TEST_NAME)
