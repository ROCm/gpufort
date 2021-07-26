include ../rules.mk

TEST_SRC = vector-add-kernel.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build.hip build.omp codegen.hip codegen.omp clean

codegen.hip:
	gpufort -w $(TEST_SRC) -E hip --config-file options.py.in
codegen.omp:
	gpufort -w $(TEST_SRC) -E omp --config-file options.py.in
build.hip: codegen.hip
	$(HIPCC) -c $(TEST_NAME).kernels.hip.cpp
	$(HIPFC) -c $(TEST_NAME).kernels.f08 $(CFLAGS)
	$(HIPFC) $(TEST_NAME).hipified.f90 -o $(TEST_NAME) $(TEST_NAME).kernels.o $(TEST_NAME).kernels.hip.o $(CFLAGS)
build.omp: codegen.omp
	$(OMPFC) -ffree-form $(OMPFC_CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

clean:
	rm -rf *.hipified.f90 *kernels.f08 *kernels.hip.cpp *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
