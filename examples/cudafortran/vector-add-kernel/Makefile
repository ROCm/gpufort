include ../rules.mk

TEST_SRC = vector-add.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build.hip build.omp codegen.hip codegen.omp clean

codegen.hip:
	gpufort -w devicelib.f90 -E hip --config-file options.py.in --log-level DEBUG
	gpufort -w $(TEST_SRC) -E hip --config-file options.py.in --log-level DEBUG
codegen.omp:
	gpufort -w $(TEST_SRC) -E omp --config-file options.py.in
build.hip: codegen.hip
	$(HIPCC) $(HIPCC_CFLAGS) -c $(TEST_SRC)-fort2hip.hip.cpp
	$(HIPFC) $(TEST_SRC)-gpufort.f08 -o $(TEST_NAME) $(TEST_SRC)-fort2hip.hip.o $(CFLAGS)
build.omp: codegen.omp
	$(OMPFC) -ffree-form $(OMPFC_CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

clean:
	rm -rf *-gpufort.* *-fort2hip.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
