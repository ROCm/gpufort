include ../rules.mk

TEST_SRC = vector-dot.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build.hip build.omp codegen.hip codegen.omp clean

codegen.hip:
	gpufort $(TEST_SRC) --wrap --dest hipgpufort --config config.py
codegen.hip-gcc:
	gpufort $(TEST_SRC) --wrap --dest hipgcc --config config.py
codegen.omp:
	gpufort $(TEST_SRC) --wrap --dest omp --config config.py
build.hip: codegen.hip
	$(HIPCC) $(HIPCC_CFLAGS) -c $(TEST_SRC)-gpufort.cpp
	$(HIPFC) $(TEST_SRC)-gpufort.f08 -o $(TEST_NAME) $(TEST_SRC)-gpufort.o $(CFLAGS) $(LDFLAGS)
compile.hip:
	gpufortfc $(TEST_SRC) --openacc -v -o $(TEST_NAME)
build.omp: codegen.omp
	$(OMPFC) -ffree-form $(OMPFC_CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

clean:
	rm -rf *-gpufort.* *-fort2*.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
