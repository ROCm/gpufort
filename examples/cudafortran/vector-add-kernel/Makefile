include ../rules.mk

TEST_SRC = vector-add.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build.hip build.omp codegen.hip codegen.omp clean

codegen.hip:
	gpufort devicelib.f90 --wrap --dest hip --config config.py
	gpufort $(TEST_SRC) --wrap --dest hip --config config.py
codegen.omp:
	gpufort $(TEST_SRC) --wrap --dest omp --config config.py
build.hip: codegen.hip
	$(HIPCC) $(HIPCC_CFLAGS) -c $(TEST_SRC)-gpufort.cpp
	$(HIPFC) $(TEST_SRC)-gpufort.f08 -o $(TEST_NAME) $(TEST_SRC)-gpufort.o $(CFLAGS) $(LDFLAGS)
build.omp: codegen.omp
	$(OMPFC) -ffree-form $(OMPFC_CFLAGS) -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $(TEST_NAME).hipified.f90 -o $(TEST_NAME)

clean:
	rm -rf *-gpufort.* *-fort2*.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
