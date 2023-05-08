include ../rules.mk

TEST_SRC      = vector-add.f90 vector-add-procedures.f90
TEST_NAME     = $(TEST_SRC:.f90=)
TEST_CONV_SRC = $(TEST_SRC:.f90=.f90-gpufort-f.f08)

.PHONY: clean codegen.hip

codegen.hip: $(TEST_CONV_SRC)

$(TEST_NAME): %: %.f90-gpufort-f.f08
	$(HIPCC) $(HIPCC_CFLAGS) -c $@.f90-gpufort.cpp
	$(HIPFC) $^ -o $@ $@.f90-gpufort.o $(CFLAGS) $(LDFLAGS)

$(TEST_CONV_SRC): %-gpufort-f.f08: %
	gpufort --wrap $^ --dest hipgpufort

clean:
	rm -rf *-gpufort.* *-gpufort-f.* *-fort2*.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod *.log
