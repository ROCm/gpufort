include ../rules.mk

TEST_SRC      = vector-add.f90 vector-add-procedures.f90
TEST_NAME     = $(TEST_SRC:.f90=)
TEST_CONV_SRC = $(TEST_SRC:.f90=.f90-gpufort.f08)

.PHONY: clean

$(TEST_NAME): %: %.f90-gpufort.f08
	$(HIPCC) $(HIPCC_CFLAGS) -c $@-fort2hip.hip.cpp
	$(HIPFC) $^ -o $@ $@.f90-fort2hip.hip.o $(CFLAGS)

$(TEST_CONV_SRC): %-gpufort.f08: %
	gpufort -w $^ -E hip-gpufort-rt
clean:
	rm -rf *-gpufort.* *-fort2hip.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
