include ../rules.mk

.PHONY: clean

FCFLAGS += -I ../include
LD_FLAGS  = ../lib/libgpufortrt.a -lstdc++

TEST_SRC = $(shell find . -name "test_*.f90")
TEST = $(TEST_SRC:.f90=)

$(TEST): %: %.f90
	$(FC) $< $(FCFLAGS) $(LD_FLAGS) -o $@

clean:
	rm -f $(TEST)