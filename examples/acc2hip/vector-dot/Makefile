include ../rules.mk

TEST_SRC = vector-dot.f90
TEST_NAME = $(TEST_SRC:.f90=)

.PHONY: build codegen clean

build: codegen
	hipcc -c $(TEST_NAME).kernels.hip.cpp
	hipfc -c $(TEST_NAME).kernels.f08 $(FOPTS)
	hipfc $(TEST_NAME).hipified.f90 -o $(TEST_NAME) $(TEST_NAME).kernels.o $(TEST_NAME).kernels.hip.o $(FOPTS) $(ACC_INC) $(ACC_LIB)

codegen:
	../../../hipfortify.py -w $(TEST_SRC)
clean:
	rm -f *.hipified.f90 *kernels.f08 *kernels.hip.cpp *.o *.mod $(TEST_NAME)
