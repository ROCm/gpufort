HIP_TEST_SRC = test_compile_headers.cpp test_gpufort_array.cpp
HIP_TEST_APP = $(HIP_TEST_SRC:.hip.cpp=.x)

FORT_TEST_SRC  = test_gpufort_array_interop.f03
FORT_TEST_APP  = $(FORT_TEST_SRC:.f03=.x)
FORT_TEST_DEPS = test_gpufort_array_kernels.hip.o

HIPCC_CFLAGS ?= $(shell gpufort --cpp-config) -g -ggdb
FC_CFLAGS    ?= $(shell gpufort --gfortran-config) -g -ggdb

LDFLAGS ?= $(shell gpufort --ldflags)

.PHONY: clean $(HIP_TEST_APP) $(FORT_TEST_APP) generate_headers

$(HIP_TEST_APP): %.x: %.hip.cpp
	hipcc $(HIPCC_CFLAGS) $^ -o $@

$(FORT_TEST_APP): %.x: %.f03 $(FORT_TEST_DEPS)
	hipfc $(FC_CFLAGS) $^ -o $@ $(LDFLAGS)

generate_headers:
	make -C ../../ gpufort_headers

%.hip.o: %.hip.cpp
	hipcc -c $(HIPCC_CFLAGS) $^ -o $@

clean:
	rm -f $(HIP_TEST_APP) $(FORT_TEST_APP) *.o *.mod